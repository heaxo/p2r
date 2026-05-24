from pathlib import Path
import argparse
import csv
import traceback

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

import osam.apis
import osam.types


# =========================
# 配置区：按你的实际情况修改
# =========================
# 可以填图片目录，也可以填某一张具体图片路径。
# 目录：批量处理目录下图片；图片文件：只处理这一张。
INPUT_DIR = r""
YOLO_MODEL_PATH = r""
OUTPUT_ROOT = r""

# 要处理的类别。只想处理钢板就写 ["plate"]；只处理A4纸就写 ["paper"]
TARGET_CLASS_NAMES = ["plate"]

# SAM2 模型：sam2 = LabelMe balanced；sam2:large = LabelMe accuracy
SAM_MODEL_NAME = "sam2"

YOLO_CONF = 0.35
YOLO_IMGSZ = 1280
RECURSIVE = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# =========================
# 人工点副方案
# =========================
# 如果这个参数不为空，则完全跳过 YOLO 自动取点逻辑，直接使用这个比例点跑 SAM2。
# 格式：
#   None：不使用人工点，走自动逻辑
#   (ratio_x, ratio_y)：使用人工点比例，例如前端展示图 1024x2048，用户点 500x1000：
#       USER_POINT_RATIO = (500 / 1024, 1000 / 2048)
#   "0.48828125,0.48828125"：也支持字符串形式，方便命令行传参
# 注意：这里传的是比例，不是实际像素坐标。程序会根据原图真实尺寸换算成真实 x/y。
USER_POINT_RATIO = None

# 命令行也可以传：
# python batch_yolo_sam2_point_with_user_ratio.py --user-point-ratio 0.48828125,0.48828125


# =========================
# YOLO 未识别到 plate 时的中心点兜底
# =========================
# 只有自动模式下生效：
#   - YOLO 正常识别到 plate：继续走 YOLO box + 多点逻辑。
#   - YOLO 没有识别到 plate：使用 0.5,0.5 比例点作为兜底点给 SAM2。
#   - 如果 0.5,0.5 落在 paper/hole 区域，会自动向周围搜索，避开整个 paper/hole mask。
FALLBACK_TO_CENTER_WHEN_NO_PLATE = True
PLATE_FALLBACK_POINT_RATIO = (0.5, 0.5)
FALLBACK_AVOID_CLASS_NAMES = ["paper", "hole"]

# 兜底点避让 paper/hole 时，额外膨胀一点，避免点刚好落在 A4 纸边缘。
FALLBACK_AVOID_DILATE_KERNEL = 35

# 中心点落在 paper/hole 上时，向外搜索可用点。
FALLBACK_POINT_SEARCH_STEP_PX = 40
FALLBACK_POINT_SEARCH_MAX_RADIUS_RATIO = 0.45
FALLBACK_POINT_SEARCH_ANGLE_COUNT = 32

# 如果处理 plate，取点时可尽量避开 paper/hole，避免点落在A4纸或孔上。
# 注意：这里仍然只是“取点过滤”，不会把 YOLO 框/mask 传给 SAM2。
AVOID_BY_TARGET = {
    "plate": ["paper", "hole"],
    "paper": [],
    "hole": [],
}
USE_AVOID_MASK_FOR_POINTS = True
AVOID_MASK_DILATE_KERNEL = 25

# 当前阶段不完全信任 YOLO，所以只用 YOLO 框做粗定位，并将框缩小后再找点。
# 0.22 表示四周各缩进去 22%，保留中间 56% 区域。
BOX_SHRINK_RATIO = 0.22

# 在缩小框里生成原始候选点。默认 3x3 = 9 个点。
# 这一步只是为了找到一个相对干净的“原始锚点”。
CANDIDATE_GRID_SIZE = 3

# 最多保留多少个候选点用于 SAM2。
MAX_SAM_TRY_POINTS = 3

# 多点模式一次性传给 SAM2 的点数。一般 3 个比较合适。
MULTI_POINT_COUNT = 3

# 原始锚点如果靠近 YOLO mask 边缘，就在它附近用距离变换吸到更靠钢板内部的位置。
# 只用于修正点，不把 YOLO mask 传给 SAM2。
ANCHOR_INNER_SEARCH_RADIUS = 180
ANCHOR_INNER_MIN_DIST = 10.0

# 围绕“内部修正后的锚点”生成上下左右点的半径。
# None 表示根据缩小框短边自动算。
POINT_AROUND_RADIUS_PX = None
POINT_AROUND_RADIUS_RATIO = 0.18
POINT_AROUND_RADIUS_MIN_PX = 50
POINT_AROUND_RADIUS_MAX_PX = 180

# 候选点之间的最小距离策略。
# 实际最小距离 = around_radius * MIN_POINT_DISTANCE_RATIO_FOR_AROUND。
# 注意：现在不是从整个 box 里为了拉开距离乱选点，而是围绕锚点扩散；
# 这个参数只是防止重复点/过近点。
MIN_POINT_DISTANCE_RATIO_FOR_AROUND = 0.70

# 候选点局部评分窗口。窗口越大，越能避开较大的字迹/油污，但也可能过于保守。
POINT_CLEAN_WINDOW_SIZE = 51

# SAM2 结果弱过滤：不强信 YOLO，只过滤明显不合理的小斑点/整图。
# plate 通常是大目标，所以最小面积比例可以稍大；paper/hole 要小一些。
MIN_SAM_MASK_AREA_RATIO_BY_CLASS = {
    "plate": 0.003,   # 图片面积的 0.3%，太小一般是点到油污/文字/小斑点
    "paper": 0.0003,
    "hole": 0.0002,
}
MAX_SAM_MASK_AREA_RATIO_BY_CLASS = {
    "plate": 0.95,    # 接近整张图一般不合理
    "paper": 0.40,
    "hole": 0.40,
}

# 是否在同时处理 plate 和 paper 时，额外输出 final_plate_mask = plate OR paper
MERGE_PAPER_TO_PLATE = True

# =========================
# A4纸区域补回钢板
# =========================
# 作用：SAM2 分割 plate 时，有时会把 A4纸覆盖区域当成“不是钢板”，导致 plate mask 缺一块。
# 这里用 YOLO 识别到的 paper mask/box，把 A4纸区域 OR 回 plate mask。
# 注意：这里只是补 mask，不把 paper box 作为 SAM2 prompt。
FILL_PAPER_TO_PLATE = True
PAPER_CLASS_NAMES = ["paper"]
PAPER_FILL_DILATE_KERNEL = 9
PAPER_FILL_CLOSE_KERNEL = 15
PAPER_FILL_USE_BOX_IF_NO_MASK = True
SAVE_RAW_PLATE_MASK_BEFORE_PAPER_FILL = True

# 人工比例点模式下，默认仍然跑一次 YOLO，只为了找 A4纸并补回 plate。
# 如果你希望人工点模式完全不跑 YOLO，可以改成 False。
RUN_YOLO_FOR_PAPER_FILL_IN_MANUAL_MODE = True

# 速度优化：多点 SAM2 结果合理时，只跑一次 SAM2。
# 如果多点结果不合理，是否再启用单点逐个兜底。
# False 更快；True 更稳但可能变慢。
ENABLE_SINGLE_POINT_FALLBACK = False


# =========================
# 基础工具函数
# =========================
def list_images(input_path, recursive=False):
    """
    获取待处理图片列表。

    input_path 可以是：
      1. 图片目录：处理目录下所有支持格式的图片
      2. 单张图片路径：只处理这一张图片

    这样前端/命令行既可以传目录批量处理，也可以传具体图片做单图处理。
    """
    input_path = Path(input_path)

    if not input_path.exists():
        return []

    # 传入的是某一张具体图片：只处理这一张
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTS:
            raise RuntimeError(f"输入路径是文件，但不是支持的图片格式：{input_path}")
        return [input_path]

    # 传入的是目录：处理目录内图片
    if input_path.is_dir():
        if recursive:
            return sorted([p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
        return sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])

    return []


def parse_user_point_ratio(value):
    """
    解析人工点比例。

    允许：
      None / "" / "none"：表示不使用人工点
      (0.5, 0.5) / [0.5, 0.5]
      "0.5,0.5"

    返回：
      None 或 (ratio_x, ratio_y)
    """
    if value is None:
        return None

    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "null", "false"}:
            return None
        text = text.replace("，", ",").replace(";", ",")
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"USER_POINT_RATIO 格式错误，应为 'ratio_x,ratio_y'，当前值：{value}")
        rx = float(parts[0])
        ry = float(parts[1])
    else:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"USER_POINT_RATIO 格式错误，应为 (ratio_x, ratio_y)，当前值：{value}")
        rx = float(value[0])
        ry = float(value[1])

    # 这里强制限制在 [0,1]，避免前端误差导致越界。
    rx = max(0.0, min(1.0, rx))
    ry = max(0.0, min(1.0, ry))

    return rx, ry


def ratio_to_xy(user_point_ratio, image_shape):
    """
    将人工点比例转换成原图真实像素坐标。

    例子：
      前端展示图：1024 x 2048
      用户点击：500 x 1000
      前端传：ratio_x = 500 / 1024，ratio_y = 1000 / 2048

      原图真实尺寸：2048 x 4096
      程序换算后：x = ratio_x * 2048，y = ratio_y * 4096
    """
    ratio = parse_user_point_ratio(user_point_ratio)
    if ratio is None:
        return None

    rx, ry = ratio
    h, w = image_shape[:2]

    x = int(round(rx * (w - 1)))
    y = int(round(ry * (h - 1)))

    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    return int(x), int(y)


def _get_mask_from_yolo_result(result, index, h, w):
    """
    从 YOLO seg 结果中取单个实例 mask，并缩放到原图尺寸。
    如果 best2.pt 不是 seg 模型，则 result.masks 会是 None。
    """
    if result.masks is None:
        return None

    mask = result.masks.data[index].cpu().numpy()

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return (mask > 0.5).astype(np.uint8)


def shrink_box(x1, y1, x2, y2, img_w, img_h, shrink_ratio=0.22):
    """
    将 YOLO 框向内缩小。
    目的：YOLO 当前只是粗定位，不能完全相信边界，所以不在完整框里取点。
    """
    x1 = float(max(0, min(img_w - 1, x1)))
    y1 = float(max(0, min(img_h - 1, y1)))
    x2 = float(max(0, min(img_w - 1, x2)))
    y2 = float(max(0, min(img_h - 1, y2)))

    if x2 <= x1 or y2 <= y1:
        return int(x1), int(y1), int(x2), int(y2)

    bw = x2 - x1
    bh = y2 - y1

    # 防止细长目标被缩没
    shrink_ratio = max(0.0, min(0.45, float(shrink_ratio)))

    nx1 = x1 + bw * shrink_ratio
    ny1 = y1 + bh * shrink_ratio
    nx2 = x2 - bw * shrink_ratio
    ny2 = y2 - bh * shrink_ratio

    if nx2 <= nx1 or ny2 <= ny1:
        return int(x1), int(y1), int(x2), int(y2)

    return int(nx1), int(ny1), int(nx2), int(ny2)


def generate_points_in_box(x1, y1, x2, y2, grid_size=3):
    """
    在缩小框内部生成原始候选点。
    这一步只是为了找到原始锚点，不是最终直接拿所有 box 点去 SAM2。
    """
    if x2 <= x1 or y2 <= y1:
        return []

    if grid_size == 3:
        ratios = [
            (0.50, 0.50),
            (0.35, 0.50),
            (0.65, 0.50),
            (0.50, 0.35),
            (0.50, 0.65),
            (0.35, 0.35),
            (0.65, 0.35),
            (0.35, 0.65),
            (0.65, 0.65),
        ]
    else:
        vals = np.linspace(0.25, 0.75, grid_size)
        ratios = [(float(rx), float(ry)) for ry in vals for rx in vals]
        ratios.sort(key=lambda p: abs(p[0] - 0.5) + abs(p[1] - 0.5))

    w = x2 - x1
    h = y2 - y1

    points = []
    seen = set()
    for rx, ry in ratios:
        x = int(round(x1 + w * rx))
        y = int(round(y1 + h * ry))
        key = (x, y)
        if key not in seen:
            seen.add(key)
            points.append((x, y))

    return points


def build_avoid_mask(result, avoid_class_names, use_box_if_no_mask=True):
    """
    构建 paper/hole 等需要避开的 mask。
    注意：该 mask 只用于过滤候选点，不传给 SAM2，不作为强约束。

    这里做了增强：
      1. 有 YOLO 分割 mask 时，优先用实例 mask。
      2. 如果 result.masks 为空，或某个实例取不到 mask，则可以用检测框 box 兜底。
         这样即使 YOLO 没有识别到 plate，只要识别到了 paper，也能避开 A4纸区域。
    """
    h, w = result.orig_shape
    avoid_mask = np.zeros((h, w), dtype=np.uint8)

    if result.boxes is None or len(result.boxes) == 0:
        return avoid_mask

    if not avoid_class_names:
        return avoid_mask

    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    avoid_class_ids = {cls_id for cls_id, cls_name in names.items() if cls_name in set(avoid_class_names)}

    if not avoid_class_ids:
        return avoid_mask

    for i, cls_id in enumerate(classes):
        if cls_id not in avoid_class_ids:
            continue

        m = None
        if result.masks is not None:
            m = _get_mask_from_yolo_result(result, i, h, w)

        if m is not None:
            avoid_mask = np.maximum(avoid_mask, m.astype(np.uint8))
        elif use_box_if_no_mask:
            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))
            if x2 > x1 and y2 > y1:
                avoid_mask[y1:y2 + 1, x1:x2 + 1] = 1

    if avoid_mask.sum() > 0 and AVOID_MASK_DILATE_KERNEL > 0:
        k = int(AVOID_MASK_DILATE_KERNEL)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        avoid_mask = cv2.dilate(avoid_mask, kernel, iterations=1)

    return (avoid_mask > 0).astype(np.uint8)



def build_class_mask_from_yolo_result(
        result,
        class_names,
        use_box_if_no_mask=True,
        dilate_kernel_size=0,
        close_kernel_size=0,
):
    """
    根据 YOLO 结果构建某些类别的合并 mask。

    当前主要用于把 A4纸 paper 区域补回 plate mask：
      - YOLO 是 seg 模型时，优先使用 paper 的分割 mask。
      - 如果没有分割 mask，但有 paper 检测框，可选用 box 矩形作为兜底。

    注意：这个 mask 不会传给 SAM2，只用于后处理补洞。
    """
    h, w = result.orig_shape
    out = np.zeros((h, w), dtype=np.uint8)

    if result.boxes is None or len(result.boxes) == 0:
        return out

    if not class_names:
        return out

    names = result.names
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    target_class_ids = {cls_id for cls_id, cls_name in names.items() if cls_name in set(class_names)}
    if not target_class_ids:
        return out

    for i, cls_id in enumerate(classes):
        if cls_id not in target_class_ids:
            continue

        m = None
        if result.masks is not None:
            m = _get_mask_from_yolo_result(result, i, h, w)

        if m is not None:
            out = np.maximum(out, m.astype(np.uint8))
        elif use_box_if_no_mask:
            x1, y1, x2, y2 = boxes[i]
            x1 = max(0, min(w - 1, int(x1)))
            y1 = max(0, min(h - 1, int(y1)))
            x2 = max(0, min(w - 1, int(x2)))
            y2 = max(0, min(h - 1, int(y2)))
            if x2 > x1 and y2 > y1:
                out[y1:y2 + 1, x1:x2 + 1] = 1

    if out.sum() <= 0:
        return out

    if dilate_kernel_size and dilate_kernel_size > 0:
        k = int(dilate_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.dilate(out, kernel, iterations=1)

    if close_kernel_size and close_kernel_size > 0:
        k = int(close_kernel_size)
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

    return (out > 0).astype(np.uint8)


def apply_paper_fill_to_plate_mask(
        plate_mask,
        paper_mask,
        output_mask_path,
        raw_mask_path=None,
):
    """
    把 A4纸区域补回 plate mask。

    原理：final_plate = plate_mask OR paper_mask
    因为 A4纸放在钢板上，A4纸下面那块实际也是钢板。
    """
    if plate_mask is None:
        raise RuntimeError("plate_mask 为空，无法补回 A4纸区域")

    plate_bin = plate_mask > 0

    if paper_mask is None or paper_mask.size == 0 or paper_mask.sum() <= 0:
        Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(plate_bin.astype(np.uint8) * 255).save(output_mask_path)
        return plate_bin.astype(np.uint8) * 255, {
            "filled": False,
            "paper_area": 0,
            "added_area": 0,
            "reason": "no_paper_mask",
        }

    paper_bin = paper_mask > 0

    if paper_bin.shape != plate_bin.shape:
        paper_bin = cv2.resize(
            paper_bin.astype(np.uint8),
            (plate_bin.shape[1], plate_bin.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ) > 0

    added = paper_bin & (~plate_bin)
    final_bin = plate_bin | paper_bin
    final_mask = final_bin.astype(np.uint8) * 255

    if raw_mask_path is not None:
        Path(raw_mask_path).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(plate_bin.astype(np.uint8) * 255).save(raw_mask_path)

    Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(final_mask).save(output_mask_path)

    return final_mask, {
        "filled": True,
        "paper_area": int(paper_bin.sum()),
        "added_area": int(added.sum()),
        "reason": "ok",
    }

def get_target_from_yolo_result(result, target_class_name="plate", avoid_class_names=None):
    """
    从 YOLO 结果里找目标实例。
    当前逻辑：同类多个实例时，取面积最大的那个。

    注意：这里只用 YOLO 做粗定位，不把 YOLO 框/mask 直接传给 SAM2。
    """
    if avoid_class_names is None:
        avoid_class_names = []

    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO 没有识别到任何目标")

    h, w = result.orig_shape
    names = result.names

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    target_class_ids = [cls_id for cls_id, cls_name in names.items() if cls_name == target_class_name]
    if not target_class_ids:
        raise RuntimeError(f"YOLO 模型中没有类别：{target_class_name}，当前类别：{names}")

    target_class_id = target_class_ids[0]
    avoid_mask = build_avoid_mask(result, avoid_class_names)

    candidates = []

    for i, cls_id in enumerate(classes):
        if cls_id != target_class_id:
            continue

        x1, y1, x2, y2 = boxes[i]
        score = float(confs[i])

        target_mask = None
        mask_area = 0

        if result.masks is not None:
            target_mask = _get_mask_from_yolo_result(result, i, h, w)
            if target_mask is not None:
                mask_area = int((target_mask > 0).sum())

        box_area = max(0, int((x2 - x1) * (y2 - y1)))
        area = mask_area if mask_area > 0 else box_area

        candidates.append({
            "box": [int(x1), int(y1), int(x2), int(y2)],
            "conf": score,
            "area": int(area),
            "box_area": int(box_area),
            "mask_area": int(mask_area),
            "class_name": target_class_name,
            "target_mask": target_mask,
            "avoid_mask": avoid_mask,
        })

    if not candidates:
        detected = sorted({names[int(cls_id)] for cls_id in classes})
        raise RuntimeError(f"没有找到目标类别：{target_class_name}，本图识别到：{detected}")

    return max(candidates, key=lambda item: item["area"])


# =========================
# 候选点评分：避开油污/字迹/锈斑/高光边界
# =========================
def score_point_cleanliness(image_rgb, x, y, window_size=51):
    """
    分数越高，说明这个点附近越干净。
    这个函数不判断目标类别，只判断局部纹理是否复杂。
    """
    h, w = image_rgb.shape[:2]

    x = int(x)
    y = int(y)
    if x < 0 or x >= w or y < 0 or y >= h:
        return -9999.0, {
            "edge_density": 1.0,
            "variance": 1.0,
            "brightness": 0.0,
        }

    k = int(window_size)
    if k % 2 == 0:
        k += 1
    r = k // 2

    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(w, x + r + 1)
    y2 = min(h, y + r + 1)

    crop = image_rgb[y1:y2, x1:x2]
    if crop.size == 0:
        return -9999.0, {
            "edge_density": 1.0,
            "variance": 1.0,
            "brightness": 0.0,
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, 50, 150)
    edge_density = float(np.mean(edges > 0))

    gray_f = gray.astype(np.float32) / 255.0
    variance = float(np.var(gray_f))
    brightness = float(np.mean(gray_f))

    # 油污圈/文字/锈斑边界：边缘密度高、灰度方差高。
    # 高光区域：亮度过高时也适当扣分。
    highlight_penalty = max(0.0, brightness - 0.88) * 1.5

    clean_score = (
        1.0
        - edge_density * 3.0
        - variance * 2.0
        - highlight_penalty
    )

    return float(clean_score), {
        "edge_density": edge_density,
        "variance": variance,
        "brightness": brightness,
    }


def is_point_valid_for_masks(x, y, image_shape, target_mask=None, avoid_mask=None):
    """
    判断点是否可用。
    target_mask 只用来保证衍生点仍在钢板内，不传给 SAM2。
    avoid_mask 只用来避开 paper/hole，不传给 SAM2。
    """
    h, w = image_shape[:2]
    x = int(x)
    y = int(y)

    if x < 0 or x >= w or y < 0 or y >= h:
        return False

    if target_mask is not None and target_mask.size > 0:
        if target_mask[y, x] <= 0:
            return False

    if USE_AVOID_MASK_FOR_POINTS and avoid_mask is not None and avoid_mask.size > 0:
        if avoid_mask[y, x] > 0:
            return False

    return True


def move_anchor_to_inner_point(anchor_x, anchor_y, target_mask, search_radius=180, min_dist=10.0):
    """
    如果原始点靠近钢板边缘，就在它附近找一个更靠钢板内部的点。
    依赖 target_mask，但只用于修正点，不传给 SAM2。

    原理：在原始点附近截取 target_mask 小区域，做 distanceTransform，
    距离值越大，说明越靠钢板内部、离边界越远。
    """
    if target_mask is None or target_mask.size == 0:
        return int(anchor_x), int(anchor_y), 0.0, False

    h, w = target_mask.shape[:2]

    ax = int(anchor_x)
    ay = int(anchor_y)

    if ax < 0 or ax >= w or ay < 0 or ay >= h:
        return ax, ay, 0.0, False

    x1 = max(0, ax - int(search_radius))
    y1 = max(0, ay - int(search_radius))
    x2 = min(w, ax + int(search_radius) + 1)
    y2 = min(h, ay + int(search_radius) + 1)

    crop = (target_mask[y1:y2, x1:x2] > 0).astype(np.uint8)

    if crop.sum() <= 0:
        return ax, ay, 0.0, False

    dist = cv2.distanceTransform(crop * 255, cv2.DIST_L2, 5)
    _, max_dist, _, max_loc = cv2.minMaxLoc(dist)

    # 附近没有明显内部区域，就不移动。
    if max_dist < float(min_dist):
        return ax, ay, float(max_dist), False

    inner_x = x1 + max_loc[0]
    inner_y = y1 + max_loc[1]

    return int(inner_x), int(inner_y), float(max_dist), True


def calc_point_around_radius(sx1, sy1, sx2, sy2):
    """
    计算围绕锚点扩散的半径。
    默认根据缩小框短边计算，也可以用 POINT_AROUND_RADIUS_PX 固定。
    """
    if POINT_AROUND_RADIUS_PX is not None:
        return max(1, int(POINT_AROUND_RADIUS_PX))

    bw = max(1, int(sx2 - sx1))
    bh = max(1, int(sy2 - sy1))

    r = int(min(bw, bh) * POINT_AROUND_RADIUS_RATIO)
    r = max(POINT_AROUND_RADIUS_MIN_PX, r)
    r = min(POINT_AROUND_RADIUS_MAX_PX, r)

    return int(r)


def generate_points_around_anchor(anchor_x, anchor_y, image_shape, target_mask=None, avoid_mask=None, radius=80):
    """
    以内部锚点为中心，向上下左右和斜角生成衍生点。

    关键：衍生点必须仍在钢板 target_mask 内。
    这样不会再从整个矩形框里乱选点，避免异形钢板时点跑到地面上。
    """
    r = int(radius)

    offsets = [
        (0, 0),       # 内部锚点
        (0, -r),      # 上
        (r, 0),       # 右
        (0, r),       # 下
        (-r, 0),      # 左
        (-r, -r),     # 左上
        (r, -r),      # 右上
        (-r, r),      # 左下
        (r, r),       # 右下
        (0, -r // 2),
        (r // 2, 0),
        (0, r // 2),
        (-r // 2, 0),
    ]

    points = []
    seen = set()
    for dx, dy in offsets:
        x = int(anchor_x + dx)
        y = int(anchor_y + dy)
        key = (x, y)
        if key in seen:
            continue
        seen.add(key)

        if not is_point_valid_for_masks(
            x=x,
            y=y,
            image_shape=image_shape,
            target_mask=target_mask,
            avoid_mask=avoid_mask,
        ):
            continue

        points.append({
            "x": int(x),
            "y": int(y),
            "from_anchor": True,
            "dx": int(dx),
            "dy": int(dy),
        })

    return points


def filter_points_by_min_distance(points, min_distance, max_points):
    """
    从已按 point_score 从高到低排序的候选点里筛选点，保证点之间不要太近。
    现在候选点是围绕锚点生成的，所以这里只做轻量去重/去过近。
    """
    if not points:
        return []

    selected = []

    for p in points:
        if len(selected) >= max_points:
            break

        x = int(p["x"])
        y = int(p["y"])

        too_close = False
        for sp in selected:
            sx = int(sp["x"])
            sy = int(sp["y"])
            dist = ((x - sx) ** 2 + (y - sy) ** 2) ** 0.5
            if dist < float(min_distance):
                too_close = True
                break

        if not too_close:
            selected.append(p)

    # 如果距离限制导致点不够，按分数补齐。
    if len(selected) < max_points:
        for p in points:
            if len(selected) >= max_points:
                break
            exists = any(int(sp["x"]) == int(p["x"]) and int(sp["y"]) == int(p["y"]) for sp in selected)
            if not exists:
                selected.append(p)

    return selected[:max_points]


def make_candidate_points_from_yolo_box(image_rgb, point_info):
    """
    自动取点主逻辑。

    流程：
      1. 用 YOLO box 缩小区域生成原始候选点。
      2. 原始点必须尽量在 YOLO target_mask 内，只用来过滤点，不传给 SAM2。
      3. 按局部干净程度选一个原始锚点。
      4. 如果锚点靠边，用 distanceTransform 吸到钢板内部。
      5. 只围绕内部锚点生成上下左右衍生点，衍生点必须仍在 target_mask 内。
      6. 选出最多 MAX_SAM_TRY_POINTS 个点给 SAM2。
    """
    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = point_info["box"]

    sx1, sy1, sx2, sy2 = shrink_box(
        x1, y1, x2, y2,
        img_w=w,
        img_h=h,
        shrink_ratio=BOX_SHRINK_RATIO,
    )

    raw_points = generate_points_in_box(
        sx1, sy1, sx2, sy2,
        grid_size=CANDIDATE_GRID_SIZE,
    )

    avoid_mask = point_info.get("avoid_mask")
    target_mask = point_info.get("target_mask")

    filtered_raw_points = []

    for x, y in raw_points:
        # target_mask 不为空时，原始锚点也要求在钢板 mask 内。
        # 如果 target_mask 为空，说明不是 seg 模型或没有 mask，只能退化为 box 内取点。
        if is_point_valid_for_masks(
            x=x,
            y=y,
            image_shape=image_rgb.shape,
            target_mask=target_mask,
            avoid_mask=avoid_mask,
        ):
            filtered_raw_points.append((x, y))

    # 如果 YOLO mask 把点全过滤掉，说明 mask 不可靠。
    # 这里退回 raw_points 只是为了找锚点；后续如果 target_mask 存在，衍生点仍会再次要求在 mask 内。
    if not filtered_raw_points:
        filtered_raw_points = raw_points

    scored_raw = []
    for x, y in filtered_raw_points:
        s, detail = score_point_cleanliness(
            image_rgb=image_rgb,
            x=x,
            y=y,
            window_size=POINT_CLEAN_WINDOW_SIZE,
        )
        scored_raw.append({
            "x": int(x),
            "y": int(y),
            "point_score": float(s),
            "point_detail": detail,
            "stage": "raw_anchor_candidate",
        })

    if not scored_raw:
        raise RuntimeError("没有可用的原始候选点")

    scored_raw.sort(key=lambda p: p["point_score"], reverse=True)

    raw_anchor = scored_raw[0]
    raw_anchor_x = int(raw_anchor["x"])
    raw_anchor_y = int(raw_anchor["y"])

    inner_x, inner_y, inner_dist, moved_to_inner = move_anchor_to_inner_point(
        anchor_x=raw_anchor_x,
        anchor_y=raw_anchor_y,
        target_mask=target_mask,
        search_radius=ANCHOR_INNER_SEARCH_RADIUS,
        min_dist=ANCHOR_INNER_MIN_DIST,
    )

    around_radius = calc_point_around_radius(sx1, sy1, sx2, sy2)

    around_points = generate_points_around_anchor(
        anchor_x=inner_x,
        anchor_y=inner_y,
        image_shape=image_rgb.shape,
        target_mask=target_mask,
        avoid_mask=avoid_mask,
        radius=around_radius,
    )

    # 如果围绕内部锚点生成不出点，至少保留内部锚点或原始锚点。
    if not around_points:
        fallback_x, fallback_y = inner_x, inner_y
        if not is_point_valid_for_masks(
            x=fallback_x,
            y=fallback_y,
            image_shape=image_rgb.shape,
            target_mask=target_mask,
            avoid_mask=avoid_mask,
        ):
            fallback_x, fallback_y = raw_anchor_x, raw_anchor_y

        around_points = [{
            "x": int(fallback_x),
            "y": int(fallback_y),
            "from_anchor": True,
            "dx": 0,
            "dy": 0,
        }]

    scored_around = []
    for p in around_points:
        x = int(p["x"])
        y = int(p["y"])
        s, detail = score_point_cleanliness(
            image_rgb=image_rgb,
            x=x,
            y=y,
            window_size=POINT_CLEAN_WINDOW_SIZE,
        )

        # 内部锚点本身稍微加分，避免全被周围点压下去。
        anchor_bonus = 0.15 if int(p.get("dx", 0)) == 0 and int(p.get("dy", 0)) == 0 else 0.0

        scored_around.append({
            "x": x,
            "y": y,
            "point_score": float(s + anchor_bonus),
            "point_detail": detail,
            "stage": "around_inner_anchor",
            "raw_anchor": (raw_anchor_x, raw_anchor_y),
            "inner_anchor": (inner_x, inner_y),
            "moved_to_inner": bool(moved_to_inner),
            "inner_dist": float(inner_dist),
            "around_radius": int(around_radius),
            "dx": int(p.get("dx", 0)),
            "dy": int(p.get("dy", 0)),
        })

    scored_around.sort(key=lambda p: p["point_score"], reverse=True)

    min_point_distance = int(max(1, around_radius * MIN_POINT_DISTANCE_RATIO_FOR_AROUND))

    selected_points = filter_points_by_min_distance(
        points=scored_around,
        min_distance=min_point_distance,
        max_points=MAX_SAM_TRY_POINTS,
    )

    if not selected_points:
        raise RuntimeError("没有可用候选点")

    meta = {
        "shrink_box": [int(sx1), int(sy1), int(sx2), int(sy2)],
        "raw_anchor": [int(raw_anchor_x), int(raw_anchor_y)],
        "inner_anchor": [int(inner_x), int(inner_y)],
        "moved_to_inner": bool(moved_to_inner),
        "inner_dist": float(inner_dist),
        "around_radius": int(around_radius),
        "min_point_distance": int(min_point_distance),
        "raw_candidates": scored_raw,
    }

    return selected_points, meta



# =========================
# YOLO 未识别到 plate 时的中心点兜底取点
# =========================
def _dilate_binary_mask(mask, kernel_size):
    """
    对二值 mask 做膨胀。
    用于兜底点避开 paper/hole 时，扩大一点安全距离，避免点贴在 A4纸边缘。
    """
    if mask is None or mask.size == 0:
        return mask

    out = (mask > 0).astype(np.uint8)
    if out.sum() <= 0:
        return out

    k = int(kernel_size)
    if k <= 0:
        return out

    if k % 2 == 0:
        k += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.dilate(out, kernel, iterations=1)
    return (out > 0).astype(np.uint8)


def build_fallback_avoid_mask(result, image_shape, avoid_class_names=None):
    """
    构建 YOLO 未识别到 plate 时的避让区域。

    主要用于：
      - 中心点 0.5,0.5 如果落在 paper/hole 上，要避开整个 paper/hole 区域。
      - 只要 YOLO 识别到了 paper/hole，就算没有 plate，也可以拿 paper/hole mask/box 做避让。
    """
    h, w = image_shape[:2]
    avoid_mask = np.zeros((h, w), dtype=np.uint8)

    if avoid_class_names is None:
        avoid_class_names = FALLBACK_AVOID_CLASS_NAMES

    if result is None:
        return avoid_mask

    try:
        avoid_mask = build_avoid_mask(
            result=result,
            avoid_class_names=avoid_class_names,
            use_box_if_no_mask=True,
        )
    except Exception:
        # 兜底逻辑不能因为避让 mask 构建失败导致整图失败。
        avoid_mask = np.zeros((h, w), dtype=np.uint8)

    if avoid_mask.shape != (h, w):
        avoid_mask = cv2.resize(avoid_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    avoid_mask = _dilate_binary_mask(avoid_mask, FALLBACK_AVOID_DILATE_KERNEL)
    return (avoid_mask > 0).astype(np.uint8)


def is_point_on_avoid_mask(x, y, image_shape, avoid_mask=None):
    """
    判断某个点是否落在避让区域上。
    """
    h, w = image_shape[:2]
    x = int(x)
    y = int(y)

    if x < 0 or x >= w or y < 0 or y >= h:
        return True

    if avoid_mask is not None and avoid_mask.size > 0:
        if avoid_mask[y, x] > 0:
            return True

    return False


def find_nearest_point_outside_avoid_mask(image_rgb, base_x, base_y, avoid_mask=None):
    """
    从 base_x/base_y 开始找一个不在 paper/hole 上的点。

    逻辑：
      1. 如果中心点本身不在 paper/hole 上，直接使用中心点。
      2. 如果中心点落在 paper/hole 上，从近到远按圆环搜索。
      3. 每个圆环里取局部最干净的点，避免点到文字、锈斑、高光边界。
    """
    h, w = image_rgb.shape[:2]
    base_x = int(max(0, min(w - 1, base_x)))
    base_y = int(max(0, min(h - 1, base_y)))

    center_hit_avoid = is_point_on_avoid_mask(
        x=base_x,
        y=base_y,
        image_shape=image_rgb.shape,
        avoid_mask=avoid_mask,
    )

    if not center_hit_avoid:
        score, detail = score_point_cleanliness(
            image_rgb=image_rgb,
            x=base_x,
            y=base_y,
            window_size=POINT_CLEAN_WINDOW_SIZE,
        )
        return {
            "x": int(base_x),
            "y": int(base_y),
            "point_score": float(score),
            "point_detail": detail,
            "stage": "center_fallback",
            "base_point": [int(base_x), int(base_y)],
            "center_hit_avoid": False,
            "avoid_adjusted": False,
            "search_radius": 0,
        }

    step = max(5, int(FALLBACK_POINT_SEARCH_STEP_PX))
    max_radius = int(min(h, w) * float(FALLBACK_POINT_SEARCH_MAX_RADIUS_RATIO))
    max_radius = max(step, max_radius)

    angle_count = max(8, int(FALLBACK_POINT_SEARCH_ANGLE_COUNT))

    for radius in range(step, max_radius + step, step):
        ring_candidates = []
        for i in range(angle_count):
            angle = 2.0 * np.pi * i / angle_count
            x = int(round(base_x + np.cos(angle) * radius))
            y = int(round(base_y + np.sin(angle) * radius))

            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            if is_point_on_avoid_mask(
                x=x,
                y=y,
                image_shape=image_rgb.shape,
                avoid_mask=avoid_mask,
            ):
                continue

            score, detail = score_point_cleanliness(
                image_rgb=image_rgb,
                x=x,
                y=y,
                window_size=POINT_CLEAN_WINDOW_SIZE,
            )

            ring_candidates.append({
                "x": int(x),
                "y": int(y),
                "point_score": float(score),
                "point_detail": detail,
                "stage": "center_fallback_avoid_adjusted",
                "base_point": [int(base_x), int(base_y)],
                "center_hit_avoid": True,
                "avoid_adjusted": True,
                "search_radius": int(radius),
            })

        if ring_candidates:
            ring_candidates.sort(key=lambda p: p["point_score"], reverse=True)
            return ring_candidates[0]

    # 极端情况：整张图都被避让 mask 覆盖，或者 YOLO paper mask 错误过大。
    # 此时只能退回原中心点，至少保证不会整张图直接失败。
    score, detail = score_point_cleanliness(
        image_rgb=image_rgb,
        x=base_x,
        y=base_y,
        window_size=POINT_CLEAN_WINDOW_SIZE,
    )
    return {
        "x": int(base_x),
        "y": int(base_y),
        "point_score": float(score),
        "point_detail": detail,
        "stage": "center_fallback_force_use_center",
        "base_point": [int(base_x), int(base_y)],
        "center_hit_avoid": True,
        "avoid_adjusted": False,
        "search_radius": -1,
        "message": "no_available_point_outside_avoid_mask",
    }


def make_candidate_point_from_center_fallback(image_rgb, result=None, target_class_name="plate"):
    """
    YOLO 没识别到 plate 时，生成一个中心兜底点。

    注意：
      - 这里只生成一个点给 SAM2。
      - 不使用 YOLO plate box，因为当前就是 plate 没识别到。
      - 仍然使用 YOLO 识别到的 paper/hole 做避让，避免点落到 A4纸上。
    """
    h, w = image_rgb.shape[:2]

    base_xy = ratio_to_xy(PLATE_FALLBACK_POINT_RATIO, image_rgb.shape)
    if base_xy is None:
        base_xy = (w // 2, h // 2)

    base_x, base_y = base_xy

    avoid_mask = build_fallback_avoid_mask(
        result=result,
        image_shape=image_rgb.shape,
        avoid_class_names=FALLBACK_AVOID_CLASS_NAMES,
    )

    p = find_nearest_point_outside_avoid_mask(
        image_rgb=image_rgb,
        base_x=base_x,
        base_y=base_y,
        avoid_mask=avoid_mask,
    )

    candidate_points = [{
        "x": int(p["x"]),
        "y": int(p["y"]),
        "point_score": float(p.get("point_score", 1.0)),
        "point_detail": p.get("point_detail", {}),
        "stage": p.get("stage", "center_fallback"),
        "base_point": p.get("base_point", [int(base_x), int(base_y)]),
        "center_hit_avoid": bool(p.get("center_hit_avoid", False)),
        "avoid_adjusted": bool(p.get("avoid_adjusted", False)),
        "search_radius": int(p.get("search_radius", 0)),
    }]

    point_info = {
        "box": None,
        "conf": None,
        "area": 0,
        "box_area": 0,
        "mask_area": 0,
        "class_name": target_class_name,
        "target_mask": None,
        "avoid_mask": avoid_mask,
        "x": int(p["x"]),
        "y": int(p["y"]),
        "mode": "center_fallback_no_plate",
        "fallback_ratio": parse_user_point_ratio(PLATE_FALLBACK_POINT_RATIO),
        "fallback_base_point": [int(base_x), int(base_y)],
        "center_hit_avoid": bool(p.get("center_hit_avoid", False)),
        "avoid_adjusted": bool(p.get("avoid_adjusted", False)),
        "avoid_mask_area": int((avoid_mask > 0).sum()) if avoid_mask is not None and avoid_mask.size > 0 else 0,
        "search_radius": int(p.get("search_radius", 0)),
    }

    point_meta = {
        "shrink_box": "",
        "raw_anchor": [int(base_x), int(base_y)],
        "inner_anchor": [int(p["x"]), int(p["y"])],
        "moved_to_inner": bool(p.get("avoid_adjusted", False)),
        "inner_dist": 0.0,
        "around_radius": "",
        "min_point_distance": "",
        "raw_candidates": candidate_points,
    }

    return candidate_points, point_info, point_meta


def is_no_plate_detected_error(error):
    """
    判断 get_target_from_yolo_result 抛出的异常是否属于“没识别到 plate”。
    """
    msg = str(error)
    return (
        "YOLO 没有识别到任何目标" in msg
        or "没有找到目标类别：plate" in msg
    )


# =========================
# SAM2 点选生成 mask
# =========================
def _sam_annotation_to_full_mask(annotation, h, w):
    bbox = annotation.bounding_box

    small_mask = np.asarray(annotation.mask)
    if small_mask.dtype != np.bool_:
        small_mask = small_mask > 0

    full_mask = np.zeros((h, w), dtype=np.uint8)

    x1 = max(0, int(bbox.xmin))
    y1 = max(0, int(bbox.ymin))

    mh, mw = small_mask.shape[:2]
    x2 = min(w, x1 + mw)
    y2 = min(h, y1 + mh)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        return None

    full_mask[y1:y2, x1:x2] = small_mask[:crop_h, :crop_w].astype(np.uint8) * 255
    return full_mask


def run_sam2_masks_by_point(image_path=None, x=None, y=None, model_name="sam2", image_rgb=None):
    """
    用单个点跑 SAM2。

    速度优化：
      - 如果传入 image_rgb，就不再重复读取图片。
      - image_path 只作为兼容旧调用的兜底。
    """
    if image_rgb is None:
        if image_path is None:
            raise RuntimeError("run_sam2_masks_by_point 需要 image_rgb 或 image_path")
        image_pil = Image.open(image_path).convert("RGB")
        image = np.asarray(image_pil)
    else:
        image = image_rgb

    h, w = image.shape[:2]

    request = osam.types.GenerateRequest(
        model=model_name,
        image=image,
        prompt=osam.types.Prompt(
            points=[[int(x), int(y)]],
            point_labels=[1],  # 1 = 前景点
        ),
    )

    response = osam.apis.generate(request=request)

    if not response.annotations:
        return []

    masks = []
    for annotation in response.annotations:
        full_mask = _sam_annotation_to_full_mask(annotation, h, w)
        if full_mask is not None:
            masks.append(full_mask)

    return masks


def run_sam2_masks_by_points(image_path=None, points=None, model_name="sam2", image_rgb=None):
    """
    用多个前景点一次性跑 SAM2。

    速度优化：
      - 一次 SAM2 调用同时传多个点。
      - 如果传入 image_rgb，就不再重复读取图片。

    注意：这里仍然不传 YOLO box，只传多个 point。
    """
    if points is None or len(points) == 0:
        return []

    if image_rgb is None:
        if image_path is None:
            raise RuntimeError("run_sam2_masks_by_points 需要 image_rgb 或 image_path")
        image_pil = Image.open(image_path).convert("RGB")
        image = np.asarray(image_pil)
    else:
        image = image_rgb

    h, w = image.shape[:2]

    sam_points = [[int(p["x"]), int(p["y"])] for p in points]

    request = osam.types.GenerateRequest(
        model=model_name,
        image=image,
        prompt=osam.types.Prompt(
            points=sam_points,
            point_labels=[1] * len(sam_points),
        ),
    )

    response = osam.apis.generate(request=request)

    if not response.annotations:
        return []

    masks = []
    for annotation in response.annotations:
        full_mask = _sam_annotation_to_full_mask(annotation, h, w)
        if full_mask is not None:
            masks.append(full_mask)

    return masks

def get_mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def score_sam_mask_weak(mask, image_shape, target_class_name, point_score=0.0):
    """
    对 SAM2 结果做弱评分。
    这里不强信 YOLO，只判断明显不合理的结果：太小、太大、局部碎片等。
    """
    h, w = image_shape[:2]
    image_area = max(1, h * w)

    mask_bin = mask > 0
    area = int(mask_bin.sum())
    area_ratio = area / image_area

    min_ratio = MIN_SAM_MASK_AREA_RATIO_BY_CLASS.get(target_class_name, 0.001)
    max_ratio = MAX_SAM_MASK_AREA_RATIO_BY_CLASS.get(target_class_name, 0.95)

    if area <= 0:
        return -9999.0, {
            "area": 0,
            "area_ratio": 0.0,
            "reason": "empty",
        }

    if area_ratio < min_ratio:
        return -1000.0 + area_ratio, {
            "area": area,
            "area_ratio": float(area_ratio),
            "reason": "too_small",
        }

    if area_ratio > max_ratio:
        return -900.0 - area_ratio, {
            "area": area,
            "area_ratio": float(area_ratio),
            "reason": "too_large",
        }

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), connectivity=8)
    component_count = max(0, num_labels - 1)
    largest_component_area = 0
    if component_count > 0:
        largest_component_area = int(stats[1:, cv2.CC_STAT_AREA].max())

    largest_ratio = largest_component_area / max(1, area)
    fragmentation_penalty = max(0.0, 1.0 - largest_ratio)

    bbox = get_mask_bbox(mask)
    bbox_area_ratio = 0.0
    if bbox is not None:
        bx1, by1, bx2, by2 = bbox
        bbox_area = max(1, (bx2 - bx1 + 1) * (by2 - by1 + 1))
        bbox_area_ratio = area / bbox_area

    if target_class_name == "plate":
        target_area_score = np.log1p(area_ratio * 100.0)
    else:
        target_area_score = np.log1p(area_ratio * 20.0)

    score = (
        target_area_score
        + 0.35 * float(point_score)
        + 0.35 * float(bbox_area_ratio)
        - 1.0 * fragmentation_penalty
    )

    return float(score), {
        "area": area,
        "area_ratio": float(area_ratio),
        "component_count": int(component_count),
        "largest_ratio": float(largest_ratio),
        "bbox_area_ratio": float(bbox_area_ratio),
        "reason": "ok",
    }


def save_best_mask_from_masks(masks, image_shape, output_mask_path, target_class_name, point_score=0.0):
    """
    从 SAM2 返回的多个 annotation mask 中选一个保存。
    这不是多点排除，只是同一次 SAM2 调用可能返回多个候选结果时，选一个弱评分最高的。
    """
    best = None

    for mask_idx, mask in enumerate(masks):
        sam_score, detail = score_sam_mask_weak(
            mask=mask,
            image_shape=image_shape,
            target_class_name=target_class_name,
            point_score=point_score,
        )

        item = {
            "mask": mask,
            "sam_score": float(sam_score),
            "sam_detail": detail,
            "mask_index": int(mask_idx),
        }

        if best is None or item["sam_score"] > best["sam_score"]:
            best = item

    if best is None:
        raise RuntimeError("SAM2 没有得到有效 mask")

    Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(best["mask"]).save(output_mask_path)

    return best["mask"], best


def run_sam2_by_user_ratio_point(
        image_path,
        user_point_ratio,
        output_mask_path,
        target_class_name="plate",
        model_name="sam2",
        image_rgb=None,
):
    """
    人工点副方案。

    如果 user_point_ratio 有值：
      1. 不做 YOLO 自动取点。
      2. 不做自动多点扩散。
      3. 不做候选点排除。
      4. 直接把比例点换算成原图真实坐标，传给 SAM2。

    注意：如果开启 FILL_PAPER_TO_PLATE，程序可能仍会跑 YOLO 来识别 A4纸并补回 plate，
    但这个 YOLO 只用于后处理补 A4纸区域，不参与 SAM2 取点。
    """
    if image_rgb is None:
        image_pil = Image.open(image_path).convert("RGB")
        image_rgb = np.asarray(image_pil)

    image_shape = image_rgb.shape[:2]

    xy = ratio_to_xy(user_point_ratio, image_shape)
    if xy is None:
        raise RuntimeError("USER_POINT_RATIO 为空，不能走人工点逻辑")

    x, y = xy

    masks = run_sam2_masks_by_point(
        image_path=image_path,
        image_rgb=image_rgb,
        x=x,
        y=y,
        model_name=model_name,
    )

    if not masks:
        raise RuntimeError("人工点 SAM2 没有生成任何 mask，请确认用户点是否在目标内部")

    mask, best = save_best_mask_from_masks(
        masks=masks,
        image_shape=image_shape,
        output_mask_path=output_mask_path,
        target_class_name=target_class_name,
        point_score=1.0,
    )

    ratio = parse_user_point_ratio(user_point_ratio)

    best.update({
        "x": int(x),
        "y": int(y),
        "mode": "user_ratio_point",
        "user_ratio": ratio,
        "used_points": [{
            "x": int(x),
            "y": int(y),
            "ratio_x": float(ratio[0]),
            "ratio_y": float(ratio[1]),
        }],
    })

    return mask, best


def run_sam2_by_candidate_points(
        image_path,
        candidate_points,
        output_mask_path,
        target_class_name="plate",
        model_name="sam2",
        image_rgb=None,
):
    """
    自动逻辑的混合方案：
      1. 优先把前 MULTI_POINT_COUNT 个点一次性传给 SAM2，通常只跑 1 次 SAM2。
      2. 如果多点结果不合理，并且 ENABLE_SINGLE_POINT_FALLBACK=True，才退回单点逐个尝试。

    速度优化：
      - 传入 image_rgb，避免重复读取图片。
      - 默认关闭单点 fallback，可减少失败图/复杂图的额外 SAM2 调用。
    """
    if image_rgb is None:
        image_pil = Image.open(image_path).convert("RGB")
        image_rgb = np.asarray(image_pil)

    image_shape = image_rgb.shape[:2]

    best = None
    tried = []

    # 1. 多点一次性 SAM2，正常情况下只跑这一次。
    multi_points = candidate_points[:MULTI_POINT_COUNT]

    if len(multi_points) >= 2:
        try:
            masks = run_sam2_masks_by_points(
                image_path=image_path,
                image_rgb=image_rgb,
                points=multi_points,
                model_name=model_name,
            )

            avg_point_score = float(np.mean([p.get("point_score", 0.0) for p in multi_points]))

            for mask_idx, mask in enumerate(masks):
                sam_score, detail = score_sam_mask_weak(
                    mask=mask,
                    image_shape=image_shape,
                    target_class_name=target_class_name,
                    point_score=avg_point_score,
                )

                item = {
                    "mask": mask,
                    "x": int(multi_points[0]["x"]),
                    "y": int(multi_points[0]["y"]),
                    "point_score": avg_point_score,
                    "sam_score": float(sam_score),
                    "sam_detail": detail,
                    "mask_index": int(mask_idx),
                    "mode": "multi_points",
                    "used_points": multi_points,
                }

                tried.append({
                    "mode": "multi_points",
                    "points": [(int(p["x"]), int(p["y"])) for p in multi_points],
                    "point_score": avg_point_score,
                    "sam_score": float(sam_score),
                    "sam_detail": detail,
                    "mask_index": int(mask_idx),
                })

                if best is None or item["sam_score"] > best["sam_score"]:
                    best = item

            if best is not None and best["sam_score"] > -100 and best["sam_detail"].get("reason") == "ok":
                Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(best["mask"]).save(output_mask_path)
                best["tried"] = tried
                return best["mask"], best

        except Exception as e:
            tried.append({
                "mode": "multi_points",
                "message": str(e),
                "sam_score": -9999.0,
            })

    # 2. 如果没有多点或者多点失败，且关闭了 fallback，就直接返回最佳结果或报错。
    if not ENABLE_SINGLE_POINT_FALLBACK:
        if best is not None:
            Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(best["mask"]).save(output_mask_path)
            best["tried"] = tried
            return best["mask"], best
        raise RuntimeError(f"多点 SAM2 未得到有效结果，且 ENABLE_SINGLE_POINT_FALLBACK=False，tried={tried[:5]}")

    # 3. 多点不理想，再单点逐个尝试。
    for p in candidate_points:
        x = int(p["x"])
        y = int(p["y"])
        point_score = float(p.get("point_score", 0.0))

        try:
            masks = run_sam2_masks_by_point(
                image_path=image_path,
                image_rgb=image_rgb,
                x=x,
                y=y,
                model_name=model_name,
            )
        except Exception as e:
            tried.append({
                "mode": "single_point",
                "x": x,
                "y": y,
                "point_score": point_score,
                "sam_score": -9999.0,
                "message": str(e),
            })
            continue

        if not masks:
            tried.append({
                "mode": "single_point",
                "x": x,
                "y": y,
                "point_score": point_score,
                "sam_score": -9999.0,
                "message": "no_mask",
            })
            continue

        for mask_idx, mask in enumerate(masks):
            sam_score, detail = score_sam_mask_weak(
                mask=mask,
                image_shape=image_shape,
                target_class_name=target_class_name,
                point_score=point_score,
            )

            item = {
                "mask": mask,
                "x": x,
                "y": y,
                "point_score": point_score,
                "sam_score": float(sam_score),
                "sam_detail": detail,
                "mask_index": int(mask_idx),
                "mode": "single_point",
                "used_points": [p],
            }

            tried.append({
                "mode": "single_point",
                "x": x,
                "y": y,
                "point_score": point_score,
                "sam_score": float(sam_score),
                "sam_detail": detail,
                "mask_index": int(mask_idx),
            })

            if best is None or item["sam_score"] > best["sam_score"]:
                best = item

            if sam_score > -100 and detail.get("reason") == "ok":
                Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(output_mask_path)
                item["tried"] = tried
                return mask, item

    if best is None or best["sam_score"] < -100:
        raise RuntimeError(f"多个候选点尝试后，SAM2 仍然没有得到合理 mask，tried={tried[:5]}")

    Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(best["mask"]).save(output_mask_path)

    best["tried"] = tried
    return best["mask"], best


# =========================
# Debug 图输出
# =========================
def save_debug_point_image(image_path, point_info, output_debug_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x = int(point_info["x"])
    y = int(point_info["y"])
    cls_name = point_info.get("class_name", "target")
    conf = point_info.get("conf")
    mode = point_info.get("mode", "auto")

    # 自动逻辑才有 YOLO box / shrink box
    if "box" in point_info and point_info["box"]:
        x1, y1, x2, y2 = point_info["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    if "shrink_box" in point_info and point_info["shrink_box"]:
        sx1, sy1, sx2, sy2 = point_info["shrink_box"]
        draw.rectangle([sx1, sy1, sx2, sy2], outline="orange", width=4)

    # 原始锚点：紫色
    if "raw_anchor" in point_info and point_info["raw_anchor"]:
        ax, ay = point_info["raw_anchor"]
        rr = 9
        draw.ellipse([ax - rr, ay - rr, ax + rr, ay + rr], fill="purple", outline="white", width=2)
        draw.text((ax + 10, ay - 10), "raw", fill="purple")

    # 内部修正锚点：绿色
    if "inner_anchor" in point_info and point_info["inner_anchor"]:
        ix, iy = point_info["inner_anchor"]
        rr = 9
        draw.ellipse([ix - rr, iy - rr, ix + rr, iy + rr], fill="green", outline="white", width=2)
        draw.text((ix + 10, iy - 10), "inner", fill="green")

    # 所有候选点：黄色小圆
    candidate_points = point_info.get("candidate_points", [])
    for idx, p in enumerate(candidate_points, start=1):
        px = int(p["x"])
        py = int(p["y"])
        rr = 7
        draw.ellipse([px - rr, py - rr, px + rr, py + rr], fill="yellow", outline="black", width=2)
        draw.text((px + 8, py - 8), str(idx), fill="yellow")

    # 最终采用点：蓝色大圆
    r = 13
    draw.ellipse([x - r, y - r, x + r, y + r], fill="blue", outline="white", width=3)

    if mode == "user_ratio_point":
        ratio = point_info.get("user_ratio")
        text = f"{cls_name} manual_ratio={ratio} selected=({x},{y})"
    else:
        if conf is None:
            text = f"{cls_name} selected=({x},{y})"
        else:
            text = f"{cls_name} conf={conf:.2f} selected=({x},{y})"

    if "sam_score" in point_info:
        text += f" sam_score={point_info['sam_score']:.3f}"

    draw.text((x + 16, y - 16), text, fill="blue")

    Path(output_debug_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_debug_path)


# =========================
# 后处理：paper 补回 plate
# =========================
def merge_paper_to_plate(plate_mask_path, paper_mask_path, output_path, close_kernel_size=15):
    plate = cv2.imread(str(plate_mask_path), cv2.IMREAD_GRAYSCALE)
    paper = cv2.imread(str(paper_mask_path), cv2.IMREAD_GRAYSCALE)

    if plate is None:
        raise RuntimeError(f"无法读取 plate mask: {plate_mask_path}")
    if paper is None:
        raise RuntimeError(f"无法读取 paper mask: {paper_mask_path}")

    final_bin = (plate > 0) | (paper > 0)
    final = final_bin.astype(np.uint8) * 255

    if close_kernel_size and close_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_kernel_size, close_kernel_size))
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, kernel)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), final)
    return final


# =========================
# 批量处理目录或单张图片
# =========================
def batch_process():
    input_path = Path(INPUT_DIR)
    output_root = Path(OUTPUT_ROOT)

    images = list_images(input_path, recursive=RECURSIVE)
    if not images:
        print(f"未找到可处理图片：{input_path}")
        return

    input_mode = "single_image" if input_path.is_file() else "directory"

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "process_log.csv"

    manual_ratio = parse_user_point_ratio(USER_POINT_RATIO)
    use_manual_point = manual_ratio is not None

    print(f"输入路径: {input_path}")
    print(f"输入模式: {'单张图片' if input_mode == 'single_image' else '目录批量'}")
    print(f"输出目录: {output_root.resolve()}")
    print(f"图片数量: {len(images)}")
    print(f"处理类别: {TARGET_CLASS_NAMES}")
    print(f"SAM_MODEL_NAME: {SAM_MODEL_NAME}")

    need_yolo = (
        (not use_manual_point)
        or (
            use_manual_point
            and FILL_PAPER_TO_PLATE
            and RUN_YOLO_FOR_PAPER_FILL_IN_MANUAL_MODE
            and "plate" in TARGET_CLASS_NAMES
        )
    )

    if use_manual_point:
        print("当前模式: 人工比例点模式，SAM2 直接使用人工比例点，不做自动多点取点")
        print(f"USER_POINT_RATIO: {manual_ratio}")
        if need_yolo:
            print("人工模式下仍会跑一次 YOLO：只用于识别 A4纸区域并补回 plate mask")
    else:
        print("当前模式: 自动模式，YOLO只做粗定位，SAM2只使用点，不使用YOLO框作为prompt")
        print(f"BOX_SHRINK_RATIO: {BOX_SHRINK_RATIO}")
        print(f"MAX_SAM_TRY_POINTS: {MAX_SAM_TRY_POINTS}")
        print(f"MULTI_POINT_COUNT: {MULTI_POINT_COUNT}")
        print(f"ENABLE_SINGLE_POINT_FALLBACK: {ENABLE_SINGLE_POINT_FALLBACK}")
        print(f"ANCHOR_INNER_SEARCH_RADIUS: {ANCHOR_INNER_SEARCH_RADIUS}")
        print(f"POINT_AROUND_RADIUS_PX: {POINT_AROUND_RADIUS_PX}")
        print(f"FALLBACK_TO_CENTER_WHEN_NO_PLATE: {FALLBACK_TO_CENTER_WHEN_NO_PLATE}")
        print(f"PLATE_FALLBACK_POINT_RATIO: {PLATE_FALLBACK_POINT_RATIO}")
        print(f"FALLBACK_AVOID_CLASS_NAMES: {FALLBACK_AVOID_CLASS_NAMES}")

    print(f"FILL_PAPER_TO_PLATE: {FILL_PAPER_TO_PLATE}")
    print(f"PAPER_CLASS_NAMES: {PAPER_CLASS_NAMES}")

    if need_yolo:
        model = YOLO(YOLO_MODEL_PATH)
        print(f"YOLO 类别: {model.names}")
    else:
        model = None

    rows = []
    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(images, start=1):
        print(f"\n[{idx}/{len(images)}] {image_path.name}")

        try:
            image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
            generated_masks = {}
            paper_fill_mask = None
            result = None

            # YOLO 只跑一次：自动模式用于粗定位；人工点模式下可选用于 A4纸补回。
            if need_yolo:
                results = model.predict(
                    source=str(image_path),
                    conf=YOLO_CONF,
                    imgsz=YOLO_IMGSZ,
                    verbose=False,
                )
                result = results[0]

                if FILL_PAPER_TO_PLATE and "plate" in TARGET_CLASS_NAMES:
                    paper_fill_mask = build_class_mask_from_yolo_result(
                        result=result,
                        class_names=PAPER_CLASS_NAMES,
                        use_box_if_no_mask=PAPER_FILL_USE_BOX_IF_NO_MASK,
                        dilate_kernel_size=PAPER_FILL_DILATE_KERNEL,
                        close_kernel_size=PAPER_FILL_CLOSE_KERNEL,
                    )

            if use_manual_point:
                # =========================
                # 人工比例点模式
                # =========================
                for cls_name in TARGET_CLASS_NAMES:
                    cls_dir = output_root / cls_name
                    mask_path = cls_dir / "masks" / f"{image_path.stem}_{cls_name}_mask.png"
                    debug_path = cls_dir / "debug_points" / f"{image_path.stem}_{cls_name}_manual_point.jpg"

                    sam_mask, sam_info = run_sam2_by_user_ratio_point(
                        image_path=str(image_path),
                        image_rgb=image_rgb,
                        user_point_ratio=manual_ratio,
                        output_mask_path=str(mask_path),
                        target_class_name=cls_name,
                        model_name=SAM_MODEL_NAME,
                    )

                    point_info = {
                        "class_name": cls_name,
                        "x": int(sam_info["x"]),
                        "y": int(sam_info["y"]),
                        "sam_score": float(sam_info["sam_score"]),
                        "sam_detail": sam_info["sam_detail"],
                        "mode": "user_ratio_point",
                        "user_ratio": manual_ratio,
                        "candidate_points": [],
                    }

                    paper_fill_info = {"filled": False, "paper_area": 0, "added_area": 0, "reason": "disabled"}
                    if cls_name == "plate" and FILL_PAPER_TO_PLATE:
                        raw_mask_path = None
                        if SAVE_RAW_PLATE_MASK_BEFORE_PAPER_FILL:
                            raw_mask_path = cls_dir / "masks_raw" / f"{image_path.stem}_{cls_name}_raw_sam_mask.png"

                        sam_mask, paper_fill_info = apply_paper_fill_to_plate_mask(
                            plate_mask=sam_mask,
                            paper_mask=paper_fill_mask,
                            output_mask_path=mask_path,
                            raw_mask_path=raw_mask_path,
                        )

                    point_info["paper_fill_info"] = paper_fill_info

                    save_debug_point_image(
                        image_path=str(image_path),
                        point_info=point_info,
                        output_debug_path=str(debug_path),
                    )

                    generated_masks[cls_name] = mask_path

                    print(
                        f"  {cls_name}: OK, manual_ratio={manual_ratio}, "
                        f"selected_point=({point_info['x']},{point_info['y']}), "
                        f"sam_score={point_info['sam_score']:.3f}, "
                        f"sam_detail={point_info['sam_detail']}"
                    )

                    rows.append({
                        "image": str(image_path),
                        "class": cls_name,
                        "status": "OK",
                        "mode": "user_ratio_point",
                        "user_ratio": str(manual_ratio),
                        "selected_x": point_info["x"],
                        "selected_y": point_info["y"],
                        "conf": "",
                        "box": "",
                        "shrink_box": "",
                        "raw_anchor": "",
                        "inner_anchor": "",
                        "moved_to_inner": "",
                        "around_radius": "",
                        "min_point_distance": "",
                        "box_area": "",
                        "mask_area_yolo": "",
                        "sam_score": f"{point_info['sam_score']:.6f}",
                        "sam_detail": str(point_info["sam_detail"]),
                        "paper_fill_info": str(paper_fill_info),
                        "candidate_points": "",
                        "mask_path": str(mask_path),
                        "debug_path": str(debug_path),
                        "message": "",
                    })

            else:
                # =========================
                # 自动模式：YOLO 粗定位 + 内部锚点 + 多点/单点 SAM2
                # =========================
                for cls_name in TARGET_CLASS_NAMES:
                    avoid_names = AVOID_BY_TARGET.get(cls_name, [])

                    try:
                        point_info = get_target_from_yolo_result(
                            result=result,
                            target_class_name=cls_name,
                            avoid_class_names=avoid_names,
                        )

                        candidate_points, point_meta = make_candidate_points_from_yolo_box(
                            image_rgb=image_rgb,
                            point_info=point_info,
                        )

                    except RuntimeError as yolo_target_error:
                        # 只有 plate 没识别到时，才使用 0.5,0.5 中心点兜底。
                        # paper/hole 等其他类别没识别到，仍然按失败处理。
                        if (
                            cls_name == "plate"
                            and FALLBACK_TO_CENTER_WHEN_NO_PLATE
                            and is_no_plate_detected_error(yolo_target_error)
                        ):
                            candidate_points, point_info, point_meta = make_candidate_point_from_center_fallback(
                                image_rgb=image_rgb,
                                result=result,
                                target_class_name=cls_name,
                            )
                        else:
                            raise

                    if not candidate_points:
                        raise RuntimeError("没有可用候选点")

                    cls_dir = output_root / cls_name
                    mask_path = cls_dir / "masks" / f"{image_path.stem}_{cls_name}_mask.png"
                    debug_path = cls_dir / "debug_points" / f"{image_path.stem}_{cls_name}_point.jpg"

                    if point_info.get("mode") == "center_fallback_no_plate":
                        # YOLO 未识别到 plate：只给 SAM2 一个兜底点。
                        fallback_point = candidate_points[0]
                        fallback_x = int(fallback_point["x"])
                        fallback_y = int(fallback_point["y"])

                        masks = run_sam2_masks_by_point(
                            image_path=str(image_path),
                            image_rgb=image_rgb,
                            x=fallback_x,
                            y=fallback_y,
                            model_name=SAM_MODEL_NAME,
                        )

                        if not masks:
                            raise RuntimeError(
                                f"中心兜底点 SAM2 没有生成任何 mask，point=({fallback_x},{fallback_y})"
                            )

                        sam_mask, sam_info = save_best_mask_from_masks(
                            masks=masks,
                            image_shape=image_rgb.shape[:2],
                            output_mask_path=str(mask_path),
                            target_class_name=cls_name,
                            point_score=float(fallback_point.get("point_score", 1.0)),
                        )

                        sam_info.update({
                            "x": fallback_x,
                            "y": fallback_y,
                            "mode": "center_fallback_no_plate",
                            "used_points": [fallback_point],
                        })
                    else:
                        sam_mask, sam_info = run_sam2_by_candidate_points(
                            image_path=str(image_path),
                            image_rgb=image_rgb,
                            candidate_points=candidate_points,
                            output_mask_path=str(mask_path),
                            target_class_name=cls_name,
                            model_name=SAM_MODEL_NAME,
                        )

                    point_info["x"] = int(sam_info["x"])
                    point_info["y"] = int(sam_info["y"])
                    point_info["sam_score"] = float(sam_info["sam_score"])
                    point_info["sam_detail"] = sam_info["sam_detail"]
                    point_info["candidate_points"] = candidate_points
                    point_info["shrink_box"] = point_meta["shrink_box"]
                    point_info["raw_anchor"] = point_meta["raw_anchor"]
                    point_info["inner_anchor"] = point_meta["inner_anchor"]
                    point_info["moved_to_inner"] = point_meta["moved_to_inner"]
                    point_info["inner_dist"] = point_meta["inner_dist"]
                    point_info["around_radius"] = point_meta["around_radius"]
                    point_info["min_point_distance"] = point_meta["min_point_distance"]
                    point_info["mode"] = sam_info.get("mode", "auto")

                    paper_fill_info = {"filled": False, "paper_area": 0, "added_area": 0, "reason": "disabled"}
                    if cls_name == "plate" and FILL_PAPER_TO_PLATE:
                        raw_mask_path = None
                        if SAVE_RAW_PLATE_MASK_BEFORE_PAPER_FILL:
                            raw_mask_path = cls_dir / "masks_raw" / f"{image_path.stem}_{cls_name}_raw_sam_mask.png"

                        sam_mask, paper_fill_info = apply_paper_fill_to_plate_mask(
                            plate_mask=sam_mask,
                            paper_mask=paper_fill_mask,
                            output_mask_path=mask_path,
                            raw_mask_path=raw_mask_path,
                        )

                    point_info["paper_fill_info"] = paper_fill_info

                    save_debug_point_image(
                        image_path=str(image_path),
                        point_info=point_info,
                        output_debug_path=str(debug_path),
                    )

                    generated_masks[cls_name] = mask_path

                    if point_info.get("mode") == "center_fallback_no_plate":
                        print(
                            f"  {cls_name}: OK, center_fallback_no_plate, "
                            f"base_point={point_info.get('fallback_base_point')}, "
                            f"selected_point=({point_info['x']},{point_info['y']}), "
                            f"center_hit_avoid={point_info.get('center_hit_avoid')}, "
                            f"avoid_adjusted={point_info.get('avoid_adjusted')}, "
                            f"avoid_mask_area={point_info.get('avoid_mask_area')}, "
                            f"search_radius={point_info.get('search_radius')}, "
                            f"sam_score={point_info['sam_score']:.3f}, "
                            f"sam_detail={point_info['sam_detail']}"
                        )
                    else:
                        print(
                            f"  {cls_name}: OK, "
                            f"selected_point=({point_info['x']},{point_info['y']}), "
                            f"conf={point_info['conf']:.3f}, "
                            f"box_area={point_info['box_area']}, "
                            f"raw_anchor={point_meta['raw_anchor']}, "
                            f"inner_anchor={point_meta['inner_anchor']}, "
                            f"moved_to_inner={point_meta['moved_to_inner']}, "
                            f"around_radius={point_meta['around_radius']}, "
                            f"sam_score={point_info['sam_score']:.3f}, "
                            f"sam_detail={point_info['sam_detail']}"
                        )

                    rows.append({
                        "image": str(image_path),
                        "class": cls_name,
                        "status": "OK",
                        "mode": point_info.get("mode", "auto"),
                        "user_ratio": "",
                        "selected_x": point_info["x"],
                        "selected_y": point_info["y"],
                        "conf": "" if point_info.get("conf") is None else f"{point_info['conf']:.6f}",
                        "box": "" if not point_info.get("box") else str(point_info["box"]),
                        "shrink_box": "" if point_meta.get("shrink_box") == "" else str(point_meta["shrink_box"]),
                        "raw_anchor": str(point_meta.get("raw_anchor", "")),
                        "inner_anchor": str(point_meta.get("inner_anchor", "")),
                        "moved_to_inner": str(point_meta.get("moved_to_inner", "")),
                        "around_radius": point_meta.get("around_radius", ""),
                        "min_point_distance": point_meta.get("min_point_distance", ""),
                        "box_area": point_info.get("box_area", ""),
                        "mask_area_yolo": point_info.get("mask_area", ""),
                        "sam_score": f"{point_info['sam_score']:.6f}",
                        "sam_detail": str(point_info["sam_detail"]),
                        "paper_fill_info": str(paper_fill_info),
                        "candidate_points": str(candidate_points),
                        "mask_path": str(mask_path),
                        "debug_path": str(debug_path),
                        "message": "",
                    })

            if MERGE_PAPER_TO_PLATE and "plate" in generated_masks and "paper" in generated_masks:
                final_path = output_root / "final_plate_masks" / f"{image_path.stem}_final_plate_mask.png"
                merge_paper_to_plate(
                    plate_mask_path=generated_masks["plate"],
                    paper_mask_path=generated_masks["paper"],
                    output_path=final_path,
                    close_kernel_size=15,
                )
                print(f"  final_plate: OK, {final_path}")

            success_count += 1

        except Exception as e:
            fail_count += 1
            msg = str(e)
            print(f"  处理失败: {msg}")
            rows.append({
                "image": str(image_path),
                "class": ";".join(TARGET_CLASS_NAMES),
                "status": "FAILED",
                "mode": "user_ratio_point" if use_manual_point else "auto",
                "user_ratio": str(manual_ratio) if use_manual_point else "",
                "selected_x": "",
                "selected_y": "",
                "conf": "",
                "box": "",
                "shrink_box": "",
                "raw_anchor": "",
                "inner_anchor": "",
                "moved_to_inner": "",
                "around_radius": "",
                "min_point_distance": "",
                "box_area": "",
                "mask_area_yolo": "",
                "sam_score": "",
                "sam_detail": "",
                "paper_fill_info": "",
                "candidate_points": "",
                "mask_path": "",
                "debug_path": "",
                "message": msg + "\n" + traceback.format_exc(limit=3),
            })

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "image",
            "class",
            "status",
            "mode",
            "user_ratio",
            "selected_x",
            "selected_y",
            "conf",
            "box",
            "shrink_box",
            "raw_anchor",
            "inner_anchor",
            "moved_to_inner",
            "around_radius",
            "min_point_distance",
            "box_area",
            "mask_area_yolo",
            "sam_score",
            "sam_detail",
            "paper_fill_info",
            "candidate_points",
            "mask_path",
            "debug_path",
            "message",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n处理完成")
    print(f"成功图片数: {success_count}")
    print(f"失败图片数: {fail_count}")
    print(f"日志文件: {log_path.resolve()}")


def str2bool(value):
    """
    argparse 用的布尔解析。
    支持：true/false、1/0、yes/no、y/n。
    """
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"无法解析布尔值：{value}")


def parse_name_list(value):
    """
    解析逗号分隔的类别名，例如："plate,paper"。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    text = text.replace("，", ",").replace(";", ",")
    return [x.strip() for x in text.split(",") if x.strip()]


def apply_cli_args():
    """
    命令行参数。

    重点：INPUT_DIR / YOLO_MODEL_PATH / OUTPUT_ROOT 已放到这里的 default 里。
    测试时可以直接改下面 default；打包后通过命令行传参覆盖。
    """
    parser = argparse.ArgumentParser()

    # =========================
    # 常用输入参数：测试时主要改这里的 default
    # =========================
    parser.add_argument(
        "--input-dir",
        default=r"D:\Desktop\现场余料图",
        # default=r"D:\lantek\project-python\p2r\test\batch_output\val\B2212013-001.jpg",
        help="图片输入目录或单张图片路径。传目录则批量处理，传图片文件则只处理这一张。",
    )
    parser.add_argument(
        "--yolo-model-path",
        default=r".\best2.pt",
        help="YOLO 模型路径。",
    )
    parser.add_argument(
        "--output-root",
        default=r"batch_output\e",
        help="输出目录。",
    )

    # =========================
    # 识别参数
    # =========================
    parser.add_argument("--target-class-names", default="plate", help="要处理的类别，逗号分隔，例如 plate 或 plate,paper")
    parser.add_argument("--sam-model-name", default="sam2:tiny", help="SAM2 模型名，例如 sam2 或 sam2:large")
    parser.add_argument("--yolo-conf", type=float, default=0.35, help="YOLO 置信度")
    parser.add_argument("--yolo-imgsz", type=int, default=1280, help="YOLO 推理图片尺寸")
    parser.add_argument("--recursive", type=str2bool, default=False, help="输入为目录时是否递归处理子目录")

    # =========================
    # 人工比例点副方案
    # =========================
    parser.add_argument(
        "--user-point-ratio",
        default=None,
        help="人工点比例，格式 ratio_x,ratio_y。例如前端图 1024x2048 上点 500x1000，则传 0.48828125,0.48828125。传了就跳过自动取点。",
    )

    # =========================
    # 自动取点参数
    # =========================
    parser.add_argument("--box-shrink-ratio", type=float, default=0.22, help="YOLO 框向内缩小比例")
    parser.add_argument("--candidate-grid-size", type=int, default=3, help="原始锚点候选网格大小")
    parser.add_argument("--max-sam-try-points", type=int, default=3, help="最多保留多少个候选点给 SAM2")
    parser.add_argument("--multi-point-count", type=int, default=3, help="多点模式一次传给 SAM2 的点数")
    parser.add_argument("--enable-single-point-fallback", type=str2bool, default=False, help="多点失败后是否逐个单点兜底。False 更快，True 更稳。")
    parser.add_argument("--anchor-inner-search-radius", type=int, default=180, help="锚点向钢板内部修正的搜索半径")
    parser.add_argument("--anchor-inner-min-dist", type=float, default=10.0, help="锚点内部修正的最小距离阈值")
    parser.add_argument("--point-around-radius-px", type=int, default=-1, help="围绕内部锚点扩散的固定半径；-1 表示自动计算")
    parser.add_argument("--point-around-radius-ratio", type=float, default=0.18, help="自动计算扩散半径时使用的比例")
    parser.add_argument("--point-around-radius-min-px", type=int, default=50, help="自动扩散半径最小值")
    parser.add_argument("--point-around-radius-max-px", type=int, default=180, help="自动扩散半径最大值")
    parser.add_argument("--point-clean-window-size", type=int, default=51, help="候选点局部干净程度评分窗口")

    # =========================
    # YOLO 未识别到 plate 时的中心点兜底参数
    # =========================
    parser.add_argument("--fallback-to-center-when-no-plate", type=str2bool, default=True, help="YOLO 未识别到 plate 时是否用 0.5,0.5 中心点兜底")
    parser.add_argument("--plate-fallback-point-ratio", default="0.5,0.5", help="plate 未识别到时的兜底比例点，默认 0.5,0.5")
    parser.add_argument("--fallback-avoid-class-names", default="paper,hole", help="中心兜底点需要避开的类别，逗号分隔，默认 paper,hole")
    parser.add_argument("--fallback-avoid-dilate-kernel", type=int, default=35, help="中心兜底避让 paper/hole 时的膨胀核大小")
    parser.add_argument("--fallback-point-search-step-px", type=int, default=40, help="中心点落在 paper/hole 上时，向外搜索的步长")
    parser.add_argument("--fallback-point-search-max-radius-ratio", type=float, default=0.45, help="中心点避让搜索最大半径比例")
    parser.add_argument("--fallback-point-search-angle-count", type=int, default=32, help="中心点避让搜索每个圆环的采样点数")

    # =========================
    # A4纸补回 plate 参数
    # =========================
    parser.add_argument("--fill-paper-to-plate", type=str2bool, default=True, help="是否把 A4纸区域补回 plate mask")
    parser.add_argument("--paper-class-names", default="paper", help="A4纸类别名，逗号分隔")
    parser.add_argument("--paper-fill-dilate-kernel", type=int, default=9, help="A4纸补回前膨胀核大小，0 表示不膨胀")
    parser.add_argument("--paper-fill-close-kernel", type=int, default=15, help="A4纸补回前闭运算核大小，0 表示不闭运算")
    parser.add_argument("--paper-fill-use-box-if-no-mask", type=str2bool, default=True, help="没有 paper mask 时是否用 paper box 兜底补回")
    parser.add_argument("--save-raw-plate-mask-before-paper-fill", type=str2bool, default=True, help="补 A4纸前是否保存原始 plate mask")
    parser.add_argument("--run-yolo-for-paper-fill-in-manual-mode", type=str2bool, default=True, help="人工点模式下是否仍跑 YOLO 用于 A4纸补回")

    args = parser.parse_args()

    global INPUT_DIR, OUTPUT_ROOT, YOLO_MODEL_PATH, TARGET_CLASS_NAMES, SAM_MODEL_NAME
    global YOLO_CONF, YOLO_IMGSZ, RECURSIVE, USER_POINT_RATIO
    global BOX_SHRINK_RATIO, CANDIDATE_GRID_SIZE, MAX_SAM_TRY_POINTS, MULTI_POINT_COUNT
    global ENABLE_SINGLE_POINT_FALLBACK, ANCHOR_INNER_SEARCH_RADIUS, ANCHOR_INNER_MIN_DIST
    global POINT_AROUND_RADIUS_PX, POINT_AROUND_RADIUS_RATIO, POINT_AROUND_RADIUS_MIN_PX, POINT_AROUND_RADIUS_MAX_PX
    global POINT_CLEAN_WINDOW_SIZE
    global FALLBACK_TO_CENTER_WHEN_NO_PLATE, PLATE_FALLBACK_POINT_RATIO, FALLBACK_AVOID_CLASS_NAMES
    global FALLBACK_AVOID_DILATE_KERNEL, FALLBACK_POINT_SEARCH_STEP_PX
    global FALLBACK_POINT_SEARCH_MAX_RADIUS_RATIO, FALLBACK_POINT_SEARCH_ANGLE_COUNT
    global FILL_PAPER_TO_PLATE, PAPER_CLASS_NAMES, PAPER_FILL_DILATE_KERNEL, PAPER_FILL_CLOSE_KERNEL
    global PAPER_FILL_USE_BOX_IF_NO_MASK, SAVE_RAW_PLATE_MASK_BEFORE_PAPER_FILL, RUN_YOLO_FOR_PAPER_FILL_IN_MANUAL_MODE

    INPUT_DIR = args.input_dir
    YOLO_MODEL_PATH = args.yolo_model_path
    OUTPUT_ROOT = args.output_root
    TARGET_CLASS_NAMES = parse_name_list(args.target_class_names)
    SAM_MODEL_NAME = args.sam_model_name
    YOLO_CONF = float(args.yolo_conf)
    YOLO_IMGSZ = int(args.yolo_imgsz)
    RECURSIVE = bool(args.recursive)
    USER_POINT_RATIO = args.user_point_ratio

    BOX_SHRINK_RATIO = float(args.box_shrink_ratio)
    CANDIDATE_GRID_SIZE = int(args.candidate_grid_size)
    MAX_SAM_TRY_POINTS = int(args.max_sam_try_points)
    MULTI_POINT_COUNT = int(args.multi_point_count)
    ENABLE_SINGLE_POINT_FALLBACK = bool(args.enable_single_point_fallback)
    ANCHOR_INNER_SEARCH_RADIUS = int(args.anchor_inner_search_radius)
    ANCHOR_INNER_MIN_DIST = float(args.anchor_inner_min_dist)
    POINT_AROUND_RADIUS_PX = None if int(args.point_around_radius_px) < 0 else int(args.point_around_radius_px)
    POINT_AROUND_RADIUS_RATIO = float(args.point_around_radius_ratio)
    POINT_AROUND_RADIUS_MIN_PX = int(args.point_around_radius_min_px)
    POINT_AROUND_RADIUS_MAX_PX = int(args.point_around_radius_max_px)
    POINT_CLEAN_WINDOW_SIZE = int(args.point_clean_window_size)

    FALLBACK_TO_CENTER_WHEN_NO_PLATE = bool(args.fallback_to_center_when_no_plate)
    PLATE_FALLBACK_POINT_RATIO = parse_user_point_ratio(args.plate_fallback_point_ratio)
    FALLBACK_AVOID_CLASS_NAMES = parse_name_list(args.fallback_avoid_class_names)
    FALLBACK_AVOID_DILATE_KERNEL = int(args.fallback_avoid_dilate_kernel)
    FALLBACK_POINT_SEARCH_STEP_PX = int(args.fallback_point_search_step_px)
    FALLBACK_POINT_SEARCH_MAX_RADIUS_RATIO = float(args.fallback_point_search_max_radius_ratio)
    FALLBACK_POINT_SEARCH_ANGLE_COUNT = int(args.fallback_point_search_angle_count)

    FILL_PAPER_TO_PLATE = bool(args.fill_paper_to_plate)
    PAPER_CLASS_NAMES = parse_name_list(args.paper_class_names)
    PAPER_FILL_DILATE_KERNEL = int(args.paper_fill_dilate_kernel)
    PAPER_FILL_CLOSE_KERNEL = int(args.paper_fill_close_kernel)
    PAPER_FILL_USE_BOX_IF_NO_MASK = bool(args.paper_fill_use_box_if_no_mask)
    SAVE_RAW_PLATE_MASK_BEFORE_PAPER_FILL = bool(args.save_raw_plate_mask_before_paper_fill)
    RUN_YOLO_FOR_PAPER_FILL_IN_MANUAL_MODE = bool(args.run_yolo_for_paper_fill_in_manual_mode)


if __name__ == "__main__":
    apply_cli_args()
    batch_process()
