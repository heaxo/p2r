from pathlib import Path
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
INPUT_DIR = r"/batch_output/val"
YOLO_MODEL_PATH = r"../best2.pt"
OUTPUT_ROOT = r"batch_yolo_sam2_point_output4"

# 要处理的类别。只想处理钢板就写 ["plate"]；只处理A4纸就写 ["paper"]
TARGET_CLASS_NAMES = ["paper"]

# SAM2 模型：sam2 = LabelMe balanced；sam2:large = LabelMe accuracy
SAM_MODEL_NAME = "sam2"

YOLO_CONF = 0.35
YOLO_IMGSZ = 1280
RECURSIVE = False

# 如果处理 plate，取点时避开 paper/hole，避免点落在A4纸或孔上
AVOID_BY_TARGET = {
    "plate": ["paper", "hole"],
    "paper": [],
    "hole": [],
}

# 是否在同时处理 plate 和 paper 时，额外输出 final_plate_mask = plate OR paper
MERGE_PAPER_TO_PLATE = True

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# YOLO 最佳点相关函数
# =========================
def _get_mask_from_yolo_result(result, index, h, w):
    """
    从 YOLO seg 结果中取单个实例 mask，并缩放到原图尺寸。
    如果你的 best2.pt 不是 seg 模型，则 result.masks 会是 None。
    """
    if result.masks is None:
        return None

    mask = result.masks.data[index].cpu().numpy()

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return (mask > 0.5).astype(np.uint8)


def get_best_point_from_yolo_result(
        result,
        image_path,
        target_class_name="plate",
        avoid_class_names=None,
):
    """
    从单张图片的 YOLO 结果里取一个最适合给 SAM2 的前景点。

    优先使用 YOLO seg mask 的距离变换，取离边界最远的点。
    如果 YOLO 不是分割模型，没有 masks，则退回到 box 中心点。
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
    avoid_class_ids = {cls_id for cls_id, cls_name in names.items() if cls_name in avoid_class_names}

    avoid_mask = np.zeros((h, w), dtype=np.uint8)
    if result.masks is not None and avoid_class_ids:
        for i, cls_id in enumerate(classes):
            if cls_id in avoid_class_ids:
                m = _get_mask_from_yolo_result(result, i, h, w)
                if m is not None:
                    avoid_mask = np.maximum(avoid_mask, m)

    candidates = []

    for i, cls_id in enumerate(classes):
        if cls_id != target_class_id:
            continue

        x1, y1, x2, y2 = boxes[i]
        score = float(confs[i])

        if result.masks is not None:
            target_mask = _get_mask_from_yolo_result(result, i, h, w)
            if target_mask is None:
                continue

            target_mask_bool = target_mask.astype(bool)
            if avoid_mask.sum() > 0:
                target_mask_bool = target_mask_bool & (~avoid_mask.astype(bool))

            target_mask_u8 = target_mask_bool.astype(np.uint8)
            area = int(target_mask_u8.sum())
            if area <= 0:
                continue

            image_rgb = np.asarray(Image.open(image_path).convert("RGB"))

            best_x, best_y, max_dist, max_score = choose_best_point_robust(
                image_rgb=image_rgb,
                target_mask=target_mask,
                avoid_mask=avoid_mask,
                edge_window_size=51,
                erode_kernel_size=31
            )

            candidates.append({
                "x": int(best_x),
                "y": int(best_y),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "conf": score,
                "area": area,
                "max_dist": float(max_dist),
                "class_name": target_class_name,
                "used_mask": True,
            })
        else:
            # 检测模型没有 mask，只能退回 box 中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            area = max(0, int((x2 - x1) * (y2 - y1)))

            candidates.append({
                "x": cx,
                "y": cy,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "conf": score,
                "area": area,
                "max_dist": 0.0,
                "class_name": target_class_name,
                "used_mask": False,
            })

    if not candidates:
        detected = sorted({names[int(cls_id)] for cls_id in classes})
        raise RuntimeError(f"没有找到可用目标点：{target_class_name}，本图识别到：{detected}")

    # 同类多个实例时，默认取面积最大的那个
    return max(candidates, key=lambda item: item["area"])


def save_debug_point_image(image_path, point_info, output_debug_path):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    x = point_info["x"]
    y = point_info["y"]
    x1, y1, x2, y2 = point_info["box"]
    cls_name = point_info["class_name"]
    conf = point_info["conf"]

    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

    r = 12
    draw.ellipse([x - r, y - r, x + r, y + r], fill="blue", outline="white", width=3)
    draw.text((x + 15, y - 15), f"{cls_name} {conf:.2f} ({x},{y})", fill="blue")

    Path(output_debug_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(output_debug_path)


# =========================
# SAM2 点选生成 mask
# =========================
def run_sam2_by_point(image_path, x, y, output_mask_path, model_name="sam2"):
    image_pil = Image.open(image_path).convert("RGB")
    image = np.asarray(image_pil)
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
        raise RuntimeError("SAM2 没有生成任何 mask，请检查点是否落在目标内部")

    # 默认取第一个结果
    annotation = response.annotations[0]
    bbox = annotation.bounding_box

    small_mask = np.asarray(annotation.mask)
    if small_mask.dtype != np.bool_:
        small_mask = small_mask > 0

    full_mask = np.zeros((h, w), dtype=np.uint8)

    x1 = max(0, int(bbox.xmin))
    y1 = max(0, int(bbox.ymin))

    # 不强依赖 bbox 是否包含右下角，直接用 small_mask 实际尺寸贴回去，更稳
    mh, mw = small_mask.shape[:2]
    x2 = min(w, x1 + mw)
    y2 = min(h, y1 + mh)

    crop_w = x2 - x1
    crop_h = y2 - y1
    if crop_w <= 0 or crop_h <= 0:
        raise RuntimeError("SAM2 返回的 mask bbox 无效")

    full_mask[y1:y2, x1:x2] = small_mask[:crop_h, :crop_w].astype(np.uint8) * 255

    Path(output_mask_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(full_mask).save(output_mask_path)

    return full_mask


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
# 批量处理目录
# =========================
def list_images(input_dir, recursive=False):
    input_dir = Path(input_dir)
    if recursive:
        return sorted([p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    return sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


import cv2
import numpy as np


def choose_best_point_robust(
        image_rgb,
        target_mask,
        avoid_mask=None,
        edge_window_size=51,
        erode_kernel_size=31
):
    """
    在 target_mask 中选择一个更适合 SAM2 的点。
    不依赖文字颜色，而是避开边缘多、纹理复杂的位置。

    image_rgb: RGB 原图，numpy数组
    target_mask: 0/1 mask，例如 plate mask
    avoid_mask: 0/1 mask，例如 paper + hole
    """

    h, w = target_mask.shape[:2]

    target = (target_mask > 0).astype(np.uint8)

    if avoid_mask is not None:
        target = target & (~(avoid_mask > 0))

    if target.sum() <= 0:
        raise RuntimeError("target_mask 扣除 avoid_mask 后为空，无法取点")

    # 1. 先腐蚀一下，避免点落到钢板边缘附近
    if erode_kernel_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (erode_kernel_size, erode_kernel_size)
        )
        inner = cv2.erode(target, kernel, iterations=1)

        # 如果腐蚀过头，就退回原 mask
        if inner.sum() > 500:
            target = inner

    # 2. 距离变换：越靠近目标内部，分数越高
    dist = cv2.distanceTransform(target * 255, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    # 3. 计算图像边缘：文字、划线、锈斑边界通常边缘多
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, 50, 150)
    edges = (edges > 0).astype(np.float32)

    # 4. 计算局部边缘密度
    k = edge_window_size
    edge_density = cv2.boxFilter(
        edges,
        ddepth=-1,
        ksize=(k, k),
        normalize=True
    )

    # 5. 计算局部灰度方差：纹理越复杂，方差越高
    gray_f = gray.astype(np.float32) / 255.0

    mean = cv2.boxFilter(
        gray_f,
        ddepth=-1,
        ksize=(k, k),
        normalize=True
    )

    mean_sq = cv2.boxFilter(
        gray_f * gray_f,
        ddepth=-1,
        ksize=(k, k),
        normalize=True
    )

    variance = mean_sq - mean * mean
    variance = np.clip(variance, 0, None)
    var_norm = cv2.normalize(variance, None, 0, 1.0, cv2.NORM_MINMAX)

    # 6. 综合打分
    # 距离边界越远越好；边缘密度越高越差；局部纹理变化越大越差
    score = (
        dist_norm
        - 0.9 * edge_density
        - 0.5 * var_norm
    )

    # 只允许在 target 区域内取点
    score[target == 0] = -9999

    _, max_score, _, max_loc = cv2.minMaxLoc(score)

    x, y = max_loc

    # 如果分数异常，退回到距离变换最大点
    if max_score <= -1000:
        _, max_dist, _, max_loc = cv2.minMaxLoc(dist)
        x, y = max_loc
        max_score = 0
    else:
        # 当前选中点距离边界的距离
        max_dist = dist[y, x]

    return int(x), int(y), float(max_dist), float(max_score)



def batch_process():
    input_dir = Path(INPUT_DIR)
    output_root = Path(OUTPUT_ROOT)

    images = list_images(input_dir, recursive=RECURSIVE)
    if not images:
        print(f"未找到图片：{input_dir}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "process_log.csv"

    print(f"图片目录: {input_dir}")
    print(f"输出目录: {output_root.resolve()}")
    print(f"图片数量: {len(images)}")
    print(f"处理类别: {TARGET_CLASS_NAMES}")

    model = YOLO(YOLO_MODEL_PATH)
    print(f"YOLO 类别: {model.names}")

    rows = []
    success_count = 0
    fail_count = 0

    for idx, image_path in enumerate(images, start=1):
        print(f"\n[{idx}/{len(images)}] {image_path.name}")

        try:
            # 一张图只跑一次 YOLO，多个类别复用 result
            results = model.predict(
                source=str(image_path),
                conf=YOLO_CONF,
                imgsz=YOLO_IMGSZ,
                verbose=False,
            )
            result = results[0]

            generated_masks = {}

            for cls_name in TARGET_CLASS_NAMES:
                avoid_names = AVOID_BY_TARGET.get(cls_name, [])

                point_info = get_best_point_from_yolo_result(
                    result=result,
                    image_path=image_path,
                    target_class_name=cls_name,
                    avoid_class_names=avoid_names,
                )

                cls_dir = output_root / cls_name
                mask_path = cls_dir / "masks" / f"{image_path.stem}_{cls_name}_mask.png"
                debug_path = cls_dir / "debug_points" / f"{image_path.stem}_{cls_name}_point.jpg"

                save_debug_point_image(
                    image_path=str(image_path),
                    point_info=point_info,
                    output_debug_path=str(debug_path),
                )

                run_sam2_by_point(
                    image_path=str(image_path),
                    x=point_info["x"],
                    y=point_info["y"],
                    output_mask_path=str(mask_path),
                    model_name=SAM_MODEL_NAME,
                )

                generated_masks[cls_name] = mask_path

                print(
                    f"  {cls_name}: OK, point=({point_info['x']},{point_info['y']}), "
                    f"conf={point_info['conf']:.3f}, area={point_info['area']}"
                )

                rows.append({
                    "image": str(image_path),
                    "class": cls_name,
                    "status": "OK",
                    "x": point_info["x"],
                    "y": point_info["y"],
                    "conf": f"{point_info['conf']:.6f}",
                    "area": point_info["area"],
                    "mask_path": str(mask_path),
                    "debug_path": str(debug_path),
                    "message": "",
                })

            # 如果同时生成了 plate 和 paper，可以输出补回 A4 纸后的 final plate mask
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
                "x": "",
                "y": "",
                "conf": "",
                "area": "",
                "mask_path": "",
                "debug_path": "",
                "message": msg + "\n" + traceback.format_exc(limit=2),
            })

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["image", "class", "status", "x", "y", "conf", "area", "mask_path", "debug_path", "message"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n处理完成")
    print(f"成功图片数: {success_count}")
    print(f"失败图片数: {fail_count}")
    print(f"日志文件: {log_path.resolve()}")


if __name__ == "__main__":
    batch_process()
