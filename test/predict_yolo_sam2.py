from ultralytics import YOLO, SAM
from pathlib import Path
import cv2
import numpy as np
import sys
import torch


# 类别名称
PLATE_CLASS_NAME = "plate"
PAPER_CLASS_NAME = "paper"
HOLE_CLASS_NAME = "hole"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_image_paths(source: str):
    source_path = Path(source)

    if source_path.is_file():
        return [source_path]

    exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    image_paths = []
    for ext in exts:
        image_paths.extend(source_path.glob(f"*{ext}"))
        image_paths.extend(source_path.glob(f"*{ext.upper()}"))

    return sorted(image_paths)


def get_best_plate_box(yolo_result, conf_min=0.15):
    """
    从 YOLO 结果里取最可信的 plate box。
    策略：
    1. 只要 class == plate
    2. conf >= conf_min
    3. 优先取面积最大的钢板框
    """
    if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
        return None

    names = yolo_result.names
    boxes = yolo_result.boxes

    best_box = None
    best_area = 0

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        conf = float(boxes.conf[i].item())
        cls_name = names.get(cls_id, str(cls_id))

        if cls_name != PLATE_CLASS_NAME:
            continue

        if conf < conf_min:
            continue

        xyxy = boxes.xyxy[i].detach().cpu().numpy().astype(float)
        x1, y1, x2, y2 = xyxy
        area = max(0, x2 - x1) * max(0, y2 - y1)

        if area > best_area:
            best_area = area
            best_box = [int(x1), int(y1), int(x2), int(y2)]

    return best_box


def expand_box(box, image_shape, expand_ratio=0.03):
    """
    适当扩大 YOLO 框，避免 YOLO 框没有完全包住钢板，导致 SAM2 也只分割一半。
    expand_ratio=0.03 表示四周扩 3%。
    """
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box

    bw = x2 - x1
    bh = y2 - y1

    dx = int(bw * expand_ratio)
    dy = int(bh * expand_ratio)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w - 1, x2 + dx)
    y2 = min(h - 1, y2 + dy)

    return [x1, y1, x2, y2]


def get_sam_mask(sam_result):
    """
    从 SAM2 结果里取 mask。
    如果有多个 mask，取面积最大的。
    """
    if sam_result is None:
        return None

    if len(sam_result) == 0:
        return None

    r = sam_result[0]

    if r.masks is None or r.masks.data is None:
        return None

    masks = r.masks.data.detach().cpu().numpy()

    if masks.ndim == 2:
        masks = masks[None, :, :]

    best_mask = None
    best_area = 0

    for mask in masks:
        binary = (mask > 0.5).astype(np.uint8)
        area = int(binary.sum())

        if area > best_area:
            best_area = area
            best_mask = binary

    return best_mask


def clean_mask(mask, min_area=3000):
    """
    简单清洗 SAM2 mask：
    1. 保留最大外轮廓
    2. 去掉小碎片
    3. 做一次闭运算，补小缺口
    """
    mask_u8 = (mask * 255).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None

    contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    if not contours:
        return None, None

    max_contour = max(contours, key=cv2.contourArea)

    clean = np.zeros_like(mask_u8)
    cv2.drawContours(clean, [max_contour], -1, 255, thickness=-1)

    return clean, max_contour


def draw_result(image, yolo_box, sam_contour):
    vis = image.copy()

    if yolo_box is not None:
        x1, y1, x2, y2 = yolo_box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            vis,
            "YOLO plate box",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    if sam_contour is not None:
        cv2.drawContours(vis, [sam_contour], -1, (0, 0, 255), 3)
        cv2.putText(
            vis,
            "SAM2 refined contour",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    return vis


def process_one_image(image_path: Path, yolo_model, sam_model, output_dir: Path):
    print(f"\n处理图片: {image_path}")

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"读取失败: {image_path}")
        return

    # 1. YOLO 先检测 plate
    yolo_results = yolo_model.predict(
        source=str(image_path),
        imgsz=1280,
        conf=0.15,
        iou=0.5,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    yolo_result = yolo_results[0]

    plate_box = get_best_plate_box(yolo_result, conf_min=0.15)

    if plate_box is None:
        print("未检测到 plate，跳过 SAM2")
        return

    print(f"YOLO plate box: {plate_box}")

    # 2. 适当扩大 box
    prompt_box = expand_box(plate_box, image.shape, expand_ratio=0.04)

    print(f"SAM2 prompt box: {prompt_box}")

    # 3. SAM2 根据 YOLO 的 box 精修 mask
    sam_results = sam_model(
        str(image_path),
        bboxes=prompt_box,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )

    sam_mask = get_sam_mask(sam_results)

    if sam_mask is None:
        print("SAM2 未输出 mask")
        return

    # 4. 清洗 mask，提取轮廓
    clean, contour = clean_mask(sam_mask, min_area=3000)

    if clean is None or contour is None:
        print("mask 清洗后没有有效轮廓")
        return

    # 5. 保存结果
    stem = image_path.stem

    mask_path = output_dir / f"{stem}_sam2_mask.png"
    vis_path = output_dir / f"{stem}_yolo_sam2_vis.jpg"

    cv2.imwrite(str(mask_path), clean)

    vis = draw_result(image, prompt_box, contour)
    cv2.imwrite(str(vis_path), vis)

    print(f"保存 mask: {mask_path}")
    print(f"保存可视化: {vis_path}")


def main():
    model_path = "../best2.pt"
    source = sys.argv[1] if len(sys.argv) > 1 else "images/val"

    output_dir = Path("predict_out/yolo_sam2_vis")
    ensure_dir(output_dir)

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # 训练好的 YOLO 模型
    yolo_model = YOLO(model_path)

    # SAM2 模型
    # 显存小/先测试：sam2.1_t.pt
    # 效果更强但更吃显存：sam2.1_b.pt / sam2.1_l.pt
    sam_model = SAM("sam2.1_t.pt")

    image_paths = get_image_paths(source)

    if not image_paths:
        print(f"没有找到图片: {source}")
        return

    print(f"共找到 {len(image_paths)} 张图片")

    for image_path in image_paths:
        process_one_image(image_path, yolo_model, sam_model, output_dir)


if __name__ == "__main__":
    main()