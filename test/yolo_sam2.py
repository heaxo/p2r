import os
import numpy as np
from PIL import Image, ImageDraw

from ultralytics import YOLO

import osam.apis
import osam.types


from pathlib import Path


def batch_run_sam2_by_yolo_box(
        input_dir,
        yolo_model_path,
        target_class_name="paper",
        output_root="batch_output",
        sam_model_name="sam2",
        yolo_conf=0.2,
        yolo_imgsz=1280,
        recursive=False
):
    input_dir = Path(input_dir)
    output_root = Path(output_root)

    mask_dir = output_root / "masks"
    debug_dir = output_root / "debug"

    mask_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if recursive:
        image_files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    else:
        image_files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not image_files:
        print(f"未找到图片文件：{input_dir}")
        return

    total = len(image_files)
    success = 0
    failed = 0

    print(f"共找到 {total} 张图片，开始处理...")

    for idx, image_path in enumerate(image_files, start=1):
        print(f"\n[{idx}/{total}] 正在处理: {image_path}")

        stem = image_path.stem

        output_mask_path = mask_dir / f"{stem}_mask.png"
        output_debug_path = debug_dir / f"{stem}_debug.jpg"

        try:
            run_sam2_by_yolo_box(
                image_path=str(image_path),
                yolo_model_path=yolo_model_path,
                target_class_name=target_class_name,
                output_mask_path=str(output_mask_path),
                output_debug_path=str(output_debug_path),
                sam_model_name=sam_model_name,
                yolo_conf=yolo_conf,
                yolo_imgsz=yolo_imgsz
            )
            print(f"处理成功: {image_path.name}")
            success += 1

        except Exception as e:
            print(f"处理失败: {image_path.name}")
            print(f"原因: {e}")
            failed += 1

    print("\n处理完成")
    print(f"总数: {total}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"输出目录: {output_root.resolve()}")


def get_yolo_box(
        image_path,
        yolo_model_path,
        target_class_name=None,
        conf=0.25,
        imgsz=1280
):
    """
    用 YOLO best2.pt 找目标框。

    target_class_name:
        None       = 不指定类别，默认取置信度最高的目标
        "plate"    = 只取钢板
        "a4"       = 只取A4纸
        具体名称要和你训练时 data.yaml 里的 names 一致
    """

    model = YOLO(yolo_model_path)

    results = model.predict(
        source=image_path,
        imgsz=imgsz,
        conf=conf,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO 没有识别到任何目标。")

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names

    candidates = []

    for i, box in enumerate(boxes):
        cls_id = classes[i]
        cls_name = names[cls_id]
        score = confs[i]

        if target_class_name is not None and cls_name != target_class_name:
            continue

        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)

        candidates.append({
            "box": box,
            "conf": score,
            "class_id": cls_id,
            "class_name": cls_name,
            "area": area
        })

    if not candidates:
        print("YOLO 识别到的类别：")
        for cls_id in sorted(set(classes)):
            print(f"  {cls_id}: {names[cls_id]}")
        raise RuntimeError(f"没有找到目标类别：{target_class_name}")

    # 钢板这种大目标，通常取面积最大更稳
    best = max(candidates, key=lambda item: item["area"])

    x1, y1, x2, y2 = best["box"]

    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "conf": float(best["conf"]),
        "class_name": best["class_name"]
    }


def run_sam2_by_yolo_box(
        image_path,
        yolo_model_path,
        target_class_name=None,
        output_mask_path="mask.png",
        output_debug_path="debug_yolo_box.jpg",
        sam_model_name="sam2",
        yolo_conf=0.25,
        yolo_imgsz=1280
):
    image_pil = Image.open(image_path).convert("RGB")
    image = np.asarray(image_pil)

    box = get_yolo_box(
        image_path=image_path,
        yolo_model_path=yolo_model_path,
        target_class_name=target_class_name,
        conf=yolo_conf,
        imgsz=yolo_imgsz
    )

    x1 = box["x1"]
    y1 = box["y1"]
    x2 = box["x2"]
    y2 = box["y2"]

    print("YOLO 选中的目标：")
    print(f"  class = {box['class_name']}")
    print(f"  conf  = {box['conf']:.4f}")
    print(f"  box   = {x1}, {y1}, {x2}, {y2}")

    # 保存调试图，方便你看 YOLO 框是否正确
    debug_img = image_pil.copy()
    draw = ImageDraw.Draw(debug_img)
    draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
    draw.text((x1, max(0, y1 - 25)), f"{box['class_name']} {box['conf']:.2f}", fill="red")
    debug_img.save(output_debug_path)

    # 用 YOLO 框作为 SAM2 的提示
    request = osam.types.GenerateRequest(
        model=sam_model_name,
        image=image,
        prompt=osam.types.Prompt(
            points=[
                [x1, y1],
                [x2, y2]
            ],
            point_labels=[
                2,  # 左上角
                3   # 右下角
            ]
        ),
    )

    response = osam.apis.generate(request=request)

    if not response.annotations:
        raise RuntimeError("SAM2 没有生成 mask。请检查 YOLO 框是否框住了目标。")

    annotation = response.annotations[0]
    bbox = annotation.bounding_box

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    small_mask = np.asarray(annotation.mask)

    if small_mask.dtype != np.bool_:
        small_mask = small_mask > 0

    full_mask[
        bbox.ymin:bbox.ymax + 1,
        bbox.xmin:bbox.xmax + 1
    ] = small_mask.astype(np.uint8) * 255

    Image.fromarray(full_mask).save(output_mask_path)

    print(f"SAM2 mask saved: {output_mask_path}")
    print(f"YOLO debug image saved: {output_debug_path}")


if __name__ == "__main__":
    batch_run_sam2_by_yolo_box(
        input_dir=r"D:\Desktop\现场余料图",
        yolo_model_path=r"../best2.pt",

        # 这里填你的类别名
        target_class_name="plate",

        output_root="batch_output",

        # sam2 = LabelMe balanced
        # sam2:large = LabelMe accuracy
        sam_model_name="sam2",

        yolo_conf=0.2,
        yolo_imgsz=1280,

        # True = 递归子目录
        # False = 只处理当前目录
        recursive=False
    )