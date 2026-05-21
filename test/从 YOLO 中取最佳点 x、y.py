import cv2
import numpy as np
import osam
from PIL import Image, ImageDraw
from ultralytics import YOLO



def run_sam2_by_point(
        image_path,
        x,
        y,
        output_path="mask.png",
        model_name="sam2"
):
    """
    model_name:
        sam2        = balanced
        sam2:large  = accuracy
    """

    image_pil = Image.open(image_path).convert("RGB")
    image = np.asarray(image_pil)

    request = osam.types.GenerateRequest(
        model=model_name,
        image=image,
        prompt=osam.types.Prompt(
            points=[[x, y]],
            point_labels=[1]  # 1 = 前景点
        ),
    )

    response = osam.apis.generate(request=request)

    if not response.annotations:
        raise RuntimeError("SAM2 没有生成任何 mask，请换一个点或者改用框选方式。")

    # 一般取第一个结果
    annotation = response.annotations[0]

    # annotation.mask 通常只是 bbox 区域内的小 mask
    bbox = annotation.bounding_box

    full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

    small_mask = np.asarray(annotation.mask)

    # 转成 0/255 图像
    if small_mask.dtype != np.bool_:
        small_mask = small_mask > 0

    full_mask[
        bbox.ymin:bbox.ymax + 1,
        bbox.xmin:bbox.xmax + 1
    ] = small_mask.astype(np.uint8) * 255

    Image.fromarray(full_mask).save(output_path)

    print(f"mask saved: {output_path}")


def _get_mask_from_yolo_result(result, index, h, w):
    """
    从 YOLO result.masks.data 里取单个实例 mask，并缩放到原图尺寸。
    """
    if result.masks is None:
        return None

    mask = result.masks.data[index].cpu().numpy()

    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    mask_bin = (mask > 0.5).astype(np.uint8)
    return mask_bin


def get_best_point_from_yolo(
        image_path,
        yolo_model_path,
        target_class_name="plate",
        avoid_class_names=None,
        conf=0.25,
        imgsz=1280,
        output_debug_path=None
):
    """
    从 YOLO 分割结果中，取一个最适合给 SAM2 的前景点。

    target_class_name:
        要取点的类别，比如 plate / paper / hole

    avoid_class_names:
        要避开的类别，比如取 plate 点时，可以避开 paper、hole
    """

    if avoid_class_names is None:
        avoid_class_names = []

    model = YOLO(yolo_model_path)

    results = model.predict(
        source=image_path,
        conf=conf,
        imgsz=imgsz,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("YOLO 没有识别到任何目标")

    h, w = result.orig_shape

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names

    # 反查类别 id
    target_class_ids = [
        cls_id for cls_id, cls_name in names.items()
        if cls_name == target_class_name
    ]

    if not target_class_ids:
        raise RuntimeError(f"YOLO 模型中没有类别：{target_class_name}，当前类别：{names}")

    target_class_id = target_class_ids[0]

    avoid_class_ids = {
        cls_id for cls_id, cls_name in names.items()
        if cls_name in avoid_class_names
    }

    # 如果 YOLO 是检测模型，不是分割模型，就没有 masks
    if result.masks is None:
        print("警告：当前 YOLO 没有 mask，可能不是 seg 模型，只能退回使用 box 中心点。")

    # 先合并需要避开的区域，比如 paper、hole
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

            # 避开 paper / hole，防止点落到 A4纸或孔里面
            if avoid_mask is not None:
                target_mask = target_mask & (~avoid_mask.astype(bool))

            target_mask = target_mask.astype(np.uint8)

            area = int(target_mask.sum())

            if area <= 0:
                continue

            # 距离变换：找 mask 内部离边界最远的点
            dist = cv2.distanceTransform(target_mask * 255, cv2.DIST_L2, 5)
            _, max_dist, _, max_loc = cv2.minMaxLoc(dist)

            best_x, best_y = max_loc

            candidates.append({
                "x": int(best_x),
                "y": int(best_y),
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "conf": score,
                "area": area,
                "max_dist": float(max_dist),
                "class_name": target_class_name
            })

        else:
            # 没有 mask，只能用 box 中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            area = max(0, int((x2 - x1) * (y2 - y1)))

            candidates.append({
                "x": cx,
                "y": cy,
                "box": [int(x1), int(y1), int(x2), int(y2)],
                "conf": score,
                "area": area,
                "max_dist": 0,
                "class_name": target_class_name
            })

    if not candidates:
        raise RuntimeError(f"没有找到可用的目标点：{target_class_name}")

    # 钢板这种大目标，一般取面积最大的实例
    best = max(candidates, key=lambda item: item["area"])

    x = best["x"]
    y = best["y"]

    print("YOLO 最佳点：")
    print(f"  class = {best['class_name']}")
    print(f"  x, y  = {x}, {y}")
    print(f"  conf  = {best['conf']:.4f}")
    print(f"  area  = {best['area']}")
    print(f"  box   = {best['box']}")

    if output_debug_path:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        x1, y1, x2, y2 = best["box"]

        # 画 YOLO 框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=5)

        # 画最佳点
        r = 12
        draw.ellipse([x - r, y - r, x + r, y + r], fill="blue", outline="white", width=3)

        draw.text((x + 15, y - 15), f"{target_class_name} point", fill="blue")

        img.save(output_debug_path)
        print(f"debug saved: {output_debug_path}")

    return x, y, best

if __name__ == "__main__":
    x, y, info = get_best_point_from_yolo(
        image_path=r"/batch_output/val/B2212013-001.jpg",
        yolo_model_path=r"../best2.pt",

        target_class_name="plate",

        # 取钢板点时，最好避开 A4纸和孔
        avoid_class_names=["paper", "hole"],

        conf=0.2,
        imgsz=1280,
        output_debug_path="debug_best_point_plate2.jpg"
    )

    run_sam2_by_point(
        image_path=r"/batch_output/val/B2212013-001.jpg",
        x=x,
        y=y,
        output_path="plate_mask.png",
        model_name="sam2"
    )