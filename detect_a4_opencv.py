import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


A4_RATIO = 297.0 / 210.0


def order_points(pts):
    """
    四点排序：左上、右上、右下、左下
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # 左上
    rect[2] = pts[np.argmax(s)]      # 右下
    rect[1] = pts[np.argmin(diff)]   # 右上
    rect[3] = pts[np.argmax(diff)]   # 左下

    return rect


def angle_between(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    cos_value = float(np.dot(v1, v2) / (n1 * n2))
    cos_value = np.clip(cos_value, -1.0, 1.0)

    return math.degrees(math.acos(cos_value))


def calc_quad_metrics(quad):
    quad = order_points(quad)

    tl, tr, br, bl = quad

    top = np.linalg.norm(tr - tl)
    right = np.linalg.norm(br - tr)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)

    avg_w = (top + bottom) / 2.0
    avg_h = (left + right) / 2.0

    if avg_w < 1e-6 or avg_h < 1e-6:
        return None

    ratio = max(avg_w, avg_h) / min(avg_w, avg_h)
    ratio_error = abs(ratio - A4_RATIO)

    width_diff_ratio = abs(top - bottom) / max(avg_w, 1e-6)
    height_diff_ratio = abs(left - right) / max(avg_h, 1e-6)

    angle_tl = angle_between(tr - tl, bl - tl)
    angle_tr = angle_between(tl - tr, br - tr)
    angle_br = angle_between(tr - br, bl - br)
    angle_bl = angle_between(tl - bl, br - bl)

    angles = [angle_tl, angle_tr, angle_br, angle_bl]
    max_angle_error = max(abs(a - 90.0) for a in angles)

    return {
        "top": float(top),
        "right": float(right),
        "bottom": float(bottom),
        "left": float(left),
        "ratio": float(ratio),
        "ratio_error": float(ratio_error),
        "width_diff_ratio": float(width_diff_ratio),
        "height_diff_ratio": float(height_diff_ratio),
        "angles": [float(a) for a in angles],
        "max_angle_error": float(max_angle_error),
    }


def contour_to_quad(cnt):
    """
    优先用 approxPolyDP 找四边形。
    如果轮廓不够规整，就退化成 minAreaRect。
    """
    peri = cv2.arcLength(cnt, True)

    for eps in [0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06]:
        approx = cv2.approxPolyDP(cnt, eps * peri, True)

        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx.reshape(4, 2))

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)

    return order_points(box)


def build_white_masks(image):
    """
    生成多组白色/低饱和度候选 mask。
    A4 一般特点：亮度高、饱和度低。
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, s, v = cv2.split(hsv)

    masks = []

    # 1. 多组固定 HSV 阈值
    v_min_list = [235, 225, 215, 205, 195, 180, 165, 150]
    s_max_list = [35, 50, 70, 90, 120, 150]

    for v_min in v_min_list:
        for s_max in s_max_list:
            mask = cv2.inRange(
                hsv,
                np.array([0, 0, v_min], dtype=np.uint8),
                np.array([180, s_max, 255], dtype=np.uint8),
            )
            masks.append(("hsv", v_min, s_max, mask))

    # 2. Otsu 亮度阈值 + 低饱和度
    _, mask_otsu_v = cv2.threshold(
        v,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    for s_max in [50, 70, 90, 120, 150]:
        mask_s = cv2.inRange(s, 0, s_max)
        mask = cv2.bitwise_and(mask_otsu_v, mask_s)
        masks.append(("otsu_v", 0, s_max, mask))

    # 3. 灰度 Otsu + 低饱和度
    _, mask_otsu_gray = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    for s_max in [50, 70, 90, 120, 150]:
        mask_s = cv2.inRange(s, 0, s_max)
        mask = cv2.bitwise_and(mask_otsu_gray, mask_s)
        masks.append(("otsu_gray", 0, s_max, mask))

    return masks


def clean_mask(mask):
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

    return mask


def detect_a4_by_opencv(image, debug_dir):
    """
    返回：
    {
        success,
        quad,
        score,
        metrics,
        mask,
        candidates
    }
    """
    h, w = image.shape[:2]
    image_area = h * w

    masks = build_white_masks(image)

    candidates = []

    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    for mask_type, v_min, s_max, raw_mask in masks:
        mask = clean_mask(raw_mask)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # 面积过滤：太小/太大都不像 A4
            if area < image_area * 0.001:
                continue

            if area > image_area * 0.40:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)

            if bw < 30 or bh < 30:
                continue

            quad = contour_to_quad(cnt)
            metrics = calc_quad_metrics(quad)

            if metrics is None:
                continue

            ratio_error = metrics["ratio_error"]

            # 透视下比例会变，所以不能卡太死
            if ratio_error > 0.60:
                continue

            rect_area = cv2.contourArea(quad.astype(np.float32))
            if rect_area <= 1:
                continue

            fill_ratio = area / rect_area

            # A4 是实心区域，填充率太低一般不是纸
            if fill_ratio < 0.35:
                continue

            # 对边长度差太离谱，通常不是 A4，或者检测很不准
            if metrics["width_diff_ratio"] > 0.65:
                continue

            if metrics["height_diff_ratio"] > 0.65:
                continue

            # 角度太离谱，通常不是纸
            if metrics["max_angle_error"] > 45:
                continue

            # 综合评分
            # 面积越大、比例越接近 A4、填充越实、角度越接近矩形，得分越高
            score = (
                area
                * fill_ratio
                / (1.0 + ratio_error * 5.0)
                / (1.0 + metrics["max_angle_error"] / 30.0)
            )

            candidates.append({
                "score": float(score),
                "quad": quad,
                "area": float(area),
                "fill_ratio": float(fill_ratio),
                "metrics": metrics,
                "mask": mask,
                "mask_type": mask_type,
                "v_min": v_min,
                "s_max": s_max,
                "bbox": (x, y, bw, bh),
            })

    if not candidates:
        return {
            "success": False,
            "message": "没有检测到可靠 A4",
            "candidates": [],
        }

    candidates.sort(key=lambda item: item["score"], reverse=True)

    best = candidates[0]

    # 输出 best mask
    cv2.imwrite(str(debug_dir / "debug_best_mask.png"), best["mask"])

    # 输出候选框调试图
    debug_candidates = image.copy()

    for i, item in enumerate(candidates[:10]):
        quad = np.int32(order_points(item["quad"]))

        color = (0, 0, 255) if i == 0 else (255, 0, 0)

        cv2.polylines(debug_candidates, [quad], True, color, 3)

        p = quad[0]
        cv2.putText(
            debug_candidates,
            f"{i + 1}",
            (int(p[0]), int(p[1]) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )

    cv2.imwrite(str(debug_dir / "debug_a4_candidates.jpg"), debug_candidates)

    # 输出最佳结果图
    debug_best = image.copy()
    q = np.int32(order_points(best["quad"]))
    cv2.polylines(debug_best, [q], True, (0, 0, 255), 4)

    for i, p in enumerate(q):
        cv2.circle(debug_best, tuple(p), 8, (0, 0, 255), -1)
        cv2.putText(
            debug_best,
            str(i + 1),
            (int(p[0]) + 10, int(p[1]) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )

    cv2.imwrite(str(debug_dir / "debug_a4_best.jpg"), debug_best)

    return {
        "success": True,
        "message": "检测到 A4 候选",
        "quad": best["quad"],
        "score": best["score"],
        "metrics": best["metrics"],
        "mask_type": best["mask_type"],
        "v_min": best["v_min"],
        "s_max": best["s_max"],
        "fill_ratio": best["fill_ratio"],
        "area": best["area"],
        "candidates": candidates,
    }


def select_roi_scaled(image, max_w=1200, max_h=850):
    """
    可选：手动框选 A4 附近区域。
    如果全图误检严重，可以加 --select-roi。
    """
    h, w = image.shape[:2]

    scale = min(max_w / w, max_h / h, 1.0)

    show_w = int(w * scale)
    show_h = int(h * scale)

    show = cv2.resize(image, (show_w, show_h), interpolation=cv2.INTER_AREA)

    roi = cv2.selectROI(
        "Select A4 ROI, press Enter",
        show,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyWindow("Select A4 ROI, press Enter")

    x, y, rw, rh = roi

    if rw <= 0 or rh <= 0:
        raise RuntimeError("没有选择 ROI")

    x = int(x / scale)
    y = int(y / scale)
    rw = int(rw / scale)
    rh = int(rh / scale)

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))

    return x, y, rw, rh


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_image_paths(input_path: Path, recursive: bool = False):
    """
    支持单张图片或目录。
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTS:
            yield input_path
        return

    if not input_path.is_dir():
        raise RuntimeError(f"路径不存在: {input_path}")

    pattern = "**/*" if recursive else "*"

    for p in input_path.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def make_image_debug_dir(root_debug_dir: Path, image_path: Path, input_root: Path | None):
    """
    给每张图片单独建调试目录。
    如果是目录模式，尽量保留相对路径，避免重名覆盖。
    """
    if input_root is not None and input_root.is_dir():
        try:
            rel = image_path.relative_to(input_root)
            parts = list(rel.with_suffix("").parts)
            safe_name = "__".join(parts)
        except ValueError:
            safe_name = image_path.stem
    else:
        safe_name = image_path.stem

    out_dir = root_debug_dir / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def process_one_image(image_path: Path, debug_dir: Path, select_roi: bool):
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "image": str(image_path),
            "success": False,
            "message": "读取图片失败",
        }

    debug_dir.mkdir(parents=True, exist_ok=True)

    if select_roi:
        x, y, w, h = select_roi_scaled(image)
        crop = image[y:y + h, x:x + w].copy()

        cv2.imwrite(str(debug_dir / "debug_roi.jpg"), crop)

        result = detect_a4_by_opencv(crop, debug_dir)

        if result["success"]:
            result["quad"] = result["quad"] + np.array([x, y], dtype=np.float32)

            debug_full = image.copy()
            q = np.int32(order_points(result["quad"]))
            cv2.polylines(debug_full, [q], True, (0, 0, 255), 4)

            for i, p in enumerate(q):
                cv2.circle(debug_full, tuple(p), 8, (0, 0, 255), -1)
                cv2.putText(
                    debug_full,
                    str(i + 1),
                    (int(p[0]) + 10, int(p[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.imwrite(str(debug_dir / "debug_a4_best_full.jpg"), debug_full)
    else:
        result = detect_a4_by_opencv(image, debug_dir)

    summary = {
        "image": str(image_path),
        "success": bool(result["success"]),
        "message": result["message"],
        "debug_dir": str(debug_dir),
    }

    if result["success"]:
        quad = order_points(result["quad"])

        summary.update({
            "score": result["score"],
            "area": result["area"],
            "fill_ratio": result["fill_ratio"],
            "mask_type": result["mask_type"],
            "v_min": result["v_min"],
            "s_max": result["s_max"],
            "p1_x": float(quad[0][0]),
            "p1_y": float(quad[0][1]),
            "p2_x": float(quad[1][0]),
            "p2_y": float(quad[1][1]),
            "p3_x": float(quad[2][0]),
            "p3_y": float(quad[2][1]),
            "p4_x": float(quad[3][0]),
            "p4_y": float(quad[3][1]),
            "ratio": result["metrics"]["ratio"],
            "ratio_error": result["metrics"]["ratio_error"],
            "max_angle_error": result["metrics"]["max_angle_error"],
            "width_diff_ratio": result["metrics"]["width_diff_ratio"],
            "height_diff_ratio": result["metrics"]["height_diff_ratio"],
        })

    return summary


def write_summary_csv(csv_path: Path, rows: list[dict]):
    if not rows:
        return

    fieldnames = [
        "image",
        "success",
        "message",
        "debug_dir",
        "score",
        "area",
        "fill_ratio",
        "mask_type",
        "v_min",
        "s_max",
        "p1_x",
        "p1_y",
        "p2_x",
        "p2_y",
        "p3_x",
        "p3_y",
        "p4_x",
        "p4_y",
        "ratio",
        "ratio_error",
        "max_angle_error",
        "width_diff_ratio",
        "height_diff_ratio",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def iter_image_paths(input_path: Path, recursive: bool = False):
    """
    支持单张图片或目录。
    """
    input_path = Path(input_path)

    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTS:
            yield input_path
        return

    if not input_path.is_dir():
        raise RuntimeError(f"路径不存在: {input_path}")

    pattern = "**/*" if recursive else "*"

    for p in input_path.glob(pattern):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p


def make_image_debug_dir(root_debug_dir: Path, image_path: Path, input_root: Path | None):
    """
    给每张图片单独建调试目录。
    如果是目录模式，尽量保留相对路径，避免重名覆盖。
    """
    if input_root is not None and input_root.is_dir():
        try:
            rel = image_path.relative_to(input_root)
            parts = list(rel.with_suffix("").parts)
            safe_name = "__".join(parts)
        except ValueError:
            safe_name = image_path.stem
    else:
        safe_name = image_path.stem

    out_dir = root_debug_dir / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return out_dir


def process_one_image(image_path: Path, debug_dir: Path, select_roi: bool):
    image = cv2.imread(str(image_path))

    if image is None:
        return {
            "image": str(image_path),
            "success": False,
            "message": "读取图片失败",
        }

    debug_dir.mkdir(parents=True, exist_ok=True)

    if select_roi:
        x, y, w, h = select_roi_scaled(image)
        crop = image[y:y + h, x:x + w].copy()

        cv2.imwrite(str(debug_dir / "debug_roi.jpg"), crop)

        result = detect_a4_by_opencv(crop, debug_dir)

        if result["success"]:
            result["quad"] = result["quad"] + np.array([x, y], dtype=np.float32)

            debug_full = image.copy()
            q = np.int32(order_points(result["quad"]))
            cv2.polylines(debug_full, [q], True, (0, 0, 255), 4)

            for i, p in enumerate(q):
                cv2.circle(debug_full, tuple(p), 8, (0, 0, 255), -1)
                cv2.putText(
                    debug_full,
                    str(i + 1),
                    (int(p[0]) + 10, int(p[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.imwrite(str(debug_dir / "debug_a4_best_full.jpg"), debug_full)
    else:
        result = detect_a4_by_opencv(image, debug_dir)

    summary = {
        "image": str(image_path),
        "success": bool(result["success"]),
        "message": result["message"],
        "debug_dir": str(debug_dir),
    }

    if result["success"]:
        quad = order_points(result["quad"])

        summary.update({
            "score": result["score"],
            "area": result["area"],
            "fill_ratio": result["fill_ratio"],
            "mask_type": result["mask_type"],
            "v_min": result["v_min"],
            "s_max": result["s_max"],
            "p1_x": float(quad[0][0]),
            "p1_y": float(quad[0][1]),
            "p2_x": float(quad[1][0]),
            "p2_y": float(quad[1][1]),
            "p3_x": float(quad[2][0]),
            "p3_y": float(quad[2][1]),
            "p4_x": float(quad[3][0]),
            "p4_y": float(quad[3][1]),
            "ratio": result["metrics"]["ratio"],
            "ratio_error": result["metrics"]["ratio_error"],
            "max_angle_error": result["metrics"]["max_angle_error"],
            "width_diff_ratio": result["metrics"]["width_diff_ratio"],
            "height_diff_ratio": result["metrics"]["height_diff_ratio"],
        })

    return summary


def write_summary_csv(csv_path: Path, rows: list[dict]):
    if not rows:
        return

    fieldnames = [
        "image",
        "success",
        "message",
        "debug_dir",
        "score",
        "area",
        "fill_ratio",
        "mask_type",
        "v_min",
        "s_max",
        "p1_x",
        "p1_y",
        "p2_x",
        "p2_y",
        "p3_x",
        "p3_y",
        "p4_x",
        "p4_y",
        "ratio",
        "ratio_error",
        "max_angle_error",
        "width_diff_ratio",
        "height_diff_ratio",
    ]

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入图片路径或图片目录")
    parser.add_argument("--debug-dir", default="a4_debug", help="调试图输出目录")
    parser.add_argument("--select-roi", action="store_true", help="手动框选 A4 附近区域后再检测")
    parser.add_argument("--recursive", action="store_true", help="目录模式下递归处理子目录")
    parser.add_argument("--summary", default="summary.csv", help="汇总 CSV 文件名")
    args = parser.parse_args()

    input_path = Path(args.image)
    root_debug_dir = Path(args.debug_dir)
    root_debug_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(iter_image_paths(input_path, recursive=args.recursive))

    if not image_paths:
        raise RuntimeError(f"没有找到图片: {input_path}")

    print(f"共找到 {len(image_paths)} 张图片")

    rows = []

    input_root = input_path if input_path.is_dir() else None

    for index, image_path in enumerate(image_paths, start=1):
        print()
        print(f"[{index}/{len(image_paths)}] 处理: {image_path}")

        image_debug_dir = make_image_debug_dir(
            root_debug_dir=root_debug_dir,
            image_path=image_path,
            input_root=input_root,
        )

        try:
            row = process_one_image(
                image_path=image_path,
                debug_dir=image_debug_dir,
                select_roi=args.select_roi,
            )
        except Exception as e:
            row = {
                "image": str(image_path),
                "success": False,
                "message": f"异常: {e}",
                "debug_dir": str(image_debug_dir),
            }

        rows.append(row)

        if row["success"]:
            print("检测成功")
            print(f"调试目录: {row['debug_dir']}")
            print(f"score: {row.get('score')}")
            print(f"ratio_error: {row.get('ratio_error')}")
            print(f"max_angle_error: {row.get('max_angle_error')}")
        else:
            print("检测失败")
            print(row["message"])

    summary_path = root_debug_dir / args.summary
    write_summary_csv(summary_path, rows)

    success_count = sum(1 for r in rows if r.get("success"))
    fail_count = len(rows) - success_count

    print()
    print("全部完成")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"汇总文件: {summary_path}")


if __name__ == "__main__":
    main()