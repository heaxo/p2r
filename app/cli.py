from __future__ import annotations

import argparse
import json
import sys
import traceback

from app.core.algorithm import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SAM_MODEL_NAME,
    YOLO_CONF,
    YOLO_IMGSZ,
    process_one_image,
)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO + SAM2 控制台版：识别钢板和A4纸，基于A4计算钢板尺寸并生成DXF")
    parser.add_argument("--image", required=False, help="输入图片路径", default=r"D:\lantek\project-python\p2r\batch_output\val\B2212013-001.jpg")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO模型路径，默认 best2.pt")
    parser.add_argument("--out", default="measure_out", help="输出目录")
    parser.add_argument("--sam-model", default=DEFAULT_SAM_MODEL_NAME, help="osam/SAM2模型名，默认 sam2")
    parser.add_argument("--imgsz", type=int, default=YOLO_IMGSZ)
    parser.add_argument("--conf", type=float, default=YOLO_CONF)
    parser.add_argument("--yolo-input-mode", default="canonical_path", choices=["canonical_path", "rgb_array", "bgr_array"])
    parser.add_argument("--plate-class", default="plate")
    parser.add_argument("--paper-class", default="paper")
    parser.add_argument("--user-point-ratio", default=None, help="可选，手动指定 plate 的 SAM2 点，例如 0.5,0.5")
    parser.add_argument("--paper-source", default="yolo", choices=["yolo", "sam2"] ,help="paper mask 来源：yolo=直接使用YOLO的paper mask/box；sam2=YOLO找paper内部点后交给SAM2。默认yolo")
    parser.add_argument("--paper-sam2-yolo-fallback", action="store_true" , help="当 --paper-source=sam2 且 SAM2 失败时，是否退回 YOLO paper mask/box")
    parser.add_argument("--a4-orientation", default="auto", choices=["auto", "landscape", "portrait"], help="A4方向。横放建议 landscape")
    parser.add_argument("--paper-points", default=None, help="可选：直接传入A4四角坐标，格式 x1,y1;x2,y2;x3,y3;x4,y4")
    parser.add_argument("--paper-rect-mode", default="approx_poly", choices=["robust_fit", "approx_poly", "min_area_rect", "raw"], help="从 paper mask 拟合 A4 四角的方法。默认 approx_poly，优先保留A4透视四边形；min_area_rect只建议兜底。")
    parser.add_argument("--simplify-mm", type=float, default=3.0, help="DXF轮廓简化精度，单位mm。越大点越少，默认3mm")

    parser.add_argument("--topdown-mm-per-px", type=float, default=2.0, help="俯视矫正图比例，默认 1px=2mm")
    parser.add_argument("--topdown-padding-mm", type=float, default=50.0, help="俯视矫正图四周留白，单位mm")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = process_one_image(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e), "traceback": traceback.format_exc(limit=10)}, ensure_ascii=False, indent=2), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
