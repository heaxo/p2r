import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def get_aruco_dictionary(name: str):
    """
    支持：
    DICT_4X4
    DICT_4X4_50
    DICT_5X5_100
    等
    """
    name = name.upper().strip()

    # 兼容你页面里看到的 DICT_4X4
    alias = {
        "DICT_4X4": "DICT_4X4_50",
        "4X4": "DICT_4X4_50",
        "DICT_5X5": "DICT_5X5_100",
        "5X5": "DICT_5X5_100",
        "DICT_6X6": "DICT_6X6_250",
        "6X6": "DICT_6X6_250",
    }

    name = alias.get(name, name)

    if not name.startswith("DICT_"):
        name = "DICT_" + name

    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"OpenCV 不支持这个字典: {name}")

    dictionary_id = getattr(cv2.aruco, name)
    dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

    return name, dictionary


def create_charuco_board(cols, rows, square_px, marker_px, dictionary):
    """
    兼容不同 OpenCV 版本。
    cols = 横向方格数
    rows = 纵向方格数
    """
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            (cols, rows),
            float(square_px),
            float(marker_px),
            dictionary,
        )

    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            cols,
            rows,
            float(square_px),
            float(marker_px),
            dictionary,
        )

    raise RuntimeError("当前 OpenCV 没有 CharucoBoard，请安装 opencv-contrib-python")


def draw_charuco_board(board, board_w_px, board_h_px):
    """
    输出灰度图，白底黑标定板。
    """
    out_size = (int(board_w_px), int(board_h_px))

    if hasattr(board, "generateImage"):
        img = board.generateImage(
            out_size,
            marginSize=0,
            borderBits=1,
        )
        return img

    if hasattr(board, "draw"):
        img = board.draw(
            out_size,
            marginSize=0,
            borderBits=1,
        )
        return img

    raise RuntimeError("当前 OpenCV 版本不支持绘制 CharucoBoard")


def save_png_with_dpi(gray_img, path, dpi):
    img = Image.fromarray(gray_img)
    img.save(path, dpi=(dpi, dpi))


def save_pdf_with_dpi(gray_img, path, dpi):
    img = Image.fromarray(gray_img).convert("RGB")
    img.save(path, "PDF", resolution=dpi)


def generate_charuco_a4(
    out_prefix: str,
    paper_w_mm: float,
    paper_h_mm: float,
    cols: int,
    rows: int,
    square_mm: float,
    marker_mm: float,
    dictionary_name: str,
    dpi: int,
):
    px_per_mm = dpi / 25.4

    paper_w_px = int(round(paper_w_mm * px_per_mm))
    paper_h_px = int(round(paper_h_mm * px_per_mm))

    square_px = int(round(square_mm * px_per_mm))
    marker_px = int(round(marker_mm * px_per_mm))

    if marker_px >= square_px:
        raise RuntimeError("marker_mm 必须小于 square_mm")

    board_w_px = cols * square_px
    board_h_px = rows * square_px

    if board_w_px > paper_w_px or board_h_px > paper_h_px:
        raise RuntimeError(
            f"标定板尺寸超过纸张。\n"
            f"纸张: {paper_w_px} x {paper_h_px}px\n"
            f"标定板: {board_w_px} x {board_h_px}px\n"
            f"请减小 rows/cols 或 square_mm。"
        )

    real_board_w_mm = cols * square_mm
    real_board_h_mm = rows * square_mm

    dict_resolved_name, dictionary = get_aruco_dictionary(dictionary_name)

    board = create_charuco_board(
        cols=cols,
        rows=rows,
        square_px=square_px,
        marker_px=marker_px,
        dictionary=dictionary,
    )

    board_img = draw_charuco_board(board, board_w_px, board_h_px)

    # A4 白底
    canvas = np.ones((paper_h_px, paper_w_px), dtype=np.uint8) * 255

    # 居中放置
    x = (paper_w_px - board_w_px) // 2
    y = (paper_h_px - board_h_px) // 2

    canvas[y:y + board_h_px, x:x + board_w_px] = board_img

    out_prefix = Path(out_prefix)
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    json_path = out_prefix.with_suffix(".json")

    save_png_with_dpi(canvas, png_path, dpi)
    save_pdf_with_dpi(canvas, pdf_path, dpi)

    meta = {
        "type": "ChArUco",
        "paper_width_mm": paper_w_mm,
        "paper_height_mm": paper_h_mm,
        "dpi": dpi,
        "dictionary": dict_resolved_name,
        "cols_squares_x": cols,
        "rows_squares_y": rows,
        "square_length_mm": square_mm,
        "marker_length_mm": marker_mm,
        "board_width_mm": real_board_w_mm,
        "board_height_mm": real_board_h_mm,
        "paper_width_px": paper_w_px,
        "paper_height_px": paper_h_px,
        "square_length_px": square_px,
        "marker_length_px": marker_px,
        "board_offset_x_px": x,
        "board_offset_y_px": y,
        "print_note": "打印时必须选择 100% / 实际大小，不要选择适应页面。",
    }

    json_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("生成完成")
    print(f"PNG:  {png_path}")
    print(f"PDF:  {pdf_path}")
    print(f"JSON: {json_path}")
    print()
    print("参数：")
    print(f"纸张: {paper_w_mm} x {paper_h_mm} mm")
    print(f"标定板: {real_board_w_mm} x {real_board_h_mm} mm")
    print(f"方格: {square_mm} mm")
    print(f"Marker: {marker_mm} mm")
    print(f"字典: {dict_resolved_name}")
    print()
    print("打印要求：")
    print("1. 选择 100% / 实际大小")
    print("2. 不要选择 适应页面 / 缩放到纸张")
    print("3. 打印后用尺子量一个方格边长，应该是 25mm")
    print("4. 建议贴到硬纸板、亚克力板或薄铝板上，保证平整")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out", default="charuco_a4_7x10_25mm", help="输出文件前缀")

    parser.add_argument("--paper-w-mm", type=float, default=210.0, help="纸张宽度，A4 是 210")
    parser.add_argument("--paper-h-mm", type=float, default=297.0, help="纸张高度，A4 是 297")

    parser.add_argument("--cols", type=int, default=7, help="横向方格数量")
    parser.add_argument("--rows", type=int, default=10, help="纵向方格数量")

    parser.add_argument("--square-mm", type=float, default=25.0, help="方格边长，单位 mm")
    parser.add_argument("--marker-mm", type=float, default=18.0, help="ArUco marker 边长，单位 mm")

    parser.add_argument("--dict", default="DICT_4X4", help="字典，比如 DICT_4X4 或 DICT_4X4_50")
    parser.add_argument("--dpi", type=int, default=600, help="输出 DPI，推荐 600")

    return parser.parse_args()


def main():
    args = parse_args()

    generate_charuco_a4(
        out_prefix=args.out,
        paper_w_mm=args.paper_w_mm,
        paper_h_mm=args.paper_h_mm,
        cols=args.cols,
        rows=args.rows,
        square_mm=args.square_mm,
        marker_mm=args.marker_mm,
        dictionary_name=args.dict,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()