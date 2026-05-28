import cv2
import numpy as np

IMG_PATH = "input.jpg"
OUT_PATH = "corrected.jpg"

CANVAS_W = 1200
CANVAS_H = 800

img = cv2.imread(IMG_PATH)
if img is None:
    raise RuntimeError(f"图片读取失败: {IMG_PATH}")

H, W = img.shape[:2]

win = "select 4 points"

points = []

# 初始缩放：完整显示图片
zoom = min(CANVAS_W / W, CANVAS_H / H)

# 当前视口左上角在原图中的坐标
view_x = (W - CANVAS_W / zoom) / 2
view_y = (H - CANVAS_H / zoom) / 2

dragging = False
last_x = 0
last_y = 0


def clamp_view():
    global view_x, view_y

    visible_w = CANVAS_W / zoom
    visible_h = CANVAS_H / zoom

    if visible_w >= W:
        view_x = (W - visible_w) / 2
    else:
        view_x = max(0, min(view_x, W - visible_w))

    if visible_h >= H:
        view_y = (H - visible_h) / 2
    else:
        view_y = max(0, min(view_y, H - visible_h))


def screen_to_image(x, y):
    ix = view_x + x / zoom
    iy = view_y + y / zoom
    return ix, iy


def image_to_screen(ix, iy):
    sx = int((ix - view_x) * zoom)
    sy = int((iy - view_y) * zoom)
    return sx, sy


def redraw():
    clamp_view()

    # 原图到显示画布的仿射变换
    M = np.array([
        [zoom, 0, -view_x * zoom],
        [0, zoom, -view_y * zoom],
    ], dtype=np.float32)

    canvas = cv2.warpAffine(
        img,
        M,
        (CANVAS_W, CANVAS_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(40, 40, 40),
    )

    for i, (px, py) in enumerate(points):
        sx, sy = image_to_screen(px, py)
        if 0 <= sx < CANVAS_W and 0 <= sy < CANVAS_H:
            cv2.circle(canvas, (sx, sy), 6, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                str(i + 1),
                (sx + 8, sy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

    cv2.imshow(win, canvas)


def mouse_callback(event, x, y, flags, param):
    global zoom, view_x, view_y
    global dragging, last_x, last_y

    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = screen_to_image(x, y)

        if 0 <= ix < W and 0 <= iy < H:
            points.append([ix, iy])
            print(f"第 {len(points)} 个点，原图坐标: {ix:.2f}, {iy:.2f}")
            redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:
        dragging = True
        last_x = x
        last_y = y

    elif event == cv2.EVENT_RBUTTONUP:
        dragging = False

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        dx = x - last_x
        dy = y - last_y

        # 拖动画面
        view_x -= dx / zoom
        view_y -= dy / zoom

        last_x = x
        last_y = y

        redraw()

    elif event == cv2.EVENT_MOUSEWHEEL:
        # 缩放前，鼠标指向的原图坐标
        anchor_x, anchor_y = screen_to_image(x, y)

        if flags > 0:
            new_zoom = zoom * 1.25
        else:
            new_zoom = zoom / 1.25

        new_zoom = max(0.05, min(new_zoom, 20.0))

        zoom = new_zoom

        # 保证滚轮缩放后，鼠标指向的原图位置不变
        view_x = anchor_x - x / zoom
        view_y = anchor_y - y / zoom

        redraw()


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)

    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # 左上
    rect[2] = pts[np.argmax(s)]      # 右下
    rect[1] = pts[np.argmin(diff)]   # 右上
    rect[3] = pts[np.argmax(diff)]   # 左下

    return rect


cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback(win, mouse_callback)

redraw()

print("操作说明:")
print("左键: 选点")
print("右键拖动: 平移图片")
print("滚轮: 放大缩小")
print("U: 撤销上一个点")
print("R: 重置视图")
print("Enter: 确认并矫正")
print("Esc: 退出")

while True:
    key = cv2.waitKey(20) & 0xFF

    if key == 27:
        cv2.destroyAllWindows()
        raise SystemExit("已退出")

    elif key in (13, 10):
        if len(points) != 4:
            print("必须选择 4 个点后再按 Enter")
        else:
            break

    elif key == ord("u"):
        if points:
            removed = points.pop()
            print(f"撤销点: {removed}")
            redraw()

    elif key == ord("r"):
        zoom = min(CANVAS_W / W, CANVAS_H / H)
        view_x = (W - CANVAS_W / zoom) / 2
        view_y = (H - CANVAS_H / zoom) / 2
        redraw()

# src 是你点的 A4 四个角，顺序：左上、右上、右下、左下
src = order_points(points)

# A4 实际比例：210 x 297
# 用 scale 控制输出清晰度，数值越大，输出图越大
scale = 4

a4_w = 210 * scale
a4_h = 297 * scale

# 判断横版还是竖版
w1 = np.linalg.norm(src[1] - src[0])
w2 = np.linalg.norm(src[2] - src[3])
h1 = np.linalg.norm(src[3] - src[0])
h2 = np.linalg.norm(src[2] - src[1])

src_w = max(w1, w2)
src_h = max(h1, h2)

if src_w >= src_h:
    # 横版 A4
    dst_a4 = np.float32([
        [0, 0],
        [a4_h - 1, 0],
        [a4_h - 1, a4_w - 1],
        [0, a4_w - 1],
    ])
else:
    # 竖版 A4
    dst_a4 = np.float32([
        [0, 0],
        [a4_w - 1, 0],
        [a4_w - 1, a4_h - 1],
        [0, a4_h - 1],
    ])

# 这个矩阵是：把原图中的 A4 拉正
M = cv2.getPerspectiveTransform(src, dst_a4)

# 关键：不要只 warp A4，而是计算整张原图变换后的范围
img_corners = np.float32([
    [0, 0],
    [W - 1, 0],
    [W - 1, H - 1],
    [0, H - 1],
]).reshape(-1, 1, 2)

warped_corners = cv2.perspectiveTransform(img_corners, M)

xs = warped_corners[:, 0, 0]
ys = warped_corners[:, 0, 1]

min_x, max_x = xs.min(), xs.max()
min_y, max_y = ys.min(), ys.max()

# 加一点边距
margin = 30

out_w = int(np.ceil(max_x - min_x)) + margin * 2
out_h = int(np.ceil(max_y - min_y)) + margin * 2

# 因为变换后可能有负坐标，所以加一个平移矩阵
T = np.array([
    [1, 0, -min_x + margin],
    [0, 1, -min_y + margin],
    [0, 0, 1],
], dtype=np.float32)

# 最终矩阵 = 平移矩阵 * A4透视矫正矩阵
M_full = T @ M

# 对整张图片做透视矫正
warped_full = cv2.warpPerspective(
    img,
    M_full,
    (out_w, out_h),
    flags=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=(40, 40, 40),
)

cv2.imwrite("corrected_full.jpg", warped_full)

print("整张图片矫正完成: corrected_full.jpg")