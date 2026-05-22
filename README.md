# Plate Measure HTTP Service

基于 YOLO + SAM2 的钢板余料识别与尺寸测量服务。项目把原控制台识别流程封装为 FastAPI HTTP 服务，并内置一个简单的 Web 上传页面。

服务会识别图片中的钢板和 A4 纸，通过 A4 纸四角建立像素到毫米坐标的 Homography，计算钢板尺寸，并输出 DXF、结果 JSON、mask、俯视矫正图和调试图。

## 主要功能

- 上传单张钢板图片并返回测量结果。
- 支持 YOLO 检测钢板和 A4 纸，SAM2 生成钢板分割 mask。
- 基于 A4 纸尺寸进行毫米换算，输出长宽、面积、周长等数据。
- 生成只包含钢板外轮廓的 `plate_outer.dxf`。
- 对 DXF 轮廓做后处理，包括去毛刺、压直线、局部凹陷修复。
- 支持圆形钢板检测，圆形件会在 DXF 中写入 `CIRCLE`。
- 内置 Web 页面：上传图片、填写参数、预览结果、下载 DXF 和 JSON。
- YOLO 模型进程内缓存，避免每次请求重复加载权重。
- 默认串行处理推理任务，降低 GPU/SAM2 并发抢占导致的不稳定。

## 目录结构

```text
p2r/
  app/
    main.py                    # FastAPI 应用入口，挂载 API、/files 和 /ui
    api/routes.py              # /health、/measure 接口
    config.py                  # 环境变量配置
    security.py                # X-Auth-Token / Bearer Token 鉴权
    schemas.py                 # HTTP 响应模型
    cli.py                     # 控制台调用入口
    core/
      algorithm.py             # YOLO + SAM2 + A4 标定 + DXF 核心流程
      dxf_postprocess.py       # DXF 轮廓后处理
    services/
      measure_service.py       # HTTP 参数适配、上传保存、结果压缩
      model_cache.py           # YOLO 模型缓存
  frontend/
    index.html                 # 内置上传页面
    debug.html                 # 调试页面
  scripts/
    install_windows.bat        # Windows 创建虚拟环境并安装依赖
    start_windows.bat          # Windows 启动服务
    start_windows.ps1          # PowerShell 启动脚本
  test/                        # 历史实验脚本、批处理脚本和样例输出
  measure_out/                 # 运行输出目录，服务自动创建
  uploads/                     # 旧版或调试上传目录
  best2.pt                     # 默认 YOLO 权重
  run_server.py                # Python 启动入口
  sample_request.py            # requests 调用示例
  requirements.txt             # Python 依赖
  WINDOWS_DEPLOY.md            # Windows 部署说明
```

## 环境要求

- Python 3.10 或 3.11。
- 可正常运行的 YOLO / Ultralytics 环境。
- 可正常运行的 `osam` / SAM2 环境。
- YOLO 权重文件，例如项目根目录下的 `best2.pt`。
- 如果使用 GPU，需要提前安装匹配的 CUDA、PyTorch 和显卡驱动。

项目依赖在 `requirements.txt` 中，但 PyTorch、CUDA、SAM2 在不同机器上可能需要按实际环境调整。

## 安装

Windows 推荐直接运行：

```bat
scripts\install_windows.bat
```

手动安装：

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Linux / macOS 可按同样思路创建虚拟环境并安装依赖。

## 配置

可以复制 `.env.example` 中的配置，也可以直接设置环境变量：

```text
APP_TOKEN=tk_c2VjcmV0LXJhbmRvbS10b2tlbi0xMjM0NTY3OA
YOLO_MODEL_PATH=best2.pt
SAM_MODEL=sam2
OUTPUT_ROOT=measure_out
YOLO_IMGSZ=1280
YOLO_CONF=0.35
MAX_UPLOAD_MB=80
SERIALIZE_PROCESSING=true
LOG_DIR=logs
LOG_LEVEL=INFO
HOST=0.0.0.0
PORT=8000
```

常用配置说明：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_TOKEN` | 内置默认 token | `/measure` 接口鉴权 token，生产环境应修改 |
| `YOLO_MODEL_PATH` | `best2.pt` | 默认 YOLO 权重路径 |
| `SAM_MODEL` | `sam2` | 传给 osam/SAM2 的模型名 |
| `OUTPUT_ROOT` | `measure_out` | 输出目录，同时通过 `/files` 对外访问 |
| `YOLO_IMGSZ` | `1280` | 默认 YOLO 推理尺寸 |
| `YOLO_CONF` | `0.35` | 默认 YOLO 置信度 |
| `MAX_UPLOAD_MB` | `80` | 单张上传图片大小限制 |
| `SERIALIZE_PROCESSING` | `true` | 是否串行处理推理请求 |
| `LOG_DIR` | `logs` | 日志输出目录 |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `HOST` | `0.0.0.0` | `run_server.py` 监听地址 |
| `PORT` | `8000` | `run_server.py` 监听端口 |

日志策略：

- 单个日志文件最大 `10 MB`。
- 日志保留 `30 days`。
- 轮转后的旧日志自动压缩为 `zip`。
- 默认文件名格式为 `logs/app_YYYY-MM-DD.log`。

## 启动服务

Windows：

```bat
scripts\start_windows.bat
```

或直接运行：

```bash
python run_server.py
```

默认访问地址：

```text
http://127.0.0.1:8000
```

服务启动后可访问：

```text
http://127.0.0.1:8000/ui/       # 内置上传页面
http://127.0.0.1:8000/docs      # FastAPI 接口文档
http://127.0.0.1:8000/health    # 健康检查
```

## HTTP 接口

### 健康检查

```text
GET /health
```

返回示例：

```json
{
  "ok": "true",
  "service": "plate-measure-http"
}
```

### 测量图片

```text
POST /measure
Content-Type: multipart/form-data
Header: X-Auth-Token: <APP_TOKEN>
```

也支持：

```text
Authorization: Bearer <APP_TOKEN>
```

表单参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `image` | 必填 | 上传图片，支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp` |
| `model_path` | `YOLO_MODEL_PATH` | 可选，覆盖默认 YOLO 权重路径 |
| `sam_model` | `SAM_MODEL` | 可选，覆盖默认 SAM2 模型名 |
| `imgsz` | `YOLO_IMGSZ` | 可选，YOLO 推理尺寸 |
| `conf` | `YOLO_CONF` | 可选，YOLO 置信度 |
| `yolo_input_mode` | `canonical_path` | `canonical_path`、`rgb_array`、`bgr_array` |
| `plate_class` | `plate` | YOLO 中钢板类别名 |
| `paper_class` | `paper` | YOLO 中 A4 纸类别名，支持逗号分隔 |
| `user_point_ratio` | 空 | 手动指定钢板 SAM2 点，例如 `0.5,0.5` |
| `paper_source` | `yolo` | A4 mask 来源：`yolo` 或 `sam2` |
| `paper_sam2_yolo_fallback` | `false` | `paper_source=sam2` 失败时是否退回 YOLO |
| `a4_orientation` | `auto` | `auto`、`landscape`、`portrait` |
| `paper_points` | 空 | 手动传 A4 四角：`x1,y1;x2,y2;x3,y3;x4,y4` |
| `paper_rect_mode` | `approx_poly` | `robust_fit`、`approx_poly`、`min_area_rect`、`raw` |
| `simplify_mm` | `3.0` | DXF 轮廓简化精度，单位 mm |
| `topdown_mm_per_px` | `2.0` | 俯视矫正图比例，1px 表示多少 mm |
| `topdown_padding_mm` | `50.0` | 俯视矫正图四周留白，单位 mm |
| `dxf_postprocess_enabled` | `true` | 是否启用 DXF 后处理 |
| `dxf_notch_fill_enabled` | `true` | 是否启用局部凹陷修复 |
| `dxf_notch_fill_max_width_mm` | `130` | 凹陷修复最大宽度，单位 mm |
| `dxf_notch_fill_max_depth_mm` | `60` | 凹陷修复最大深度，单位 mm |

curl 示例：

```bash
curl -X POST "http://127.0.0.1:8000/measure" \
  -H "X-Auth-Token: tk_c2VjcmV0LXJhbmRvbS10b2tlbi0xMjM0NTY3OA" \
  -F "image=@test.jpg" \
  -F "a4_orientation=auto" \
  -F "paper_source=yolo" \
  -F "paper_rect_mode=approx_poly" \
  -F "simplify_mm=3.0"
```

Python 示例见 `sample_request.py`。

## 返回结果

返回结构示例：

```json
{
  "ok": true,
  "run_dir": "measure_out/test_ab12cd34",
  "paths": {
    "dxf": "measure_out/test_ab12cd34/plate_outer.dxf",
    "result_json": "measure_out/test_ab12cd34/result.json",
    "canonical_image": "measure_out/test_ab12cd34/input_canonical_used_by_yolo_and_sam2.png",
    "paper_mask": "measure_out/test_ab12cd34/paper_mask.png",
    "plate_raw_mask": "measure_out/test_ab12cd34/plate_raw_mask.png",
    "plate_final_with_paper_fill": "measure_out/test_ab12cd34/plate_final_with_paper_fill.png",
    "debug_overlay": "measure_out/test_ab12cd34/debug_overlay.jpg",
    "debug_mm_preview": "measure_out/test_ab12cd34/debug_mm_preview.png"
  },
  "urls": {
    "dxf": "/files/test_ab12cd34/plate_outer.dxf",
    "result_json": "/files/test_ab12cd34/result.json"
  },
  "plate_dimensions": {
    "length_mm": 1000.0,
    "width_mm": 500.0,
    "area_m2": 0.5,
    "perimeter_mm": 3000.0
  }
}
```

字段说明：

- `paths` 是服务端本地文件路径。
- `urls` 是通过 FastAPI 静态文件映射暴露的下载路径，前缀默认是 `/files`。
- `plate_dimensions` 包含钢板尺寸、面积、周长；圆形件还会包含圆心、半径、直径等字段。
- `a4` 包含 A4 方向、四角点和标定信息。
- `paper`、`plate`、`fill_paper_to_plate` 包含识别和 mask 处理过程信息。
- `topdown` 包含俯视矫正图输出信息。

## 输出文件

每次请求会在 `OUTPUT_ROOT` 下生成一个独立目录，例如：

```text
measure_out/
  _uploads/
  test_ab12cd34/
    input_canonical_used_by_yolo_and_sam2.png
    paper_mask.png
    plate_raw_mask.png
    plate_final_with_paper_fill.png
    plate_outer.dxf
    debug_overlay.jpg
    debug_mm_preview.png
    debug_dxf_postprocess_preview.png
    result.json
```

`plate_outer.dxf` 只包含钢板外轮廓，不包含 A4 纸轮廓。

## 控制台模式

可以不启动 HTTP 服务，直接调用核心算法：

```bash
python -m app.cli --image test.jpg --model best2.pt --out measure_out
```

常用参数与 HTTP 接口基本一致：

```bash
python -m app.cli ^
  --image test.jpg ^
  --model best2.pt ^
  --out measure_out ^
  --paper-source yolo ^
  --a4-orientation auto ^
  --paper-rect-mode approx_poly ^
  --simplify-mm 3.0
```

## Windows 部署

完整说明见 `WINDOWS_DEPLOY.md`。最小流程：

```bat
scripts\install_windows.bat
scripts\start_windows.bat
```

部署到局域网机器时，需要：

- 修改 `scripts\start_windows.bat` 中的 `APP_TOKEN` 和 `YOLO_MODEL_PATH`。
- 确认服务器防火墙放行 8000 端口。
- 访问 `http://服务器IP:8000/ui/` 使用上传页面。
- 访问 `http://服务器IP:8000/docs` 查看接口文档。

## 使用建议

- 生产环境不要使用默认 `APP_TOKEN`。
- 初期保持 `SERIALIZE_PROCESSING=true`，确认 GPU/SAM2 并发稳定后再考虑关闭。
- 如果尺寸偏差明显，优先检查 A4 纸是否和钢板在同一平面。
- 对 A4 四角识别不准的图片，可传 `paper_points` 手动指定四角坐标。
- A4 方向明显时，可把 `a4_orientation` 固定为 `landscape` 或 `portrait`。
- 广角畸变、钢板弯曲、A4 遮挡、A4 与钢板不共面都会影响尺寸精度。

## 常见问题

### 401 Invalid auth token

请求头中的 token 与 `APP_TOKEN` 不一致。检查页面或调用方填写的 `X-Auth-Token`。

### YOLO 权重文件不存在

确认 `YOLO_MODEL_PATH` 或请求中的 `model_path` 指向真实存在的 `.pt` 文件。

### 未得到 paper mask

图片中 A4 纸未被 YOLO/SAM2 正确识别。可以尝试：

- 降低或调整 `conf`。
- 确认 `paper_class` 与模型类别名一致。
- 使用 `paper_source=sam2` 并开启 `paper_sam2_yolo_fallback=true`。
- 直接传入 `paper_points`。

### SAM2 调用失败

确认当前 Python 环境中 `osam` / SAM2 能独立运行，并且 `SAM_MODEL` 名称与本机环境一致。

### 其他电脑无法访问服务

确认服务使用 `0.0.0.0` 监听，并放行 Windows 防火墙端口：

```bat
netsh advfirewall firewall add rule name="PlateMeasure8000" dir=in action=allow protocol=TCP localport=8000
```
