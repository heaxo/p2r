# plate_measure_http_service

这是把原控制台版 `plate_measure_yolo_sam2_console.py` 改成的 HTTP 服务版。核心识别、A4 标定、尺寸计算、DXF 生成逻辑保持在 `app/core/algorithm.py`，HTTP 层只负责上传图片、鉴权、组装参数、调用核心流程并返回文件路径。

## 目录结构

```text
plate_measure_http_service/
  app/
    main.py                  # FastAPI 入口
    api/routes.py             # HTTP 接口
    config.py                 # 环境变量配置
    security.py               # 简单 token 鉴权
    schemas.py                # 响应结构
    cli.py                    # 保留控制台入口
    core/algorithm.py         # 原控制台核心算法逻辑
    services/model_cache.py   # YOLO 模型缓存
    services/measure_service.py
  run_server.py
  sample_request.py
  requirements.txt
  .env.example
```

## 安装

```bash
pip install -r requirements.txt
```

还需要你原来可以运行 SAM2 的 `osam/SAM2` 环境，否则 SAM2 调用会失败。

## 启动

Windows CMD 示例：

```bat
set APP_TOKEN=your-token
set YOLO_MODEL_PATH=D:\lantek\project-python\p2r\best2.pt
set OUTPUT_ROOT=D:\lantek\project-python\p2r\measure_out
python run_server.py
```

PowerShell 示例：

```powershell
$env:APP_TOKEN="your-token"
$env:YOLO_MODEL_PATH="D:\lantek\project-python\p2r\best2.pt"
$env:OUTPUT_ROOT="D:\lantek\project-python\p2r\measure_out"
python run_server.py
```

默认地址：

```text
http://127.0.0.1:8000
```

## 接口

### 健康检查

```text
GET /health
```

### 识别并生成结果

```text
POST /measure
Header: X-Auth-Token: your-token
Content-Type: multipart/form-data
```

常用表单参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| image | 必填 | 上传图片 |
| model_path | 环境变量 YOLO_MODEL_PATH | 可选，YOLO 权重路径 |
| sam_model | 环境变量 SAM_MODEL | SAM2 模型名 |
| imgsz | 1280 | YOLO 推理尺寸 |
| conf | 0.35 | YOLO 置信度 |
| plate_class | plate | 钢板类别名 |
| paper_class | paper | A4纸类别名，可逗号分隔 |
| paper_source | yolo | yolo 或 sam2 |
| a4_orientation | auto | auto、landscape、portrait |
| paper_points | 空 | 可直接传 A4 四角坐标，格式 `x1,y1;x2,y2;x3,y3;x4,y4` |
| paper_rect_mode | approx_poly | robust_fit、approx_poly、min_area_rect、raw |
| simplify_mm | 3.0 | DXF 轮廓简化精度，单位 mm |

curl 示例：

```bash
curl -X POST "http://127.0.0.1:8000/measure" \
  -H "X-Auth-Token: your-token" \
  -F "image=@test.jpg" \
  -F "a4_orientation=auto" \
  -F "paper_source=yolo" \
  -F "paper_rect_mode=approx_poly"
```

返回示例字段：

```json
{
  "ok": true,
  "run_dir": "measure_out/test_ab12cd34",
  "paths": {
    "dxf": "measure_out/test_ab12cd34/plate_outer.dxf",
    "result_json": "measure_out/test_ab12cd34/result.json",
    "paper_mask": "measure_out/test_ab12cd34/paper_mask.png",
    "plate_raw_mask": "measure_out/test_ab12cd34/plate_raw_mask.png",
    "plate_final_with_paper_fill": "measure_out/test_ab12cd34/plate_final_with_paper_fill.png",
    "debug_overlay": "measure_out/test_ab12cd34/debug_overlay.jpg",
    "debug_mm_preview": "measure_out/test_ab12cd34/debug_mm_preview.png"
  },
  "urls": {
    "dxf": "/files/test_ab12cd34/plate_outer.dxf"
  },
  "plate_dimensions": {
    "length_mm": 1000.0,
    "width_mm": 500.0
  }
}
```

`paths` 是服务器本地路径；`urls` 是通过 FastAPI 静态文件映射出来的访问路径。

## 执行效率上的调整

1. YOLO 模型使用 `app/services/model_cache.py` 做进程内缓存，避免每次请求重复加载模型。
2. 默认开启 `SERIALIZE_PROCESSING=true`，串行处理推理任务，避免多请求同时抢占 GPU 或 SAM2 资源导致不稳定。
3. 保留原来的 canonical 中间图逻辑，YOLO 和 SAM2 仍共用同一张图，避免 EXIF、旋转和尺寸导致坐标错位。

## 保留控制台调用

```bash
python -m app.cli --image test.jpg --model best2.pt --out measure_out
```


## 内置HTML页面

本版本已内置测试页面：

```text
http://127.0.0.1:8000/ui/
```

页面支持上传图片、填写接口参数、查看返回 JSON、预览 mask/debug 图片，并下载 DXF。

Windows 部署可参考：

```text
WINDOWS_DEPLOY.md
scripts\install_windows.bat
scripts\start_windows.bat
```
