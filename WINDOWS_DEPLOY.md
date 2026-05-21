# Windows服务器部署说明

## 1. 环境要求

建议使用 Windows Server 2019/2022 或 Windows 10/11。

需要准备：

1. Python 3.10 或 3.11。
2. 你原来控制台程序能正常运行的 osam / SAM2 环境。
3. YOLO 权重文件，例如 `best2.pt`。
4. 如果使用 GPU，需要提前装好 CUDA、PyTorch GPU 版本，并保证原控制台程序可正常推理。

本项目只是把原控制台程序包成 HTTP 服务，没有重新实现 SAM2 环境安装。

## 2. 安装依赖

进入项目目录，执行：

```bat
scripts\install_windows.bat
```

如果服务器不能联网，可以在能联网的机器上提前下载 whl 包，再拷贝到服务器离线安装。

## 3. 修改启动配置

打开：

```text
scripts\start_windows.bat
```

至少修改这两个值：

```bat
set APP_TOKEN=change-me-please
set YOLO_MODEL_PATH=best2.pt
```

`APP_TOKEN` 是接口简单授权 token，HTML 页面里也要填写同一个值。

## 4. 启动服务

```bat
scripts\start_windows.bat
```

启动后访问：

```text
http://服务器IP:8000/ui/
```

接口文档：

```text
http://服务器IP:8000/docs
```

健康检查：

```text
http://服务器IP:8000/health
```

## 5. Windows防火墙

如果其他电脑访问不了，需要放开 8000 端口。

管理员 CMD 执行：

```bat
netsh advfirewall firewall add rule name="PlateMeasure8000" dir=in action=allow protocol=TCP localport=8000
```

## 6. 生产环境建议

1. `APP_TOKEN` 不要使用默认值。
2. `OUTPUT_ROOT` 建议放到固定磁盘目录，例如 `D:\PlateMeasure\measure_out`。
3. 初期保留 `SERIALIZE_PROCESSING=true`，避免多个请求同时抢 GPU/SAM2。
4. 运行稳定后，再考虑用 NSSM 或 Windows 服务托管 `scripts\start_windows.bat`。
5. 如果图片很大，可以根据实际情况调大或调小 `MAX_UPLOAD_MB`。

## 7. HTML页面说明

页面文件位置：

```text
frontend\index.html
```

服务启动后通过 `/ui/` 访问，不需要单独部署 nginx 或 IIS。

页面会调用：

```text
POST /measure
```

请求头：

```text
X-Auth-Token: 你的APP_TOKEN
```

上传成功后页面会展示：

1. 长宽、面积、周长。
2. DXF 下载按钮。
3. result.json 下载按钮。
4. mask、debug、topdown 等图片预览。
5. 完整接口返回 JSON。


## 常见启动报错

### No module named uvicorn

说明当前没有使用项目里的 `.venv`，或者依赖没有安装成功。先执行：

```bat
scripts\install_windows.bat
```

再执行：

```bat
scripts\start_windows.bat
```

如果日志里显示使用的是 `python-embed`，不要用 embedded Python 部署 Web 服务。请安装官方 Python 3.10 或 3.11，并勾选 pip，然后重新执行安装脚本。

### 系统找不到指定的路径

通常是 `.venv` 没创建成功，或者直接运行了启动脚本但没有先安装依赖。新版脚本会在启动前检查 `.venv\Scripts\python.exe` 是否存在。
