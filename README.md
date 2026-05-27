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

## 桌面版开发与打包

桌面版在 `desktop-electron` 分支开发，使用 Electron 作为桌面壳，复用现有 FastAPI 后端和 `frontend/index.html` 页面。

### 调试运行

在项目根目录运行：

```powershell
npm run desktop:dev
```

这个命令会启动 `desktop/main.js`。Electron 会自动启动 `desktop/backend_entry.py`，后端监听 `127.0.0.1` 的随机空闲端口，健康检查通过后打开现有 `/ui/` 页面。

首次运行前需要安装依赖：

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
npm install
```

如果只想单独调试后端，可以运行：

```powershell
.venv\Scripts\python desktop\backend_entry.py
```

默认访问地址：

```text
http://127.0.0.1:8000/ui/
```

### 打包安装包

先安装桌面打包依赖：

```powershell
.venv\Scripts\python -m pip install -r desktop\requirements-build.txt
```

然后执行：

```powershell
npm run desktop:pack
```

打包流程会先用 PyInstaller 构建 Python 后端：

```text
dist/p2r-backend/
```

然后用 electron-builder 生成 Windows 安装包：

```text
desktop-dist/Plate Measure Setup 1.2.0.exe
```

### 运行目录

桌面版运行时会把输出文件、日志和任务数据库放到 Electron 的用户数据目录下：

```text
runtime/measure_out
runtime/logs
runtime/data/tasks.sqlite3
```

开发模式下仍然可以继续使用 HTTP 服务版入口：

```powershell
.venv\Scripts\python run_server.py
```
