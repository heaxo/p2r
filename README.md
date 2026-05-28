# Pic2Remnant

钢板图片识别与 DXF 生成工具。项目同时支持 HTTP 服务版和 Electron 桌面版，桌面版复用同一套 FastAPI 后端、算法代码和前端页面。

## 功能简要说明

- 批量识别：通过 `/ui/` 上传多张图片，设置透视、目标尺寸、夹钳修复等参数，生成 DXF、预览图和 JSON 结果。
- 数据集管理：通过 `/ui/datasets.html` 创建数据集，录入图片、板材编号、数量、材质、厚度、尺寸参数和识别开关。
- Excel 导入：下载 Excel 模板后批量填写数据，再上传创建数据集。
- 历史记录：数据集和数据项保存到 SQLite，可查看历史识别状态、结果链接和 DXF。
- 数据集复制：复制已有数据集，复制后的状态为未识别。
- 导入至 Expert：数据集可选择 `Procesos` 或 `Masterlink` 导入器，并生成对应的 `.LST/.prc` 或 XML 后调用 Lantek 导入程序。
- 桌面菜单：Electron 菜单栏提供“数据集”和“批量识别”两个入口，桌面版默认进入数据集页面。

## 目录结构

```text
app/
  api/routes.py                  FastAPI 路由
  core/                          原有识别、DXF 后处理算法
  services/measure_service.py    单图/批量识别服务
  services/task_store.py         批量上传任务 SQLite 存储
  services/dataset_store.py      数据集 SQLite 存储
  services/dataset_service.py    数据集、Excel、识别编排服务
  services/lantek_registry.py    读取 Lantek 注册表安装目录
desktop/
  main.js                        Electron 主进程和桌面菜单
  preload.js                     Electron preload
  icon.ico / icon.png             桌面应用图标
  backend_entry.py               桌面版后端启动入口
  p2r-backend.spec               PyInstaller 打包配置
frontend/
  index.html                     批量识别页面
  datasets.html                  数据集管理页面
  vendor/bootstrap/              Bootstrap 本地静态资源
data/
  tasks.sqlite3                  HTTP 开发模式默认 SQLite 数据库
measure_out/
  _uploads/                      批量识别上传图片
  _datasets/                     数据集图片和识别输出
```

## 首次安装依赖

在项目根目录执行：

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
npm install
```

数据集页面使用 Bootstrap 5，本地资源位于 `frontend/vendor/bootstrap/`，桌面版离线运行时不依赖 CDN。

如果需要重新创建虚拟环境：

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## HTTP 服务版调试

启动后端：

```powershell
.venv\Scripts\python run_server.py
```

默认访问：

```text
http://127.0.0.1:8000/ui/
http://127.0.0.1:8000/ui/datasets.html
```

常用环境变量：

```text
HOST=0.0.0.0
PORT=8000
APP_TOKEN=tk_c2VjcmV0LXJhbmRvbS10b2tlbi0xMjM0NTY3OA
YOLO_MODEL_PATH=best2.pt
OUTPUT_ROOT=measure_out
TASK_DB_PATH=data/tasks.sqlite3
LOG_DIR=logs
```

## 桌面版调试

在项目根目录运行：

```powershell
npm run desktop:dev
```

Electron 会自动启动 `desktop/backend_entry.py`，后端监听 `127.0.0.1` 的随机空闲端口，健康检查通过后打开数据集页面。菜单栏“功能”中可以在“数据集”和“批量识别”之间切换。

如果只想单独调试桌面后端：

```powershell
.venv\Scripts\python desktop\backend_entry.py
```

## 数据集和 Excel 模板

数据集页面支持两种创建方式：

- 手动创建：名称为空时自动使用 `年月日_时分秒`；如果手动填写名称，后端会做唯一校验。
- Excel 导入：页面点击“下载 Excel 模板”，填写后上传。

Excel 模板列：

```text
图片地址
板材编号
数量
材质
厚度
尺寸1
尺寸2
x
y
是否启用钢板透视
是否启用夹钳修复
夹钳修复最大宽度mm
夹钳修复最大深度mm
```

说明：

- `图片地址` 必填，桌面版建议填写本机绝对路径。
- `数量` 可选，填写大于 0 的整数。
- `尺寸1` 和 `尺寸2` 不指定方向，两个值同时填写时，会按识别结果的长短边自动匹配。
- `x` 和 `y` 是明确指定 DXF X/Y 方向目标尺寸；当 `尺寸1/尺寸2` 有值时优先使用 `尺寸1/尺寸2`。
- 布尔列可填写 `是/否`、`true/false` 或 `1/0`。

## 打包命令

先安装打包依赖：

```powershell
.venv\Scripts\python -m pip install -r desktop\requirements-build.txt
npm install
```

### 只打包后端

后端使用 PyInstaller 打包，入口是 `desktop/backend_entry.py`，配置文件是 `desktop/p2r-backend.spec`。

```powershell
npm run backend:build
```

等价命令：

```powershell
.venv\Scripts\python -m PyInstaller desktop\p2r-backend.spec --noconfirm
```

输出目录：

```text
dist/p2r-backend/
```

### 前端资源

当前前端是 `frontend/` 下的静态 HTML/CSS/JS，没有单独的前端构建命令。打包后端时，`desktop/p2r-backend.spec` 会把整个 `frontend/` 目录一起打进 `dist/p2r-backend/`。

如果只改了前端页面，重新执行后端打包即可：

```powershell
npm run backend:build
```

### 整体桌面版打包

整体桌面版会先打后端，再用 electron-builder 打 Windows 安装包：

```powershell
npm run desktop:pack
```

打包流程：

```text
PyInstaller -> dist/p2r-backend/
electron-builder -> desktop-dist/Pic2Remnant Setup 1.2.0.exe
```

未安装版可从这里运行：

```text
desktop-dist/win-unpacked/Pic2Remnant.exe
```

## 桌面版运行数据位置

桌面版运行时会把输出、日志和 SQLite 数据库放到 Electron 用户数据目录下：

```text
runtime/measure_out
runtime/logs
runtime/data/tasks.sqlite3
```

在桌面程序菜单中点击“帮助 -> 打开运行目录”可直接打开该目录。

## 导入至 Expert

数据集页面点击“导入设置”可选择导入器：

- `Procesos`：生成 `.LST` 和 `.prc`，再调用 `Expert\Procesos.exe`。
- `Masterlink`：生成带 UTF-8 BOM 的 XML，再调用 `System\Common\XMLImporter.exe`。

Lantek 安装目录从注册表读取：

```text
HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Lantek
MainDir
```

导入命令执行完成后，程序只提示“已执行完成”，不判断 Lantek 内部是否导入成功，需要打开套料软件确认结果。

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

### 其他电脑无法访问 HTTP 服务

确认服务使用 `0.0.0.0` 监听，并放行 Windows 防火墙端口：

```bat
netsh advfirewall firewall add rule name="Pic2Remnant8000" dir=in action=allow protocol=TCP localport=8000
```
