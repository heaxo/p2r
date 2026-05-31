const { app, BrowserWindow, dialog, shell, Menu, Tray } = require('electron');
const fs = require('fs');
const http = require('http');
const net = require('net');
const path = require('path');
const { spawn } = require('child_process');

app.setName('Pic2Remnant');

const gotSingleInstanceLock = app.requestSingleInstanceLock();
if (!gotSingleInstanceLock) {
  app.quit();
}

let mainWindow = null;
let startupWindow = null;
let tray = null;
let backendProcess = null;
let backendPort = null;
let licenseUsable = false;
let licenseMonitor = null;
let isQuitting = false;
let startupProgress = 1;
let startupTargetProgress = 1;
let startupMessage = '正在启动...';
let startupProgressTimer = null;
let startupStartedAt = 0;
const STARTUP_ANIMATION_MS = 7000;
const STARTUP_AUTO_PROGRESS_CAP = 95;

function appUrl(page = '') {
  const cleanPage = String(page || '').replace(/^\/+/, '');
  return `http://127.0.0.1:${backendPort}/ui/${cleanPage}`;
}

function loadUiPage(page) {
  if (!mainWindow || !backendPort) {
    return;
  }
  if (!licenseUsable) {
    showUnavailablePage();
    return;
  }
  mainWindow.loadURL(appUrl(page));
}

function showMainWindow() {
  if (!mainWindow) {
    if (backendPort) {
      createWindow();
    }
    return;
  }

  if (mainWindow.isMinimized()) {
    mainWindow.restore();
  }
  mainWindow.show();
  mainWindow.focus();
}

function createTray() {
  if (tray) {
    return;
  }

  tray = new Tray(path.join(__dirname, 'icon.ico'));
  tray.setToolTip('Pic2Remnant');
  tray.setContextMenu(Menu.buildFromTemplate([
    {
      label: '打开',
      click: showMainWindow,
    },
    { type: 'separator' },
    {
      label: '退出',
      click: () => {
        isQuitting = true;
        stopBackend();
        app.quit();
      },
    },
  ]));
  tray.on('double-click', showMainWindow);
}

function startupHtml(percent, message) {
  const safePercent = Math.max(0, Math.min(100, Number(percent) || 0));
  const safeMessage = String(message || '正在启动...');
  return `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>启动中</title>
  <style>
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: #1f2937;
      background: #f3f6fb;
      overflow: hidden;
    }
    body {
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .shell {
      width: min(460px, calc(100vw - 64px));
      padding: 0;
    }
    .message {
      min-height: 22px;
      margin-bottom: 14px;
      font-size: 14px;
      color: #344054;
      text-align: center;
    }
    .track {
      width: 100%;
      height: 10px;
      overflow: hidden;
      background: #d9e2ef;
      border-radius: 999px;
    }
    .bar {
      width: ${safePercent}%;
      height: 100%;
      background: linear-gradient(90deg, #2d579b, #4c7fd3);
      border-radius: 999px;
      transition: width .25s ease;
    }
    .percent {
      margin-top: 10px;
      text-align: right;
      font-size: 12px;
      color: #667085;
    }
  </style>
</head>
<body>
  <div class="shell">
    <div class="message">${safeMessage}</div>
    <div class="track"><div class="bar"></div></div>
    <div class="percent">${safePercent}%</div>
  </div>
</body>
</html>`;
}

function renderStartupWindow() {
  if (!startupWindow || startupWindow.isDestroyed()) {
    return;
  }
  startupWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(startupHtml(startupProgress, startupMessage))}`);
}

function updateStartupWindow(percent, message) {
  startupMessage = String(message || startupMessage);
  const requestedProgress = Math.max(1, Math.min(100, Number(percent) || 1));
  if (requestedProgress >= 100) {
    startupTargetProgress = 100;
  }
  renderStartupWindow();
}

function completeStartupWindow(message) {
  startupMessage = String(message || startupMessage);
  startupTargetProgress = 100;
  startupProgress = 100;
  renderStartupWindow();
}

function startStartupProgressAnimation() {
  if (startupProgressTimer) {
    clearInterval(startupProgressTimer);
  }

  startupProgressTimer = setInterval(() => {
    if (!startupWindow || startupWindow.isDestroyed()) {
      clearInterval(startupProgressTimer);
      startupProgressTimer = null;
      return;
    }

    const elapsed = Date.now() - startupStartedAt;
    const timedProgress = Math.min(
      STARTUP_AUTO_PROGRESS_CAP,
      1 + Math.floor((elapsed / STARTUP_ANIMATION_MS) * (STARTUP_AUTO_PROGRESS_CAP - 1)),
    );
    const nextTarget = Math.max(startupTargetProgress, timedProgress);

    if (startupProgress < nextTarget) {
      const distance = nextTarget - startupProgress;
      startupProgress += Math.max(1, Math.ceil(distance / 8));
      startupProgress = Math.min(startupProgress, nextTarget);
      renderStartupWindow();
    }
  }, 160);
}

function closeStartupWindow() {
  if (startupProgressTimer) {
    clearInterval(startupProgressTimer);
    startupProgressTimer = null;
  }
  if (startupWindow && !startupWindow.isDestroyed()) {
    startupWindow.close();
  }
  startupWindow = null;
}

function createStartupWindow() {
  startupProgress = 1;
  startupTargetProgress = 1;
  startupMessage = '正在启动...';
  startupStartedAt = Date.now();
  startupWindow = new BrowserWindow({
    width: 520,
    height: 320,
    resizable: false,
    maximizable: false,
    minimizable: false,
    show: false,
    icon: path.join(__dirname, 'icon.ico'),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  startupWindow.setMenuBarVisibility(false);
  startupWindow.once('ready-to-show', () => {
    if (startupWindow && !startupWindow.isDestroyed()) {
      startupWindow.show();
    }
  });
  renderStartupWindow();
  startStartupProgressAnimation();
}

function backendUrl(pathname) {
  return `http://127.0.0.1:${backendPort}${pathname}`;
}

function requestJson(pathname) {
  return new Promise((resolve) => {
    const request = http.get(backendUrl(pathname), (response) => {
      let body = '';
      response.setEncoding('utf8');
      response.on('data', (chunk) => {
        body += chunk;
      });
      response.on('end', () => {
        try {
          resolve({ statusCode: response.statusCode, body: JSON.parse(body || '{}') });
        } catch (_) {
          resolve({ statusCode: response.statusCode, body: {} });
        }
      });
    });

    request.on('error', () => resolve({ statusCode: 0, body: {} }));
    request.setTimeout(3000, () => {
      request.destroy();
      resolve({ statusCode: 0, body: {} });
    });
  });
}

async function checkLicenseStatus() {
  if (!backendPort) {
    return false;
  }
  const response = await requestJson('/license/status');
  return response.statusCode === 200 && response.body && response.body.ok === true;
}

function showUnavailablePage() {
  if (!mainWindow) {
    return;
  }
  const html = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>无法使用</title>
  <style>
    html, body {
      width: 100%;
      height: 100%;
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      color: #1f2937;
      background: #eef2f7;
    }
    body {
      display: flex;
      align-items: center;
      justify-content: center;
    }
    h1 {
      margin: 0;
      font-size: 34px;
      font-weight: 700;
      letter-spacing: 0;
    }
  </style>
</head>
<body>
  <h1>无法使用</h1>
</body>
</html>`;
  mainWindow.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(html)}`);
}

function startLicenseMonitor() {
  if (licenseMonitor) {
    clearInterval(licenseMonitor);
  }

  licenseMonitor = setInterval(async () => {
    const usable = await checkLicenseStatus();
    if (!licenseUsable && usable) {
      licenseUsable = true;
      loadUiPage('datasets.html');
      return;
    }

    if (licenseUsable && !usable) {
      licenseUsable = false;
      showUnavailablePage();
      return;
    }

    licenseUsable = usable;
  }, 60000);
}

function createAppMenu() {
  const runtimeRoot = path.join(app.getPath('userData'), 'runtime');
  const template = [
    {
      label: '功能',
      submenu: [
        {
          label: '数据集',
          accelerator: 'Ctrl+1',
          click: () => loadUiPage('datasets.html'),
        },
        /*{
          label: '批量识别',
          accelerator: 'Ctrl+2',
          click: () => loadUiPage(''),
        },*/
        { type: 'separator' },
        {
          label: '退出',
          role: 'quit',
        },
      ],
    },
    {
      label: '视图',
      submenu: [
        { label: '刷新', role: 'reload' },
        { label: '强制刷新', role: 'forceReload' },
        { type: 'separator' },
        { label: '开发者工具', role: 'toggleDevTools' },
        { type: 'separator' },
        { label: '放大', role: 'zoomIn' },
        { label: '缩小', role: 'zoomOut' },
        { label: '实际大小', role: 'resetZoom' },
      ],
    },
    {
      label: '帮助',
      submenu: [
        {
          label: '打开运行目录',
          click: () => shell.openPath(runtimeRoot),
        },
        {
          label: '关于',
          click: () => {
            dialog.showMessageBox(mainWindow, {
              type: 'info',
              title: '关于 Pic2Remnant',
              message: 'Pic2Remnant',
              detail: 'Lantek 余料图识别',
            });
          },
        },
      ],
    },
  ];

  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function findFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      const port = address.port;
      server.close(() => resolve(port));
    });
  });
}

function getPythonExecutable(projectRoot) {
  const candidates = [
    path.join(projectRoot, '.venv', 'Scripts', 'python.exe'),
    path.join(projectRoot, 'venv', 'Scripts', 'python.exe'),
    process.platform === 'win32' ? 'python' : 'python3',
  ];

  for (const candidate of candidates) {
    if (candidate === 'python' || candidate === 'python3' || fs.existsSync(candidate)) {
      return candidate;
    }
  }

  return process.platform === 'win32' ? 'python' : 'python3';
}

function getBackendCommand(projectRoot) {
  if (app.isPackaged) {
    const exeName = process.platform === 'win32' ? 'p2r-backend.exe' : 'p2r-backend';
    return {
      command: path.join(process.resourcesPath, 'backend', exeName),
      args: [],
      cwd: process.resourcesPath,
    };
  }

  return {
    command: getPythonExecutable(projectRoot),
    args: [path.join(projectRoot, 'desktop', 'backend_entry.py')],
    cwd: projectRoot,
  };
}

function getModelPath(projectRoot) {
  const candidates = app.isPackaged
    ? [
        path.join(process.resourcesPath, 'backend', '_internal', 'best2.pt'),
        path.join(process.resourcesPath, 'backend', 'best2.pt'),
      ]
    : [
        path.join(projectRoot, 'best2.pt'),
      ];

  return candidates.find((candidate) => fs.existsSync(candidate));
}

function appendLog(logStream, chunk) {
  if (!chunk) {
    return;
  }
  logStream.write(chunk);
}

async function startBackend(onProgress = () => {}) {
  onProgress(12, '正在准备运行目录...');
  const projectRoot = path.resolve(__dirname, '..');
  const userData = app.getPath('userData');
  const runtimeRoot = path.join(userData, 'runtime');
  const outputRoot = path.join(runtimeRoot, 'measure_out');
  const logDir = path.join(runtimeRoot, 'logs');
  const dataDir = path.join(runtimeRoot, 'data');
  const licenseRoot = app.isPackaged ? path.dirname(process.execPath) : projectRoot;

  fs.mkdirSync(outputRoot, { recursive: true });
  fs.mkdirSync(logDir, { recursive: true });
  fs.mkdirSync(dataDir, { recursive: true });

  onProgress(24, '正在分配本地端口...');
  backendPort = await findFreePort();
  const backend = getBackendCommand(projectRoot);
  const backendLogPath = path.join(logDir, 'backend-process.log');
  const logStream = fs.createWriteStream(backendLogPath, { flags: 'a' });
  const modelPath = getModelPath(projectRoot);

  const env = {
    ...process.env,
    HOST: '127.0.0.1',
    PORT: String(backendPort),
    OUTPUT_ROOT: outputRoot,
    LOG_DIR: logDir,
    TASK_DB_PATH: path.join(dataDir, 'tasks.sqlite3'),
    LICENSE_ROOT: licenseRoot,
    LICENSE_STATE_PATH: path.join(dataDir, 'license_state.json'),
    DESKTOP_APP: '1',
    PYTHONUNBUFFERED: '1',
  };
  if (modelPath) {
    env.YOLO_MODEL_PATH = modelPath;
  }

  onProgress(42, '正在启动后台服务...');
  backendProcess = spawn(backend.command, backend.args, {
    cwd: backend.cwd,
    env,
    windowsHide: true,
  });

  backendProcess.stdout.on('data', (chunk) => appendLog(logStream, chunk));
  backendProcess.stderr.on('data', (chunk) => appendLog(logStream, chunk));
  const backendProcessError = new Promise((_, reject) => {
    backendProcess.once('error', (error) => {
    appendLog(logStream, `\nBackend process error: ${error.message}\n`);
      reject(error);
    });
  });
  backendProcess.on('exit', (code, signal) => {
    appendLog(logStream, `\nBackend exited: code=${code} signal=${signal}\n`);
    logStream.end();
    backendProcess = null;
  });

  try {
    onProgress(66, '正在等待后台服务就绪...');
    await Promise.race([waitForBackend(backendPort), backendProcessError]);
    onProgress(86, '后台服务已就绪...');
  } catch (error) {
    closeStartupWindow();
    throw new Error(`${error.message}\nBackend log: ${backendLogPath}`);
  }
}

function waitForBackend(port) {
  const startedAt = Date.now();
  const timeoutMs = 120000;

  return new Promise((resolve, reject) => {
    const check = () => {
      const request = http.get(`http://127.0.0.1:${port}/health`, (response) => {
        response.resume();
        if (response.statusCode === 200) {
          resolve();
          return;
        }
        retry();
      });

      request.on('error', retry);
      request.setTimeout(2000, () => {
        request.destroy();
        retry();
      });
    };

    const retry = () => {
      if (Date.now() - startedAt > timeoutMs) {
        reject(new Error('Backend did not become ready in time.'));
        return;
      }
      setTimeout(check, 500);
    };

    check();
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1280,
    height: 840,
    minWidth: 980,
    minHeight: 680,
    show: false,
    icon: path.join(__dirname, 'icon.ico'),
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  createAppMenu();
  mainWindow.setMenuBarVisibility(true);
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    closeStartupWindow();
  });
  mainWindow.on('close', (event) => {
    if (isQuitting) {
      return;
    }
    event.preventDefault();
    mainWindow.hide();
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  if (licenseUsable) {
    mainWindow.loadURL(appUrl('datasets.html'));
  } else {
    showUnavailablePage();
  }
}

function stopBackend() {
  if (licenseMonitor) {
    clearInterval(licenseMonitor);
    licenseMonitor = null;
  }
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
}

app.whenReady().then(async () => {
  try {
    createTray();
    createStartupWindow();
    await startBackend(updateStartupWindow);
    updateStartupWindow(92, '正在检查使用权限...');
    licenseUsable = await checkLicenseStatus();
    completeStartupWindow('正在打开界面...');
    createWindow();
    startLicenseMonitor();
  } catch (error) {
    closeStartupWindow();
    dialog.showErrorBox('启动失败', String(error && error.message ? error.message : error));
    app.quit();
  }
});

app.on('second-instance', () => {
  showMainWindow();
});

app.on('window-all-closed', () => {
  if (isQuitting) {
    stopBackend();
  }
});

app.on('before-quit', () => {
  isQuitting = true;
  stopBackend();
});

app.on('activate', () => {
  showMainWindow();
});
