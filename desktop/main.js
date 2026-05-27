const { app, BrowserWindow, dialog, shell } = require('electron');
const fs = require('fs');
const http = require('http');
const net = require('net');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow = null;
let backendProcess = null;
let backendPort = null;

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

async function startBackend() {
  const projectRoot = path.resolve(__dirname, '..');
  const userData = app.getPath('userData');
  const runtimeRoot = path.join(userData, 'runtime');
  const outputRoot = path.join(runtimeRoot, 'measure_out');
  const logDir = path.join(runtimeRoot, 'logs');
  const dataDir = path.join(runtimeRoot, 'data');

  fs.mkdirSync(outputRoot, { recursive: true });
  fs.mkdirSync(logDir, { recursive: true });
  fs.mkdirSync(dataDir, { recursive: true });

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
    DESKTOP_APP: '1',
    PYTHONUNBUFFERED: '1',
  };
  if (modelPath) {
    env.YOLO_MODEL_PATH = modelPath;
  }

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
    await Promise.race([waitForBackend(backendPort), backendProcessError]);
  } catch (error) {
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
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.setMenuBarVisibility(false);
  mainWindow.once('ready-to-show', () => mainWindow.show());
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.loadURL(`http://127.0.0.1:${backendPort}/ui/`);
}

function stopBackend() {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
}

app.whenReady().then(async () => {
  try {
    await startBackend();
    createWindow();
  } catch (error) {
    dialog.showErrorBox('启动失败', String(error && error.message ? error.message : error));
    app.quit();
  }
});

app.on('window-all-closed', () => {
  stopBackend();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', stopBackend);

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0 && backendPort) {
    createWindow();
  }
});
