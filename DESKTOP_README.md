# Desktop App

This branch adds an Electron desktop shell around the existing FastAPI and HTML app.

## Development

Install the existing Python dependencies first, then install the desktop dependencies:

```powershell
.venv\Scripts\python -m pip install -r requirements.txt
npm install
npm run desktop:dev
```

The Electron main process starts the Python backend on `127.0.0.1` with a free local port, waits for `/health`, then opens `/ui/`.

Runtime output is stored under Electron's user data directory:

- `runtime/measure_out`
- `runtime/logs`
- `runtime/data/tasks.sqlite3`

## Packaging

Install PyInstaller into the Python environment:

```powershell
.venv\Scripts\python -m pip install -r desktop\requirements-build.txt
```

Then package the app:

```powershell
npm run desktop:pack
```

The build creates a PyInstaller backend under `dist/p2r-backend`, then copies it into the Electron installer as an extra resource.
