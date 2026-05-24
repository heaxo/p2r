from __future__ import annotations

import os
import socket
import sys

import uvicorn


def _port_is_available(host: str, port: int) -> bool:
    bind_host = "" if host in {"0.0.0.0", "::"} else host
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((bind_host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() in {"1", "true", "yes"}

    if not reload and not _port_is_available(host, port):
        print(
            f"端口 {port} 已被占用，服务可能已经在后台运行。"
            f"请先关闭旧的 python/uvicorn 进程，或设置 PORT 使用其它端口。",
            file=sys.stderr,
        )
        sys.exit(1)

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=reload,
    )
