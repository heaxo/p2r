from __future__ import annotations

import sys


class RegistryHelper:
    """Read Lantek install information from Windows registry."""

    REGISTRY_PATH = r"SOFTWARE\WOW6432Node\Lantek"
    VALUE_NAME = "MainDir"

    @staticmethod
    def get_install_dir() -> str | None:
        if sys.platform != "win32":
            return None

        try:
            import winreg

            access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, RegistryHelper.REGISTRY_PATH, 0, access) as key:
                value, _ = winreg.QueryValueEx(key, RegistryHelper.VALUE_NAME)
        except Exception:
            return None

        text = str(value).strip() if value is not None else ""
        return text or None
