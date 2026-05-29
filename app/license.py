from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:
    class _FallbackLogger:
        def debug(self, *args: object, **kwargs: object) -> None:
            return None

        def warning(self, *args: object, **kwargs: object) -> None:
            return None

        def exception(self, *args: object, **kwargs: object) -> None:
            return None

    logger = _FallbackLogger()

LICENSE_FILENAME = "license.key"
LICENSE_TOKEN_PREFIX = "P2R1"
CLOCK_TOLERANCE_SECONDS = 300

PUBLIC_RSA_N = int(
    "76aa821f8a7c0cf462674b85b0cc952817b232570a12b63bd2f6fb1b2d061d2970c069614dfe5249f4307e40e6bcce20c4540f1d1f68e73e0b3ca5ed52c7b4f11cc405cdca3867cb2be4c29f70c8690b8e4c25cd6be3a26a0f6176a214136b52995fd9bc0f03f09afaafc69ebb1720d3061bde41238bfcc2faa32b3608d178deee70ef39abd2149c13b9ef00d1bdf35e28c95fe8d57019da149809694b759c9cb2f431a59bc14b34cf485f469313d9dca9c7d177fe367ede8bc2f71f69ee9e45dc5c9e167bd37b2d8bccb65ce8612d69f807c99ae81ec70115782c6188f7078d3240e94400e6ad77f003def0ff141a878086215f9467c2dc76e922de6956d7c3",
    16,
)
PUBLIC_RSA_E = 65537

_SHA256_DIGEST_INFO_PREFIX = bytes.fromhex("3031300d060960864801650304020105000420")
_STATE_SECRET = bytes.fromhex("f880a5390359e7767bd57feb991880a5e22c379e7039e59c89fb4fb3831b7cf3")


class LicenseError(Exception):
    def __init__(self, code: str, message: str = "无法使用") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class LicenseInfo:
    license_id: str
    days: int
    issued_at: datetime
    valid_from: datetime
    expires_at: datetime
    customer: str | None = None

    @property
    def remaining_seconds(self) -> int:
        seconds = int((self.expires_at - _utc_now()).total_seconds())
        return max(0, seconds)

    @property
    def remaining_days(self) -> int:
        seconds = self.remaining_seconds
        return (seconds + 86399) // 86400

    def public_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "license_id": self.license_id,
            "days": self.days,
            "issued_at": _format_dt(self.issued_at),
            "valid_from": _format_dt(self.valid_from),
            "expires_at": _format_dt(self.expires_at),
            "remaining_seconds": self.remaining_seconds,
            "remaining_days": self.remaining_days,
            "customer": self.customer,
            "file": str(license_file_path()),
        }


@dataclass(frozen=True)
class LicenseStatus:
    ok: bool
    code: str
    message: str
    info: LicenseInfo | None = None

    def public_dict(self) -> dict[str, Any]:
        if self.ok and self.info:
            return self.info.public_dict()
        return {
            "ok": False,
            "code": self.code,
            "detail": self.message,
            "file": str(license_file_path()),
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _format_dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_dt(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LicenseError("invalid_license_file") from exc
    if parsed.tzinfo is None:
        raise LicenseError("invalid_license_file")
    return parsed.astimezone(timezone.utc)


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _emsa_pkcs1_v1_5_encode(digest: bytes, key_bytes: int) -> bytes:
    digest_info = _SHA256_DIGEST_INFO_PREFIX + digest
    padding_len = key_bytes - len(digest_info) - 3
    if padding_len < 8:
        raise LicenseError("invalid_signature")
    return b"\x00\x01" + (b"\xff" * padding_len) + b"\x00" + digest_info


def _verify_signature(payload: bytes, signature: bytes) -> bool:
    key_bytes = (PUBLIC_RSA_N.bit_length() + 7) // 8
    if len(signature) != key_bytes:
        return False

    signature_int = int.from_bytes(signature, "big")
    if signature_int >= PUBLIC_RSA_N:
        return False

    actual = pow(signature_int, PUBLIC_RSA_E, PUBLIC_RSA_N).to_bytes(key_bytes, "big")
    expected = _emsa_pkcs1_v1_5_encode(hashlib.sha256(payload).digest(), key_bytes)
    return hmac.compare_digest(actual, expected)


def decode_license_key(token: str) -> LicenseInfo:
    parts = token.strip().split(".")
    if len(parts) != 3 or parts[0] != LICENSE_TOKEN_PREFIX:
        raise LicenseError("invalid_license_file")

    try:
        payload_bytes = _b64url_decode(parts[1])
        signature = _b64url_decode(parts[2])
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise LicenseError("invalid_license_file") from exc

    if not isinstance(payload, dict) or not _verify_signature(payload_bytes, signature):
        raise LicenseError("invalid_signature")

    try:
        version = int(payload.get("version"))
        days = int(payload.get("days"))
        license_id = str(payload.get("license_id") or "").strip()
    except (TypeError, ValueError) as exc:
        raise LicenseError("invalid_license_file") from exc

    if version != 1 or days <= 0 or not license_id:
        raise LicenseError("invalid_license_file")

    return LicenseInfo(
        license_id=license_id,
        days=days,
        issued_at=_parse_dt(str(payload.get("issued_at")), "issued_at"),
        valid_from=_parse_dt(str(payload.get("valid_from")), "valid_from"),
        expires_at=_parse_dt(str(payload.get("expires_at")), "expires_at"),
        customer=str(payload["customer"]).strip() if payload.get("customer") else None,
    )


def _program_root() -> Path:
    configured = os.getenv("LICENSE_ROOT")
    if configured:
        return Path(configured)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def license_file_path() -> Path:
    configured = os.getenv("LICENSE_FILE")
    if configured:
        return Path(configured)
    return _program_root() / LICENSE_FILENAME


def _state_paths() -> list[Path]:
    paths: list[Path] = []
    configured = os.getenv("LICENSE_STATE_PATH")
    if configured:
        paths.append(Path(configured))

    program_data = os.getenv("PROGRAMDATA")
    if program_data:
        paths.append(Path(program_data) / "Pic2Remnant" / "license_state.json")
    else:
        paths.append(Path.home() / ".pic2remnant" / "license_state.json")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def _state_signature(state: dict[str, Any]) -> str:
    body = {key: value for key, value in state.items() if key != "signature"}
    return hmac.new(_STATE_SECRET, _json_bytes(body), hashlib.sha256).hexdigest()


def _read_state(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise LicenseError("invalid_license_state") from exc
    if not isinstance(state, dict):
        raise LicenseError("invalid_license_state")
    signature = str(state.get("signature") or "")
    if not hmac.compare_digest(signature, _state_signature(state)):
        raise LicenseError("invalid_license_state")
    return state


def _load_last_seen() -> datetime | None:
    last_seen: datetime | None = None
    for path in _state_paths():
        state = _read_state(path)
        if not state:
            continue
        value = _parse_dt(str(state.get("last_seen_utc")), "last_seen_utc")
        if last_seen is None or value > last_seen:
            last_seen = value
    return last_seen


def _write_state(last_seen: datetime) -> None:
    state = {
        "version": 1,
        "last_seen_utc": _format_dt(last_seen),
    }
    state["signature"] = _state_signature(state)
    content = json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2)

    for path in _state_paths():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not write license state: path={}, error={}", path, exc)


def check_license() -> LicenseInfo:
    path = license_file_path()
    if not path.exists() or not path.is_file():
        raise LicenseError("license_file_missing")

    token = path.read_text(encoding="utf-8").strip()
    info = decode_license_key(token)
    now = _utc_now()

    if now.timestamp() + CLOCK_TOLERANCE_SECONDS < info.valid_from.timestamp():
        raise LicenseError("system_clock_invalid")
    if now > info.expires_at:
        raise LicenseError("expired")

    last_seen = _load_last_seen()
    if last_seen and now.timestamp() + CLOCK_TOLERANCE_SECONDS < last_seen.timestamp():
        raise LicenseError("system_clock_rollback")

    _write_state(max(now, last_seen) if last_seen else now)
    return info


class LicenseGate:
    def __init__(self) -> None:
        self._last_status = LicenseStatus(ok=False, code="not_checked", message="无法使用")

    def check(self) -> LicenseStatus:
        try:
            info = check_license()
        except LicenseError as exc:
            self._last_status = LicenseStatus(ok=False, code=exc.code, message=exc.message)
            logger.warning("License check failed: code={}, file={}", exc.code, license_file_path())
        except Exception as exc:
            self._last_status = LicenseStatus(ok=False, code="license_check_failed", message="无法使用")
            logger.exception("License check failed unexpectedly: {}", exc)
        else:
            self._last_status = LicenseStatus(ok=True, code="ok", message="ok", info=info)
            logger.debug("License check passed: license_id={}, expires_at={}", info.license_id, _format_dt(info.expires_at))
        return self._last_status

    @property
    def status(self) -> LicenseStatus:
        return self._last_status


license_gate = LicenseGate()
