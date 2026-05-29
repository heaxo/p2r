from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.license import LICENSE_TOKEN_PREFIX, _SHA256_DIGEST_INFO_PREFIX

PRIVATE_RSA_N = int(
    "76aa821f8a7c0cf462674b85b0cc952817b232570a12b63bd2f6fb1b2d061d2970c069614dfe5249f4307e40e6bcce20c4540f1d1f68e73e0b3ca5ed52c7b4f11cc405cdca3867cb2be4c29f70c8690b8e4c25cd6be3a26a0f6176a214136b52995fd9bc0f03f09afaafc69ebb1720d3061bde41238bfcc2faa32b3608d178deee70ef39abd2149c13b9ef00d1bdf35e28c95fe8d57019da149809694b759c9cb2f431a59bc14b34cf485f469313d9dca9c7d177fe367ede8bc2f71f69ee9e45dc5c9e167bd37b2d8bccb65ce8612d69f807c99ae81ec70115782c6188f7078d3240e94400e6ad77f003def0ff141a878086215f9467c2dc76e922de6956d7c3",
    16,
)
PRIVATE_RSA_D = int(
    "2f0d58df2cc81752e799d1a646fe37be57f6fe62e8c9969c8e00047d38130e8919fe114ad5e41bb8db2c474370ba3879619f7b06af27a759409761ab8281514bb17e4056e2f20ea5ced7bc1c005a56dc9025bd6f8072183a39b3099926a0b49361e151b271b9bc3397670f386d32b47f3812e0c3af569878af6cf77b4b6fd39ef0e11215192baceee5135b6900ca79b78c55208a39e493690384b2ed651649ad01a8b37a3028328ec2b7d15efb281e697abb25c495ce8d52deb4e9067e4a87e394aa2f3fd92261a83b8d6b4cd632e24fcacc53d8a25af908999960f579e901b7f76552f6aeda46fb248e236b1470fd338313065ccbc09b79652498f0a3f8dcb1",
    16,
)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _format_dt(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_start(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _sign(payload_bytes: bytes) -> bytes:
    key_bytes = (PRIVATE_RSA_N.bit_length() + 7) // 8
    digest_info = _SHA256_DIGEST_INFO_PREFIX + hashlib.sha256(payload_bytes).digest()
    padding_len = key_bytes - len(digest_info) - 3
    if padding_len < 8:
        raise RuntimeError("RSA key is too short")
    encoded = b"\x00\x01" + (b"\xff" * padding_len) + b"\x00" + digest_info
    signature = pow(int.from_bytes(encoded, "big"), PRIVATE_RSA_D, PRIVATE_RSA_N)
    return signature.to_bytes(key_bytes, "big")


def generate_license(days: int, customer: str | None, start: datetime) -> str:
    if days <= 0:
        raise ValueError("days must be greater than 0")

    payload: dict[str, Any] = {
        "version": 1,
        "license_id": uuid.uuid4().hex,
        "days": days,
        "issued_at": _format_dt(datetime.now(timezone.utc)),
        "valid_from": _format_dt(start),
        "expires_at": _format_dt(start + timedelta(days=days)),
    }
    if customer:
        payload["customer"] = customer

    payload_bytes = _json_bytes(payload)
    return f"{LICENSE_TOKEN_PREFIX}.{_b64url(payload_bytes)}.{_b64url(_sign(payload_bytes))}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Pic2Remnant license.key file.")
    parser.add_argument("--days", type=int, required=True, help="Allowed usage days.")
    parser.add_argument("--out", default="license.key", help="Output file path. Default: license.key")
    parser.add_argument("--customer", default=None, help="Optional customer note stored in the key.")
    parser.add_argument(
        "--start",
        default=None,
        help="Optional UTC start time, for example 2026-05-29T00:00:00Z. Default: now.",
    )
    args = parser.parse_args()

    token = generate_license(days=args.days, customer=args.customer, start=_parse_start(args.start))
    out = Path(args.out)
    out.write_text(token + "\n", encoding="utf-8")
    print(str(out.resolve()))


if __name__ == "__main__":
    main()
