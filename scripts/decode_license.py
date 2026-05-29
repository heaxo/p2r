from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.license import LicenseError, decode_license_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode and verify a Pic2Remnant license.key file.")
    parser.add_argument("file", nargs="?", default="license.key", help="License file path. Default: license.key")
    args = parser.parse_args()

    try:
        info = decode_license_key(Path(args.file).read_text(encoding="utf-8").strip())
    except (OSError, LicenseError) as exc:
        print(f"无法使用: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    data = info.public_dict()
    data["file"] = str(Path(args.file).resolve())
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
