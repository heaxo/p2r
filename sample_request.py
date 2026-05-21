from __future__ import annotations

import requests


def main() -> None:
    url = "http://127.0.0.1:8000/measure"
    token = "change-me"
    image_path = "test.jpg"

    with open(image_path, "rb") as f:
        resp = requests.post(
            url,
            headers={"X-Auth-Token": token},
            files={"image": (image_path, f, "image/jpeg")},
            data={
                "a4_orientation": "auto",
                "paper_source": "yolo",
                "paper_rect_mode": "approx_poly",
                "simplify_mm": "3.0",
            },
            timeout=600,
        )

    print(resp.status_code)
    print(resp.text)


if __name__ == "__main__":
    main()
