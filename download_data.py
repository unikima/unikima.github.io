"""دانلود مجموعه‌داده goodbooks-10k.

این اسکریپت فقط یک‌بار لازم است اجرا شود. فایل‌ها در پوشه‌ی data/
ذخیره می‌شوند و اگر از قبل موجود باشند، دوباره دانلود نمی‌شوند.

منبع: https://github.com/zygmuntz/goodbooks-10k
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master"
FILES = ["books.csv", "ratings.csv", "book_tags.csv", "tags.csv"]
DATA_DIR = Path(__file__).resolve().parent / "data"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    for name in FILES:
        target = DATA_DIR / name
        if target.exists() and target.stat().st_size > 1000:
            print(f"{name} از قبل موجود است — رد شد.")
            continue
        print(f"در حال دانلود {name} ...", flush=True)
        try:
            urllib.request.urlretrieve(f"{BASE_URL}/{name}", target)
        except Exception as exc:                                # noqa: BLE001
            print(f"دانلود {name} ناموفق بود: {exc}", file=sys.stderr)
            print("می‌توانید فایل‌ها را دستی از آدرس زیر بگیرید و در پوشه‌ی "
                  "data/ بگذارید:", file=sys.stderr)
            print("https://github.com/zygmuntz/goodbooks-10k", file=sys.stderr)
            sys.exit(1)
        size_mb = target.stat().st_size / 1e6
        print(f"  {name} دانلود شد ({size_mb:.1f} مگابایت)")
    print("همه‌ی فایل‌ها آماده‌اند.")


if __name__ == "__main__":
    main()
