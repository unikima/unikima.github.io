"""استخراج مدل‌های آموزش‌دیده به قالب قابل استفاده در مرورگر.

چرا این مرحله لازم است: مدل‌های آموزش‌دیده اشیای پایتونی‌اند و جاوااسکریپت
نمی‌تواند آن‌ها را بخواند. اما پس از آموزش، هرچه لازم داریم چند ماتریس عددی
است. این اسکریپت آن ماتریس‌ها را به فایل‌های دودویی فشرده تبدیل می‌کند تا
رابط وب بتواند بدون هیچ سرور پایتونی، محاسبه‌ی توصیه را در خود مرورگر
انجام دهد.

خروجی‌ها در web/public/model/ ذخیره می‌شوند:

  books.json        فراداده‌ی کتاب‌ها
  meta.json         ابعاد و پارامترها
  svd_v.bin         ماتریس عامل‌های PureSVD           float32[n_items × k]
  item_sim_idx.bin  اندیس همسایه‌های آیتم‌محور        uint16[n_items × K]
  item_sim_val.bin  مقدار شباهت آیتم‌محور             float32[n_items × K]
  cont_sim_idx.bin  اندیس همسایه‌های محتوایی          uint16[n_items × K]
  cont_sim_val.bin  مقدار شباهت محتوایی               float32[n_items × K]
  popularity.bin    تعداد خواننده هر کتاب             float32[n_items]
  results.json      جدول‌های ارزیابی برای صفحه نتایج

اجرا:  python -m src.export_web
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from . import config as cfg
from .data_loader import load_dataset

WEB_MODEL_DIR = cfg.ROOT / "web" / "public" / "model"
TOP_K_NEIGHBOURS = 40
TOP_GENRES = 5

# برچسب‌های Goodreads ترکیبی از ژانر واقعی و برچسب‌های مدیریت قفسه‌ی شخصی
# هستند. برچسب‌هایی مانند to-read یا owned هیچ اطلاعاتی درباره‌ی محتوای کتاب
# نمی‌دهند و فقط وضعیت خواندن کاربر را نشان می‌دهند، بنابراین پیش از نمایش
# حذف می‌شوند. توجه: این پالایش فقط برای نمایش است؛ مدل محتوامحور روی متن
# کامل برچسب‌ها آموزش دیده است.
SHELF_TAGS = {
    "to-read", "currently-reading", "favorites", "favourites", "owned",
    "owned-books", "books-i-own", "i-own", "own", "my-books", "my-library",
    "default", "kindle", "library", "audiobook", "audiobooks", "audio",
    "ebook", "ebooks", "e-books", "to-buy", "wish-list", "wishlist", "unread",
    "re-read", "reread", "read-more-than-once", "all-time-favorites",
    "shelfari-favorites", "calibre", "english", "books", "book", "read",
    "reads", "favorite-books", "faves", "my-favorites", "my-favourites",
    "bookshelf", "general", "stuff", "misc", "want-to-read", "have",
    "finished", "completed", "abandoned", "dnf", "tbr", "on-hold",
    "purchased", "bought", "gift", "borrowed", "book-club", "series",
    "5-stars", "4-stars", "3-stars", "novels", "novel", "fiction-books",
    "paper-books", "hardcover", "paperback", "physical-books", "home-library",
    "adult", "books-read", "read-in-2016", "read-in-2015", "recommended",
}


def display_genres(tag_names: list[str], authors: str) -> list[str]:
    """انتخاب برچسب‌های محتوایی و آماده‌سازی آن‌ها برای نمایش."""
    surnames = {
        part.strip().split()[-1].lower()
        for part in str(authors).split(",") if part.strip()
    }
    picked: list[str] = []
    for tag in tag_names:
        key = str(tag).strip().lower()
        if not key or key in SHELF_TAGS:
            continue
        # برچسب‌هایی که فقط نام نویسنده‌اند اطلاعات تازه‌ای اضافه نمی‌کنند
        if key.replace("-", " ") in surnames or key in surnames:
            continue
        pretty = key.replace("-", " ").strip().title()
        if pretty not in picked:
            picked.append(pretty)
        if len(picked) == TOP_GENRES:
            break
    return picked


def _top_k_neighbours(similarity_block_fn, n_items: int, k: int,
                      block: int = 512):
    """استخراج k همسایه برتر هر آیتم به‌صورت بلوکی.

    ماتریس کامل شباهت ۱۰٬۰۰۰ در ۱۰٬۰۰۰ است که هم پرحافظه است و هم برای
    ارسال به مرورگر بسیار بزرگ. تنها k همسایه‌ی برتر نگه داشته می‌شود که
    حجم را به چند مگابایت کاهش می‌دهد بدون آنکه کیفیت توصیه افت محسوسی کند.
    """
    idx_out = np.zeros((n_items, k), dtype=np.uint16)
    val_out = np.zeros((n_items, k), dtype=np.float32)

    for start in range(0, n_items, block):
        end = min(start + block, n_items)
        sim = similarity_block_fn(start, end)          # (n_items, end-start)
        sim[np.arange(start, end), np.arange(end - start)] = 0.0

        top = np.argpartition(-sim, k - 1, axis=0)[:k]
        for c in range(end - start):
            order = top[:, c]
            values = sim[order, c]
            rank = np.argsort(-values)
            idx_out[start + c] = order[rank]
            val_out[start + c] = np.maximum(values[rank], 0.0)
    return idx_out, val_out


def main() -> None:
    WEB_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("بارگذاری داده و مدل‌ها ...")
    ds = load_dataset()
    with open(cfg.ARTIFACT_DIR / "models.pkl", "rb") as fh:
        models = pickle.load(fh)

    pure_svd = models["pure_svd"]
    item_knn = models["item_knn"]
    content = models["content"]
    hybrid = models["hybrid"]

    n_items = ds.n_items

    # ---------------------------------------------------------- فراداده کتاب‌ها
    books = ds.books

    # حدود یک‌سوم کتاب‌ها در Goodreads جلد ندارند و نشانی تصویرشان به یک
    # تصویر جایگزین عمومی اشاره می‌کند. ISBN را هم صادر می‌کنیم تا رابط وب
    # بتواند برای این کتاب‌ها جلد را از منبع دیگری (Open Library) بگیرد.
    # شناسه‌ها به‌صورت رشته خوانده می‌شوند تا صفرهای ابتدایی حفظ شود.
    #
    # فقط ستون isbn استفاده می‌شود. ستون isbn13 در فایل اصلی به‌صورت نماد
    # علمی ذخیره شده (مثلاً 9.78043902348e+12) و رقم آخرش را از دست داده،
    # بنابراین قابل اتکا نیست. شناسه‌ی ده‌رقمی سالم است و Open Library هم
    # همان را می‌پذیرد.
    raw = pd.read_csv(cfg.DATA_DIR / "books.csv", dtype={"isbn": str})
    isbn_by_book = raw.set_index("book_id")["isbn"].to_dict()

    # برچسب‌های هر کتاب به‌ترتیب فراوانی، برای نمایش در پیش‌نمایش
    book_tags = pd.read_csv(cfg.DATA_DIR / "book_tags.csv")
    tags = pd.read_csv(cfg.DATA_DIR / "tags.csv")
    tagged = (book_tags.merge(tags, on="tag_id", how="left")
              .sort_values("count", ascending=False))
    tags_by_gid = tagged.groupby("goodreads_book_id")["tag_name"].apply(list)

    popularity_counts = models["popularity"].item_counts_

    def clean_isbn(value):
        if not isinstance(value, str):
            return ""
        digits = value.strip().upper()
        return digits.rjust(10, "0") if digits else ""

    payload = []
    missing_cover = 0
    for i in range(n_items):
        row = books.loc[i]
        url = str(row["image_url"]) if pd.notna(row["image_url"]) else ""
        if "nophoto" in url:
            url = ""                      # جلد واقعی وجود ندارد
            missing_cover += 1

        payload.append({
            "t": str(row["title"]),
            "a": str(row["authors"]),
            "img": url,
            "isbn": clean_isbn(isbn_by_book.get(int(row["book_id"]))),
            "g": display_genres(
                tags_by_gid.get(int(row["goodreads_book_id"]), []),
                row["authors"]),
            "y": int(row["original_publication_year"])
            if pd.notna(row["original_publication_year"]) else None,
            "r": round(float(row["average_rating"]), 2),
            # تعداد کاربران این مجموعه‌داده که به کتاب امتیاز داده‌اند
            "n": int(popularity_counts[i]),
        })
    print(f"  کتاب‌های بدون جلد در Goodreads: {missing_cover:,}")
    with_genre = sum(1 for x in payload if x["g"])
    print(f"  کتاب‌های دارای برچسب ژانر: {with_genre:,}")
    (WEB_MODEL_DIR / "books.json").write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    print(f"  books.json — {len(payload):,} کتاب")

    # ------------------------------------------------- ماتریس عامل‌های PureSVD
    # V با ابعاد (n_items × k) ذخیره می‌شود. برای کاربر تازه، بردار انتخاب او
    # در V ضرب می‌شود تا بردار پنهان به دست آید و سپس در ترانهاده‌ی V ضرب
    # می‌شود؛ این همان «تازاندن» (fold-in) استاندارد PureSVD است.
    V = np.ascontiguousarray(pure_svd.Vt_.T, dtype=np.float32)   # (n_items, k)
    V.tofile(WEB_MODEL_DIR / "svd_v.bin")
    print(f"  svd_v.bin — {V.shape} ({V.nbytes / 1e6:.1f} مگابایت)")

    # ------------------------------------------------ همسایه‌های آیتم‌محور
    # sim_ به‌صورت سطری ذخیره شده: سطر i همسایه‌های کتاب i را نگه می‌دارد.
    # برای گرفتن بلوکی به شکل (n_items × block) ترانهاده می‌شود.
    sim_csr = item_knn.sim_.tocsr()

    def knn_block(start, end):
        return np.asarray(sim_csr[start:end].T.todense(), dtype=np.float32)

    print("استخراج همسایه‌های آیتم‌محور ...")
    knn_idx, knn_val = _top_k_neighbours(knn_block, n_items, TOP_K_NEIGHBOURS)
    knn_idx.tofile(WEB_MODEL_DIR / "item_sim_idx.bin")
    knn_val.tofile(WEB_MODEL_DIR / "item_sim_val.bin")
    print(f"  item_sim — {knn_idx.shape} "
          f"({(knn_idx.nbytes + knn_val.nbytes) / 1e6:.1f} مگابایت)")

    # -------------------------------------------------- همسایه‌های محتوایی
    print("استخراج همسایه‌های محتوایی ...")
    # سطرهای item_vectors_ بردار هر کتاب‌اند و از پیش نرمال شده‌اند،
    # بنابراین ضرب داخلی دو سطر مستقیماً شباهت کسینوسی می‌دهد.
    Vc = normalize(content.item_vectors_).tocsr().astype(np.float32)

    def content_block(start, end):
        return np.asarray((Vc @ Vc[start:end].T).todense(), dtype=np.float32)

    cont_idx, cont_val = _top_k_neighbours(content_block, n_items,
                                           TOP_K_NEIGHBOURS)
    cont_idx.tofile(WEB_MODEL_DIR / "cont_sim_idx.bin")
    cont_val.tofile(WEB_MODEL_DIR / "cont_sim_val.bin")
    print(f"  cont_sim — {cont_idx.shape} "
          f"({(cont_idx.nbytes + cont_val.nbytes) / 1e6:.1f} مگابایت)")

    # ------------------------------------------------------------- محبوبیت
    popularity = models["popularity"].item_counts_.astype(np.float32)
    popularity.tofile(WEB_MODEL_DIR / "popularity.bin")

    # ------------------------------------------------------------- فراداده
    meta = {
        "nItems": int(n_items),
        "nUsers": int(ds.n_users),
        "nRatings": int(len(ds.train_df) + len(ds.valid_df) + len(ds.test_df)),
        "svdFactors": int(V.shape[1]),
        "topK": TOP_K_NEIGHBOURS,
        "alpha": float(hybrid.alpha),
        "globalMean": round(float(ds.global_mean), 4),
    }
    (WEB_MODEL_DIR / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("  meta.json —", meta)

    # -------------------------------------------------------- نتایج ارزیابی
    comparison = pd.read_csv(cfg.RESULTS_DIR / "comparison.csv", index_col=0)
    results = {
        "comparison": [
            {"model": name, **{k: round(float(v), 4) for k, v in row.items()}}
            for name, row in comparison.iterrows()
        ],
        "alphaSweep": pd.read_csv(cfg.RESULTS_DIR / "alpha_sweep.csv")
        .round(4).to_dict(orient="records"),
        "factorSweep": pd.read_csv(cfg.RESULTS_DIR / "factor_sweep.csv")
        .round(4).to_dict(orient="records"),
        "coldStart": pd.read_csv(cfg.RESULTS_DIR / "cold_start.csv")
        .round(4).to_dict(orient="records")[0],
        "trainingCurve": pd.read_csv(cfg.RESULTS_DIR / "mf_training_curve.csv")
        .round(4).to_dict(orient="records"),
    }
    (WEB_MODEL_DIR / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8")
    print("  results.json")

    total = sum(f.stat().st_size for f in WEB_MODEL_DIR.iterdir())
    print(f"\nمجموع حجم خروجی: {total / 1e6:.1f} مگابایت")
    print(f"مسیر: {WEB_MODEL_DIR}")


if __name__ == "__main__":
    main()
