"""
بارگذاری، پاک‌سازی و تقسیم داده‌های goodbooks-10k.

خروجی اصلی این ماژول یک شیء Dataset است که شامل ماتریس اسپارس
کاربر×کتاب برای آموزش، امتیازهای آزمون، و فراداده‌ی کتاب‌هاست.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from . import config as cfg


@dataclass
class Dataset:
    """نگه‌دارنده‌ی همه‌ی داده‌های آماده‌ی مدل‌سازی."""

    train_matrix: csr_matrix          # ماتریس اسپارس کاربر × کتاب (امتیاز خام)
    train_df: pd.DataFrame            # ستون‌های u, i, rating (اندیس‌های داخلی)
    valid_df: pd.DataFrame            # اعتبارسنجی — تنظیم پارامتر و پایش آموزش
    test_df: pd.DataFrame             # آزمون — فقط یک‌بار، برای گزارش نهایی
    books: pd.DataFrame               # فراداده‌ی کتاب‌ها، ایندکس‌شده با i
    user_index: dict[int, int]        # user_id اصلی → اندیس داخلی
    item_index: dict[int, int]        # book_id اصلی → اندیس داخلی
    item_ids: np.ndarray              # اندیس داخلی → book_id اصلی

    @property
    def n_users(self) -> int:
        return self.train_matrix.shape[0]

    @property
    def n_items(self) -> int:
        return self.train_matrix.shape[1]

    @property
    def global_mean(self) -> float:
        return float(self.train_df["rating"].mean())


def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """خواندن فایل‌های خام و ساخت ستون برچسب‌های متنی برای هر کتاب."""
    ratings = pd.read_csv(cfg.DATA_DIR / "ratings.csv")
    books = pd.read_csv(cfg.DATA_DIR / "books.csv")

    # برچسب‌های goodreads برای استفاده در مدل محتوامحور
    book_tags = pd.read_csv(cfg.DATA_DIR / "book_tags.csv")
    tags = pd.read_csv(cfg.DATA_DIR / "tags.csv")
    book_tags = book_tags.merge(tags, on="tag_id", how="left")

    # برای هر کتاب، پرتکرارترین برچسب‌ها را نگه می‌داریم
    book_tags = book_tags.sort_values("count", ascending=False)
    top_tags = (
        book_tags.groupby("goodreads_book_id")["tag_name"]
        .apply(lambda s: " ".join(s.head(cfg.CONTENT_TOP_TAGS).astype(str)))
        .rename("tag_text")
    )
    books = books.merge(top_tags, left_on="goodreads_book_id",
                        right_index=True, how="left")
    books["tag_text"] = books["tag_text"].fillna("")
    return ratings, books, book_tags


def _clean(ratings: pd.DataFrame) -> pd.DataFrame:
    """حذف رکوردهای تکراری و کاربران/کتاب‌های کم‌تعامل.

    فیلتر به‌صورت تکرارشونده اعمال می‌شود، چون حذف کاربران کم‌تعامل ممکن است
    باعث شود کتابی که قبلاً واجد شرایط بود از حد نصاب بیفتد و بالعکس.
    """
    ratings = ratings.drop_duplicates(subset=["user_id", "book_id"])
    ratings = ratings[ratings["rating"].between(1, 5)]

    while True:
        n_before = len(ratings)
        user_counts = ratings["user_id"].value_counts()
        keep_u = user_counts[user_counts >= cfg.MIN_RATINGS_PER_USER].index
        ratings = ratings[ratings["user_id"].isin(keep_u)]

        item_counts = ratings["book_id"].value_counts()
        keep_i = item_counts[item_counts >= cfg.MIN_RATINGS_PER_BOOK].index
        ratings = ratings[ratings["book_id"].isin(keep_i)]

        if len(ratings) == n_before:
            break
    return ratings


def _split(ratings: pd.DataFrame,
           seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """تقسیم سه‌بخشی لایه‌بندی‌شده بر اساس کاربر.

    از هر کاربر سهم مشخصی از امتیازهایش کنار گذاشته می‌شود، نه اینکه کل
    داده به‌صورت تصادفی تقسیم شود. این کار تضمین می‌کند هر کاربرِ مجموعه‌ی
    ارزیابی، در مجموعه‌ی آموزش هم سابقه دارد — یعنی مسئله‌ی سرد بودن کاربر
    به‌طور مصنوعی وارد ارزیابی نمی‌شود.

    چرا سه بخش و نه دو بخش: هر تصمیمی که با نگاه کردن به داده گرفته شود —
    انتخاب وزن مدل ترکیبی، تعداد دوره‌های آموزش، تعداد عامل‌های پنهان —
    آن داده را «مصرف» می‌کند و دیگر نمی‌شود از آن برای گزارش نتیجه‌ی نهایی
    استفاده کرد. پس یک بخش اعتبارسنجی برای تصمیم‌گیری کنار گذاشته می‌شود و
    بخش آزمون تا لحظه‌ی گزارش نهایی دست‌نخورده می‌ماند.
    """
    shuffled = ratings.sample(frac=1.0, random_state=seed)
    position = shuffled.groupby("user_id").cumcount()
    per_user_total = shuffled.groupby("user_id")["user_id"].transform("size")

    n_test = np.maximum(1, np.floor(per_user_total * cfg.TEST_FRACTION)).astype(int)
    n_valid = np.maximum(1, np.floor(per_user_total * cfg.VALID_FRACTION)).astype(int)

    is_test = position < n_test
    is_valid = (position >= n_test) & (position < n_test + n_valid)

    train = shuffled[~(is_test | is_valid)].copy()
    valid = shuffled[is_valid].copy()
    test = shuffled[is_test].copy()
    return train, valid, test


def build_dataset(verbose: bool = True) -> Dataset:
    """خط لوله‌ی کامل: خواندن → پاک‌سازی → تقسیم → ساخت ماتریس اسپارس."""
    ratings, books, _ = _load_raw()
    if verbose:
        print(f"امتیازهای خام: {len(ratings):,}")

    ratings = _clean(ratings)
    if verbose:
        print(f"پس از پاک‌سازی: {len(ratings):,} "
              f"({ratings.user_id.nunique():,} کاربر، "
              f"{ratings.book_id.nunique():,} کتاب)")

    train, valid, test = _split(ratings, cfg.RANDOM_SEED)

    # نگاشت شناسه‌های اصلی به اندیس‌های پیوسته‌ی صفر-مبنا
    user_ids = np.sort(ratings["user_id"].unique())
    item_ids = np.sort(ratings["book_id"].unique())
    user_index = {uid: k for k, uid in enumerate(user_ids)}
    item_index = {iid: k for k, iid in enumerate(item_ids)}

    for frame in (train, valid, test):
        frame["u"] = frame["user_id"].map(user_index).astype(np.int32)
        frame["i"] = frame["book_id"].map(item_index).astype(np.int32)
        frame["rating"] = frame["rating"].astype(np.float32)

    train_matrix = csr_matrix(
        (train["rating"].values, (train["u"].values, train["i"].values)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )

    books = books[books["book_id"].isin(item_index)].copy()
    books["i"] = books["book_id"].map(item_index)
    books = books.sort_values("i").set_index("i")

    if verbose:
        density = 100 * train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])
        print(f"آموزش: {len(train):,} | اعتبارسنجی: {len(valid):,} | "
              f"آزمون: {len(test):,} | چگالی ماتریس: {density:.2f}%")

    return Dataset(
        train_matrix=train_matrix,
        train_df=train[["u", "i", "rating"]].reset_index(drop=True),
        valid_df=valid[["u", "i", "rating"]].reset_index(drop=True),
        test_df=test[["u", "i", "rating"]].reset_index(drop=True),
        books=books,
        user_index=user_index,
        item_index=item_index,
        item_ids=item_ids,
    )


def save_dataset(ds: Dataset, path=None) -> None:
    path = path or (cfg.ARTIFACT_DIR / "dataset.pkl")
    with open(path, "wb") as fh:
        pickle.dump(ds, fh, protocol=4)


def load_dataset(path=None) -> Dataset:
    path = path or (cfg.ARTIFACT_DIR / "dataset.pkl")
    with open(path, "rb") as fh:
        return pickle.load(fh)
