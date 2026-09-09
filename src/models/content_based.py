"""مدل محتوامحور مبتنی بر TF-IDF.

این مدل به‌جای رفتار کاربران، به «محتوای» کتاب نگاه می‌کند: عنوان، نویسنده و
برچسب‌هایی که خوانندگان به آن زده‌اند. هر کتاب به یک بردار TF-IDF تبدیل
می‌شود و پروفایل سلیقه‌ی کاربر، میانگین وزن‌دار بردار کتاب‌هایی است که
پسندیده است.

چرا این لایه لازم است: فیلترینگ مشارکتی برای کتابی که هنوز کسی امتیازش
نداده هیچ خروجی ندارد، چون هیچ الگوی رفتاری درباره‌اش وجود ندارد. این همان
مسئله‌ی «شروع سرد» (cold-start) است. مدل محتوامحور این مشکل را ندارد، چون
عنوان و نویسنده‌ی کتاب از روز اول موجود است.

وزن‌دهی امتیازها: امتیاز خام ۱ تا ۵ مستقیماً به‌عنوان وزن استفاده نمی‌شود،
چون در آن صورت کتابی که کاربر ۱ داده هم با وزن مثبت وارد پروفایل می‌شود.
به‌جای آن، امتیاز نسبت به میانگین کاربر مرکزی می‌شود تا کتاب‌های نپسندیده
وزن منفی بگیرند و پروفایل را از خود دور کنند.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .. import config as cfg
from .base import BaseRecommender


class ContentBasedRecommender(BaseRecommender):
    name = "محتوامحور (TF-IDF)"

    def __init__(self, max_features: int = cfg.CONTENT_MAX_FEATURES):
        self.max_features = max_features

    @staticmethod
    def _corpus(books) -> list[str]:
        """ساخت متن توصیفی هر کتاب از عنوان، نویسنده و برچسب‌ها.

        نام نویسنده سه بار تکرار می‌شود تا وزن بیشتری از تک‌کلمه‌های عنوان
        بگیرد؛ در عمل «نویسنده» قوی‌ترین سیگنال محتوایی برای کتاب است.
        """
        title = books["title"].fillna("").astype(str)
        authors = books["authors"].fillna("").astype(str)
        tags = books["tag_text"].fillna("").astype(str)
        return (title + " " + (authors + " ") * 3 + tags).str.lower().tolist()

    def fit(self, ds):
        self.n_items_ = ds.n_items
        corpus = self._corpus(ds.books)

        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )
        self.item_vectors_ = normalize(self.vectorizer_.fit_transform(corpus))

        # میانگین امتیاز هر کاربر، برای مرکزی‌سازی وزن‌ها
        train = ds.train_matrix.tocsr()
        counts = np.diff(train.indptr)
        sums = np.asarray(train.sum(axis=1)).ravel()
        self.user_means_ = np.where(counts > 0, sums / np.maximum(counts, 1),
                                    ds.global_mean).astype(np.float32)
        self.train_ = train
        self.global_mean_ = ds.global_mean
        return self

    def _profile(self, user: int):
        """بردار پروفایل سلیقه‌ی کاربر در فضای TF-IDF."""
        row = self.train_[user]
        if row.nnz == 0:
            return None
        weights = row.data - self.user_means_[user]
        return normalize(csr_matrix(weights) @ self.item_vectors_[row.indices])

    def profile_from_items(self, item_ids, ratings=None):
        """ساخت پروفایل برای کاربر تازه‌وارد، فقط از روی چند کتاب پسندیده.

        این متد همان چیزی است که مسئله‌ی شروع سرد کاربر را حل می‌کند: بدون
        هیچ سابقه‌ای در ماتریس آموزش، از روی دو-سه انتخاب اولیه توصیه می‌سازد.
        """
        item_ids = np.asarray(item_ids, dtype=int)
        if len(item_ids) == 0:
            return None
        if ratings is None:
            weights = np.ones(len(item_ids), dtype=np.float32)
        else:
            weights = np.asarray(ratings, dtype=np.float32) - self.global_mean_
            if not np.any(np.abs(weights) > 1e-6):
                weights = np.ones(len(item_ids), dtype=np.float32)
        return normalize(csr_matrix(weights) @ self.item_vectors_[item_ids])

    def score_all_items(self, user):
        return self.score_from_profile(self._profile(user))

    def score_from_profile(self, profile) -> np.ndarray:
        """امتیازدهی به همه‌ی کتاب‌ها از روی یک بردار پروفایل.

        پروفایل به بردار چگال تبدیل می‌شود و سپس ضرب «ماتریس اسپارس در بردار
        چگال» انجام می‌گیرد. این کار چند برابر سریع‌تر از ضرب اسپارس-در-اسپارس
        است و چون طول بردار واژگان حدود ۲۳ هزار است، هزینه‌ی حافظه‌اش ناچیز.
        """
        if profile is None:
            return np.zeros(self.n_items_, dtype=np.float32)
        dense = np.asarray(profile.todense(), dtype=np.float32).ravel()
        return np.asarray(self.item_vectors_ @ dense, dtype=np.float32).ravel()

    def predict(self, users, items):
        """تبدیل شباهت محتوایی به مقیاس امتیاز ۱ تا ۵.

        شباهت کسینوسی در بازه‌ی [-۱, ۱] است و مستقیماً «امتیاز» نیست. برای
        اینکه این مدل هم در جدول RMSE قابل مقایسه باشد، شباهت حول میانگین
        کاربر نگاشت می‌شود. انتظار نداریم اینجا رقیب مدل‌های مشارکتی باشد؛
        قوت این مدل در رتبه‌بندی و پوشش است، نه در پیش‌بینی عددی.
        """
        similarity = self._predict_grouped(users, items, self.score_all_items)
        return np.clip(self.user_means_[np.asarray(users)] + 2.0 * similarity, 1.0, 5.0)
