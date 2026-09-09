"""مدل پایه: توصیه بر اساس محبوبیت.

این مدل شخصی‌سازی نمی‌کند و به همه‌ی کاربران یک فهرست یکسان می‌دهد.
هدف از آن، فراهم کردن یک خط مبنا (baseline) است: هر مدل شخصی‌سازی‌شده‌ای
که نتواند از این مدل بهتر عمل کند، عملاً ارزشی اضافه نکرده است.

نکته‌ی مهم پیاده‌سازی: به‌جای میانگین ساده‌ی امتیازها از «میانگین هموارشده‌ی
بیزی» استفاده شده است. میانگین ساده باعث می‌شود کتابی که فقط سه نفر آن را
خوانده و هر سه ۵ داده‌اند، بالاتر از کتابی بنشیند که هزار نفر خوانده‌اند و
میانگین ۴.۶ گرفته است. فرمول هموارسازی، میانگین هر کتاب را به نسبت تعداد
امتیازهایش به سمت میانگین کل جمعیت می‌کشد:

    score(i) = (n_i * mean_i + C * mu) / (n_i + C)

که در آن C ثابت هموارسازی و mu میانگین کل امتیازهاست.
"""
from __future__ import annotations

import numpy as np

from .. import config as cfg
from .base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    name = "محبوبیت (پایه)"

    def __init__(self, prior: float = cfg.POPULARITY_PRIOR):
        self.prior = prior

    def fit(self, ds):
        self.global_mean_ = ds.global_mean
        counts = np.asarray((ds.train_matrix != 0).sum(axis=0)).ravel()
        sums = np.asarray(ds.train_matrix.sum(axis=0)).ravel()

        with np.errstate(invalid="ignore", divide="ignore"):
            means = np.where(counts > 0, sums / np.maximum(counts, 1), self.global_mean_)

        self.item_scores_ = (
            (counts * means + self.prior * self.global_mean_) / (counts + self.prior)
        ).astype(np.float32)
        self.item_counts_ = counts
        return self

    def predict(self, users, items):
        return self.item_scores_[items]

    def score_all_items(self, user):
        """برای رتبه‌بندی، معیار «تعداد خواننده» است نه «میانگین امتیاز».

        این تفاوت مهم است و در ابتدا به‌سادگی از قلم می‌افتد. کتابی که ۴۰۰
        نفر خوانده‌اند و میانگین ۴.۸ گرفته، بالاترین میانگین هموارشده را
        دارد؛ ولی احتمال اینکه یک کاربر تصادفی آن را خوانده باشد بسیار کمتر
        از «هری پاتر» با ۱۷ هزار خواننده است. خط مبنای استاندارد در
        توصیه‌ی Top-N، پرخواننده‌ترین‌هاست (MostPopular) نه پرامتیازترین‌ها.
        """
        return self.item_counts_.astype(np.float32)
