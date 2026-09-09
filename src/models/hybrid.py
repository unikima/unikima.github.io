"""مدل ترکیبی: تجزیه ماتریس + محتوامحور.

دو مدل قبلی نقاط ضعف مکمل هم دارند. تجزیه ماتریس در پیش‌بینی دقیق است ولی
برای کاربر یا کتاب تازه هیچ حرفی برای گفتن ندارد. مدل محتوامحور همیشه
خروجی دارد ولی چون فقط متن را می‌بیند، نمی‌تواند بفهمد «کسانی که این را
دوست داشتند، آن یکی را هم دوست داشتند».

ترکیب به‌صورت وزن‌دار انجام می‌شود:

    score = α · z(score_MF) + (1-α) · z(score_content)

نکته‌ی کلیدی، تابع z است. امتیاز تجزیه ماتریس در مقیاس ۱ تا ۵ است و شباهت
محتوایی در بازه‌ی ۰ تا ۱؛ جمع مستقیم آنها بی‌معناست و عملاً وزن را به مدلی
می‌دهد که مقیاس بزرگ‌تری دارد. به همین دلیل هر دو بردار پیش از ترکیب
استانداردسازی (z-score) می‌شوند تا میانگین صفر و انحراف معیار یک داشته باشند.

رفتار تطبیقی: اگر کاربر سابقه‌ای در ماتریس آموزش نداشته باشد، وزن α به‌طور
خودکار صفر می‌شود و مدل کاملاً به لایه‌ی محتوامحور تکیه می‌کند. این همان
سازوکاری است که سیستم را در برابر شروع سرد مقاوم می‌کند.
"""
from __future__ import annotations

import numpy as np

from .. import config as cfg
from .base import BaseRecommender


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    std = x.std()
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


class HybridRecommender(BaseRecommender):
    name = "ترکیبی (MF + محتوا)"

    def __init__(self, ranker, content, mf=None,
                 alpha: float = cfg.HYBRID_ALPHA,
                 cold_start_threshold: int = 5):
        """
        ranker  — موتور مشارکتیِ رتبه‌بندی (PureSVD)
        content — موتور محتوامحور
        mf      — مدل تجزیه ماتریس، فقط برای پیش‌بینی عددی امتیاز

        اینکه رتبه‌بندی و پیش‌بینی امتیاز به دو زیرمدل متفاوت سپرده شده،
        یک تصمیم آگاهانه است نه یک ناسازگاری: هر کدام در کاری که به آن
        سپرده شده بهترین عملکرد را دارد.
        """
        self.ranker = ranker
        self.content = content
        self.mf = mf if mf is not None else ranker
        self.alpha = alpha
        self.cold_start_threshold = cold_start_threshold

    def fit(self, ds):
        """هر دو زیرمدل از پیش آموزش دیده‌اند؛ اینجا فقط سابقه ثبت می‌شود."""
        self.train_ = ds.train_matrix.tocsr()
        self.user_activity_ = np.diff(self.train_.indptr)
        self.n_items_ = ds.n_items
        return self

    def _alpha_for(self, user: int) -> float:
        """وزن تطبیقی: هرچه سابقه‌ی کاربر کمتر، سهم لایه‌ی محتوایی بیشتر."""
        activity = self.user_activity_[user]
        if activity == 0:
            return 0.0
        if activity < self.cold_start_threshold:
            return self.alpha * activity / self.cold_start_threshold
        return self.alpha

    def score_all_items(self, user):
        a = self._alpha_for(user)
        content_scores = _zscore(self.content.score_all_items(user))
        if a == 0.0:
            return content_scores
        collaborative = _zscore(self.ranker.score_all_items(user))
        return a * collaborative + (1 - a) * content_scores

    def score_for_new_user(self, item_ids, ratings=None) -> np.ndarray:
        """توصیه برای کاربری که اصلاً در داده‌ی آموزش نیست.

        فقط لایه‌ی محتوامحور فعال است، چون هیچ بردار پنهانی برای این کاربر
        وجود ندارد. این مسیر همان چیزی است که در رابط کاربری برای کاربر
        تازه‌وارد اجرا می‌شود.
        """
        profile = self.content.profile_from_items(item_ids, ratings)
        scores = _zscore(self.content.score_from_profile(profile))
        # حذف کتاب‌هایی که کاربر همین حالا انتخابشان کرده
        scores[np.asarray(item_ids, dtype=int)] = -np.inf
        return scores

    def predict(self, users, items):
        """پیش‌بینی امتیاز عددی بر عهده‌ی زیرمدل تجزیه ماتریس است.

        ترکیب z-score برای رتبه‌بندی طراحی شده و خروجی‌اش مقیاس امتیاز ندارد،
        بنابراین برای معیار RMSE از پیش‌بینی مستقیم MF استفاده می‌شود.
        """
        return self.mf.predict(users, items)
