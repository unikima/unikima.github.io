"""رابط مشترک همه‌ی مدل‌های توصیه‌گر.

هر مدل دو کار انجام می‌دهد:
  predict          → پیش‌بینی امتیاز عددی برای زوج‌های (کاربر، کتاب)
  score_all_items  → امتیازدهی به همه‌ی کتاب‌ها برای یک کاربر، جهت رتبه‌بندی

جدا کردن این دو عمداً انجام شده است: برخی مدل‌ها (مثل محتوامحور) در
پیش‌بینی امتیاز عددی ضعیف‌اند ولی در رتبه‌بندی خوب عمل می‌کنند، و
معیارهای ارزیابی این دو جنبه را مستقل می‌سنجند.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseRecommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, ds) -> "BaseRecommender":
        ...

    @abstractmethod
    def predict(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """پیش‌بینی امتیاز برای آرایه‌های هم‌طول کاربر و کتاب."""

    @abstractmethod
    def score_all_items(self, user: int) -> np.ndarray:
        """بردار امتیاز به طول تعداد کل کتاب‌ها برای یک کاربر."""

    def _predict_grouped(self, users, items, score_fn) -> np.ndarray:
        """پیش‌بینی دسته‌ای با گروه‌بندی بر اساس کاربر.

        مدل‌های همسایگی و محتوامحور برای هر کاربر یک بردار امتیاز کامل
        می‌سازند. اگر این کار به‌ازای هر سطر آزمون تکرار شود، برای یک میلیون
        سطر آزمون میلیون‌ها بار محاسبه‌ی تکراری انجام می‌شود. اینجا سطرها
        بر اساس کاربر مرتب و گروه‌بندی می‌شوند تا بردار امتیاز هر کاربر
        دقیقاً یک بار ساخته شود.
        """
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        out = np.empty(len(users), dtype=np.float32)

        order = np.argsort(users, kind="stable")
        sorted_users, sorted_items = users[order], items[order]

        boundaries = np.flatnonzero(np.diff(sorted_users)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(sorted_users)]))

        for start, end in zip(starts, ends):
            scores = score_fn(int(sorted_users[start]))
            out[order[start:end]] = scores[sorted_items[start:end]]
        return out

    def recommend(self, user: int, n: int = 10,
                  exclude_seen: bool = True, seen=None) -> np.ndarray:
        """برگرداندن اندیس n کتاب برتر برای یک کاربر."""
        scores = self.score_all_items(user).copy()
        if exclude_seen and seen is not None and len(seen) > 0:
            scores[seen] = -np.inf
        n = min(n, int(np.isfinite(scores).sum()))
        if n <= 0:
            return np.array([], dtype=int)
        top = np.argpartition(-scores, n - 1)[:n]
        return top[np.argsort(-scores[top])]
