"""PureSVD — تجزیه‌ی مقادیر منفرد روی ماتریس کامل، برای رتبه‌بندی.

این مدل پاسخ پروژه به یک مشکل واقعی است که در ارزیابی آشکار شد.

مدل تجزیه ماتریس (FunkSVD) فقط روی خانه‌های پرشده‌ی ماتریس آموزش می‌بیند و
خانه‌های خالی را نادیده می‌گیرد. نتیجه این می‌شود که کمترین خطای پیش‌بینی را
دارد، ولی وقتی از آن می‌خواهیم ده کتاب برتر را نام ببرد، کتاب‌هایی را بالا
می‌آورد که فقط چند ده نفر خوانده‌اند و میانگین امتیاز بالایی دارند. مدل
هیچ‌وقت یاد نگرفته که «خوانده‌نشدن» خودش یک سیگنال است.

PureSVD دقیقاً همین را جبران می‌کند: ماتریس را کامل در نظر می‌گیرد و
خانه‌های خالی را صفر می‌گذارد، یعنی «این کاربر با این کتاب تعاملی نداشته».
سپس تجزیه‌ی مقادیر منفرد کوتاه‌شده روی همان ماتریس اجرا می‌شود:

    R ≈ U · Σ · Vᵀ

و امتیاز رتبه‌بندی هر کاربر از بازسازی سطر او به دست می‌آید. چون صفرها در
تابع هزینه شرکت می‌کنند، مدل به‌طور ضمنی محبوبیت را هم یاد می‌گیرد و
کتاب‌های بسیار کم‌خواننده را بالا نمی‌آورد.

نکته‌ی مهم برای دفاع: این ضعفِ FunkSVD نیست که «اشتباه» باشد؛ دو مدل دو
مسئله‌ی متفاوت را حل می‌کنند. پیش‌بینی امتیاز و رتبه‌بندی Top-N دو کار
جدا هستند و بهترین مدل برای یکی، لزوماً بهترین برای دیگری نیست. این
موضوع در ادبیات سیستم‌های توصیه‌گر شناخته‌شده است
(Cremonesi و همکاران، ۲۰۱۰).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import svds

from .. import config as cfg
from .base import BaseRecommender


class PureSVDRecommender(BaseRecommender):
    name = "PureSVD"

    def __init__(self, n_factors: int = cfg.PURE_SVD_FACTORS,
                 seed: int = cfg.RANDOM_SEED):
        self.n_factors = n_factors
        self.seed = seed

    def fit(self, ds):
        R = ds.train_matrix.astype(np.float32)
        rng = np.random.default_rng(self.seed)
        v0 = rng.normal(size=min(R.shape))

        # svds مقادیر منفرد را به ترتیب صعودی برمی‌گرداند؛ معکوس می‌کنیم
        U, sigma, Vt = svds(R, k=self.n_factors, v0=v0)
        order = np.argsort(-sigma)
        self.U_ = np.ascontiguousarray(U[:, order], dtype=np.float32)
        self.sigma_ = sigma[order].astype(np.float32)
        self.Vt_ = np.ascontiguousarray(Vt[order], dtype=np.float32)

        # سطر کاربر در فضای پنهان، از پیش ضرب‌شده در مقادیر منفرد
        self.user_factors_ = self.U_ * self.sigma_
        self.train_ = ds.train_matrix.tocsr()
        self.global_mean_ = ds.global_mean
        return self

    def score_all_items(self, user):
        return self.user_factors_[user] @ self.Vt_

    def predict(self, users, items):
        """بازسازی درایه‌های ماتریس.

        چون خانه‌های خالی صفر فرض شده‌اند، خروجی این مدل به سمت پایین
        اریب است و برای معیار RMSE رقیب FunkSVD نیست. عدد آن را گزارش
        می‌کنیم چون همین اریبی، نکته‌ی اصلی مقایسه است.
        """
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        pred = np.einsum("ij,ji->i", self.user_factors_[users], self.Vt_[:, items])
        return np.clip(pred, 1.0, 5.0)
