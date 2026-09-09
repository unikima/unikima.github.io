"""تجزیه ماتریس با بایاس (FunkSVD) — هسته‌ی مشارکتی سیستم.

هر کاربر و هر کتاب با یک بردار پنهان k-بُعدی نمایش داده می‌شود. این بردارها
مفاهیمی را می‌آموزند که در داده صریح نیستند — مثلاً «گرایش به ادبیات کلاسیک»
یا «علاقه به داستان‌های پلیسی» — بی‌آنکه کسی این مفاهیم را به مدل گفته باشد.

فرمول پیش‌بینی:

    r̂(u,i) = μ + b_u + b_i + p_u · q_i

سه جمله‌ی اول بایاس‌اند و بخش مهمی از تغییرپذیری داده را توضیح می‌دهند:
μ میانگین کلی، b_u سخت‌گیری یا سخاوت کاربر در امتیازدهی، و b_i کیفیت ذاتی
کتاب. تنها چیزی که به ضرب داخلی بردارها سپرده می‌شود، «تعامل سلیقه‌ای»
باقی‌مانده است.

آموزش با گرادیان کاهشی تصادفی روی درایه‌های مشاهده‌شده انجام می‌شود. توجه
داشته باشید که برخلاف SVD کلاسیک جبر خطی، اینجا درایه‌های خالی ماتریس صفر
فرض نمی‌شوند؛ آنها صرفاً در تابع هزینه شرکت نمی‌کنند. این تفاوت، دلیل اصلی
برتری این روش بر تجزیه‌ی مستقیم ماتریس اسپارس است.
"""
from __future__ import annotations

import numpy as np

from .. import config as cfg
from .base import BaseRecommender


class MatrixFactorizationRecommender(BaseRecommender):
    name = "تجزیه ماتریس (SVD)"

    def __init__(self, n_factors: int = cfg.MF_FACTORS,
                 n_epochs: int = cfg.MF_EPOCHS,
                 lr: float = cfg.MF_LR,
                 reg: float = cfg.MF_REG,
                 batch_size: int = cfg.MF_BATCH,
                 seed: int = cfg.RANDOM_SEED,
                 verbose: bool = True):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.verbose = verbose
        self.history_: list[dict] = []

    def fit(self, ds, valid_df=None):
        rng = np.random.default_rng(self.seed)
        u = ds.train_df["u"].values.astype(np.int32)
        i = ds.train_df["i"].values.astype(np.int32)
        y = ds.train_df["rating"].values.astype(np.float32)

        n_users, n_items = ds.n_users, ds.n_items
        self.mu_ = float(y.mean())
        self.bu_ = np.zeros(n_users, dtype=np.float32)
        self.bi_ = np.zeros(n_items, dtype=np.float32)
        scale = 0.05
        self.P_ = rng.normal(0, scale, (n_users, self.n_factors)).astype(np.float32)
        self.Q_ = rng.normal(0, scale, (n_items, self.n_factors)).astype(np.float32)

        n = len(y)
        for epoch in range(self.n_epochs):
            order = rng.permutation(n)
            # کاهش تدریجی نرخ یادگیری برای پایداری در دوره‌های پایانی
            lr = self.lr * (0.92 ** epoch)

            for start in range(0, n, self.batch_size):
                b = order[start:start + self.batch_size]
                ub, ib, yb = u[b], i[b], y[b]

                pu, qi = self.P_[ub], self.Q_[ib]
                pred = self.mu_ + self.bu_[ub] + self.bi_[ib] + np.einsum("ij,ij->i", pu, qi)
                err = (yb - np.clip(pred, -20, 20)).astype(np.float32)

                np.add.at(self.bu_, ub, lr * (err - self.reg * self.bu_[ub]))
                np.add.at(self.bi_, ib, lr * (err - self.reg * self.bi_[ib]))
                np.add.at(self.P_, ub, lr * (err[:, None] * qi - self.reg * pu))
                np.add.at(self.Q_, ib, lr * (err[:, None] * pu - self.reg * qi))

            if self.verbose:
                train_rmse = self._rmse(u, i, y)
                entry = {"epoch": epoch + 1, "train_rmse": train_rmse}
                if valid_df is not None:
                    entry["valid_rmse"] = self._rmse(
                        valid_df["u"].values, valid_df["i"].values,
                        valid_df["rating"].values)
                self.history_.append(entry)
                msg = f"  epoch {epoch + 1:>2}/{self.n_epochs}  train RMSE={train_rmse:.4f}"
                if valid_df is not None:
                    msg += f"  valid RMSE={entry['valid_rmse']:.4f}"
                print(msg)
        return self

    def _rmse(self, u, i, y) -> float:
        pred = self.predict(u, i)
        return float(np.sqrt(np.mean((y - pred) ** 2)))

    def predict(self, users, items):
        users = np.asarray(users, dtype=np.int32)
        items = np.asarray(items, dtype=np.int32)
        pred = (self.mu_ + self.bu_[users] + self.bi_[items]
                + np.einsum("ij,ij->i", self.P_[users], self.Q_[items]))
        return np.clip(pred, 1.0, 5.0)

    def score_all_items(self, user):
        return self.mu_ + self.bu_[user] + self.bi_ + self.Q_ @ self.P_[user]
