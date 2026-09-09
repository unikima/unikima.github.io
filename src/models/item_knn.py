"""فیلترینگ مشارکتی آیتم‌محور با شباهت کسینوسی.

ایده: دو کتاب شبیه‌اند اگر کاربران یکسانی به آنها امتیازهای مشابه داده باشند.
برای پیش‌بینی امتیاز کاربر u به کتاب i، به سراغ کتاب‌هایی می‌رویم که u قبلاً
امتیاز داده و به i شبیه‌اند، و میانگین وزن‌دار آن امتیازها را می‌گیریم.

سه تصمیم پیاده‌سازی که ارزش توضیح دادن دارند:

۱. مرکزی‌سازی بر اساس میانگین هر کتاب: قبل از محاسبه‌ی شباهت، میانگین امتیاز
   هر کتاب از آن کم می‌شود. بدون این کار، شباهت‌ها بیشتر بازتاب «محبوبیت
   عمومی» هستند تا «سلیقه‌ی مشترک».

۲. کوچک‌سازی (shrinkage): شباهتی که تنها بر پایه‌ی سه کاربر مشترک محاسبه شده
   قابل اعتماد نیست. ضریب  n/(n+λ)  شباهت‌های کم‌پشتوانه را به سمت صفر می‌برد.

۳. نگه‌داشتن فقط K همسایه‌ی برتر: ماتریس شباهت کامل ۱۰٬۰۰۰×۱۰٬۰۰۰ است؛ نگه
   داشتن آن هم پرحافظه است و هم همسایه‌های دور فقط نویز اضافه می‌کنند.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from .. import config as cfg
from .base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    name = "شباهت آیتم‌محور (KNN)"

    def __init__(self, k: int = cfg.ITEM_KNN_K,
                 shrinkage: float = cfg.ITEM_KNN_SHRINKAGE):
        self.k = k
        self.shrinkage = shrinkage

    def fit(self, ds):
        X = ds.train_matrix.tocsc()
        self.global_mean_ = ds.global_mean

        counts = np.diff(X.indptr).astype(np.float32)
        sums = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
        self.item_means_ = np.where(counts > 0, sums / np.maximum(counts, 1),
                                    self.global_mean_).astype(np.float32)

        # مرکزی‌سازی فقط روی درایه‌های موجود (نه روی صفرهای ساختاری)
        Xc = X.copy()
        for col in range(X.shape[1]):
            start, end = X.indptr[col], X.indptr[col + 1]
            Xc.data[start:end] -= self.item_means_[col]

        Xc = csr_matrix(Xc)
        Xc.data = np.nan_to_num(Xc.data)

        # نرمال‌سازی ستون‌ها تا ضرب داخلی مستقیماً کسینوس بدهد
        norms = np.sqrt(np.asarray(Xc.multiply(Xc).sum(axis=0))).ravel()
        norms[norms == 0] = 1.0
        Xn = (Xc @ diags(1.0 / norms)).tocsc().astype(np.float32)

        # تعداد کاربران مشترک بین هر جفت کتاب، برای اعمال کوچک‌سازی
        B = (X != 0).astype(np.float32).tocsc()

        n_items = X.shape[1]
        rows, cols, vals = [], [], []
        block = 512
        for start in range(0, n_items, block):
            end = min(start + block, n_items)
            sim = (Xn.T @ Xn[:, start:end]).toarray()          # (n_items, block)
            common = (B.T @ B[:, start:end]).toarray()         # کاربران مشترک
            sim *= common / (common + self.shrinkage)
            # حذف قطر اصلی: شباهت هر کتاب با خودش نباید در توصیه دخالت کند
            sim[np.arange(start, end), np.arange(end - start)] = 0.0

            kk = min(self.k, n_items - 1)
            top = np.argpartition(-sim, kk - 1, axis=0)[:kk]    # (kk, block)
            for c in range(end - start):
                idx = top[:, c]
                v = sim[idx, c]
                keep = v > 0
                rows.append(idx[keep])
                cols.append(np.full(keep.sum(), start + c, dtype=np.int32))
                vals.append(v[keep])

        self.sim_ = csr_matrix(
            (np.concatenate(vals).astype(np.float32),
             (np.concatenate(cols), np.concatenate(rows))),
            shape=(n_items, n_items),
        )
        self.sim_abs_ = abs(self.sim_)

        # نگه‌داری ماتریس آموزش برای دسترسی سریع به سابقه‌ی هر کاربر
        self.train_ = ds.train_matrix.tocsr()
        self.train_centered_ = csr_matrix(Xc)
        return self

    def _user_scores(self, user: int) -> np.ndarray:
        row = self.train_centered_[user]
        seen = self.train_[user].indices

        numerator = np.asarray(self.sim_ @ row.T.todense()).ravel()
        indicator = np.zeros(self.sim_.shape[0], dtype=np.float32)
        indicator[seen] = 1.0
        denominator = self.sim_abs_ @ indicator

        with np.errstate(invalid="ignore", divide="ignore"):
            deviation = np.where(denominator > 1e-8, numerator / denominator, 0.0)
        return self.item_means_ + deviation

    def explain(self, user: int, item: int, top_n: int = 3):
        """چرا این کتاب به این کاربر پیشنهاد شد.

        امتیاز هر کتاب پیشنهادی، جمع سهم تک‌تک کتاب‌هایی است که کاربر قبلاً
        خوانده: sim(i, j) × (امتیاز کاربر به j منهای میانگین j). پس برای
        توضیح دادن پیشنهاد، کافی است بزرگ‌ترین سهم‌ها را پیدا کنیم — نیازی
        به هیچ مدل جداگانه‌ای برای تفسیر نیست.

        این ویژگی مزیت اصلی روش همسایگی نسبت به تجزیه ماتریس است: در
        تجزیه ماتریس، امتیاز از ضرب دو بردار پنهان می‌آید و هیچ کتاب مشخصی
        را نمی‌توان به‌عنوان دلیل نام برد.

        خروجی: فهرستی از (اندیس کتاب، سهم) به ترتیب نزولی سهم.
        """
        neighbours = self.sim_[item]
        if neighbours.nnz == 0:
            return []

        history = self.train_centered_[user]
        if history.nnz == 0:
            return []

        shared = np.intersect1d(neighbours.indices, history.indices)
        if len(shared) == 0:
            return []

        similarity = np.asarray(neighbours[:, shared].todense()).ravel()
        deviation = np.asarray(history[:, shared].todense()).ravel()
        contribution = similarity * deviation

        order = np.argsort(-contribution)[:top_n]
        return [(int(shared[k]), float(contribution[k]))
                for k in order if contribution[k] > 0]

    def predict(self, users, items):
        out = self._predict_grouped(users, items, self._user_scores)
        return np.clip(out, 1.0, 5.0)

    def score_all_items(self, user):
        """امتیاز رتبه‌بندی: جمع وزن‌دار شباهت‌ها، بدون نرمال‌سازی.

        برای پیش‌بینی امتیاز، تقسیم بر مجموع شباهت‌ها لازم است تا خروجی در
        مقیاس ۱ تا ۵ بماند. ولی برای رتبه‌بندی همین تقسیم مضر است: کتابی که
        فقط با یکی از کتاب‌های کاربر شباهت دارد، پس از نرمال‌سازی همان‌قدر
        امتیاز می‌گیرد که کتابی که با بیست کتاب او مرتبط است. حذف مخرج باعث
        می‌شود «شواهد بیشتر» به رتبه‌ی بالاتر ترجمه شود، که همان رفتار
        استاندارد ItemKNN در توصیه‌ی Top-N است.
        """
        row = self.train_centered_[user]
        return np.asarray(self.sim_ @ row.T.todense()).ravel()
