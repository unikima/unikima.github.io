"""چارچوب ارزیابی مدل‌های توصیه‌گر.

سیستم توصیه‌گر را نمی‌توان با یک عدد سنجید، چون دو کار متفاوت انجام می‌دهد:

۱. پیش‌بینی امتیاز — «کاربر به این کتاب چند می‌دهد؟»
   معیار: RMSE و MAE. هرچه کمتر بهتر.

۲. رتبه‌بندی — «کدام ده کتاب را نشان بدهیم؟»
   معیار: Precision@K، Recall@K و NDCG@K. هرچه بیشتر بهتر.

این دو همیشه هم‌جهت نیستند: مدلی می‌تواند RMSE عالی داشته باشد ولی فهرست
ده‌تایی بی‌ربطی بسازد. در عمل آنچه کاربر می‌بیند فهرست است، نه عدد پیش‌بینی،
بنابراین معیارهای گروه دوم اهمیت عملی بیشتری دارند.

معیار سوم، پوشش (Coverage)، از جنس دیگری است: چه سهمی از کل کتاب‌های موجود
اصلاً شانس دیده شدن دارند. مدل محبوبیت معمولاً RMSE بدی ندارد ولی پوشش
فاجعه‌باری دارد، چون همیشه همان چند کتاب پرفروش را به همه پیشنهاد می‌دهد.
گزارش کردن پوشش در کنار دقت، این ضعف را آشکار می‌کند.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


# ---------------------------------------------------------------- خطای پیش‌بینی

def rating_metrics(model, test_df: pd.DataFrame) -> dict[str, float]:
    """محاسبه‌ی RMSE و MAE روی امتیازهای مجموعه‌ی آزمون."""
    users = test_df["u"].values
    items = test_df["i"].values
    truth = test_df["rating"].values.astype(np.float32)

    pred = np.asarray(model.predict(users, items), dtype=np.float32)
    error = truth - pred
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
    }


# ------------------------------------------------------------------- رتبه‌بندی

def _dcg(relevances: np.ndarray) -> float:
    """بهره‌ی تجمعی تنزیل‌یافته: هرچه آیتم مرتبط پایین‌تر باشد، ارزش کمتری دارد."""
    if len(relevances) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(relevances) + 2))
    return float(np.sum(relevances * discounts))


def ranking_metrics_per_user(model, ds, k: int = cfg.TOP_K,
                             n_users: int = cfg.EVAL_USER_SAMPLE,
                             seed: int = cfg.RANDOM_SEED,
                             eval_df: pd.DataFrame | None = None) -> dict:
    """همان معیارهای رتبه‌بندی، ولی مقدار هر کاربر جداگانه برگردانده می‌شود.

    داشتن مقادیر تک‌تک کاربران لازم است تا بتوان بازه‌ی اطمینان را با
    بازنمونه‌گیری بوت‌استرپ محاسبه کرد. میانگین به‌تنهایی نمی‌گوید که
    اختلاف دو مدل واقعی است یا اثر تصادفیِ انتخاب نمونه‌ی کاربران.
    """
    if eval_df is None:
        eval_df = ds.test_df

    rng = np.random.default_rng(seed)
    train = ds.train_matrix.tocsr()

    relevant_by_user: dict[int, set] = {}
    hits = eval_df[eval_df["rating"] >= cfg.RELEVANCE_THRESHOLD]
    for u, group in hits.groupby("u")["i"]:
        relevant_by_user[int(u)] = set(group.tolist())

    candidates = np.array(sorted(relevant_by_user.keys()))
    if len(candidates) > n_users:
        candidates = rng.choice(candidates, size=n_users, replace=False)

    precisions, recalls, ndcgs = [], [], []
    recommended_items: set[int] = set()

    for u in candidates:
        u = int(u)
        relevant = relevant_by_user[u]
        seen = train[u].indices
        top = model.recommend(u, n=k, exclude_seen=True, seen=seen)
        if len(top) == 0:
            continue

        recommended_items.update(top.tolist())
        gains = np.array([1.0 if int(x) in relevant else 0.0 for x in top])

        precisions.append(gains.sum() / k)
        recalls.append(gains.sum() / len(relevant))

        ideal = np.ones(min(len(relevant), k))
        idcg = _dcg(ideal)
        ndcgs.append(_dcg(gains) / idcg if idcg > 0 else 0.0)

    return {
        "precision": np.asarray(precisions),
        "recall": np.asarray(recalls),
        "ndcg": np.asarray(ndcgs),
        "coverage": len(recommended_items) / ds.n_items,
    }


def bootstrap_ci(values: np.ndarray, n_boot: int = cfg.N_BOOTSTRAP,
                 alpha: float = 0.05,
                 seed: int = cfg.RANDOM_SEED) -> tuple[float, float]:
    """بازه‌ی اطمینان با بازنمونه‌گیری بوت‌استرپ روی کاربران.

    روش: از میان کاربران ارزیابی‌شده، با جایگذاری نمونه‌ای هم‌اندازه
    برداشته و میانگین آن محاسبه می‌شود؛ این کار هزار بار تکرار می‌شود و
    صدک‌های ۲.۵ و ۹۷.۵ توزیع حاصل، بازه‌ی اطمینان ۹۵ درصدی را می‌دهند.

    این روش هیچ فرضی درباره‌ی توزیع داده نمی‌گذارد — که مهم است، چون
    توزیع Precision@10 به‌شدت غیرنرمال است (بسیاری از کاربران دقیقاً صفر
    می‌گیرند) و فرمول‌های مبتنی بر توزیع نرمال اینجا معتبر نیستند.
    """
    if len(values) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lower, upper


def ranking_metrics(model, ds, k: int = cfg.TOP_K,
                    n_users: int = cfg.EVAL_USER_SAMPLE,
                    seed: int = cfg.RANDOM_SEED,
                    eval_df=None, with_ci: bool = False) -> dict[str, float]:
    """معیارهای رتبه‌بندی روی نمونه‌ای از کاربران.

    «مرتبط» یعنی کاربر در مجموعه‌ی ارزیابی به آن کتاب امتیاز ≥ آستانه داده
    است. کتاب‌هایی که کاربر در مجموعه‌ی آموزش دیده از فهرست حذف می‌شوند، چون
    پیشنهاد کتابی که کاربر قبلاً خوانده ارزشی ندارد.

    ارزیابی روی نمونه انجام می‌شود نه کل کاربران: محاسبه‌ی فهرست ده‌تایی برای
    هر ۵۳ هزار کاربر در شش مدل زمان‌بر است. دقیقاً به همین دلیل، گزارش بازه‌ی
    اطمینان اهمیت دارد — با with_ci=True مشخص می‌شود عدد به‌دست‌آمده چقدر به
    انتخاب تصادفی نمونه حساس است.
    """
    per_user = ranking_metrics_per_user(model, ds, k=k, n_users=n_users,
                                        seed=seed, eval_df=eval_df)

    result = {
        f"Precision@{k}": float(per_user["precision"].mean()) if len(per_user["precision"]) else 0.0,
        f"Recall@{k}": float(per_user["recall"].mean()) if len(per_user["recall"]) else 0.0,
        f"NDCG@{k}": float(per_user["ndcg"].mean()) if len(per_user["ndcg"]) else 0.0,
        "Coverage": per_user["coverage"],
    }

    if with_ci:
        for label, key in ((f"Precision@{k}", "precision"),
                           (f"NDCG@{k}", "ndcg")):
            low, high = bootstrap_ci(per_user[key], seed=seed)
            result[f"{label} CI_low"] = low
            result[f"{label} CI_high"] = high
    return result


# ------------------------------------------------------------ ارزیابی سرد

def cold_start_evaluation(hybrid, ds, n_users: int = 500,
                          n_seed_items: int = 3, k: int = cfg.TOP_K,
                          seed: int = cfg.RANDOM_SEED) -> dict[str, float]:
    """سنجش عملکرد سیستم برای کاربر تازه‌وارد.

    شبیه‌سازی: از هر کاربر فقط سه کتاب پسندیده‌اش را برمی‌داریم، وانمود
    می‌کنیم هیچ سابقه‌ی دیگری ندارد، و می‌سنجیم چقدر از کتاب‌های مرتبطِ
    مجموعه‌ی آزمون را پیدا می‌کند. این سناریو دقیقاً همان چیزی است که در
    استفاده‌ی واقعی، برای کاربری که تازه ثبت‌نام کرده اتفاق می‌افتد.
    """
    rng = np.random.default_rng(seed)
    train = ds.train_matrix.tocsr()

    liked = ds.test_df[ds.test_df["rating"] >= cfg.RELEVANCE_THRESHOLD]
    relevant_by_user = {int(u): set(g.tolist())
                        for u, g in liked.groupby("u")["i"]}

    eligible = [u for u in relevant_by_user
                if len(train[u].indices) >= n_seed_items]
    if len(eligible) > n_users:
        eligible = rng.choice(eligible, size=n_users, replace=False)

    precisions, ndcgs = [], []
    for u in eligible:
        u = int(u)
        history = train[u]
        best = np.argsort(-history.data)[:n_seed_items]
        seed_items = history.indices[best]
        seed_ratings = history.data[best]

        scores = hybrid.score_for_new_user(seed_items, seed_ratings)
        scores[history.indices] = -np.inf          # حذف کل سابقه‌ی آموزش
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        relevant = relevant_by_user[u]
        gains = np.array([1.0 if int(x) in relevant else 0.0 for x in top])
        precisions.append(gains.sum() / k)
        idcg = _dcg(np.ones(min(len(relevant), k)))
        ndcgs.append(_dcg(gains) / idcg if idcg > 0 else 0.0)

    return {
        f"ColdStart Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"ColdStart NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_tested": len(precisions),
    }


# ------------------------------------------------------------------- گزارش

def evaluate_all(models: dict, ds, k: int = cfg.TOP_K,
                 verbose: bool = True, with_ci: bool = False) -> pd.DataFrame:
    """اجرای همه‌ی معیارها روی همه‌ی مدل‌ها و ساخت جدول مقایسه.

    این تابع فقط یک‌بار و در انتهای کار روی داده‌ی آزمون اجرا می‌شود؛
    تنظیم پارامترها پیش‌تر و روی داده‌ی اعتبارسنجی انجام شده است.
    """
    rows = []
    for name, model in models.items():
        if verbose:
            print(f"ارزیابی: {name} ...", flush=True)
        row = {"مدل": name}
        row.update(rating_metrics(model, ds.test_df))
        row.update(ranking_metrics(model, ds, k=k, eval_df=ds.test_df,
                                   with_ci=with_ci))
        rows.append(row)

    df = pd.DataFrame(rows)
    return df.set_index("مدل")
