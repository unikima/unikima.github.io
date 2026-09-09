"""تنظیم پارامترها روی مجموعه‌ی اعتبارسنجی.

دو پارامتر در این پروژه دلخواه انتخاب شده بودند و همین، ضعف قابل‌حمله‌ای
در دفاع بود: وزن مدل ترکیبی (آلفا) و تعداد عامل‌های پنهان PureSVD.

نکته‌ی مهم روش‌شناختی: هر دو جاروب روی **مجموعه‌ی اعتبارسنجی** انجام
می‌شوند، نه مجموعه‌ی آزمون. اگر مقدار بهینه را روی داده‌ی آزمون پیدا کنیم و
بعد نتیجه‌ی نهایی را هم روی همان گزارش کنیم، عدد گزارش‌شده خوش‌بینانه است —
چون پارامتر را طوری تنظیم کرده‌ایم که روی همان داده خوب جواب بدهد. این
همان اشتباهی است که تفکیک سه‌بخشی داده برای جلوگیری از آن انجام شد.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from .evaluate import ranking_metrics
from .models import HybridRecommender, PureSVDRecommender


def sweep_alpha(ranker, content, mf, ds,
                grid: list[float] | None = None,
                k: int = cfg.TOP_K,
                n_users: int = cfg.TUNING_USER_SAMPLE,
                verbose: bool = True) -> pd.DataFrame:
    """یافتن وزن بهینه‌ی ترکیب، روی داده‌ی اعتبارسنجی.

    آلفا برابر ۱ یعنی فقط مدل مشارکتی و آلفا برابر ۰ یعنی فقط مدل
    محتوامحور. مقادیر میانی، ترکیب وزن‌دار این دو هستند. نمودار حاصل
    نشان می‌دهد آیا ترکیب واقعاً از هر دو جزء خود بهتر است یا نه.
    """
    grid = grid or cfg.ALPHA_GRID
    rows = []
    for alpha in grid:
        hybrid = HybridRecommender(ranker, content, mf, alpha=alpha).fit(ds)
        scores = ranking_metrics(hybrid, ds, k=k, n_users=n_users,
                                 eval_df=ds.valid_df)
        rows.append({
            "alpha": alpha,
            f"Precision@{k}": scores[f"Precision@{k}"],
            f"NDCG@{k}": scores[f"NDCG@{k}"],
            "Coverage": scores["Coverage"],
        })
        if verbose:
            print(f"  alpha={alpha:.2f}  NDCG@{k}={rows[-1][f'NDCG@{k}']:.4f}  "
                  f"Coverage={rows[-1]['Coverage']:.3f}", flush=True)
    return pd.DataFrame(rows)


def sweep_factors(ds, grid: list[int] | None = None,
                  k: int = cfg.TOP_K,
                  n_users: int = cfg.TUNING_USER_SAMPLE,
                  verbose: bool = True) -> pd.DataFrame:
    """تحلیل حساسیت نسبت به تعداد عامل‌های پنهان PureSVD.

    این نمودار به یک سؤال رایج داوران جواب می‌دهد: «چرا این تعداد بُعد؟»
    اگر منحنی در محدوده‌ی وسیعی صاف باشد، یعنی نتیجه به این انتخاب حساس
    نیست و می‌توان با اطمینان گفت عدد انتخاب‌شده بحرانی نبوده است.
    """
    grid = grid or cfg.FACTOR_GRID
    rows = []
    for n_factors in grid:
        model = PureSVDRecommender(n_factors=n_factors).fit(ds)
        scores = ranking_metrics(model, ds, k=k, n_users=n_users,
                                 eval_df=ds.valid_df)
        rows.append({
            "n_factors": n_factors,
            f"Precision@{k}": scores[f"Precision@{k}"],
            f"NDCG@{k}": scores[f"NDCG@{k}"],
            "Coverage": scores["Coverage"],
        })
        if verbose:
            print(f"  factors={n_factors:>4}  NDCG@{k}={rows[-1][f'NDCG@{k}']:.4f}",
                  flush=True)
    return pd.DataFrame(rows)


def best_of(table: pd.DataFrame, column: str, metric: str):
    """برگرداندن مقدار پارامتری که بیشترین مقدار معیار را داده است."""
    return table.loc[table[metric].idxmax(), column]
