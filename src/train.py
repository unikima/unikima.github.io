"""اسکریپت آموزش: همه‌ی مدل‌ها را می‌سازد، تنظیم می‌کند، ارزیابی و ذخیره می‌کند.

اجرا:  python -m src.train

ترتیب کار عمداً به این شکل است:
  ۱. داده به سه بخش آموزش / اعتبارسنجی / آزمون تقسیم می‌شود.
  ۲. مدل‌ها روی بخش آموزش یاد می‌گیرند.
  ۳. پارامترهای آزاد روی بخش اعتبارسنجی تنظیم می‌شوند.
  ۴. و تنها در پایان، یک‌بار، نتیجه روی بخش آزمون گزارش می‌شود.

این ترتیب تضمین می‌کند اعداد نهایی خوش‌بینانه نباشند: هیچ تصمیمی با نگاه
کردن به داده‌ی آزمون گرفته نشده است.

خروجی‌ها در پوشه‌ی artifacts/ و results/ ذخیره می‌شوند و رابط کاربری
Streamlit مستقیماً از همان‌ها می‌خواند.
"""
from __future__ import annotations

import pickle
import time

import pandas as pd

from . import config as cfg
from .data_loader import build_dataset, save_dataset
from .evaluate import cold_start_evaluation, evaluate_all
from .models import (ContentBasedRecommender, HybridRecommender,
                     ItemKNNRecommender, MatrixFactorizationRecommender,
                     PopularityRecommender, PureSVDRecommender)
from .tuning import best_of, sweep_alpha, sweep_factors


def main() -> None:
    t_start = time.time()

    print("=" * 62)
    print("مرحله ۱ — آماده‌سازی داده")
    print("=" * 62)
    ds = build_dataset()
    save_dataset(ds)

    print()
    print("=" * 62)
    print("مرحله ۲ — آموزش مدل‌ها")
    print("=" * 62)

    models = {}

    t = time.time()
    models["محبوبیت (پایه)"] = PopularityRecommender().fit(ds)
    print(f"مدل محبوبیت آموزش دید ({time.time() - t:.1f} ثانیه)")

    t = time.time()
    models["شباهت آیتم‌محور (KNN)"] = ItemKNNRecommender().fit(ds)
    print(f"مدل آیتم‌محور آموزش دید ({time.time() - t:.1f} ثانیه)")

    t = time.time()
    content = ContentBasedRecommender().fit(ds)
    models["محتوامحور (TF-IDF)"] = content
    print(f"مدل محتوامحور آموزش دید ({time.time() - t:.1f} ثانیه)")

    t = time.time()
    print("آموزش تجزیه ماتریس (پایش روی داده‌ی اعتبارسنجی):")
    mf = MatrixFactorizationRecommender().fit(ds, valid_df=ds.valid_df)
    models["تجزیه ماتریس (SVD)"] = mf
    print(f"مدل تجزیه ماتریس آموزش دید ({time.time() - t:.1f} ثانیه)")

    print()
    print("=" * 62)
    print("مرحله ۳ — تنظیم پارامترها روی داده‌ی اعتبارسنجی")
    print("=" * 62)

    print("تحلیل حساسیت PureSVD به تعداد عامل‌های پنهان:")
    factor_table = sweep_factors(ds)
    best_factors = int(best_of(factor_table, "n_factors", f"NDCG@{cfg.TOP_K}"))
    print(f"→ تعداد عامل انتخاب‌شده: {best_factors}")

    pure_svd = PureSVDRecommender(n_factors=best_factors).fit(ds)
    models["PureSVD"] = pure_svd

    print()
    print("جاروب وزن مدل ترکیبی:")
    alpha_table = sweep_alpha(pure_svd, content, mf, ds)
    best_alpha = float(best_of(alpha_table, "alpha", f"NDCG@{cfg.TOP_K}"))
    print(f"→ وزن انتخاب‌شده: alpha = {best_alpha}")

    hybrid = HybridRecommender(pure_svd, content, mf, alpha=best_alpha).fit(ds)
    models["ترکیبی (PureSVD + محتوا)"] = hybrid

    print()
    print("=" * 62)
    print("مرحله ۴ — ارزیابی نهایی روی داده‌ی آزمون")
    print("=" * 62)
    table = evaluate_all(models, ds, with_ci=True)
    print()

    display_columns = [c for c in table.columns if "CI_" not in c]
    print(table[display_columns].round(4).to_string())

    print()
    print(f"بازه اطمینان ۹۵٪ برای NDCG@{cfg.TOP_K}:")
    low_col, high_col = f"NDCG@{cfg.TOP_K} CI_low", f"NDCG@{cfg.TOP_K} CI_high"
    for name, row in table.iterrows():
        print(f"  {name:<26} {row[f'NDCG@{cfg.TOP_K}']:.4f}  "
              f"[{row[low_col]:.4f}, {row[high_col]:.4f}]")

    print()
    print("ارزیابی سناریوی شروع سرد (کاربر تازه‌وارد با ۳ کتاب) ...")
    cold = cold_start_evaluation(hybrid, ds)
    for key, value in cold.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # ---------------------------------------------------------- ذخیره‌سازی
    table.to_csv(cfg.RESULTS_DIR / "comparison.csv", encoding="utf-8-sig")
    alpha_table.to_csv(cfg.RESULTS_DIR / "alpha_sweep.csv", index=False,
                       encoding="utf-8-sig")
    factor_table.to_csv(cfg.RESULTS_DIR / "factor_sweep.csv", index=False,
                        encoding="utf-8-sig")
    pd.DataFrame([cold]).to_csv(cfg.RESULTS_DIR / "cold_start.csv",
                                index=False, encoding="utf-8-sig")
    pd.DataFrame(mf.history_).to_csv(cfg.RESULTS_DIR / "mf_training_curve.csv",
                                     index=False)
    pd.DataFrame([{"best_alpha": best_alpha,
                   "best_n_factors": best_factors}]).to_csv(
        cfg.RESULTS_DIR / "chosen_hyperparameters.csv", index=False)

    from .plots import make_all
    figures = make_all(table, pd.DataFrame(mf.history_), ds,
                       alpha_table=alpha_table, factor_table=factor_table)
    print()
    print("نمودارها ساخته شد:")
    for f in figures:
        print(f"  {f}")

    with open(cfg.ARTIFACT_DIR / "models.pkl", "wb") as fh:
        pickle.dump({"popularity": models["محبوبیت (پایه)"],
                     "item_knn": models["شباهت آیتم‌محور (KNN)"],
                     "content": content,
                     "mf": mf,
                     "pure_svd": pure_svd,
                     "hybrid": hybrid}, fh, protocol=4)

    print()
    print(f"پایان. زمان کل: {(time.time() - t_start) / 60:.1f} دقیقه")
    print(f"نتایج در {cfg.RESULTS_DIR} و مدل‌ها در {cfg.ARTIFACT_DIR} ذخیره شد.")


if __name__ == "__main__":
    main()
