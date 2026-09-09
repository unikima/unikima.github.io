"""تولید نمودارهای گزارش.

برچسب نمودارها فارسی است. رسم فارسی در matplotlib سه چیز لازم دارد و
هر سه اینجا فراهم شده است:
  ۱. یک فونت فارسی واقعی (وزیرمتن، در پوشه‌ی assets).
  ۲. شکل‌دهی حروف: در فارسی هر حرف بسته به جایگاهش در کلمه شکل متفاوتی
     دارد و matplotlib این کار را انجام نمی‌دهد؛ arabic_reshaper می‌کند.
  ۳. الگوریتم دوجهته: بدون آن، متن راست‌به‌چپ وارونه رسم می‌شود.

اگر فونت یا این دو کتابخانه در دسترس نباشند، ماژول به‌جای خطا دادن به
برچسب‌های انگلیسی برمی‌گردد تا اجرای آموزش هرگز به‌خاطر نمودار نشکند.

انتخاب‌های بصری بر پایه‌ی چند قاعده‌ی ثابت است:
  • هر نمودار فقط یک محور مقداری دارد (هرگز دو محور y).
  • وقتی یک کمیت را بین چند مدل مقایسه می‌کنیم، یک مجموعه‌ی داده داریم،
    پس یک رنگ استفاده می‌شود؛ رنگ‌های متفاوت وقتی معنا دارند که هویت‌های
    متفاوتی را نشان دهند.
  • خطوط راهنما کم‌رنگ و پس‌زمینه ساده است تا داده برجسته بماند.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as cfg

# پالت: آبی و نارنجی به‌عنوان دو رنگ هویتی، خاکستری برای اجزای فرعی
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

# ------------------------------------------------------- آماده‌سازی فارسی
_FONT = cfg.ROOT / "assets" / "Vazirmatn.ttf"
PERSIAN = False

try:  # pragma: no cover - وابسته به محیط
    import arabic_reshaper
    from bidi.algorithm import get_display
    from matplotlib import font_manager

    if _FONT.exists():
        font_manager.fontManager.addfont(str(_FONT))
        plt.rcParams["font.family"] = "Vazirmatn"
        PERSIAN = True
except Exception:  # فونت یا کتابخانه نبود؛ به انگلیسی برمی‌گردیم
    PERSIAN = False

_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def fa(text: str) -> str:
    """متن فارسی آماده‌ی رسم: حروف به‌هم‌چسبیده و ترتیب راست‌به‌چپ."""
    if not PERSIAN:
        return text
    return get_display(arabic_reshaper.reshape(str(text)))


def num(value: str) -> str:
    """ارقام لاتین را به فارسی تبدیل می‌کند تا با متن گزارش هم‌خوان باشد."""
    return str(value).translate(_DIGITS) if PERSIAN else str(value)


def _fa_ticks(ax, axis: str = "both") -> None:
    """ارقام روی محورها را فارسی می‌کند."""
    if not PERSIAN:
        return
    if axis in ("x", "both"):
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels([num(t.get_text()) for t in ax.get_xticklabels()])
    if axis in ("y", "both"):
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels([num(t.get_text()) for t in ax.get_yticklabels()])


# نام کوتاه مدل‌ها برای محور نمودارها. نام کامل جدول‌ها برای برچسب محور
# بلند است و ستون نمودار را می‌بلعد.
SHORT_NAMES = {
    "محبوبیت (پایه)": "محبوبیت (پایه)",
    "شباهت آیتم‌محور (KNN)": "شباهت آیتم‌محور",
    "محتوامحور (TF-IDF)": "محتوامحور",
    "تجزیه ماتریس (SVD)": "تجزیه ماتریس",
    "PureSVD": "PureSVD",
    "ترکیبی (PureSVD + محتوا)": "ترکیبی",
}


def _label(name: str) -> str:
    short = SHORT_NAMES.get(name, name)
    # نام‌های لاتین مثل PureSVD نباید از الگوریتم دوجهته رد شوند
    return short if short.isascii() else fa(short)


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def sweep_curve(table: pd.DataFrame, x_column: str, metric: str,
                filename: str, x_label: str, title: str,
                log_x: bool = False) -> str:
    """منحنی یک معیار برحسب یک پارامتر، با علامت‌گذاری نقطه‌ی بهینه.

    یک سری داده داریم، پس یک رنگ استفاده می‌شود. نقطه‌ی بیشینه با یک
    نشانگر بزرگ‌تر و برچسب مستقیم مشخص می‌شود تا خواننده بدون خواندن محور،
    مقدار انتخاب‌شده و دلیلش را ببیند.
    """
    x = table[x_column].values
    y = table[metric].values
    best = int(np.argmax(y))

    fig, ax = plt.subplots(figsize=(6.8, 3.5), dpi=160)
    ax.plot(x, y, color=BLUE, linewidth=2, marker="o", markersize=5, zorder=3)
    ax.plot(x[best], y[best], marker="o", markersize=11, color=BLUE,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
    # این برچسب عمداً از get_display رد نمی‌شود: هر دو سطرش با یک نام
    # لاتین شروع می‌شوند و ارقام فارسی هم در جهت چپ‌به‌راست درست می‌نشینند.
    ax.annotate(f"{x_column} = {num(f'{x[best]:g}')}\n"
                f"{metric} = {num(f'{y[best]:.4f}')}",
                xy=(x[best], y[best]), xytext=(0, -34),
                textcoords="offset points", ha="center",
                fontsize=8.5, color=INK)

    if log_x:
        ax.set_xscale("log")
        ax.set_xticks(x, [num(int(v)) for v in x])
    span = y.max() - y.min()
    ax.set_ylim(y.min() - span * 0.45, y.max() + span * 0.18)
    ax.set_xlabel(fa(x_label), fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.set_title(fa(title), fontsize=11, pad=12, loc="right")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _fa_ticks(ax, "y" if log_x else "both")
    _style(ax)

    path = cfg.RESULTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def metric_bars_with_ci(table: pd.DataFrame, metric: str,
                        filename: str) -> str:
    """میله‌های افقی همراه با بازه‌ی اطمینان ۹۵ درصد.

    میله‌ی خطا مهم‌ترین بخش این شکل است: اگر بازه‌های دو مدل هم‌پوشانی
    داشته باشند، نمی‌توان ادعا کرد یکی قطعاً بهتر از دیگری است. بدون این
    میله‌ها، جدول نتایج دقتی را وانمود می‌کند که واقعاً وجود ندارد.
    """
    low_col, high_col = f"{metric} CI_low", f"{metric} CI_high"
    values = table[metric]
    names = [_label(n) for n in values.index]

    order = np.argsort(values.values)[::-1]
    ordered_names = [names[k] for k in order]
    mean = values.values[order]
    low = table[low_col].values[order]
    high = table[high_col].values[order]

    fig, ax = plt.subplots(figsize=(7.4, 3.6), dpi=160)
    y = np.arange(len(mean))[::-1]
    colors = [BLUE if k == 0 else "#9ec5f4" for k in range(len(mean))]

    ax.barh(y, mean, height=0.55, color=colors, zorder=3)
    ax.errorbar(mean, y, xerr=[mean - low, high - mean], fmt="none",
                ecolor=INK, elinewidth=1.3, capsize=4, zorder=4)

    for pos, m, h in zip(y, mean, high):
        ax.text(h + high.max() * 0.02, pos, num(f"{m:.4f}"), va="center",
                ha="left", fontsize=9, color=INK)

    ax.set_yticks(y, ordered_names, fontsize=9)
    ax.set_xlim(0, high.max() * 1.22)
    ax.set_title(f"{metric} " + fa("با بازه اطمینان ۹۵ درصد — بیشتر بهتر"),
                 fontsize=11, pad=12, loc="right")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _fa_ticks(ax, "x")
    _style(ax)

    path = cfg.RESULTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def metric_bars(table: pd.DataFrame, metric: str, lower_is_better: bool,
                filename: str) -> str:
    """نمودار میله‌ای افقی مقایسه‌ی یک معیار بین مدل‌ها.

    میله‌ی افقی انتخاب شده چون نام مدل‌ها بلند است؛ در حالت عمودی برچسب‌ها
    یا کج می‌شوند یا روی هم می‌افتند. بهترین مدل با رنگ پررنگ‌تر و برچسب
    مستقیم مشخص می‌شود تا خواننده بدون مراجعه به محور، نتیجه را ببیند.
    """
    values = table[metric]
    names = [_label(n) for n in values.index]
    order = np.argsort(values.values)
    if not lower_is_better:
        order = order[::-1]

    ordered_names = [names[k] for k in order]
    ordered_values = values.values[order]

    fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
    y = np.arange(len(ordered_values))[::-1]

    colors = [BLUE if k == 0 else "#9ec5f4" for k in range(len(ordered_values))]
    ax.barh(y, ordered_values, height=0.58, color=colors, zorder=3)

    for pos, value in zip(y, ordered_values):
        ax.text(value + max(ordered_values) * 0.015, pos, num(f"{value:.4f}"),
                va="center", ha="left", fontsize=9, color=INK)

    ax.set_yticks(y, ordered_names, fontsize=9)
    ax.set_xlim(0, max(ordered_values) * 1.18)
    direction = "کمتر بهتر" if lower_is_better else "بیشتر بهتر"
    ax.set_title(f"{metric} — " + fa(direction),
                 fontsize=11, pad=12, loc="right")
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _fa_ticks(ax, "x")
    _style(ax)

    path = cfg.RESULTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def training_curve(history: pd.DataFrame, filename: str = "mf_curve.png") -> str:
    """منحنی همگرایی مدل تجزیه ماتریس.

    این نمودار دو کار می‌کند: نشان می‌دهد آموزش همگرا شده، و نشان می‌دهد
    فاصله‌ی خطای آموزش و آزمون کم مانده — یعنی مدل بیش‌برازش نکرده است.
    همین شکل، پاسخ آماده‌ای است به سؤال «از کجا می‌دانید overfit نشده؟»
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=160)
    epochs = history["epoch"]

    ax.plot(epochs, history["train_rmse"], color=BLUE, linewidth=2,
            marker="o", markersize=4, label=fa("خطای آموزش"), zorder=3)
    if "valid_rmse" in history:
        ax.plot(epochs, history["valid_rmse"], color=ORANGE, linewidth=2,
                marker="s", markersize=4, label=fa("خطای اعتبارسنجی"),
                zorder=3)
        final = history["valid_rmse"].iloc[-1]
        ax.annotate(num(f"{final:.4f}"),
                    xy=(epochs.iloc[-1], final),
                    xytext=(-6, 10), textcoords="offset points",
                    fontsize=9, color=INK, ha="right")

    ax.set_xlabel(fa("دوره آموزش"), fontsize=9)
    ax.set_ylabel("RMSE", fontsize=9)
    ax.set_title(fa("همگرایی مدل تجزیه ماتریس"),
                 fontsize=11, pad=12, loc="right")
    ax.set_xticks(epochs[::2], [num(int(e)) for e in epochs[::2]])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    _fa_ticks(ax, "y")
    _style(ax)

    path = cfg.RESULTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def rating_distribution(ds, filename: str = "rating_distribution.png") -> str:
    """توزیع امتیازها در دیتاست — شکل مقدماتی فصل «داده‌ها»."""
    counts = ds.train_df["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=160)
    ax.bar(counts.index, counts.values / 1e6, width=0.6, color=BLUE, zorder=3)
    for x, v in zip(counts.index, counts.values / 1e6):
        ax.text(x, v + 0.03, num(f"{v:.2f}"), ha="center", fontsize=8.5,
                color=INK)
    ax.set_xlabel(fa("امتیاز"), fontsize=9)
    ax.set_ylabel(fa("تعداد (میلیون)"), fontsize=9)
    ax.set_title(fa("توزیع امتیازها در مجموعه‌ی آموزش"),
                 fontsize=11, pad=12, loc="right")
    ax.set_ylim(0, (counts.values / 1e6).max() * 1.15)
    ax.set_xticks(list(counts.index), [num(int(v)) for v in counts.index])
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    _fa_ticks(ax, "y")
    _style(ax)

    path = cfg.RESULTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def make_all(table: pd.DataFrame, history: pd.DataFrame, ds,
             alpha_table: pd.DataFrame | None = None,
             factor_table: pd.DataFrame | None = None) -> list[str]:
    top_k = cfg.TOP_K
    outputs = [
        metric_bars(table, "RMSE", True, "fig_rmse.png"),
        metric_bars(table, f"Precision@{top_k}", False, "fig_precision.png"),
        metric_bars(table, f"NDCG@{top_k}", False, "fig_ndcg.png"),
        metric_bars(table, "Coverage", False, "fig_coverage.png"),
        training_curve(history),
        rating_distribution(ds),
    ]
    if f"NDCG@{top_k} CI_low" in table.columns:
        outputs.append(metric_bars_with_ci(table, f"NDCG@{top_k}",
                                           "fig_ndcg_ci.png"))
    if alpha_table is not None:
        outputs.append(sweep_curve(
            alpha_table, "alpha", f"NDCG@{top_k}", "fig_alpha_sweep.png",
            "وزن ترکیب آلفا  (یک = فقط مشارکتی، صفر = فقط محتوامحور)",
            "جاروب وزن مدل ترکیبی روی داده‌ی اعتبارسنجی"))
    if factor_table is not None:
        outputs.append(sweep_curve(
            factor_table, "n_factors", f"NDCG@{top_k}", "fig_factor_sweep.png",
            "تعداد عامل‌های پنهان",
            "حساسیت PureSVD به تعداد عامل‌های پنهان", log_x=True))
    return outputs
