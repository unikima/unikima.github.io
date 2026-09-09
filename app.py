"""رابط کاربری وب سیستم توصیه‌گر کتاب.

اجرا:
    streamlit run app.py

پیش‌نیاز: ابتدا باید `python -m src.train` اجرا شده باشد تا مدل‌های آموزش‌دیده
در پوشه‌ی artifacts/ ساخته شوند.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src import config as cfg
from src.data_loader import load_dataset

st.set_page_config(page_title="سیستم توصیه‌گر کتاب",
                   page_icon="📚", layout="wide")

# ---------------------------------------------------------------- ظاهر فارسی
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton, .stSelectbox, .stSlider {
    font-family: 'Vazirmatn', Tahoma, sans-serif;
}
.main .block-container { direction: rtl; text-align: right; max-width: 1180px; }
h1, h2, h3, h4 { font-weight: 700; }
[data-testid="stSidebar"] { direction: rtl; text-align: right; }
[data-testid="stMetricValue"] { direction: ltr; }
.book-card {
    border: 1px solid rgba(11,11,11,0.10);
    border-radius: 12px; padding: 12px; height: 100%;
    background: #fcfcfb;
}
.book-cover {
    width: 100%; aspect-ratio: 2/3; object-fit: cover;
    border-radius: 8px; display: block; margin-bottom: 10px;
    background: #f0efec;
}
.book-cover.fallback {
    display: flex; align-items: center; justify-content: center;
    font-size: 2rem; font-weight: 700; color: #898781;
    border: 1px dashed #c3c2b7;
}
.book-title { font-weight: 600; font-size: 0.92rem; line-height: 1.5; margin: 8px 0 2px; }
.book-author { font-size: 0.8rem; color: #52514e; }
.book-score { font-size: 0.78rem; color: #2a78d6; direction: ltr; text-align: right; }
.book-reason {
    font-size: 0.74rem; color: #52514e; line-height: 1.6;
    margin-top: 8px; padding-top: 8px;
    border-top: 1px solid rgba(11,11,11,0.08);
}
.rank-badge {
    display: inline-block; background: #2a78d6; color: #fff;
    border-radius: 6px; padding: 1px 8px; font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- بارگذاری
@st.cache_resource(show_spinner="در حال بارگذاری مدل‌ها ...")
def load_everything():
    models_path = cfg.ARTIFACT_DIR / "models.pkl"
    if not models_path.exists():
        return None, None
    ds = load_dataset()
    with open(models_path, "rb") as fh:
        models = pickle.load(fh)
    return ds, models


@st.cache_data
def load_results():
    path = cfg.RESULTS_DIR / "comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)


ds, models = load_everything()

if ds is None:
    st.error("مدل‌های آموزش‌دیده پیدا نشد.")
    st.code("python -m src.train", language="bash")
    st.stop()

books = ds.books
TITLES = books["title"].fillna("بدون عنوان").tolist()
AUTHORS = books["authors"].fillna("نامشخص").tolist()
IMAGES = books["image_url"].fillna("").tolist()


def _escape(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def book_cards(indices, scores=None, columns: int = 5,
               explanations: dict | None = None) -> None:
    """نمایش شبکه‌ای کتاب‌ها با جلد، عنوان و نویسنده.

    تصویر جلد از سرور Goodreads بارگذاری می‌شود. اگر دسترسی به اینترنت
    نباشد یا نشانی تصویر از کار افتاده باشد، به‌جای آیکون شکسته یک کادر
    ساده با حرف اول عنوان نشان داده می‌شود.
    """
    indices = list(indices)
    for start in range(0, len(indices), columns):
        chunk = indices[start:start + columns]
        cols = st.columns(columns)
        for col, (offset, idx) in zip(cols, enumerate(chunk, start=start)):
            idx = int(idx)
            title = _escape(TITLES[idx])
            author = _escape(AUTHORS[idx])
            initial = _escape(TITLES[idx][:1].upper() or "؟")

            if IMAGES[idx]:
                cover = (f'<img class="book-cover" src="{_escape(IMAGES[idx])}" '
                         f'alt="{title}" loading="lazy" '
                         f'onerror="this.outerHTML='
                         f'&quot;<div class=\'book-cover fallback\'>'
                         f'{initial}</div>&quot;">')
            else:
                cover = f'<div class="book-cover fallback">{initial}</div>'

            score_html = ""
            if scores is not None:
                score_html = (f'<div class="book-score">'
                              f'امتیاز: {float(scores[idx]):.3f}</div>')

            reason_html = ""
            if explanations and explanations.get(idx):
                names = "، ".join(_escape(TITLES[j][:34])
                                  for j, _ in explanations[idx])
                reason_html = (f'<div class="book-reason">چون خوانده‌اید: '
                               f'{names}</div>')

            with col:
                st.markdown(
                    f'<div class="book-card">{cover}'
                    f'<span class="rank-badge">{offset + 1}</span>'
                    f'<div class="book-title">{title}</div>'
                    f'<div class="book-author">{author}</div>'
                    f'{score_html}{reason_html}</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------- نوار کناری
st.sidebar.title("سیستم توصیه‌گر کتاب")
st.sidebar.caption("پروژه کارشناسی مهندسی کامپیوتر")

MODEL_LABELS = {
    "ترکیبی (PureSVD + محتوا) — پیشنهادی": "hybrid",
    "PureSVD": "pure_svd",
    "تجزیه ماتریس (SVD)": "mf",
    "شباهت آیتم‌محور (KNN)": "item_knn",
    "محتوامحور (TF-IDF)": "content",
    "محبوبیت (پایه)": "popularity",
}
model_label = st.sidebar.selectbox("مدل توصیه‌گر", list(MODEL_LABELS.keys()))
model = models[MODEL_LABELS[model_label]]
top_n = st.sidebar.slider("تعداد پیشنهاد", 5, 20, 10, step=5)

st.sidebar.divider()
st.sidebar.metric("تعداد کاربران", f"{ds.n_users:,}")
st.sidebar.metric("تعداد کتاب‌ها", f"{ds.n_items:,}")
st.sidebar.metric("تعداد امتیازها", f"{len(ds.train_df) + len(ds.test_df):,}")

# ---------------------------------------------------------------- صفحه اصلی
st.title("📚 سیستم توصیه‌گر کتاب")

tab_new, tab_user, tab_similar, tab_eval, tab_about = st.tabs(
    ["کاربر جدید", "کاربر موجود", "کتاب‌های مشابه", "نتایج ارزیابی", "درباره پروژه"]
)

# ----- کاربر جدید (سناریوی شروع سرد) -----
with tab_new:
    st.subheader("چند کتابی که دوست داشته‌اید را انتخاب کنید")
    st.caption("این حالت، مسئله‌ی «شروع سرد» را نشان می‌دهد: کاربری که هیچ "
               "سابقه‌ای در سامانه ندارد و سیستم باید تنها از روی چند انتخاب "
               "اولیه، سلیقه‌اش را حدس بزند.")

    picked = st.multiselect(
        "جستجو و انتخاب کتاب (حداقل یک مورد)",
        options=list(range(len(TITLES))),
        format_func=lambda k: f"{TITLES[k]} — {AUTHORS[k]}",
        max_selections=10,
    )

    ratings_input = []
    if picked:
        st.write("چقدر از هرکدام خوشتان آمد؟")
        rating_cols = st.columns(min(len(picked), 4))
        for pos, idx in enumerate(picked):
            with rating_cols[pos % len(rating_cols)]:
                value = st.slider(TITLES[idx][:38], 1, 5, 5, key=f"rate_{idx}")
                ratings_input.append(value)

    if st.button("پیشنهاد بده", type="primary", disabled=not picked):
        hybrid = models["hybrid"]
        scores = hybrid.score_for_new_user(np.array(picked),
                                           np.array(ratings_input, dtype=float))
        top = np.argpartition(-scores, top_n)[:top_n]
        top = top[np.argsort(-scores[top])]
        st.divider()
        st.subheader("پیشنهادهای سیستم")
        book_cards(top, scores)

# ----- کاربر موجود -----
with tab_user:
    st.subheader("توصیه برای کاربری از مجموعه‌داده")
    st.caption("کاربران این بخش در داده‌ی آموزش سابقه دارند، بنابراین هر پنج "
               "مدل قابل استفاده‌اند و می‌توان خروجی‌شان را مقایسه کرد.")

    train_csr = ds.train_matrix.tocsr()
    activity = np.diff(train_csr.indptr)
    active_users = np.argsort(-activity)[:2000]

    user = st.selectbox(
        "انتخاب کاربر",
        options=active_users[:300],
        format_func=lambda u: f"کاربر شماره {u}  ({activity[u]} امتیاز ثبت‌شده)",
    )

    history = train_csr[int(user)]
    order = np.argsort(-history.data)[:6]
    favourites = history.indices[order]

    st.markdown("**کتاب‌هایی که این کاربر بالاترین امتیاز را داده:**")
    book_cards(favourites, columns=6)

    st.divider()
    st.markdown(f"**پیشنهاد مدل «{model_label}»:**")
    show_reasons = st.checkbox("نمایش دلیل هر پیشنهاد", value=True,
                              key="show_reasons")
    scores = model.score_all_items(int(user))
    top = model.recommend(int(user), n=top_n, exclude_seen=True,
                          seen=history.indices)

    reasons = None
    if show_reasons:
        knn_model = models["item_knn"]
        reasons = {int(i): knn_model.explain(int(user), int(i), top_n=2)
                   for i in top}

    book_cards(top, np.asarray(scores, dtype=float), explanations=reasons)

    if show_reasons:
        st.caption("دلیل‌ها از ماتریس شباهت آیتم‌محور استخراج می‌شوند: سهم هر "
                   "کتابِ خوانده‌شده در امتیاز نهایی محاسبه و بزرگ‌ترین سهم‌ها "
                   "نمایش داده می‌شود. مدل تجزیه ماتریس چنین قابلیتی ندارد، "
                   "چون امتیازش از ضرب دو بردار پنهان می‌آید و به کتاب مشخصی "
                   "قابل نسبت دادن نیست.")

# ----- کتاب‌های مشابه -----
with tab_similar:
    st.subheader("کتاب‌های مشابه یک کتاب")
    st.caption("این بخش مستقیماً ماتریس شباهت آیتم‌محور را نشان می‌دهد و برای "
               "بررسی چشمی کیفیت مدل مفید است: اگر شباهت‌ها منطقی به نظر برسند، "
               "مدل چیز درستی یاد گرفته است.")

    seed_book = st.selectbox(
        "یک کتاب انتخاب کنید",
        options=list(range(len(TITLES))),
        format_func=lambda k: f"{TITLES[k]} — {AUTHORS[k]}",
        key="similar_seed",
    )

    knn = models["item_knn"]
    row = knn.sim_[seed_book]
    if row.nnz == 0:
        st.info("برای این کتاب همسایه‌ی معتبری یافت نشد.")
    else:
        order = np.argsort(-row.data)[:top_n]
        neighbours = row.indices[order]
        similarity = np.zeros(ds.n_items)
        similarity[neighbours] = row.data[order]
        book_cards(neighbours, similarity)

# ----- نتایج ارزیابی -----
with tab_eval:
    st.subheader("مقایسه‌ی کمّی مدل‌ها")
    table = load_results()
    if table is None:
        st.info("فایل نتایج پیدا نشد. ابتدا `python -m src.train` را اجرا کنید.")
    else:
        st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
        st.caption("در ستون‌های RMSE و MAE مقدار کمتر بهتر است؛ در بقیه‌ی "
                   "ستون‌ها مقدار بیشتر بهتر است.")

        params_path = cfg.RESULTS_DIR / "chosen_hyperparameters.csv"
        if params_path.exists():
            chosen = pd.read_csv(params_path).iloc[0]
            col_a, col_b = st.columns(2)
            col_a.metric("وزن بهینه مدل ترکیبی (α)", f"{chosen['best_alpha']:.2f}")
            col_b.metric("تعداد عامل پنهان PureSVD", int(chosen["best_n_factors"]))
            st.caption("هر دو مقدار روی مجموعه‌ی اعتبارسنجی انتخاب شده‌اند، "
                       "نه روی داده‌ی آزمون.")

        figures = [
            ("fig_ndcg_ci.png",
             f"کیفیت رتبه‌بندی با بازه اطمینان ۹۵٪ (NDCG@{cfg.TOP_K}) — "
             "اگر بازه‌ها هم‌پوشانی داشته باشند، برتری قطعی نیست"),
            ("fig_alpha_sweep.png",
             "جاروب وزن مدل ترکیبی روی داده‌ی اعتبارسنجی"),
            ("fig_factor_sweep.png",
             "حساسیت PureSVD به تعداد عامل‌های پنهان"),
            ("fig_rmse.png", "خطای پیش‌بینی امتیاز (RMSE)"),
            ("fig_precision.png", f"دقت در ده پیشنهاد اول (Precision@{cfg.TOP_K})"),
            ("fig_ndcg.png", f"کیفیت رتبه‌بندی (NDCG@{cfg.TOP_K})"),
            ("fig_coverage.png", "پوشش فهرست کتاب‌ها (Coverage)"),
            ("mf_curve.png", "منحنی همگرایی مدل تجزیه ماتریس"),
        ]
        for filename, caption in figures:
            path = cfg.RESULTS_DIR / filename
            if path.exists():
                st.image(str(path), caption=caption, use_container_width=True)

        cold_path = cfg.RESULTS_DIR / "cold_start.csv"
        if cold_path.exists():
            st.divider()
            st.markdown("**نتیجه‌ی آزمون شروع سرد** — کاربر تازه‌وارد با تنها ۳ کتاب:")
            st.dataframe(pd.read_csv(cold_path), use_container_width=True)

# ----- درباره -----
with tab_about:
    st.markdown(f"""
### معرفی

این سامانه یک سیستم توصیه‌گر کتاب است که روی مجموعه‌داده‌ی
**goodbooks-10k** آموزش دیده و پنج رویکرد مختلف را پیاده‌سازی و
با هم مقایسه می‌کند.

### مدل‌های پیاده‌سازی‌شده

| مدل | ایده‌ی اصلی | نقطه‌ی قوت | نقطه‌ی ضعف |
|---|---|---|---|
| محبوبیت | پرخواننده‌ترین کتاب‌ها برای همه | ساده و پایدار | اصلاً شخصی‌سازی نمی‌کند |
| شباهت آیتم‌محور | کتاب‌های شبیه به آنچه خوانده‌اید | قابل تفسیر | برای کتاب تازه کار نمی‌کند |
| تجزیه ماتریس | عامل‌های پنهان، آموزش روی خانه‌های پرشده | کمترین خطای پیش‌بینی | در رتبه‌بندی ضعیف |
| PureSVD | تجزیه‌ی ماتریس کامل با صفر برای خانه‌های خالی | بهترین رتبه‌بندی مشارکتی | خطای پیش‌بینی بالاتر |
| محتوامحور | شباهت متنی عنوان، نویسنده و برچسب | حل شروع سرد، پوشش بالا | سلیقه‌ی جمعی را نمی‌بیند |
| ترکیبی | جمع وزن‌دار PureSVD و محتوامحور | بهترین رتبه‌بندی کلی | پیچیدگی بیشتر |

### یک نکته‌ی مهم در نتایج

مدل تجزیه ماتریس کمترین خطای پیش‌بینی (RMSE) را دارد، ولی وقتی از آن
می‌خواهیم ده کتاب برتر را نام ببرد، عملکرد ضعیفی نشان می‌دهد. علتش این
است که این مدل فقط روی خانه‌های پرشده‌ی ماتریس آموزش می‌بیند و هرگز
نمی‌آموزد که «خوانده‌نشدن» هم خودش یک سیگنال است؛ در نتیجه کتاب‌های
کم‌خواننده‌ی پرامتیاز را بالا می‌آورد.

این یعنی **پیش‌بینی امتیاز و رتبه‌بندی دو مسئله‌ی متفاوت‌اند** و بهترین
مدل برای یکی، لزوماً بهترین برای دیگری نیست. به همین دلیل در این پروژه
مدل PureSVD اضافه شد که خانه‌های خالی را صفر در نظر می‌گیرد و برای
رتبه‌بندی طراحی شده است.

### داده‌ها

- کاربران: {ds.n_users:,}
- کتاب‌ها: {ds.n_items:,}
- امتیازها: {len(ds.train_df) + len(ds.test_df):,}
- تقسیم آموزش/آزمون: {100 * (1 - cfg.TEST_FRACTION):.0f}٪ / {100 * cfg.TEST_FRACTION:.0f}٪ به تفکیک هر کاربر

### ساختار کد

```
src/config.py      تنظیمات
src/data_loader.py خواندن، پاک‌سازی و تقسیم داده
src/models/        پنج مدل توصیه‌گر
src/evaluate.py    معیارهای ارزیابی
src/train.py       اسکریپت آموزش
app.py             همین رابط کاربری
```
""")
