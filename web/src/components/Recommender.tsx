"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  loadModel, mostPopular, recommend, type LoadedModel,
} from "../lib/model";
import type { ModelName, Selection } from "../lib/types";
import { fa } from "../lib/format";
import BookCard from "./BookCard";
import BookSearch from "./BookSearch";
import ModelComparison from "./ModelComparison";

const MODEL_OPTIONS: { value: ModelName; label: string; note: string }[] = [
  { value: "hybrid", label: "ترکیبی (پیشنهادی)",
    note: "جمع وزن‌دار مدل مشارکتی و محتوامحور — بهترین نتیجه در ارزیابی" },
  { value: "collaborative", label: "مشارکتی (PureSVD)",
    note: "فقط بر پایه رفتار جمعی کاربران، بدون نگاه به محتوای کتاب" },
  { value: "content", label: "محتوامحور (TF-IDF)",
    note: "فقط بر پایه شباهت عنوان، نویسنده و برچسب‌ها" },
  { value: "itemKnn", label: "شباهت آیتم‌محور",
    note: "جمع وزن‌دار شباهت کتاب‌ها با آنچه خوانده‌اید" },
  { value: "popularity", label: "محبوبیت (پایه)",
    note: "پرخواننده‌ترین کتاب‌ها — عمداً هیچ شخصی‌سازی نمی‌کند" },
];

/** ذخیره‌ی انتخاب‌ها در نشانی صفحه، برای اشتراک‌گذاری */
function encode(selections: Selection[]): string {
  return selections.map((s) => `${s.item}.${s.rating}`).join("_");
}

function decode(value: string | null): Selection[] {
  if (!value) return [];
  return value.split("_").flatMap((part) => {
    const [item, rating] = part.split(".").map(Number);
    return Number.isInteger(item) && rating >= 1 && rating <= 5
      ? [{ item, rating }] : [];
  });
}

export default function Recommender() {
  const [model, setModel] = useState<LoadedModel | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selections, setSelections] = useState<Selection[]>([]);
  const [modelName, setModelName] = useState<ModelName>("hybrid");
  const [alpha, setAlpha] = useState(0.6);
  const [count, setCount] = useState(12);
  const [showReasons, setShowReasons] = useState(true);
  const [showComparison, setShowComparison] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    loadModel()
      .then((m) => {
        setModel(m);
        setAlpha(m.meta.alpha);
        const params = new URLSearchParams(window.location.search);
        const restored = decode(params.get("b"));
        if (restored.length) setSelections(restored);
      })
      .catch((e) => setError(String(e)));
  }, []);

  // همگام‌سازی نشانی صفحه با انتخاب‌ها، بدون افزودن به تاریخچه مرورگر
  useEffect(() => {
    if (!model) return;
    const url = new URL(window.location.href);
    if (selections.length) url.searchParams.set("b", encode(selections));
    else url.searchParams.delete("b");
    window.history.replaceState(null, "", url.toString());
  }, [selections, model]);

  const chosen = useMemo(
    () => new Set(selections.map((s) => s.item)),
    [selections],
  );

  const suggestions = useMemo(() => {
    if (!model || selections.length === 0) return [];
    return recommend(model, selections, modelName, count, alpha);
  }, [model, selections, modelName, count, alpha]);

  const starters = useMemo(
    () => (model ? mostPopular(model, 12) : []),
    [model],
  );

  // مجموعه‌ی ژانرهای کتاب‌های انتخاب‌شده، تا در پیش‌نمایش پیشنهادها
  // برچسب‌های مشترک برجسته شوند
  const selectedGenres = useMemo(() => {
    const set = new Set<string>();
    if (!model) return set;
    for (const { item } of selections) {
      for (const genre of model.books[item].g) set.add(genre);
    }
    return set;
  }, [model, selections]);

  const share = useCallback(() => {
    navigator.clipboard.writeText(window.location.href).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }, []);

  if (error) {
    return (
      <p className="rounded-xl border border-accent/40 bg-accent/10 p-4 text-sm">
        بارگذاری مدل ناموفق بود: {error}
      </p>
    );
  }

  if (!model) {
    return (
      <div className="flex flex-col items-center gap-3 py-32 text-sm text-muted">
        <div className="h-9 w-9 animate-spin rounded-full border-2 border-line border-t-brand" />
        در حال بارگذاری مدل‌ها ...
      </div>
    );
  }

  const add = (item: number) =>
    setSelections((s) =>
      s.some((x) => x.item === item) ? s : [...s, { item, rating: 5 }]);

  const remove = (item: number) =>
    setSelections((s) => s.filter((x) => x.item !== item));

  const rate = (item: number, rating: number) =>
    setSelections((s) =>
      s.map((x) => (x.item === item ? { ...x, rating } : x)));

  const isHybrid = modelName === "hybrid";
  const isPopularity = modelName === "popularity";

  return (
    <div className="space-y-12">
      {/* ------------------------------------------------ معرفی و آمار */}
      <section className="rounded-2xl border border-line bg-gradient-to-bl from-brand-soft/70 to-panel p-6 sm:p-8">
        <h1 className="text-2xl font-bold sm:text-3xl">
          کتاب بعدی‌ات را پیدا کن
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-7 text-ink-soft">
          چند کتابی که دوست داشته‌ای را انتخاب کن. سامانه از روی همین چند
          انتخاب، الگوی سلیقه‌ات را حدس می‌زند و فهرستی شخصی‌سازی‌شده
          می‌سازد — و برای هر پیشنهاد می‌گوید چرا آن را انتخاب کرده است.
        </p>

        {/*
          این آمار برای کاربری که دنبال کتاب است ضروری نیست، ولی مقیاس داده‌ی
          پشت سامانه را در یک نگاه نشان می‌دهد. اعداد گرد شده‌اند چون در این
          جایگاه «حدود شش میلیون» خواناتر از رقم دقیق است؛ عدد دقیق در گزارش
          و صفحه‌ی نتایج آمده.
        */}
        <dl className="mt-6 flex flex-wrap gap-x-8 gap-y-3">
          {[
            [`${fa("5.98")} میلیون`, "امتیاز واقعی که مدل از آن یاد گرفته"],
            [`${fa(53)} هزار`, "کاربر پشت این امتیازها"],
            [`${fa(10)} هزار`, "کتاب در کاتالوگ"],
            [fa(6), "مدل پیاده‌سازی و مقایسه‌شده"],
          ].map(([value, label]) => (
            <div key={label}>
              <dd className="text-xl font-bold text-brand">{value}</dd>
              <dt className="mt-0.5 text-xs text-ink-soft">{label}</dt>
            </div>
          ))}
        </dl>

        <div className="mt-6 max-w-xl">
          <BookSearch
            books={model.books}
            popularity={model.popularity}
            onPick={add}
            excluded={chosen}
          />
          {/*
            هرچه کاربر کتاب بیشتری بدهد پیشنهادها دقیق‌تر می‌شوند، ولی
            این از خودِ صفحه پیدا نبود و کاربر بعد از کتاب اول گمان
            می‌کرد کار تمام است. این یک خط، هر دو راهِ افزودن را
            می‌گوید.
          */}
          {selections.length > 0 && (
            <p className="mt-2 text-xs leading-6 text-muted">
              برای دقیق‌تر شدن پیشنهادها، کتاب‌های بیشتری اضافه کن — از همین
              کادر جست‌وجو، یا با کلیک روی هر کتاب در فهرست پیشنهادها.
            </p>
          )}
        </div>
      </section>

      {selections.length === 0 && (
        <section>
          <h2 className="text-sm font-semibold text-ink-soft">
            یا از میان پرخواننده‌ترین‌ها شروع کن:
          </h2>
          <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
            {starters.map((i) => (
              <BookCard
                key={i}
                book={model.books[i]}
                onClick={() => add(i)}
                compact
              />
            ))}
          </div>
        </section>
      )}

      {/* --------------------------------------------- انتخاب‌های کاربر */}
      {selections.length > 0 && (
        <section>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-bold">
              انتخاب‌های شما{" "}
              <span className="text-sm font-normal text-muted">
                ({fa(selections.length)} کتاب)
              </span>
            </h2>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={share}
                className="rounded-lg border border-line px-3 py-1.5 text-xs text-ink-soft transition-colors hover:border-brand hover:text-brand"
              >
                {copied ? "لینک کپی شد ✓" : "کپی لینک این انتخاب"}
              </button>
              <button
                type="button"
                onClick={() => setSelections([])}
                className="rounded-lg border border-line px-3 py-1.5 text-xs text-ink-soft transition-colors hover:border-accent hover:text-accent"
              >
                پاک کردن همه
              </button>
            </div>
          </div>

          <div className="mt-3 space-y-2">
            {selections.map(({ item, rating }) => (
              <div
                key={item}
                className="flex flex-wrap items-center gap-3 rounded-xl border border-line bg-panel px-4 py-3"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium">
                    {model.books[item].t}
                  </p>
                  <p className="truncate text-xs text-ink-soft">
                    {model.books[item].a}
                  </p>
                </div>

                <label className="flex items-center gap-2 text-xs text-ink-soft">
                  امتیاز شما
                  <input
                    type="range" min={1} max={5} value={rating}
                    onChange={(e) => rate(item, Number(e.target.value))}
                    className="accent-[var(--brand)]"
                  />
                  <span className="w-4 text-center font-semibold text-ink">
                    {fa(rating)}
                  </span>
                </label>

                <button
                  type="button"
                  onClick={() => remove(item)}
                  className="rounded-lg border border-line px-2.5 py-1 text-xs text-ink-soft transition-colors hover:border-accent hover:text-accent"
                >
                  حذف
                </button>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* ------------------------------------------------- پیشنهادها */}
      {suggestions.length > 0 && (
        <section>
          <div className="flex flex-wrap items-end justify-between gap-4">
            <h2 className="text-lg font-bold">پیشنهادهای سامانه</h2>

            <div className="flex flex-wrap items-center gap-4 text-xs">
              <label className="flex items-center gap-2">
                <span className="text-ink-soft">مدل</span>
                <select
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value as ModelName)}
                  className="rounded-lg border border-line bg-panel px-2.5 py-1.5 outline-none focus:border-brand"
                >
                  {MODEL_OPTIONS.map((o) => (
                    <option key={o.value} value={o.value}>{o.label}</option>
                  ))}
                </select>
              </label>

              <label className="flex items-center gap-2">
                <span className="text-ink-soft">تعداد</span>
                <select
                  value={count}
                  onChange={(e) => setCount(Number(e.target.value))}
                  className="rounded-lg border border-line bg-panel px-2.5 py-1.5 outline-none focus:border-brand"
                >
                  {[6, 12, 18, 24].map((n) => (
                    <option key={n} value={n}>{fa(n)}</option>
                  ))}
                </select>
              </label>

              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={showReasons}
                  onChange={(e) => setShowReasons(e.target.checked)}
                  className="accent-[var(--brand)]"
                />
                <span className="text-ink-soft">نمایش دلیل</span>
              </label>
            </div>
          </div>

          <p className="mt-2 text-xs leading-6 text-muted">
            {MODEL_OPTIONS.find((o) => o.value === modelName)?.note}
          </p>

          {/* هشدار مدل پایه */}
          {isPopularity && (
            <div className="mt-4 rounded-xl border border-accent/40 bg-accent/8 p-4 text-xs leading-6">
              <strong className="text-accent">توجه:</strong> این مدل عمداً به
              انتخاب‌های شما نگاه نمی‌کند و به همه‌ی کاربران فهرست یکسانی
              می‌دهد. وجودش برای فراهم کردن یک خط مبنا است: هر مدل
              شخصی‌سازی‌شده‌ای باید بتواند از این بهتر عمل کند، وگرنه ارزشی
              نیفزوده است. در ارزیابی، مدل ترکیبی حدود{" "}
              <span className="ltr inline-block font-semibold">۱۸۶٪</span>{" "}
              از این مدل بهتر عمل کرد.
            </div>
          )}

          {/* اسلایدر زنده وزن ترکیب */}
          {isHybrid && (
            <div className="mt-4 rounded-xl border border-line bg-panel p-4">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h3 className="text-sm font-semibold">
                  وزن ترکیب دو مدل{" "}
                  <span className="ltr inline-block font-bold text-brand">
                    α = {fa(alpha.toFixed(2))}
                  </span>
                </h3>
                {Math.abs(alpha - model.meta.alpha) > 0.001 && (
                  <button
                    type="button"
                    onClick={() => setAlpha(model.meta.alpha)}
                    className="rounded-lg border border-line px-2.5 py-1 text-[11px] text-ink-soft hover:border-brand hover:text-brand"
                  >
                    بازگشت به مقدار بهینه ({fa(model.meta.alpha)})
                  </button>
                )}
              </div>

              {/*
                اسلایدر و برچسب‌هایش صریحاً چپ‌به‌راست‌اند. در ظرف
                راست‌به‌چپ، مرورگر کمینه‌ی بازه را سمت راست می‌گذارد و
                برچسب‌ها با موقعیت واقعی اهرم جابه‌جا می‌شوند. چون این یک
                محور عددی صفر تا یک است، همان جهت محورهای نمودارها را
                می‌گیرد.
              */}
              <div dir="ltr" className="mt-3">
                <input
                  dir="ltr"
                  type="range" min={0} max={1} step={0.05} value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))}
                  className="w-full accent-[var(--brand)]"
                  aria-label="وزن ترکیب"
                />
                <div className="flex justify-between text-[11px] text-muted">
                  <span>{fa(0)} — فقط محتوامحور</span>
                  <span>{fa(1)} — فقط مشارکتی</span>
                </div>
              </div>

              <p className="mt-3 text-xs leading-6 text-ink-soft">
                اسلایدر را حرکت بدهید و ببینید پیشنهادها چطور عوض می‌شوند.
                مقدار{" "}
                <span className="ltr inline-block font-semibold">
                  {fa(model.meta.alpha)}
                </span>{" "}
                از روی جاروب بر داده‌ی اعتبارسنجی به دست آمده و بهترین کیفیت
                رتبه‌بندی را می‌دهد — نه دو انتها. همین نشان می‌دهد ترکیب
                واقعاً از هر دو جزء خودش بهتر عمل می‌کند.
              </p>
            </div>
          )}

          <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {suggestions.map((s, k) => (
              <BookCard
                key={s.item}
                book={model.books[s.item]}
                rank={k + 1}
                match={s.match}
                rawScore={s.score}
                collaborativeShare={s.collaborativeShare}
                highlightGenres={selectedGenres}
                // کلیک روی هر پیشنهاد، آن را به انتخاب‌ها اضافه می‌کند و
                // فهرست بلافاصله دقیق‌تر می‌شود. بدون این، کارت‌ها ظاهر
                // تعاملی داشتند اما کلیک هیچ کاری نمی‌کرد.
                onClick={() => add(s.item)}
                reasons={
                  showReasons
                    ? s.reasons.map((r) => model.books[r.item].t)
                    : undefined
                }
              />
            ))}
          </div>

          <div className="mt-4 space-y-2 text-xs leading-6 text-muted">
            <p>
              «میزان تطابق» نسبت به قوی‌ترین پیشنهاد همین فهرست سنجیده
              می‌شود. عدد «نمره مدل» خروجی خام الگوریتم است و مقیاس ۱ تا ۵
              ندارد؛ تنها ترتیب آن معنا دارد، نه مقدار مطلقش.
            </p>
            {showReasons && (
              <p>
                دلیل‌ها از ماتریس شباهت آیتم‌محور استخراج می‌شوند: سهم هر
                کتابِ انتخاب‌شده در امتیاز نهایی محاسبه و بزرگ‌ترین سهم‌ها
                نمایش داده می‌شود. مدل تجزیه ماتریس چنین قابلیتی ندارد، چون
                امتیازش از ضرب دو بردار پنهان می‌آید و به کتاب مشخصی قابل
                نسبت دادن نیست.
              </p>
            )}
          </div>
        </section>
      )}

      {/* ------------------------------------------- مقایسه مدل‌ها */}
      {selections.length > 0 && (
        <section>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-lg font-bold">مقایسه مدل‌ها روی همین ورودی</h2>
            <button
              type="button"
              onClick={() => setShowComparison((v) => !v)}
              className="rounded-lg border border-line px-3 py-1.5 text-xs transition-colors hover:border-brand hover:text-brand"
            >
              {showComparison ? "بستن" : "نمایش مقایسه"}
            </button>
          </div>

          {showComparison && (
            <>
              <p className="mt-2 max-w-3xl text-xs leading-6 text-ink-soft">
                هر چهار مدل دقیقاً همین انتخاب‌ها را دریافت کرده‌اند. تفاوت
                خروجی‌شان نشان می‌دهد چرا انتخاب مدل اهمیت دارد و چرا در
                نهایت مدل ترکیبی انتخاب شد.
              </p>
              <div className="mt-4">
                <ModelComparison model={model} selections={selections} />
              </div>
            </>
          )}
        </section>
      )}

      {/* اعداد مجموعه‌داده بالای صفحه آمده‌اند و تکرارشان اینجا لازم نیست */}
      <footer className="border-t border-line pt-5 text-xs leading-6 text-muted">
        تمام محاسبات در مرورگر شما انجام می‌شود و هیچ داده‌ای به سروری ارسال
        نمی‌گردد.
      </footer>
    </div>
  );
}
