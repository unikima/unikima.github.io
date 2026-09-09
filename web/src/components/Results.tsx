"use client";

import { useEffect, useState } from "react";
import type { EvaluationResults } from "../lib/types";
import { fa, faFixed, MODEL_LABELS } from "../lib/format";
import { BarChart, LineChart } from "./Charts";

const METRIC_COLUMNS = [
  { key: "RMSE", label: "RMSE", lower: true },
  { key: "MAE", label: "MAE", lower: true },
  { key: "Precision@10", label: "Precision@10", lower: false },
  { key: "Recall@10", label: "Recall@10", lower: false },
  { key: "NDCG@10", label: "NDCG@10", lower: false },
  { key: "Coverage", label: "پوشش", lower: false },
] as const;

export default function Results() {
  const [data, setData] = useState<EvaluationResults | null>(null);

  useEffect(() => {
    const base = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
    fetch(`${base}/model/results.json`).then((r) => r.json()).then(setData);
  }, []);

  if (!data) {
    return (
      <div className="flex flex-col items-center gap-3 py-24 text-sm text-muted">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-brand" />
        در حال بارگذاری نتایج ...
      </div>
    );
  }

  const label = (m: string) => MODEL_LABELS[m] ?? m;
  const bestAlpha = data.alphaSweep.reduce((a, b) =>
    b["NDCG@10"] > a["NDCG@10"] ? b : a).alpha;
  const bestFactors = data.factorSweep.reduce((a, b) =>
    b["NDCG@10"] > a["NDCG@10"] ? b : a).n_factors;

  return (
    <div className="space-y-12">
      <section>
        <h1 className="text-2xl font-bold">نتایج ارزیابی</h1>
        <p className="mt-2 max-w-3xl text-sm leading-7 text-ink-soft">
          شش مدل بر روی مجموعه‌ی آزمون ارزیابی شده‌اند. ارزیابی در دو بعد
          مستقل انجام شده است: خطای پیش‌بینی امتیاز و کیفیت رتبه‌بندی. این
          دو الزاماً هم‌جهت نیستند و همین، یافته‌ی محوری پژوهش است.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-bold">جدول مقایسه</h2>
        <div className="thin-scroll mt-3 overflow-x-auto rounded-xl border border-line">
          <table className="w-full min-w-[680px] border-collapse text-sm">
            <thead>
              <tr className="bg-brand-soft/60">
                <th className="px-3 py-2.5 text-right font-semibold">مدل</th>
                {METRIC_COLUMNS.map((c) => (
                  <th key={c.key} className="px-3 py-2.5 text-center font-semibold">
                    <span className="ltr inline-block">{c.label}</span>
                    <span className="block text-[10px] font-normal text-muted">
                      {c.lower ? "کمتر بهتر" : "بیشتر بهتر"}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.comparison.map((row) => {
                const best = row.model.includes("ترکیبی");
                return (
                  <tr
                    key={row.model}
                    className={
                      "border-t border-line " +
                      (best ? "bg-brand-soft/40 font-semibold" : "")
                    }
                  >
                    <td className="px-3 py-2.5 text-right">{label(row.model)}</td>
                    {METRIC_COLUMNS.map((c) => (
                      <td key={c.key} className="ltr px-3 py-2.5 text-center">
                        {faFixed(row[c.key] as number)}
                      </td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-bold">کیفیت رتبه‌بندی با بازه اطمینان</h2>
        <p className="mt-2 max-w-3xl text-sm leading-7 text-ink-soft">
          خط افقی روی هر میله، بازه‌ی اطمینان ۹۵ درصدی است که با
          بازنمونه‌گیری بوت‌استرپ روی کاربران محاسبه شده. اهمیتش این است که
          اگر بازه‌ی دو مدل هم‌پوشانی داشته باشد، نمی‌توان ادعا کرد یکی
          قطعاً بهتر از دیگری است. اینجا بازه‌های سه مدل برتر هیچ هم‌پوشانی
          ندارند.
        </p>
        <div className="mt-4 rounded-xl border border-line bg-panel p-4">
          <BarChart
            data={data.comparison.map((r) => ({
              label: label(r.model),
              value: r["NDCG@10"],
              low: r["NDCG@10 CI_low"],
              high: r["NDCG@10 CI_high"],
            }))}
            caption="NDCG@10 به همراه بازه اطمینان ۹۵ درصد — بیشتر بهتر"
          />
        </div>
      </section>

      {/*
        min-w-0 روی فرزندهای grid لازم است. فرزند grid به‌طور پیش‌فرض
        حاضر نیست از عرضِ محتوایش باریک‌تر شود، و چون نمودار داخلش حداقل
        عرض ۵۲۰ پیکسل دارد، روی موبایل ستون پهن می‌ماند و کل صفحه را
        افقی می‌کشد — حتی سربرگ سایت هم جابه‌جا می‌شد. با این کلاس ستون
        باریک می‌شود و به‌جای صفحه، خودِ نمودار در قاب خودش اسکرول
        می‌خورد.
      */}
      <section className="grid gap-6 lg:grid-cols-2">
        <div className="min-w-0 rounded-xl border border-line bg-panel p-4">
          <h3 className="text-base font-bold">جاروب وزن مدل ترکیبی</h3>
          <p className="mt-1.5 text-xs leading-6 text-ink-soft">
            وزن یک یعنی فقط مدل مشارکتی و صفر یعنی فقط محتوامحور. قله در
            میانه است، یعنی ترکیب واقعاً از هر دو جزء خود بهتر عمل می‌کند.
          </p>
          <div className="mt-3">
            <LineChart
              data={data.alphaSweep.map((d) => ({ x: d.alpha, y: d["NDCG@10"] }))}
              xLabel="وزن ترکیب (α)"
              yLabel="NDCG@10"
              highlight={bestAlpha}
              xTickFormat={(v) => fa(v.toFixed(2))}
              caption={`مقدار بهینه: α = ${fa(bestAlpha)}`}
            />
          </div>
        </div>

        <div className="min-w-0 rounded-xl border border-line bg-panel p-4">
          <h3 className="text-base font-bold">حساسیت به تعداد عامل‌های پنهان</h3>
          <p className="mt-1.5 text-xs leading-6 text-ink-soft">
            منحنی در بازه‌ی گسترده‌ای صاف است، یعنی نتیجه به این انتخاب
            حساسیت زیادی ندارد و مقدار انتخاب‌شده بحرانی نیست.
          </p>
          <div className="mt-3">
            <LineChart
              data={data.factorSweep.map((d) => ({
                x: d.n_factors, y: d["NDCG@10"],
              }))}
              xLabel="تعداد عامل پنهان"
              yLabel="NDCG@10"
              highlight={bestFactors}
              caption={`مقدار انتخاب‌شده: ${fa(bestFactors)}`}
            />
          </div>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-bold">منحنی همگرایی مدل تجزیه ماتریس</h2>
        <p className="mt-2 max-w-3xl text-sm leading-7 text-ink-soft">
          فاصله‌ی اندک میان خطای آموزش و خطای اعتبارسنجی نشان می‌دهد مدل
          دچار بیش‌برازش نشده است.
        </p>
        <div className="mt-4 rounded-xl border border-line bg-panel p-4">
          <LineChart
            data={data.trainingCurve.map((d) => ({
              x: d.epoch, y: d.valid_rmse,
            }))}
            xLabel="دوره آموزش"
            yLabel="RMSE اعتبارسنجی"
            caption="خطای مدل روی داده اعتبارسنجی در هر دوره — کمتر بهتر"
          />
        </div>
      </section>

      <section>
        <h2 className="text-lg font-bold">آزمون شروع سرد</h2>
        <p className="mt-2 max-w-3xl text-sm leading-7 text-ink-soft">
          شبیه‌سازی کاربر تازه‌وارد: از هر کاربر تنها سه کتاب پسندیده برداشته
          و بقیه‌ی سابقه‌اش نادیده گرفته شد.
        </p>
        <div className="mt-3 flex flex-wrap gap-4">
          {Object.entries(data.coldStart).map(([k, v]) => (
            <div
              key={k}
              className="rounded-xl border border-line bg-panel px-5 py-4"
            >
              <p className="ltr text-2xl font-bold">
                {Number.isInteger(v) ? fa(v) : faFixed(v)}
              </p>
              <p className="mt-1 text-xs text-muted">
                <span className="ltr inline-block">{k}</span>
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
