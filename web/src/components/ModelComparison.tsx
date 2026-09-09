"use client";

import { useMemo } from "react";
import { compareModels, type LoadedModel } from "../lib/model";
import type { ModelName, Selection } from "../lib/types";
import { fa } from "../lib/format";

const COLUMNS: {
  key: ModelName; title: string; note: string; tone: "good" | "bad" | "plain";
}[] = [
  { key: "hybrid", title: "ترکیبی", tone: "good",
    note: "بهترین کیفیت رتبه‌بندی در ارزیابی" },
  { key: "collaborative", title: "PureSVD", tone: "plain",
    note: "مشارکتی، مناسب رتبه‌بندی" },
  { key: "content", title: "محتوامحور", tone: "plain",
    note: "شباهت متنی، بیشترین پوشش" },
  { key: "popularity", title: "محبوبیت", tone: "bad",
    note: "بدون شخصی‌سازی — خط مبنا" },
];

interface Props {
  model: LoadedModel;
  selections: Selection[];
}

/**
 * مقایسه‌ی کنار هم خروجی مدل‌ها روی یک ورودی یکسان.
 *
 * این نما برای دفاع ساخته شده است: با یک نگاه می‌توان دید که مدل‌های
 * مختلف روی داده‌ی یکسان چه رفتار متفاوتی دارند، و چرا مدل ترکیبی
 * انتخاب شده است.
 */
export default function ModelComparison({ model, selections }: Props) {
  const results = useMemo(
    () => compareModels(model, selections, COLUMNS.map((c) => c.key), 5),
    [model, selections],
  );

  if (selections.length === 0) return null;

  return (
    <div className="thin-scroll overflow-x-auto">
      <div className="grid min-w-[840px] grid-cols-4 gap-4">
        {COLUMNS.map((col) => (
          <section key={col.key} className="flex flex-col">
            <header
              className={
                "rounded-t-xl border border-b-0 px-3 py-2.5 " +
                (col.tone === "good"
                  ? "border-brand bg-brand-soft"
                  : "border-line bg-panel")
              }
            >
              <h3 className="text-sm font-bold">{col.title}</h3>
              <p className="mt-0.5 text-[11px] leading-5 text-ink-soft">
                {col.note}
              </p>
            </header>

            <ol
              className={
                "flex-1 space-y-0 rounded-b-xl border " +
                (col.tone === "good" ? "border-brand" : "border-line") +
                " bg-panel"
              }
            >
              {results[col.key]?.map((rec, i) => (
                <li
                  key={rec.item}
                  className="flex gap-2.5 border-b border-line px-3 py-2.5 last:border-b-0"
                >
                  <span className="mt-0.5 shrink-0 text-[11px] font-semibold text-muted">
                    {fa(i + 1)}
                  </span>
                  {/*
                    عنوان‌ها لاتین‌اند و در ظرف راست‌به‌چپ، بریدن با سه‌نقطه
                    ابتدای عنوان را حذف می‌کند. به‌جای بریدن، متن در دو خط
                    می‌شکند تا شروع عنوان همیشه دیده شود.
                  */}
                  <span className="min-w-0">
                    <span className="line-clamp-2 block text-xs font-medium leading-5">
                      {model.books[rec.item].t}
                    </span>
                    <span className="line-clamp-1 block text-[11px] text-ink-soft">
                      {model.books[rec.item].a}
                    </span>
                  </span>
                </li>
              ))}
            </ol>
          </section>
        ))}
      </div>
    </div>
  );
}
