"use client";

import { useId, useState } from "react";
import { fa } from "../lib/format";

/**
 * نمودارها با SVG درون‌خطی رسم می‌شوند، بدون کتابخانه‌ی بیرونی.
 *
 * قاعده‌های رعایت‌شده: هر نمودار فقط یک محور مقداری دارد؛ چون در هر نمودار
 * یک کمیت را بین چند مورد مقایسه می‌کنیم، یک مجموعه‌داده داریم و بنابراین
 * یک رنگ استفاده می‌شود (رنگ‌های متفاوت وقتی معنا دارند که هویت‌های متفاوتی
 * را نشان دهند)؛ خطوط راهنما کم‌رنگ‌اند تا داده برجسته بماند؛ و مقدار هر
 * میله مستقیماً کنارش نوشته می‌شود تا خواندن نمودار به محور وابسته نباشد.
 */

export interface BarDatum {
  label: string;
  value: number;
  low?: number;
  high?: number;
}

interface BarProps {
  data: BarDatum[];
  /** مقدار بیشتر بهتر است؟ برای مرتب‌سازی و مشخص کردن بهترین */
  higherIsBetter?: boolean;
  digits?: number;
  caption?: string;
}

export function BarChart({
  data, higherIsBetter = true, digits = 4, caption,
}: BarProps) {
  const rows = [...data].sort((a, b) =>
    higherIsBetter ? b.value - a.value : a.value - b.value);

  const max = Math.max(...rows.map((r) => r.high ?? r.value));
  const rowHeight = 34;
  const labelWidth = 132;
  const valueWidth = 62;
  const chartWidth = 560;
  const plotWidth = chartWidth - labelWidth - valueWidth;
  const height = rows.length * rowHeight + 12;

  const x = (v: number) => (v / (max * 1.02)) * plotWidth;

  return (
    <figure className="overflow-x-auto">
      {/*
        جهت SVG صریحاً چپ‌به‌راست تنظیم می‌شود. صفحه راست‌به‌چپ است و اگر
        این جهت به SVG هم ارث برسد، معنای text-anchor برعکس می‌شود و
        برچسب‌ها از کادر بیرون می‌زنند. متن فارسی داخل هر برچسب همچنان
        درست شکل می‌گیرد، چون جهت‌دهی حروف به خود کاراکترها وابسته است.
      */}
      <svg
        viewBox={`0 0 ${chartWidth} ${height}`}
        className="w-full min-w-[520px]"
        style={{ direction: "ltr" }}
        role="img"
        aria-label={caption}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <line
            key={f}
            x1={valueWidth + plotWidth * f}
            x2={valueWidth + plotWidth * f}
            y1={0}
            y2={rows.length * rowHeight}
            stroke="var(--line)"
            strokeWidth={1}
          />
        ))}

        {rows.map((r, i) => {
          const y = i * rowHeight;
          const best = i === 0;
          const barW = Math.max(x(r.value), 2);
          return (
            <g key={r.label}>
              <text
                x={chartWidth - 4}
                y={y + rowHeight / 2 + 4}
                textAnchor="end"
                fontSize={12}
                fill="var(--ink-soft)"
              >
                {r.label}
              </text>

              <rect
                x={valueWidth + plotWidth - barW}
                y={y + 8}
                width={barW}
                height={rowHeight - 18}
                rx={4}
                fill={best ? "var(--brand)" : "var(--brand)"}
                opacity={best ? 1 : 0.42}
              />

              {r.low !== undefined && r.high !== undefined && (
                <g stroke="var(--ink)" strokeWidth={1.3}>
                  <line
                    x1={valueWidth + plotWidth - x(r.low)}
                    x2={valueWidth + plotWidth - x(r.high)}
                    y1={y + rowHeight / 2}
                    y2={y + rowHeight / 2}
                  />
                  <line
                    x1={valueWidth + plotWidth - x(r.low)}
                    x2={valueWidth + plotWidth - x(r.low)}
                    y1={y + rowHeight / 2 - 4}
                    y2={y + rowHeight / 2 + 4}
                  />
                  <line
                    x1={valueWidth + plotWidth - x(r.high)}
                    x2={valueWidth + plotWidth - x(r.high)}
                    y1={y + rowHeight / 2 - 4}
                    y2={y + rowHeight / 2 + 4}
                  />
                </g>
              )}

              <text
                x={valueWidth - 8}
                y={y + rowHeight / 2 + 4}
                textAnchor="end"
                fontSize={11.5}
                fill="var(--ink)"
                style={{ direction: "ltr" }}
              >
                {fa(r.value.toFixed(digits))}
              </text>
            </g>
          );
        })}
      </svg>
      {caption && (
        <figcaption className="mt-2 text-xs text-muted">{caption}</figcaption>
      )}
    </figure>
  );
}

export interface LinePoint {
  x: number;
  y: number;
}

interface LineProps {
  data: LinePoint[];
  xLabel: string;
  yLabel: string;
  caption?: string;
  /** مقدار x انتخاب‌شده، که با نشانگر بزرگ‌تر مشخص می‌شود */
  highlight?: number;
  xTickFormat?: (v: number) => string;
}

export function LineChart({
  data, xLabel, yLabel, caption, highlight, xTickFormat,
}: LineProps) {
  const [hover, setHover] = useState<number | null>(null);
  const gradientId = useId();

  const width = 560;
  const height = 240;
  const pad = { top: 16, right: 18, bottom: 42, left: 54 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const xs = data.map((d) => d.x);
  const ys = data.map((d) => d.y);
  const xMin = Math.min(...xs);
  const xMax = Math.max(...xs);
  const yMin = Math.min(...ys);
  const yMax = Math.max(...ys);
  const yPad = (yMax - yMin) * 0.18 || 0.01;

  const px = (v: number) =>
    pad.left + ((v - xMin) / (xMax - xMin || 1)) * plotW;
  const py = (v: number) =>
    pad.top + plotH -
    ((v - (yMin - yPad)) / ((yMax + yPad) - (yMin - yPad))) * plotH;

  const path = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${px(d.x)} ${py(d.y)}`)
    .join(" ");

  const yTicks = 4;

  return (
    <figure className="overflow-x-auto">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full min-w-[520px]"
        style={{ direction: "ltr" }}
        role="img"
        aria-label={caption}
        onMouseLeave={() => setHover(null)}
      >
        <defs>
          <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="var(--brand)" stopOpacity={0.16} />
            <stop offset="100%" stopColor="var(--brand)" stopOpacity={0} />
          </linearGradient>
        </defs>

        {Array.from({ length: yTicks + 1 }, (_, i) => {
          const v = (yMin - yPad) +
            (i / yTicks) * ((yMax + yPad) - (yMin - yPad));
          return (
            <g key={i}>
              <line
                x1={pad.left} x2={pad.left + plotW}
                y1={py(v)} y2={py(v)}
                stroke="var(--line)" strokeWidth={1}
              />
              <text
                x={pad.left - 8} y={py(v) + 4}
                textAnchor="end" fontSize={10.5} fill="var(--muted)"
                style={{ direction: "ltr" }}
              >
                {fa(v.toFixed(3))}
              </text>
            </g>
          );
        })}

        <path
          d={`${path} L ${px(xMax)} ${pad.top + plotH} L ${px(xMin)} ${pad.top + plotH} Z`}
          fill={`url(#${gradientId})`}
        />
        <path d={path} fill="none" stroke="var(--brand)" strokeWidth={2} />

        {data.map((d, i) => {
          const isBest = highlight !== undefined && d.x === highlight;
          const isHover = hover === i;
          return (
            <g key={d.x}>
              <circle
                cx={px(d.x)} cy={py(d.y)}
                r={isBest ? 6 : 4}
                fill="var(--brand)"
                stroke="var(--surface)"
                strokeWidth={isBest ? 2 : 1}
              />
              <circle
                cx={px(d.x)} cy={py(d.y)} r={14}
                fill="transparent"
                onMouseEnter={() => setHover(i)}
              />
              {isHover && (
                <text
                  x={px(d.x)} y={py(d.y) - 12}
                  textAnchor="middle" fontSize={11} fill="var(--ink)"
                  style={{ direction: "ltr" }}
                >
                  {fa(d.y.toFixed(4))}
                </text>
              )}
              <text
                x={px(d.x)} y={pad.top + plotH + 16}
                textAnchor="middle" fontSize={10.5} fill="var(--muted)"
                style={{ direction: "ltr" }}
              >
                {xTickFormat ? xTickFormat(d.x) : fa(d.x)}
              </text>
            </g>
          );
        })}

        <text
          x={pad.left + plotW / 2} y={height - 6}
          textAnchor="middle" fontSize={11} fill="var(--muted)"
        >
          {xLabel}
        </text>
        <text
          x={14} y={pad.top + plotH / 2}
          textAnchor="middle" fontSize={11} fill="var(--muted)"
          transform={`rotate(-90 14 ${pad.top + plotH / 2})`}
        >
          {yLabel}
        </text>
      </svg>
      {caption && (
        <figcaption className="mt-2 text-xs text-muted">{caption}</figcaption>
      )}
    </figure>
  );
}
