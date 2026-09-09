"use client";

import { useMemo, useState } from "react";
import type { Book } from "../lib/types";
import { fa } from "../lib/format";

interface Props {
  books: Book[];
  popularity: Float32Array;
  onPick: (item: number) => void;
  excluded: Set<number>;
  placeholder?: string;
}

/**
 * جستجوی کتاب بر اساس عنوان یا نویسنده.
 *
 * جستجو روی هر ده هزار کتاب و در هر بار تایپ انجام می‌شود، بنابراین باید
 * ارزان باشد: تطبیق ساده‌ی زیررشته روی متن از پیش کوچک‌شده، و توقف پس از
 * یافتن پنجاه نتیجه. نتایج بر اساس تعداد خواننده مرتب می‌شوند تا
 * شناخته‌شده‌ترین کتاب‌ها بالا بیایند.
 */
export default function BookSearch({
  books, popularity, onPick, excluded, placeholder = "نام کتاب یا نویسنده...",
}: Props) {
  const [query, setQuery] = useState("");

  const haystack = useMemo(
    () => books.map((b) => `${b.t} ${b.a}`.toLowerCase()),
    [books],
  );

  const results = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (q.length < 2) return [];

    const found: number[] = [];
    for (let i = 0; i < haystack.length && found.length < 60; i++) {
      if (!excluded.has(i) && haystack[i].includes(q)) found.push(i);
    }
    return found
      .sort((a, b) => popularity[b] - popularity[a])
      .slice(0, 12);
  }, [query, haystack, excluded, popularity]);

  return (
    <div className="relative">
      <input
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-xl border border-line bg-panel px-4 py-3 text-sm outline-none placeholder:text-muted focus:border-brand"
      />

      {query.trim().length >= 2 && (
        <div className="thin-scroll absolute z-10 mt-2 max-h-80 w-full overflow-y-auto rounded-xl border border-line bg-panel shadow-lg">
          {results.length === 0 ? (
            <p className="px-4 py-3 text-sm text-muted">کتابی پیدا نشد.</p>
          ) : (
            results.map((i) => (
              <button
                key={i}
                type="button"
                onClick={() => {
                  onPick(i);
                  setQuery("");
                }}
                className="flex w-full items-baseline justify-between gap-3 border-b border-line px-4 py-2.5 text-right last:border-b-0 hover:bg-brand-soft"
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm font-medium">
                    {books[i].t}
                  </span>
                  <span className="block truncate text-xs text-ink-soft">
                    {books[i].a}
                  </span>
                </span>
                <span className="ltr shrink-0 text-xs text-muted">
                  {fa(popularity[i].toLocaleString("en-US"))} خواننده
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
