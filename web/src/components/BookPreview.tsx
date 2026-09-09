"use client";

import { useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import type { Book } from "../lib/types";
import { fa, faNumber } from "../lib/format";

interface Props {
  book: Book;
  /** مستطیل کارتی که نشانگر رویش است */
  anchor: DOMRect;
  /** ژانرهایی که با انتخاب‌های کاربر مشترک‌اند و برجسته می‌شوند */
  highlight?: Set<string>;
}

export const PREVIEW_WIDTH = 300;
const GAP = 10;

/**
 * پیش‌نمایش شناور کتاب.
 *
 * دو نکته‌ی فنی که رعایتشان لازم است:
 *
 * ۱. پنل با portal مستقیم به body منتقل می‌شود. کارت‌ها هنگام هاور یک
 *    transform می‌گیرند و در CSS هر عنصری که transform داشته باشد برای
 *    فرزندان position:fixed خود مبدأ مختصات می‌شود؛ در نتیجه پنل به‌جای
 *    صفحه، نسبت به کارت جای می‌گرفت و از کادر بیرون می‌زد. انتقال به body
 *    این وابستگی را به‌کلی حذف می‌کند و مشکل لایه‌بندی را هم حل می‌کند.
 *
 * ۲. جای پنل پس از رندر و با اندازه‌گیری ارتفاع واقعی محاسبه می‌شود، نه با
 *    حدس زدن. ارتفاع به تعداد برچسب‌ها و طول عنوان بستگی دارد و ثابت نیست.
 *    تا وقتی اندازه‌گیری انجام نشده، پنل نامرئی است تا پرش دیده نشود.
 */
export default function BookPreview({ book, anchor, highlight }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ top: number; left: number } | null>(
    null);

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;

    const { width, height } = node.getBoundingClientRect();

    let left = anchor.left + anchor.width / 2 - width / 2;
    left = Math.min(Math.max(GAP, left), window.innerWidth - width - GAP);

    // ترجیح: بالای کارت. اگر جا نبود، پایین. اگر باز هم نبود، چسبیده به لبه.
    let top = anchor.top - height - GAP;
    if (top < GAP) top = anchor.bottom + GAP;
    if (top + height > window.innerHeight - GAP) {
      top = Math.max(GAP, window.innerHeight - height - GAP);
    }

    setPosition({ top, left });
  }, [anchor, book]);

  return createPortal(
    <div
      ref={ref}
      role="tooltip"
      style={{
        width: PREVIEW_WIDTH,
        top: position?.top ?? 0,
        left: position?.left ?? 0,
        opacity: position ? 1 : 0,
      }}
      className="pointer-events-none fixed z-[999] rounded-xl border border-line bg-panel p-3 text-right shadow-2xl transition-opacity duration-150"
      dir="rtl"
    >
      <p className="text-sm font-bold leading-6 text-ink">{book.t}</p>
      <p className="mt-0.5 text-xs text-ink-soft">{book.a}</p>

      <div className="mt-2.5 space-y-1 text-[11px] text-ink-soft">
        {book.y !== null && (
          <p>
            سال انتشار{" "}
            <span className="ltr inline-block font-semibold text-ink">
              {fa(book.y)}
            </span>
          </p>
        )}
        <p>
          امتیاز Goodreads{" "}
          <span className="ltr inline-block font-semibold text-ink">
            {fa(book.r.toFixed(2))}
          </span>{" "}
          از {fa(5)}
        </p>
        <p>
          <span className="ltr inline-block font-semibold text-ink">
            {faNumber(book.n)}
          </span>{" "}
          خواننده در این مجموعه‌داده
        </p>
      </div>

      {book.g.length > 0 && (
        <div className="mt-3 border-t border-line pt-2.5">
          <p className="text-[10px] text-muted">ژانر و برچسب‌ها</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {book.g.map((genre) => {
              const shared = highlight?.has(genre);
              return (
                <span
                  key={genre}
                  className={
                    "ltr rounded-md px-2 py-0.5 text-[11px] " +
                    (shared
                      ? "bg-brand font-semibold text-white"
                      : "bg-brand-soft text-ink-soft")
                  }
                >
                  {genre}
                </span>
              );
            })}
          </div>
          {highlight && book.g.some((g) => highlight.has(g)) && (
            <p className="mt-2 text-[10px] leading-4 text-muted">
              برچسب‌های پررنگ با کتاب‌های انتخابی شما مشترک‌اند — همان چیزی
              که مدل محتوامحور شباهت را بر پایه‌ی آن می‌سنجد.
            </p>
          )}
        </div>
      )}
    </div>,
    document.body,
  );
}
