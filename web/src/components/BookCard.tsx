"use client";

import { useEffect, useRef, useState } from "react";
import type { Book } from "../lib/types";
import { fa } from "../lib/format";
import BookPreview from "./BookPreview";

interface Props {
  book: Book;
  rank?: number;
  /** میزان تطابق نسبی، بین ۰ تا ۱ */
  match?: number;
  /** نمره خام مدل — ریز و کم‌رنگ نمایش داده می‌شود */
  rawScore?: number;
  /** سهم بخش مشارکتی از امتیاز، بین ۰ تا ۱ */
  collaborativeShare?: number;
  reasons?: string[];
  onClick?: () => void;
  selected?: boolean;
  compact?: boolean;
  /** ژانرهای مشترک با انتخاب‌های کاربر، برای برجسته‌سازی در پیش‌نمایش */
  highlightGenres?: Set<string>;
}

const PREVIEW_DELAY_MS = 260;

/**
 * جلد کتاب، با زنجیره‌ی جایگزین.
 *
 * حدود یک‌سوم کتاب‌های این مجموعه‌داده در Goodreads جلد ندارند. برای آن‌ها
 * ابتدا Open Library بر اساس شناسه‌ی ده‌رقمی امتحان می‌شود، و اگر آنجا هم
 * تصویری نبود یا شبکه در دسترس نبود، یک جلد طراحی‌شده با عنوان و نویسنده
 * نمایش داده می‌شود. پارامتر default=false باعث می‌شود Open Library به‌جای
 * برگرداندن یک تصویر خالی، خطای واقعی بدهد تا بتوان تشخیصش داد.
 */
function Cover({ book }: { book: Book }) {
  const sources = [
    book.img,
    book.isbn
      ? `https://covers.openlibrary.org/b/isbn/${book.isbn}-M.jpg?default=false`
      : "",
  ].filter(Boolean);

  const [attempt, setAttempt] = useState(0);
  const src = sources[attempt];

  if (!src) {
    return (
      <div className="flex aspect-2/3 w-full flex-col justify-between rounded-lg border border-line bg-gradient-to-br from-brand-soft to-panel p-2.5 text-right">
        {/* فضای خالی سمت راست، تا نشان رتبه روی عنوان نیفتد */}
        <span className="line-clamp-4 pr-7 text-[11px] font-semibold leading-5 text-ink-soft">
          {book.t}
        </span>
        <span className="line-clamp-1 text-[10px] text-muted">{book.a}</span>
      </div>
    );
  }

  return (
    // eslint-disable-next-line @next/next/no-img-element
    <img
      key={src}
      src={src}
      alt={book.t}
      loading="lazy"
      onError={() => setAttempt((n) => n + 1)}
      className="aspect-2/3 w-full rounded-lg border border-line bg-brand-soft/30 object-cover"
    />
  );
}

/** نوار میزان تطابق — جایگزین نمایش عدد خام */
function MatchBar({ match, rawScore }: { match: number; rawScore?: number }) {
  const percent = Math.round(match * 100);
  return (
    <div
      className="mt-2.5"
      title={
        rawScore === undefined ? undefined
          : `نمره خام مدل: ${rawScore.toFixed(3)} — مقیاس ۱ تا ۵ نیست و ` +
            "فقط ترتیب آن معنا دارد"
      }
    >
      <div className="flex items-baseline justify-between text-[11px]">
        <span className="text-ink-soft">میزان تطابق</span>
        <span className="ltr font-semibold text-brand">{fa(percent)}٪</span>
      </div>
      <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-line">
        <div
          className="h-full rounded-full bg-brand transition-[width] duration-500"
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

/**
 * تفکیک سهم دو مدل در امتیاز نهایی.
 * عمداً فقط یک نوار نازک است و برچسب عددی ندارد: اگر هر کارت دو نوار با
 * برچسب کامل داشته باشد، شبکه‌ی پیشنهادها ناخوانا می‌شود. توضیح کامل با
 * نگه داشتن نشانگر روی نوار دیده می‌شود.
 */
function ShareBar({ share }: { share: number }) {
  const collab = Math.round(share * 100);
  return (
    <div
      className="mt-2 flex items-center gap-1.5"
      title={`سهم مدل مشارکتی ${collab}٪ · سهم مدل محتوامحور ${100 - collab}٪`}
    >
      <span className="flex h-1 flex-1 overflow-hidden rounded-full">
        <span
          className="h-full bg-brand transition-[width] duration-500"
          style={{ width: `${collab}%` }}
        />
        <span
          className="h-full bg-accent transition-[width] duration-500"
          style={{ width: `${100 - collab}%` }}
        />
      </span>
      {/* یک عدد بس است: سهم دوم مکمل آن است و ذکر هر دو فقط ابهام می‌سازد */}
      <span className="shrink-0 text-[10px] whitespace-nowrap text-muted">
        <span className="ltr inline-block">{fa(collab)}٪</span> مشارکتی
      </span>
    </div>
  );
}

export default function BookCard({
  book, rank, match, rawScore, collaborativeShare, reasons, onClick,
  selected, compact, highlightGenres,
}: Props) {
  const Wrapper = onClick ? "button" : "div";

  // مستطیل کارت نگه داشته می‌شود، نه مختصات نهایی پنل. جای دقیق پنل را خود
  // آن پس از اندازه‌گیری ارتفاع واقعی‌اش تعیین می‌کند.
  const [anchor, setAnchor] = useState<DOMRect | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const openPreview = (event: React.MouseEvent<HTMLElement>) => {
    // روی صفحه‌های لمسی «نگه داشتن نشانگر» وجود ندارد؛ مرورگر هنگام لمس
    // یک mouseenter ساختگی می‌فرستد و پیش‌نمایش درست وسط کاری که کاربر
    // می‌خواهد بکند باز می‌شود و جلوی دیدش را می‌گیرد. پس فقط روی
    // دستگاه‌هایی که واقعاً نشانگر دارند باز می‌شود.
    if (!window.matchMedia("(hover: hover) and (pointer: fine)").matches) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (timer.current) clearTimeout(timer.current);
    // تأخیر کوتاه تا وقتی نشانگر فقط از روی کارت رد می‌شود چیزی باز نشود
    timer.current = setTimeout(() => setAnchor(rect), PREVIEW_DELAY_MS);
  };

  const closePreview = () => {
    if (timer.current) clearTimeout(timer.current);
    setAnchor(null);
  };

  // اگر کاربر هنگام باز بودن پیش‌نمایش صفحه را بپیماید، پنل باید بسته شود
  // وگرنه در جای قبلی معلق می‌ماند.
  useEffect(() => {
    if (!anchor) return;
    const close = () => setAnchor(null);
    window.addEventListener("scroll", close, { passive: true });
    window.addEventListener("resize", close);
    return () => {
      window.removeEventListener("scroll", close);
      window.removeEventListener("resize", close);
    };
  }, [anchor]);

  return (
    <Wrapper
      onClick={onClick}
      onMouseEnter={openPreview}
      onMouseLeave={closePreview}
      onFocus={(e) => setAnchor(e.currentTarget.getBoundingClientRect())}
      onBlur={closePreview}
      type={onClick ? "button" : undefined}
      className={
        "group flex h-full flex-col rounded-xl border p-3 text-right transition-all duration-200 " +
        (selected
          ? "border-brand bg-brand-soft shadow-sm"
          : "border-line bg-panel " +
            (onClick
              ? "hover:-translate-y-0.5 hover:border-brand hover:shadow-md"
              : "hover:shadow-sm"))
      }
    >
      <div className="relative overflow-hidden rounded-lg">
        <Cover book={book} />
        {rank !== undefined && (
          <span className="absolute top-1.5 right-1.5 rounded-md bg-brand px-2 py-0.5 text-xs font-semibold text-white shadow-sm">
            {fa(rank)}
          </span>
        )}
        {/*
          نشانه‌ی «این کارت را می‌شود کلیک کرد».
          بدون آن، کارت‌ها فقط با تغییر رنگ حاشیه واکنش نشان می‌دادند و
          کاربر نمی‌فهمید کلیک کردنشان کاری می‌کند یا نه. روی صفحه‌های
          لمسی که hover ندارند همیشه دیده می‌شود.
        */}
        {onClick && !selected && (
          <span className="pointer-events-none absolute inset-x-1.5 bottom-1.5 flex items-center justify-center gap-1 rounded-md bg-brand/95 py-1 text-[11px] font-semibold text-white opacity-0 shadow-sm transition-opacity duration-150 group-hover:opacity-100 group-focus-visible:opacity-100 [@media(hover:none)]:opacity-100">
            <span aria-hidden="true">＋</span> افزودن
          </span>
        )}
      </div>

      <p className="mt-2.5 line-clamp-3 text-sm font-semibold leading-6">
        {book.t}
      </p>
      <p className="mt-1 line-clamp-2 text-xs leading-5 text-ink-soft">
        {book.a}
      </p>

      {!compact && match !== undefined && (
        <MatchBar match={match} rawScore={rawScore} />
      )}

      {!compact && collaborativeShare !== undefined && (
        <ShareBar share={collaborativeShare} />
      )}

      {reasons && reasons.length > 0 && (
        <p className="mt-2 border-t border-line pt-2 text-xs leading-6 text-ink-soft">
          چون خوانده‌اید: {reasons.join("، ")}
        </p>
      )}

      {anchor && (
        <BookPreview book={book} anchor={anchor} highlight={highlightGenres} />
      )}
    </Wrapper>
  );
}
