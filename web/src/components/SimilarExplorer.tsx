"use client";

import { useEffect, useMemo, useState } from "react";
import { loadModel, mostPopular, similarBooks, type LoadedModel }
  from "../lib/model";
import BookCard from "./BookCard";
import BookSearch from "./BookSearch";

const EMPTY = new Set<number>();

export default function SimilarExplorer() {
  const [model, setModel] = useState<LoadedModel | null>(null);
  const [seed, setSeed] = useState<number | null>(null);

  useEffect(() => {
    loadModel().then((m) => {
      setModel(m);
      setSeed(mostPopular(m, 1)[0]);
    });
  }, []);

  const neighbours = useMemo(
    () => (model && seed !== null ? similarBooks(model, seed, 12) : []),
    [model, seed],
  );

  if (!model) {
    return (
      <div className="flex flex-col items-center gap-3 py-24 text-sm text-muted">
        <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-brand" />
        در حال بارگذاری مدل‌ها ...
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <section>
        <h1 className="text-2xl font-bold">کتاب‌های مشابه</h1>
        <p className="mt-2 max-w-3xl text-sm leading-7 text-ink-soft">
          این بخش ماتریس شباهت آیتم‌محور را مستقیماً نشان می‌دهد. دو کتاب
          وقتی شبیه شمرده می‌شوند که کاربران یکسانی به آن‌ها امتیازهای مشابه
          داده باشند — نه به این دلیل که موضوعشان یکی است. بررسی چشمی این
          فهرست، ساده‌ترین راه سنجش کیفیت مدل است.
        </p>

        <div className="mt-5 max-w-xl">
          <BookSearch
            books={model.books}
            popularity={model.popularity}
            onPick={setSeed}
            excluded={EMPTY}
            placeholder="یک کتاب انتخاب کنید..."
          />
        </div>
      </section>

      {seed !== null && (
        <>
          <section>
            <h2 className="text-sm font-semibold text-ink-soft">کتاب مبنا</h2>
            <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6">
              <BookCard book={model.books[seed]} selected />
            </div>
          </section>

          <section>
            <h2 className="text-sm font-semibold text-ink-soft">
              شبیه‌ترین کتاب‌ها
            </h2>
            {neighbours.length === 0 ? (
              <p className="mt-3 text-sm text-muted">
                برای این کتاب همسایه معتبری یافت نشد.
              </p>
            ) : (
              <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                {neighbours.map((n, k) => (
                  <BookCard
                    key={n.item}
                    book={model.books[n.item]}
                    rank={k + 1}
                    match={n.score}
                    rawScore={n.score}
                    onClick={() => setSeed(n.item)}
                  />
                ))}
              </div>
            )}
            <p className="mt-4 text-xs text-muted">
              روی هر کتاب کلیک کنید تا همسایه‌های آن را ببینید.
            </p>
          </section>
        </>
      )}
    </div>
  );
}
