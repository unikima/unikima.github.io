"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const LINKS = [
  { href: "/", label: "پیشنهاد کتاب" },
  { href: "/similar", label: "کتاب‌های مشابه" },
  { href: "/results", label: "نتایج ارزیابی" },
  { href: "/about", label: "درباره پروژه" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-20 border-b border-line bg-surface/85 backdrop-blur">
      <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3">
        <Link href="/" className="flex items-center gap-2 text-base font-bold">
          <span aria-hidden>📚</span>
          <span>سیستم توصیه‌گر کتاب</span>
        </Link>

        <nav className="flex flex-wrap items-center gap-1 text-sm">
          {LINKS.map(({ href, label }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                aria-current={active ? "page" : undefined}
                className={
                  "rounded-lg px-3 py-1.5 transition-colors " +
                  (active
                    ? "bg-brand-soft font-semibold text-brand"
                    : "text-ink-soft hover:bg-brand-soft/60 hover:text-ink")
                }
              >
                {label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
