import type { Metadata } from "next";
import localFont from "next/font/local";
import Nav from "../components/Nav";
import "./globals.css";

/**
 * فونت وزیرمتن به‌صورت محلی همراه پروژه ارسال می‌شود، نه از سرویس بیرونی.
 * این کار سه مزیت دارد: صفحه بدون وابستگی به اینترنتِ بیرونی بالا می‌آید،
 * سرعت بارگذاری بیشتر است، و اگر روزی آن سرویس در دسترس نباشد ظاهر سایت
 * به‌هم نمی‌ریزد. نسخه‌ی متغیر فونت تنها ۱۱۰ کیلوبایت است و همه‌ی وزن‌ها
 * را پوشش می‌دهد.
 */
const vazir = localFont({
  src: "../fonts/Vazirmatn.woff2",
  weight: "100 900",
  variable: "--font-vazir",
  display: "swap",
});

const TITLE = "سیستم توصیه‌گر کتاب";
const DESCRIPTION =
  "سامانه پیشنهاد کتاب بر اساس سلیقه کاربر، با رویکرد ترکیبی فیلترینگ " +
  "مشارکتی و محتوامحور و توضیح دلیل هر پیشنهاد — پروژه کارشناسی مهندسی " +
  "کامپیوتر، دانشگاه صنعتی همدان";

/**
 * فراداده‌ی سایت.
 *
 * بخش openGraph تعیین می‌کند که وقتی نشانی سایت در تلگرام، واتس‌اپ، ایمیل
 * یا هر جای دیگری فرستاده می‌شود، به‌جای یک نشانی خشک، کارتی با عنوان،
 * توضیح و تصویر نمایش داده شود. تصویر آن به‌صورت خودکار از فایل
 * opengraph-image.png در همین پوشه برداشته می‌شود.
 *
 * metadataBase لازم است، چون نشانی تصویر در این کارت‌ها باید مطلق باشد و
 * بدون آن پیام‌رسان‌ها تصویر را پیدا نمی‌کنند.
 */
export const metadata: Metadata = {
  metadataBase: new URL("https://unikima.github.io"),
  title: { default: TITLE, template: `%s — ${TITLE}` },
  description: DESCRIPTION,
  applicationName: TITLE,
  keywords: [
    "سیستم توصیه‌گر", "پیشنهاد کتاب", "فیلترینگ مشارکتی", "یادگیری ماشین",
    "recommender system", "collaborative filtering", "matrix factorization",
  ],
  openGraph: {
    type: "website",
    locale: "fa_IR",
    siteName: TITLE,
    title: TITLE,
    description: DESCRIPTION,
    url: "/",
  },
  twitter: {
    card: "summary_large_image",
    title: TITLE,
    description: DESCRIPTION,
  },
};

export const viewport = {
  themeColor: "#2a78d6",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="fa" dir="rtl" className={`${vazir.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <Nav />
        <main className="mx-auto w-full max-w-6xl grow px-4 pb-24 pt-8">
          {children}
        </main>
      </body>
    </html>
  );
}
