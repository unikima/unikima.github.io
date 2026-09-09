import type { NextConfig } from "next";

/**
 * پیکربندی برای دو مسیر انتشار.
 *
 * حالت پیش‌فرض (Vercel یا اجرای محلی): هیچ تنظیم اضافه‌ای لازم نیست.
 *
 * حالت خروجی ایستا (GitHub Pages و هر میزبان فایل ساکن): با متغیر محیطی
 * STATIC_EXPORT=1 فعال می‌شود و کل سایت به فایل‌های HTML ساده تبدیل
 * می‌گردد. چون همه‌ی صفحه‌های این پروژه از پیش ساخته می‌شوند و هیچ کد
 * سمت سروری وجود ندارد، این تبدیل بدون از دست رفتن هیچ قابلیتی انجام
 * می‌شود.
 *
 * روی GitHub Pages سایت زیر مسیر /<نام مخزن>/ سرو می‌شود، بنابراین
 * basePath هم باید تنظیم شود تا نشانی فایل‌های مدل درست بماند.
 */
const isStaticExport = process.env.STATIC_EXPORT === "1";
const basePath = process.env.BASE_PATH ?? "";

const nextConfig: NextConfig = {
  // trailingSlash باعث می‌شود هر صفحه به‌شکل about/index.html ساخته شود.
  // میزبان‌های فایل ساکن این ساختار را مطمئن‌تر از about.html سرو می‌کنند.
  ...(isStaticExport
    ? { output: "export" as const, trailingSlash: true }
    : {}),
  ...(basePath ? { basePath, assetPrefix: basePath } : {}),

  // جلد کتاب‌ها از دامنه‌های بیرونی می‌آیند و با تگ img ساده نمایش داده
  // می‌شوند، پس بهینه‌سازی تصویر Next نه لازم است و نه در خروجی ایستا
  // پشتیبانی می‌شود.
  images: { unoptimized: true },

  // تا کد سمت کارخواه بتواند نشانی فایل‌های مدل را درست بسازد
  env: { NEXT_PUBLIC_BASE_PATH: basePath },
};

export default nextConfig;
