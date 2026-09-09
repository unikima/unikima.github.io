export const metadata = { title: "درباره پروژه | سیستم توصیه‌گر کتاب" };

const MODELS = [
  ["محبوبیت", "پرخواننده‌ترین کتاب‌ها", "ساده و پایدار", "بدون شخصی‌سازی"],
  ["شباهت آیتم‌محور", "شباهت کسینوسی میان کتاب‌ها", "قابل تفسیر",
    "ناتوان برای کتاب تازه"],
  ["تجزیه ماتریس", "عامل‌های پنهان با گرادیان کاهشی",
    "کمترین خطای پیش‌بینی", "ضعیف در رتبه‌بندی"],
  ["PureSVD", "تجزیه ماتریس کامل با صفر برای خانه‌های خالی",
    "بهترین رتبه‌بندی مشارکتی", "خطای پیش‌بینی بالا"],
  ["محتوامحور", "شباهت متنی عنوان، نویسنده و برچسب",
    "حل شروع سرد و پوشش بالا", "نادیده‌گرفتن سلیقه جمعی"],
  ["ترکیبی", "جمع وزن‌دار دو مدل بالا", "بهترین عملکرد کلی",
    "پیچیدگی بیشتر"],
];

export default function AboutPage() {
  return (
    <article className="max-w-3xl space-y-8">
      <header>
        <h1 className="text-2xl font-bold">درباره پروژه</h1>
        <p className="mt-2 text-sm leading-7 text-ink-soft">
          پروژه کارشناسی مهندسی کامپیوتر — دانشگاه صنعتی همدان. طراحی و
          پیاده‌سازی سیستم توصیه‌گر کتاب بر اساس سلیقه کاربر با رویکرد
          فیلترینگ مشارکتی.
        </p>
      </header>

      <section>
        <h2 className="text-lg font-bold">معماری سامانه</h2>
        <p className="mt-2 text-sm leading-7 text-ink-soft">
          سامانه دو بخش دارد. بخش نخست با پایتون نوشته شده و کار آموزش و
          ارزیابی مدل‌ها را انجام می‌دهد. بخش دوم همین برنامه‌ی وب است که با
          Next.js و TypeScript ساخته شده.
        </p>
        <p className="mt-3 text-sm leading-7 text-ink-soft">
          نکته‌ی طراحی این است که پس از آموزش، آنچه برای تولید پیشنهاد لازم
          است چند ماتریس عددی بیش نیست. این ماتریس‌ها به فایل‌های دودویی
          فشرده تبدیل و همراه برنامه ارسال می‌شوند، و محاسبه‌ی توصیه کاملاً
          در مرورگر شما انجام می‌گیرد. نتیجه آنکه سامانه به هیچ سرور پشتیبانی
          نیاز ندارد، پاسخ آنی است، و هیچ داده‌ای از شما جایی ارسال نمی‌شود.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-bold">مدل‌های پیاده‌سازی‌شده</h2>
        <div className="thin-scroll mt-3 overflow-x-auto rounded-xl border border-line">
          <table className="w-full min-w-[600px] border-collapse text-sm">
            <thead>
              <tr className="bg-brand-soft/60">
                {["مدل", "ایده اصلی", "نقطه قوت", "نقطه ضعف"].map((h) => (
                  <th key={h} className="px-3 py-2.5 text-right font-semibold">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {MODELS.map((row) => (
                <tr key={row[0]} className="border-t border-line">
                  {row.map((cell, i) => (
                    <td
                      key={i}
                      className={
                        "px-3 py-2.5 " +
                        (i === 0 ? "font-semibold" : "text-ink-soft")
                      }
                    >
                      {cell}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-bold">یافته‌ی محوری</h2>
        <p className="mt-2 text-sm leading-7 text-ink-soft">
          مدل تجزیه ماتریس کمترین خطای پیش‌بینی را دارد، ولی وقتی از آن
          می‌خواهیم ده کتاب برتر را نام ببرد، عملکردی نزدیک به تصادفی نشان
          می‌دهد. علتش این است که این مدل فقط روی خانه‌های پرشده‌ی ماتریس
          آموزش می‌بیند و هرگز نمی‌آموزد که «خوانده‌نشدن» خودش یک سیگنال
          است؛ در نتیجه کتاب‌های کم‌خواننده‌ی پرامتیاز را بالا می‌آورد.
        </p>
        <p className="mt-3 text-sm leading-7 text-ink-soft">
          یعنی پیش‌بینی امتیاز و رتبه‌بندی دو مسئله‌ی متفاوت‌اند و بهترین
          مدل برای یکی لزوماً بهترین برای دیگری نیست. مدل PureSVD که
          خانه‌های خالی را صفر در نظر می‌گیرد، دقیقاً برای پر کردن همین خلأ
          به پروژه اضافه شد.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-bold">داده‌ها</h2>
        <p className="mt-2 text-sm leading-7 text-ink-soft">
          مجموعه‌داده‌ی متن‌باز goodbooks-10k شامل نزدیک به شش میلیون امتیاز
          واقعی کاربران به ده هزار کتاب. این مجموعه انگلیسی‌زبان است، چون
          هیچ سکوی فارسی ماتریس امتیاز کاربرانش را به‌صورت عمومی منتشر نکرده
          و بدون چنین ماتریسی فیلترینگ مشارکتی قابل پیاده‌سازی نیست.
          الگوریتم‌های به‌کاررفته مستقل از زبان‌اند.
        </p>
      </section>

      <section>
        <h2 className="text-lg font-bold">فناوری‌های استفاده‌شده</h2>
        <ul className="mt-2 list-inside list-disc space-y-1.5 text-sm leading-7 text-ink-soft">
          <li>آموزش و ارزیابی: Python، NumPy، SciPy، pandas، scikit-learn</li>
          <li>رابط کاربری: Next.js، TypeScript، React، Tailwind CSS</li>
          <li>نمودارها: SVG درون‌خطی، بدون کتابخانه بیرونی</li>
        </ul>
      </section>
    </article>
  );
}
