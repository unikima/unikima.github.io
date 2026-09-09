const PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹";

/** تبدیل ارقام لاتین به فارسی */
export function fa(value: string | number): string {
  return String(value).replace(/[0-9]/g, (d) => PERSIAN_DIGITS[+d]);
}

/** عدد با جداکننده هزارگان و ارقام فارسی */
export function faNumber(value: number): string {
  return fa(value.toLocaleString("en-US"));
}

/** عدد اعشاری با تعداد رقم مشخص و ارقام فارسی */
export function faFixed(value: number, digits = 4): string {
  return fa(value.toFixed(digits));
}

/** نام فارسی مدل‌ها، برای نمایش در جدول نتایج */
export const MODEL_LABELS: Record<string, string> = {
  "محبوبیت (پایه)": "محبوبیت (پایه)",
  "شباهت آیتم‌محور (KNN)": "شباهت آیتم‌محور",
  "محتوامحور (TF-IDF)": "محتوامحور",
  "تجزیه ماتریس (SVD)": "تجزیه ماتریس",
  PureSVD: "PureSVD",
  "ترکیبی (PureSVD + محتوا)": "ترکیبی",
};
