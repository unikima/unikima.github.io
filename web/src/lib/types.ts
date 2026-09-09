export interface Book {
  /** عنوان */
  t: string;
  /** نویسنده */
  a: string;
  /** نشانی تصویر جلد در Goodreads — برای یک‌سوم کتاب‌ها خالی است */
  img: string;
  /** شناسه ده‌رقمی کتاب، برای گرفتن جلد از Open Library */
  isbn: string;
  /** ژانرها — برچسب‌های محتوایی پالایش‌شده کاربران Goodreads */
  g: string[];
  /** سال انتشار */
  y: number | null;
  /** میانگین امتیاز در Goodreads */
  r: number;
  /** تعداد کاربران این مجموعه‌داده که به کتاب امتیاز داده‌اند */
  n: number;
}

export interface ModelMeta {
  nItems: number;
  nUsers: number;
  nRatings: number;
  svdFactors: number;
  topK: number;
  alpha: number;
  globalMean: number;
}

/** انتخاب کاربر: اندیس کتاب و امتیازی که به آن داده است */
export interface Selection {
  item: number;
  rating: number;
}

/** یک پیشنهاد، به همراه دلیل و تفکیک سهم دو مدل */
export interface Recommendation {
  item: number;
  /** نمره خام مدل — فقط برای مقایسه نسبی معنا دارد، مقیاس ۱ تا ۵ نیست */
  score: number;
  /** میزان تطابق نسبت به قوی‌ترین پیشنهاد همین فهرست، بین ۰ تا ۱ */
  match: number;
  /** سهم بخش مشارکتی از امتیاز نهایی، بین ۰ تا ۱ (فقط در مدل ترکیبی) */
  collaborativeShare?: number;
  /** کتاب‌هایی از انتخاب کاربر که بیشترین سهم را در این پیشنهاد داشته‌اند */
  reasons: { item: number; weight: number }[];
}

export type ModelName =
  | "hybrid"
  | "collaborative"
  | "content"
  | "itemKnn"
  | "popularity";

export interface ComparisonRow {
  model: string;
  RMSE: number;
  MAE: number;
  "Precision@10": number;
  "Recall@10": number;
  "NDCG@10": number;
  Coverage: number;
  "Precision@10 CI_low": number;
  "Precision@10 CI_high": number;
  "NDCG@10 CI_low": number;
  "NDCG@10 CI_high": number;
}

export interface EvaluationResults {
  comparison: ComparisonRow[];
  alphaSweep: { alpha: number; "Precision@10": number; "NDCG@10": number;
    Coverage: number }[];
  factorSweep: { n_factors: number; "Precision@10": number;
    "NDCG@10": number; Coverage: number }[];
  coldStart: Record<string, number>;
  trainingCurve: { epoch: number; train_rmse: number; valid_rmse: number }[];
}
