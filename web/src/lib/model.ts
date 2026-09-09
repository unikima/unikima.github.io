/**
 * موتور توصیه‌گر — کاملاً در مرورگر اجرا می‌شود.
 *
 * پس از آموزش در پایتون، آنچه برای تولید پیشنهاد لازم است چند ماتریس عددی
 * بیش نیست. این ماژول آن ماتریس‌ها را یک‌بار بارگذاری می‌کند و سپس هر
 * درخواست توصیه فقط چند ضرب بردار در ماتریس است که در چند میلی‌ثانیه انجام
 * می‌شود. به همین دلیل این برنامه به هیچ سرور پشتیبانی نیاز ندارد.
 */
import type { Book, ModelMeta, ModelName, Recommendation, Selection }
  from "./types";

export interface LoadedModel {
  meta: ModelMeta;
  books: Book[];
  /** ماتریس عامل‌های پنهان PureSVD، به شکل تخت: [item * k + factor] */
  svdV: Float32Array;
  itemSimIdx: Uint16Array;
  itemSimVal: Float32Array;
  contSimIdx: Uint16Array;
  contSimVal: Float32Array;
  popularity: Float32Array;
}

// روی میزبان‌هایی مانند GitHub Pages سایت زیر یک زیرمسیر سرو می‌شود، پس
// نشانی فایل‌های مدل باید همان پیشوند را داشته باشد. در حالت عادی این
// متغیر خالی است و نشانی‌ها از ریشه خوانده می‌شوند.
const BASE = `${process.env.NEXT_PUBLIC_BASE_PATH ?? ""}/model`;

async function fetchBinary(name: string): Promise<ArrayBuffer> {
  const response = await fetch(`${BASE}/${name}`);
  if (!response.ok) throw new Error(`بارگذاری ${name} ناموفق بود`);
  return response.arrayBuffer();
}

let cache: Promise<LoadedModel> | null = null;

export function loadModel(): Promise<LoadedModel> {
  if (cache) return cache;

  cache = (async () => {
    const [meta, books, svd, isIdx, isVal, csIdx, csVal, pop] =
      await Promise.all([
        fetch(`${BASE}/meta.json`).then((r) => r.json() as Promise<ModelMeta>),
        fetch(`${BASE}/books.json`).then((r) => r.json() as Promise<Book[]>),
        fetchBinary("svd_v.bin"),
        fetchBinary("item_sim_idx.bin"),
        fetchBinary("item_sim_val.bin"),
        fetchBinary("cont_sim_idx.bin"),
        fetchBinary("cont_sim_val.bin"),
        fetchBinary("popularity.bin"),
      ]);

    return {
      meta,
      books,
      svdV: new Float32Array(svd),
      itemSimIdx: new Uint16Array(isIdx),
      itemSimVal: new Float32Array(isVal),
      contSimIdx: new Uint16Array(csIdx),
      contSimVal: new Float32Array(csVal),
      popularity: new Float32Array(pop),
    };
  })();

  return cache;
}

// ------------------------------------------------------------------ کمکی‌ها

/** استانداردسازی نمره: میانگین صفر و انحراف معیار یک */
function zScore(x: Float32Array): Float32Array {
  let sum = 0;
  for (let i = 0; i < x.length; i++) sum += x[i];
  const mean = sum / x.length;

  let variance = 0;
  for (let i = 0; i < x.length; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  const sd = Math.sqrt(variance / x.length);
  const out = new Float32Array(x.length);
  if (sd < 1e-8) return out;
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - mean) / sd;
  return out;
}

/**
 * وزن هر انتخاب: امتیاز نسبت به میانگین کلی مرکزی می‌شود تا کتابی که کاربر
 * نپسندیده وزن منفی بگیرد و پروفایل را از خود دور کند. اگر همه‌ی امتیازها
 * دقیقاً برابر میانگین باشند، وزن یکنواخت استفاده می‌شود.
 */
function weightsOf(selections: Selection[], globalMean: number): number[] {
  const w = selections.map((s) => s.rating - globalMean);
  return w.some((v) => Math.abs(v) > 1e-6) ? w : selections.map(() => 1);
}

// --------------------------------------------------------------- مدل‌ها

/**
 * امتیاز مشارکتی با «تازاندن» کاربر تازه در فضای PureSVD.
 *
 * بردار انتخاب کاربر در ماتریس V ضرب می‌شود تا نمایش او در فضای پنهان به
 * دست آید، سپس همان نمایش در ترانهاده‌ی V ضرب می‌شود تا امتیاز همه‌ی
 * کتاب‌ها حاصل شود. این کار معادل ساختن سطر کاربر در ماتریس بازسازی‌شده
 * است، بدون آنکه لازم باشد مدل دوباره آموزش ببیند.
 */
export function collaborativeScores(
  model: LoadedModel, selections: Selection[],
): Float32Array {
  const { svdV, meta } = model;
  const k = meta.svdFactors;
  const n = meta.nItems;

  const userFactors = new Float32Array(k);
  for (const { item, rating } of selections) {
    const base = item * k;
    for (let f = 0; f < k; f++) userFactors[f] += rating * svdV[base + f];
  }

  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const base = i * k;
    let dot = 0;
    for (let f = 0; f < k; f++) dot += userFactors[f] * svdV[base + f];
    scores[i] = dot;
  }
  return scores;
}

/** جمع وزن‌دار شباهت‌ها روی یک فهرست همسایه‌ی فشرده */
function neighbourScores(
  idx: Uint16Array, val: Float32Array, topK: number, nItems: number,
  selections: Selection[], weights: number[],
): Float32Array {
  const scores = new Float32Array(nItems);
  selections.forEach(({ item }, s) => {
    const w = weights[s];
    const base = item * topK;
    for (let t = 0; t < topK; t++) {
      scores[idx[base + t]] += val[base + t] * w;
    }
  });
  return scores;
}

export function contentScores(
  model: LoadedModel, selections: Selection[],
): Float32Array {
  const weights = weightsOf(selections, model.meta.globalMean);
  return neighbourScores(model.contSimIdx, model.contSimVal, model.meta.topK,
    model.meta.nItems, selections, weights);
}

export function itemKnnScores(
  model: LoadedModel, selections: Selection[],
): Float32Array {
  const weights = weightsOf(selections, model.meta.globalMean);
  return neighbourScores(model.itemSimIdx, model.itemSimVal, model.meta.topK,
    model.meta.nItems, selections, weights);
}

/**
 * مدل ترکیبی — همان رابطه‌ی به‌کاررفته در پایان‌نامه.
 * استانداردسازی پیش از جمع ضروری است، چون خروجی دو مدل مقیاس‌های کاملاً
 * متفاوتی دارند و جمع مستقیم، وزن را به مدلی می‌دهد که عدد بزرگ‌تری تولید
 * می‌کند نه مدلی که اطلاعات بیشتری می‌آورد.
 */
export function hybridScores(
  model: LoadedModel, selections: Selection[], alpha = model.meta.alpha,
): Float32Array {
  const collab = zScore(collaborativeScores(model, selections));
  const content = zScore(contentScores(model, selections));
  const out = new Float32Array(collab.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = alpha * collab[i] + (1 - alpha) * content[i];
  }
  return out;
}

export function popularityScores(model: LoadedModel): Float32Array {
  return model.popularity;
}

// ------------------------------------------------------------- توصیه نهایی

/**
 * دلیل هر پیشنهاد از ماتریس شباهت آیتم‌محور استخراج می‌شود: سهم هر کتابِ
 * انتخاب‌شده در امتیاز کتاب پیشنهادی محاسبه و بزرگ‌ترین سهم‌ها برگردانده
 * می‌شود. مدل تجزیه ماتریس چنین قابلیتی ندارد، چون امتیازش از ضرب دو بردار
 * پنهان می‌آید و به کتاب مشخصی قابل نسبت دادن نیست.
 */
function explain(
  model: LoadedModel, target: number, selections: Selection[],
  weights: number[], limit = 2,
): { item: number; weight: number }[] {
  const { itemSimIdx, itemSimVal, meta } = model;
  const contributions: { item: number; weight: number }[] = [];

  selections.forEach(({ item }, s) => {
    const base = item * meta.topK;
    for (let t = 0; t < meta.topK; t++) {
      if (itemSimIdx[base + t] === target) {
        const c = itemSimVal[base + t] * weights[s];
        if (c > 0) contributions.push({ item, weight: c });
        break;
      }
    }
  });

  return contributions.sort((a, b) => b.weight - a.weight).slice(0, limit);
}

export function recommend(
  model: LoadedModel,
  selections: Selection[],
  modelName: ModelName = "hybrid",
  count = 12,
  alpha = model.meta.alpha,
): Recommendation[] {
  if (selections.length === 0) return [];

  // نمره‌های دو جزء جداگانه نگه داشته می‌شوند تا بتوان سهم هرکدام را در
  // امتیاز نهایی به کاربر نشان داد.
  let collab: Float32Array | null = null;
  let content: Float32Array | null = null;
  let scores: Float32Array;

  switch (modelName) {
    case "collaborative":
      scores = collaborativeScores(model, selections);
      break;
    case "content":
      scores = contentScores(model, selections);
      break;
    case "itemKnn":
      scores = itemKnnScores(model, selections);
      break;
    case "popularity":
      scores = popularityScores(model);
      break;
    default: {
      collab = zScore(collaborativeScores(model, selections));
      content = zScore(contentScores(model, selections));
      scores = new Float32Array(collab.length);
      for (let i = 0; i < scores.length; i++) {
        scores[i] = alpha * collab[i] + (1 - alpha) * content[i];
      }
    }
  }

  const chosen = new Set(selections.map((s) => s.item));
  const weights = weightsOf(selections, model.meta.globalMean);

  const ranked: { item: number; score: number }[] = [];
  for (let i = 0; i < scores.length; i++) {
    if (!chosen.has(i)) ranked.push({ item: i, score: scores[i] });
  }
  ranked.sort((a, b) => b.score - a.score);

  const top = ranked.slice(0, count);

  /*
   * «میزان تطابق» نسبت به قوی‌ترین پیشنهاد همین فهرست سنجیده می‌شود، نه
   * به‌صورت مطلق. نمره‌ی خام مدل مقیاس معناداری ندارد (می‌تواند منفی یا
   * بزرگ‌تر از ده باشد) و نمایش مستقیم آن گمراه‌کننده است، چون ذهن آن را
   * با امتیاز یک تا پنج اشتباه می‌گیرد. کف بازه روی ۰٫۵۵ گذاشته شده تا
   * نوارِ آخرین پیشنهاد هم قابل دیدن بماند.
   */
  const best = top.length ? top[0].score : 0;
  const worst = top.length ? top[top.length - 1].score : 0;
  const span = best - worst || 1;

  return top.map(({ item, score }) => {
    const entry: Recommendation = {
      item,
      score,
      match: 0.55 + 0.45 * ((score - worst) / span),
      reasons: explain(model, item, selections, weights),
    };

    if (collab && content) {
      // سهم نسبی هر جزء، بر پایه‌ی قدرمطلق مشارکت آن در امتیاز نهایی
      const a = Math.abs(alpha * collab[item]);
      const b = Math.abs((1 - alpha) * content[item]);
      entry.collaborativeShare = a + b > 1e-9 ? a / (a + b) : alpha;
    }
    return entry;
  });
}

/**
 * پنج پیشنهاد برتر چند مدل، برای نمایش کنار هم.
 *
 * این نما مهم‌ترین یافته‌ی پژوهش را در یک نگاه نشان می‌دهد: مدل تجزیه
 * ماتریس با وجود کمترین خطای پیش‌بینی، در همین جدول کتاب‌هایی را بالا
 * می‌آورد که آشکارا بی‌ربط‌اند.
 */
export function compareModels(
  model: LoadedModel,
  selections: Selection[],
  models: ModelName[],
  count = 5,
): Record<string, Recommendation[]> {
  const out: Record<string, Recommendation[]> = {};
  for (const name of models) {
    out[name] = recommend(model, selections, name, count);
  }
  return out;
}

/** k کتاب شبیه به یک کتاب مشخص، مستقیماً از ماتریس شباهت آیتم‌محور */
export function similarBooks(
  model: LoadedModel, item: number, count = 12,
): { item: number; score: number }[] {
  const { itemSimIdx, itemSimVal, meta } = model;
  const base = item * meta.topK;
  const out: { item: number; score: number }[] = [];
  for (let t = 0; t < meta.topK && out.length < count; t++) {
    const value = itemSimVal[base + t];
    if (value > 0) out.push({ item: itemSimIdx[base + t], score: value });
  }
  return out;
}

/** پرخواننده‌ترین کتاب‌ها — برای پیشنهاد اولیه به کاربری که هنوز چیزی نگفته */
export function mostPopular(model: LoadedModel, count = 12): number[] {
  const order = Array.from(model.popularity.keys());
  order.sort((a, b) => model.popularity[b] - model.popularity[a]);
  return order.slice(0, count);
}
