import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 07. Есть ли внутри принятого каталога место, где аудиоэмбеддинги дают фору,
# и чего стоило бы завести отдельный срез холодных треков, расширив каталог.
DAY = 86400
TEST0 = 280 * DAY
N, K, SEED, BUDGET = 3000, 10000, 42, 2_000_000
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
EMB = pl.scan_parquet(os.path.join(raw, "embed_item_ids.parquet"))

pairs = (LF.select("uid", "item_id", "timestamp")
           .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
           .collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("uf"))
zero = (LF.select("uid", "played_ratio_pct").group_by("uid")
          .agg((pl.col("played_ratio_pct") == 0).mean().alias("z"))
          .collect(engine="streaming"))
ntr = (LF.filter(pl.col("timestamp") < TEST0).select("uid")
         .group_by("uid").len().collect(engine="streaming"))
pool = sorted((set(ufirst.filter(pl.col("uf") < TEST0)["uid"].to_list())
               - set(zero.filter(pl.col("z") > 0.5)["uid"].to_list()))
              & set(ntr.filter(pl.col("len") >= 5)["uid"].to_list()))
P = pl.Series("uid", pool)
rnd = random.Random(SEED)
samp = rnd.sample(pool, N)
S = pl.Series("uid", samp)
emb = EMB.select("item_id").unique().collect()

trm_pool = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
              .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
pop = (trm_pool.group_by("item_id").agg(pl.col("uid").n_unique().alias("users"))
               .sort(["users", "item_id"], descending=[True, False]))
CAT = pop.head(K)
C = pl.Series("item_id", CAT["item_id"].to_list())
THR = CAT["users"].min()
tm = trm_pool.filter(pl.col("uid").is_in(S.implode()))
sub_pop = (tm.group_by("item_id").agg(pl.col("uid").n_unique().alias("su")))
allp = (LF.select("uid", "item_id").unique().collect(engine="streaming")
          .filter(pl.col("uid").is_in(S.implode())))
rel = (LF.filter(pl.col("timestamp") >= TEST0)
         .select("uid", "item_id", "played_ratio_pct")
         .filter(pl.col("played_ratio_pct") >= 50)
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .filter(pl.col("t0") >= TEST0).filter(pl.col("uid").is_in(S.implode())))
BASE_PAIRS = allp.filter(pl.col("item_id").is_in(C.implode())).height
print("каталог %d треков (порог >= %d слушателей), пар в подвыборке %d из бюджета %d"
      % (K, THR, BASE_PAIRS, BUDGET))

# --- 1. молодость трека каталога против объёма обучающего сигнала ---
print("\n=== 1. «молодые» треки каталога: слабее ли у них коллаборативный сигнал? ===")
dbut = (pairs.group_by("item_id").agg(pl.col("t0").min().alias("f"))
             .with_columns((pl.col("f") // DAY).alias("d0")))
cd = (CAT.join(dbut.select("item_id", "d0"), on="item_id", how="left")
         .join(sub_pop, on="item_id", how="left").with_columns(pl.col("su").fill_null(0)))
print("  %-22s %7s %12s %12s %14s" % ("дебют", "треков", "польз. пула",
                                      "польз. подв.", "дней истории"))
for lab, lo, hi in [("сутки 0 (бэк-каталог)", 0, 0), ("сутки 1–150", 1, 150),
                    ("сутки 151–200", 151, 200), ("сутки 201–250", 201, 250),
                    ("сутки 251–279", 251, 279)]:
    b = cd.filter((pl.col("d0") >= lo) & (pl.col("d0") <= hi))
    if b.height == 0:
        continue
    print("  %-22s %7d %12.0f %12.0f %14.0f"
          % (lab, b.height, b["users"].median(), b["su"].median(), 280 - b["d0"].median()))
print("  видно, слабеет ли сигнал с молодостью: порог каталога гарантирует >= %d"
      % THR + " слушателей пула\n  независимо от даты дебюта")

# --- 2. цена отдельного среза холодных треков ---
print("\n=== 2. чего стоило бы завести срез холодных треков (расширить каталог) ===")
print("  %-18s %8s %10s %9s %10s %9s %10s"
      % ("полоса (слушат.)", "треков", "+пар", "бюджет", "новинок", "польз.", "эмбеддинг"))
for lo, hi in [(1, 4), (5, 19), (20, 49), (50, 118), (119, THR - 1)]:
    band = pop.filter((pl.col("users") >= lo) & (pl.col("users") <= hi))
    if band.height == 0:
        continue
    B = pl.Series("item_id", band["item_id"].to_list())
    add = allp.filter(pl.col("item_id").is_in(B.implode())).height
    rb = rel.filter(pl.col("item_id").is_in(B.implode()))
    e = band.select("item_id").join(emb, on="item_id", how="semi").height
    tot = BASE_PAIRS + add
    print("  %-18s %8d %10d %8s %10d %9d %9.1f %%"
          % ("%d–%d" % (lo, hi), band.height, add,
             "да" if tot <= BUDGET else "нет", rb.height, rb["uid"].n_unique(),
             100 * e / band.height))
print("  «+пар» — сколько пар добавит полоса к %d; бюджет задания %d"
      % (BASE_PAIRS, BUDGET))
print("  «новинок» — релевантных новинок test, которые стали бы достижимы")

# --- 3. сколько обучающих наблюдений у треков такой полосы ---
print("\n=== 3. обучающий сигнал в холодных полосах (пользователей подвыборки на трек) ===")
for lo, hi in [(1, 4), (5, 19), (20, 49), (50, 118), (119, THR - 1), (THR, 10 ** 9)]:
    band = pop.filter((pl.col("users") >= lo) & (pl.col("users") <= hi))
    if band.height == 0:
        continue
    sp = (band.select("item_id").join(sub_pop, on="item_id", how="left")
              .with_columns(pl.col("su").fill_null(0)))
    lab = "%d–%d" % (lo, hi) if hi < 10 ** 9 else ">= %d (каталог)" % lo
    print("  %-18s треков %7d: медиана %5.0f, q25 %4.0f, q75 %5.0f, "
          "совсем нет в подвыборке у %d"
          % (lab, band.height, sp["su"].median(), sp["su"].quantile(.25),
             sp["su"].quantile(.75), (sp["su"] == 0).sum()))
