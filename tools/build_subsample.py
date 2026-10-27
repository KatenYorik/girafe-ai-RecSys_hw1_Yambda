import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
# Сборщик подвыборки для ДЗ 1. Детерминированный: при тех же аргументах даёт
# побайтово те же файлы. Параметры приняты блоками 02/03/04, см. eda/08-subsample.
#
#   python -I tools/build_subsample.py data/raw [data/subsample]
#
# Исходные данные в data/raw только читаются.
raw = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(raw), "subsample")

DAY = 86400
VAL0 = 259 * DAY                 # 22 377 600 — граница train/val
TEST0 = 280 * DAY                # 24 192 000 — граница val/test
N, K, SEED = 3000, 10000, 42
MIN_LISTENS = 5                  # порог активности в истории до test (блок 03)
MAX_ZERO = 0.5                   # доля played_ratio_pct == 0, выше которой выкидываем
BUDGET = (100_000, 2_000_000)

LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
EMB = pl.scan_parquet(os.path.join(raw, "embed_item_ids.parquet"))
os.makedirs(out, exist_ok=True)
print("источник %s -> приёмник %s" % (raw, out))

# ---------- 1. пул пользователей (блок 03) ----------
pairs = (LF.select("uid", "item_id", "timestamp")
           .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
           .collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("uf"))
zero = (LF.select("uid", "played_ratio_pct").group_by("uid")
          .agg((pl.col("played_ratio_pct") == 0).mean().alias("z"))
          .collect(engine="streaming"))
npre = (LF.filter(pl.col("timestamp") < TEST0)
          .select("uid", "timestamp")
          .with_columns((pl.col("timestamp") // DAY).alias("d"))
          .group_by("uid").agg(pl.len().alias("n_pre"),
                               pl.col("d").n_unique().alias("days_pre"))
          .collect(engine="streaming"))
pool = sorted((set(ufirst.filter(pl.col("uf") < TEST0)["uid"].to_list())
               - set(zero.filter(pl.col("z") > MAX_ZERO)["uid"].to_list()))
              & set(npre.filter(pl.col("n_pre") >= MIN_LISTENS)["uid"].to_list()))
P = pl.Series("uid", pool)
print("пул после фильтра блока 03: %d пользователей" % len(pool))

# ---------- 2. каталог (блок 04): топ-K по числу слушателей ПУЛА до test ----------
trm_pool = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
              .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
pop = (trm_pool.group_by("item_id").agg(pl.col("uid").n_unique().alias("pool_users"))
               .sort(["pool_users", "item_id"], descending=[True, False])
               .head(K).with_row_index("rank", offset=1))
C = pl.Series("item_id", pop["item_id"].to_list())
print("каталог: %d треков, порог «слушали >= %d из %d»"
      % (K, pop["pool_users"].min(), len(pool)))

# ---------- 3. подвыборка пользователей ----------
samp = sorted(random.Random(SEED).sample(pool, N))
S = pl.Series("uid", samp)
print("подвыборка: %d пользователей (сид %d)" % (N, SEED))

# ---------- 4. события по окнам, только каталог ----------
COLS = ["uid", "item_id", "timestamp", "played_ratio_pct", "is_organic"]
ev = (LF.select(COLS).collect(engine="streaming")
        .filter(pl.col("uid").is_in(S.implode()) & pl.col("item_id").is_in(C.implode()))
        .join(pairs, on=["uid", "item_id"], how="left"))
WIN = [("train", 0, VAL0), ("val", VAL0, TEST0), ("test", TEST0, None)]
sizes = {}
for name, lo, hi in WIN:
    d = ev.filter(pl.col("timestamp") >= lo)
    if hi is not None:
        d = d.filter(pl.col("timestamp") < hi)
    d = d.with_columns((pl.col("t0") >= lo).alias("is_new"),
                       (pl.col("played_ratio_pct") >= 50).alias("is_relevant"))
    d = d.select(COLS + ["is_new", "is_relevant"]).sort(["uid", "timestamp", "item_id"])
    d.write_parquet(os.path.join(out, name + ".parquet"))
    sizes[name] = d
    print("  %-5s: строк %8d, пар %8d, треков %6d, пользователей %5d"
          % (name, d.height, d.select(pl.struct("uid", "item_id").n_unique()).item(),
             d["item_id"].n_unique(), d["uid"].n_unique()))

# ---------- 5. справочники ----------
amap = AM.unique(subset=["item_id"], keep="first").collect()
emb = EMB.select("item_id").unique().with_columns(pl.lit(True).alias("has_embed")).collect()
dbut = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id", "timestamp")
          .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode()))
          .group_by("item_id").agg((pl.col("timestamp").min() // DAY).alias("debut_day")))
cat = (pop.join(amap, on="item_id", how="left")
          .join(emb, on="item_id", how="left")
          .join(dbut, on="item_id", how="left")
          .with_columns(pl.col("has_embed").fill_null(False))
          .select("item_id", "rank", "pool_users", "artist_id", "has_embed", "debut_day")
          .sort("rank"))
cat.write_parquet(os.path.join(out, "catalog.parquet"))
users = (pl.DataFrame({"uid": pl.Series(samp, dtype=pl.UInt32)})
           .join(npre, on="uid", how="left")
           .join(sizes["test"].group_by("uid").agg(pl.len().alias("n_test")),
                 on="uid", how="left")
           .with_columns(pl.col("n_test").fill_null(0)).sort("uid"))
users.write_parquet(os.path.join(out, "users.parquet"))

# ---------- 6. отчёт ----------
tr = sizes["train"].select(pl.struct("uid", "item_id").n_unique()).item()
mat = (pl.concat([sizes["train"], sizes["val"]])
         .select(pl.struct("uid", "item_id").n_unique()).item())
allpairs = ev.select(pl.struct("uid", "item_id").n_unique()).item()
print("\n=== итог ===")
print("  пар (пользователь, трек) по всей оставляемой истории: %d (бюджет %d–%d: %s)"
      % (allpairs, BUDGET[0], BUDGET[1],
         "ВЛЕЗАЕТ" if BUDGET[0] <= allpairs <= BUDGET[1] else "НЕ ВЛЕЗАЕТ"))
print("  матрица обучения train: %d пар, плотность %.2f %%" % (tr, 100 * tr / (N * K)))
print("  матрица обучения train+val (для финальной модели): %d пар, плотность %.2f %%"
      % (mat, 100 * mat / (N * K)))
print("  треков %d, артистов %d, пользователей %d"
      % (K, cat["artist_id"].n_unique(), N))
print("  эмбеддинг есть у %d треков каталога (%.2f %%)"
      % (cat["has_embed"].sum(), 100 * cat["has_embed"].mean()))
t = sizes["test"]
k = (t.filter(pl.col("is_new") & pl.col("is_relevant"))
      .group_by("uid").agg(pl.col("item_id").n_unique().alias("k")))
full = (users.select("uid").join(k, on="uid", how="left")
          .with_columns(pl.col("k").fill_null(0))["k"])
print("  оцениваемых (есть релевантная новинка в каталоге): %d; с полным idealDCG@10: %d"
      % ((full >= 1).sum(), (full >= 10).sum()))
print("\nзаписано: " + ", ".join(sorted(os.listdir(out))))
print("размер: %.1f МБ" % (sum(os.path.getsize(os.path.join(out, f))
                               for f in os.listdir(out)) / 1024 ** 2))
