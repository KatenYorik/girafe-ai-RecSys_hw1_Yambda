import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 04. Что получается внутри ПРИНЯТОЙ конфигурации:
#   каталог = топ-10 000 треков по числу слушателей в пуле, история до test (сутки 0…279)
#   подвыборка = 3 000 пользователей, сид 42 (блок 03)
# Считаем артистов каталога, покрытие метаданными и срез «новый артист».
DAY = 86400
TEST0 = 280 * DAY
N, K, SEED = 3000, 10000, 42
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
ALB = pl.scan_parquet(os.path.join(raw, "album_item_mapping.parquet"))
EMB = pl.scan_parquet(os.path.join(raw, "embed_item_ids.parquet"))

pairs_lf = (LF.select("uid", "item_id", "timestamp")
              .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0")))
pairs = pairs_lf.collect(engine="streaming")
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

trm_pool = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
              .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
pop = (trm_pool.group_by("item_id").agg(pl.col("uid").n_unique().alias("users"))
               .sort(["users", "item_id"], descending=[True, False]))
CAT = pop.head(K)
C = pl.Series("item_id", CAT["item_id"].to_list())
print("пул %d; каталог = топ-%d по пулу (порог >= %d слушателей)"
      % (len(pool), K, CAT["users"].min()))

# --- 1. метаданные внутри каталога ---
print("\n=== 1. покрытие метаданными: каталог против всего, что слушали ===")
amap = AM.unique(subset=["item_id"], keep="first").collect()
alb = ALB.select("item_id").unique().collect()
emb = EMB.select("item_id").unique().collect()
allit = pop.select("item_id")
for lab, df in [("каталог топ-%d" % K, CAT.select("item_id")),
                ("все треки истории до test", allit)]:
    n = df.height
    na = df.join(amap, on="item_id", how="inner").height
    nl = df.join(alb, on="item_id", how="semi").height
    ne = df.join(emb, on="item_id", how="semi").height
    print("  %-28s треков %7d: артист %6.2f %%, альбом %6.2f %%, эмбеддинг %6.2f %%"
          % (lab, n, 100 * na / n, 100 * nl / n, 100 * ne / n))
print("  (для FM важен эмбеддинг: внутри каталога его нет у %d треков)"
      % (CAT.height - CAT.select("item_id").join(emb, on="item_id", how="semi").height))

# --- 2. артисты каталога ---
print("\n=== 2. артисты внутри каталога ===")
cat_a = CAT.join(amap, on="item_id", how="left")
print("  треков без артиста: %d (%.2f %%)"
      % (cat_a["artist_id"].null_count(), 100 * cat_a["artist_id"].null_count() / K))
ar = (cat_a.filter(pl.col("artist_id").is_not_null())
           .group_by("artist_id").agg(pl.col("item_id").n_unique().alias("tracks"),
                                      pl.col("users").sum().alias("pairs")))
print("  различных артистов в каталоге: %d" % ar.height)
s = ar["tracks"]
print("  треков на артиста: среднее %.2f, " % s.mean()
      + " ".join("q%d=%d" % (int(100 * q), s.quantile(q)) for q in [.5, .75, .9, .99])
      + " max=%d" % s.max())
print("  артистов с одним треком в каталоге: %d (%.1f %%)"
      % ((s == 1).sum(), 100 * (s == 1).mean()))
asr = ar.sort("pairs", descending=True).with_columns(pl.col("pairs").cum_sum().alias("cp"))
TOT = ar["pairs"].sum()
for f in [.5, .8]:
    n = (asr["cp"] < f * TOT).sum() + 1
    print("  %.0f %% пар каталога дают %d артистов (%.1f %% артистов каталога)"
          % (100 * f, n, 100 * n / ar.height))

# --- 3. срез «новый артист» внутри эксперимента ---
print("\n=== 3. срез «новый артист» внутри принятой конфигурации ===")
afirst = (pairs_lf.join(AM, on="item_id", how="inner")
            .group_by(["uid", "artist_id"]).agg(pl.col("t0").min().alias("a0"))
            .collect(engine="streaming"))
rnd = random.Random(SEED)
samp = rnd.sample(pool, N)
S = pl.Series("uid", samp)
rel = (LF.filter(pl.col("timestamp") >= TEST0)
         .select("uid", "item_id", "played_ratio_pct")
         .filter(pl.col("played_ratio_pct") >= 50)
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .filter(pl.col("t0") >= TEST0)
         .filter(pl.col("uid").is_in(S.implode()))
         .join(amap, on="item_id", how="left")
         .join(afirst, on=["uid", "artist_id"], how="left"))
for lab, df in [("без ограничения каталога", rel),
                ("внутри каталога топ-%d" % K,
                 rel.filter(pl.col("item_id").is_in(C.implode())))]:
    na = df.filter(pl.col("artist_id").is_not_null() & (pl.col("a0") >= TEST0))
    ka = na.group_by("uid").agg(pl.col("artist_id").n_unique().alias("k"))
    full = (pl.DataFrame({"uid": samp}).join(ka, on="uid", how="left")
              .with_columns(pl.col("k").fill_null(0))["k"])
    print("  %-30s событий %7d, пользователей %4d, пар (uid, artist) %6d"
          % (lab, na.height, na["uid"].n_unique(),
             na.select(pl.struct("uid", "artist_id").n_unique()).item()))
    print("      новых артистов на пользователя: медиана %d, q75 %d, q90 %d; "
          "с >= 1 — %d, >= 3 — %d, >= 5 — %d, >= 10 — %d"
          % (full.filter(full >= 1).median(), full.filter(full >= 1).quantile(.75),
             full.filter(full >= 1).quantile(.9),
             (full >= 1).sum(), (full >= 3).sum(), (full >= 5).sum(), (full >= 10).sum()))
    print("      различных новых артистов всего: %d" % na["artist_id"].n_unique())
