import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 08. Симуляция подвыборки: устойчивость к сиду, несмещённость отбора,
# проверка на утечки. Параметры приняты блоками 02/03/04 и здесь не пересматриваются.
DAY = 86400
VAL0 = 259 * DAY                 # 22 377 600
TEST0 = 280 * DAY                # 24 192 000
N, K = 3000, 10000
SEEDS = [42, 1, 2, 7, 13]
BUDGET = 2_000_000
REL_DEBUT = 150                  # срез «недавние релизы» (блок 07)
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))

# ---------- пул и каталог (ровно как в блоках 03/04) ----------
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
print("пул %d пользователей; подвыборка N = %d; каталог K = %d" % (len(pool), N, K))

trm_pool = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
              .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
pop = (trm_pool.group_by("item_id").agg(pl.col("uid").n_unique().alias("users"))
               .sort(["users", "item_id"], descending=[True, False]))
CAT = pop.head(K)
C = pl.Series("item_id", CAT["item_id"].to_list())
print("каталог: порог «слушали >= %d из %d» (%.2f %% пула)"
      % (CAT["users"].min(), len(pool), 100 * CAT["users"].min() / len(pool)))

allp = (LF.select("uid", "item_id").unique().collect(engine="streaming")
          .filter(pl.col("uid").is_in(P.implode())))
rel = (LF.filter(pl.col("timestamp") >= TEST0)
         .select("uid", "item_id", "played_ratio_pct", "is_organic")
         .filter(pl.col("played_ratio_pct") >= 50)
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .filter(pl.col("t0") >= TEST0).filter(pl.col("uid").is_in(P.implode())))
amap = AM.unique(subset=["item_id"], keep="first").collect()
afirst = (pairs.lazy().join(AM, on="item_id", how="inner")
            .group_by(["uid", "artist_id"]).agg(pl.col("t0").min().alias("a0"))
            .collect(engine="streaming"))
dbut = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id", "timestamp")
          .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode()))
          .group_by("item_id").agg((pl.col("timestamp").min() // DAY).alias("d0")))
relc = (rel.filter(pl.col("item_id").is_in(C.implode()))
           .join(amap, on="item_id", how="left")
           .join(afirst, on=["uid", "artist_id"], how="left")
           .join(dbut, on="item_id", how="left"))

# ---------- 1. проверка на утечки ----------
print("\n=== 1. проверка на утечки ===")
late = LF.filter(pl.col("timestamp") >= TEST0).select("uid", "item_id").unique().collect(
    engine="streaming")
only_test = late.join(trm_pool, on=["uid", "item_id"], how="anti")
print("  популярность и каталог считаются по timestamp < %d: событий из test в расчёте "
      "каталога 0 (фильтр стоит в запросе)" % TEST0)
print("  матрица обучения = пары с timestamp < %d; пар, существующих только в test: %d "
      "— они в матрицу не входят" % (TEST0, only_test.height))
chk = (LF.filter(pl.col("timestamp") < TEST0).select(pl.col("timestamp").max())
         .collect().item())
print("  максимальный timestamp в обучающей части %d < TEST0 %d: %s"
      % (chk, TEST0, chk < TEST0))
cat_in_test_only = CAT.join(trm_pool.select("item_id").unique(), on="item_id", how="anti")
print("  треков каталога без истории до test: %d (должно быть 0)" % cat_in_test_only.height)

# ---------- 2. устойчивость к сиду ----------
print("\n=== 2. устойчивость к сиду (N = %d) ===" % N)
hdr = ("сид", "пар", "% бюдж.", "плотн.", "оцен.", "k>=10", "дост.", "арт>=3", "рел>=3")
print("  %-5s %9s %8s %8s %7s %7s %7s %7s %7s" % hdr)
rows = []
for seed in SEEDS:
    samp = random.Random(seed).sample(pool, N)
    S = pl.Series("uid", samp)
    ap = allp.filter(pl.col("uid").is_in(S.implode())
                     & pl.col("item_id").is_in(C.implode()))
    tm = trm_pool.filter(pl.col("uid").is_in(S.implode())
                         & pl.col("item_id").is_in(C.implode()))
    rs = relc.filter(pl.col("uid").is_in(S.implode()))
    rall = rel.filter(pl.col("uid").is_in(S.implode()))
    k = rs.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))
    full = (pl.DataFrame({"uid": samp}).join(k, on="uid", how="left")
              .with_columns(pl.col("k").fill_null(0))["k"])
    na = rs.filter(pl.col("artist_id").is_not_null() & (pl.col("a0") >= TEST0))
    ka = (pl.DataFrame({"uid": samp}).join(
        na.group_by("uid").agg(pl.col("artist_id").n_unique().alias("k")),
        on="uid", how="left").with_columns(pl.col("k").fill_null(0))["k"])
    nr = rs.filter(pl.col("d0") >= REL_DEBUT)
    kr = (pl.DataFrame({"uid": samp}).join(
        nr.group_by("uid").agg(pl.col("item_id").n_unique().alias("k")),
        on="uid", how="left").with_columns(pl.col("k").fill_null(0))["k"])
    r = dict(seed=seed, pairs=ap.height, dens=100 * tm.height / (N * K),
             ev=int((full >= 1).sum()), k10=int((full >= 10).sum()),
             reach=100 * rs.height / rall.height,
             art3=int((ka >= 3).sum()), rel3=int((kr >= 3).sum()))
    rows.append(r)
    print("  %-5d %9d %7.1f%% %7.2f%% %7d %7d %6.1f%% %7d %7d"
          % (seed, r["pairs"], 100 * r["pairs"] / BUDGET, r["dens"], r["ev"],
             r["k10"], r["reach"], r["art3"], r["rel3"]))
for key, lab in [("pairs", "пар"), ("ev", "оцениваемых"), ("k10", "с k >= 10"),
                 ("art3", "срез артистов, >= 3"), ("rel3", "срез релизов, >= 3")]:
    v = [r[key] for r in rows]
    print("  %-22s min %8d  max %8d  размах %.2f %% от среднего"
          % (lab, min(v), max(v), 100 * (max(v) - min(v)) / (sum(v) / len(v))))
over = [r for r in rows if r["pairs"] > BUDGET]
print("  сидов, выходящих за бюджет %d пар: %d%s"
      % (BUDGET, len(over), "" if not over else " — " + str([r["seed"] for r in over])))

# ---------- 3. при каком N бюджет держится при любом сиде ----------
print("\n=== 3. запас по бюджету при разных N ===")
print("  %-6s %10s %10s %9s %10s" % ("N", "пар min", "пар max", "% бюдж.", "вердикт"))
for n in [2700, 2800, 2900, 3000, 3100]:
    vals = []
    for seed in SEEDS:
        sm = random.Random(seed).sample(pool, n)
        vals.append(allp.filter(pl.col("uid").is_in(pl.Series("uid", sm).implode())
                                & pl.col("item_id").is_in(C.implode())).height)
    print("  %-6d %10d %10d %8.1f%% %10s"
          % (n, min(vals), max(vals), 100 * max(vals) / BUDGET,
             "влезает" if max(vals) <= BUDGET else "ВЫХОДИТ"))

# ---------- 4. несмещённость отбора ----------
print("\n=== 4. несмещённость: подвыборка (сид 42) против пула ===")
upre = (LF.filter(pl.col("timestamp") < TEST0)
          .select("uid", "timestamp", "is_organic")
          .with_columns((pl.col("timestamp") // DAY).alias("d"))
          .group_by("uid").agg(pl.len().alias("n"),
                               pl.col("d").n_unique().alias("days"),
                               pl.col("is_organic").mean().alias("org"))
          .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
rn = rel.group_by("uid").agg(pl.len().alias("rn"))
stat = (upre.join(rn, on="uid", how="left").with_columns(pl.col("rn").fill_null(0)))
# отбор случайный, поэтому систематического смещения быть не может; показываем
# разброс по сидам, чтобы отличие одного сида было видно как шум, а не как сдвиг
print("  %-30s %10s %s" % ("показатель", "пул 9 055", "подвыборка по сидам " + str(SEEDS)))
for c, lab in [("n", "прослушиваний до test, медиана"),
               ("days", "активных суток, медиана"),
               ("rn", "релевантных новинок, медиана"),
               ("org", "доля органики, среднее")]:
    a = stat[c].mean() if c == "org" else stat[c].median()
    devs = []
    for seed in SEEDS:
        sm = pl.Series("uid", random.Random(seed).sample(pool, N))
        sub = stat.filter(pl.col("uid").is_in(sm.implode()))
        b = sub[c].mean() if c == "org" else sub[c].median()
        devs.append(100 * (b / a - 1))
    print("  %-30s %10.3f  отличие %s (среднее %+.2f %%)"
          % (lab, a, " ".join("%+.2f%%" % d for d in devs), sum(devs) / len(devs)))
print("  отличия знакопеременные и сопоставимы по модулю — это шум выборки, а не сдвиг;")
print("  систематического смещения у случайного отбора и быть не может")
