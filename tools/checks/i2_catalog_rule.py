import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 04. Чем задавать каталог: топ-K внутри подвыборки, порог по доле слушателей
# или фиксированный каталог, посчитанный один раз на пуле.
# Популярность — только по истории до test (сутки 0…279), иначе утечка.
DAY = 86400
TEST0 = 280 * DAY
N = 3000            # принято блоком 03
K = 10000           # принято блоком 03
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))

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
print("пул %d пользователей, N = %d, каталог K = %d" % (len(pool), N, K))

trm_pool = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
              .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
rel_pool = (LF.filter(pl.col("timestamp") >= TEST0)
              .select("uid", "item_id", "played_ratio_pct")
              .filter(pl.col("played_ratio_pct") >= 50)
              .collect(engine="streaming")
              .join(pairs, on=["uid", "item_id"], how="left")
              .filter(pl.col("t0") >= TEST0).filter(pl.col("uid").is_in(P.implode())))
pop_pool = (trm_pool.group_by("item_id").agg(pl.col("uid").n_unique().alias("users"))
                    .sort(["users", "item_id"], descending=[True, False]))
cat_pool = set(pop_pool.head(K)["item_id"].to_list())
THR_POOL = pop_pool["users"][K - 1]
print("каталог по пулу: топ-%d = порог «слушали >= %d из %d человек» (%.2f %% пула)"
      % (K, THR_POOL, len(pool), 100 * THR_POOL / len(pool)))

# --- устойчивость каталога «топ-K внутри подвыборки» к сиду ---
print("\n=== 1. устойчивость каталога к сиду подвыборки ===")
cats, samps = {}, {}
for seed in [42, 1, 2, 7]:
    rnd = random.Random(seed)
    samp = rnd.sample(pool, N)
    S = pl.Series("uid", samp)
    tm = trm_pool.filter(pl.col("uid").is_in(S.implode()))
    pp = (tm.group_by("item_id").len()
            .sort(["len", "item_id"], descending=[True, False]))
    cats[seed] = set(pp.head(K)["item_id"].to_list())
    samps[seed] = samp
    print("  сид %-3d: треков в подвыборке %7d, порог на границе каталога %d слушателей, "
          "пересечение с каталогом пула %.1f %%"
          % (seed, pp.height, pp["len"][K - 1],
             100 * len(cats[seed] & cat_pool) / K))
ks = list(cats)
print("  попарные пересечения каталогов между сидами:")
for a in range(len(ks)):
    for b in range(a + 1, len(ks)):
        print("    сид %-3d vs сид %-3d: %.1f %%"
              % (ks[a], ks[b], 100 * len(cats[ks[a]] & cats[ks[b]]) / K))

# --- сравнение трёх правил на сиде 42 ---
print("\n=== 2. три правила задания каталога (сид 42, N = %d) ===" % N)
samp = samps[42]
S = pl.Series("uid", samp)
tm = trm_pool.filter(pl.col("uid").is_in(S.implode()))
allp = (LF.select("uid", "item_id").unique().collect(engine="streaming")
          .filter(pl.col("uid").is_in(S.implode())))
rs = rel_pool.filter(pl.col("uid").is_in(S.implode()))
pp = (tm.group_by("item_id").len()
            .sort(["len", "item_id"], descending=[True, False]))
SHARE = THR_POOL / len(pool)          # та же доля слушателей, но внутри подвыборки
thr_sub = max(1, round(SHARE * N))

RULES = [
    ("A: топ-%d внутри подвыборки" % K, set(pp.head(K)["item_id"].to_list())),
    ("B: >= %d слушателей в подвыборке (та же доля %.2f %%)" % (thr_sub, 100 * SHARE),
     set(pp.filter(pl.col("len") >= thr_sub)["item_id"].to_list())),
    ("C: фиксированный каталог пула (топ-%d на 9 055)" % K, cat_pool),
]
print("  %-52s %7s %9s %10s %7s %7s %9s"
      % ("правило", "треков", "пар", "плотность", "оцен.", "k>=10", "достижимо"))
for lab, cat in RULES:
    C = pl.Series("item_id", sorted(cat))
    tk = tm.filter(pl.col("item_id").is_in(C.implode()))
    ak = allp.filter(pl.col("item_id").is_in(C.implode()))
    rk = rs.filter(pl.col("item_id").is_in(C.implode()))
    present = tk["item_id"].n_unique()
    k = rk.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))
    full = (pl.DataFrame({"uid": samp}).join(k, on="uid", how="left")
              .with_columns(pl.col("k").fill_null(0))["k"])
    print("  %-52s %7d %9d %9.2f %% %7d %7d %8.1f %%"
          % (lab, present, ak.height, 100 * tk.height / (N * max(present, 1)),
             (full >= 1).sum(), (full >= 10).sum(), 100 * rk.height / rs.height))
print("  «треков» — сколько треков каталога реально встречается в подвыборке;")
print("  «пар» — пары (пользователь, трек) по всей истории, бюджет задания 100 тыс. – 2 млн;")
print("  «достижимо» — доля релевантных новинок test внутри каталога")
inter = cat_pool & RULES[0][1]
print("\n  каталог пула и каталог подвыборки пересекаются на %.1f %% (%d треков); "
      "из каталога пула в подвыборке встречается %d треков"
      % (100 * len(inter) / K, len(inter),
         tm.filter(pl.col("item_id").is_in(pl.Series("item_id", sorted(cat_pool))
                                           .implode()))["item_id"].n_unique()))
