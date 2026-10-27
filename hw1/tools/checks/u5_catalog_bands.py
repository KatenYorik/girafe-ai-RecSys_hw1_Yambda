import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 03 (вход и в блок 04). Что несут полосы каталога по популярности:
# стоит ли расширять каталог с топ-10 000 до топ-15 000.
# Принятые границы (блок 02): train = сутки 0…258, val = 259…279, test = 280…300.
# Принятый фильтр (блок 03): >= 5 прослушиваний в истории до test.
DAY = 86400
TEST0 = 280 * DAY
SEED = 42
N = int(sys.argv[2]) if len(sys.argv) > 2 else 2600
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
rnd = random.Random(SEED)
samp = pool if N >= len(pool) else rnd.sample(pool, N)
S = pl.Series("uid", samp)
print("пул %d, подвыборка N = %d (сид %d)" % (len(pool), len(samp), SEED))

trm = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
         .collect(engine="streaming").filter(pl.col("uid").is_in(S.implode())))
rel = (LF.filter(pl.col("timestamp") >= TEST0)
         .select("uid", "item_id", "played_ratio_pct")
         .filter(pl.col("played_ratio_pct") >= 50)
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .filter(pl.col("t0") >= TEST0).filter(pl.col("uid").is_in(S.implode())))
pop = (trm.group_by("item_id").len().sort("len", descending=True)
          .with_row_index("rank", offset=1))
print("пар в матрице обучения %d, различных треков %d; релевантных новинок в test %d "
      "у %d пользователей" % (trm.height, pop.height, rel.height, rel["uid"].n_unique()))

# --- что лежит в полосах по популярности ---
BANDS = [(1, 1000), (1001, 5000), (5001, 10000), (10001, 15000), (15001, 30000)]
print("\n=== полосы каталога (ранг по числу пользователей в матрице обучения) ===")
print("  %-14s %7s %10s %9s %8s %8s %11s" % ("полоса", "треков", "взаимод.",
                                             "доля вз.", "медиана", "среднее", "новинок test"))
relp = rel.join(pop.select("item_id", "rank"), on="item_id", how="left")
for a, b in BANDS:
    band = pop.filter((pl.col("rank") >= a) & (pl.col("rank") <= b))
    if band.height == 0:
        continue
    rb = relp.filter((pl.col("rank") >= a) & (pl.col("rank") <= b))
    print("  %-14s %7d %10d %8.1f %% %8.0f %8.2f %11d"
          % ("%d–%d" % (a, b), band.height, band["len"].sum(),
             100 * band["len"].sum() / trm.height,
             band["len"].median(), band["len"].mean(), rb.height))
print("  «медиана/среднее» — сколько пользователей подвыборки знают трек из полосы")
nr = relp.filter(pl.col("rank").is_null())
print("  треков вне матрицы обучения (новинки, которых никто из подвыборки не слушал "
      "до test): новинок test %d" % nr.height)

# --- сколько оцениваемых добавляет расширение каталога ---
print("\n=== оцениваемые по размеру каталога ===")
ev_full = rel["uid"].n_unique()
prev = None
print("  %-10s %9s %11s %9s %12s %11s" % ("каталог", "пар", "оцениваемых",
                                          "+к пред.", "плотность", "покрытие"))
for K in [1000, 3000, 5000, 8000, 10000, 15000, pop.height]:
    if K > pop.height:
        continue
    top = pop.head(K)["item_id"]
    tk = trm.filter(pl.col("item_id").is_in(top.implode()))
    ek = rel.filter(pl.col("item_id").is_in(top.implode()))["uid"].n_unique()
    lab = "весь (%d)" % K if K == pop.height else "топ-%d" % K
    print("  %-10s %9d %11d %9s %10.2f %% %9.1f %%"
          % (lab, tk.height, ek, "—" if prev is None else "+%d" % (ek - prev),
             100 * tk.height / (len(samp) * K), 100 * tk.height / trm.height))
    prev = ek
print("  всего оцениваемых при неограниченном каталоге: %d" % ev_full)

# --- насколько полоса 10–15 тыс. вообще обучаема ---
print("\n=== обучаемость полосы 10 001–15 000 ===")
band = pop.filter((pl.col("rank") >= 10001) & (pl.col("rank") <= 15000))
if band.height:
    for thr in [1, 2, 3, 5, 10]:
        k = (band["len"] <= thr).sum()
        print("    треков, известных <= %2d пользователям: %d из %d (%.1f %%)"
              % (thr, k, band.height, 100 * k / band.height))
    b5 = pop.filter((pl.col("rank") >= 5001) & (pl.col("rank") <= 10000))
    print("    для сравнения, полоса 5 001–10 000: известных <= 3 пользователям "
          "%d из %d (%.1f %%)"
          % ((b5["len"] <= 3).sum(), b5.height, 100 * (b5["len"] <= 3).mean()))
    print("    EASE оценивает похожесть треков по совместным прослушиваниям: трек, который "
          "знают 1–2 человека,\n    получает вес из одного-двух наблюдений, то есть шум")

# --- потолок метрики: сколько у пользователя ДОСТИЖИМЫХ релевантных новинок ---
# NDCG@10 нормируется на idealDCG@min(10, k), поэтому важно не общее число
# новинок, а есть ли у человека хотя бы 10 внутри каталога
print("\n=== потолок метрики по размеру каталога ===")
print("  %-10s %9s %9s %9s %9s %12s" % ("каталог", ">=1", ">=3", ">=5", ">=10",
                                        "медиана k"))
for K in [5000, 8000, 10000, 15000, pop.height]:
    if K > pop.height:
        continue
    top = pop.head(K)["item_id"]
    rk = rel.filter(pl.col("item_id").is_in(top.implode()))
    k = rk.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))
    full = (pl.DataFrame({"uid": samp})
              .join(k, on="uid", how="left")
              .with_columns(pl.col("k").fill_null(0))["k"])
    lab = "весь" if K == pop.height else "топ-%d" % K
    print("  %-10s %9d %9d %9d %9d %12d"
          % (lab, (full >= 1).sum(), (full >= 3).sum(), (full >= 5).sum(),
             (full >= 10).sum(), full.filter(full >= 1).median()))
print("  (пользователей в подвыборке %d; >= 10 — те, у кого idealDCG@10 полон)" % len(samp))

print("\n=== доля релевантных новинок, достижимых внутри каталога ===")
tot = rel.height
for K in [5000, 10000, 15000, pop.height]:
    if K > pop.height:
        continue
    top = pop.head(K)["item_id"]
    rk = rel.filter(pl.col("item_id").is_in(top.implode()))
    lab = "весь" if K == pop.height else "топ-%d" % K
    print("  %-10s %9d из %d (%.1f %%)" % (lab, rk.height, tot, 100 * rk.height / tot))
