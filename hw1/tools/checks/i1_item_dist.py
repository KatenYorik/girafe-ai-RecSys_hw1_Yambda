import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# Блок 04. Хвост популярности треков и артистов, покрытие прослушиваний, пороги.
# Всё считается по истории ДО test (сутки 0…279): каталог, построенный с участием
# тестового окна, означал бы утечку в метрику.
# Границы — блок 02, фильтр пользователей — блок 03 (>= 5 прослушиваний, без
# «перелистывающих», без холодных).
DAY = 86400
TEST0 = 280 * DAY
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))

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
print("пул пользователей (блок 03): %d" % len(pool))

# трек -> (число пользователей пула, число прослушиваний) в истории до test
it = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id")
        .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode()))
        .group_by("item_id").agg(pl.col("uid").n_unique().alias("users"),
                                 pl.len().alias("plays")))
NIT, NPL, NPR = it.height, it["plays"].sum(), it["users"].sum()
print("треков в истории до test: %d; прослушиваний %d; пар (пользователь, трек) %d"
      % (NIT, NPL, NPR))

QS = [.5, .75, .9, .95, .99, .999]
print("\n=== 1. распределение популярности трека ===")
for c, lab in [("users", "слушателей на трек"), ("plays", "прослушиваний на трек")]:
    s = it[c]
    print("  %-22s среднее %6.2f, " % (lab, s.mean())
          + " ".join("q%.1f=%d" % (100 * q, s.quantile(q)) for q in QS)
          + " max=%d" % s.max())
for k in [1, 2, 3]:
    print("  треков, которых слушали ровно %d человек: %d (%.1f %%)"
          % (k, (it["users"] == k).sum(), 100 * (it["users"] == k).mean()))

print("\n=== 2. сколько треков покрывают долю прослушиваний ===")
srt = it.sort("plays", descending=True).with_columns(
    pl.col("plays").cum_sum().alias("cp"), pl.col("users").cum_sum().alias("cu"))
print("  %-8s %10s %12s %10s" % ("доля", "треков", "доля каталога", "доля пар"))
for f in [.5, .8, .9, .95, .99]:
    n = (srt["cp"] < f * NPL).sum() + 1
    print("  %-8.0f%% %10d %11.2f %% %9.1f %%"
          % (100 * f, n, 100 * n / NIT, 100 * srt["cu"][n - 1] / NPR))
print("  обратно: топ-K треков по числу слушателей покрывает прослушиваний и пар:")
su = it.sort(["users", "item_id"], descending=[True, False]).with_columns(
    pl.col("plays").cum_sum().alias("cp"), pl.col("users").cum_sum().alias("cu"))
for K in [1000, 5000, 10000, 15000, 30000, 50000, 100000]:
    if K > NIT:
        continue
    print("    топ-%6d: прослушиваний %5.1f %%, пар %5.1f %%"
          % (K, 100 * su["cp"][K - 1] / NPL, 100 * su["cu"][K - 1] / NPR))

print("\n=== 3. пороги по числу слушателей (на пуле %d человек) ===" % len(pool))
print("  %-8s %10s %12s %10s %12s" % ("порог", "треков", "доля каталога", "доля пар",
                                      "доля прослуш."))
for M in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]:
    sub = it.filter(pl.col("users") >= M)
    if sub.height == 0:
        continue
    print("  >=%-6d %10d %11.2f %% %9.1f %% %11.1f %%"
          % (M, sub.height, 100 * sub.height / NIT,
             100 * sub["users"].sum() / NPR, 100 * sub["plays"].sum() / NPL))
# какой порог даёт каталог нужного размера
print("  порог, дающий каталог заданного размера:")
for K in [5000, 10000, 15000]:
    thr = su["users"][K - 1]
    n_at = (it["users"] >= thr).sum()
    print("    %6d треков  ->  порог «слушали >= %d человек» (даёт %d треков: "
          "на границе много треков с одинаковой популярностью)" % (K, thr, n_at))

print("\n=== 4. артисты ===")
amap = AM.unique(subset=["item_id"], keep="first").collect()
ia = it.join(amap, on="item_id", how="left")
print("  треков без артиста в маппинге: %d (%.2f %%)"
      % (ia["artist_id"].null_count(), 100 * ia["artist_id"].null_count() / NIT))
ar = (ia.filter(pl.col("artist_id").is_not_null())
        .group_by("artist_id").agg(pl.col("item_id").n_unique().alias("tracks"),
                                   pl.col("plays").sum().alias("plays"),
                                   pl.col("users").sum().alias("pairs")))
print("  различных артистов: %d" % ar.height)
for c, lab in [("tracks", "треков на артиста"), ("plays", "прослушиваний на артиста")]:
    s = ar[c]
    print("  %-26s среднее %7.2f, " % (lab, s.mean())
          + " ".join("q%.1f=%d" % (100 * q, s.quantile(q)) for q in QS)
          + " max=%d" % s.max())
print("  артистов с одним треком: %d (%.1f %%)"
      % ((ar["tracks"] == 1).sum(), 100 * (ar["tracks"] == 1).mean()))
asrt = ar.sort("plays", descending=True).with_columns(pl.col("plays").cum_sum().alias("cp"))
APL = ar["plays"].sum()
for f in [.5, .8, .9]:
    n = (asrt["cp"] < f * APL).sum() + 1
    print("  %.0f %% прослушиваний дают %d артистов (%.2f %% всех)"
          % (100 * f, n, 100 * n / ar.height))
print("  пороги по артистам:")
for M in [1, 5, 10, 50, 100, 500]:
    sub = ar.filter(pl.col("pairs") >= M)
    print("    артистов с >= %4d парами (пользователь, трек): %7d (%.1f %%), "
          "прослушиваний %.1f %%"
          % (M, sub.height, 100 * sub.height / ar.height,
             100 * sub["plays"].sum() / APL))
