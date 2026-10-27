import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# Блок 03. Распределение активности пользователей.
# Единица timestamp = 1 с, сутки = 86400 (блок 02, t1_time_unit.py).
DAY = 86400
NDAYS = 301
TEST_L = 21                   # предварительные границы блока 02
VAL_L = 21
TEST0 = (NDAYS - TEST_L) * DAY            # сутки 280
VAL0 = (NDAYS - TEST_L - VAL_L) * DAY     # сутки 259
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
NUSERS_ALL = 10_000           # по объединению всех типов событий (блок 01)

u = (LF.select("uid", "item_id", "timestamp", "played_ratio_pct")
       .with_columns((pl.col("timestamp") // DAY).alias("d"))
       .group_by("uid")
       .agg(pl.len().alias("n"),
            pl.col("item_id").n_unique().alias("items"),
            pl.col("d").n_unique().alias("days"),
            pl.col("d").min().alias("d_first"),
            pl.col("d").max().alias("d_last"),
            (pl.col("played_ratio_pct") >= 50).sum().alias("n_rel"),
            (pl.col("played_ratio_pct") == 0).sum().alias("n_zero"),
            (pl.col("timestamp") < VAL0).sum().alias("n_train"),
            (pl.col("item_id").filter(pl.col("timestamp") < VAL0)
               .n_unique()).alias("items_train"),
            (pl.col("timestamp") >= TEST0).sum().alias("n_test"))
       .collect(engine="streaming"))
print("пользователей с прослушиваниями: %d (из %d по объединению событий)"
      % (u.height, NUSERS_ALL))

QS = [.01, .05, .1, .25, .5, .75, .9, .95, .99]


def dist(s, lab, unit=""):
    print("  %s: среднее %.1f, " % (lab, s.mean())
          + " ".join("q%02d=%d" % (int(q * 100), s.quantile(q)) for q in QS)
          + " max=%d%s" % (s.max(), unit))
    tot = s.sum()
    srt = s.sort(descending=True)
    for p in [1, 5, 10, 25, 50]:
        k = max(1, int(len(s) * p / 100))
        print("     верхние %2d %% пользователей дают %.1f %% объёма (%d из %d)"
              % (p, 100 * srt.head(k).sum() / tot, srt.head(k).sum(), tot))
    print("     скошенность: среднее / медиана = %.2f, коэф. Джини = %.3f"
          % (s.mean() / max(s.median(), 1), gini(s)))


def gini(s):
    v = s.sort().cast(pl.Float64).to_numpy()
    n = len(v)
    c = v.cumsum()
    return (n + 1 - 2 * (c.sum() / c[-1])) / n


print("\n=== 1. прослушивания на пользователя (вся история) ===")
dist(u["n"], "прослушиваний")
print("\n=== 2. различных треков на пользователя ===")
dist(u["items"], "треков")
print("\n=== 3. активных суток на пользователя (из %d) ===" % NDAYS)
dist(u["days"], "активных суток")
span = (u["d_last"] - u["d_first"] + 1)
print("  интервал от первых до последних суток: медиана %d, q10=%d, q90=%d"
      % (span.median(), span.quantile(.1), span.quantile(.9)))
dens = (u["days"] / span)
print("  плотность активности (активные сутки / интервал): медиана %.3f, "
      "q10=%.3f, q25=%.3f, q75=%.3f" % (dens.median(), dens.quantile(.1),
                                        dens.quantile(.25), dens.quantile(.75)))
print("  активны во ВСЕ сутки своего интервала: %d пользователей" % (dens == 1).sum())
print("  активны >= 290 из 301 суток: %d" % (u["days"] >= 290).sum())
print("\n=== 4. прослушиваний в сутки активности ===")
dist((u["n"] / u["days"]).round(0).cast(pl.Int64), "прослушиваний/сутки активности")

print("\n=== 5. первые сутки пользователя (левая цензура и новички) ===")
f = u["d_first"]
print("  d_first: " + " ".join("q%02d=%d" % (int(q * 100), f.quantile(q)) for q in QS))
for lo, hi, lab in [(0, 0, "сутки 0"), (0, 6, "первая неделя"),
                    (0, 29, "первый месяц"), (240, 300, "после суток 240"),
                    (259, 300, "внутри val+test"), (280, 300, "внутри test")]:
    k = ((f >= lo) & (f <= hi)).sum()
    print("    первое прослушивание в %-18s: %4d пользователей (%.1f %%)"
          % (lab, k, 100 * k / u.height))

print("\n=== 6. активность в train / val / test (границы блока 02, предварительные) ===")
print("  train = сутки 0..%d, val = %d..%d, test = %d..300"
      % (NDAYS - TEST_L - VAL_L - 1, NDAYS - TEST_L - VAL_L,
         NDAYS - TEST_L - 1, NDAYS - TEST_L))
print("  активны в train: %d, в test: %d, и там и там: %d"
      % ((u["n_train"] > 0).sum(), (u["n_test"] > 0).sum(),
         ((u["n_train"] > 0) & (u["n_test"] > 0)).sum()))
print("  только в train: %d, только в test: %d, ни там ни там (только val): %d"
      % (((u["n_train"] > 0) & (u["n_test"] == 0)).sum(),
         ((u["n_train"] == 0) & (u["n_test"] > 0)).sum(),
         ((u["n_train"] == 0) & (u["n_test"] == 0)).sum()))
print("\n=== 7. прослушивания в train на пользователя ===")
tr = u.filter(pl.col("n_train") > 0)
dist(tr["n_train"], "прослушиваний в train (у активных в train)")
print("  различных треков в train: " + " ".join(
    "q%02d=%d" % (int(q * 100), tr["items_train"].quantile(q)) for q in QS))
print("\n=== 8. пороги по активности в train ===")
print("  %-10s %10s %12s %14s" % ("порог", "польз.", "доля польз.", "прослуш. теряем"))
TOT = u["n_train"].sum()
for thr in [1, 3, 5, 10, 20, 30, 50, 100, 200, 500]:
    k = (u["n_train"] >= thr).sum()
    kept = u.filter(pl.col("n_train") >= thr)["n_train"].sum()
    print("  >=%-8d %10d %11.1f%% %13.3f%%"
          % (thr, k, 100 * k / u.height, 100 * (TOT - kept) / TOT))
print("\n=== 9. доля played_ratio == 0 на пользователя (перелистывание) ===")
z = (u["n_zero"] / u["n"])
print("  доля нулевых: среднее %.4f, " % z.mean()
      + " ".join("q%02d=%.4f" % (int(q * 100), z.quantile(q)) for q in QS)
      + " max=%.4f" % z.max())
for thr in [.3, .4, .5, .6, .8]:
    k = (z > thr).sum()
    print("    доля нулевых > %.0f %%: %d пользователей (%.2f %%), "
          "у них прослушиваний %d" % (100 * thr, k, 100 * k / u.height,
                                      u.filter(z > thr)["n"].sum()))
print("  у них же доля релевантных (>=50 %%): среднее %.3f против %.3f у остальных"
      % ((u.filter(z > .5)["n_rel"] / u.filter(z > .5)["n"]).mean(),
         (u.filter(z <= .5)["n_rel"] / u.filter(z <= .5)["n"]).mean()))
