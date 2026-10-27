import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, time
raw = sys.argv[1]
# ВНИМАНИЕ (поправка блока 02): DAY = 17280 — это НЕ сутки, а 1/5 суток (4 ч 48 мин).
# Единица timestamp равна 1 секунде, реальные сутки = 86400 единиц (см. t1_time_unit.py).
# Константа оставлена как есть, чтобы числа блока воспроизводились: любой горизонт,
# который этот скрипт печатает «в днях», читается как (дни / 5) реальных суток.
DAY = 17280
TRAIN_DAYS = 1414                 # блок 06: test = дни 1414..1503 (90 дней)
TRAIN_END = TRAIN_DAYS * DAY      # train = дни 0..1413
KEY = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct"]
t = time.time()
LF = (pl.scan_parquet(os.path.join(raw, "listens.parquet"))
        .filter(pl.col("timestamp") < TRAIN_END)
        .select(KEY).unique(subset=KEY)
        .with_columns(pl.min_horizontal(pl.col("played_ratio_pct"), pl.lit(100))
                        .cast(pl.Int32).alias("r")))
# повтор = отдельное прослушивание (строка после удаления полных дублей), не строка-дубль
p = (LF.group_by(["uid", "item_id"]).agg(
        pl.len().alias("n"),
        pl.col("timestamp").min().alias("t0"),
        pl.col("timestamp").max().alias("tl"),
        pl.col("r").mean().alias("rm"),
        (pl.col("r") >= 50).sum().alias("n50"),
        (pl.col("r") == 100).sum().alias("n100"))
      .collect(engine="streaming"))
tot_pairs, tot_listens = p.height, p["n"].sum()
print("пар (пользователь, трек) в train: %d; прослушиваний %d; %.1f с"
      % (tot_pairs, tot_listens, time.time() - t))
print("прослушиваний на пару: mean %.3f  median %d  q75 %d  q90 %d  q95 %d  q99 %d  max %d"
      % (p["n"].mean(), p["n"].median(), p["n"].quantile(.75), p["n"].quantile(.9),
         p["n"].quantile(.95), p["n"].quantile(.99), p["n"].max()))

print("\n=== распределение числа прослушиваний пары ===")
print("  n   пар          доля пар, %   накопл. пар, %   прослушиваний   доля прослуш., %   накопл. прослуш., %")
g = p.group_by("n").agg(pl.len().alias("pairs")).sort("n")
cp = cl = 0
rows = {n: pr for n, pr in g.iter_rows()}
for n in sorted(rows):
    pr = rows[n]; cp += pr; cl += pr * n
    if n <= 20 or n in (25, 30, 40, 50, 100, 200):
        print("  %3d %9d   %11.3f   %13.3f   %13d   %14.3f   %16.3f"
              % (n, pr, 100*pr/tot_pairs, 100*cp/tot_pairs, pr*n, 100*pr*n/tot_listens, 100*cl/tot_listens))
print("\n  доля пар с n == 1: %.2f %% (они дают %.2f %% прослушиваний)"
      % (100*rows.get(1,0)/tot_pairs, 100*rows.get(1,0)/tot_listens))
for thr in [2, 3, 5, 10, 20, 50, 100]:
    pp = p.filter(pl.col("n") >= thr)
    print("  пар с n >= %3d: %8d (%5.2f %% пар), прослушиваний %9d (%5.2f %%)"
          % (thr, pp.height, 100*pp.height/tot_pairs, pp["n"].sum(), 100*pp["n"].sum()/tot_listens))

print("\n=== хвост: сколько прослушиваний «съедает» усечение n на уровне C ===")
for C in [1, 2, 3, 5, 8, 10, 15, 20, 30, 50]:
    kept = p.select(pl.min_horizontal(pl.col("n"), pl.lit(C)).sum()).item()
    print("  n -> min(n, %2d): остаётся %10d из %d прослушиваний (%.2f %%); "
          "усечение затрагивает %7d пар (%.2f %%)"
          % (C, kept, tot_listens, 100*kept/tot_listens,
             p.filter(pl.col("n") > C).height, 100*p.filter(pl.col("n") > C).height/tot_pairs))

print("\n=== у кого копятся повторы: пары с большим n ===")
hv = p.filter(pl.col("n") >= 50)
print("  пар с n >= 50: %d, у %d пользователей (из %d); доля прослушиваний %.2f %%"
      % (hv.height, hv["uid"].n_unique(), p["uid"].n_unique(),
         100*hv["n"].sum()/tot_listens))
pu = p.group_by("uid").agg(pl.col("n").max().alias("nmax"), pl.len().alias("items"),
                           pl.col("n").sum().alias("listens"))
print("  максимум прослушиваний одного трека у пользователя: median %d q90 %d q99 %d max %d"
      % (pu["nmax"].median(), pu["nmax"].quantile(.9), pu["nmax"].quantile(.99), pu["nmax"].max()))

print("\n=== добавляют ли повторы информацию: завершаемость против числа повторов ===")
print("   n      пар     средний played_ratio пары   доля >=50%% среди прослушиваний   доля ==100%%")
for lo, hi in [(1,1),(2,2),(3,3),(4,5),(6,10),(11,20),(21,50),(51,10**9)]:
    b = p.filter((pl.col("n") >= lo) & (pl.col("n") <= hi))
    if not b.height: continue
    lab = "%d" % lo if lo == hi else "%d-%d" % (lo, min(hi, 10**8))
    print("  %-6s %8d   %21.2f   %29.2f %%   %10.2f %%"
          % (lab, b.height, (b["rm"]*b["n"]).sum()/b["n"].sum(),
             100*b["n50"].sum()/b["n"].sum(), 100*b["n100"].sum()/b["n"].sum()))

print("\n=== как растянуты повторы по времени (дни между первым и последним) ===")
sp = p.filter(pl.col("n") >= 2).with_columns(((pl.col("tl") - pl.col("t0")) / DAY).alias("span"))
print("  пар с n >= 2: %d; разброс первое..последнее, дней: median %.2f q25 %.2f q75 %.2f q90 %.1f"
      % (sp.height, sp["span"].median(), sp["span"].quantile(.25),
         sp["span"].quantile(.75), sp["span"].quantile(.9)))
print("  доля пар с n >= 2, у которых все прослушивания в один день: %.1f %%; в пределах часа: %.1f %%"
      % (100*(sp["span"] < 1).mean(), 100*(sp["span"] < 1/24).mean()))
