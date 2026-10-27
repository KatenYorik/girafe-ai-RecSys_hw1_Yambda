import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# ВНИМАНИЕ (поправка блока 02): единица timestamp = 1 с, сутки = 86400 единиц.
# Поэтому корзины 150*17280 ниже — это НЕ 150 дней, а 30 реальных суток.
# Оставлено как есть, чтобы числа блока 01 воспроизводились (см. t1_time_unit.py).
O = pl.col("is_organic"); r = pl.col("played_ratio_pct"); T = pl.col("timestamp")
print("=== share of is_organic by event type ===")
for f in ["listens","likes","dislikes","unlikes","undislikes"]:
    lf = pl.scan_parquet(os.path.join(raw,f+".parquet"))
    a = lf.select(O.mean().alias("m"), pl.len().alias("n"), O.n_unique().alias("nu")).collect().row(0)
    print("  %-11s is_organic=1 : %6.2f%%  (n=%d, distinct values=%d)" % (f, 100*a[0], a[1], a[2]))

lf = pl.scan_parquet(os.path.join(raw,"listens.parquet"))
print("\n=== listens: completion by is_organic ===")
g = lf.group_by("is_organic").agg(
    pl.len().alias("n"), r.mean().alias("ratio_mean"),
    (r==0).mean().alias("skip0"), (r>=50).mean().alias("listen_plus"),
    (r==100).mean().alias("full"), pl.col("item_id").n_unique().alias("items"),
    pl.col("track_length_seconds").median().alias("len_med")).sort("is_organic").collect()
for row in g.iter_rows(named=True):
    print("  is_organic=%d: n=%9d  mean_ratio=%5.1f  ratio==0=%5.1f%%  >=50=%5.1f%%  ==100=%5.1f%%  items=%7d  len_med=%s"
      % (row["is_organic"], row["n"], row["ratio_mean"], 100*row["skip0"],
         100*row["listen_plus"], 100*row["full"], row["items"], row["len_med"]))

print("\n=== organic share over time (by 150-day bucket) ===")
t = lf.select((T//(150*17280)).alias("b"), O).group_by("b").agg(
      pl.len().alias("n"), O.mean().alias("org")).sort("b").collect()
for row in t.iter_rows(): print("   days %4d-%4d: n=%8d  organic=%5.1f%%" % (row[0]*150, row[0]*150+149, row[1], 100*row[2]))

print("\n=== per-user mix ===")
u = lf.group_by("uid").agg(O.mean().alias("org"), pl.len().alias("n")).collect()
print("  users with organic share ==0: %d, ==1: %d, between: %d" % (
  (u["org"]==0).sum(), (u["org"]==1).sum(), ((u["org"]>0)&(u["org"]<1)).sum()))
print("  organic share across users: q10=%.2f median=%.2f q90=%.2f" % (
  u["org"].quantile(.1), u["org"].median(), u["org"].quantile(.9)))

print("\n=== popularity concentration by is_organic (top-1%% items share) ===")
for v in [0,1]:
    c = lf.filter(O==v).group_by("item_id").len().collect().sort("len", descending=True)
    k = max(1, c.height//100)
    print("   is_organic=%d: items=%d, top1%% items hold %.1f%% of listens" % (
        v, c.height, 100*c["len"].head(k).sum()/c["len"].sum()))
