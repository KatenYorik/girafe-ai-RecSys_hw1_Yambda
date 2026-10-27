import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# ВНИМАНИЕ (поправка блока 02): этот скрипт считает «сутки» как (timestamp * 5 / 86400),
# то есть исходит из единицы 5 с. Единица равна 1 секунде, поэтому все «дни» здесь
# впятеро короче реальных суток, а горизонт данных — 301 сутки, не 1505 (t1_time_unit.py).
# Поток по реальным суткам и разбор «провалов» — в t2_seasonality.py.
T = pl.col("timestamp")
print("=== global timestamp range per file (units of 5 s) ===")
gmin=gmax=None
for f in ["listens","likes","dislikes","unlikes","undislikes"]:
    a = pl.scan_parquet(os.path.join(raw,f+".parquet")).select(
        T.min().alias("mn"), T.max().alias("mx")).collect().row(0)
    span = (a[1]-a[0])*5/86400
    print("  %-11s min=%9d max=%9d  span=%.2f days" % (f, a[0], a[1], span))
    gmin = a[0] if gmin is None else min(gmin,a[0]); gmax = a[1] if gmax is None else max(gmax,a[1])
print("  ALL        min=%9d max=%9d  span=%.2f days" % (gmin, gmax, (gmax-gmin)*5/86400))

lf = pl.scan_parquet(os.path.join(raw,"listens.parquet"))
u = lf.group_by("uid").agg(T.min().alias("mn"), T.max().alias("mx"), pl.len().alias("n")).collect()
print("\n=== per-user first/last listen (shared origin?) ===")
for lab,c in [("first", "mn"), ("last","mx")]:
    s = u[c]
    print("  %-5s: min=%d  q10=%d  median=%d  q90=%d  max=%d" % (
        lab, s.min(), s.quantile(.1), s.median(), s.quantile(.9), s.max()))
print("  users whose last listen is in final 1%% of span: %d / %d" % (
    (u["mx"] > gmax - 0.01*(gmax-gmin)).sum(), u.height))
print("  users whose first listen is in first 1%% of span: %d / %d" % (
    (u["mn"] < gmin + 0.01*(gmax-gmin)).sum(), u.height))
print("  per-user active span (days): median=%.1f q10=%.1f q90=%.1f" % (
    ((u["mx"]-u["mn"])*5/86400).median(), ((u["mx"]-u["mn"])*5/86400).quantile(.1),
    ((u["mx"]-u["mn"])*5/86400).quantile(.9)))

print("\n=== listens per day ===")
day = lf.select(((T - gmin)*5//86400).alias("d")).group_by("d").len().sort("d").collect()
print("  days covered: %d (d=%d..%d)" % (day.height, day["d"].min(), day["d"].max()))
rows = list(day.iter_rows())
print("  first 5 days:", [(d,n) for d,n in rows[:5]])
print("  last 5 days :", [(d,n) for d,n in rows[-5:]])
print("  median listens/day: %.0f   min day: %s   max day: %s" % (
    day["len"].median(), rows[day["len"].arg_min()], rows[day["len"].arg_max()]))
print("\n=== sortedness check (uid, timestamp) ===")
h = lf.select("uid","timestamp").head(5_000_000).collect()
print("  first 5M rows sorted by (uid,timestamp):",
      h.equals(h.sort(["uid","timestamp"])))
