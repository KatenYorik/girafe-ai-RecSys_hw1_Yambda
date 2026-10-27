import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
lf = pl.scan_parquet(os.path.join(raw,"listens.parquet"))
N = 46467212
r = pl.col("played_ratio_pct"); L = pl.col("track_length_seconds")
agg = lf.select(
    r.min().alias("r_min"), r.max().alias("r_max"), r.mean().alias("r_mean"),
    r.median().alias("r_med"),
    (r==0).sum().alias("r_eq0"), (r<=1).sum().alias("r_le1"),
    (r>=50).sum().alias("r_ge50"), (r>100).sum().alias("r_gt100"),
    (r==100).sum().alias("r_eq100"), (r>200).sum().alias("r_gt200"),
    L.min().alias("L_min"), L.max().alias("L_max"), L.median().alias("L_med"),
    (L==0).sum().alias("L_eq0"), (L<30).sum().alias("L_lt30"),
    (L>1800).sum().alias("L_gt30min"), (L>7200).sum().alias("L_gt2h"),
).collect()
d = agg.to_dicts()[0]
print("played_ratio_pct: min=%(r_min)s max=%(r_max)s mean=%(r_mean).1f median=%(r_med)s" % d)
for k,lab in [("r_eq0","== 0"),("r_le1","<= 1"),("r_ge50",">= 50 (Listen+)"),
              ("r_eq100","== 100"),("r_gt100","> 100"),("r_gt200","> 200")]:
    print("   %-16s %10d  %6.2f%%" % (lab, d[k], 100*d[k]/N))
print("track_length_seconds: min=%(L_min)s max=%(L_max)s median=%(L_med)s" % d)
for k,lab in [("L_eq0","== 0"),("L_lt30","< 30 s"),("L_gt30min","> 30 min"),("L_gt2h","> 2 h")]:
    print("   %-16s %10d  %6.2f%%" % (lab, d[k], 100*d[k]/N))
print("\nquantiles played_ratio_pct:")
q = lf.select([r.quantile(p).alias(str(p)) for p in [.01,.1,.25,.5,.75,.9,.99,.999]]).collect()
print("  ", {k:v for k,v in zip(q.columns, q.row(0))})
print("quantiles track_length_seconds:")
q = lf.select([L.quantile(p).alias(str(p)) for p in [.001,.01,.1,.5,.9,.99,.999]]).collect()
print("  ", {k:v for k,v in zip(q.columns, q.row(0))})
print("\nratio>100 buckets:")
b = (lf.filter(r>100).select(((r-1)//100*100).alias("b")).group_by("b").len()
       .sort("b").collect())
for row in b.head(8).iter_rows(): print("   %4d-%4d : %d" % (row[0]+1, row[0]+100, row[1]))
print("\nL==0: distinct items=%s, share of ratio>=50 there=%s" % (
    lf.filter(L==0).select(pl.col("item_id").n_unique()).collect().item(),
    lf.filter(L==0).select((r>=50).mean()).collect().item()))
