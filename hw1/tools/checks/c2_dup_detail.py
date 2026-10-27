import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
lf = pl.scan_parquet(os.path.join(raw,"listens.parquet"))
d = (lf.group_by(["uid","item_id","timestamp"]).agg(
        pl.len().alias("n"), pl.col("played_ratio_pct").max().alias("r_max"))
     .filter(pl.col("n")>1).collect(engine="streaming"))
h = d.group_by("n").len().sort("n")
print("n_repeat -> keys")
for r in h.head(10).iter_rows(): print("  %4d -> %d" % r)
print("  n>=10 :", d.filter(pl.col("n")>=10).height, "keys")
print("  n>=50 :", d.filter(pl.col("n")>=50).height, "keys")
print("  max n :", d["n"].max())
hv = d.filter(pl.col("n")>=20)
print("  keys n>=20: %d, distinct uids: %d, max played_ratio there: %s"
      % (hv.height, hv["uid"].n_unique(), hv["r_max"].max()))
