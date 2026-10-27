import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, sys, os
raw = sys.argv[1]
KEY = ["uid","item_id","timestamp"]
for f in ["listens","likes","dislikes","unlikes","undislikes"]:
    lf = pl.scan_parquet(os.path.join(raw,f+".parquet"))
    cols = list(lf.collect_schema())
    n = lf.select(pl.len()).collect().item()
    # пропуски
    nulls = lf.select([pl.col(c).null_count().alias(c) for c in cols]).collect().row(0)
    nk = lf.group_by(KEY).len().filter(pl.col("len")>1).select(
            pl.len().alias("dup_keys"), (pl.col("len")-1).sum().alias("extra_rows"),
            pl.col("len").max().alias("max_rep")).collect(engine="streaming").row(0)
    full = lf.group_by(cols).len().filter(pl.col("len")>1).select(
            (pl.col("len")-1).sum().alias("extra_rows")).collect(engine="streaming").row(0)[0] or 0
    print("=== %s  rows=%d" % (f, n))
    print("    nulls:", dict(zip(cols, nulls)))
    print("    дубли ключа (uid,item_id,timestamp): ключей=%s, лишних строк=%s (%.4f%%), макс повтор=%s"
          % (nk[0], nk[1], 100*(nk[1] or 0)/n, nk[2]))
    print("    полные дубли (все колонки): лишних строк=%d (%.4f%%)" % (full, 100*full/n))
    sys.stdout.flush()
