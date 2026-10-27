import polars as pl, sys, os
raw = sys.argv[1]
exp = {"listens":46_467_212, "likes":881_456, "dislikes":107_776}
for f in ["listens","likes","dislikes","unlikes","undislikes","artist_item_mapping","album_item_mapping"]:
    p = os.path.join(raw, f+".parquet")
    lf = pl.scan_parquet(p)
    n = lf.select(pl.len()).collect().item()
    sch = dict(lf.collect_schema())
    note = ""
    if f in exp:
        note = "  card=%d %s" % (exp[f], "OK" if exp[f]==n else "MISMATCH")
    print("%-22s rows=%12d%s" % (f, n, note))
    print("   ", {k:str(v) for k,v in sch.items()})
