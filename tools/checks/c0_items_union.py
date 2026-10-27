import polars as pl, sys, os
raw = sys.argv[1]
parts=[]
for f in ["listens","likes","dislikes","unlikes","undislikes"]:
    d = pl.scan_parquet(os.path.join(raw,f+".parquet")).select("item_id","uid").collect()
    parts.append(d)
    print("%-11s items=%8d users=%6d" % (f, d["item_id"].n_unique(), d["uid"].n_unique()))
al = pl.concat(parts)
print("UNION all events: items=%d users=%d  (card: 934057 items, 10000 users)"
      % (al["item_id"].n_unique(), al["uid"].n_unique()))
