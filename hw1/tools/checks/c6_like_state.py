import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
def load(f, ev):
    return pl.read_parquet(os.path.join(raw,f+".parquet")).select(
        "uid","item_id","timestamp").with_columns(pl.lit(ev).alias("ev"))
lk, ul = load("likes","like"), load("unlikes","unlike")
dl, ud = load("dislikes","dislike"), load("undislikes","undislike")

for pos, neg, name in [(lk, ul, "like/unlike"), (dl, ud, "dislike/undislike")]:
    a = pos.join(neg, on=["uid","item_id"], how="semi").select(["uid","item_id"]).unique().height
    orphan = neg.join(pos, on=["uid","item_id"], how="anti").select(["uid","item_id"]).unique().height
    allp = pl.concat([pos, neg]).sort(["uid","item_id","timestamp"])
    last = allp.group_by(["uid","item_id"]).agg(pl.col("ev").last().alias("final"))
    cnt = last.group_by("final").len().sort("final")
    print("=== %s ===" % name)
    print("   pairs with %s events: %d ; pairs with reverting events: %d"
          % (name.split("/")[0], pos.select(["uid","item_id"]).unique().height,
             neg.select(["uid","item_id"]).unique().height))
    print("   pairs having both: %d ; reverts with NO matching positive: %d" % (a, orphan))
    print("   final state by last event:", dict(cnt.iter_rows()))

both = lk.select(["uid","item_id"]).unique().join(
        dl.select(["uid","item_id"]).unique(), on=["uid","item_id"], how="semi")
print("\npairs that are BOTH liked and disliked (ever): %d" % both.height)
