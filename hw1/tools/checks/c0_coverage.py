import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, sys, os
raw = sys.argv[1]
items = (pl.scan_parquet(os.path.join(raw,"listens.parquet")).select("item_id").unique().collect())
print("unique items in 50m listens:", items.height, "(card: 934057)")
s = set(items["item_id"].to_list())

art = pl.scan_parquet(os.path.join(raw,"artist_item_mapping.parquet")).collect()
print("artist_item_mapping: rows=%d unique items=%d" % (art.height, art["item_id"].n_unique()))
a50 = art.filter(pl.col("item_id").is_in(items["item_id"].implode()))
cnt = a50.group_by("item_id").len()
print("  items of 50m with artist: %d (%.2f%%)" % (cnt.height, 100*cnt.height/items.height))
print("  multi-artist items: %d (%.2f%% of covered); max artists per item: %d"
      % ((cnt["len"]>1).sum(), 100*(cnt["len"]>1).mean(), cnt["len"].max()))
print("  distinct artists touching 50m:", a50["artist_id"].n_unique())

alb = pl.scan_parquet(os.path.join(raw,"album_item_mapping.parquet")).collect()
b = alb.filter(pl.col("item_id").is_in(items["item_id"].implode())).group_by("item_id").len()
print("album: items of 50m covered %d (%.2f%%), multi-album %d" % (b.height, 100*b.height/items.height, (b["len"]>1).sum()))

e = pl.read_parquet(os.path.join(raw,"embed_item_ids.parquet"))
hit = e.filter(pl.col("item_id").is_in(items["item_id"].implode())).height
print("embeddings: items of 50m with embed: %d (%.2f%%)" % (hit, 100*hit/items.height))
