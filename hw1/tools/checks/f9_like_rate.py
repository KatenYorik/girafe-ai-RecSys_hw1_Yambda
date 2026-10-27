"""Откуда брать порог «массового лайкера»: сравнение скорости лайков со скоростью прослушивания.

Возражение (27.10.2026): 20 лайков в час — нормальное поведение, за час можно
прослушать альбом и залайкать его целиком. Порог должен опираться на то, сколько треков
физически успеваешь услышать за час, а не выбираться на глаз.

    python -I tools/checks/f9_like_rate.py data/raw data/subsample
"""
import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os

raw, sub = sys.argv[1], sys.argv[2]
TEST0 = 280 * 86400
U = pl.read_parquet(os.path.join(sub, "users.parquet"))["uid"].implode()
amap = (pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
          .unique(subset=["item_id"], keep="first").collect())

ls = (pl.scan_parquet(os.path.join(raw, "listens.parquet"))
        .filter((pl.col("timestamp") < TEST0) & pl.col("uid").is_in(U))
        .select("uid", "item_id", "timestamp", "played_ratio_pct")
        .collect(engine="streaming").with_columns((pl.col("timestamp") // 3600).alias("h")))

print("=== сколько РАЗНЫХ треков человек трогает за час (только непустые часы) ===")
touch = ls.group_by(["uid", "h"]).agg(pl.col("item_id").n_unique().alias("k"))
heard = (ls.filter(pl.col("played_ratio_pct") >= 50)
           .group_by(["uid", "h"]).agg(pl.col("item_id").n_unique().alias("k")))
same = (ls.join(amap, on="item_id", how="left")
          .group_by(["uid", "artist_id", "h"]).agg(pl.col("item_id").n_unique().alias("k")))
for lab, t in [("любое прослушивание", touch), ("дослушано >= 50 %", heard),
               ("одного артиста, любое прослушивание", same)]:
    s = t["k"]
    print("   %-36s медиана %2.0f  q90 %3.0f  q99 %3.0f  q999 %3.0f  максимум %4d"
          % (lab, s.median(), s.quantile(.9), s.quantile(.99), s.quantile(.999), s.max()))

print("\n=== то же для лайков: сколько треков одного артиста лайкается за час ===")
lk = (pl.scan_parquet(os.path.join(raw, "likes.parquet"))
        .filter((pl.col("timestamp") < TEST0) & pl.col("uid").is_in(U))
        .select("uid", "item_id", "timestamp")
        .collect(engine="streaming").join(amap, on="item_id", how="left")
        .with_columns((pl.col("timestamp") // 3600).alias("h")))
bl = lk.group_by(["uid", "artist_id", "h"]).agg(pl.col("item_id").n_unique().alias("k"))
s = bl["k"]
print("   залпов всего %d: медиана %.0f  q90 %.0f  q99 %.0f  q999 %.0f  максимум %d"
      % (bl.height, s.median(), s.quantile(.9), s.quantile(.99), s.quantile(.999), s.max()))
for thr in [10, 15, 20, 30, 60, 100]:
    g = bl.filter(pl.col("k") >= thr)
    print("      залпов >= %3d треков: %5d (%.3f%% залпов), пользователей %4d"
          % (thr, g.height, 100 * g.height / bl.height, g["uid"].n_unique()))

# сколько из залпа человек реально слушал в тот же час
hp = ls.select("uid", "item_id", "h").unique().with_columns(pl.lit(True).alias("same_hour"))
print("\n=== из треков залпа прослушано в тот же час ===")
for thr in [20, 30, 60, 100]:
    big = bl.filter(pl.col("k") >= thr).select("uid", "artist_id", "h")
    ev = (lk.join(big, on=["uid", "artist_id", "h"], how="semi")
            .join(hp, on=["uid", "item_id", "h"], how="left")
            .with_columns(pl.col("same_hour").fill_null(False)))
    if ev.height:
        print("   порог >= %3d: лайков в таких залпах %6d, из них прослушано в тот же час %5.1f%%"
              % (thr, ev.height, 100 * ev["same_hour"].mean()))

print("\n=== профиль группы при разных порогах ===")
cnt = (ls.group_by("uid").agg(pl.len().alias("listens"))
         .join(lk.group_by("uid").agg(pl.len().alias("likes")), on="uid", how="left")
         .with_columns(pl.col("likes").fill_null(0)))
te = pl.read_parquet(os.path.join(sub, "test.parquet"))
ev = (te.filter(pl.col("is_relevant") & pl.col("is_new")).group_by("uid")
        .agg(pl.col("item_id").n_unique().alias("k")))
everheard = ls.select("uid", "item_id").unique().with_columns(pl.lit(True).alias("hh"))
lk2 = lk.join(everheard, on=["uid", "item_id"], how="left").with_columns(pl.col("hh").fill_null(False))
for thr in [20, 30, 60, 100]:
    G = bl.filter(pl.col("k") >= thr)["uid"].unique().implode()
    c = cnt.filter(pl.col("uid").is_in(G))
    g = lk2.filter(pl.col("uid").is_in(G))
    e = ev.filter(pl.col("uid").is_in(G))
    print("   >= %3d: пользователей %4d (%4.1f%%)  их доля лайков %5.1f%%  "
          "не слушали %5.1f%%  лайков медиана %5.0f  оцениваемых %4d  новинок медиана %4.0f"
          % (thr, c.height, 100 * c.height / 3000, 100 * g.height / lk2.height,
             100 * (~g["hh"]).mean(), c["likes"].median(), e.height,
             e["k"].median() if e.height else -1))
print("   (для сравнения: по всем 3 000 лайков медиана %.0f, новинок медиана %.0f)"
      % (cnt["likes"].median(), ev["k"].median()))
