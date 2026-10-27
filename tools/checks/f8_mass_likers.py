"""Массовые лайкеры: кто ставит десятки и сотни лайков одному артисту за час.

Группа A — хотя бы один залп >= 20 различных треков одного артиста в пределах часа.
Группа B — отношение «лайков на прослушивание» выше 0,5.
Считает профиль группы, её вклад в лайки, влияние на метрику и на признаки.

    python -I tools/checks/f8_mass_likers.py data/raw data/subsample
"""
import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os

raw, sub = sys.argv[1], sys.argv[2]
TEST0 = 280 * 86400
U = pl.read_parquet(os.path.join(sub, "users.parquet"))["uid"].implode()
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
amap = AM.unique(subset=["item_id"], keep="first").collect()
asize = amap.group_by("artist_id").len().rename({"len": "tracks_of_artist"})

lk = (pl.scan_parquet(os.path.join(raw, "likes.parquet"))
        .filter((pl.col("timestamp") < TEST0) & pl.col("uid").is_in(U))
        .select("uid", "item_id", "timestamp", "is_organic")
        .collect(engine="streaming").join(amap, on="item_id", how="left"))
ls = (pl.scan_parquet(os.path.join(raw, "listens.parquet"))
        .filter((pl.col("timestamp") < TEST0) & pl.col("uid").is_in(U))
        .select("uid", "item_id", "timestamp", "played_ratio_pct", "is_organic")
        .collect(engine="streaming"))

# группа A: залпы
burst = (lk.with_columns((pl.col("timestamp") // 3600).alias("h"))
           .group_by(["uid", "artist_id", "h"]).agg(pl.col("item_id").n_unique().alias("k")))
A = burst.filter(pl.col("k") >= 20)["uid"].unique()
# группа B: отношение лайков к прослушиваниям
cnt = (ls.group_by("uid").agg(pl.len().alias("listens"))
         .join(lk.group_by("uid").agg(pl.len().alias("likes")), on="uid", how="left")
         .with_columns(pl.col("likes").fill_null(0)))
cnt = cnt.with_columns((pl.col("likes") / pl.col("listens")).alias("ratio"))
B = cnt.filter(pl.col("ratio") > 0.5)["uid"]
both = set(A.to_list()) & set(B.to_list())
print("группа A (залп >= 20 треков одного артиста за час): %d пользователей" % len(A))
print("группа B (лайков на прослушивание > 0,5):            %d пользователей" % len(B))
print("пересечение: %d; объединение: %d из 3 000"
      % (len(both), len(set(A.to_list()) | set(B.to_list()))))

heard = ls.select("uid", "item_id").unique().with_columns(pl.lit(True).alias("h"))
lk2 = lk.join(heard, on=["uid", "item_id"], how="left").with_columns(pl.col("h").fill_null(False))
G = A.implode()
print("\n=== профиль группы A против остальных ===")
for lab, cond in [("группа A", pl.col("uid").is_in(G)), ("остальные", ~pl.col("uid").is_in(G))]:
    g = lk2.filter(cond); c = cnt.filter(cond)
    print("   %-10s пользователей %4d  лайков %7d (%5.1f%% всех)  не слушали %5.1f%%  "
          "органичность %5.1f%%"
          % (lab, c.height, g.height, 100 * g.height / lk2.height,
             100 * (~g["h"]).mean(), 100 * g["is_organic"].mean()))
    print("   %-10s прослушиваний медиана %6.0f  лайков медиана %5.0f  отношение медиана %.3f"
          % ("", c["listens"].median(), c["likes"].median(), c["ratio"].median()))

print("\n=== слушают ли они как обычные люди ===")
for lab, cond in [("группа A", pl.col("uid").is_in(G)), ("остальные", ~pl.col("uid").is_in(G))]:
    g = ls.filter(cond)
    days = g.with_columns((pl.col("timestamp") // 86400).alias("d")).group_by("uid").agg(
        pl.col("d").n_unique().alias("ad"))
    print("   %-10s played_ratio медиана %4.0f  доля >=50%% %5.1f%%  органичность %5.1f%%  "
          "активных суток медиана %3.0f"
          % (lab, g["played_ratio_pct"].median(), 100 * (g["played_ratio_pct"] >= 50).mean(),
             100 * g["is_organic"].mean(), days["ad"].median()))

print("\n=== на каких артистов идут их лайки ===")
for lab, cond in [("группа A", pl.col("uid").is_in(G)), ("остальные", ~pl.col("uid").is_in(G))]:
    g = lk.filter(cond).join(asize, on="artist_id", how="left")
    print("   %-10s треков у артиста: медиана %6.0f  q90 %7.0f  доля лайков на артистах "
          ">10 000 треков %5.1f%%"
          % (lab, g["tracks_of_artist"].median(), g["tracks_of_artist"].quantile(.9),
             100 * g.filter(pl.col("tracks_of_artist") > 10_000).height / g.height))

print("\n=== искажают ли они метрику ===")
te = pl.read_parquet(os.path.join(sub, "test.parquet"))
ev = (te.filter(pl.col("is_relevant") & pl.col("is_new")).group_by("uid")
        .agg(pl.col("item_id").n_unique().alias("k")))
inA = ev.filter(pl.col("uid").is_in(G))
print("   оцениваемых всего %d; из группы A — %d (%.1f%%)"
      % (ev.height, inA.height, 100 * inA.height / ev.height))
if inA.height:
    print("   релевантных новинок у них медиана %.0f против %.0f у остальных"
          % (inA["k"].median(), ev.filter(~pl.col("uid").is_in(G))["k"].median()))
