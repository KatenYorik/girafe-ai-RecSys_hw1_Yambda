"""Преждевременные лайки: лайк раньше НАЧАЛА первого воспроизведения трека.

Граница — решение от 27.10.2026. Прослушивание помечается концом
воспроизведения (блок 09, f5), поэтому начало = timestamp - проигранные секунды.
Пара без прослушиваний у этого пользователя считается преждевременной по построению.

Срезы: популярность артиста, знаком ли артист пользователю до лайка,
пресейв (лайк раньше глобального дебюта трека) и залпы по одному артисту.

    python -I tools/checks/f6_premature_likes.py data/raw data/subsample
"""
import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os

raw, sub = sys.argv[1], sys.argv[2]
TEST0 = 280 * 86400                      # блок 02: test = сутки 280..300
users = pl.read_parquet(os.path.join(sub, "users.parquet"))["uid"]
U = users.implode()
cat = pl.read_parquet(os.path.join(sub, "catalog.parquet"))["item_id"]

LS = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
amap = AM.unique(subset=["item_id"], keep="first").collect()

# --- лайки до теста: первое событие лайка на паре
lk = (pl.scan_parquet(os.path.join(raw, "likes.parquet"))
        .filter((pl.col("timestamp") < TEST0) & pl.col("uid").is_in(U))
        .select("uid", "item_id", "timestamp", "is_organic")
        .collect(engine="streaming")
        .sort("timestamp").group_by(["uid", "item_id"])
        .agg(pl.col("timestamp").first().alias("t_like"),
             pl.col("is_organic").first().alias("org")))

# --- первое прослушивание пары у этого пользователя и начало воспроизведения
mine = (LS.filter(pl.col("uid").is_in(U))
          .select("uid", "item_id", "timestamp", "track_length_seconds", "played_ratio_pct")
          .collect(engine="streaming").sort("timestamp"))
first = (mine.group_by(["uid", "item_id"])
             .agg(pl.col("timestamp").first().alias("t_end"),
                  (pl.col("track_length_seconds").first() *
                   pl.col("played_ratio_pct").first() / 100).alias("played")))
first = first.with_columns((pl.col("t_end") - pl.col("played")).alias("t_start"))

d = lk.join(first, on=["uid", "item_id"], how="left").with_columns(
        pl.when(pl.col("t_start").is_null()).then(True)
          .otherwise(pl.col("t_like") < pl.col("t_start")).alias("prem"))
tot = d.height
print("пар с лайком до теста: %d" % tot)
print("преждевременных (лайк раньше начала первого воспроизведения): %d (%.1f%%)"
      % (d["prem"].sum(), 100 * d["prem"].mean()))
print("  из них вообще без прослушивания: %d" % d.filter(pl.col("t_start").is_null()).height)
print("органичность: у преждевременных %.1f%%, у остальных %.1f%%"
      % (100 * d.filter(pl.col("prem"))["org"].mean(),
         100 * d.filter(~pl.col("prem"))["org"].mean()))

d = d.join(amap, on="item_id", how="left")

# --- срез 1: популярность артиста (слушатели в пуле 3 000 до теста)
pop = (mine.filter(pl.col("timestamp") < TEST0).join(amap, on="item_id", how="inner")
           .group_by("artist_id").agg(pl.col("uid").n_unique().alias("listeners")))
print("\n=== срез 1: доля преждевременных лайков по популярности артиста ===")
print("   (популярность = сколько из 3 000 пользователей слушали артиста до теста)")
dp = d.join(pop, on="artist_id", how="left").with_columns(pl.col("listeners").fill_null(0))
bins = [(0, 0), (1, 5), (6, 20), (21, 100), (101, 500), (501, 10**9)]
for lo, hi in bins:
    g = dp.filter(pl.col("listeners").is_between(lo, hi))
    if g.height == 0:
        continue
    lab = "%d+" % lo if hi > 10**8 else "%d–%d" % (lo, hi)
    print("   слушателей %-9s лайков %7d  преждевременных %6.1f%%  органичность %5.1f%%"
          % (lab, g.height, 100 * g["prem"].mean(), 100 * g["org"].mean()))

# --- срез 2: знаком ли артист пользователю до лайка
ua = (mine.join(amap, on="item_id", how="inner")
          .group_by(["uid", "artist_id"]).agg(pl.col("timestamp").min().alias("a_first")))
da = d.join(ua, on=["uid", "artist_id"], how="left").with_columns(
        (pl.col("a_first") < pl.col("t_like")).fill_null(False).alias("art_known"))
print("\n=== срез 2: знаком ли артист пользователю ДО лайка ===")
for known in [True, False]:
    g = da.filter(pl.col("art_known") == known)
    print("   артист %-10s лайков %7d (%4.1f%%)  преждевременных %6.1f%%"
          % ("знаком" if known else "новый", g.height, 100 * g.height / tot,
             100 * g["prem"].mean()))
print("   среди ПРЕЖДЕВРЕМЕННЫХ лайков артист уже был знаком у %.1f%%"
      % (100 * da.filter(pl.col("prem"))["art_known"].mean()))

# --- срез 3: пресейвы — лайк раньше глобального дебюта трека
deb = (LS.select("item_id", "timestamp").group_by("item_id")
         .agg(pl.col("timestamp").min().alias("debut")).collect(engine="streaming"))
dd = d.join(deb, on="item_id", how="left")
pre = dd.filter(pl.col("t_like") < pl.col("debut"))
print("\n=== срез 3: пресейвы (лайк раньше первого прослушивания трека КЕМ УГОДНО) ===")
print("   таких пар: %d (%.2f%% всех лайков, %.2f%% преждевременных)"
      % (pre.height, 100 * pre.height / tot,
         100 * pre.height / max(1, d["prem"].sum())))
if pre.height:
    lead = (pre["debut"] - pre["t_like"]) / 86400
    print("   опережение дебюта: медиана %.1f суток, q75 %.1f, q90 %.1f, максимум %.1f"
          % (lead.median(), lead.quantile(.75), lead.quantile(.9), lead.max()))
    print("   органичность таких лайков: %.1f%%" % (100 * pre["org"].mean()))
    print("   различных пользователей: %d, различных треков: %d, артистов: %d"
          % (pre["uid"].n_unique(), pre["item_id"].n_unique(), pre["artist_id"].n_unique()))

# --- срез 4: залпы по одному артисту (подпись пресейва релиза без альбомов)
print("\n=== срез 4: залпы — сколько треков одного артиста лайкается в пределах часа ===")
for name, g in [("преждевременные", d.filter(pl.col("prem"))),
                ("обычные", d.filter(~pl.col("prem")))]:
    b = (g.with_columns((pl.col("t_like") // 3600).alias("h"))
          .group_by(["uid", "artist_id", "h"]).agg(pl.col("item_id").n_unique().alias("k")))
    print("   %-16s залпов %7d  медиана %d  доля залпов с >=3 треками %5.1f%%  максимум %d"
          % (name, b.height, b["k"].median(), 100 * b.filter(pl.col("k") >= 3).height / b.height,
             b["k"].max()))

# --- то же внутри каталога
dc = d.filter(pl.col("item_id").is_in(cat.implode()))
print("\nвнутри каталога (10 000 треков): лайков %d, преждевременных %.1f%%"
      % (dc.height, 100 * dc["prem"].mean()))

# --- срез 5: популярность артиста ПРИ КОНТРОЛЕ знакомства
print("\n=== срез 5: доля преждевременных при контроле знакомства с артистом ===")
dx = da.join(pop, on="artist_id", how="left").with_columns(pl.col("listeners").fill_null(0))
print("   %-11s | %s" % ("слушателей", " | ".join("%14s" % s for s in ["артист знаком", "артист новый"])))
for lo, hi in [(1, 20), (21, 100), (101, 500), (501, 10**9)]:
    row = []
    for known in [True, False]:
        g = dx.filter(pl.col("listeners").is_between(lo, hi) & (pl.col("art_known") == known))
        row.append("%5.1f%% (n=%6d)" % (100 * g["prem"].mean(), g.height) if g.height else "—")
    lab = "%d+" % lo if hi > 10**8 else "%d–%d" % (lo, hi)
    print("   %-11s | %s" % (lab, " | ".join(row)))
lead = (pre["debut"] - pre["t_like"]) / 86400
for thr in [1, 7, 30]:
    k = (lead > thr).sum()
    print("\n   пресейвов с опережением дебюта больше %2d суток: %d (%.2f%% всех лайков)"
          % (thr, k, 100 * k / tot))
