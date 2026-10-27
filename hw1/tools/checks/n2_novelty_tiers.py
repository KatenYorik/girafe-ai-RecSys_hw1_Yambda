import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
# ВНИМАНИЕ (поправка блока 02): DAY = 17280 — это НЕ сутки, а 1/5 суток (4 ч 48 мин).
# Единица timestamp равна 1 секунде, реальные сутки = 86400 единиц (см. t1_time_unit.py).
# Константа оставлена как есть, чтобы числа блока воспроизводились: любой горизонт,
# который этот скрипт печатает «в днях», читается как (дни / 5) реальных суток.
DAY = 17280
LAST = 1503
WINDOWS = [7, 14, 30, 60, 90]

# первое прослушивание пары (uid, item) и пары (uid, artist)
pairs_lf = (LF.select("uid", "item_id", "timestamp")
              .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0")))
pairs = pairs_lf.collect(engine="streaming")
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("uf"))
afirst = (pairs_lf.join(AM, on="item_id", how="inner")
            .group_by(["uid", "artist_id"]).agg(pl.col("t0").min().alias("a0"))
            .collect(engine="streaming"))
print("пар (uid, item): %d; пар (uid, artist): %d" % (pairs.height, afirst.height))

# прослушивания в максимальном окне: трек -> его артист (у трека практически всегда один)
amap = AM.unique(subset=["item_id"], keep="first").collect()
tst = (LF.filter((pl.col("timestamp") >= (LAST + 1 - max(WINDOWS)) * DAY)
                 & (pl.col("timestamp") < (LAST + 1) * DAY))
         .select("uid", "item_id", "timestamp", "played_ratio_pct", "is_organic")
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .join(amap, on="item_id", how="left")
         .join(afirst, on=["uid", "artist_id"], how="left"))
print("прослушиваний в последних %d днях: %d\n" % (max(WINDOWS), tst.height))

def tiers(df, ts0):
    return df.with_columns(
        pl.when(pl.col("t0") < ts0).then(pl.lit("1 знакомый трек"))
          .when(pl.col("artist_id").is_null()).then(pl.lit("4 артист неизвестен"))
          .when(pl.col("a0") < ts0).then(pl.lit("2 новый трек знакомого артиста"))
          .otherwise(pl.lit("3 новый артист")).alias("tier"))

for L in WINDOWS:
    ts0 = (LAST + 1 - L) * DAY
    print("=== окно test = %d дней (дни %d..%d) ===" % (L, LAST + 1 - L, LAST))
    # исключаем пользователей, начавших слушать внутри окна: они были бы холодными
    tru = ufirst.filter(pl.col("uf") < ts0)["uid"]
    rel = tiers(tst.filter((pl.col("timestamp") >= ts0)
                           & (pl.col("played_ratio_pct") >= 50)
                           & (pl.col("uid").is_in(tru.implode()))), ts0)
    tot = rel.height
    g = rel.group_by("tier").agg(pl.len().alias("n"),
                                 pl.col("uid").n_unique().alias("users"),
                                 pl.struct("uid", "item_id").n_unique().alias("pairs"),
                                 pl.col("is_organic").mean().alias("org")).sort("tier")
    print("  релевантных (>=50%%) прослушиваний в test: %d" % tot)
    for t, n, us, pr, org in g.iter_rows():
        print("    %-32s %8d (%5.1f%%)  пользователей %4d  пар (uid,item) %7d  органика %.0f%%"
              % (t, n, 100 * n / tot, us, pr, 100 * org))
    na = rel.filter(pl.col("tier") == "3 новый артист")
    print("  срез «новый артист»: события %d, пользователей %d, "
          "различных новых артистов у пользователей %d"
          % (na.height, na["uid"].n_unique(), na.select(pl.struct("uid", "artist_id").n_unique()).item()))
    kc = na.group_by("uid").agg(pl.struct("item_id").n_unique().alias("k"))["k"]
    ka = na.group_by("uid").agg(pl.col("artist_id").n_unique().alias("k"))["k"]
    print("    новых треков новых артистов на пользователя (у кого >=1): "
          "median=%d q25=%d q75=%d q90=%d; новых артистов: median=%d q75=%d q90=%d"
          % (kc.median(), kc.quantile(.25), kc.quantile(.75), kc.quantile(.9),
             ka.median(), ka.quantile(.75), ka.quantile(.9)))
    for thr in [1, 3, 5]:
        print("    пользователей с >=%d новыми артистами: %d" % (thr, (ka >= thr).sum()))
    # органика отдельно
    print("  доли ступеней внутри органики / внутри рекомендаций:")
    for v, lab in [(1, "органика"), (0, "рекомендации")]:
        s = rel.filter(pl.col("is_organic") == v)
        gg = s.group_by("tier").agg(pl.len().alias("n")).sort("tier")
        print("    %-13s (%7d событий): " % (lab, s.height)
              + "  ".join("%s %.1f%%" % (t[0], 100 * n / s.height) for t, n in gg.iter_rows()))
    print()
