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
L = 30                       # окно test, по которому смотрим цензуру
TS0 = (LAST + 1 - L) * DAY
KS = [30, 90, 180, 365, 730, 1474]   # длина train-истории, дней (1474 = вся)

# по каждой паре (uid,item): первое прослушивание вообще и последнее ДО теста
pairs_lf = (LF.select("uid", "item_id", "timestamp")
              .group_by(["uid", "item_id"]).agg(
                  pl.col("timestamp").min().alias("t0"),
                  pl.col("timestamp").filter(pl.col("timestamp") < TS0).max().alias("tpre")))
pairs = pairs_lf.collect(engine="streaming")
amap = AM.unique(subset=["item_id"], keep="first").collect()
apre = (pairs.join(amap, on="item_id", how="inner")
             .group_by(["uid", "artist_id"]).agg(pl.col("tpre").max().alias("apre")))
ufirst = pairs.group_by("uid").agg(
    pl.col("t0").min().alias("uf"),
    pl.col("tpre").is_not_null().sum().alias("n_train_items"))
ufirst = ufirst.with_columns((pl.col("uf") // DAY).alias("fday"))

rel = (LF.filter((pl.col("timestamp") >= TS0) & (pl.col("timestamp") < (LAST + 1) * DAY)
                 & (pl.col("played_ratio_pct") >= 50))
         .select("uid", "item_id", "timestamp")
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .join(amap, on="item_id", how="left")
         .join(apre, on=["uid", "artist_id"], how="left")
         .join(ufirst, on="uid", how="left"))

def mark(df, kmin):
    """знакомый трек / новый трек знакомого артиста / новый артист при train = [kmin, TS0)"""
    known_i = pl.col("tpre").is_not_null() & (pl.col("tpre") >= kmin)
    known_a = pl.col("apre").is_not_null() & (pl.col("apre") >= kmin)
    return df.with_columns(
        known_i.alias("fam_item"),
        (~known_i & known_a).alias("new_item_fam_artist"),
        (~known_i & ~known_a & pl.col("artist_id").is_not_null()).alias("new_artist"))

print("=== 1. Новизна в test (окно %d дней) по дню первого прослушивания пользователя ===" % L)
print("    train — вся история до теста; новизна считается только внутри окна данных")
m = mark(rel, 0)
buckets = [(0, 0), (1, 7), (8, 100), (101, 400), (401, 800), (801, 1200), (1201, 1473)]
print("  %-14s %6s %10s %9s | %7s %7s %7s | %6s" %
      ("первый день", "юзеров", "треков tr", "событий", "знаком", "нов.трек", "нов.арт", "медиана нов.треков"))
for lo, hi in buckets:
    s = m.filter((pl.col("fday") >= lo) & (pl.col("fday") <= hi))
    if s.height == 0:
        continue
    u = ufirst.filter((pl.col("fday") >= lo) & (pl.col("fday") <= hi))
    cnt = (s.filter(~pl.col("fam_item")).group_by("uid")
            .agg(pl.col("item_id").n_unique().alias("k"))["k"])
    print("  %-14s %6d %10.0f %9d | %6.1f%% %6.1f%% %6.1f%% | %6.0f" % (
        "%d..%d" % (lo, hi), u.height, u["n_train_items"].median(), s.height,
        100 * s["fam_item"].mean(), 100 * s["new_item_fam_artist"].mean(),
        100 * s["new_artist"].mean(), cnt.median() if len(cnt) else 0))

print("\n=== 2. Сколько добавляет старая история: train обрезан последними K днями ===")
print("  %6s %11s %9s %9s %9s %9s" %
      ("K дней", "треков tr", "знаком", "нов.трек", "нов.арт", "юзеров с нов.арт"))
for K in KS:
    kmin = max(0, TS0 - K * DAY)
    s = mark(rel, kmin)
    ntr = (pairs.filter(pl.col("tpre") >= kmin).group_by("uid").len()["len"])
    print("  %6d %11.0f %8.1f%% %8.1f%% %8.1f%% %9d" % (
        K, ntr.median(), 100 * s["fam_item"].mean(),
        100 * s["new_item_fam_artist"].mean(), 100 * s["new_artist"].mean(),
        s.filter("new_artist")["uid"].n_unique()))

print("\n=== 3. То же только для пользователей, начавших внутри окна (первый день > 30) ===")
sub = rel.filter(pl.col("fday") > 30)
print("  пользователей %d, событий %d" % (sub["uid"].n_unique(), sub.height))
print("  %6s %9s %9s %9s" % ("K дней", "знаком", "нов.трек", "нов.арт"))
for K in KS:
    s = mark(sub, max(0, TS0 - K * DAY))
    print("  %6d %8.1f%% %8.1f%% %8.1f%%" % (
        K, 100 * s["fam_item"].mean(), 100 * s["new_item_fam_artist"].mean(),
        100 * s["new_artist"].mean()))

print("\n=== 4. Псевдо-цензура: у пользователей с первым днём <= 7 режем САМУЮ СТАРУЮ историю ===")
print("    (train = последние K дней вместо ~1474; оценка масштаба левой цензуры)")
cen = rel.filter(pl.col("fday") <= 7)
print("  пользователей %d, событий %d" % (cen["uid"].n_unique(), cen.height))
print("  %6s %9s %9s %9s %9s" % ("K дней", "срезано", "знаком", "нов.трек", "нов.арт"))
for K in [1474, 1200, 900, 600, 300]:
    s = mark(cen, max(0, TS0 - K * DAY))
    print("  %6d %8d %8.1f%% %8.1f%% %8.1f%%" % (
        K, 1474 - K, 100 * s["fam_item"].mean(),
        100 * s["new_item_fam_artist"].mean(), 100 * s["new_artist"].mean()))
