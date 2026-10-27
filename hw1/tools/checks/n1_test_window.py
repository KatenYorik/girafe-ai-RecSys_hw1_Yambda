import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
# ВНИМАНИЕ (поправка блока 02): DAY = 17280 — это НЕ сутки, а 1/5 суток (4 ч 48 мин).
# Единица timestamp равна 1 секунде, реальные сутки = 86400 единиц (см. t1_time_unit.py).
# Константа оставлена как есть, чтобы числа блока воспроизводились: любой горизонт,
# который этот скрипт печатает «в днях», читается как (дни / 5) реальных суток.
DAY = 17280          # НЕ сутки: 1/5 суток, см. пометку выше
LAST = 1503          # «день» 1504 — последние 3 часа данных
WINDOWS = [7, 14, 30, 60, 90]

# первое прослушивание каждой пары (uid, item) по всем данным
pairs = (LF.select("uid", "item_id", "timestamp")
           .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
           .collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("ufirst"))
print("пар (uid, item): %d, пользователей с прослушиваниями: %d" % (pairs.height, ufirst.height))

# все прослушивания внутри максимального тестового окна
tst = (LF.filter((pl.col("timestamp") >= (LAST + 1 - max(WINDOWS)) * DAY)
                 & (pl.col("timestamp") < (LAST + 1) * DAY))
         .select("uid", "item_id", "timestamp", "played_ratio_pct")
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left"))
print("прослушиваний в последних %d днях: %d\n" % (max(WINDOWS), tst.height))

QS = [.1, .25, .5, .75, .9]
for L in WINDOWS:
    ts0 = (LAST + 1 - L) * DAY
    d0 = LAST + 1 - L
    print("=== окно test = %d дней (дни %d..%d, train = дни 0..%d) ===" % (L, d0, LAST, d0 - 1))
    n_train = (ufirst["ufirst"] < ts0).sum()
    # только пользователи с историей ДО окна: при длинном окне часть юзеров
    # начинает слушать внутри него и была бы холодной (их исключаем)
    tru = ufirst.filter(pl.col("ufirst") < ts0)["uid"]
    ev = tst.filter(pl.col("timestamp") >= ts0).filter(pl.col("uid").is_in(tru.implode()))
    rel = ev.filter(pl.col("played_ratio_pct") >= 50)
    new = rel.filter(pl.col("t0") >= ts0)
    print("  прослушиваний в test: %d, из них >=50%%: %d (%.1f%%)"
          % (ev.height, rel.height, 100 * rel.height / ev.height))
    print("  из релевантных на новые треки: %d (%.1f%%)"
          % (new.height, 100 * new.height / rel.height))
    print("  пользователей: с историей в train %d, активных в test %d, с релевантным "
          "прослушиванием %d, с релевантным НОВЫМ треком %d (%.1f%% от train-юзеров)"
          % (n_train, ev["uid"].n_unique(), rel["uid"].n_unique(),
             new["uid"].n_unique(), 100 * new["uid"].n_unique() / n_train))
    cnt = new.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))["k"]
    print("  релевантных новых треков на пользователя (среди тех, у кого >=1): "
          "mean=%.1f " % cnt.mean()
          + " ".join("q%02d=%d" % (int(q * 100), cnt.quantile(q)) for q in QS)
          + " max=%d" % cnt.max())
    # с нулями — по всем пользователям, у которых есть история в train
    full = (ufirst.filter(pl.col("ufirst") < ts0).select("uid")
            .join(new.group_by("uid").agg(pl.col("item_id").n_unique().alias("k")),
                  on="uid", how="left").with_columns(pl.col("k").fill_null(0))["k"])
    print("  то же по всем train-пользователям (с нулями): "
          "mean=%.1f " % full.mean()
          + " ".join("q%02d=%d" % (int(q * 100), full.quantile(q)) for q in QS))
    for thr in [1, 3, 5, 10]:
        print("    >=%2d релевантных новых треков: %d пользователей (%.1f%%)"
              % (thr, (full >= thr).sum(), 100 * (full >= thr).sum() / len(full)))
    print()
