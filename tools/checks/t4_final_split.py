import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# Блок 02. Числа по ПРИНЯТОМУ сплиту (решение от 22.10.2026):
#   train = сутки   0…258 (37 недель)
#   val   = сутки 259…279 (3 недели)
#   test  = сутки 280…300 (3 недели; последние сутки заполнены на 92,6 %)
# Единица timestamp = 1 с, сутки = 86400 единиц (t1_time_unit.py).
# Скрипт — единственный источник чисел по принятому окну: n1…n4 блока 06 оставлены
# на старой сетке 17280, чтобы опубликованные там числа воспроизводились.
DAY = 86400
TRAIN_END = 259 * DAY          # = 22 377 600
TEST0 = 280 * DAY              # = 24 192 000
TMAX = 26_000_000
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))

pairs_lf = (LF.select("uid", "item_id", "timestamp")
              .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0")))
pairs = pairs_lf.collect(engine="streaming")
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("uf"))
afirst = (pairs_lf.join(AM, on="item_id", how="inner")
            .group_by(["uid", "artist_id"]).agg(pl.col("t0").min().alias("a0"))
            .collect(engine="streaming"))
amap = AM.unique(subset=["item_id"], keep="first").collect()
print("пар (uid, item): %d; пар (uid, artist): %d; пользователей с прослушиваниями: %d"
      % (pairs.height, afirst.height, ufirst.height))

TOTAL = LF.select(pl.len()).collect().item()
n_train = LF.filter(pl.col("timestamp") < TRAIN_END).select(pl.len()).collect().item()
n_val = (LF.filter((pl.col("timestamp") >= TRAIN_END) & (pl.col("timestamp") < TEST0))
           .select(pl.len()).collect().item())
n_test = LF.filter(pl.col("timestamp") >= TEST0).select(pl.len()).collect().item()
print("прослушиваний: всего %d; train %d (%.1f %%); val %d (%.1f %%); test %d (%.1f %%)"
      % (TOTAL, n_train, 100 * n_train / TOTAL, n_val, 100 * n_val / TOTAL,
         n_test, 100 * n_test / TOTAL))

ev_all = (LF.filter(pl.col("timestamp") >= TRAIN_END)
            .select("uid", "item_id", "timestamp", "played_ratio_pct", "is_organic")
            .collect(engine="streaming")
            .join(pairs, on=["uid", "item_id"], how="left")
            .join(amap, on="item_id", how="left")
            .join(afirst, on=["uid", "artist_id"], how="left"))


def tiers(df, ts0):
    return df.with_columns(
        pl.when(pl.col("t0") < ts0).then(pl.lit("1 знакомый трек"))
          .when(pl.col("artist_id").is_null()).then(pl.lit("4 артист неизвестен"))
          .when(pl.col("a0") < ts0).then(pl.lit("2 новый трек знакомого артиста"))
          .otherwise(pl.lit("3 новый артист")).alias("tier"))


def report(lab, ts0, ts1, d0, d1):
    print("\n" + "=" * 72)
    print("=== %s: сутки %d…%d (%d суток), история — всё до суток %d ==="
          % (lab, d0, d1, d1 - d0 + 1, d0))
    tru = ufirst.filter(pl.col("uf") < ts0)["uid"]
    n_hist, cold = len(tru), (ufirst["uf"] >= ts0).sum()
    ev = (ev_all.filter((pl.col("timestamp") >= ts0) & (pl.col("timestamp") < ts1))
                .filter(pl.col("uid").is_in(tru.implode())))
    rel = tiers(ev.filter(pl.col("played_ratio_pct") >= 50), ts0)
    new = rel.filter(pl.col("t0") >= ts0)
    print("  пользователей с историей до окна %d, холодных (начали внутри) %d"
          % (n_hist, cold))
    print("  прослушиваний %d, релевантных (>=50 %%) %d (%.1f %%), "
          "из них на новые треки %d (%.1f %%)"
          % (ev.height, rel.height, 100 * rel.height / ev.height,
             new.height, 100 * new.height / rel.height))
    k = new.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))
    print("  активных в окне %d, с релевантным %d, ОЦЕНИВАЕМЫХ %d (%.1f %% от имеющих историю)"
          % (ev["uid"].n_unique(), rel["uid"].n_unique(), k.height,
             100 * k.height / n_hist))
    s = k["k"]
    print("  релевантных новых треков на оцениваемого: среднее %.1f, "
          "q25=%d median=%d q75=%d q90=%d max=%d"
          % (s.mean(), s.quantile(.25), s.median(), s.quantile(.75), s.quantile(.9), s.max()))
    full = (ufirst.filter(pl.col("uf") < ts0).select(pl.col("uid"))
            .join(k, on="uid", how="left").with_columns(pl.col("k").fill_null(0))["k"])
    for thr in [1, 3, 5, 10]:
        print("    >= %2d релевантных новинок: %d пользователей (%.1f %%)"
              % (thr, (full >= thr).sum(), 100 * (full >= thr).mean()))
    print("  искажение от усреднения по всем, а не по оцениваемым: в %.2f раза"
          % (n_hist / k.height))

    tot = rel.height
    g = rel.group_by("tier").agg(pl.len().alias("n"),
                                 pl.col("uid").n_unique().alias("users"),
                                 pl.struct("uid", "item_id").n_unique().alias("prs"),
                                 pl.col("is_organic").mean().alias("org")).sort("tier")
    print("  три ступени новизны (от релевантных прослушиваний окна):")
    for t, n, us, pr, org in g.iter_rows():
        print("    %-32s %8d (%5.1f %%)  пользователей %4d  пар %7d  органика %.0f %%"
              % (t, n, 100 * n / tot, us, pr, 100 * org))
    na = rel.filter(pl.col("tier") == "3 новый артист")
    ka = na.group_by("uid").agg(pl.col("artist_id").n_unique().alias("k"))["k"]
    kc = na.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))["k"]
    print("  срез «новый артист»: событий %d, пользователей %d, пар (uid, artist) %d"
          % (na.height, na["uid"].n_unique(),
             na.select(pl.struct("uid", "artist_id").n_unique()).item()))
    print("    новых артистов на пользователя: median=%d q75=%d q90=%d; "
          "новых треков новых артистов: median=%d q75=%d"
          % (ka.median(), ka.quantile(.75), ka.quantile(.9), kc.median(), kc.quantile(.75)))
    for thr in [1, 3, 5]:
        print("    пользователей с >= %d новыми артистами: %d" % (thr, (ka >= thr).sum()))
    print("  ступени внутри органики / рекомендаций:")
    for v, l2 in [(1, "органика"), (0, "рекомендации")]:
        sub = rel.filter(pl.col("is_organic") == v)
        gg = sub.group_by("tier").agg(pl.len().alias("n")).sort("tier")
        print("    %-13s (%8d событий): " % (l2, sub.height)
              + "  ".join("%s %.1f %%" % (t[0], 100 * n / sub.height)
                           for t, n in gg.iter_rows()))


report("TEST (метрика работы)", TEST0, TMAX + 1, 280, 300)
report("VAL (подбор гиперпараметров, обучение только на train)",
       TRAIN_END, TEST0, 259, 279)
