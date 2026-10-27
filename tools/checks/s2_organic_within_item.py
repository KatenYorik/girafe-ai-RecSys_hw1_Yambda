import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# ВНИМАНИЕ (поправка блока 02): DAY = 17280 — это НЕ сутки, а 1/5 суток (4 ч 48 мин).
# Единица timestamp равна 1 секунде, реальные сутки = 86400 единиц (см. t1_time_unit.py).
# Константа оставлена как есть, чтобы числа блока воспроизводились: любой горизонт,
# который этот скрипт печатает «в днях», читается как (дни / 5) реальных суток.
DAY = 17280
TRAIN_DAYS = 1414                 # блок 06: test = дни 1414..1503 (90 дней)
TRAIN_END = TRAIN_DAYS * DAY      # train = дни 0..1413
KEY = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct"]
LF = (pl.scan_parquet(os.path.join(raw, "listens.parquet"))
        .filter(pl.col("timestamp") < TRAIN_END)
        .select(KEY).unique(subset=KEY)
        .with_columns(pl.min_horizontal(pl.col("played_ratio_pct"), pl.lit(100))
                        .cast(pl.Int32).alias("r")))      # клип до 100, блок 01

AGG = [pl.len().alias("n"), pl.col("r").mean().alias("mean"),
       (pl.col("r") <= 1).mean().alias("skip"),
       (pl.col("r") >= 50).mean().alias("lplus"),
       (pl.col("r") == 100).mean().alias("full")]

print("=== 1. сырое сравнение по train (уточнение блока 01, где считалось по всем данным) ===")
g = LF.group_by("is_organic").agg(AGG).sort("is_organic").collect(engine="streaming")
for org, n, m, sk, lp, fu in g.iter_rows():
    print("  is_organic=%d: прослушиваний %d, средний played_ratio %.2f, скипов (<=1%%) %.2f %%, "
          ">=50%% %.2f %%, ==100%% %.2f %%" % (org, n, m, 100*sk, 100*lp, 100*fu))

print("\n=== 2. то же внутри одного трека (контроль репертуара) ===")
it = (LF.group_by(["item_id", "is_organic"]).agg(AGG)
        .collect(engine="streaming")
        .pivot(on="is_organic", index="item_id", values=["n", "mean", "skip", "lplus", "full"]))
it = it.rename({"n_0": "n0", "n_1": "n1", "mean_0": "m0", "mean_1": "m1",
                "skip_0": "sk0", "skip_1": "sk1", "lplus_0": "lp0", "lplus_1": "lp1",
                "full_0": "fu0", "full_1": "fu1"})
print("  треков в train: %d" % it.height)
both = it.filter(pl.col("n0").is_not_null() & pl.col("n1").is_not_null())
print("  из них с прослушиваниями и в рекомендациях, и в органике: %d (%.1f %%)"
      % (both.height, 100 * both.height / it.height))

for kmin in [1, 5, 20, 100]:
    b = both.filter((pl.col("n0") >= kmin) & (pl.col("n1") >= kmin))
    if b.height == 0: continue
    # пулинг: взвешиваем по числу прослушиваний внутри отобранных треков
    tot0, tot1 = b["n0"].sum(), b["n1"].sum()
    pm0 = (b["m0"] * b["n0"]).sum() / tot0;  pm1 = (b["m1"] * b["n1"]).sum() / tot1
    ps0 = (b["sk0"] * b["n0"]).sum() / tot0; ps1 = (b["sk1"] * b["n1"]).sum() / tot1
    pl0 = (b["lp0"] * b["n0"]).sum() / tot0; pl1 = (b["lp1"] * b["n1"]).sum() / tot1
    # парное сравнение: один трек — одно наблюдение, вес min(n0, n1)
    d = (b["m1"] - b["m0"])
    w = pl.Series([min(a, c) for a, c in zip(b["n0"], b["n1"])])
    wd = (d * w).sum() / w.sum()
    print("\n  -- треки с >= %d прослушиваний КАЖДОГО типа: %d треков "
          "(%d рекомендательных + %d органических прослушиваний) --"
          % (kmin, b.height, tot0, tot1))
    print("     пулинг: средний played_ratio рекоменд. %.2f / органика %.2f (разрыв %+.2f)"
          % (pm0, pm1, pm1 - pm0))
    print("     пулинг: скипов рекоменд. %.2f %% / органика %.2f %%; >=50%% %.2f %% / %.2f %%"
          % (100*ps0, 100*ps1, 100*pl0, 100*pl1))
    print("     парно по трекам: средняя разница (органика - рекоменд.) %+.2f "
          "(невзвеш. %+.2f, медиана %+.2f)" % (wd, d.mean(), d.median()))
    print("     треков, где органика завершается ХУЖЕ: %d (%.1f %%); лучше: %d (%.1f %%)"
          % ((d < 0).sum(), 100*(d < 0).sum()/b.height,
             (d > 0).sum(), 100*(d > 0).sum()/b.height))

print("\n=== 3. контроль популярности: разрыв по корзинам числа прослушиваний трека ===")
tn = it.with_columns((pl.col("n0").fill_null(0) + pl.col("n1").fill_null(0)).alias("nt"))
bins = [(1,1),(2,5),(6,20),(21,100),(101,1000),(1001,10**9)]
for lo, hi in bins:
    b = tn.filter((pl.col("nt") >= lo) & (pl.col("nt") <= hi))
    s0 = b.filter(pl.col("n0").is_not_null()); s1 = b.filter(pl.col("n1").is_not_null())
    m0 = (s0["m0"] * s0["n0"]).sum() / max(s0["n0"].sum(), 1)
    m1 = (s1["m1"] * s1["n1"]).sum() / max(s1["n1"].sum(), 1)
    print("  прослушиваний у трека %5d..%-9d треков %7d: рекоменд. %.2f / органика %.2f (%+.2f)"
          % (lo, min(hi, 10**8), b.height, m0, m1, m1 - m0))
