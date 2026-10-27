import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# Блок 03. Воронка фильтрации пользователей и размер случайной подвыборки.
# Единица timestamp = 1 с, сутки = 86400 (блок 02).
DAY = 86400
NDAYS = 301
NUSERS_ALL = 10_000
SEED = 42
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))

# два варианта границ: принятая блоком 06 и предварительная блока 02
VARIANTS = [("блок 06 (test 18,1 суток)", 1414 * 17280, None),
            ("блок 02 (test 21 / val 21)", 280 * DAY, 259 * DAY)]

pairs = (LF.select("uid", "item_id", "timestamp")
           .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
           .collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("ufirst"))
NL = ufirst.height
print("пользователей с прослушиваниями %d, пар (uid, item) всего %d"
      % (NL, pairs.height))

# признаки пользователя для фильтров
feat = (LF.select("uid", "timestamp", "played_ratio_pct")
          .group_by("uid")
          .agg(pl.len().alias("n"),
               (pl.col("played_ratio_pct") == 0).mean().alias("zero"),
               ((pl.col("timestamp") % DAY) // 3600 <= 4).mean().alias("night"))
          .collect(engine="streaming"))

tst_raw = (LF.filter(pl.col("timestamp") >= 259 * DAY)
             .select("uid", "item_id", "timestamp", "played_ratio_pct")
             .collect(engine="streaming")
             .join(pairs, on=["uid", "item_id"], how="left"))

for LAB, TS0, VS0 in VARIANTS:
    print("\n" + "=" * 70)
    print("=== ВОРОНКА: %s ===" % LAB)
    print("  граница test = %d (сутки %.2f)%s"
          % (TS0, TS0 / DAY,
             "" if VS0 is None else ", граница val = %d (сутки %.2f)" % (VS0, VS0 / DAY)))
    hist = ufirst.filter(pl.col("ufirst") < TS0)["uid"]
    rel = (tst_raw.filter(pl.col("timestamp") >= TS0)
                  .filter(pl.col("played_ratio_pct") >= 50))
    new = rel.filter(pl.col("t0") >= TS0).filter(pl.col("uid").is_in(hist.implode()))
    evalu = new["uid"].unique()
    deg = feat.filter(pl.col("zero") > 0.5)["uid"]
    # для метрики на test историей служит всё до test (train + val): финальная
    # модель обучается на train + val, поэтому порог активности считаем по ней
    ntr = (LF.filter(pl.col("timestamp") < TS0).select("uid", "timestamp")
             .group_by("uid").len().collect(engine="streaming"))
    if VS0 is not None:
        nva = (LF.filter(pl.col("timestamp") < VS0).select("uid")
                 .group_by("uid").len().collect(engine="streaming"))
    print("  %-52s %6s %8s" % ("шаг", "осталось", "выпало"))
    step = [("всего пользователей в данных (объединение событий)", NUSERS_ALL, 0)]
    step.append(("есть хотя бы одно прослушивание", NL, NUSERS_ALL - NL))
    s2 = len(hist)
    step.append(("есть история до начала test (иначе холодный)", s2, NL - s2))
    keep = set(hist.to_list()) - set(deg.to_list())
    s3 = len(keep)
    step.append(("убрать «перелистывающих» (доля played_ratio == 0 > 50 %%)",
                 s3, s2 - s3))
    act = set(ntr.filter(pl.col("len") >= 5)["uid"].to_list())
    s4 = len(keep & act)
    step.append(("есть >= 5 прослушиваний в истории до test", s4, s3 - s4))
    s5 = len((keep & act) & set(evalu.to_list()))
    step.append(("есть >= 1 релевантный НОВЫЙ трек в test = ОЦЕНИВАЕМЫЕ",
                 s5, s4 - s5))
    for lab, left, out in step:
        print("  %-52s %6d %8d" % (lab, left, out))
    print("  (без фильтров оцениваемых было бы %d)" % len(evalu))
    print("\n  пороги по активности в train (после снятия «перелистывающих»):")
    print("    %-10s %8s %12s %14s" % ("порог", "польз.", "оцениваемых", "теряем оцен."))
    for thr in [1, 5, 10, 20, 30, 50, 100, 200]:
        a = set(ntr.filter(pl.col("len") >= thr)["uid"].to_list())
        kk = keep & a
        ev = len(kk & set(evalu.to_list()))
        print("    >=%-8d %8d %12d %13d" % (thr, len(kk), ev, s5 - ev))
    if VS0 is not None:
        print("  отдельно — пул для подбора на val (модель обучается только на train,")
        print("  сутки 0..%d): пользователей с >= 5 прослушиваниями в train %d,"
              % (VS0 // DAY - 1, len(keep & set(nva.filter(pl.col("len") >= 5)["uid"]
                                                 .to_list()))))
        print("  то есть %d из %d имеют историю только внутри val и для подбора "
              "гиперпараметров недоступны"
              % (s3 - len(keep & set(nva.filter(pl.col("len") >= 5)["uid"].to_list())),
                 s3))

# ---------------- подвыборка ----------------
print("\n" + "=" * 70)
print("=== РАЗМЕР СЛУЧАЙНОЙ ПОДВЫБОРКИ (границы блока 02: train = сутки 0..258) ===")
TS0, VS0 = 280 * DAY, 259 * DAY
hist = ufirst.filter(pl.col("ufirst") < TS0)["uid"]
deg = set(feat.filter(pl.col("zero") > 0.5)["uid"].to_list())
ntr = (LF.filter(pl.col("timestamp") < TS0).select("uid")
         .group_by("uid").len().collect(engine="streaming"))
act5 = set(ntr.filter(pl.col("len") >= 5)["uid"].to_list())
pool = sorted((set(hist.to_list()) - deg) & act5)
print("  пул после правил очистки: %d пользователей" % len(pool))

# бюджет задания считаем по всей оставляемой истории пользователя (train+val+test):
# именно столько пар (пользователь, трек) лежит в подвыборке
tr_pairs = (LF.select("uid", "item_id").unique().collect(engine="streaming"))
trm = (LF.filter(pl.col("timestamp") < TS0).select("uid", "item_id")
         .unique().collect(engine="streaming"))
print("  пар (пользователь, трек) всего по пулу: %d, из них в истории до test "
      "(матрица обучения): %d" % (
          tr_pairs.filter(pl.col("uid").is_in(pl.Series("uid", pool).implode())).height,
          trm.filter(pl.col("uid").is_in(pl.Series("uid", pool).implode())).height))
rel = (tst_raw.filter(pl.col("timestamp") >= TS0)
              .filter(pl.col("played_ratio_pct") >= 50)
              .filter(pl.col("t0") >= TS0))

import random
for N in [1000, 2000, 3000, 4000, 5000, len(pool)]:
    rnd = random.Random(SEED)
    samp = pool if N >= len(pool) else rnd.sample(pool, N)
    S = pl.Series("uid", samp)
    tp = tr_pairs.filter(pl.col("uid").is_in(S.implode()))
    tm = trm.filter(pl.col("uid").is_in(S.implode()))
    pop = tm.group_by("item_id").len().sort("len", descending=True)
    rs = rel.filter(pl.col("uid").is_in(S.implode()))
    print("\n  --- N = %d пользователей ---" % len(samp))
    nit = tp["item_id"].n_unique()
    print("    весь каталог: пар %d (треков %d, плотность %.5f %%, "
          "пар на пользователя %.0f); матрица обучения %d пар"
          % (tp.height, nit, 100 * tp.height / (len(samp) * nit),
             tp.height / len(samp), tm.height))
    print("    оцениваемых (релевантная новинка в test): %d (%.1f %%)"
          % (rs["uid"].n_unique(), 100 * rs["uid"].n_unique() / len(samp)))
    print("    бюджет 100 тыс. – 2 млн пар: %s"
          % ("ВЛЕЗАЕТ" if 100_000 <= tp.height <= 2_000_000 else
             "НЕ влезает (%s)" % ("мало" if tp.height < 100_000 else "много")))
    for K in [5000, 10000, 15000]:
        if pop.height < K:
            continue
        top = pop.head(K)["item_id"]
        tpk = tp.filter(pl.col("item_id").is_in(top.implode()))
        tmk = tm.filter(pl.col("item_id").is_in(top.implode()))
        rk = rs.filter(pl.col("item_id").is_in(top.implode()))
        cov = tpk.height / tp.height
        print("      каталог топ-%5d треков: пар %8d (%.1f %% взаимодействий), "
              "плотность %.3f %%, матрица обучения %8d пар, оцениваемых %d "
              "(%.1f %%), бюджет: %s"
              % (K, tpk.height, 100 * cov,
                 100 * tmk.height / (len(samp) * K),
                 tmk.height, rk["uid"].n_unique(),
                 100 * rk["uid"].n_unique() / len(samp),
                 "да" if 100_000 <= tpk.height <= 2_000_000 else "нет"))
