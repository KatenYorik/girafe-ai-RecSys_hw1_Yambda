import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 03. Сетка конфигураций подвыборки: сколько пользователей × какой каталог
# укладывается в бюджет задания (100 тыс. – 2 млн пар, >= 1000 польз., >= 1000 треков).
# Принятые границы (блок 02): train = сутки 0…258, val = 259…279, test = 280…300.
DAY = 86400
VAL0 = 259 * DAY
TEST0 = 280 * DAY
BUDGET = 2_000_000
SEED = 42
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))

pairs = (LF.select("uid", "item_id", "timestamp")
           .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
           .collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("uf"))
feat = (LF.select("uid", "played_ratio_pct")
          .group_by("uid").agg((pl.col("played_ratio_pct") == 0).mean().alias("zero"))
          .collect(engine="streaming"))
ntr = (LF.filter(pl.col("timestamp") < TEST0).select("uid")
         .group_by("uid").len().collect(engine="streaming"))
pool = sorted((set(ufirst.filter(pl.col("uf") < TEST0)["uid"].to_list())
               - set(feat.filter(pl.col("zero") > 0.5)["uid"].to_list()))
              & set(ntr.filter(pl.col("len") >= 5)["uid"].to_list()))
P = pl.Series("uid", pool)
print("пул после принятых правил очистки: %d пользователей" % len(pool))

allp = (LF.select("uid", "item_id").unique().collect(engine="streaming")
          .filter(pl.col("uid").is_in(P.implode())))
trp = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id")
         .unique().collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
rel = (LF.filter(pl.col("timestamp") >= TEST0)
         .select("uid", "item_id", "played_ratio_pct")
         .filter(pl.col("played_ratio_pct") >= 50)
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .filter(pl.col("t0") >= TEST0).filter(pl.col("uid").is_in(P.implode())))
print("по всему пулу: пар всего %d, в матрице обучения (сутки 0…279) %d, "
      "релевантных новинок в test %d у %d пользователей"
      % (allp.height, trp.height, rel.height, rel["uid"].n_unique()))

# --- сколько пар на пользователя даёт каталог топ-K: отсюда предельное N ---
pop_all = trp.group_by("item_id").len().sort("len", descending=True)
print("\n=== предельное N при бюджете %d пар ===" % BUDGET)
print("  %-9s %12s %11s %10s %12s %12s" % ("каталог", "пар/польз.", "предел N",
                                           "плотность", "покрытие", "EASE K*K, ГБ"))
rates = {}
for K in [3000, 5000, 8000, 10000, 15000, None]:
    if K is None:
        top, lab = pop_all["item_id"], "весь"
    else:
        if pop_all.height < K:
            continue
        top, lab = pop_all.head(K)["item_id"], "топ-%d" % K
    sub = allp.filter(pl.col("item_id").is_in(top.implode()))
    rate = sub.height / len(pool)
    KK = K if K else pop_all.height
    rates[lab] = (rate, KK)
    # EASE решает систему K x K: сама матрица + рабочая копия при обращении
    gb = KK * KK * 8 * 2 / 1024 ** 3
    print("  %-9s %12.0f %11d %9.2f %% %10.1f %% %12.2f"
          % (lab, rate, int(BUDGET / rate), 100 * rate / KK,
             100 * sub.height / allp.height, gb))
print("  (плотность и покрытие взаимодействий почти не зависят от N, "
      "см. замеры по конфигурациям ниже)")

# --- конкретные конфигурации ---
CONF = [(4000, 5000), (3000, 5000), (3000, 10000), (2500, 10000),
        (2000, 10000), (2600, 15000), (2000, 15000), (1300, None)]
print("\n=== конфигурации (сид %d; каталог — топ-K по популярности внутри подвыборки) ===" % SEED)
for N, K in CONF:
    rnd = random.Random(SEED)
    samp = pool if N >= len(pool) else rnd.sample(pool, N)
    S = pl.Series("uid", samp)
    ap = allp.filter(pl.col("uid").is_in(S.implode()))
    tp = trp.filter(pl.col("uid").is_in(S.implode()))
    pop = tp.group_by("item_id").len().sort("len", descending=True)
    rs = rel.filter(pl.col("uid").is_in(S.implode()))
    if K is None:
        top, lab, KK = pop["item_id"], "весь каталог", pop.height
    else:
        top, lab, KK = pop.head(K)["item_id"], "топ-%d" % K, K
    apk = ap.filter(pl.col("item_id").is_in(top.implode()))
    tpk = tp.filter(pl.col("item_id").is_in(top.implode()))
    rk = rs.filter(pl.col("item_id").is_in(top.implode()))
    ev_full, ev_k = rs["uid"].n_unique(), rk["uid"].n_unique()
    ok = 100_000 <= apk.height <= BUDGET and len(samp) >= 1000 and KK >= 1000
    print("\n  N = %5d, %s" % (len(samp), lab))
    print("    пар всего %9d  | матрица обучения %9d | треков %6d | плотность %5.2f %%"
          % (apk.height, tpk.height, KK, 100 * tpk.height / (len(samp) * KK)))
    print("    оцениваемых %5d (%.1f %% подвыборки); каталог отрезает %d из %d (%.1f %%)"
          % (ev_k, 100 * ev_k / len(samp), ev_full - ev_k, ev_full,
             100 * (ev_full - ev_k) / ev_full))
    print("    покрытие взаимодействий каталогом %.1f %%; EASE K*K %.2f ГБ; бюджет: %s"
          % (100 * apk.height / ap.height, KK * KK * 8 * 2 / 1024 ** 3,
             "ВЛЕЗАЕТ" if ok else "НЕ влезает"))
