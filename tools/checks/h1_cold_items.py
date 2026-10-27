import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 07. Холодные треки — в исходной формулировке и в переформулировках под
# принятый каталог (блоки 03/04): каталог = топ-10 000 по числу слушателей в пуле,
# только по истории до test (сутки 0…279); подвыборка 3 000 пользователей, сид 42.
DAY = 86400
TEST0 = 280 * DAY
N, K, SEED = 3000, 10000, 42
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
EMB = pl.scan_parquet(os.path.join(raw, "embed_item_ids.parquet"))

pairs = (LF.select("uid", "item_id", "timestamp")
           .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
           .collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("uf"))
zero = (LF.select("uid", "played_ratio_pct").group_by("uid")
          .agg((pl.col("played_ratio_pct") == 0).mean().alias("z"))
          .collect(engine="streaming"))
ntr = (LF.filter(pl.col("timestamp") < TEST0).select("uid")
         .group_by("uid").len().collect(engine="streaming"))
pool = sorted((set(ufirst.filter(pl.col("uf") < TEST0)["uid"].to_list())
               - set(zero.filter(pl.col("z") > 0.5)["uid"].to_list()))
              & set(ntr.filter(pl.col("len") >= 5)["uid"].to_list()))
P = pl.Series("uid", pool)
rnd = random.Random(SEED)
samp = rnd.sample(pool, N)
S = pl.Series("uid", samp)
emb = EMB.select("item_id").unique().collect()

# популярность и каталог — по истории до test, на пуле
trm_pool = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
              .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
pop = (trm_pool.group_by("item_id").agg(pl.col("uid").n_unique().alias("users"))
               .sort(["users", "item_id"], descending=[True, False]))
CAT = pop.head(K)
C = pl.Series("item_id", CAT["item_id"].to_list())
print("пул %d, подвыборка %d, каталог %d (порог >= %d слушателей)"
      % (len(pool), N, K, CAT["users"].min()))

# релевантные новинки test у подвыборки
rel = (LF.filter(pl.col("timestamp") >= TEST0)
         .select("uid", "item_id", "played_ratio_pct")
         .filter(pl.col("played_ratio_pct") >= 50)
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .filter(pl.col("t0") >= TEST0)
         .filter(pl.col("uid").is_in(S.implode())))
REL = rel.height
print("релевантных новинок test в подвыборке: %d событий у %d пользователей"
      % (REL, rel["uid"].n_unique()))

known = set(pop["item_id"].to_list())          # треки с историей до test (у пула)
print("\n=== 1. исходная формулировка: трек не встречался в train ни у кого ===")
cold = rel.filter(~pl.col("item_id").is_in(pl.Series("item_id", sorted(known)).implode()))
ci = cold["item_id"].unique()
print("  релевантных новинок на таких треках: %d событий (%.1f %% всех новинок) "
      "у %d пользователей, различных треков %d"
      % (cold.height, 100 * cold.height / REL, cold["uid"].n_unique(), len(ci)))
ne = pl.DataFrame({"item_id": ci}).join(emb, on="item_id", how="semi").height
print("  эмбеддинг есть у %d из %d таких треков (%.1f %%)"
      % (ne, len(ci), 100 * ne / max(len(ci), 1)))
in_cat = len(set(ci.to_list()) & set(C.to_list()))
print("  из них попадают в каталог: %d — срез по построению ПУСТ, потому что каталог"
      % in_cat)
print("  строится по популярности в истории до test, а у этих треков её нет")

print("\n=== 2. переформулировка A: «молодые» треки каталога (поздний дебют) ===")
# день первого появления трека у кого-либо (по всем данным)
dbut = (pairs.group_by("item_id").agg(pl.col("t0").min().alias("first"))
             .with_columns((pl.col("first") // DAY).alias("d0")))
cat_d = CAT.join(dbut, on="item_id", how="left")
print("  день первого появления трека каталога: " + " ".join(
    "q%d=%d" % (int(100 * q), cat_d["d0"].quantile(q)) for q in [.05, .25, .5, .75, .95])
      + " max=%d" % cat_d["d0"].max())
relc = rel.filter(pl.col("item_id").is_in(C.implode())).join(dbut, on="item_id", how="left")
RELC = relc.height
print("  релевантных новинок внутри каталога: %d событий" % RELC)
for d in [150, 200, 230, 250, 270]:
    sub = cat_d.filter(pl.col("d0") >= d)
    rs = relc.filter(pl.col("d0") >= d)
    e = sub.select("item_id").join(emb, on="item_id", how="semi").height
    print("    дебют после суток %3d: треков каталога %5d (%.1f %%), "
          "релевантных новинок %6d (%.1f %% внутрикаталожных) у %4d польз., эмбеддинг %.1f %%"
          % (d, sub.height, 100 * sub.height / K, rs.height,
             100 * rs.height / RELC, rs["uid"].n_unique(),
             100 * e / max(sub.height, 1)))

print("\n=== 3. переформулировка B: нижняя полоса каталога по популярности ===")
# «холодный» относительно матрицы, на которой реально учимся: число пользователей
# ПОДВЫБОРКИ, знающих трек каталога до test
tm = trm_pool.filter(pl.col("uid").is_in(S.implode()))
sub_pop = (CAT.select("item_id").join(
    tm.group_by("item_id").agg(pl.col("uid").n_unique().alias("su")),
    on="item_id", how="left").with_columns(pl.col("su").fill_null(0)))
print("  пользователей подвыборки на трек каталога: min=%d " % sub_pop["su"].min()
      + " ".join("q%d=%d" % (int(100 * q), sub_pop["su"].quantile(q))
                 for q in [.05, .25, .5, .75, .95]) + " max=%d" % sub_pop["su"].max())
print("  треков каталога, известных в подвыборке менее чем: "
      + ", ".join("%d польз. — %d" % (m, (sub_pop["su"] < m).sum())
                  for m in [10, 20, 30, 50]))
relb = (rel.filter(pl.col("item_id").is_in(C.implode()))
           .join(sub_pop, on="item_id", how="left")
           .join(CAT.rename({"users": "pool_users"}), on="item_id", how="left"))
cat_srt = CAT.with_row_index("rank", offset=1)
relb = relb.join(cat_srt.select("item_id", "rank"), on="item_id", how="left")
print("  полосы каталога по рангу популярности:")
print("    %-14s %7s %9s %10s %9s %11s" % ("полоса", "треков", "медиана",
                                           "новинок", "польз.", "эмбеддинг"))
for a, b in [(1, 2000), (2001, 5000), (5001, 8000), (8001, 10000)]:
    bandi = cat_srt.filter((pl.col("rank") >= a) & (pl.col("rank") <= b))
    sp = sub_pop.join(bandi.select("item_id"), on="item_id", how="semi")
    rb = relb.filter((pl.col("rank") >= a) & (pl.col("rank") <= b))
    e = bandi.select("item_id").join(emb, on="item_id", how="semi").height
    print("    %-14s %7d %9.0f %10d %9d %10.1f %%"
          % ("%d–%d" % (a, b), bandi.height, sp["su"].median(), rb.height,
             rb["uid"].n_unique(), 100 * e / bandi.height))
print("  «медиана» — сколько пользователей подвыборки знают трек полосы до test")
print("  (это и есть объём обучающего сигнала на трек: у EASE и iALS похожесть")
print("   оценивается именно по этим наблюдениям)")
