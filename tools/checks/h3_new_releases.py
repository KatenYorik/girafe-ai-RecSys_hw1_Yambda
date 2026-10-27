import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 07. Срез «новые релизы»: релевантные новинки на треках каталога, появившихся
# в данных незадолго до теста. Замена отменённому срезу холодных треков.
# Конфигурация — принятая: train 0…258 / val 259…279 / test 280…300 (блок 02),
# пул и фильтр из блока 03, каталог топ-10 000 по пулу из блока 04, сид 42.
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

trm_pool = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
              .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
pop = (trm_pool.group_by("item_id").agg(pl.col("uid").n_unique().alias("users"))
               .sort(["users", "item_id"], descending=[True, False]))
CAT = pop.head(K).with_row_index("rank", offset=1)
C = pl.Series("item_id", CAT["item_id"].to_list())

# дебют трека — первое прослушивание кем-либо из пула ДО теста (без участия test)
dbut = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id", "timestamp")
          .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode()))
          .group_by("item_id").agg(pl.col("timestamp").min().alias("f"))
          .with_columns((pl.col("f") // DAY).alias("d0")))
cat = CAT.join(dbut.select("item_id", "d0"), on="item_id", how="left")
sub_pop = trm_pool.filter(pl.col("uid").is_in(S.implode())).group_by("item_id").agg(
    pl.col("uid").n_unique().alias("su"))
print("пул %d, подвыборка %d, каталог %d (порог >= %d слушателей)"
      % (len(pool), N, K, pop["users"][K - 1]))

rel = (LF.filter(pl.col("timestamp") >= TEST0)
         .select("uid", "item_id", "played_ratio_pct", "is_organic")
         .filter(pl.col("played_ratio_pct") >= 50)
         .collect(engine="streaming")
         .join(pairs, on=["uid", "item_id"], how="left")
         .filter(pl.col("t0") >= TEST0).filter(pl.col("uid").is_in(S.implode()))
         .filter(pl.col("item_id").is_in(C.implode()))
         .join(cat.select("item_id", "rank", "d0", "users"), on="item_id", how="left"))
RELC = rel.height
print("релевантных новинок внутри каталога: %d событий у %d пользователей (база среза)"
      % (RELC, rel["uid"].n_unique()))

print("\n=== срез «новые релизы» при разных порогах новизны ===")
print("  %-34s %7s %9s %8s %7s %7s %7s %8s"
      % ("определение", "треков", "событий", ">=1", ">=3", ">=5", ">=10", "медиана"))
for d in [150, 200, 230, 250, 260, 270]:
    ct = cat.filter(pl.col("d0") >= d)
    rs = rel.filter(pl.col("d0") >= d)
    k = rs.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))
    full = (pl.DataFrame({"uid": samp}).join(k, on="uid", how="left")
              .with_columns(pl.col("k").fill_null(0))["k"])
    nz = full.filter(full >= 1)
    print("  %-34s %7d %9d %8d %7d %7d %7d %8d"
          % ("дебют >= суток %d (не старше %d сут.)" % (d, 280 - d),
             ct.height, rs.height, (full >= 1).sum(), (full >= 3).sum(),
             (full >= 5).sum(), (full >= 10).sum(),
             nz.median() if len(nz) else 0))
print("  «>=K» — пользователей, у которых K и больше релевантных новинок внутри среза;")
print("  «медиана» — по тем, у кого есть хотя бы одна")

D = int(sys.argv[2]) if len(sys.argv) > 2 else 150
# порог новизны (сутки дебюта); 150 — принятое определение среза: к началу
# теста треку не больше 130 суток. Выбран потому, что при более узких порогах
# пересечение со срезом по новым артистам становится вырожденным.
print("\n=== подробно по выбранному порогу: дебют >= суток %d (не старше %d суток) ==="
      % (D, 280 - D))
ct = cat.filter(pl.col("d0") >= D)
rs = rel.filter(pl.col("d0") >= D)
k = rs.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))
full = (pl.DataFrame({"uid": samp}).join(k, on="uid", how="left")
          .with_columns(pl.col("k").fill_null(0))["k"])
print("  треков в срезе: %d (%.2f %% каталога); пар (пользователь, трек) %d"
      % (ct.height, 100 * ct.height / K,
         rs.select(pl.struct("uid", "item_id").n_unique()).item()))
print("  событий %d (%.1f %% внутрикаталожных новинок), пользователей %d (%.1f %% подвыборки)"
      % (rs.height, 100 * rs.height / RELC, (full >= 1).sum(),
         100 * (full >= 1).sum() / N))
nz = full.filter(full >= 1)
print("  новинок среза на пользователя: среднее %.1f, медиана %d, q75 %d, q90 %d, max %d"
      % (nz.mean(), nz.median(), nz.quantile(.75), nz.quantile(.9), nz.max()))
print("  популярность треков среза в каталоге: ранг медиана %d, q25 %d, q75 %d; "
      "слушателей пула медиана %d"
      % (ct["rank"].median(), ct["rank"].quantile(.25), ct["rank"].quantile(.75),
         ct["users"].median()))
sp = ct.select("item_id").join(sub_pop, on="item_id", how="left").with_columns(
    pl.col("su").fill_null(0))
print("  обучающий сигнал: пользователей подвыборки на трек среза медиана %d "
      "(по каталогу в целом 128)" % sp["su"].median())
e = ct.select("item_id").join(emb, on="item_id", how="semi").height
print("  эмбеддинг есть у %d из %d треков среза (%.1f %%)"
      % (e, ct.height, 100 * e / ct.height))
print("  органика: %.1f %% событий среза против %.1f %% по всем внутрикаталожным новинкам"
      % (100 * rs["is_organic"].mean(), 100 * rel["is_organic"].mean()))
bands = [(D + i, min(D + i + 19, 279)) for i in range(0, 280 - D, 20)]
print("  дебют треков среза по суткам: " + ", ".join(
    "%d–%d: %d" % (a, b, ct.filter((pl.col("d0") >= a) & (pl.col("d0") <= b)).height)
    for a, b in bands))

# --- пересечение со срезом «новый артист»: новинка нового релиза у нового артиста ---
print("\n=== срез по артистам внутри среза по новым релизам ===")
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
amap = AM.unique(subset=["item_id"], keep="first").collect()
afirst = (LF.select("uid", "item_id", "timestamp")
            .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
            .join(AM, on="item_id", how="inner")
            .group_by(["uid", "artist_id"]).agg(pl.col("t0").min().alias("a0"))
            .collect(engine="streaming"))
r2 = (rel.join(amap, on="item_id", how="left")
         .join(afirst, on=["uid", "artist_id"], how="left")
         .with_columns((pl.col("d0") >= D).alias("new_rel"),
                       (pl.col("artist_id").is_not_null()
                        & (pl.col("a0") >= TEST0)).alias("new_art")))
print("  таблица 2x2 по релевантным новинкам внутри каталога (всего %d событий):" % r2.height)
print("    %-34s %9s %8s %9s %9s" % ("клетка", "событий", "доля", "польз.", "артистов"))
for nr in [True, False]:
    for na in [True, False]:
        c = r2.filter((pl.col("new_rel") == nr) & (pl.col("new_art") == na))
        print("    %-34s %9d %7.1f %% %9d %9d"
              % ("%s релиз, %s артист"
                 % ("новый" if nr else "старый", "новый" if na else "знакомый"),
                 c.height, 100 * c.height / r2.height, c["uid"].n_unique(),
                 c["artist_id"].n_unique()))
cross = r2.filter(pl.col("new_rel") & pl.col("new_art"))
k = cross.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"),
                              pl.col("artist_id").n_unique().alias("ka"))
full = (pl.DataFrame({"uid": samp}).join(k, on="uid", how="left")
          .with_columns(pl.col("k").fill_null(0), pl.col("ka").fill_null(0)))
print("\n  пересечение «новый релиз + новый артист» подробно:")
print("    событий %d (%.1f %% внутрикаталожных новинок), треков %d, артистов %d, пар %d"
      % (cross.height, 100 * cross.height / r2.height, cross["item_id"].n_unique(),
         cross["artist_id"].n_unique(),
         cross.select(pl.struct("uid", "artist_id").n_unique()).item()))
for c, lab in [("k", "новых треков"), ("ka", "новых артистов")]:
    s = full[c]
    nz = s.filter(s >= 1)
    print("    %-16s на пользователя: с >=1 — %d, >=3 — %d, >=5 — %d; "
          "медиана %d, q90 %d, max %d"
          % (lab, (s >= 1).sum(), (s >= 3).sum(), (s >= 5).sum(),
             nz.median() if len(nz) else 0, nz.quantile(.9) if len(nz) else 0, s.max()))
print("    органика %.1f %% (в срезе новых релизов %.1f %%, по всем новинкам %.1f %%)"
      % (100 * cross["is_organic"].mean(),
         100 * r2.filter(pl.col("new_rel"))["is_organic"].mean(),
         100 * r2["is_organic"].mean()))
nrel = r2.filter(pl.col("new_rel"))
print("    внутри среза новых релизов новые артисты дают %.1f %% событий; "
      "по всем внутрикаталожным новинкам — %.1f %%"
      % (100 * cross.height / nrel.height,
         100 * r2.filter(pl.col("new_art")).height / r2.height))
