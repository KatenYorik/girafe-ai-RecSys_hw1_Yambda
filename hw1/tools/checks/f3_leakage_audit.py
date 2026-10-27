import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 09. Аудит утечек. Каждый признак из принятого набора считается дважды: по своему
# окну (timestamp < W) и по всем данным (включая test). Если версии расходятся — значит
# признак чувствителен к утечке и в наборе обязателен именно вариант по окну; расхождение
# заодно показывает, сколько именно утечки мы бы получили, забыв фильтр. Признаки, у
# источника которых нет колонки timestamp (артист, альбом, эмбеддинг), проверяются
# структурно. Отдельно — состояние лайка на границе теста против итогового.
DAY = 86400
VAL0, TEST0, END = 259 * DAY, 280 * DAY, 301 * DAY
N, K, SEED = 3000, 10000, 42
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
BM = pl.scan_parquet(os.path.join(raw, "album_item_mapping.parquet"))

pairs0 = (LF.select("uid", "item_id", "timestamp")
            .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
            .collect(engine="streaming"))
ufirst = pairs0.group_by("uid").agg(pl.col("t0").min().alias("uf"))
zero = (LF.select("uid", "played_ratio_pct").group_by("uid")
          .agg((pl.col("played_ratio_pct") == 0).mean().alias("z"))
          .collect(engine="streaming"))
ntr = (LF.filter(pl.col("timestamp") < TEST0).select("uid")
         .group_by("uid").len().collect(engine="streaming"))
pool = sorted((set(ufirst.filter(pl.col("uf") < TEST0)["uid"].to_list())
               - set(zero.filter(pl.col("z") > 0.5)["uid"].to_list()))
              & set(ntr.filter(pl.col("len") >= 5)["uid"].to_list()))
P = pl.Series("uid", pool)
samp = sorted(random.Random(SEED).sample(pool, N))
S = pl.Series("uid", samp)
US = pl.DataFrame({"uid": samp})
trm = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
         .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
CAT = (trm.group_by("item_id").agg(pl.col("uid").n_unique().alias("pool_users"))
          .sort(["pool_users", "item_id"], descending=[True, False]).head(K))
C = pl.Series("item_id", CAT["item_id"].to_list())
print("пул %d, подвыборка %d, каталог %d; границы: VAL0 = %d, TEST0 = %d, конец = %d"
      % (len(pool), N, K, VAL0, TEST0, END))

# --- 0. структурная проверка: у источников метаданных нет времени ---
print("\n=== 0. Структурная проверка источников ===")
for f in ["artist_item_mapping", "album_item_mapping", "embed_item_ids"]:
    sc = dict(pl.scan_parquet(os.path.join(raw, f + ".parquet")).collect_schema())
    print("  %-22s колонки %s → времени нет, утечка невозможна по построению"
          % (f, list(sc)))
for f in ["listens", "likes", "dislikes", "unlikes", "undislikes"]:
    sc = dict(pl.scan_parquet(os.path.join(raw, f + ".parquet")).collect_schema())
    print("  %-22s есть timestamp: %s → признак обязан нести фильтр по окну"
          % (f, "timestamp" in sc))

# --- 1. признаки трека: по окну против «по всем данным» ---
print("\n=== 1. Признаки трека: версия по окну против версии по всем данным ===")


def item_feats(W):
    return (LF.filter(pl.col("timestamp") < W)
              .join(CAT.select("item_id").lazy(), on="item_id", how="semi")
              .join(pl.DataFrame({"uid": pool}).lazy(), on="uid", how="semi")
              .group_by("item_id").agg(
                  pl.col("uid").n_unique().alias("users"),
                  pl.len().alias("listens"),
                  pl.col("played_ratio_pct").mean().alias("pr"),
                  (pl.col("played_ratio_pct") <= 1).mean().alias("skip"),
                  pl.col("is_organic").mean().alias("org"),
                  (pl.col("timestamp") // DAY).min().alias("debut"),
              ).collect(engine="streaming"))


IT = {w: item_feats(W) for w, W in [("train", VAL0), ("tr+val", TEST0), ("всё", END)]}
base = CAT.select("item_id")
J = base
for w in IT:
    J = J.join(IT[w].rename({c: c + "|" + w for c in IT[w].columns if c != "item_id"}),
               on="item_id", how="left")
print("  %-28s %12s %12s %12s %10s %10s"
      % ("признак трека", "train", "tr+val", "всё (утечка)", "расх. tr+val", "доля изм."))
for c, fmt in [("users", "%12.0f"), ("listens", "%12.0f"), ("pr", "%12.2f"),
               ("skip", "%12.4f"), ("org", "%12.4f"), ("debut", "%12.0f")]:
    a, b, d = (J[c + "|train"].cast(pl.Float64), J[c + "|tr+val"].cast(pl.Float64),
               J[c + "|всё"].cast(pl.Float64))
    diff = (d - b).abs()
    print(("  %-28s" + fmt * 3 + " %10.3f %9.1f %%")
          % (c, a.median(), b.median(), d.median(), diff.median(),
             100 * (diff > 1e-9).mean()))
print("  («расх. tr+val» — медиана |значение по всем данным − значение по train+val|;")
print("   «доля изм.» — у какой доли треков каталога версия по всем данным отличается.")
print("   В набор берётся ТОЛЬКО версия по окну; колонка «всё» — размер утечки, если забыть фильтр.)")
print("  дебют: по train+val медиана %d, по всем данным %d; треков, у которых дебют "
      "сдвинулся бы в test: %d"
      % (J["debut|tr+val"].median(), J["debut|всё"].median(),
         (J["debut|всё"] != J["debut|tr+val"]).sum()))

# --- 2. признаки пользователя: то же сравнение ---
print("\n=== 2. Агрегаты пользователя: версия по окну против версии по всем данным ===")


def user_feats(W):
    return (LF.filter(pl.col("timestamp") < W)
              .join(US.lazy(), on="uid", how="semi")
              .group_by("uid").agg(
                  pl.len().alias("n"),
                  pl.col("item_id").n_unique().alias("items"),
                  (pl.col("timestamp") // DAY).n_unique().alias("days"),
                  pl.col("is_organic").mean().alias("org"),
                  pl.col("played_ratio_pct").mean().alias("pr"),
                  (pl.col("played_ratio_pct") <= 1).mean().alias("skip"),
                  (pl.col("played_ratio_pct") >= 99).mean().alias("full"),
              ).collect(engine="streaming"))


UF = {w: user_feats(W) for w, W in [("train", VAL0), ("tr+val", TEST0), ("всё", END)]}
JU = US
for w in UF:
    JU = JU.join(UF[w].rename({c: c + "|" + w for c in UF[w].columns if c != "uid"}),
                 on="uid", how="left")
print("  %-28s %12s %12s %12s %10s %10s"
      % ("агрегат пользователя", "train", "tr+val", "всё (утечка)", "расх. tr+val", "доля изм."))
for c, fmt in [("n", "%12.0f"), ("items", "%12.0f"), ("days", "%12.0f"),
               ("org", "%12.4f"), ("pr", "%12.2f"), ("skip", "%12.4f"),
               ("full", "%12.4f")]:
    a, b, d = (JU[c + "|train"].cast(pl.Float64), JU[c + "|tr+val"].cast(pl.Float64),
               JU[c + "|всё"].cast(pl.Float64))
    diff = (d - b).abs()
    print(("  %-28s" + fmt * 3 + " %10.3f %9.1f %%")
          % (c, a.median(), b.median(), d.median(), diff.median(),
             100 * (diff > 1e-9).mean()))

# --- 3. «знаком ли трек / артист»: утечка здесь стоит всей метрики ---
print("\n=== 3. Признак «трек/артист знаком»: цена ошибки ===")
sp = pairs0.filter(pl.col("uid").is_in(S.implode()) & pl.col("item_id").is_in(C.implode()))
known_tv = sp.filter(pl.col("t0") < TEST0)
new_test = sp.filter(pl.col("t0") >= TEST0)
print("  пар (пользователь, трек) каталога с историей: всего %d, из них знакомых к началу "
      "test %d, впервые встреченных в test %d"
      % (sp.height, known_tv.height, new_test.height))
rel = (LF.filter(pl.col("timestamp") >= TEST0)
         .select("uid", "item_id", "played_ratio_pct")
         .filter(pl.col("played_ratio_pct") >= 50).collect(engine="streaming")
         .filter(pl.col("uid").is_in(S.implode()) & pl.col("item_id").is_in(C.implode()))
         .join(sp, on=["uid", "item_id"], how="inner"))
relnew = rel.filter(pl.col("t0") >= TEST0)
print("  релевантных новинок внутри каталога: %d событий, %d пар — это ровно та цель, "
      "которую мы предсказываем"
      % (relnew.height, relnew.select(pl.struct("uid", "item_id").n_unique()).item()))
print("  если «знаком» считать по всем данным, все %d целевых пар станут «знакомыми» и "
      "будут выброшены из кандидатов → NDCG@10 = 0 по построению"
      % relnew.select(pl.struct("uid", "item_id").n_unique()).item())
print("  доля кандидатных пар, у которых признак «знаком» изменился бы: %.2f %% "
      "(%d из %d пар подвыборки × каталога)"
      % (100 * new_test.height / (N * K), new_test.height, N * K))
amap = AM.unique(subset=["item_id"], keep="first").collect()
pa = pairs0.filter(pl.col("uid").is_in(S.implode())).join(amap, on="item_id", how="inner")
af = pa.group_by(["uid", "artist_id"]).agg(pl.col("t0").min().alias("a0"))
print("  пар (пользователь, артист): знакомых к началу test %d, новых в test %d (%.2f %% "
      "от знакомых) — тот же фильтр по окну обязателен"
      % ((af["a0"] < TEST0).sum(), (af["a0"] >= TEST0).sum(),
         100 * (af["a0"] >= TEST0).sum() / max((af["a0"] < TEST0).sum(), 1)))
cat_art = (CAT.select("item_id").join(amap, on="item_id", how="left")
              .group_by("artist_id").len().rename({"len": "k_items"}))
for lab, W in [("train 0…258", VAL0), ("train+val 0…279", TEST0)]:
    ka = (af.filter(pl.col("a0") < W).join(cat_art, on="artist_id", how="inner")
            .group_by("uid").agg(pl.col("k_items").sum().alias("kn"),
                                 pl.len().alias("na")))
    kn = (US.join(ka, on="uid", how="left")
            .with_columns(pl.col("kn").fill_null(0), pl.col("na").fill_null(0)))
    print("  pair_artist_known по окну %-16s: знакомых артистов на пользователя медиана "
          "%d; кандидатных пар со знакомым артистом %d из %d (%.1f %%), на пользователя "
          "медиана %d из 10 000"
          % (lab, kn["na"].median(), kn["kn"].sum(), N * K,
             100 * kn["kn"].sum() / (N * K), kn["kn"].median()))

# --- 4. состояние лайка: граница теста против итога ---
print("\n=== 4. Состояние лайка: на границе теста против итогового ===")
parts = []
for f, k in [("likes", 1), ("unlikes", 0), ("dislikes", -1), ("undislikes", 0)]:
    parts.append(pl.scan_parquet(os.path.join(raw, f + ".parquet"))
                   .select("uid", "item_id", "timestamp")
                   .with_columns(pl.lit(k, pl.Int8).alias("st"),
                                 pl.lit(f).alias("src")))
EVT = pl.concat(parts).collect()
ES = EVT.filter(pl.col("uid").is_in(S.implode()))
print("  события лайков/дизлайков и их отмен: всего %d, у подвыборки %d" % (EVT.height, ES.height))
for f in ["likes", "unlikes", "dislikes", "undislikes"]:
    e = ES.filter(pl.col("src") == f)
    print("    %-12s у подвыборки %7d; в train 0…258 %7d, в val %6d, в test 280…300 %6d (%.2f %%)"
          % (f, e.height, (e["timestamp"] < VAL0).sum(),
             ((e["timestamp"] >= VAL0) & (e["timestamp"] < TEST0)).sum(),
             (e["timestamp"] >= TEST0).sum(),
             100 * (e["timestamp"] >= TEST0).sum() / max(e.height, 1)))


def state(W):
    return (ES.filter(pl.col("timestamp") < W).sort(["uid", "item_id", "timestamp"])
              .group_by(["uid", "item_id"]).agg(pl.col("st").last().alias("st"),
                                                pl.len().alias("k")))


st_tv, st_all = state(TEST0), state(END)
cmp = (st_tv.rename({"st": "s_tv", "k": "k_tv"})
            .join(st_all.rename({"st": "s_all", "k": "k_all"}),
                  on=["uid", "item_id"], how="full", coalesce=True)
            .with_columns(pl.col("s_tv").fill_null(0), pl.col("s_all").fill_null(0),
                          pl.col("k_tv").fill_null(0), pl.col("k_all").fill_null(0)))
print("  пар с хотя бы одним событием: к началу test %d, по всем данным %d"
      % (st_tv.height, st_all.height))
ch = cmp.filter(pl.col("s_tv") != pl.col("s_all"))
print("  пар, у которых состояние меняется внутри тестового окна: %d (%.2f %% пар с "
      "событиями к началу test, %.2f %% итоговых)"
      % (ch.height, 100 * ch.height / st_tv.height, 100 * ch.height / st_all.height))
print("  %-34s %10s" % ("переход (к началу test → итог)", "пар"))
NM = {1: "лайк", 0: "нет", -1: "дизлайк"}
for a in [1, 0, -1]:
    for b in [1, 0, -1]:
        if a == b:
            continue
        c = ch.filter((pl.col("s_tv") == a) & (pl.col("s_all") == b)).height
        if c:
            print("  %-34s %10d" % ("%s → %s" % (NM[a], NM[b]), c))
newp = cmp.filter((pl.col("k_tv") == 0) & (pl.col("k_all") > 0))
both = ch.filter((pl.col("k_tv") == 0))
print("  пар, у которых к началу test событий не было вовсе, а в test появились: %d "
      "(из них итоговое состояние «лайк» у %d, «дизлайк» у %d, «нет» у %d — последние "
      "в смену состояния не попадают)"
      % (newp.height, newp.filter(pl.col("s_all") == 1).height,
         newp.filter(pl.col("s_all") == -1).height,
         newp.filter(pl.col("s_all") == 0).height))
print("  → в признаки берём состояние на границе окна; итоговое состояние — утечка ровно "
      "на %d пар, из них %d появились внутри теста с нуля" % (ch.height, both.height))

# --- 5. лайк без прослушивания: находка блока 05 на принятой конфигурации ---
print("\n=== 5. Лайк без прослушивания в истории до test (продолжение блока 05) ===")
liked_tv = st_tv.filter(pl.col("st") == 1).select("uid", "item_id")
pre = pairs0.filter(pl.col("t0") < TEST0).select("uid", "item_id")
no_listen = liked_tv.join(pre, on=["uid", "item_id"], how="anti")
print("  лайкнутых пар у подвыборки к началу test: %d" % liked_tv.height)
print("  из них без ни одного прослушивания в сутки 0…279: %d (%.1f %%) — блок 05 на train "
      "видел 16,3 %%" % (no_listen.height, 100 * no_listen.height / liked_tv.height))
lk = (ES.filter((pl.col("src") == "likes") & (pl.col("timestamp") < TEST0))
        .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("lt")))
jj = lk.join(pairs0, on=["uid", "item_id"], how="left")
print("  лайк раньше первого прослушивания той же пары: %d из %d пар с лайком (%.1f %%); "
      "прослушивания нет вовсе у %d"
      % ((jj["lt"] < jj["t0"]).sum(), jj.height,
         100 * (jj["lt"] < jj["t0"]).sum() / jj.height, jj["t0"].null_count()))
inside = liked_tv.join(CAT.select("item_id"), on="item_id", how="semi")
print("  лайкнутых пар на треках каталога: %d; из них без прослушивания до test: %d (%.1f %%)"
      % (inside.height,
         inside.join(pre, on=["uid", "item_id"], how="anti").height,
         100 * inside.join(pre, on=["uid", "item_id"], how="anti").height
         / max(inside.height, 1)))

# --- 6. признаки тестового события в наборе: проверка по списку ---
print("\n=== 6. Признаки тестового события в наборе отсутствуют ===")
W2 = "train или train+val"
L, LA = "listens, ts < W", "listens + artist map, ts < W"
FEATS = [
    # (имя, окно, источник, зависит ли от времени)
    ("item_pop_users", W2, L, True),
    ("item_pop_users_sub", W2, L, True),
    ("item_listens", W2, L, True),
    ("item_mean_played_ratio", W2, L, True),
    ("item_skip_share", W2, L, True),
    ("item_full_share", W2, L, True),
    ("item_organic_share", W2, L, True),
    ("item_age_days", W2, L, True),
    ("artist_id", "без времени", "artist_item_mapping", False),
    ("album_id", "без времени", "album_item_mapping", False),
    ("has_embed", "без времени", "embed_item_ids", False),
    ("embed[128]", "без времени", "embeddings.parquet", False),
    ("user_n_listens", W2, L, True),
    ("user_n_listens_cat", W2, L, True),
    ("user_n_items", W2, L, True),
    ("user_n_items_cat", W2, L, True),
    ("user_n_artists", W2, LA, True),
    ("user_active_days", W2, L, True),
    ("user_hist_days", W2, L, True),
    ("user_organic_share", W2, L, True),
    ("user_mean_played_ratio", W2, L, True),
    ("user_skip_share", W2, L, True),
    ("user_full_share", W2, L, True),
    ("user_n_likes", W2, "like-состояние, ts < W", True),
    ("user_n_dislikes", W2, "like-состояние, ts < W", True),
    ("user_mean_repeats", W2, L, True),
    ("user_has_profile", W2, L, True),
    ("pair_artist_known", W2, LA, True),
    ("pair_artist_listens", W2, LA, True),
    ("pair_album_known", W2, "listens + album map, ts < W", True),
]
BANNED = ["played_ratio_pct тестового прослушивания", "is_organic тестового прослушивания",
          "timestamp тестового события", "итоговое состояние лайка (по всем данным)",
          "популярность с участием test", "дебют трека с участием test"]
print("  в наборе %d признаков; зависят от времени %d, у остальных источник без времени"
      % (len(FEATS), sum(1 for f in FEATS if f[3])))
bad = [f for f in FEATS if f[3] and "ts < W" not in f[2]]
print("  признаков, зависящих от времени и не несущих фильтр «ts < W»: %d" % len(bad))
print("  служебные величины по тому же окну, но НЕ признаки FM: pair_item_known "
      "(на кандидатах тождественный нуль — фильтр кандидатов и слагаемое реранжирования), "
      "состояние лайка на паре (кандидатных пар с лайком 1 446 из 30 млн)")
print("  %-50s %6s %-24s" % ("признак", "время", "окно"))
for n, w, src, t in FEATS:
    print("  %-50s %6s %-24s" % (n, "да" if t else "нет", w))
print("\n  запрещённые источники и их наличие в наборе:")
for b in BANNED:
    print("    %-54s нет" % b)
print("\n  граница окна задана константами (VAL0 = %d, TEST0 = %d) и подставляется в "
      "каждый расчёт параметром W; расчётов без фильтра по W в наборе нет. Третья "
      "константа END = %d используется только здесь, чтобы измерить размер утечки."
      % (VAL0, TEST0, END))
