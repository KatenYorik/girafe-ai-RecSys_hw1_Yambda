import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 09. Признаки трека на принятом каталоге: альбомы (сколько треков с несколькими
# и правило выбора одного), эмбеддинги (размерность и тип), популярность и возраст как
# признаки. Всё считается по двум окнам: train = сутки 0…258 (для подбора на val) и
# train+val = 0…279 (для финальной модели). Тестовое окно не участвует нигде.
DAY = 86400
VAL0, TEST0 = 259 * DAY, 280 * DAY
N, K, SEED = 3000, 10000, 42
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))
BM = pl.scan_parquet(os.path.join(raw, "album_item_mapping.parquet"))
EMB = pl.scan_parquet(os.path.join(raw, "embed_item_ids.parquet"))

# --- пул, подвыборка, каталог: конфигурация блоков 03/04 ---
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
samp = random.Random(SEED).sample(pool, N)
S = pl.Series("uid", samp)
trm = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
         .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
pop = (trm.group_by("item_id").agg(pl.col("uid").n_unique().alias("pool_users"))
          .sort(["pool_users", "item_id"], descending=[True, False]))
CAT = pop.head(K).with_row_index("rank", offset=1)
C = pl.Series("item_id", CAT["item_id"].to_list())
print("пул %d, подвыборка %d, каталог %d (порог >= %d слушателей пула)"
      % (len(pool), N, K, CAT["pool_users"][K - 1]))

# --- 1. альбомы: сколько треков каталога имеют несколько альбомов ---
print("\n=== 1. Альбом трека: сколько вариантов и как выбрать один ===")
alb = BM.collect().filter(pl.col("item_id").is_in(C.implode()))
alb_size = BM.group_by("album_id").len().collect().rename({"len": "alb_tracks"})
cnt = alb.group_by("item_id").len()
full = CAT.select("item_id").join(cnt, on="item_id", how="left").with_columns(
    pl.col("len").fill_null(0))
print("  треков каталога с альбомом: %d из %d (%.2f %%)"
      % ((full["len"] >= 1).sum(), K, 100 * (full["len"] >= 1).sum() / K))
print("  с несколькими альбомами: %d (%.2f %%); по всему каталогу 50M было 22,5 %% (блок 01)"
      % ((full["len"] > 1).sum(), 100 * (full["len"] > 1).sum() / K))
print("  число альбомов на трек: " + ", ".join(
    "%d: %d" % (v, full.filter(pl.col("len") == v).height) for v in [0, 1, 2, 3, 4, 5]) +
    ", >5: %d" % full.filter(pl.col("len") > 5).height)
print("  всего альбомов, в которые входят треки каталога: %d" % alb["album_id"].n_unique())
alb2 = alb.join(alb_size, on="album_id", how="left")
rules = {
    "A. минимальный album_id":
        alb2.sort(["item_id", "album_id"]).group_by("item_id").first(),
    "B. самый крупный альбом, при равенстве min album_id":
        alb2.sort(["item_id", "alb_tracks", "album_id"],
                  descending=[False, True, False]).group_by("item_id").first(),
    "C. самый мелкий альбом (сингл), при равенстве min album_id":
        alb2.sort(["item_id", "alb_tracks", "album_id"]).group_by("item_id").first(),
}
print("\n  %-56s %9s %9s %9s" % ("правило", "альбомов", "медиана", "макс"))
chosen = {}
for name, df in rules.items():
    g = df.group_by("album_id").len()
    chosen[name] = df.select("item_id", "album_id").sort("item_id")
    print("  %-56s %9d %9d %9d"
          % (name, df["album_id"].n_unique(), g["len"].median(), g["len"].max()))
print("  (альбомов = сколько разных album_id получает каталог; медиана/макс = "
      "треков каталога на выбранный альбом)")
print("  насколько альбом вообще группирует каталог:")
for name, df in rules.items():
    g = df.group_by("album_id").len()
    big = g.filter(pl.col("len") >= 2)
    print("    %-56s альбомов с >= 2 треками каталога %5d, в них %5d треков (%.1f %%)"
          % (name, big.height, big["len"].sum(), 100 * big["len"].sum() / K))
ka = list(chosen)
multi = sorted(set(full.filter(pl.col("len") > 1)["item_id"].to_list()))
M = pl.Series("item_id", multi)
print("  согласие правил на треках с несколькими альбомами (%d шт.):" % len(multi))
for i in range(len(ka)):
    for j in range(i + 1, len(ka)):
        m = (chosen[ka[i]].rename({"album_id": "a1"})
             .join(chosen[ka[j]].rename({"album_id": "a2"}), on="item_id")
             .filter(pl.col("item_id").is_in(M.implode())))
        print("    %s vs %s: совпадает у %d из %d (%.1f %%)"
              % (ka[i][0], ka[j][0], (m["a1"] == m["a2"]).sum(), m.height,
                 100 * (m["a1"] == m["a2"]).sum() / max(m.height, 1)))
det = all(d.group_by("item_id").len()["len"].max() == 1 for d in chosen.values())
print("  однозначность: каждое правило даёт ровно один альбом на трек — %s"
      % ("да" if det else "НЕТ"))
sz = alb2.group_by("item_id").agg(pl.col("alb_tracks").min().alias("mn"),
                                  pl.col("alb_tracks").max().alias("mx"))
sz = sz.filter(pl.col("item_id").is_in(M.implode()))
print("  у треков с несколькими альбомами размеры расходятся: медиана мелкого %d "
      "трек(ов), крупного %d; размеры равны у %d из %d"
      % (sz["mn"].median(), sz["mx"].median(),
         (sz["mn"] == sz["mx"]).sum(), sz.height))

# --- 2. эмбеддинги ---
print("\n=== 2. Аудиоэмбеддинги ===")
embids = EMB.select("item_id").unique().collect()
he = CAT.select("item_id").join(embids, on="item_id", how="semi").height
print("  вектор есть у %d из %d треков каталога (%.2f %%); остальным нулевой вектор "
      "и флаг has_embed = 0 (правило блока 01)" % (he, K, 100 * he / K))
loc = os.path.join(os.path.dirname(raw.rstrip("/\\")), "catalog_embeddings.parquet")
if os.path.exists(loc):
    e = pl.read_parquet(loc)
    print("  локальная выгрузка: %s (%.1f МБ), %d строк, схема %s"
          % (loc, os.path.getsize(loc) / 1e6, e.height,
             {k: str(v) for k, v in dict(e.collect_schema()).items()}))
    for c in ["embed", "normalized_embed"]:
        if c in e.columns:
            L = sorted(set(e[c].list.len().to_list()))
            nrm = e.select(pl.col(c).list.eval(pl.element() ** 2).list.sum()
                           .sqrt().alias("n"))["n"]
            print("    %-16s длина %s, норма: медиана %.4f, min %.4f, max %.4f"
                  % (c, L, nrm.median(), nrm.min(), nrm.max()))
    print("  треков каталога без вектора в выгрузке: %d"
          % CAT.select("item_id").join(e.select("item_id"), on="item_id",
                                       how="anti").height)
else:
    print("  локальной выгрузки нет; схема удалённого embeddings.parquet: "
          "item_id UInt32, embed List(Float64), normalized_embed List(Float64), "
          "размерность 128 (измерено удалённым head)")
    print("  выгрузить: python -I tools/checks/f4_export_embeddings.py data/raw")

# --- 3. популярность и возраст: по двум окнам ---
print("\n=== 3. Популярность и возраст трека по двум окнам ===")
ev = (LF.filter(pl.col("timestamp") < TEST0)
        .join(CAT.select("item_id").lazy(), on="item_id", how="semi")
        .join(pl.DataFrame({"uid": pool}).lazy(), on="uid", how="semi")
        .with_columns((pl.col("timestamp") < VAL0).alias("tr"),
                      pl.col("uid").is_in(S.implode()).alias("sm"))
        .group_by("item_id").agg(
            pl.len().alias("n_tv"),
            pl.col("tr").sum().alias("n_tr"),
            pl.col("uid").n_unique().alias("u_tv"),
            pl.col("uid").filter("tr").n_unique().alias("u_tr"),
            pl.col("uid").filter("sm").n_unique().alias("su_tv"),
            pl.col("uid").filter(pl.col("tr") & pl.col("sm")).n_unique().alias("su_tr"),
            pl.col("timestamp").min().alias("f_tv"),
            pl.col("timestamp").filter("tr").min().alias("f_tr"),
            pl.col("played_ratio_pct").mean().alias("pr_tv"),
            pl.col("played_ratio_pct").filter("tr").mean().alias("pr_tr"),
            (pl.col("played_ratio_pct") <= 1).mean().alias("skip_tv"),
            (pl.col("played_ratio_pct") >= 99).mean().alias("full_tv"),
            pl.col("is_organic").mean().alias("org_tv"),
        ).collect(engine="streaming"))
it = (CAT.join(ev, on="item_id", how="left")
         .with_columns((pl.col("f_tv") // DAY).alias("d_tv"),
                       (pl.col("f_tr") // DAY).alias("d_tr"))
         .with_columns((280 - pl.col("d_tv")).alias("age_tv"),
                       (280 - pl.col("d_tr")).alias("age_tr")))


def q(s, lab, fmt="%9.0f"):
    s = s.drop_nulls()
    print(("  %-44s" + fmt * 7) % (lab, s.min(), s.quantile(.1), s.quantile(.25),
                                   s.median(), s.quantile(.75), s.quantile(.9), s.max()))


print("  %-44s %9s %9s %9s %9s %9s %9s %9s"
      % ("признак (окно)", "min", "q10", "q25", "медиана", "q75", "q90", "max"))
q(it["u_tr"], "слушателей пула, train 0…258")
q(it["u_tv"], "слушателей пула, train+val 0…279")
q(it["su_tr"], "слушателей подвыборки, train")
q(it["su_tv"], "слушателей подвыборки, train+val")
q(it["n_tr"], "прослушиваний, train")
q(it["n_tv"], "прослушиваний, train+val")
q(it["pr_tr"], "средний played_ratio_pct, train", "%9.1f")
q(it["pr_tv"], "средний played_ratio_pct, train+val", "%9.1f")
q(it["skip_tv"] * 100, "доля скипов (<= 1 %), train+val", "%9.1f")
q(it["full_tv"] * 100, "доля полных (>= 99 %), train+val", "%9.1f")
q(it["org_tv"] * 100, "доля органики, train+val", "%9.1f")
q(it["age_tr"], "возраст к началу test, дебют по train, сут.")
q(it["age_tv"], "возраст к началу test, дебют по train+val, сут.")
print("  треков с дебютом в сутки 0 (возраст = левая граница наблюдения): %d (%.1f %%)"
      % ((it["d_tv"] == 0).sum(), 100 * (it["d_tv"] == 0).sum() / K))
print("  треков каталога без прослушиваний в train 0…258 (признаки по train вырождены "
      "в нуль): %d; из них дебютировали внутри val: %d"
      % ((it["u_tr"] == 0).sum(), (it["d_tv"] >= 259).sum()))
print("  дебют внутри val (сутки 259…279): %d треков; дебют в сутки 250…258: %d"
      % ((it["d_tv"] >= 259).sum(),
         it.filter((pl.col("d_tv") >= 250) & (pl.col("d_tv") <= 258)).height))
lg = (it["su_tv"].fill_null(0) + 1).log()
print("  log1p(слушателей подвыборки, train+val): медиана %.2f, разброс %.2f…%.2f"
      % (lg.median(), lg.min(), lg.max()))

print("\n  расхождение популярности между окнами (train против train+val):")
a = it.select("item_id", "u_tr", "u_tv").with_columns(pl.col("u_tr").fill_null(0))
r_tr = a.sort(["u_tr", "item_id"], descending=[True, False]).with_row_index("r_tr", 1)
r_tv = a.sort(["u_tv", "item_id"], descending=[True, False]).with_row_index("r_tv", 1)
m = r_tr.select("item_id", "r_tr").join(r_tv.select("item_id", "r_tv"), on="item_id")
d = (m["r_tr"].cast(pl.Int64) - m["r_tv"].cast(pl.Int64)).abs()
print("    |сдвиг ранга|: медиана %d, q90 %d, q99 %d, max %d"
      % (d.median(), d.quantile(.9), d.quantile(.99), d.max()))
for t in [100, 1000, 5000]:
    o = len(set(r_tr.head(t)["item_id"].to_list())
            & set(r_tv.head(t)["item_id"].to_list()))
    print("    пересечение топ-%d: %d (%.1f %%)" % (t, o, 100 * o / t))
print("    корреляция Спирмена по рангам: %.4f"
      % pl.DataFrame({"a": m["r_tr"].cast(pl.Float64),
                      "b": m["r_tv"].cast(pl.Float64)})
          .select(pl.corr("a", "b")).item())
gr = ((it["u_tv"] - it["u_tr"].fill_null(0)) / it["u_tr"].fill_null(0).clip(1))
print("    прирост слушателей за val: медиана %.1f %%, q90 %.1f %% от train-значения"
      % (100 * gr.median(), 100 * gr.quantile(.9)))

# --- 4. артист как признак ---
print("\n=== 4. Артист как признак ===")
amap = AM.unique(subset=["item_id"], keep="first").collect()
nart = AM.collect().filter(pl.col("item_id").is_in(C.implode()))
g = nart.group_by("item_id").len()
print("  треков каталога с артистом: %d; с несколькими артистами: %d"
      % (g.height, (g["len"] > 1).sum()))
ca = CAT.join(amap, on="item_id", how="left")
ga = ca.group_by("artist_id").len()
print("  артистов в каталоге %d; треков каталога на артиста: медиана %d, q75 %d, "
      "q90 %d, max %d" % (ga.height, ga["len"].median(), ga["len"].quantile(.75),
                          ga["len"].quantile(.9), ga["len"].max()))
ap = (ca.join(it.select("item_id", "u_tv", "su_tv"), on="item_id", how="left")
        .group_by("artist_id").agg(pl.col("su_tv").sum().alias("asu")))
q(ap["asu"], "слушателей подвыборки на артиста (сумма)")
print("  артистов с одним треком в каталоге: %d (%.1f %%) — для них признак артиста "
      "дублирует признак трека"
      % ((ga["len"] == 1).sum(), 100 * (ga["len"] == 1).sum() / ga.height))
