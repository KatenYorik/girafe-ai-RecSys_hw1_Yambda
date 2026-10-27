import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, random
raw = sys.argv[1]
# Блок 09. Агрегаты пользователя как признаки FM. Считаются на подвыборке из 3 000
# по двум окнам и только по ним: train = сутки 0…258 (для подбора гиперпараметров на
# val) и train+val = сутки 0…279 (для финальной модели). Тестовое окно (сутки 280…300)
# не участвует ни в одном агрегате. Лайк и дизлайк берутся как СОСТОЯНИЕ на конец окна
# (правило блока 01: итог по паре — последнее событие среди like/unlike/dislike/undislike).
DAY = 86400
VAL0, TEST0 = 259 * DAY, 280 * DAY
N, K, SEED = 3000, 10000, 42
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
AM = pl.scan_parquet(os.path.join(raw, "artist_item_mapping.parquet"))

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
print("пул %d, подвыборка %d, каталог %d" % (len(pool), N, K))

# --- событийные агрегаты по двум окнам ---
ev = (LF.filter(pl.col("timestamp") < TEST0)
        .join(US.lazy(), on="uid", how="semi")
        .with_columns((pl.col("timestamp") < VAL0).alias("tr"),
                      (pl.col("timestamp") // DAY).alias("d"))
        .group_by("uid").agg(
            pl.len().alias("n_tv"), pl.col("tr").sum().alias("n_tr"),
            pl.col("d").n_unique().alias("days_tv"),
            pl.col("d").filter("tr").n_unique().alias("days_tr"),
            pl.col("d").min().alias("d0_tv"), pl.col("d").filter("tr").min().alias("d0_tr"),
            pl.col("d").max().alias("dl_tv"), pl.col("d").filter("tr").max().alias("dl_tr"),
            pl.col("is_organic").mean().alias("org_tv"),
            pl.col("is_organic").filter("tr").mean().alias("org_tr"),
            pl.col("played_ratio_pct").mean().alias("pr_tv"),
            pl.col("played_ratio_pct").filter("tr").mean().alias("pr_tr"),
            (pl.col("played_ratio_pct") <= 1).mean().alias("sk_tv"),
            (pl.col("played_ratio_pct") <= 1).filter("tr").mean().alias("sk_tr"),
            (pl.col("played_ratio_pct") >= 99).mean().alias("fu_tv"),
            (pl.col("played_ratio_pct") >= 99).filter("tr").mean().alias("fu_tr"),
        ).collect(engine="streaming"))

# --- парные агрегаты (треки, артисты, повторы) по двум окнам ---
pr = (LF.filter(pl.col("timestamp") < TEST0)
        .join(US.lazy(), on="uid", how="semi")
        .with_columns((pl.col("timestamp") < VAL0).alias("tr"))
        .group_by(["uid", "item_id"]).agg(
            pl.len().alias("n_tv"), pl.col("tr").sum().alias("n_tr"))
        .collect(engine="streaming"))
amap = AM.unique(subset=["item_id"], keep="first").collect()
pa = pr.join(amap, on="item_id", how="left")
up = pa.group_by("uid").agg(
    pl.col("item_id").n_unique().alias("it_tv"),
    pl.col("item_id").filter(pl.col("n_tr") > 0).n_unique().alias("it_tr"),
    pl.col("artist_id").n_unique().alias("ar_tv"),
    pl.col("artist_id").filter(pl.col("n_tr") > 0).n_unique().alias("ar_tr"),
    pl.col("n_tv").mean().alias("rep_tv"),
    pl.col("n_tr").filter(pl.col("n_tr") > 0).mean().alias("rep_tr"),
)
cat_pr = pr.filter(pl.col("item_id").is_in(C.implode()))
uc = cat_pr.group_by("uid").agg(
    pl.col("n_tv").sum().alias("cn_tv"), pl.col("n_tr").sum().alias("cn_tr"),
    pl.col("item_id").n_unique().alias("cit_tv"),
    pl.col("item_id").filter(pl.col("n_tr") > 0).n_unique().alias("cit_tr"))

# --- лайки и дизлайки как состояние на конец окна ---
parts = []
for f, k in [("likes", 1), ("unlikes", 0), ("dislikes", -1), ("undislikes", 0)]:
    parts.append(pl.scan_parquet(os.path.join(raw, f + ".parquet"))
                   .select("uid", "item_id", "timestamp")
                   .with_columns(pl.lit(k, pl.Int8).alias("st"),
                                 pl.lit(f).alias("src")))
EVT = pl.concat(parts).join(US.lazy(), on="uid", how="semi").collect()


def like_state(W):
    e = EVT.filter(pl.col("timestamp") < W)
    last = (e.sort(["uid", "item_id", "timestamp"])
             .group_by(["uid", "item_id"]).agg(pl.col("st").last().alias("st"),
                                               pl.col("timestamp").last().alias("t"),
                                               pl.len().alias("k"),
                                               pl.col("timestamp").max().alias("tm")))
    ties = last.filter((pl.col("k") > 1) & (pl.col("t") == pl.col("tm"))).height
    return last, ties


ul = {}
for lab, W in [("tr", VAL0), ("tv", TEST0)]:
    last, _ = like_state(W)
    ul[lab] = last.group_by("uid").agg(
        (pl.col("st") == 1).sum().alias("lik_" + lab),
        (pl.col("st") == -1).sum().alias("dis_" + lab))

U = (US.join(ev, on="uid", how="left").join(up, on="uid", how="left")
       .join(uc, on="uid", how="left")
       .join(ul["tr"], on="uid", how="left").join(ul["tv"], on="uid", how="left")
       .with_columns(pl.col("lik_tr", "dis_tr", "lik_tv", "dis_tv").fill_null(0),
                     (pl.col("dl_tv") - pl.col("d0_tv") + 1).alias("span_tv"),
                     (pl.col("dl_tr") - pl.col("d0_tr") + 1).alias("span_tr"),
                     (259 - pl.col("d0_tr")).alias("hist_tr"),
                     (280 - pl.col("d0_tv")).alias("hist_tv")))

ROWS = [
    ("прослушиваний", "n_tr", "n_tv", "%9.0f"),
    ("прослушиваний на треках каталога", "cn_tr", "cn_tv", "%9.0f"),
    ("различных треков", "it_tr", "it_tv", "%9.0f"),
    ("различных треков каталога", "cit_tr", "cit_tv", "%9.0f"),
    ("различных артистов", "ar_tr", "ar_tv", "%9.0f"),
    ("активных суток", "days_tr", "days_tv", "%9.0f"),
    ("длина истории до конца окна, сут.", "hist_tr", "hist_tv", "%9.0f"),
    ("от первого до последнего события, сут.", "span_tr", "span_tv", "%9.0f"),
    ("доля органики, %", "org_tr", "org_tv", "%9.1f"),
    ("средний played_ratio_pct", "pr_tr", "pr_tv", "%9.1f"),
    ("доля скипов (<= 1 %), %", "sk_tr", "sk_tv", "%9.1f"),
    ("доля полных (>= 99 %), %", "fu_tr", "fu_tv", "%9.1f"),
    ("лайков (состояние на конец окна)", "lik_tr", "lik_tv", "%9.0f"),
    ("дизлайков (состояние на конец окна)", "dis_tr", "dis_tv", "%9.0f"),
    ("повторов на пару (среднее)", "rep_tr", "rep_tv", "%9.2f"),
]
SC = {"org_tr", "org_tv", "sk_tr", "sk_tv", "fu_tr", "fu_tv"}


def col(c):
    s = U[c].cast(pl.Float64)
    return s * 100 if c in SC else s


print("\n=== Агрегаты пользователя: квантили по 3 000 пользователям ===")
print("  %-40s %5s %9s %9s %9s %9s %9s %9s %9s %7s"
      % ("признак", "окно", "min", "q10", "q25", "медиана", "q75", "q90", "max", "н/д"))
for lab, ctr, ctv, fmt in ROWS:
    for w, c in [("train", ctr), ("tr+val", ctv)]:
        s = col(c)
        nd = s.null_count()
        s = s.drop_nulls()
        print(("  %-40s %5s" + fmt * 7 + " %7d")
              % (lab if w == "train" else "", w, s.min(), s.quantile(.1),
                 s.quantile(.25), s.median(), s.quantile(.75), s.quantile(.9),
                 s.max(), nd))
print("  «н/д» — пользователей из 3 000, у которых признак не определён в этом окне")

print("\n=== Расхождение train против train+val ===")
print("  %-40s %12s %12s %10s %10s %10s"
      % ("признак", "медиана train", "медиана tr+val", "прирост", "медиана |d|", "corr"))
for lab, ctr, ctv, fmt in ROWS:
    a, b = col(ctr), col(ctv)
    msk = a.is_not_null() & b.is_not_null()
    a2, b2 = a.filter(msk), b.filter(msk)
    rel = ((b2 - a2) / a2.abs().clip(1e-9))
    cr = pl.DataFrame({"a": a2, "b": b2}).select(pl.corr("a", "b")).item()
    print("  %-40s %12.2f %12.2f %9.1f %% %10.3f %10.4f"
          % (lab, a2.median(), b2.median(), 100 * rel.median(),
             (b2 - a2).abs().median(), cr if cr is not None else float("nan")))
print("  «прирост» — медиана относительного изменения признака при переходе train → "
      "train+val; «corr» — корреляция Пирсона между значениями в двух окнах")

print("\n=== Пользователи, у которых агрегаты особенно подвижны ===")
for lab, ctr, ctv, fmt in [r for r in ROWS if r[1] in
                           ("n_tr", "it_tr", "ar_tr", "lik_tr", "days_tr")]:
    a, b = col(ctr), col(ctv)
    rel = ((b - a) / a.abs().clip(1e-9)).drop_nulls()
    print("  %-40s доля с ростом >= 25 %%: %5.1f %%, >= 50 %%: %5.1f %%, без изменения: %5.1f %%"
          % (lab, 100 * (rel >= .25).mean(), 100 * (rel >= .5).mean(),
             100 * (rel.abs() < 1e-9).mean()))

print("\n=== Пустые и вырожденные случаи ===")
print("  без прослушиваний в train 0…258: %d; в train+val 0…279: %d"
      % ((U["n_tr"].fill_null(0) == 0).sum(), (U["n_tv"].fill_null(0) == 0).sum()))
print("  без прослушиваний на треках каталога: train %d, train+val %d"
      % ((U["cn_tr"].fill_null(0) == 0).sum(), (U["cn_tv"].fill_null(0) == 0).sum()))
print("  без ни одного лайка: train %d (%.1f %%), train+val %d (%.1f %%)"
      % ((U["lik_tr"] == 0).sum(), 100 * (U["lik_tr"] == 0).sum() / N,
         (U["lik_tv"] == 0).sum(), 100 * (U["lik_tv"] == 0).sum() / N))
print("  без ни одного дизлайка: train %d (%.1f %%), train+val %d (%.1f %%)"
      % ((U["dis_tr"] == 0).sum(), 100 * (U["dis_tr"] == 0).sum() / N,
         (U["dis_tv"] == 0).sum(), 100 * (U["dis_tv"] == 0).sum() / N))
print("  доля пользователей, у которых все прослушивания организованы (org = 1): %.2f %%; "
      "ни одного органического: %.2f %%"
      % (100 * (U["org_tv"] == 1).mean(), 100 * (U["org_tv"] == 0).mean()))
print("  history: первое событие в сутки 0 у %d пользователей (левая цензура)"
      % (U["d0_tv"] == 0).sum())
