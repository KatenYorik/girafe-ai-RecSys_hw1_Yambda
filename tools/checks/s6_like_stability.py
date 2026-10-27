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

def load(f, pr, ev):
    return (pl.read_parquet(os.path.join(raw, f + ".parquet"))
              .select("uid", "item_id", "timestamp", "is_organic")
              .filter(pl.col("timestamp") < TRAIN_END)
              .unique(subset=["uid", "item_id", "timestamp", "is_organic"])   # блок 01
              .with_columns(pl.lit(pr).alias("pr"), pl.lit(ev).alias("ev")))
LK, UL = load("likes", 0, "like"), load("unlikes", 1, "unlike")
DL, UD = load("dislikes", 2, "dislike"), load("undislikes", 3, "undislike")
print("события в train (после дедупа): likes %d, unlikes %d, dislikes %d, undislikes %d"
      % (LK.height, UL.height, DL.height, UD.height))
print("пар: с лайком %d, с анлайком %d, с дизлайком %d, с андизлайком %d"
      % (LK.select("uid","item_id").unique().height, UL.select("uid","item_id").unique().height,
         DL.select("uid","item_id").unique().height, UD.select("uid","item_id").unique().height))
ev = pl.concat([LK, UL, DL, UD])
st = (ev.sort(["uid", "item_id", "timestamp", "pr"])
        .group_by(["uid", "item_id"]).agg(pl.col("ev").last().alias("state")))
print("итоговое состояние пар на конец train:", dict(st.group_by("state").len().sort("state").iter_rows()))

print("\n=== 1. как часто отменяют лайк ===")
# для каждого анлайка — последний лайк перед ним
pairs_lk = LK.select("uid", "item_id").unique()
ul_j = (UL.select("uid", "item_id", pl.col("timestamp").alias("u_ts"))
          .join(LK.select("uid", "item_id", pl.col("timestamp").alias("l_ts")),
                on=["uid", "item_id"], how="inner")
          .filter(pl.col("l_ts") < pl.col("u_ts"))
          .group_by(["uid", "item_id", "u_ts"]).agg(pl.col("l_ts").max()))
orph = UL.join(pairs_lk, on=["uid", "item_id"], how="anti")
print("  анлайков в train %d: из них с лайком внутри окна %d (%.1f %%), "
      "без лайка %d (%.1f %%) — левая цензура (блок 01)"
      % (UL.height, ul_j.height, 100*ul_j.height/UL.height, orph.height, 100*orph.height/UL.height))
# доля лайков, отменённых когда-либо внутри окна
lk_pairs = LK.group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("l0"),
                                               pl.col("timestamp").max().alias("l1"))
canc = lk_pairs.join(ul_j.group_by(["uid", "item_id"]).agg(pl.col("u_ts").max()),
                     on=["uid", "item_id"], how="left")
print("  пар с лайком внутри окна %d; из них когда-либо отменён %d (%.2f %%); "
      "итоговое состояние не «лайк» у %d (%.2f %%)"
      % (canc.height, canc["u_ts"].is_not_null().sum(),
         100*canc["u_ts"].is_not_null().mean(),
         canc.height - st.filter(pl.col("state") == "like").height,
         100*(canc.height - st.filter(pl.col("state") == "like").height)/canc.height))

print("\n=== 2. задержка «лайк -> анлайк», дни ===")
d = (ul_j.with_columns(((pl.col("u_ts") - pl.col("l_ts")) / DAY).alias("lag")))["lag"]
print("  наблюдений %d: median %.1f  q10 %.2f  q25 %.1f  q75 %.1f  q90 %.1f  mean %.1f  max %.1f"
      % (d.len(), d.median(), d.quantile(.1), d.quantile(.25), d.quantile(.75),
         d.quantile(.9), d.mean(), d.max()))
for h in [1/24, 1, 7, 30, 90, 180, 365, 730]:
    lab = "1 час" if h < 1 else "%g дн." % h
    print("  отменено в пределах %-8s %8d (%5.1f %% всех отмен)" % (lab, (d <= h).sum(), 100*(d <= h).mean()))

print("\n=== 3. поправка на цензуру: доля лайков, отменённых за H дней ===")
print("  (берём только лайки, у которых до конца train остаётся >= H дней)")
print("   H, дней   лайков с полным окном   отменено за H дней   доля")
lkh = (LK.select("uid", "item_id", pl.col("timestamp").alias("l_ts"))
         .join(ul_j.select("uid", "item_id", "l_ts", "u_ts"), on=["uid", "item_id", "l_ts"], how="left"))
for H in [1, 7, 30, 90, 180, 365, 730]:
    s = lkh.filter(pl.col("l_ts") <= TRAIN_END - H * DAY)
    c = s.filter(pl.col("u_ts").is_not_null() & ((pl.col("u_ts") - pl.col("l_ts")) <= H * DAY))
    print("  %7d   %21d   %18d   %5.2f %%" % (H, s.height, c.height, 100*c.height/s.height))

print("\n=== 4. дизлайки ===")
dl_pairs = DL.select("uid", "item_id").unique()
print("  дизлайков %d у %d пользователей; пар с дизлайком %d"
      % (DL.height, DL["uid"].n_unique(), dl_pairs.height))
print("  отмен дизлайка (undislike) %d; из них с дизлайком внутри окна %d (%.1f %%)"
      % (UD.height, UD.join(dl_pairs, on=["uid","item_id"], how="semi").height,
         100*UD.join(dl_pairs, on=["uid","item_id"], how="semi").height/UD.height))
both = dl_pairs.join(pairs_lk, on=["uid", "item_id"], how="semi")
print("  пар, которые когда-либо и лайкнуты, и дизлайкнуты: %d (%.1f %% пар с дизлайком)"
      % (both.height, 100*both.height/dl_pairs.height))
ud_j = (UD.select("uid","item_id", pl.col("timestamp").alias("u_ts"))
          .join(DL.select("uid","item_id", pl.col("timestamp").alias("l_ts")),
                on=["uid","item_id"], how="inner").filter(pl.col("l_ts") < pl.col("u_ts"))
          .group_by(["uid","item_id","u_ts"]).agg(pl.col("l_ts").max()))
dd = (ud_j.with_columns(((pl.col("u_ts")-pl.col("l_ts"))/DAY).alias("lag")))["lag"]
print("  задержка «дизлайк -> отмена», дней: median %.1f q25 %.1f q75 %.1f; в пределах суток %.1f %%"
      % (dd.median(), dd.quantile(.25), dd.quantile(.75), 100*(dd <= 1).mean()))
dlp = DL.group_by(["uid","item_id"]).agg(pl.col("timestamp").min().alias("d0"))
print("  дизлайков, снятых внутри окна: %d (%.1f %% пар с дизлайком)"
      % (ud_j.select("uid","item_id").unique().height,
         100*ud_j.select("uid","item_id").unique().height/dl_pairs.height))

print("\n=== 5. слушают ли дальше после отмены лайка ===")
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet")).filter(pl.col("timestamp") < TRAIN_END)
u_last = ul_j.group_by(["uid", "item_id"]).agg(pl.col("u_ts").max())
after = (LF.select("uid", "item_id", "timestamp")
           .join(u_last.lazy(), on=["uid", "item_id"], how="inner")
           .group_by(["uid", "item_id"]).agg(
                (pl.col("timestamp") > pl.col("u_ts")).sum().alias("post"),
                (pl.col("timestamp") <= pl.col("u_ts")).sum().alias("pre"),
                pl.col("u_ts").first())
           .collect(engine="streaming"))
# учитываем только те отмены, после которых в окне остаётся хотя бы 30 дней
a30 = after.filter(pl.col("u_ts") <= TRAIN_END - 30 * DAY)
print("  пар «лайк -> анлайк» с прослушиваниями в train: %d (у %d остаётся >= 30 дней окна)"
      % (after.height, a30.height))
print("  прослушиваний до отмены: median %d; после отмены: median %d, среднее %.2f"
      % (a30["pre"].median(), a30["post"].median(), a30["post"].mean()))
print("  доля пар, у которых после отмены лайка есть хотя бы одно прослушивание: %.1f %%; "
      ">= 3 прослушивания: %.1f %%" % (100*(a30["post"] >= 1).mean(), 100*(a30["post"] >= 3).mean()))
