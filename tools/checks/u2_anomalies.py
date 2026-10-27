import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# Блок 03. Три вида аномалий: ботоподобные повторы, почти круглосуточная
# активность, перелистывание. Единица timestamp = 1 с (блок 02).
DAY = 86400
NDAYS = 301
TEST0 = 280 * DAY             # предварительные границы блока 02
VAL0 = 259 * DAY
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))

base = (LF.select("uid", "item_id", "timestamp", "played_ratio_pct")
          .with_columns((pl.col("timestamp") // DAY).alias("d"),
                        ((pl.col("timestamp") % DAY) // 3600).alias("h")))
per_user = (base.group_by("uid")
                .agg(pl.len().alias("n"),
                     pl.col("item_id").n_unique().alias("items"),
                     pl.col("d").n_unique().alias("days"),
                     pl.col("played_ratio_pct").mean().alias("pr"),
                     (pl.col("played_ratio_pct") >= 50).mean().alias("rel"),
                     (pl.col("played_ratio_pct") == 0).mean().alias("zero"),
                     (pl.col("h") <= 4).mean().alias("night"),
                     (pl.col("timestamp") < VAL0).sum().alias("n_train"))
                .collect(engine="streaming"))
N = per_user.height
print("пользователей с прослушиваниями: %d" % N)
print("для справки: доля потока в часах 00–04 по всем данным 8,62 %%, "
      "средняя по пользователям %.2f %%" % (100 * per_user["night"].mean()))

# --- 1. ботоподобные повторы: >= 20 строк на один ключ (uid, item, timestamp) ---
print("\n=== 1. повторы одного трека в одну секунду (определение блока 01) ===")
key = (base.group_by(["uid", "item_id", "timestamp"]).len()
           .filter(pl.col("len") >= 2).collect(engine="streaming"))
print("  ключей с повтором: %d, лишних строк %d (%.2f %% данных)"
      % (key.height, (key["len"] - 1).sum(),
         100 * (key["len"] - 1).sum() / LF.select(pl.len()).collect().item()))
for thr in [2, 5, 10, 20, 50, 100]:
    s = key.filter(pl.col("len") >= thr)
    print("    ключей с >= %3d повторами: %6d, пользователей %4d, максимум %d"
          % (thr, s.height, s["uid"].n_unique(), s["len"].max() if s.height else 0))
bots = key.filter(pl.col("len") >= 20)["uid"].unique()
print("  «ботов» (хотя бы один ключ с >= 20 повторами): %d" % len(bots))
B = per_user.filter(pl.col("uid").is_in(bots.implode()))
R = per_user.filter(~pl.col("uid").is_in(bots.implode()))
print("  %-28s %12s %12s" % ("", "«боты» (%d)" % B.height, "остальные (%d)" % R.height))
for c, lab, f in [("n", "прослушиваний, медиана", "%12.0f"),
                  ("items", "треков, медиана", "%12.0f"),
                  ("days", "активных суток, медиана", "%12.0f"),
                  ("pr", "played_ratio, среднее", "%12.1f"),
                  ("rel", "доля >= 50 %", "%12.3f"),
                  ("zero", "доля == 0", "%12.3f"),
                  ("night", "доля часов 00–04", "%12.3f")]:
    bv = B[c].median() if c in ("n", "items", "days") else B[c].mean()
    rv = R[c].median() if c in ("n", "items", "days") else R[c].mean()
    print(("  %-28s " + f + " " + f) % (lab, bv, rv))
print("  доля всех прослушиваний у «ботов»: %.2f %% (%d из %d)"
      % (100 * B["n"].sum() / per_user["n"].sum(), B["n"].sum(), per_user["n"].sum()))
# сколько у них самих приходится на залипшие повторы
bk = key.filter(pl.col("uid").is_in(bots.implode()) & (pl.col("len") >= 20))
agg = bk.group_by("uid").agg((pl.col("len") - 1).sum().alias("extra"),
                             pl.len().alias("keys"))
j = agg.join(B.select("uid", "n"), on="uid")
sh = (j["extra"] / j["n"])
print("  у «ботов» доля строк, лежащих в тяжёлых повторах: медиана %.4f, "
      "q90 %.4f, max %.4f" % (sh.median(), sh.quantile(.9), sh.max()))
print("  то есть даже у них залипшие повторы — %.2f %% их собственных прослушиваний"
      % (100 * j["extra"].sum() / j["n"].sum()))
print("  played_ratio внутри тяжёлых повторов против остальных прослушиваний "
      "тех же пользователей:")
heavy = key.filter(pl.col("len") >= 20).select("uid", "item_id", "timestamp").lazy()
hv = (base.join(heavy, on=["uid", "item_id", "timestamp"], how="semi")
          .select("played_ratio_pct").collect(engine="streaming"))["played_ratio_pct"]
ov = (base.filter(pl.col("uid").is_in(bots.implode()))
          .join(heavy, on=["uid", "item_id", "timestamp"], how="anti")
          .select("played_ratio_pct").collect(engine="streaming"))["played_ratio_pct"]
print("    в повторах: строк %d, среднее %.1f, доля == 0 %.3f, доля >= 50 %.3f"
      % (len(hv), hv.mean(), (hv == 0).mean(), (hv >= 50).mean()))
print("    вне повторов: строк %d, среднее %.1f, доля == 0 %.3f, доля >= 50 %.3f"
      % (len(ov), ov.mean(), (ov == 0).mean(), (ov >= 50).mean()))

# --- 2. почти круглосуточная активность ---
print("\n=== 2. почти круглосуточная активность (оставленный плеер?) ===")
ud = (base.group_by(["uid", "d"])
          .agg(pl.col("h").n_unique().alias("hrs"), pl.len().alias("n"))
          .collect(engine="streaming"))
print("  пар (пользователь, активные сутки): %d" % ud.height)
hh = ud["hrs"]
print("  занятых часов в активные сутки: " + " ".join(
    "q%02d=%d" % (int(q * 100), hh.quantile(q)) for q in [.25, .5, .75, .9, .95, .99])
      + " max=%d" % hh.max())
for thr in [20, 22, 23, 24]:
    print("    суток с >= %d занятыми часами: %d (%.3f %% активных суток), "
          "пользователей %d"
          % (thr, (hh >= thr).sum(), 100 * (hh >= thr).mean(),
             ud.filter(pl.col("hrs") >= thr)["uid"].n_unique()))
per = (ud.group_by("uid").agg(pl.col("hrs").max().alias("h_max"),
                              pl.col("hrs").median().alias("h_med"),
                              (pl.col("hrs") >= 23).sum().alias("d23"),
                              pl.len().alias("days")))
print("  по пользователям: медиана занятых часов в сутки — медиана %.0f, q90 %.0f, "
      "max %.0f" % (per["h_med"].median(), per["h_med"].quantile(.9), per["h_med"].max()))
for thr in [1, 5, 20, 50]:
    k = (per["d23"] >= thr).sum()
    print("    пользователей с >= %2d сутками по >= 23 часа: %d (%.2f %%)"
          % (thr, k, 100 * k / N))
hard = per.filter(pl.col("d23") >= 20)
if hard.height:
    hj = hard.join(per_user, on="uid")
    print("  срез «>= 20 суток по >= 23 часа» (%d чел.): медиана прослушиваний %.0f, "
          "медиана активных суток %.0f, доля >= 50 %% %.3f, доля часов 00–04 %.3f"
          % (hard.height, hj["n"].median(), hj["days_right"].median()
             if "days_right" in hj.columns else hj["days"].median(),
             hj["rel"].mean(), hj["night"].mean()))
print("  доля ночных прослушиваний (часы 00–04) по пользователям: " + " ".join(
    "q%02d=%.3f" % (int(q * 100), per_user["night"].quantile(q))
    for q in [.5, .9, .95, .99]) + " max=%.3f" % per_user["night"].max())
for thr in [.25, .35, .5]:
    k = (per_user["night"] > thr).sum()
    print("    доля ночных > %.0f %% (норма 8,6 %%): %d пользователей (%.2f %%)"
          % (100 * thr, k, 100 * k / N))

# --- 3. пересечение аномалий ---
print("\n=== 3. пересечение трёх признаков аномалии ===")
flag = (per_user.with_columns(
    pl.col("uid").is_in(bots.implode()).alias("f_bot"),
    (pl.col("zero") > 0.5).alias("f_zero"),
    (pl.col("night") > 0.35).alias("f_night"))
    .join(per.select("uid", "d23"), on="uid", how="left")
    .with_columns((pl.col("d23").fill_null(0) >= 20).alias("f_24h")))
for c, lab in [("f_bot", ">= 20 повторов в секунду"),
               ("f_zero", "доля played_ratio == 0 > 50 %"),
               ("f_night", "доля часов 00–04 > 35 %"),
               ("f_24h", ">= 20 суток по >= 23 часа")]:
    s = flag.filter(pl.col(c))
    print("  %-32s %4d чел., прослушиваний %9d (%.2f %%), медиана активных суток %.0f"
          % (lab, s.height, s["n"].sum(), 100 * s["n"].sum() / flag["n"].sum(),
             s["days"].median() if s.height else 0))
anyf = flag.filter(pl.col("f_bot") | pl.col("f_zero") | pl.col("f_night")
                   | pl.col("f_24h"))
print("  хотя бы один признак: %d чел. (%.2f %%), прослушиваний %d (%.2f %%)"
      % (anyf.height, 100 * anyf.height / N, anyf["n"].sum(),
         100 * anyf["n"].sum() / flag["n"].sum()))
print("  два и более признака: %d чел."
      % flag.filter((pl.col("f_bot").cast(pl.Int8) + pl.col("f_zero").cast(pl.Int8)
                     + pl.col("f_night").cast(pl.Int8)
                     + pl.col("f_24h").cast(pl.Int8)) >= 2).height)

# --- 4. ночная активность: аномалия или другой часовой пояс? ---
# если у «ночных» пользователей профиль остаётся одногорбым, но сдвинут,
# это часовой пояс, а не оставленный плеер
print("\n=== 4. «ночные» пользователи: сдвинутый профиль или размазанный? ===")
prof = (base.group_by(["uid", "h"]).len()
            .collect(engine="streaming")
            .pivot(on="h", index="uid", values="len", aggregate_function="sum")
            .fill_null(0))
cols = [str(h) for h in range(24) if str(h) in prof.columns]
import numpy as np
M = prof.select(cols).to_numpy().astype(float)
uids = prof["uid"].to_numpy()
tot = M.sum(axis=1, keepdims=True)
P = M / np.maximum(tot, 1)
peak = np.array([int(cols[i]) for i in P.argmax(axis=1)])
top5 = np.sort(P, axis=1)[:, -5:].sum(axis=1)   # доля в пяти самых активных часах
night = P[:, [int(c) for c in cols].index(0):5].sum(axis=1) if len(cols) == 24 \
    else P[:, :5].sum(axis=1)
# «размазанность»: энтропия профиля, нормированная на максимум (log 24)
ent = -(P * np.log(np.maximum(P, 1e-12))).sum(axis=1) / np.log(24)
grp = {"все": np.ones(len(uids), bool),
       "ночные > 35 %": night > .35,
       "ночные > 50 %": night > .50,
       "обычные (<= 25 %)": night <= .25}
for lab, m in grp.items():
    if m.sum() == 0:
        continue
    print("  %-18s %4d чел.: медиана часа пика %2d, доля в 5 самых активных часах %.3f, "
          "медиана энтропии %.3f"
          % (lab, m.sum(), int(np.median(peak[m])), float(np.median(top5[m])),
             float(np.median(ent[m]))))
m = night > .35
print("  часы пика у «ночных > 35 %%»: " + ", ".join(
    "%02dч: %d" % (h, int((peak[m] == h).sum())) for h in range(24)
    if (peak[m] == h).sum() > 0))
print("  для сравнения, часы пика у обычных: " + ", ".join(
    "%02dч: %d" % (h, int((peak[night <= .25] == h).sum())) for h in range(24)
    if (peak[night <= .25] == h).sum() > 0))
print("  энтропия 1.0 = активность ровно размазана по суткам; "
      "сколько «ночных» с энтропией > 0.95: %d из %d"
      % (int((ent[m] > .95).sum()), int(m.sum())))
