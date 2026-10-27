import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os, time
raw = sys.argv[1]
# ВНИМАНИЕ (поправка блока 02): DAY = 17280 — это НЕ сутки, а 1/5 суток (4 ч 48 мин).
# Единица timestamp равна 1 секунде, реальные сутки = 86400 единиц (см. t1_time_unit.py).
# Константа оставлена как есть, чтобы числа блока воспроизводились: любой горизонт,
# который этот скрипт печатает «в днях», читается как (дни / 5) реальных суток.
DAY = 17280
TRAIN_DAYS = 1414                 # блок 06: test = дни 1414..1503 (90 дней)
TRAIN_END = TRAIN_DAYS * DAY      # train = дни 0..1413
KEY = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct"]
t = time.time()
LF = (pl.scan_parquet(os.path.join(raw, "listens.parquet"))
        .filter(pl.col("timestamp") < TRAIN_END)
        .select(KEY).unique(subset=KEY)
        .with_columns(pl.min_horizontal(pl.col("played_ratio_pct"), pl.lit(100))
                        .cast(pl.Int32).alias("r")))
p = (LF.group_by(["uid", "item_id"]).agg(
        pl.len().alias("n"),
        pl.col("timestamp").min().alias("t0"),
        # первое прослушивание пары; если в одной 5-секундной корзине их несколько (блок 01),
        # берём самое полное
        pl.col("r").filter(pl.col("timestamp") == pl.col("timestamp").min()).max().alias("r0"),
        pl.col("r").filter(pl.col("timestamp") == pl.col("timestamp").min()).len().alias("k0"),
        pl.col("is_organic").filter(pl.col("timestamp") == pl.col("timestamp").min()).max().alias("o0"))
      .collect(engine="streaming"))
print("пар в train: %d (%.1f с); у %d пар (%.2f %%) в первой корзине больше одного прослушивания"
      % (p.height, time.time() - t, p.filter(pl.col("k0") > 1).height,
         100*p.filter(pl.col("k0") > 1).height/p.height))

# --- состояние лайка на конец train (правила блока 01) ---
def load(f, ev, pr):
    return (pl.read_parquet(os.path.join(raw, f + ".parquet"))
              .select("uid", "item_id", "timestamp", "is_organic")
              .filter(pl.col("timestamp") < TRAIN_END)
              .unique(subset=["uid", "item_id", "timestamp", "is_organic"])
              .with_columns(pl.lit(ev).alias("ev"), pl.lit(pr).alias("pr")))
ev = pl.concat([load("likes", "like", 0), load("unlikes", "unlike", 1),
                load("dislikes", "dislike", 2), load("undislikes", "undislike", 3)])
st = (ev.sort(["uid", "item_id", "timestamp", "pr"])
        .group_by(["uid", "item_id"]).agg(pl.col("ev").last().alias("state"),
                                          pl.col("timestamp").last().alias("st_ts")))
lk_ts = (ev.filter(pl.col("ev") == "like").group_by(["uid", "item_id"])
           .agg(pl.col("timestamp").max().alias("like_ts")))
print("пар с событиями like/dislike в train: %d; итоговое состояние:" % st.height,
      dict(st.group_by("state").len().sort("state").iter_rows()))

p = (p.join(st, on=["uid", "item_id"], how="left")
       .join(lk_ts, on=["uid", "item_id"], how="left")
       .with_columns((pl.col("state") == "like").fill_null(False).alias("like"),
                     (pl.col("state") == "dislike").fill_null(False).alias("dis")))
TOT = p.height
print("пар с итоговым лайком: %d (%.3f %% всех пар); с итоговым дизлайком: %d (%.3f %%)"
      % (p["like"].sum(), 100*p["like"].mean(), p["dis"].sum(), 100*p["dis"].mean()))
lp = p.filter(pl.col("like"))
print("из лайкнутых пар: лайк позже первого прослушивания у %d (%.1f %%), "
      "раньше или одновременно у %d (%.1f %%) — левая цензура, лайк пришёл из истории до окна"
      % ((lp["like_ts"] >= lp["t0"]).sum(), 100*(lp["like_ts"] >= lp["t0"]).mean(),
         (lp["like_ts"] < lp["t0"]).sum(), 100*(lp["like_ts"] < lp["t0"]).mean()))
# лайк, который точно поставлен после знакомства внутри окна
p = p.with_columns((pl.col("like") & (pl.col("like_ts") >= pl.col("t0"))).alias("like_after"))

BUCK = [(0,1,"0-1 (скип)"),(2,10,"2-10"),(11,25,"11-25"),(26,49,"26-49"),(50,80,"50-80"),
        (81,98,"81-98"),(99,99,"99"),(100,100,"100 (дослушан)")]
def table(df, title, note=""):
    print("\n=== %s ===%s" % (title, note))
    print("  played_ratio 1-го    пар        доля пар   P(повтор)  P(n>=5)  E[n]   P(like)  P(like после 1-го)  P(dislike)")
    for lo, hi, lab in BUCK:
        b = df.filter((pl.col("r0") >= lo) & (pl.col("r0") <= hi))
        if not b.height: continue
        print("  %-18s %9d  %7.2f %%  %8.2f %% %7.2f %% %5.2f  %7.3f %%  %13.3f %%  %9.3f %%"
              % (lab, b.height, 100*b.height/df.height, 100*(b["n"] >= 2).mean(),
                 100*(b["n"] >= 5).mean(), b["n"].mean(), 100*b["like"].mean(),
                 100*b["like_after"].mean(), 100*b["dis"].mean()))
    print("  (база по всем парам: P(like) %.3f %%, P(dislike) %.3f %%)"
          % (100*df["like"].mean(), 100*df["dis"].mean()))
    for cond, lab in [((pl.col("r0") <= 1), "итого скип (<=1)"),
                      ((pl.col("r0") >= 2) & (pl.col("r0") <= 49), "итого 2-49"),
                      ((pl.col("r0") >= 50) & (pl.col("r0") <= 98), "итого 50-98"),
                      ((pl.col("r0") >= 99), "итого 99-100"),
                      ((pl.col("r0") >= 50), "итого >=50"),
                      ((pl.col("r0") == 100), "итого ==100")]:
        b = df.filter(cond)
        print("  %-18s %9d  %7.2f %%  %8.2f %% %7.2f %% %5.2f  %7.3f %%  %13.3f %%  %9.3f %%"
              % (lab, b.height, 100*b.height/df.height, 100*(b["n"] >= 2).mean(),
                 100*(b["n"] >= 5).mean(), b["n"].mean(), 100*b["like"].mean(),
                 100*b["like_after"].mean(), 100*b["dis"].mean()))

table(p, "влияние первого прослушивания — все пары train")
CUT = TRAIN_END - 180 * DAY
table(p.filter(pl.col("t0") < CUT),
      "то же, только пары, у которых первое прослушивание раньше дня %d" % (TRAIN_DAYS - 180),
      "\n  (контроль: у поздних пар физически меньше времени на повторы и лайк)")

print("\n=== P(like | число прослушиваний пары) ===")
print("   n          пар        P(like)     P(like после 1-го)   P(dislike)   прирост P(like)   на один повтор, п.п.")
prev = prevn = None
for lo, hi in [(1,1),(2,2),(3,3),(4,4),(5,5),(6,7),(8,10),(11,15),(16,20),(21,30),
               (31,50),(51,100),(101,10**9)]:
    b = p.filter((pl.col("n") >= lo) & (pl.col("n") <= hi))
    if not b.height: continue
    lab = "%d" % lo if lo == hi else "%d-%d" % (lo, min(hi, 10**8))
    cur, curn = 100*b["like"].mean(), b["n"].mean()
    print("  %-8s %10d  %9.3f %%  %13.3f %%  %10.3f %%   %13s   %18s"
          % (lab, b.height, cur, 100*b["like_after"].mean(), 100*b["dis"].mean(),
             "—" if prev is None else "%+.2f" % (cur - prev),
             "—" if prev is None else "%+.3f" % ((cur - prev) / (curn - prevn))))
    prev, prevn = cur, curn

print("\n=== совместно: P(like) по (первое прослушивание x число повторов) ===")
print("  строки — played_ratio первого прослушивания, столбцы — n; в скобках число пар")
cols = [(1,1),(2,3),(4,10),(11,10**9)]
print("  %-18s" % "" + "".join("%22s" % ("n=%d" % lo if lo == hi else "n=%d-%s" % (lo, "∞" if hi > 10**8 else hi)) for lo, hi in cols))
for lo, hi, lab in [(0,1,"0-1 (скип)"),(2,49,"2-49"),(50,98,"50-98"),(99,100,"99-100")]:
    row = p.filter((pl.col("r0") >= lo) & (pl.col("r0") <= hi))
    cells = []
    for a, bb in cols:
        c = row.filter((pl.col("n") >= a) & (pl.col("n") <= bb))
        cells.append("%9.3f %% (%8d)" % (100*c["like"].mean() if c.height else 0, c.height))
    print("  %-18s" % lab + "".join("%22s" % x for x in cells))

print("\n=== для сравнения: завершаемость первого прослушивания и органика ===")
for v, lab in [(0, "первое из рекомендаций"), (1, "первое органическое")]:
    b = p.filter(pl.col("o0") == v)
    print("  %-23s пар %9d: средний r0 %.2f, скипов %.2f %%, r0==100 %.2f %%, "
          "P(повтор) %.2f %%, P(like) %.3f %%"
          % (lab, b.height, b["r0"].mean(), 100*(b["r0"] <= 1).mean(),
             100*(b["r0"] == 100).mean(), 100*(b["n"] >= 2).mean(), 100*b["like"].mean()))
