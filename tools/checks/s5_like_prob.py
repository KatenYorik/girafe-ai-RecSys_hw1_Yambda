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
LF = (pl.scan_parquet(os.path.join(raw, "listens.parquet"))
        .filter(pl.col("timestamp") < TRAIN_END)
        .select(KEY + ["track_length_seconds"]).unique(subset=KEY)
        .with_columns(pl.min_horizontal(pl.col("played_ratio_pct"), pl.lit(100))
                        .cast(pl.Int32).alias("r")))
FST = pl.col("timestamp") == pl.col("timestamp").min()
p = (LF.group_by(["uid", "item_id"]).agg(
        pl.len().alias("n"),
        pl.col("timestamp").min().alias("t0"),
        pl.col("r").filter(FST).max().alias("r0"),
        pl.col("is_organic").filter(FST).max().alias("o0"),
        pl.col("track_length_seconds").first().alias("len"),
        pl.col("r").max().alias("rmax"))
      .collect(engine="streaming"))

def load(f, ev, pr):
    return (pl.read_parquet(os.path.join(raw, f + ".parquet"))
              .select("uid", "item_id", "timestamp", "is_organic")
              .filter(pl.col("timestamp") < TRAIN_END)
              .unique(subset=["uid", "item_id", "timestamp", "is_organic"])
              .with_columns(pl.lit(ev).alias("ev"), pl.lit(pr).alias("pr")))
ev = pl.concat([load("likes", "like", 0), load("unlikes", "unlike", 1),
                load("dislikes", "dislike", 2), load("undislikes", "undislike", 3)])
st = (ev.sort(["uid", "item_id", "timestamp", "pr"])
        .group_by(["uid", "item_id"]).agg(pl.col("ev").last().alias("state")))
p = (p.join(st, on=["uid", "item_id"], how="left")
       .with_columns((pl.col("state") == "like").fill_null(False).alias("like"),
                     (pl.col("state") == "dislike").fill_null(False).alias("dis")))
print("пар в train %d; с итоговым лайком %d (%.3f %%)" % (p.height, p["like"].sum(), 100*p["like"].mean()))

print("\n=== P(like) по played_ratio первого прослушивания, мелкий шаг ===")
print("  r0      пар       P(like)    P(dislike)   E[n]")
STEPS = [(0,0),(1,1),(2,5),(6,10),(11,20),(21,30),(31,40),(41,50),(51,60),(61,70),
         (71,80),(81,90),(91,95),(96,97),(98,98),(99,99),(100,100)]
for lo, hi in STEPS:
    b = p.filter((pl.col("r0") >= lo) & (pl.col("r0") <= hi))
    lab = "%d" % lo if lo == hi else "%d-%d" % (lo, hi)
    print("  %-7s %9d  %8.3f %%  %8.3f %%  %5.2f"
          % (lab, b.height, 100*b["like"].mean(), 100*b["dis"].mean(), b["n"].mean()))

print("\n=== та же кривая при n == 1 (чистое знакомство, повторов не было) ===")
q = p.filter(pl.col("n") == 1)
print("  r0      пар       P(like)    P(dislike)")
for lo, hi in STEPS:
    b = q.filter((pl.col("r0") >= lo) & (pl.col("r0") <= hi))
    lab = "%d" % lo if lo == hi else "%d-%d" % (lo, hi)
    print("  %-7s %9d  %8.3f %%  %8.3f %%" % (lab, b.height, 100*b["like"].mean(), 100*b["dis"].mean()))

print("\n=== проверка аномалии у 99-100: контроль органики ===")
for v, lab in [(0, "рекомендации"), (1, "органика")]:
    s = p.filter(pl.col("o0") == v)
    print("  первое прослушивание — %s (%d пар):" % (lab, s.height))
    for lo, hi in [(0,1),(2,49),(50,80),(81,98),(99,99),(100,100)]:
        b = s.filter((pl.col("r0") >= lo) & (pl.col("r0") <= hi))
        print("     r0 %-7s пар %9d  P(like) %7.3f %%  P(dislike) %6.3f %%  E[n] %.2f"
              % ("%d-%d" % (lo, hi), b.height, 100*b["like"].mean(), 100*b["dis"].mean(), b["n"].mean()))

print("\n=== проверка аномалии у 99-100: контроль длины трека ===")
print("  длина, с        r0 0-1     r0 2-49    r0 50-80   r0 81-98   r0 99      r0 100     (P(like), в скобках пары)")
for lo, hi in [(0,90),(91,150),(151,210),(211,300),(301,600),(601,10**9)]:
    s = p.filter((pl.col("len") >= lo) & (pl.col("len") <= hi))
    cells = []
    for a, b2 in [(0,1),(2,49),(50,80),(81,98),(99,99),(100,100)]:
        c = s.filter((pl.col("r0") >= a) & (pl.col("r0") <= b2))
        cells.append("%5.2f%%(%6d)" % (100*c["like"].mean() if c.height else 0, c.height))
    print("  %-14s " % ("%d-%s" % (lo, "∞" if hi > 10**8 else hi)) + " ".join(cells))

print("\n=== доля первых прослушиваний с r0 99 / 100 по корзинам длины ===")
for lo, hi in [(0,90),(91,150),(151,210),(211,300),(301,600),(601,10**9)]:
    s = p.filter((pl.col("len") >= lo) & (pl.col("len") <= hi))
    print("  длина %-8s пар %9d: r0==100 %5.2f %%, r0==99 %5.2f %%, r0<=1 %5.2f %%"
          % ("%d-%s" % (lo, "∞" if hi > 10**8 else hi), s.height,
             100*(s["r0"] == 100).mean(), 100*(s["r0"] == 99).mean(), 100*(s["r0"] <= 1).mean()))

print("\n=== P(like) по максимальному played_ratio пары (для сравнения с первым) ===")
print("  rmax    пар       P(like)    P(dislike)")
for lo, hi in [(0,1),(2,49),(50,80),(81,98),(99,99),(100,100)]:
    b = p.filter((pl.col("rmax") >= lo) & (pl.col("rmax") <= hi))
    print("  %-7s %9d  %8.3f %%  %8.3f %%" % ("%d-%d" % (lo, hi), b.height,
          100*b["like"].mean(), 100*b["dis"].mean()))
