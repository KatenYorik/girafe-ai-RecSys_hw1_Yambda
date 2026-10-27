import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
LF = pl.scan_parquet(os.path.join(raw,"listens.parquet"))
# ВНИМАНИЕ (поправка блока 02): DAY = 17280 — это НЕ сутки, а 1/5 суток (4 ч 48 мин).
# Единица timestamp равна 1 секунде, реальные сутки = 86400 единиц (см. t1_time_unit.py).
# Константа оставлена как есть, чтобы числа блока воспроизводились: любой горизонт,
# который этот скрипт печатает «в днях», читается как (дни / 5) реальных суток.
DAY, LAST, L = 17280, 1503, 90
ts0 = (LAST+1-L)*DAY
pairs = (LF.select("uid","item_id","timestamp").group_by(["uid","item_id"])
           .agg(pl.col("timestamp").min().alias("t0")).collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("uf"))
trainu = ufirst.filter(pl.col("uf") < ts0)["uid"]
print("train-пользователей (первое прослушивание до начала окна): %d из %d"
      % (len(trainu), ufirst.height))

ev = (LF.filter((pl.col("timestamp")>=ts0) & (pl.col("timestamp")<(LAST+1)*DAY))
        .filter(pl.col("played_ratio_pct")>=50)
        .select("uid","item_id","timestamp").collect(engine="streaming")
        .join(pairs, on=["uid","item_id"], how="left"))
new = (ev.filter(pl.col("t0")>=ts0)
         .filter(pl.col("uid").is_in(trainu.implode()))
         .with_columns(((pl.col("timestamp")-ts0)//DAY).alias("off")))
print("релевантных прослушиваний новых треков в окне 90 дней: %d" % new.height)

print("\n=== доля релевантных новинок по 30-дневным третям окна ===")
th = new.with_columns((pl.col("off")//30).alias("t")).group_by("t").len().sort("t")
tot = new.height
for t,n in th.iter_rows():
    print("   дни %2d-%2d: %7d событий  (%4.1f%%)" % (t*30+1, t*30+30, n, 100*n/tot))

print("\n=== когда у пользователя появляется ПЕРВАЯ релевантная новинка ===")
f = new.group_by("uid").agg(pl.col("off").min().alias("first"))
base = len(trainu)
for lim in [7,14,30,60,90]:
    k = (f["first"] < lim).sum()
    print("   к дню %2d: %d пользователей (%.1f%% от train-юзеров)" % (lim, k, 100*k/base))
print("   только после дня 30: %d пользователей (%.1f%%)"
      % ((f["first"]>=30).sum(), 100*(f["first"]>=30).sum()/base))
print("   только после дня 60: %d пользователей (%.1f%%)"
      % ((f["first"]>=60).sum(), 100*(f["first"]>=60).sum()/base))

print("\n=== сколько train теряет ===")
n_all = LF.select(pl.len()).collect().item()
for L2 in [30,60,90]:
    lost = LF.filter(pl.col("timestamp")>=(LAST+1-L2)*DAY).select(pl.len()).collect().item()
    print("   окно %2d дней: train теряет %d прослушиваний (%.1f%%)" % (L2, lost, 100*lost/n_all))

print("\n=== 'новизна' внутри окна размывается? ===")
k30 = (new.filter(pl.col("off")<30).select(["uid","item_id"]).unique())
late = (new.filter(pl.col("off")>=30).select(["uid","item_id"]).unique())
ov = late.join(k30, on=["uid","item_id"], how="semi").height
print("   пар, релевантных и в первые 30 дней, и позже: %d из %d поздних пар (%.1f%%)"
      % (ov, late.height, 100*ov/late.height))
print("   -> столько 'новинок' во второй половине окна уже знакомы пользователю по его же тесту")
