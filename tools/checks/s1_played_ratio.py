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
LF = (pl.scan_parquet(os.path.join(raw, "listens.parquet"))
        .filter(pl.col("timestamp") < TRAIN_END))
# блок 01: удаляем только полные дубли; повторы с разным played_ratio_pct — разные прослушивания
KEY = ["uid", "item_id", "timestamp", "is_organic", "played_ratio_pct"]
h = (LF.select(KEY).unique(subset=KEY)
       .group_by("played_ratio_pct").agg(pl.len().alias("n"))
       .sort("played_ratio_pct").collect(engine="streaming"))
raw_n = LF.select(pl.len()).collect().item()
tot = h["n"].sum()
print("прослушиваний в train (дни 0..%d): %d строк, %d после удаления полных дублей (-%.2f%%)"
      % (TRAIN_DAYS - 1, raw_n, tot, 100 * (raw_n - tot) / raw_n))
d = dict(h.iter_rows())
cum, c = {}, 0
for v in range(0, 160):
    c += d.get(v, 0); cum[v] = c

print("\n--- мелкий шаг в зоне скипов (0..20 %) ---")
print(" %  доля, %   накопл., %   событий")
for v in range(0, 21):
    print("%3d  %7.3f   %8.3f   %9d" % (v, 100*d.get(v,0)/tot, 100*cum[v]/tot, d.get(v,0)))
print("\n--- середина, шаг 5 % ---")
for a in range(20, 90, 5):
    s = sum(d.get(v,0) for v in range(a, a+5))
    print("%3d-%3d  %6.3f %%  накопл. %6.2f %%  событий %9d" % (a, a+4, 100*s/tot, 100*cum[a+4]/tot, s))
print("\n--- мелкий шаг у потолка (90..100 %) и перемотка ---")
for v in range(90, 101):
    print("%3d  %7.3f   %8.3f   %9d" % (v, 100*d.get(v,0)/tot, 100*cum[v]/tot, d.get(v,0)))
over = sum(d.get(v,0) for v in range(101, 160))
print("> 100  %7.3f %%   событий %9d" % (100*over/tot, over))

print("\n--- где «провал»: сравнение соседних процентов в зоне 0..15 ---")
for v in range(1, 16):
    prev = d.get(v-1, 0); cur = d.get(v, 0)
    print("  %2d -> %2d: %9d -> %9d  (x%.2f)" % (v-1, v, prev, cur, cur/max(prev,1)))
mid = min(range(2, 99), key=lambda v: d.get(v, 0))
print("минимум плотности на [2;98]: %d %% (%d событий, %.4f %%)"
      % (mid, d.get(mid,0), 100*d.get(mid,0)/tot))
print("плотность на процент по зонам:")
for a, b in [(2,5),(5,10),(10,20),(20,40),(40,60),(60,80),(80,95),(95,99)]:
    print("    %2d..%2d %%: %8.0f событий на процент" % (a, b, sum(d.get(v,0) for v in range(a,b))/(b-a)))
print("    у 0 %%: %d, у 1 %%: %d, у 99 %%: %d, у 100 %%: %d" % (d.get(0,0), d.get(1,0), d.get(99,0), d.get(100,0)))

print("\n--- пороги ---")
for thr in [1, 2, 3, 5, 10, 15, 20, 25, 30, 50, 80, 90, 95, 99, 100]:
    print("  <= %3d %%: %6.2f %% событий   | >= %3d %%: %6.2f %%"
          % (thr, 100*cum[thr]/tot, thr, 100*(tot-cum[thr-1] if thr else tot)/tot))
s = h.select(pl.col("played_ratio_pct").repeat_by("n").list.explode(empty_as_null=True))["played_ratio_pct"]
print("\nквантили: q10=%d q25=%d median=%d q75=%d q90=%d  mean=%.2f"
      % (s.quantile(.1), s.quantile(.25), s.median(), s.quantile(.75), s.quantile(.9), s.mean()))

print("")
print("--- та же зона скипов в секундах (played_ratio_pct * length / 100) ---")
sec = (LF.filter(pl.col("played_ratio_pct") <= 25).select(KEY + ["track_length_seconds"])
         .unique(subset=KEY)
         .select(((pl.col("played_ratio_pct") * pl.col("track_length_seconds")) // 100)
                 .alias("sec"))
         .group_by("sec").agg(pl.len().alias("n")).sort("sec").collect(engine="streaming"))
ds = dict(sec.iter_rows()); st = sec["n"].sum()
print("прослушиваний с ratio <= 25 %%: %d (%.2f %% всех train-прослушиваний)" % (st, 100*st/tot))
cs, c = {}, 0
for v in range(0, 3000):
    c += ds.get(v, 0); cs[v] = c
print("  сек   доля зоны, %   накопл. зоны, %   событий")
for v in list(range(0, 21)) + [25, 30, 35, 40, 45, 50, 60]:
    print("  %3d   %8.3f      %9.2f      %9d" % (v, 100*ds.get(v,0)/st, 100*cs[v]/st, ds.get(v,0)))
print("  доля зоны с 0 сек: %.2f %%; <= 5 сек: %.2f %%; <= 10 сек: %.2f %%; <= 30 сек: %.2f %%"
      % (100*cs[0]/st, 100*cs[5]/st, 100*cs[10]/st, 100*cs[30]/st))
print("  в пересчёте на все прослушивания train: <= 5 сек %.2f %%, <= 10 сек %.2f %%, <= 30 сек %.2f %%"
      % (100*cs[5]/tot, 100*cs[10]/tot, 100*cs[30]/tot))
