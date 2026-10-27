import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# Блок 02. Единица timestamp: проверка, а не допущение.
# Блок 01 принял «единица = 5 с, сутки = 17280 единиц». Этот скрипт проверяет
# допущение тремя независимыми способами.
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))
ts = LF.select("timestamp").collect(engine="streaming")["timestamp"].cast(pl.Int64)
print("прослушиваний %d, timestamp от %d до %d" % (len(ts), ts.min(), ts.max()))

# --- тест 1: сетка значений ---
print("\n=== тест 1: на какую сетку ложатся значения ===")
for m in [2, 3, 5, 10, 15, 60]:
    k = (ts % m == 0).sum()
    print("  кратны %2d: %d из %d (%.2f %%)" % (m, k, len(ts), 100 * k / len(ts)))
print("  различных значений timestamp: %d" % ts.n_unique())

# --- тест 2: суточный период по «пикастости» профиля ---
print("\n=== тест 2: суточный период (CV профиля из 24 корзин) ===")
res = []
for P, lab in [(17280, "гипотеза блока 01: 5 с/единицу"),
               (86400, "гипотеза: 1 с/единицу"),
               (21600, "—"), (28800, "—"), (43200, "полсуток при 1 с"),
               (57600, "—"), (64800, "—"), (100800, "—"), (129600, "—"),
               (172800, "двое суток при 1 с"), (604800, "неделя при 1 с")]:
    cnt = ((ts % P) * 24 // P).value_counts()["count"].cast(pl.Float64)
    res.append((cnt.std() / cnt.mean(), P, lab))
for cv, P, lab in sorted(res, reverse=True):
    print("  P=%7d  CV=%.4f  %s" % (P, cv, lab))
for P in [86400, 17280]:
    c = ((ts % P) * 24 // P).value_counts().sort("timestamp")
    print("  профиль по 1/24 периода P=%d:\n    %s"
          % (P, " ".join("%d" % v for v in c["count"].to_list())))

# --- тест 3: решающий — интервал между соседними прослушиваниями ---
# трек играет track_length_seconds * played_ratio/100 секунд; если единица
# timestamp = 1 с, интервал до следующего прослушивания равен этому числу,
# если 5 с — впятеро меньше
print("\n=== тест 3: интервал между соседними прослушиваниями против длины трека ===")
g = (LF.select("uid", "timestamp", "track_length_seconds", "played_ratio_pct")
       .sort("uid", "timestamp")
       .with_columns(pl.col("timestamp").diff().alias("gap"),
                     pl.col("uid").diff().alias("du"))
       .filter((pl.col("du") == 0) & (pl.col("gap") > 0))
       .select("gap", "track_length_seconds", "played_ratio_pct")
       .collect(engine="streaming"))
print("  пар «подряд у одного пользователя»: %d" % g.height)
print("  интервал, единиц: " + " ".join(
    "q%02d=%d" % (int(q * 100), g["gap"].quantile(q)) for q in [.1, .25, .5, .75, .9]))
played = (g["track_length_seconds"] * g["played_ratio_pct"].clip(0, 100) / 100)
print("  медиана длины трека %d с, медиана фактически проигранных %.0f с"
      % (g["track_length_seconds"].median(), played.median()))
sub = g.with_columns(played.alias("sec")).filter(pl.col("gap") < 20000)
print("  медиана (интервал / проигранные секунды) = %.3f "
      "(1.0 → единица 1 с, 0.2 → единица 5 с)"
      % (sub["gap"] / sub["sec"].clip(1, None)).median())
for lo, hi in [(100, 150), (150, 250), (250, 400), (400, 600)]:
    s = g.filter((pl.col("track_length_seconds") >= lo)
                 & (pl.col("track_length_seconds") < hi)
                 & (pl.col("played_ratio_pct") >= 98))
    print("  дослушанные треки %3d–%3d с: медиана интервала %4d единиц "
          "(медиана длины %3d с, пар %d)"
          % (lo, hi, s["gap"].median(), s["track_length_seconds"].median(), s.height))

# --- что из этого следует ---
D = 86400
print("\n=== следствие: реальный горизонт данных ===")
print("  при единице 1 с: %d с = %.2f суток = %.1f месяцев"
      % (ts.max(), ts.max() / D, ts.max() / D / 30.44))
print("  при единице 5 с (блок 01): %.2f суток" % (ts.max() * 5 / D))
print("  принятая граница train/test (1414 * 17280 = %d) при единице 1 с = сутки %.2f"
      % (1414 * 17280, 1414 * 17280 / D))
print("  принятое тестовое окно (90 * 17280 = %d единиц) = %.2f реальных суток"
      % (90 * 17280, 90 * 17280 / D))
for L in [7, 14, 30, 60, 90]:
    print("    окно блока 06 «%2d дней» = %.1f реальных суток" % (L, L * 17280 / D))
