import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# Блок 02. Сезонность, провалы и скачки — в РЕАЛЬНЫХ сутках.
# Единица timestamp = 1 с (доказано в t1_time_unit.py), сутки = 86400 единиц.
DAY = 86400
OLD = 17280          # «сутки» блока 01 — на самом деле 1/5 суток (4 ч 48 мин)
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))

day = (LF.select((pl.col("timestamp") // DAY).alias("d"))
         .group_by("d").len().sort("d").collect(engine="streaming"))
TMAX = LF.select(pl.col("timestamp").max()).collect().item()
LAST = day["d"].max()
print("=== поток по реальным суткам ===")
print("  суток покрыто: %d (d=%d..%d), медиана %.0f прослушиваний/сутки"
      % (day.height, day["d"].min(), LAST, day["len"].median()))
print("  последние сутки %d заполнены на %.1f %% (timestamp max = %d)"
      % (LAST, 100 * (TMAX - LAST * DAY) / DAY, TMAX))
rows = list(day.iter_rows())
print("  первые 5 суток: %s" % rows[:5])
print("  последние 5 суток: %s" % rows[-5:])
print("  минимум: сутки %d (%d), максимум: сутки %d (%d)"
      % (*rows[day["len"].arg_min()], *rows[day["len"].arg_max()]))
# рост потока: средние по десятидневкам в начале и в конце
h1 = day.filter(pl.col("d") < 30)["len"].mean()
h2 = day.filter((pl.col("d") >= 270) & (pl.col("d") <= 299))["len"].mean()
print("  первые 30 суток в среднем %.0f, сутки 270–299 — %.0f: рост в %.2f раза за 10 месяцев"
      % (h1, h2, h2 / h1))

# --- внутрисуточный профиль: он и объясняет «провалы» блока 01 ---
print("\n=== внутрисуточный профиль (24 часовые корзины, вся история) ===")
hr = (LF.select(((pl.col("timestamp") % DAY) // 3600).alias("h"))
        .group_by("h").len().sort("h").collect(engine="streaming"))
tot = hr["len"].sum()
print("  " + " ".join("%02d:%.2f%%" % (h, 100 * c / tot) for h, c in hr.iter_rows()))
print("  минимум час %d (%.2f %% потока), максимум час %d (%.2f %%), max/min = %.2f"
      % (hr["h"][hr["len"].arg_min()], 100 * hr["len"].min() / tot,
         hr["h"][hr["len"].arg_max()], 100 * hr["len"].max() / tot,
         hr["len"].max() / hr["len"].min()))
print("  то есть timestamp = 0 приходится на ночной минимум — сетка суток "
      "совпадает с календарной")
print("  пятёрки по 4 ч 48 мин (это и есть «сутки» блока 01), доля потока:")
seg = (LF.select(((pl.col("timestamp") % DAY) // OLD).alias("s"))
         .group_by("s").len().sort("s").collect(engine="streaming"))
for s, c in seg.iter_rows():
    print("    сегмент %d (%02d:%02d–%02d:%02d): %.2f %% потока — «день d %% 5 = %d» блока 01"
          % (s, s * 288 // 60, s * 288 % 60, (s + 1) * 288 // 60, (s + 1) * 288 % 60,
             100 * c / seg["len"].sum(), s))
print("  max/min между сегментами = %.2f — ровно та «пятидневная сезонность», "
      "что видна на сетке 17280" % (seg["len"].max() / seg["len"].min()))

# --- недельная сезонность ---
# абсолютной привязки к календарю нет: сутки 0 считаем нулевым днём недели
full = day.filter((pl.col("d") >= 1) & (pl.col("d") < LAST))   # без неполных краёв
print("\n=== недельная сезонность (день недели = d %% 7, сутки 1..%d) ===" % (LAST - 1))
dow = (full.with_columns((pl.col("d") % 7).alias("w"))
           .group_by("w").agg(pl.col("len").mean().alias("m"),
                              pl.col("len").median().alias("md"),
                              pl.len().alias("k")).sort("w"))
base = dow["m"].mean()
for w, m, md, k in dow.iter_rows():
    print("  dow %d: среднее %7.0f (%+5.1f%% к общему), медиана %7.0f, суток %d"
          % (w, m, 100 * (m / base - 1), md, k))
print("  амплитуда сырая: max/min = %.3f, разброс %.1f п. п."
      % (dow["m"].max() / dow["m"].min(),
         100 * (dow["m"].max() - dow["m"].min()) / base))
det = (full.with_columns(pl.col("len").rolling_median(7, center=True, min_samples=4)
                           .alias("m7"))
           .with_columns((pl.col("len") / pl.col("m7")).alias("r"),
                         (pl.col("d") % 7).alias("w")))
dd = det.group_by("w").agg(pl.col("r").mean().alias("m"),
                           pl.col("r").std().alias("s")).sort("w")
print("  детренд (к скользящей медиане 7 суток) — тренд роста убран:")
for w, m, s in dd.iter_rows():
    print("    dow %d: %.4f (%+5.1f%%), разброс по неделям sd = %.3f" % (w, m, 100 * (m - 1), s))
print("  амплитуда детренд: max/min = %.4f, разброс %.1f п. п."
      % (dd["m"].max() / dd["m"].min(), 100 * (dd["m"].max() - dd["m"].min())))
print("  устойчивость: самый высокий dow = %d, самый низкий = %d "
      "(на сырых средних %d и %d)"
      % (dd["w"][dd["m"].arg_max()], dd["w"][dd["m"].arg_min()],
         dow["w"][dow["m"].arg_max()], dow["w"][dow["m"].arg_min()]))

# --- фаза недели на границах окон-кандидатов ---
print("\n=== фаза недели на границах (сутки 0..300, dow = d %% 7) ===")
CAND = [("test 18", 283, 18), ("test 21", 280, 21), ("test 30", 271, 30),
        ("val 18", 265, 18), ("val 21", 259, 21), ("val 30", 241, 30),
        ("train при test/val 21", 0, 259)]
for lab, d0, L in CAND:
    sub = full.filter((pl.col("d") >= d0) & (pl.col("d") < d0 + L))
    c = (sub.with_columns((pl.col("d") % 7).alias("w"))
            .group_by("w").agg(pl.len().alias("k"), pl.col("len").sum().alias("s"))
            .sort("w"))
    tot, nd = c["s"].sum(), c["k"].sum()
    even = sum(c["s"][i] / c["k"][i] for i in range(c.height)) / 7 * nd if c.height == 7 else tot
    print("  %-22s сутки %3d..%3d (%3d): старт dow %d, конец dow %d, %s 7; "
          "дней каждого dow %s; смещение объёма %+.2f %%"
          % (lab, d0, d0 + L - 1, L, d0 % 7, (d0 + L - 1) % 7,
             "кратно" if L % 7 == 0 else "НЕ кратно", c["k"].to_list(),
             100 * (tot / even - 1)))
print("  принятая граница блока 06 (1414 * 17280 = %d) = сутки %.2f, "
      "то есть разрез внутри суток в %02d:%02d, dow %d"
      % (1414 * OLD, 1414 * OLD / DAY, (1414 * OLD % DAY) // 3600,
         (1414 * OLD % DAY) % 3600 // 60, (1414 * OLD // DAY) % 7))

# --- провалы и скачки ---
print("\n=== провалы и скачки (скользящая медиана 15 суток) ===")
roll = (day.with_columns(pl.col("len").rolling_median(15, center=True, min_samples=8)
                           .alias("m15"))
           .with_columns((pl.col("len") / pl.col("m15") - 1).alias("dev")))
for thr in [.10, .15, .20, .30]:
    k = (roll["dev"].abs() > thr).sum()
    print("  суток с |откл| > %2d %%: %d из %d (%.1f %%)"
          % (100 * thr, k, roll.height, 100 * k / roll.height))
ext = roll.sort("dev")
print("  пять самых глубоких провалов:")
for d, c, m, dv in list(ext.iter_rows())[:5]:
    print("    сутки %3d: %7d против медианы %7.0f (%+.1f %%)" % (d, c, m, 100 * dv))
print("  пять самых высоких скачков:")
for d, c, m, dv in list(ext.iter_rows())[-5:]:
    print("    сутки %3d: %7d против медианы %7.0f (%+.1f %%)" % (d, c, m, 100 * dv))
# отдельно: сколько из отклонений объясняется днём недели
print("  для сравнения, амплитуда недельной сезонности даёт отклонения до %.1f %%"
      % (100 * max(abs(dd["m"].max() - 1), abs(dd["m"].min() - 1))))

# --- «день 195» блока 01 ---
print("\n=== что было «днём 195» блока 01 ===")
d195 = 195 * OLD
print("  195 * 17280 = %d единиц = реальные сутки %.2f, сегмент %d (%s)"
      % (d195, d195 / DAY, (d195 % DAY) // OLD, "ночь 00:00–04:48"))
print("  реальные сутки 39: %d прослушиваний (медиана по суткам %.0f, откл %+.1f %%)"
      % (day.filter(pl.col("d") == 39)["len"][0], day["len"].median(),
         100 * roll.filter(pl.col("d") == 39)["dev"][0]))
print("  соседи: " + ", ".join(
    "сутки %d=%d" % (d, day.filter(pl.col("d") == d)["len"][0]) for d in range(36, 44)))
print("  то есть одиночного провала нет: это ночной сегмент обычных суток")
