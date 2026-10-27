"""Асимметрия «знакомый трек» против «релевантный трек».

Релевантность на тесте требует played_ratio_pct >= 50, а «знаком» — любого
прослушивания, даже скипа. Скрипт измеряет, сколько «знакомых» треков человек
фактически не слышал, и сколько целевых пар мы из-за этого не засчитываем.
Решение от 27.10.2026 — оставить как есть, см. LIMITATIONS.md.

    python -I tools/checks/f7_known_asymmetry.py data/subsample
"""
import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os

d = sys.argv[1]
tr = pl.read_parquet(os.path.join(d, "train.parquet"))
va = pl.read_parquet(os.path.join(d, "val.parquet"))
te = pl.read_parquet(os.path.join(d, "test.parquet"))

hp = (pl.concat([tr, va]).select("uid", "item_id", "played_ratio_pct")
        .group_by(["uid", "item_id"])
        .agg(pl.col("played_ratio_pct").max().alias("best")))
print("пар в истории до теста: %d" % hp.height)
LEVELS = [("ни разу не больше 1 % (чистый скип)", 1), ("ни разу не больше 10 %", 10)]
for lab, thr in LEVELS:
    g = hp.filter(pl.col("best") <= thr)
    print("   %-38s %7d пар (%5.1f%%)" % (lab, g.height, 100 * g.height / hp.height))
g = hp.filter(pl.col("best") < 50)
print("   %-38s %7d пар (%5.1f%%)" % ("ни разу не дотянули до 50 %", g.height,
                                      100 * g.height / hp.height))

rel = te.filter(pl.col("is_relevant")).select("uid", "item_id").unique()
newrel = te.filter(pl.col("is_relevant") & pl.col("is_new")).select("uid", "item_id").unique()
print("\nрелевантных пар в тесте: %d, из них новых (цель метрики): %d"
      % (rel.height, newrel.height))
for lab, cond in [("чистый скип (<=1 %)", pl.col("best") <= 1),
                  ("<=10 %", pl.col("best") <= 10),
                  ("<50 %", pl.col("best") < 50)]:
    k = rel.join(hp.filter(cond).select("uid", "item_id"), on=["uid", "item_id"],
                 how="semi").height
    print("   в истории %-20s, в тесте дослушали >=50 %%: %6d (%.1f%% цели)"
          % (lab, k, 100 * k / newrel.height))
