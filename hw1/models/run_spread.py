"""Разброс между запусками: тот же iALS и та же модель 5, другое случайное зерно.

Нужен, чтобы знать, какие различия между моделями вообще можно обсуждать. Сравнения в
отчёте опираются на разрывы в несколько раз больше этого разброса.

    python -I run_spread.py
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load, evaluate, report
from run_ials import fit_ials
from target import VARIANTS, build

# те же значения, что в run_model5.py (импортировать его нельзя: он считает при импорте)
FACTORS, REG, ALPHA, CHOSEN = 128, 0.05, 1.0, "B"

d = load("test")
print(d, "\n")
for seed in (42, 1, 7):
    report("iALS seed=%d" % seed,
           evaluate(d, fit_ials(d, factors=FACTORS, reg=REG, alpha=ALPHA, seed=seed)))

cfg = VARIANTS[CHOSEN]
ui, ii, w, _ = build(d, cap=cfg["cap"], like_bonus=cfg["like_bonus"])
for seed in (42, 1, 7):
    report("Модель 5 seed=%d" % seed,
           evaluate(d, fit_ials(d, weights_coo=(ui, ii, w), factors=FACTORS, reg=REG,
                                alpha=ALPHA, seed=seed)))
