"""Модель 5: iALS на своём таргете «понравилось при знакомстве».

Форма формулы и множители качества — из блока 05, см. target.py.
Коэффициенты: вариант B (C = 20, beta = 15), решение от 27.10.2026.
Запуск без аргументов считает все три варианта на val; с аргументом `test` —
только принятый вариант B на тесте.
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, evaluate, report
from target import VARIANTS, build
from run_ials import fit_ials

CHOSEN = "B"          # решение от 27.10.2026

# Единый протокол сравнения с базовым iALS: те же factors, reg, alpha, что выбрала
# сетка в run_ials.py. Отличается только матрица, на которой модель учится.
FACTORS, REG, ALPHA = 128, 0.05, 1.0

if len(sys.argv) > 1 and sys.argv[1] == "test":
    d = load("test")
    print(d, "\n")
    cfg = VARIANTS[CHOSEN]
    ui, ii, w, stats = build(d, cap=cfg["cap"], like_bonus=cfg["like_bonus"])
    print("вариант %s: cap=%d, beta=%g; пар %d, лайков при знакомстве %d, "
          "доля массы от лайков %.1f%%, верхние 2,5%% пар держат %.1f%%"
          % (CHOSEN, cfg["cap"], cfg["like_bonus"], stats["pairs"], stats["likes"],
             100 * stats["mass_from_likes"], 100 * stats["mass_top"]))
    sc = fit_ials(d, weights_coo=(ui, ii, w), factors=FACTORS, reg=REG, alpha=ALPHA)
    report("Модель 5 (вариант %s)" % CHOSEN, evaluate(d, sc))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores_model5.npy"), sc)
    sys.exit(0)

d = load("val")
print(d, "\n")
for key, cfg in VARIANTS.items():
    ui, ii, w, stats = build(d, **{k: v for k, v in cfg.items() if k != "label"})
    print("вариант %s (%s): cap=%d, like_bonus=%g" % (key, cfg["label"], cfg["cap"], cfg["like_bonus"]))
    print("   пар %d, лайков при знакомстве %d, доля массы таргета от лайков %.1f%%, "
          "верхние 2,5%% пар держат %.1f%% массы, вес медиана %.2f максимум %.0f"
          % (stats["pairs"], stats["likes"], 100*stats["mass_from_likes"],
             100*stats["mass_top"], stats["w_median"], stats["w_max"]))
    sc = fit_ials(d, weights_coo=(ui, ii, w), factors=FACTORS, reg=REG, alpha=ALPHA)
    report("   Модель 5 вариант " + key, evaluate(d, sc))
    print()
