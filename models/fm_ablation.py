"""Абляция FM по срезу «новый артист»: отвечает ли новизна выдачи за аудио.

Отчёт честно оговаривал, что абляция «с аудио / без аудио» мерилась только по общей
метрике, поэтому сдвиг FM к незнакомым исполнителям нельзя было отнести именно к аудио:
тот же сдвиг мог давать эмбеддинг артиста, который есть в обеих ветвях. Этот скрипт
закрывает вопрос — считает для обеих ветвей те же величины, что и run_analysis.py
(доля новых артистов в топ-10, концентрация на популярном, Recall@10 на срезе
«новый артист»), плюс реакцию на штраф mu.

Требует двух файлов оценок на test:
    python -I fm_one.py test audio      -> scores_fm.npy
    python -I fm_one.py test noaudio    -> scores_fm_noaudio.npy

    python -I fm_ablation.py
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load
from run_analysis import known_artists, novelty_report, standardize

HERE = os.path.dirname(os.path.abspath(__file__))

d = load("test")
print(d)
art, n_art = d.artists()
ka = known_artists(d, art, n_art)

branches = [("FM с аудио", "scores_fm.npy"), ("FM без аудио", "scores_fm_noaudio.npy")]
scores = {}
for name, f in branches:
    p = os.path.join(HERE, f)
    if not os.path.exists(p):
        sys.exit("нет файла оценок %s — сначала прогоните fm_one.py test" % f)
    scores[name] = np.load(p)

print("\n=== без реранжирования ===")
for name in scores:
    novelty_report(d, scores[name], art, ka, name)

print("\n=== со штрафом за знакомого артиста ===")
for name, s in scores.items():
    z = standardize(s, d)
    for mu in (0.25, 1.0):
        novelty_report(d, z, art, ka, name, mu=mu)
    print()
