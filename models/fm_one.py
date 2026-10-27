"""Один прогон FM: стадия и состав признаков задаются аргументами.

    python -I fm_one.py val audio     # FM с аудио и артистом на val
    python -I fm_one.py val noaudio   # та же модель без аудио (абляция)
    python -I fm_one.py test audio    # финальный замер на test
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, evaluate, report
from run_fm import FM

stage, mode = sys.argv[1], sys.argv[2]
EPOCHS = 20
d = load(stage)
print(d)
m = FM(d, dim=64, lr=0.05, reg=0.01, use_audio=(mode == "audio")).fit(epochs=EPOCHS)
sc = m.scores()
name = "FM %s (%d эпох)" % ("аудио+артист" if mode == "audio" else "без аудио", EPOCHS)
report(name, evaluate(d, sc))
if stage == "test":
    # аудиоветвь — под именем scores_fm.npy, его подхватывает run_analysis.py;
    # ветвь без аудио нужна, чтобы померить срез «новый артист» у обеих (fm_ablation.py)
    f = "scores_fm.npy" if mode == "audio" else "scores_fm_noaudio.npy"
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), sc)
