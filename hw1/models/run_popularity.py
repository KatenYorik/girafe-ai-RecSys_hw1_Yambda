"""Бейзлайн Popularity: трек тем выше, чем больше разных слушателей у него в обучающем окне."""
import sys, os
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, evaluate, report

for stage in ("val", "test"):
    d = load(stage)
    print(d)
    listeners = d.known.sum(0).astype(np.float32)          # разных слушателей на трек
    scores = np.broadcast_to(listeners, (d.n_users, d.n_items))
    report("Popularity", evaluate(d, scores))
    plays = d.counts().sum(0)                               # всего прослушиваний на трек
    report("Popularity (плейсы)", evaluate(d, np.broadcast_to(plays, (d.n_users, d.n_items))))
    rng = np.random.default_rng(42)
    report("Random", evaluate(d, rng.random((d.n_users, d.n_items), dtype=np.float32)))
    print()
