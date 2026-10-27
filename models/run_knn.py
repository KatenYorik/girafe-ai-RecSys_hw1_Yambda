"""Модель 1: item-based коллаборативная фильтрация на соседях.

**Постановка.** Профиль пользователя — бинарный вектор прослушанного в обучающем окне
(«знаком» = любое прослушивание, как и у остальных моделей). Близость двух треков —
косинус по их столбцам в этой матрице, с настраиваемым показателем подавления
популярности $q$:

$$s_{ij} = \frac{|U_i \cap U_j|}{(n_i)^q (n_j)^q},\qquad s_{ii} = 0,$$

где $n_i$ — число слушателей трека. При $q = 0{,}5$ это ровно косинус; меньшие $q$
оставляют популярным трекам больше веса, большие — сильнее их прижимают. Нужен этот
показатель именно в discovery: иначе соседями всего подряд оказываются хиты, а
рекомендовать надо незнакомое.

У каждого трека оставляются только $k$ ближайших соседей (остальные близости
обнуляются) — обычное отсечение kNN, оно же защищает от шумных хвостовых совпадений.
Оценка кандидата — сумма близостей от знакомых треков пользователя:

$$\hat r_{uj} = \sum_{i \in I_u} s_{ij}.$$

Диагональ обнулена, поэтому знакомый трек не поддерживает сам себя; из кандидатов он
всё равно исключён (`common.topk`).

Подбор на val по сетке $(q, k)$, затем переобучение на train+val и замер на test.
Матрица близостей 10 000 x 10 000 во float32 — это 400 МБ, в память укладывается.
"""
import sys, os, time
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, evaluate, report


def cooccurrence(d):
    """Бинарный профиль и матрица совстречаемости треков (10 000 x 10 000)."""
    X = d.known.astype(np.float32)
    return X, X.T @ X


def similarity(co, n, q, k):
    """Близости с подавлением популярности и отсечением топ-k соседей по столбцу."""
    den = np.power(np.maximum(n, 1.0), q, dtype=np.float32)
    S = co / (den[:, None] * den[None, :])
    np.fill_diagonal(S, 0.0)
    if k < S.shape[0]:
        drop = np.argpartition(-S, k, axis=0)[k:]      # всё, что ниже топ-k, в ноль
        np.put_along_axis(S, drop, 0.0, axis=0)
    return S


def fit_knn(d, q=0.5, k=200, parts=None):
    """Оценки (пользователи x треки) для item-based kNN."""
    X, co = (parts if parts is not None else cooccurrence(d))
    n = co.diagonal().copy()                            # число слушателей трека
    S = similarity(co, n, q, k)
    scores = (X @ S).astype(np.float32)
    del S
    return scores


if __name__ == "__main__":
    dv = load("val")
    print(dv, "\n")
    parts = cooccurrence(dv)
    best, best_ndcg = None, -1.0
    for q in (0.3, 0.5, 0.7, 0.9):
        for k in (50, 100, 200, 500):
            t0 = time.time()
            res = evaluate(dv, fit_knn(dv, q=q, k=k, parts=parts))
            report("kNN q=%-4g k=%-3d" % (q, k), res)
            print("      (%.0f с)" % (time.time() - t0))
            if res["NDCG@10"] > best_ndcg:
                best, best_ndcg = (q, k), res["NDCG@10"]
    del parts
    print("\nлучшее на val: q=%g k=%d (NDCG@10 %.5f)\n" % (*best, best_ndcg))

    dt = load("test")
    print(dt)
    q, k = best
    sc = fit_knn(dt, q=q, k=k)
    report("kNN q=%g k=%d" % best, evaluate(dt, sc))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores_knn.npy"), sc)
