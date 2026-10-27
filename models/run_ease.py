"""EASE: B = -G^-1 / diag(G^-1) при G = X^T X + lambda*I, диагональ B обнуляется.

Лямбда подбирается на val, затем модель переобучается на train+val и мерится на test.
Каталог 10 000 треков выбран в блоке 04 именно под эту модель.
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, evaluate, report


def ease(X, lam):
    G = (X.T @ X).astype(np.float32)
    G[np.diag_indices_from(G)] += np.float32(lam)
    P = np.linalg.inv(G)
    del G
    B = P / (-np.diag(P))
    del P
    B[np.diag_indices_from(B)] = 0.0
    return B


def binary(d):
    return (d.known.astype(np.float32))


if __name__ == "__main__":
    grid = [50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0]
    dv = load("val")
    print(dv)
    best, best_ndcg = None, -1.0
    for lam in grid:
        X = binary(dv)
        B = ease(X, lam)
        res = evaluate(dv, X @ B)
        report("EASE lambda=%-6g" % lam, res)
        if res["NDCG@10"] > best_ndcg:
            best, best_ndcg = lam, res["NDCG@10"]
        del B
    print("\nлучшая lambda на val: %g (NDCG@10 %.5f)\n" % (best, best_ndcg))

    dt = load("test")
    print(dt)
    X = binary(dt)
    B = ease(X, best)
    sc = (X @ B).astype(np.float32)
    report("EASE lambda=%g" % best, evaluate(dt, sc))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores_ease.npy"), sc)
