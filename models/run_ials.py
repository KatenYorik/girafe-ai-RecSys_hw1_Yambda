"""iALS на неявном отклике — своя реализация (Hu-Koren-Volinsky, 2008).

Задание запрещает готовые реализации рекомендательных моделей, поэтому ALS написан
здесь целиком на numpy.

**Постановка.** Наблюдения бинаризуются: $p_{ui} = 1$, если пара встречалась в обучающем
окне, иначе 0. Сила наблюдения уходит не в цель, а в вес — уверенность
$c_{ui} = \alpha \cdot r_{ui}$, где $r_{ui}$ — значение матрицы (по умолчанию число
прослушиваний, у модели 5 — свой таргет). Минимизируется

$$\sum_{u,i} w_{ui}\,(p_{ui} - x_u^\top y_i)^2 + \lambda\Big(\sum_u \|x_u\|^2 + \sum_i \|y_i\|^2\Big),$$

где $w_{ui} = c_{ui}$ на наблюдённых парах и $w_{ui} = 1$ на всех остальных. Сумма идёт
по **всем** парам: ненаблюдённые тянут вектора к нулю с единичным весом и дают член
$Y^\top Y$ ниже — один и тот же для всех пользователей.

Отсюда следствие, которое объясняет, почему на val выигрывает $\alpha = 1$: при нём пара,
прослушанная один раз, получает вес 1, то есть ровно вес ненаблюдённой пары, и отличается
от неё только целью $p$. Повторы начинают весить больше только при $r > 1$, а большие
$\alpha$ раздувают вес популярных повторов и тем вредят discovery-постановке.

**Шаг.** При фиксированной матрице предметов $Y$ каждый пользователь решается отдельно:

$$x_u = \big(Y^\top Y + Y_u^\top (C_u - I) Y_u + \lambda I\big)^{-1} Y_u^\top c_u,$$

где $Y_u$ — строки $Y$ по предметам пользователя, $c_u$ — их уверенности. Первый член
общий для всех пользователей и считается один раз за полушаг, поэтому цена шага линейна
по числу наблюдений, а не по размеру матрицы. Шаг по предметам симметричен. Система
решается точно (`np.linalg.solve`) — это ближе к формуле, чем сопряжённые градиенты.

Пользователи без истории остаются нулевыми: правая часть у них пустая. Из оценки они и
так выпадают, `common.py` держит их вне пула.
"""
import sys, os, time
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, evaluate, report


CAP = 8_000_000      # элементов во временном буфере батча (~32 МБ во float32)


def _half_step(C, F, reg, cap=CAP):
    """Решает вектора для всех строк C при фиксированной F (вторая сторона).

    Считает батчами: строки сортируются по числу наблюдений, соседние по длине
    собираются в один паддингованный тензор, и накопление Y_u^T (C_u - I) Y_u для
    всей группы идёт одним пакетным matmul вместо отдельного умножения на каждую
    строку. На паддинге c = 0, поэтому он не вносит ничего ни в матрицу, ни в правую
    часть. Строка без наблюдений остаётся нулевой.
    """
    n, f = C.shape[0], F.shape[1]
    base = ((F.T @ F) + reg * np.eye(f)).astype(np.float32)
    out = np.zeros((n, f), dtype=np.float32)
    indptr, indices, data = C.indptr, C.indices, C.data
    lens = np.diff(indptr)
    order = np.argsort(lens, kind="stable")
    order = order[lens[order] > 0]
    i, m = 0, len(order)
    while i < m:
        b = 1
        while (i + b < m and b < 4096
               and (b + 1) * int(lens[order[i + b]]) * f <= cap):
            b += 1
        rows = order[i:i + b]
        i += b
        L = int(lens[rows].max())
        Yb = np.zeros((len(rows), L, f), dtype=np.float32)
        cb = np.zeros((len(rows), L), dtype=np.float32)
        for t, r in enumerate(rows):
            s0, e0 = indptr[r], indptr[r + 1]
            Yb[t, :e0 - s0] = F[indices[s0:e0]]
            cb[t, :e0 - s0] = data[s0:e0]
        w = cb - (cb != 0)                            # (c - 1) на наблюдённых, 0 на паддинге
        A = base + np.matmul((Yb * w[:, :, None]).transpose(0, 2, 1), Yb)
        rhs = np.matmul(cb[:, None, :], Yb)[:, 0, :]
        out[rows] = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
    return out


def als(C, factors=128, reg=0.05, iters=20, seed=42, verbose=False):
    """C — csr уверенностей (пользователи x предметы). Возвращает (X, Y)."""
    Ct = C.T.tocsr()
    rng = np.random.default_rng(seed)
    Y = (rng.random((C.shape[1], factors)) * 0.01).astype(np.float32)
    X = np.zeros((C.shape[0], factors), dtype=np.float32)
    for it in range(iters):
        t0 = time.time()
        X = _half_step(C, Y, reg)
        Y = _half_step(Ct, X, reg)
        if verbose:
            print("      итерация %2d за %.1f с" % (it + 1, time.time() - t0))
    return X, Y


def fit_ials(d, weights=None, factors=128, reg=0.05, alpha=10.0, iters=20, seed=42,
             weights_coo=None, verbose=False):
    """Оценки (пользователи x треки). weights_coo=(строки, столбцы, веса) — свой таргет."""
    if weights_coo is not None:
        from scipy.sparse import csr_matrix
        ui, ii, w = weights_coo
        m = csr_matrix((w, (ui, ii)), shape=(d.n_users, d.n_items))
        m.sum_duplicates()
    else:
        m = d.csr(weights)
    m.data = (alpha * m.data).astype(np.float32)
    X, Y = als(m, factors=factors, reg=reg, iters=iters, seed=seed, verbose=verbose)
    return (X @ Y.T).astype(np.float32)


if __name__ == "__main__":
    # Пространство поиска и почему оно такое. Своя реализация считает ощутимо
    # медленнее библиотечной (замер: 2,4 минуты на обучение при factors=64 и
    # 10,5 минут при 128), поэтому прежняя сетка из 27 комбинаций целиком не
    # прогоняется. Порядок подбора: сначала reg x alpha при дешёвом factors=64
    # (9 комбинаций), затем проверка, помогает ли удвоение factors при лучших
    # (reg, alpha). Так сетка опирается на прежний замер, где alpha решала всё
    # (выигрывала во всех девяти комбинациях factors x reg), reg почти не влиял,
    # а factors 128 и 256 различались в пятом знаке.
    dv = load("val")
    print(dv, "\n")
    best, best_ndcg = None, -1.0
    for reg in (0.01, 0.05, 0.2):
        for alpha in (1.0, 10.0, 40.0):
            t0 = time.time()
            res = evaluate(dv, fit_ials(dv, factors=64, reg=reg, alpha=alpha))
            report("iALS f=64  reg=%-4g a=%-4g" % (reg, alpha), res)
            print("      (%.0f с)" % (time.time() - t0))
            if res["NDCG@10"] > best_ndcg:
                best, best_ndcg = (64, reg, alpha), res["NDCG@10"]
    _, r, a = best
    print("\nлучшее при factors=64: reg=%g alpha=%g (NDCG@10 %.5f)" % (r, a, best_ndcg))

    print("\nпроверка factors=128 при reg=%g alpha=%g:" % (r, a))
    t0 = time.time()
    res128 = evaluate(dv, fit_ials(dv, factors=128, reg=r, alpha=a))
    report("iALS f=128 reg=%-4g a=%-4g" % (r, a), res128)
    print("      (%.0f с)" % (time.time() - t0))
    if res128["NDCG@10"] > best_ndcg:
        best, best_ndcg = (128, r, a), res128["NDCG@10"]
    print("\nитог подбора: factors=%d reg=%g alpha=%g (NDCG@10 %.5f)\n" % (*best, best_ndcg))

    dt = load("test")
    print(dt)
    f, r, a = best
    sc = fit_ials(dt, factors=f, reg=r, alpha=a)
    report("iALS f=%d reg=%g a=%g" % best, evaluate(dt, sc))
    np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scores_ials.npy"), sc)
