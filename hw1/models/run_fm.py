"""FM с аудиоэмбеддингами: гибридная факторизационная машина, обученная на BPR.

Представление трека складывается из трёх частей, то есть это FM по взаимодействиям
«пользователь x признаки трека»:

    q_i = V_i + E_i W + A_{artist(i)}         (своё id, аудио, артист)
    score(u, i) = <P_u, q_i> + b_i

Аудио — `normalized_embed` длины 128 из data/catalog_embeddings.parquet: блок 09
намерил, что нормы `embed` различаются в восемь раз, поэтому в модель идёт
нормированный вектор. У 325 треков каталога вектора нет — им нули, как и решено
в блоке 01.

Обучение — BPR на мини-батчах: на каждую положительную пару берётся случайный
отрицательный кандидат. Положительные пары — прослушивания обучающего окна.
Модель написана на numpy целиком: готовых реализаций FM в работе нет.
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from common import load, evaluate, report, DATA


def scatter_add(M, idx, vals, lr):
    """M[idx] += lr * vals, но быстро: np.add.at на порядок медленнее."""
    n = M.shape[0]
    S = csr_matrix((np.ones(len(idx), dtype=np.float32),
                    (idx, np.arange(len(idx)))), shape=(n, len(idx)))
    M += lr * (S @ vals)

def audio(d):
    """Матрица (n_items x 128) нормированных аудиовекторов; у кого нет — нули."""
    e = pl.read_parquet(os.path.join(DATA, "catalog_embeddings.parquet"))
    j = pl.DataFrame({"item_id": d.items}).join(
        e.select("item_id", "normalized_embed"), on="item_id", how="left")
    dim = len(e["normalized_embed"][0])
    out = np.zeros((d.n_items, dim), dtype=np.float32)
    have = 0
    for k, v in enumerate(j["normalized_embed"].to_list()):
        if v is not None:
            out[k] = v
            have += 1
    return out, have


class FM:
    def __init__(self, d, dim=64, lr=0.05, reg=0.01, seed=42, use_audio=True,
                 use_artist=True):
        rng = np.random.default_rng(seed)
        self.d, self.dim, self.lr, self.reg = d, dim, lr, reg
        self.use_audio, self.use_artist = use_audio, use_artist
        self.P = rng.normal(0, 0.05, (d.n_users, dim)).astype(np.float32)
        self.V = rng.normal(0, 0.05, (d.n_items, dim)).astype(np.float32)
        self.b = np.zeros(d.n_items, dtype=np.float32)
        self.E, have = audio(d)
        self.W = rng.normal(0, 0.01, (self.E.shape[1], dim)).astype(np.float32)
        self.art, n_art = d.artists()
        self.A = rng.normal(0, 0.05, (n_art, dim)).astype(np.float32)
        self.n_art = n_art
        print("   аудиовекторов %d/%d, артистов %d" % (have, d.n_items, n_art))

    def q(self, idx):
        out = self.V[idx]
        if self.use_audio:
            out = out + self.E[idx] @ self.W
        if self.use_artist:
            out = out + self.A[self.art[idx]]
        return out

    def scores(self):
        allq = self.q(np.arange(self.d.n_items))
        return (self.P @ allq.T + self.b).astype(np.float32)

    def fit(self, epochs=20, batch=32768, seed=0):
        d = self.d
        rng = np.random.default_rng(seed)
        pu, pi = d.ui, d.ii
        n = len(pu)
        for ep in range(epochs):
            perm = rng.permutation(n)
            for s in range(0, n, batch):
                b = perm[s:s + batch]
                u, i = pu[b], pi[b]
                jn = rng.integers(0, d.n_items, len(b))
                bad = d.known[u, jn]
                while bad.any():                       # отрицательный не должен быть знакомым
                    jn[bad] = rng.integers(0, d.n_items, int(bad.sum()))
                    bad = d.known[u, jn]
                qi, qj = self.q(i), self.q(jn)
                pu_b = self.P[u]
                x = (pu_b * (qi - qj)).sum(1) + self.b[i] - self.b[jn]
                # клип перед exp: иначе при больших |x| сигмоида переполняется
                g = (1.0 / (1.0 + np.exp(np.clip(x, -30.0, 30.0)))).astype(np.float32)[:, None]
                lr, reg = self.lr, self.reg
                dP = g * (qi - qj) - reg * pu_b
                dqi, dqj = g * pu_b, -g * pu_b
                scatter_add(self.P, u, dP, lr)
                scatter_add(self.V, i, dqi - reg * self.V[i], lr)
                scatter_add(self.V, jn, dqj - reg * self.V[jn], lr)
                scatter_add(self.b[:, None], i, (g[:, 0] - reg * self.b[i])[:, None], lr)
                scatter_add(self.b[:, None], jn, (-g[:, 0] - reg * self.b[jn])[:, None], lr)
                if self.use_artist:
                    ai, aj = self.art[i], self.art[jn]
                    scatter_add(self.A, ai, dqi - reg * self.A[ai], lr)
                    scatter_add(self.A, aj, dqj - reg * self.A[aj], lr)
                if self.use_audio:
                    # W общая для всех примеров батча, поэтому её градиент надо УСРЕДНИТЬ.
                    # Остальные параметры обновляются построчно через scatter, то есть уже
                    # «на пример»; без деления на размер батча шаг для W оказывается на
                    # четыре порядка больше, и модель расходится.
                    self.W += lr * ((self.E[i].T @ dqi + self.E[jn].T @ dqj) / len(b)
                                    - reg * self.W)
        return self


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        dt = load("test")
        print(dt)
        m = FM(dt, dim=64, lr=0.05, reg=0.01).fit(epochs=20)
        sc = m.scores()
        report("FM (аудио + артист)", evaluate(dt, sc))
        np.save(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "scores_fm.npy"), sc)
        sys.exit(0)

    dv = load("val")
    print(dv)
    for tag, kw in [("FM аудио+артист", dict()),
                    ("FM без аудио", dict(use_audio=False))]:
        print(tag)
        m = FM(dv, dim=64, lr=0.05, reg=0.01, **kw).fit(epochs=20)
        report("   " + tag, evaluate(dv, m.scores()))
