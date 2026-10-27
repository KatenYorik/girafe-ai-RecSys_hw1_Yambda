"""Общий слой для всех моделей: данные, кандидаты, метрики.

Все решения EDA зашиты здесь в одном месте, чтобы пять моделей сравнивались честно:

* сплит (блок 02): train = сутки 0…258, val = 259…279, test = 280…300;
* две стадии — `val` (учимся на train, мерим на val) и `test` (учимся на train+val,
  мерим на test), признаки и популярность пересчитываются на каждой (блок 09);
* кандидаты (блок 06): каталог 10 000 треков минус всё, что пользователь уже слушал
  в обучающем окне. «Знаком» = любое прослушивание, без порога (см. LIMITATIONS.md);
* релевантность (блок 06): новый трек, прослушанный на >= 50 % в оценочном окне;
* метрика (блок 06): NDCG@10 с нормировкой на idealDCG@min(10, число релевантных)
  и Recall@10; считается на пользователях, у которых есть и профиль, и релевантные
  новинки (блок 09).

Использование::

    from common import load, evaluate
    d = load("val")
    scores = my_model(d)            # матрица (пользователи x треки) float32
    print(evaluate(d, scores))
"""
import os
import numpy as np
import polars as pl

DAY = 86400
VAL0, TEST0 = 259 * DAY, 280 * DAY
K = 10

_HERE = os.path.dirname(os.path.abspath(__file__))
_UP = os.path.dirname(_HERE)                     # каталог над models/


def data_dir():
    """Каталог с данными — ищется, а не зашивается.

    Данные (523 МБ) в репозиторий не входят, поэтому путь не прописан константой.
    Порядок поиска: переменная окружения `YAMBDA_DATA`, затем `data/` и `data/`
    рядом с кодом и на несколько каталогов выше. Так один и тот же код работает и
    когда данные скачаны внутрь репозитория, и когда они лежат по соседству.
    """
    cands = [os.environ.get("YAMBDA_DATA")]
    d = _UP
    for _ in range(4):
        cands += [os.path.join(d, "data"), os.path.join(d, "hw1", "data")]
        d = os.path.dirname(d)
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "subsample")):
            return c
    raise RuntimeError(
        "не найден каталог данных: положите его рядом как data/ или укажите путь "
        "в переменной окружения YAMBDA_DATA (нужны подкаталоги subsample/ и raw/)")


DATA = data_dir()
SUB = os.path.join(DATA, "subsample")
RAW = os.path.join(DATA, "raw")


class Data:
    """Готовая к работе стадия: матрица обучения, кандидаты, ответы."""

    def __init__(self, stage):
        assert stage in ("val", "test")
        self.stage = stage
        items = pl.read_parquet(os.path.join(SUB, "catalog.parquet"))["item_id"].sort()
        users = pl.read_parquet(os.path.join(SUB, "users.parquet"))["uid"].sort()
        self.items = items.to_numpy()
        self.users = users.to_numpy()
        self.n_users, self.n_items = len(self.users), len(self.items)
        self.uidx = {u: i for i, u in enumerate(self.users.tolist())}
        self.iidx = {t: i for i, t in enumerate(self.items.tolist())}

        fit_files = ["train"] if stage == "val" else ["train", "val"]
        fit = pl.concat([pl.read_parquet(os.path.join(SUB, f + ".parquet")) for f in fit_files])
        ev = pl.read_parquet(os.path.join(SUB, (stage if stage == "val" else "test") + ".parquet"))
        self.fit_events, self.eval_events = fit, ev

        self.ui = self._rows(fit, "uid")
        self.ii = self._rows(fit, "item_id", items=True)

        # кандидаты: маска того, что пользователь уже слушал в обучающем окне
        self.known = np.zeros((self.n_users, self.n_items), dtype=bool)
        self.known[self.ui, self.ii] = True

        # ответы: новые релевантные пары оценочного окна
        rel = ev.filter(pl.col("is_new") & pl.col("is_relevant")).select("uid", "item_id").unique()
        ru = self._rows(rel, "uid")
        ri = self._rows(rel, "item_id", items=True)
        self.answer = np.zeros((self.n_users, self.n_items), dtype=bool)
        self.answer[ru, ri] = True

        self.n_rel = self.answer.sum(1)
        self.has_profile = np.asarray(self.known.sum(1) > 0)
        self.pool = np.where((self.n_rel > 0) & self.has_profile)[0]

    def _rows(self, df, col, items=False):
        m = self.iidx if items else self.uidx
        return df[col].replace_strict(m, return_dtype=pl.Int32).to_numpy()

    def counts(self):
        """Матрица обучения в виде плотного счётчика прослушиваний (3 000 x 10 000)."""
        c = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        np.add.at(c, (self.ui, self.ii), 1.0)
        return c

    def csr(self, weights=None):
        """Разреженная матрица: её берут свой ALS (run_ials) и EASE."""
        from scipy.sparse import csr_matrix
        w = np.ones(len(self.ui), dtype=np.float32) if weights is None else weights
        m = csr_matrix((w, (self.ui, self.ii)), shape=(self.n_users, self.n_items))
        m.sum_duplicates()
        return m

    def artists(self):
        """Вектор длины n_items: индекс артиста для каждого трека каталога."""
        am = (pl.scan_parquet(os.path.join(RAW, "artist_item_mapping.parquet"))
                .unique(subset=["item_id"], keep="first").collect())
        j = pl.DataFrame({"item_id": self.items}).join(am, on="item_id", how="left")
        a = j["artist_id"].fill_null(-1).to_numpy()
        uniq = {v: i for i, v in enumerate(sorted(set(a.tolist())))}
        return np.array([uniq[v] for v in a.tolist()], dtype=np.int32), len(uniq)

    def __repr__(self):
        return ("Data(stage=%s, пользователей %d, треков %d, оцениваемых %d, "
                "релевантных пар %d, профиль есть у %d)"
                % (self.stage, self.n_users, self.n_items, len(self.pool),
                   int(self.n_rel.sum()), int(self.has_profile.sum())))


def load(stage):
    return Data(stage)


def topk(scores, d, k=K):
    """Топ-k кандидатов для каждого оцениваемого пользователя; знакомое исключено."""
    s = scores[d.pool].astype(np.float32, copy=True)
    s[d.known[d.pool]] = -np.inf
    part = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.take_along_axis(s, part, 1).argsort(1)[:, ::-1]
    return np.take_along_axis(part, order, 1)


def evaluate(d, scores, k=K):
    """NDCG@k с idealDCG@min(k, n_rel) и Recall@k, усреднённые по оцениваемым."""
    rec = topk(scores, d, k)
    hits = np.take_along_axis(d.answer[d.pool], rec, 1)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * disc).sum(1)
    n = d.n_rel[d.pool]
    ideal = np.array([disc[:min(k, int(x))].sum() for x in n])
    return {"stage": d.stage, "users": len(d.pool),
            "NDCG@%d" % k: float((dcg / ideal).mean()),
            "Recall@%d" % k: float((hits.sum(1) / n).mean()),
            "HitRate@%d" % k: float((hits.sum(1) > 0).mean())}


def report(name, res):
    print("%-22s %s  пользователей %4d  NDCG@10 %.5f  Recall@10 %.5f  HitRate@10 %.4f"
          % (name, res["stage"], res["users"], res["NDCG@10"], res["Recall@10"],
             res["HitRate@10"]))
