"""Анализ по артистам и реранжирование — одинаковые для всех моделей.

Три ступени новизны (блок 06) в выдаче: знакомый трек в топ-10 невозможен по
построению, кандидаты его исключают, поэтому разбор идёт по двум оставшимся —
«новый трек знакомого артиста» и «новый артист».

Реранжирование (README.md): s' = s - lambda*[трек знаком] - mu*[артист знаком].
Первое слагаемое на кандидатном множестве тождественно нулевое (знакомые треки уже
выброшены), так что сетка строится по mu. Чтобы mu значило одно и то же для разных
моделей, оценки предварительно стандартизуются внутри пользователя по его кандидатам.

    python -I run_analysis.py
"""
import sys, os
if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import load, evaluate, topk, K

HERE = os.path.dirname(os.path.abspath(__file__))


def known_artists(d, art, n_art):
    """Маска (пользователи x артисты): слушал ли пользователь артиста в обучающем окне."""
    ka = np.zeros((d.n_users, n_art), dtype=bool)
    ka[d.ui, art[d.ii]] = True
    return ka


def standardize(scores, d):
    """z-оценка внутри пользователя по его кандидатам, чтобы mu была сравнима."""
    s = scores.astype(np.float32, copy=True)
    cand = ~d.known
    out = np.zeros_like(s)
    for u in d.pool:
        v = s[u][cand[u]]
        m, sd = v.mean(), v.std() + 1e-8
        out[u] = (s[u] - m) / sd
    return out


def novelty_report(d, scores, art, ka, name, mu=0.0):
    s = scores
    if mu:
        s = scores - mu * ka[:, art].astype(np.float32)
    rec = topk(s, d, K)
    rart = art[rec]
    known = np.take_along_axis(ka[d.pool], rart, 1)
    hits = np.take_along_axis(d.answer[d.pool], rec, 1)
    disc = 1.0 / np.log2(np.arange(2, K + 2))
    n = d.n_rel[d.pool]
    ideal = np.array([disc[:min(K, int(x))].sum() for x in n])
    ndcg = ((hits * disc).sum(1) / ideal).mean()
    recall = (hits.sum(1) / n).mean()
    # срез «новый артист»: релевантные новинки, у которых артист пользователю новый
    ans_new_art = d.answer[d.pool] & ~ka[d.pool][:, art]
    n_na = ans_new_art.sum(1)
    hits_na = np.take_along_axis(ans_new_art, rec, 1)
    m = n_na > 0
    recall_na = (hits_na[m].sum(1) / n_na[m]).mean()
    # концентрация на популярном
    pop = d.known.sum(0)
    top100 = np.argsort(-pop)[:100]
    is_top = np.isin(rec, top100)
    distinct = np.array([len(set(r.tolist())) for r in rart])
    print("%-26s mu=%-4g NDCG@10 %.5f  Recall@10 %.5f  новых артистов в топ-10 %5.1f%%  "
          "разных артистов %4.2f  из топ-100 по популярности %5.1f%%  Recall@10 на срезе "
          "«новый артист» %.5f (польз. %d)"
          % (name, mu, ndcg, recall, 100 * (~known).mean(), distinct.mean(),
             100 * is_top.mean(), recall_na, int(m.sum())))
    return dict(mu=mu, ndcg=float(ndcg), recall=float(recall),
                new_artist_share=float((~known).mean()), distinct=float(distinct.mean()),
                top100=float(is_top.mean()), recall_new_artist=float(recall_na))


SAVED = [("kNN", "scores_knn.npy"), ("EASE", "scores_ease.npy"),
         ("iALS", "scores_ials.npy"), ("Модель 5", "scores_model5.npy"),
         ("FM (аудио)", "scores_fm.npy")]


def model_scores(d):
    """Оценки всех моделей: Popularity считается на месте, остальные берутся из .npy.

    Файлы оставляют за собой run_knn / run_ease / run_ials / run_model5 / fm_one на
    стадии test. Так анализ разбирает ровно те модели, числа которых стоят в таблице,
    и ничего не переобучает заново.
    """
    out = {}
    pop = d.known.sum(0).astype(np.float32)
    out["Popularity"] = np.broadcast_to(pop, (d.n_users, d.n_items)).copy()
    missing = []
    for name, f in SAVED:
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[name] = np.load(p)
        else:
            missing.append(f)
    if missing:
        sys.exit("нет файлов оценок: %s — сначала прогоните соответствующие run_*.py"
                 % ", ".join(missing))
    return out


if __name__ == "__main__":
    d = load("test")
    print(d)
    art, n_art = d.artists()
    ka = known_artists(d, art, n_art)
    print("артистов в каталоге %d; знакомых артистов на пользователя: медиана %.0f\n"
          % (n_art, np.median(ka.sum(1))))

    scores = model_scores(d)
    print("=== без реранжирования ===")
    base = {}
    for name, s in scores.items():
        base[name] = novelty_report(d, s, art, ka, name)

    print("\n=== кривая по mu (реранжирование), стандартизованные оценки ===")
    curves = {}
    for name, s in scores.items():
        z = standardize(s, d)
        curves[name] = [novelty_report(d, z, art, ka, name, mu=mu)
                        for mu in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)]
        print()
    np.save(os.path.join(HERE, "curves.npy"), curves, allow_pickle=True)
