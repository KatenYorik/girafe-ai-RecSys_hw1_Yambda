"""Таргет модели 5: «понравилось при знакомстве».

Форма формулы взята из измерений блока 05, свободными оставлены только три
коэффициента. Таргет считается на обучающем окне стадии и ни в одной своей части
не смотрит в оценочное окно.

    r(u, i) = base(n) * quality(r0) + like_bonus * L

* `n` — сколько раз пользователь слушал трек в обучающем окне.
  `base(n) = min(n, C)`: блок 05 показал, что P(лайк) растёт по повторам почти
  ЛИНЕЙНО (+1,6…1,8 п. п. на повтор) до n ≈ 20, а дальше прирост падает до +1,21
  (21–30), +0,67 (31–50), +0,12 (51–100) и уходит в минус. Поэтому линейный рост
  с обрезанием вернее логарифма, который занижал бы участок 1…20.

* `r0` — `played_ratio_pct` первого прослушивания пары в окне, то есть качество
  самого знакомства. `quality(r0)` — множители из двумерной таблицы блока 05,
  нормированные на зону 50–98 %:

      0–1 %   (скип)      0,35     — худшая зона, но НЕ ноль: при n >= 11 такие
                                     пары дают P(лайк) 21,4 %
      2–49 %               0,75
      50–98 %              1,00     — максимум
      99–100 %             0,60     — ниже, чем 50–98: трек доиграл до конца,
                                     потому что им никто не управлял (фон)

  Множители намеренно немонотонны по r0 — блок 05 прямо запрещает монотонную
  добавку, кривая разворачивается на 99–100 %.

* `L` — лайк, поставленный «при знакомстве»: не раньше начала первого
  воспроизведения пары (решение от 27.10.2026) и не отменённый в пределах
  часа (мисклики, блок 05). Начало воспроизведения = метка прослушивания минус
  проигранные секунды, потому что прослушивание помечается концом (блок 09).

`is_organic` в таргет не входит: блоки 01, 05, 06 и 09 согласно показали, что он
маркирует канал знакомства, а не силу предпочтения.
"""
import os
import numpy as np
import polars as pl

from common import RAW

QUALITY = [(0, 1, 0.35), (2, 49, 0.75), (50, 98, 1.00), (99, 1000, 0.60)]

VARIANTS = {
    "A": dict(cap=20, like_bonus=5.0,  label="повторы главные, лайк скромный"),
    "B": dict(cap=20, like_bonus=15.0, label="сбалансированный: лайк ≈ 15 прослушиваний"),
    "C": dict(cap=10, like_bonus=30.0, label="лайк главный, раннее насыщение"),
}


def quality(r0):
    q = np.full(len(r0), 0.60, dtype=np.float32)
    for lo, hi, v in QUALITY:
        q[(r0 >= lo) & (r0 <= hi)] = v
    return q


def pair_stats(d):
    """Для каждой пары обучающего окна: n, r0 и начало первого воспроизведения."""
    fit = d.fit_events.sort("timestamp")
    g = fit.group_by(["uid", "item_id"]).agg(
        pl.len().alias("n"),
        pl.col("played_ratio_pct").first().alias("r0"),
        pl.col("timestamp").first().alias("t_end"))
    # длительность нужна из raw: в подвыборке её нет
    lens = (pl.scan_parquet(os.path.join(RAW, "listens.parquet"))
              .select("uid", "item_id", "timestamp", "track_length_seconds")
              .collect(engine="streaming"))
    g = g.join(lens, left_on=["uid", "item_id", "t_end"],
               right_on=["uid", "item_id", "timestamp"], how="left")
    g = g.with_columns(
        (pl.col("t_end") - (pl.col("track_length_seconds") * pl.col("r0") / 100)
         ).alias("t_start"))
    return g


def discovery_likes(d, g):
    """Лайки «при знакомстве»: не раньше начала первого воспроизведения, без мискликов."""
    hi = d.fit_events["timestamp"].max() + 1
    U = pl.Series(d.users).implode()
    lk = (pl.scan_parquet(os.path.join(RAW, "likes.parquet"))
            .filter((pl.col("timestamp") < hi) & pl.col("uid").is_in(U))
            .select("uid", "item_id", "timestamp").collect(engine="streaming")
            .sort("timestamp").group_by(["uid", "item_id"])
            .agg(pl.col("timestamp").first().alias("t_like")))
    ul = (pl.scan_parquet(os.path.join(RAW, "unlikes.parquet"))
            .filter((pl.col("timestamp") < hi) & pl.col("uid").is_in(U))
            .select("uid", "item_id", "timestamp").collect(engine="streaming")
            .sort("timestamp").group_by(["uid", "item_id"])
            .agg(pl.col("timestamp").first().alias("t_unlike")))
    lk = lk.join(ul, on=["uid", "item_id"], how="left").filter(
        pl.col("t_unlike").is_null() | ((pl.col("t_unlike") - pl.col("t_like")) > 3600))
    j = g.join(lk, on=["uid", "item_id"], how="left")
    return j.with_columns(
        (pl.col("t_like").is_not_null() & (pl.col("t_like") >= pl.col("t_start"))
         ).fill_null(False).alias("L"))


def build(d, cap, like_bonus, **kw):
    """Возвращает (строки, столбцы, веса) для матрицы обучения модели 5."""
    g = discovery_likes(d, pair_stats(d))
    n = g["n"].to_numpy().astype(np.float32)
    r0 = g["r0"].to_numpy()
    L = g["L"].to_numpy().astype(np.float32)
    w = np.minimum(n, cap) * quality(r0) + like_bonus * L
    ui = np.array([d.uidx[u] for u in g["uid"].to_list()], dtype=np.int32)
    ii = np.array([d.iidx[t] for t in g["item_id"].to_list()], dtype=np.int32)
    keep = w > 0
    return ui[keep], ii[keep], w[keep].astype(np.float32), dict(
        pairs=int(keep.sum()), likes=int(L.sum()),
        mass_from_likes=float((like_bonus * L)[keep].sum() / w[keep].sum()),
        mass_top=float(np.sort(w[keep])[::-1][:max(1, int(0.025 * keep.sum()))].sum()
                       / w[keep].sum()),
        w_median=float(np.median(w[keep])), w_max=float(w[keep].max()))
