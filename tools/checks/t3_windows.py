import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, os
raw = sys.argv[1]
# Блок 02. Границы train/val/test в РЕАЛЬНЫХ сутках (единица timestamp = 1 с,
# см. t1_time_unit.py). Логика повторяет n1_test_window.py блока 06, но окна
# задаются в реальных сутках, а не в 1/5 суток.
DAY = 86400
NDAYS = 301                  # сутки 0..300, последние заполнены на 92.6 %
TMAX = 26_000_000
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))

pairs = (LF.select("uid", "item_id", "timestamp")
           .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
           .collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("ufirst"))
NTOT = ufirst.height
print("пар (uid, item): %d, пользователей с прослушиваниями: %d" % (pairs.height, NTOT))

MAXW = 120           # самое раннее нужное начало окна: val 90 суток перед test 30
ev_all = (LF.filter(pl.col("timestamp") >= (NDAYS - MAXW) * DAY)
            .select("uid", "item_id", "timestamp", "played_ratio_pct")
            .collect(engine="streaming")
            .join(pairs, on=["uid", "item_id"], how="left"))
print("прослушиваний в последних %d сутках: %d\n" % (MAXW, ev_all.height))
TOTAL = LF.select(pl.len()).collect().item()

QS = [.25, .5, .75, .9]


def window(ts0, ts1, label, hist_src=None):
    """Метрики окна [ts0, ts1). hist_src — срез, по которому считается «знакомое»
    (по умолчанию всё до ts0, то есть окно идёт сразу за обучением)."""
    h = ts0 if hist_src is None else hist_src
    tru = ufirst.filter(pl.col("ufirst") < h)["uid"]
    n_hist = len(tru)
    cold = (ufirst["ufirst"] >= h).sum()
    ev = (ev_all.filter((pl.col("timestamp") >= ts0) & (pl.col("timestamp") < ts1))
                .filter(pl.col("uid").is_in(tru.implode())))
    rel = ev.filter(pl.col("played_ratio_pct") >= 50)
    new = rel.filter(pl.col("t0") >= h)
    k = new.group_by("uid").agg(pl.col("item_id").n_unique().alias("k"))
    full = (ufirst.filter(pl.col("ufirst") < h).select("uid")
                  .join(k, on="uid", how="left")
                  .with_columns(pl.col("k").fill_null(0))["k"])
    print("  %s" % label)
    print("    пользователей с историей до окна %d (холодных, начавших внутри, %d)"
          % (n_hist, cold))
    print("    прослушиваний %d, релевантных (>=50 %%) %d (%.1f %%), из них новых "
          "%d (%.1f %%)" % (ev.height, rel.height,
                            100 * rel.height / max(ev.height, 1), new.height,
                            100 * new.height / max(rel.height, 1)))
    print("    активных в окне %d, с релевантным %d, ОЦЕНИВАЕМЫХ (есть релевантный "
          "новый трек) %d (%.1f %% от имеющих историю)"
          % (ev["uid"].n_unique(), rel["uid"].n_unique(), len(k),
             100 * len(k) / max(n_hist, 1)))
    if len(k):
        s = k["k"]
        print("    новых треков на оцениваемого: среднее %.1f, %s, max %d"
              % (s.mean(), " ".join("q%02d=%d" % (int(q * 100), s.quantile(q))
                                    for q in QS), s.max()))
        print("    с >= 10 релевантными новинками: %d (%.1f %% от имеющих историю)"
              % ((full >= 10).sum(), 100 * (full >= 10).sum() / n_hist))
    return dict(n_hist=n_hist, cold=cold, ev=ev.height, rel=rel.height,
                new=new.height, users=len(k),
                u10=int((full >= 10).sum()))


print("=== A. принятая граница блока 06 в реальном времени ===")
TS_ACC = 1414 * 17280
print("  граница timestamp = %d = реальные сутки %.2f; окно до конца данных = "
      "%.2f суток" % (TS_ACC, TS_ACC / DAY, (TMAX - TS_ACC) / DAY))
a = window(TS_ACC, TMAX + 1, "test = [%d, конец), 18.1 реальных суток" % TS_ACC)
print("    train теряет %.1f %% прослушиваний"
      % (100 * (TOTAL - LF.filter(pl.col("timestamp") < TS_ACC).select(pl.len())
                          .collect().item()) / TOTAL))

print("\n=== B. тестовое окно L реальных суток (test = сутки %d-L..300) ===" % NDAYS)
res = {}
for L in [7, 14, 18, 21, 28, 30, 45, 60, 90]:
    d0 = NDAYS - L
    ts0 = d0 * DAY
    lost = 100 * (TOTAL - LF.filter(pl.col("timestamp") < ts0).select(pl.len())
                            .collect().item()) / TOTAL
    res[L] = window(ts0, TMAX + 1,
                    "L = %2d суток: test = сутки %d..300, train = сутки 0..%d, "
                    "train теряет %.1f %%" % (L, d0, d0 - 1, lost))
    res[L]["lost"] = lost
    print()

print("=== C. сводка по тестовому окну ===")
print("  %-6s %9s %9s %11s %9s %9s %8s"
      % ("L, сут", "с истор.", "холодн.", "оцениваемых", "доля", ">=10 нов.", "train-"))
for L, r in res.items():
    print("  %-6d %9d %9d %11d %8.1f%% %9d %7.1f%%"
          % (L, r["n_hist"], r["cold"], r["users"],
             100 * r["users"] / r["n_hist"], r["u10"], r["lost"]))

print("\n=== D. окно val перед тестом ===")
# test фиксирован (выбранный вариант, пересчитывается в реальных сутках);
# val — V суток непосредственно перед ним, обучение для val — всё до val
vres = {}
for TEST_L in [18, 21, 30]:
    print("\n  -- при test = %d суток (сутки %d..300) --" % (TEST_L, NDAYS - TEST_L))
    vres[TEST_L] = {}
    for V in [7, 14, 18, 21, 28, 30, 45, 60, 90]:
        v1 = (NDAYS - TEST_L) * DAY
        v0 = (NDAYS - TEST_L - V) * DAY
        tr_end = v0
        lost_v = 100 * (TOTAL - LF.filter(pl.col("timestamp") < tr_end)
                                  .select(pl.len()).collect().item()) / TOTAL
        r = window(v0, v1, "val = %2d суток (сутки %d..%d), train = сутки 0..%d, "
                           "train теряет %.1f %% (с учётом test)"
                   % (V, NDAYS - TEST_L - V, NDAYS - TEST_L - 1,
                      NDAYS - TEST_L - V - 1, lost_v))
        r["lost"] = lost_v
        vres[TEST_L][V] = r
        print()

print("=== E. сводка по окну val ===")
for TEST_L, tab in vres.items():
    print("  при test = %d суток:" % TEST_L)
    print("    %-6s %9s %9s %11s %9s %9s %9s"
          % ("V, сут", "с истор.", "холодн.", "оцениваемых", "доля", "нов. соб.", "train-"))
    for V, r in tab.items():
        print("    %-6d %9d %9d %11d %8.1f%% %9d %8.1f%%"
              % (V, r["n_hist"], r["cold"], r["users"],
                 100 * r["users"] / r["n_hist"], r["new"], r["lost"]))
