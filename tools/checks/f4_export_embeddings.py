import sys; if hasattr(sys.stdout, "reconfigure"):      # в ядре Jupyter stdout его не имеет
    sys.stdout.reconfigure(encoding="utf-8")
import polars as pl, numpy as np, os, time, shutil, glob
raw = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    os.path.dirname(os.path.abspath(raw.rstrip("/\\"))), "catalog_embeddings.parquet")
CHUNK = int(sys.argv[3]) if len(sys.argv) > 3 else 800_000
# Блок 09. Выгрузка аудиоэмбеддингов для 10 000 треков каталога из удалённого
# embeddings.parquet (13,8 ГБ). Конфигурация каталога — блок 04.
#
# Читаем ЧАНКАМИ по строкам: polars проталкивает slice в сканирование паркета и
# пропускает лишние row group'ы (проверено: slice на офсете 4 млн стоит столько же,
# сколько на нуле). Один collect на весь файл не проходит — процесс убивает по памяти.
# Накладные расходы на вызов (~40 с) сильно больше предельной скорости чтения
# (~24 тыс. строк/с), поэтому чанк берём большой: 800 тыс. строк ≈ 1 ГБ в памяти.
#
# Качаем только embed, без normalized_embed: это ровно вдвое меньше трафика, а
# normalized_embed = embed / ||embed|| — скрипт это проверяет на выборке перед загрузкой
# и считает нормированный вектор локально.
#
# Чанки складываются по одному в <out>.parts/ и при повторном запуске не перекачиваются,
# так что обрыв не теряет работу. Папка удаляется после успешной сборки.
DAY = 86400
TEST0 = 280 * DAY
K = 10000
URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/embeddings.parquet"
parts = out + ".parts"
LF = pl.scan_parquet(os.path.join(raw, "listens.parquet"))

pairs = (LF.select("uid", "item_id", "timestamp")
           .group_by(["uid", "item_id"]).agg(pl.col("timestamp").min().alias("t0"))
           .collect(engine="streaming"))
ufirst = pairs.group_by("uid").agg(pl.col("t0").min().alias("uf"))
zero = (LF.select("uid", "played_ratio_pct").group_by("uid")
          .agg((pl.col("played_ratio_pct") == 0).mean().alias("z"))
          .collect(engine="streaming"))
ntr = (LF.filter(pl.col("timestamp") < TEST0).select("uid")
         .group_by("uid").len().collect(engine="streaming"))
pool = sorted((set(ufirst.filter(pl.col("uf") < TEST0)["uid"].to_list())
               - set(zero.filter(pl.col("z") > 0.5)["uid"].to_list()))
              & set(ntr.filter(pl.col("len") >= 5)["uid"].to_list()))
P = pl.Series("uid", pool)
trm = (LF.filter(pl.col("timestamp") < TEST0).select("uid", "item_id").unique()
         .collect(engine="streaming").filter(pl.col("uid").is_in(P.implode())))
CAT = (trm.group_by("item_id").agg(pl.col("uid").n_unique().alias("users"))
          .sort(["users", "item_id"], descending=[True, False]).head(K))
ids = pl.Series("item_id", sorted(CAT["item_id"].to_list()))
del pairs, ufirst, zero, ntr, trm
print("каталог собран: %d треков, пул %d" % (len(ids), len(pool)), flush=True)

# --- сверка: normalized_embed = embed / ||embed|| ---
t = time.time()
chk = pl.scan_parquet(URL).select("item_id", "embed", "normalized_embed").slice(0, 2000)
chk = chk.collect()
a = np.asarray(chk["embed"].to_list(), dtype=np.float64)
b = np.asarray(chk["normalized_embed"].to_list(), dtype=np.float64)
dev = np.abs(a / np.linalg.norm(a, axis=1, keepdims=True) - b).max()
print("сверка на %d строках за %.0f с: max |embed/||embed|| − normalized_embed| = %.3e "
      "→ качаем только embed, нормируем локально" % (chk.height, time.time() - t, dev),
      flush=True)
assert dev < 1e-6, "normalized_embed не равен embed/||embed||, качать надо обе колонки"
del chk, a, b

total = pl.scan_parquet(URL).select(pl.len()).collect().item()
nch = (total + CHUNK - 1) // CHUNK
print("в файле %d строк, чанков по %d: %d" % (total, CHUNK, nch), flush=True)
os.makedirs(parts, exist_ok=True)

t0 = time.time()
got = 0
for i in range(nch):
    p = os.path.join(parts, "part%03d.parquet" % i)
    if os.path.exists(p):
        got += pl.read_parquet(p).height
        print("  чанк %3d/%d: уже выгружен" % (i + 1, nch), flush=True)
        continue
    t = time.time()
    d = (pl.scan_parquet(URL).select("item_id", "embed")
           .slice(i * CHUNK, CHUNK)
           .filter(pl.col("item_id").is_in(ids.implode()))
           .with_columns(pl.col("embed").cast(pl.List(pl.Float32)))
           .collect())
    d.write_parquet(p)
    got += d.height
    print("  чанк %3d/%d: совпало %5d строк, %.0f с (всего %5d из %d, прошло %.0f мин)"
          % (i + 1, nch, d.height, time.time() - t, got, len(ids), (time.time() - t0) / 60),
          flush=True)
    del d

emb = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(os.path.join(parts, "*.parquet")))])
emb = emb.unique(subset=["item_id"], keep="first").sort("item_id")
v = np.asarray(emb["embed"].to_list(), dtype=np.float32)
nrm = np.linalg.norm(v, axis=1, keepdims=True)
emb = emb.with_columns(
    pl.Series("normalized_embed", (v / nrm).astype(np.float32).tolist(),
              dtype=pl.List(pl.Float32)),
    pl.Series("embed_norm", nrm[:, 0].astype(np.float32)))
emb.write_parquet(out)
shutil.rmtree(parts, ignore_errors=True)
print("\nзаписано: %s (%.1f МБ) за %.0f мин"
      % (out, os.path.getsize(out) / 1e6, (time.time() - t0) / 60))
print("треков каталога с вектором: %d из %d (%.2f %%)"
      % (emb.height, len(ids), 100 * emb.height / len(ids)))
print("размерность: %s; типы: %s"
      % (sorted(set(emb["embed"].list.len().to_list())),
         {k: str(w) for k, w in dict(emb.collect_schema()).items()}))
print("норма embed: медиана %.3f, min %.3f, max %.3f; норма normalized_embed: "
      "min %.6f, max %.6f"
      % (emb["embed_norm"].median(), emb["embed_norm"].min(), emb["embed_norm"].max(),
         float(np.linalg.norm(np.asarray(emb["normalized_embed"].to_list(),
                                         dtype=np.float64), axis=1).min()),
         float(np.linalg.norm(np.asarray(emb["normalized_embed"].to_list(),
                                         dtype=np.float64), axis=1).max())))
miss = pl.DataFrame({"item_id": ids}).join(emb.select("item_id"), on="item_id", how="anti")
print("треков каталога без вектора: %d (ожидается 325 — блок 04)" % miss.height)
