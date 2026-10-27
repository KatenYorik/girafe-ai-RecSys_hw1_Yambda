# Данные (Yambda-50M)

Скачано скриптом [`tools/download_data.sh`](../tools/download_data.sh) в `data/raw/`
(в репозиторий не коммитим). Источник — [`yandex/yambda`](https://huggingface.co/datasets/yandex/yambda),
Apache 2.0, доступ без токена.

| Файл | Размер | Строк |
|---|---:|---:|
| `listens.parquet` | 369 МБ | 46 467 212 |
| `likes.parquet` | 7,2 МБ | 881 456 |
| `dislikes.parquet` | 1,0 МБ | 107 776 |
| `unlikes.parquet` | 2,2 МБ | 312 972 |
| `undislikes.parquet` | 0,2 МБ | 21 033 |
| `artist_item_mapping.parquet` | 38 МБ | 9 271 906 |
| `album_item_mapping.parquet` | 55 МБ | 9 651 644 |
| `embed_item_ids.parquet` | 21 МБ | 7 721 749 |
| **итого** | **471 МБ** | |

Маппинги артистов/альбомов — общие для всех размеров Yambda (весь каталог 9,39 млн треков),
отдельной версии под 50M нет. `multi_event.parquet` (384 МБ) не качали: это объединение тех же
событий, нужные таблицы есть по отдельности.

## Эмбеддинги: не качаем
`embeddings.parquet` — **13,8 ГБ** (7 721 749 треков, `item_id`, `embed`, `normalized_embed`,
оба `List(Float64)`). На диске свободно ~17 ГБ, так что локально он не живёт.
Polars читает его удалённо с отбором колонок — `item_id` целиком (7,7 млн) приезжает за ~5 с:

```python
import polars as pl
URL = "https://huggingface.co/datasets/yandex/yambda/resolve/main/embeddings.parquet"
ids = pl.scan_parquet(URL).select("item_id").collect()
```

Когда дойдём до FM с аудио (блоки 07/09): выгрузить векторы **только для треков финальной
подвыборки** (~5–15 тыс.), сразу кастуя во `Float32`, и сохранить отдельным parquet на несколько
МБ. Отбор строк по `item_id` не экономит трафик (нужные id разбросаны по всем row group'ам),
поэтому читать стримом и держать только совпавшие строки — на диск полная версия не ложится.

## Подвыборка (`data/subsample/`)
Собирается детерминированно скриптом [`tools/build_subsample.py`](../tools/build_subsample.py)
из `raw`, в репозиторий не коммитим (как и `raw`):

```bash
python -I tools/build_subsample.py data/raw
```

Параметры приняты блоками 02/03/04 и зашиты в скрипт: сплит train 0…258 / val 259…279 /
test 280…300 суток, пул 9 055 пользователей после фильтра блока 03, подвыборка 3 000
пользователей (сид 42), каталог топ-10 000 треков по числу слушателей пула в истории до test.
Обоснование — [`eda/08-subsample/README.md`](../eda/08-subsample/README.md).

| Файл | Что внутри |
|---|---|
| `train.parquet`, `val.parquet`, `test.parquet` | события окна для 3 000 пользователей, только треки каталога: `uid`, `item_id`, `timestamp`, `played_ratio_pct`, `is_organic`, `is_new` (пара впервые встречается в этом окне), `is_relevant` (`played_ratio_pct >= 50`) |
| `catalog.parquet` | `item_id`, `rank`, `pool_users`, `artist_id`, `has_embed`, `debut_day` |
| `users.parquet` | `uid`, `n_pre`, `days_pre`, `n_test` |

Эмбеддинги для FM выгружать по `item_id` из `catalog.parquet` — это ровно 10 000 треков,
у 96,75 % из них вектор есть (блок 04).

## Эмбеддинги каталога (`catalog_embeddings.parquet`)
Выгружены скриптом [`tools/checks/f4_export_embeddings.py`](../tools/checks/f4_export_embeddings.py)
для 10 000 треков каталога, 9,2 МБ, в репозиторий не коммитим:

```bash
python -I tools/checks/f4_export_embeddings.py data/raw
```

| колонка | тип | заметка |
|---|---|---|
| `item_id` | `UInt32` | 9 675 треков каталога из 10 000 (96,75 %); у 325 вектора нет — им нулевой вектор и `has_embed = 0` |
| `embed` | `List(Float32)` | 128 чисел, норма по каталогу: медиана 28,47 (16,56…137,00) |
| `normalized_embed` | `List(Float32)` | тот же вектор с нормой 1; посчитан локально |
| `embed_norm` | `Float32` | норма `embed`, чтобы не считать заново |

Скачивать обе колонки векторов незачем: проверено, что
`max |embed/‖embed‖ − normalized_embed| = 2,2 · 10⁻¹⁶`, поэтому по сети идёт только `embed`
(вдвое меньше трафика). Чтение — чанками по 800 тыс. строк: `slice` проталкивается в
сканирование паркета и пропускает лишние row group'ы, а один `collect` на весь файл процесс
не выдерживает (убивает по памяти). Обоснование и замеры —
[`eda/09-features-leakage/README.md`](../eda/09-features-leakage/README.md).

## Чем читаем
Установлены `polars 1.44`, `pandas 2.2.3`, `numpy 2.2.5`. **`pyarrow` нет**, поэтому
`pandas.read_parquet` не работает — читаем через polars. Моделям нужен ещё `scipy`
(разреженные матрицы), графикам — `matplotlib`. Библиотек рекомендательных моделей в
зависимостях нет: задание требует писать модели с нуля, поэтому `implicit` из работы убран,
а ALS свой. Полный список — в `requirements.txt` в корне.

## Где лежат данные
Путь к каталогу данных нигде не зашит: `models/common.py` ищет его функцией `data_dir()` —
сначала переменная окружения `YAMBDA_DATA`, затем `data/` рядом с кодом, затем `../hw1/data`.
Так репозиторий сдачи работает, не копируя 523 МБ: данные остаются здесь.
