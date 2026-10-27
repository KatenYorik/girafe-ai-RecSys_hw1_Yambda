#!/bin/sh
# Скачивает Yambda-50M (flat) + маппинги артистов/альбомов в data/raw/.
# Итого ~473 МБ. embeddings.parquet (13,8 ГБ) НЕ качаем — читаем удалённо,
# см. data/README.md.
set -e
DEST="$(dirname "$0")/../data/raw"
BASE="https://huggingface.co/datasets/yandex/yambda/resolve/main"
mkdir -p "$DEST"
for f in flat/50m/listens.parquet flat/50m/likes.parquet flat/50m/dislikes.parquet \
         flat/50m/unlikes.parquet flat/50m/undislikes.parquet \
         artist_item_mapping.parquet album_item_mapping.parquet ; do
  name=$(basename "$f")
  [ -f "$DEST/$name" ] && { echo "skip $name"; continue; }
  echo ">>> $f"
  curl -fL --retry 3 --retry-delay 2 -o "$DEST/$name" "$BASE/$f"
done
echo "OK -> $DEST"
