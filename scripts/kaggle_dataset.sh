#!/usr/bin/env bash
# F1.8 — Publica (o versiona) el dataset privado de Kaggle con SOLO el manifiesto y los splits.
# Requiere `kaggle` (grupo dev) y ~/.kaggle/kaggle.json. Uso:
#   KAGGLE_USERNAME=<usuario> scripts/kaggle_dataset.sh          # primera vez: create
#   KAGGLE_USERNAME=<usuario> scripts/kaggle_dataset.sh version  # actualizaciones
set -euo pipefail
cd "$(dirname "$0")/.."
: "${KAGGLE_USERNAME:?exporta KAGGLE_USERNAME}"
mode="${1:-create}"
out=kaggle/dataset
rm -rf "$out" && mkdir -p "$out"
cp data/manifests/isic2020.csv data/splits/train.txt data/splits/val.txt data/splits/test.txt data/splits/SHA256SUMS "$out/"
sed "s/KAGGLE_USERNAME/$KAGGLE_USERNAME/" kaggle/dataset-metadata.json > "$out/dataset-metadata.json"
ls -la "$out"
if [ "$mode" = create ]; then
  uv run --no-sync kaggle datasets create -p "$out" --dir-mode skip
else
  uv run --no-sync kaggle datasets version -p "$out" -m "actualización $(date +%F)" --dir-mode skip
fi
echo "Dataset privado: https://www.kaggle.com/datasets/$KAGGLE_USERNAME/melanoma-isic2020-splits"
