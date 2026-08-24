#!/usr/bin/env bash
set -euo pipefail

# Edit these three paths for a real training run.
EXPRESSION_CPM="Training_Input/my_training_cpm.tsv"
PHENOTYPES="Training_Input/my_phenotypes.tsv"
OUTPUT_DIR="User_Training_Run"

python3 train/seq2pheno_train.py \
  --expression-cpm "${EXPRESSION_CPM}" \
  --phenotypes "${PHENOTYPES}" \
  --output-dir "${OUTPUT_DIR}" \
  --autoencoder-epochs 100 \
  --cv-splits 5
