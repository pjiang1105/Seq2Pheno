# User Training Pipeline

This directory contains the separate Seq2Pheno 2.0 training pipeline. It does not overwrite the bundled `Ref/` directory used by the original example workflow.

## Required Inputs

1. Expression CPM matrix, tab-separated:

```text
Sample  GeneA  GeneB  GeneC
S1      10.2   0      5.5
S2      3.1    8.4    0
```

2. Phenotype table, tab-separated:

```text
Sample  Fibrosis  CD163
S1      12.4      0.31
S2      9.8       0.22
```

Sample IDs must match between the two files. The phenotype file may contain any numeric phenotype columns.

## Run

```bash
python3 train/seq2pheno_train.py \
  --expression-cpm Training_Input/my_training_cpm.tsv \
  --phenotypes Training_Input/my_phenotypes.tsv \
  --output-dir User_Training_Run
```

The main output for future prediction is:

```text
User_Training_Run/Ref/
```

Use that custom reference with the standard Seq2Pheno prediction workflow:

```bash
SEQ2PHENO_REF_DIR=User_Training_Run/Ref snakemake --cores 8
```

