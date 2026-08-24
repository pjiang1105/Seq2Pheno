<h1 align="center">Seq2Pheno: An AI/ML Framework for Predicting Immunological and Histological Tissue Phenotypes from RNA-seq Data</h1>

<p align="center">
  Developed by <strong>Peng Jiang, Ph.D.</strong><br>
  Assistant Professor, Cleveland State University, Cleveland, OH, USA <br>
  Honorary Fellow, University of Wisconsin–Madison, Madison, WI, USA <br>
  📧 p.jiang@csuohio.edu
</p>

---

## Introduction

Tissue phenotypes such as fibrosis, macrophage polarization, and regenerative responses are commonly evaluated using histological methods such as Masson’s Trichrome staining and immunohistochemistry. While informative, these approaches are labor-intensive, costly, and often impractical for frozen biobank specimens. RNA-seq, by contrast, offers scalable and cost-effective transcriptomic profiling but lacks direct phenotypic readouts.  

To address this gap, we developed **Seq2Pheno**, an AI/ML framework designed to provide quantitative estimates of tissue phenotypes from transcriptomic profiles. Seq2Pheno operates in two stages. In the first stage, an encoder–decoder was trained on RNA-seq profiles from 751 canine wound-healing biopsies spanning diverse biological and perturbation contexts. This step compressed high-dimensional transcriptomes into a compact latent representation that distilled core biological signals while retaining the ability to reconstruct the original data. In the second stage, subsets of samples with paired histological measurements were used to train and evaluate phenotype-prediction models. Phenotypes included fibrosis markers (e.g., α-SMA), fibrosis burden estimates (e.g., percent fibrosis and semi-quantitative scoring), immunoregulatory macrophages (CD163), macrophage activation (IBA-1 and MAC387), pro-inflammatory cytokine (TNF-α), and regenerative markers such as PAX7.

**Input Files:** RNA-seq (counts per million, CPM)  
**Output:** Predicted tissue phenotype values (e.g., fibrosis burden, IHC optical density for α-SMA, CD163, IBA-1, MAC387, TNF-α, and PAX7).

---

## 📂 Tool Structure
```
Seq2Pheno_2.0/
├── Snakefile
├── Run_snakemake.sh
├── bin/
│   ├── 1_filter_genes_with_order.py
│   ├── 2_Ref_quantile_Normalization.py
│   ├── 3_log10_transform.py
│   ├── 4_Encoder.py
│   └── 5_ML_Pred.py
├── train/
│   ├── seq2pheno_train.py
│   ├── Run_user_training.example.sh
│   └── README.md
├── Input_files/
│   ├── Sandbox_CMP_Trunk_01.tsv
│   └── Sandbox_CMP_Trunk_02.tsv
├── Ref/
│   ├── Gene_names_with_order.tsv
│   ├── Reference_Rank_Ave_CPM.tsv
│   ├── scaler_parameters.tsv
│   ├── encoder_model.h5
│   ├── a-SMA/ (trained models + Best_Model.txt)
│   ├── CD163/ (trained models + Best_Model.txt)
│   ├── IBA-1/ (trained models + Best_Model.txt)
│   ├── MAC387/ (trained models + Best_Model.txt)
│   ├── PAX7/ (trained models + Best_Model.txt)
│   ├── Semi-fibrosis-score/ (trained models + Best_Model.txt)
│   ├── TNF-a/ (trained models + Best_Model.txt)
│   └── TZ_Percentage_Fibrosis/ (trained models + Best_Model.txt)
├── Outputs_Examples/
│   ├── Encoder/...
│   └── Pred_Phenotype/...
└── Outputs_<timestamp>/ (generated automatically)
```

Seq2Pheno 2.0 keeps two workflows separate:

1. **Prediction workflow**: the original Snakemake pipeline using the bundled `Ref/` models.
2. **User training workflow**: a separate Python pipeline under `train/` that creates a custom reference directory such as `User_Training_Run/Ref/`.

---

## ⚙️ System Requirements
- **Operating System:** Linux/macOS  
- **Python:** ≥3.8  
- **Snakemake:** ≥7.32.4  
- **Recommended Hardware:** ≥8 GB RAM, ≥4 CPU cores  
- **Optional GPU Support:** NVIDIA GPU with CUDA for TensorFlow acceleration  

---

## 📦 Software Dependencies
Python libraries:
- numpy  
- pandas  
- scipy  
- scikit-learn  
- tensorflow (optional, for GPU support)  
- snakemake  
- joblib  ← used for parallel processing in normalization and model loading

---

## 🛠 Installation

### Conda (Recommended)
**CPU version:**
```bash
conda create -n seq2pheno python=3.8 snakemake numpy pandas scipy scikit-learn tensorflow joblib
conda activate seq2pheno
```

**GPU version:**
```bash
conda create -n seq2pheno_gpu python=3.8 snakemake numpy pandas scipy scikit-learn tensorflow-gpu joblib
conda activate seq2pheno_gpu
```
Make sure CUDA and cuDNN are installed properly for GPU acceleration.

### Alternative (pip)
```bash
pip install numpy pandas scipy scikit-learn snakemake tensorflow joblib
# or (legacy CUDA wheels)
pip install numpy pandas scipy scikit-learn snakemake tensorflow-gpu joblib
```

---

## 🧪 Verify Installation
```bash
python3 -c "import numpy, pandas, scipy, sklearn, tensorflow, joblib; print('All libraries installed correctly.')"
```
Expected output:
```
All libraries installed correctly.
```

---

## ⚠️ Python Interpreter Clarification
Seq2Pheno leverages Snakemake’s integrated Python script execution, automatically selecting the Python interpreter based on your active environment.

---

## 🧩 Quick Start (Example input data can be found in Input_files sub-folder)

This quick start runs the full pipeline on the **two included example matrices** located in `Input_files/`:
- `Sandbox_CMP_Trunk_01.tsv`
- `Sandbox_CMP_Trunk_02.tsv`

> Each example file is a **CPM matrix with samples as rows and genes as columns**, with the first column named `Sample`.

### 1) Run the pipeline
From the repository root:
```bash
# Option A: use the provided script (adjust cores if needed)
bash Run_snakemake.sh

# Option B: run Snakemake directly
snakemake --cores 8
```

### 2) Examine generated outputs
A timestamped directory is created, e.g., `Outputs_2025-07-04-1200/`, containing:

```
Outputs_<timestamp>/
├── Encoder/
│   ├── Sandbox_CMP_Trunk_01.gene_filter_order.tsv
│   ├── Sandbox_CMP_Trunk_01.gene_filter_order.Unmatched_Genes_Log.txt
│   ├── Sandbox_CMP_Trunk_01.Ref_Normalized_CPM.tsv
│   ├── Sandbox_CMP_Trunk_01.log10_Ref_Normalized_CPM.tsv
│   ├── Sandbox_CMP_Trunk_01.encoder.tsv
│   └── (corresponding files for Sandbox_CMP_Trunk_02)
└── Pred_Phenotype/
    ├── Sandbox_CMP_Trunk_01.Phenotype_Pred.tsv
    └── Sandbox_CMP_Trunk_02.Phenotype_Pred.tsv
```

### 3) Preview predictions
```bash
# Show header and first 3 rows for Trunk 01 predictions
head -n 4 Outputs_*/Pred_Phenotype/Sandbox_CMP_Trunk_01.Phenotype_Pred.tsv
```
**Expected columns:**
```
Sample	a-SMA	CD163	IBA-1	MAC387	PAX7	Semi-fibrosis-score	TNF-a	TZ_Percentage_Fibrosis
```

### 4) (Optional) Sanity check vs. provided example outputs
We provide expected results under `Outputs_Examples/`. You can compare your run to those files:
```bash
# Compare the phenotype predictions (ignores timestamped output path)
diff -q Outputs_Examples/Pred_Phenotype/Sandbox_CMP_Trunk_01.Phenotype_Pred.tsv        Outputs_*/Pred_Phenotype/Sandbox_CMP_Trunk_01.Phenotype_Pred.tsv

diff -q Outputs_Examples/Pred_Phenotype/Sandbox_CMP_Trunk_02.Phenotype_Pred.tsv        Outputs_*/Pred_Phenotype/Sandbox_CMP_Trunk_02.Phenotype_Pred.tsv
```

---

## 🛠 Modifying the Snakemake Workflow
The pipeline is defined in the `Snakefile` at the project root. To edit:
```bash
nano Snakefile
```
Common modification:
```python
INPUT_DIR = "Input_files"  # change to match your dataset location
```

---

## 📥 Input Files (Format)
- Place tab-separated CPM matrices in the `Input_files/` directory.  
- **Orientation:** the first column is `Sample` (sample IDs), and the remaining columns are gene symbols (one column per gene).  
- Filenames should end with `.tsv` (e.g., `MyCohort.tsv`).

**Example:**
```
Input_files/
├── Sandbox_CMP_Trunk_01.tsv
└── Sandbox_CMP_Trunk_02.tsv
```

---

## 📤 Output Files (What the pipeline produces)
- **Encoder/ (intermediate):**  
  - `<trunk>.gene_filter_order.tsv` (reordered/filtered gene matrix)  
  - `<trunk>.gene_filter_order.Unmatched_Genes_Log.txt` (genes not found in reference; informational)  
  - `<trunk>.Ref_Normalized_CPM.tsv` (reference rank-based quantile normalized CPM)  
  - `<trunk>.log10_Ref_Normalized_CPM.tsv` (log10 transformed)  
  - `<trunk>.encoder.tsv` (latent representation)  

- **Pred_Phenotype/ (final):**  
  - `<trunk>.Phenotype_Pred.tsv` (columns: `Sample`, `a-SMA`, `CD163`, `IBA-1`, `MAC387`, `PAX7`, `Semi-fibrosis-score`, `TNF-a`, `TZ_Percentage_Fibrosis`)

---

## 📚 Workflow Summary
1. **Filter & reorder genes (`bin/1_filter_genes_with_order.py`)** – aligns input genes to `Ref/Gene_names_with_order.tsv`; logs unmatched genes.  
2. **Reference rank-based normalization (`bin/2_Ref_quantile_Normalization.py`)** – maps CPM ranks to `Ref/Reference_Rank_Ave_CPM.tsv` (parallelized with joblib).  
3. **Log10 transform (`bin/3_log10_transform.py`)** – stabilizes expression distributions.  
4. **Encoding (`bin/4_Encoder.py`)** – scales using `Ref/scaler_parameters.tsv` and encodes with `Ref/encoder_model.h5`.  
5. **Phenotype prediction (`bin/5_ML_Pred.py`)** – applies phenotype-specific models in `Ref/*/Best_Model.txt` to produce final predictions.

---

## 🧬 Seq2Pheno 2.0: Train Models With Your Own Data

Seq2Pheno 2.0 adds a separate training pipeline for users who have their own RNA-seq CPM matrix and matched phenotype measurements. This training pipeline does **not** overwrite the bundled `Ref/` directory. It writes a new custom reference directory that can later be used by the same prediction Snakemake workflow.

### Training Input 1: Expression CPM Matrix

The expression matrix must be tab-separated, with samples as rows and genes as columns. The first column is the sample ID column unless a different column is specified.

```
Sample	GeneA	GeneB	GeneC
S1	10.2	0	5.5
S2	3.1	8.4	0
S3	0	1.4	22.0
```

### Training Input 2: Phenotype Table

The phenotype table must be tab-separated. The first column is the sample ID column unless a different column is specified. Every numeric non-sample column is trained as a phenotype by default.

```
Sample	Fibrosis	CD163	My_New_Phenotype
S1	12.4	0.31	5.0
S2	9.8	0.22	7.1
S3	14.0	0.40	6.3
```

Sample IDs must match between the expression and phenotype files. Samples without a value for a given phenotype are dropped only for that phenotype model.

### Recommended Directory Layout

```
Seq2Pheno_2.0/
├── Training_Input/
│   ├── my_training_cpm.tsv
│   └── my_phenotypes.tsv
├── train/
│   └── seq2pheno_train.py
└── User_Training_Run/          # generated by the training command
```

### Run User Training

From the `Seq2Pheno_2.0/` directory:

```bash
python3 train/seq2pheno_train.py \
  --expression-cpm Training_Input/my_training_cpm.tsv \
  --phenotypes Training_Input/my_phenotypes.tsv \
  --output-dir User_Training_Run
```

To train only selected phenotype columns:

```bash
python3 train/seq2pheno_train.py \
  --expression-cpm Training_Input/my_training_cpm.tsv \
  --phenotypes Training_Input/my_phenotypes.tsv \
  --phenotype-columns Fibrosis,CD163 \
  --output-dir User_Training_Run
```

If the sample ID columns are not named or positioned the same way:

```bash
python3 train/seq2pheno_train.py \
  --expression-cpm Training_Input/my_training_cpm.tsv \
  --phenotypes Training_Input/my_phenotypes.tsv \
  --expression-sample-column Sample \
  --phenotype-sample-column Biopsy_ID \
  --output-dir User_Training_Run
```

### Training Outputs

```
User_Training_Run/
├── Ref/
│   ├── Gene_names_with_order.tsv
│   ├── Reference_Rank_Ave_CPM.tsv
│   ├── scaler_parameters.tsv
│   ├── encoder_model.h5
│   ├── Fibrosis/
│   │   ├── Best_Model.txt
│   │   └── final_*_model.*
│   └── CD163/
│       ├── Best_Model.txt
│       └── final_*_model.*
├── Preprocessed/
│   ├── training.gene_filter_order.tsv
│   ├── training.Ref_Normalized_CPM.tsv
│   └── training.log10_Ref_Normalized_CPM.tsv
├── Autoencoder/
│   ├── autoencoder_model.h5
│   ├── encoder_model.h5
│   ├── encoded_features.tsv
│   ├── reconstructed_data.tsv
│   ├── reconstruction_errors.tsv
│   └── training_history.tsv
├── Phenotype_Models/
│   ├── phenotype_training_summary.tsv
│   └── <Phenotype>/
│       ├── 2_Predicted_vs_Observed.tsv
│       ├── 2_crossval_performance_metrics_all_models.tsv
│       ├── Best_Model.txt
│       └── final_*_model.*
└── training_manifest.json
```

The most important output is:

```
User_Training_Run/Ref/
```

This directory contains the gene order, reference normalization values, scaler parameters, encoder, phenotype model folders, and `Best_Model.txt` files needed for prediction.

### Use a User-Trained Model for Prediction

Place new CPM files for prediction in `Input_files/` using the same input format as the original workflow:

```
Input_files/
└── My_New_Cohort.tsv
```

Then run the normal prediction workflow while pointing Seq2Pheno to the custom reference:

```bash
SEQ2PHENO_REF_DIR=User_Training_Run/Ref snakemake --cores 8
```

The generated output will contain phenotype columns matching the custom trained phenotypes:

```
Outputs_<timestamp>/Pred_Phenotype/My_New_Cohort.Phenotype_Pred.tsv
```

### Important Notes for User Training

- The training CPM matrix should represent the same type of expression unit expected by Seq2Pheno: counts per million (CPM).
- The custom `Ref/` generated by training should be used for future prediction on compatible data from the same gene space and preprocessing assumptions.
- More matched samples generally make cross-validation model selection more reliable.
- By default, Seq2Pheno trains Random Forest, Gradient Boosting, SVR, Ridge Regression, Lasso Regression, and Neural Network models, then selects the model with the highest cross-validation Spearman correlation for each phenotype.
- To skip neural-network phenotype predictors while still training the autoencoder, add `--skip-neural-network`.

---

## 📝 Citation
Jiang, P. et al. (2026) *Seq2Pheno: An AI/ML Farmwork to Predict Immunological and Histological Tissue Phenotypes from RNA-seq Data*

---

## ⚖️ License
Seq2Pheno tool is licensed under the **Non-Commercial MIT License (NC-MIT)**.  

**Copyright (c) 2025 Peng Jiang**

Permission is hereby granted, free of charge, to any person obtaining a copy  
of this software and associated documentation files (the “Software”), to deal  
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, and distribute copies of the Software,  
**for non-commercial purposes only**, subject to the following conditions:  

1. **Non-Commercial Use:** The Software and its derivatives may be used, copied,  
   and modified solely for non-commercial research, academic, or educational  
   purposes. Any commercial use — including but not limited to selling, licensing,  
   or incorporating the Software into a commercial product or service — is  
   **prohibited without prior written permission** from the copyright holder.  

2. **Attribution:** The above copyright notice and this permission notice shall  
   be included in all copies or substantial portions of the Software.  

---

**Disclaimer:**  
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,  
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE  
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER  
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  
SOFTWARE.


---

## 📬 Contact
**Peng Jiang, Ph.D.**  
**Assistant Professor (Tenure-track)**<br>
Center for Gene Regulation in Health and Disease<br>
Center for Applied Data Analysis and Modeling<br>
Cleveland State University, 2121 Euclid Ave, Cleveland, OH 44115, USA<br>
**Honorary Fellow**<br>
School of Medicine and Public Health (SMPH)<br>
University of Wisconsin–Madison, 750 Highland Avenue, Madison, WI 53705, USA<br><br>
📧 p.jiang@csuohio.edu
