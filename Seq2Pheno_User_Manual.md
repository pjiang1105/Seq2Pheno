<h1 align="center">Seq2Pheno: an AI/ML-based pipeline to predict tissue phenotypes from RNA-seq data</h1>

<p align="center">
  Developed by <strong>Peng Jiang, Ph.D.</strong><br>
  Assistant Professor, Cleveland State University, Cleveland, OH, USA <br>
  Honorary Fellow, University of Wisconsin–Madison, Madison, WI, USA <br>
  📧 p.jiang@csuohio.edu
</p>

---

## Introduction

Tissue phenotypes such as fibrosis, macrophage polarization, and regenerative responses are commonly evaluated using histological methods such as hematoxylin–eosin staining and immunohistochemistry. While informative, these approaches are labor-intensive, costly, and often impractical for frozen biobank specimens. RNA-seq, by contrast, offers scalable and cost-effective transcriptomic profiling but lacks direct phenotypic readouts.  

To address this gap, we developed **Seq2Pheno**, an AI/ML framework designed to provide quantitative estimates of tissue phenotypes from transcriptomic profiles. Seq2Pheno operates in two stages. In the first stage, an encoder–decoder was trained on RNA-seq profiles from 751 canine wound-healing biopsies spanning diverse biological and perturbation contexts. This step compressed high-dimensional transcriptomes into a compact latent representation that distilled core biological signals while retaining the ability to reconstruct the original data. In the second stage, subsets of samples with paired histological measurements were used to train and evaluate phenotype-prediction models. Phenotypes included fibrosis markers (e.g., α-SMA), fibrosis burden estimates (e.g., percent fibrosis and semi-quantitative scoring), immunoregulatory macrophages (CD163), macrophage activation (IBA-1 and MAC387), pro-inflammatory cytokine (TNF-α), and regenerative markers such as PAX7.

**Input Files:** RNA-seq (counts per million, CPM)  
**Output:** Predicted tissue phenotype values (e.g., fibrosis burden, IHC optical density for α-SMA, CD163, IBA-1, MAC387, TNF-α, and PAX7).

---

## 📂 Tool Structure
```
Seq2Pheno/
├── Snakefile
├── Run_snakemake.sh
├── bin/
│   ├── 1_filter_genes_with_order.py
│   ├── 2_Ref_quantile_Normalization.py
│   ├── 3_log10_transform.py
│   ├── 4_Encoder.py
│   └── 5_ML_Pred.py
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

## 🧩 Quick Start (Aligned with Included Example Data)

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

## 📝 Citation
Jiang, P. et al. *Seq2Pheno: Predicting Tissue Phenotypes from RNA-seq Data.* (2025)

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


