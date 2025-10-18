import os
import glob
from datetime import datetime

# Get current timestamp for directory naming
timestamp = datetime.now().strftime('%Y-%m-%d-%H%M')

# Define directories
INPUT_DIR = "Input_files"
ENCODER_OUTPUT_DIR = f"Outputs_{timestamp}/Encoder"
PHENOTYPE_OUTPUT_DIR = f"Outputs_{timestamp}/Pred_Phenotype"

# Create output directories if they don't exist
os.makedirs(ENCODER_OUTPUT_DIR, exist_ok=True)
os.makedirs(PHENOTYPE_OUTPUT_DIR, exist_ok=True)

# Identify input samples
input_files = glob.glob(os.path.join(INPUT_DIR, "*.tsv"))
SAMPLES = [os.path.basename(f).replace(".tsv", "") for f in input_files]

# Define final target
rule all:
    input:
        expand(os.path.join(PHENOTYPE_OUTPUT_DIR, "{sample}.Phenotype_Pred.tsv"), sample=SAMPLES)

# Step 1: Filter genes
rule filter_genes:
    input:
        lambda wildcards: os.path.join(INPUT_DIR, f"{wildcards.sample}.tsv")
    output:
        os.path.join(ENCODER_OUTPUT_DIR, "{sample}.gene_filter_order.tsv")
    shell:
        "python3 ./bin/1_filter_genes_with_order.py {input} -o {output}"

# Step 2: Quantile normalization
rule quantile_normalization:
    input:
        os.path.join(ENCODER_OUTPUT_DIR, "{sample}.gene_filter_order.tsv")
    output:
        os.path.join(ENCODER_OUTPUT_DIR, "{sample}.Ref_Normalized_CPM.tsv")
    shell:
        "python3 ./bin/2_Ref_quantile_Normalization.py {input} {output}"

# Step 3: Log10 transform
rule log10_transform:
    input:
        os.path.join(ENCODER_OUTPUT_DIR, "{sample}.Ref_Normalized_CPM.tsv")
    output:
        os.path.join(ENCODER_OUTPUT_DIR, "{sample}.log10_Ref_Normalized_CPM.tsv")
    shell:
        "python3 ./bin/3_log10_transform.py {input} {output}"

# Step 4: Encoder to latent space
rule encode:
    input:
        os.path.join(ENCODER_OUTPUT_DIR, "{sample}.log10_Ref_Normalized_CPM.tsv")
    output:
        os.path.join(ENCODER_OUTPUT_DIR, "{sample}.encoder.tsv")
    shell:
        "python3 ./bin/4_Encoder.py {input} {output}"

# Step 5: Predict phenotype
rule process_encoder:
    input:
        os.path.join(ENCODER_OUTPUT_DIR, "{sample}.encoder.tsv")
    output:
        os.path.join(PHENOTYPE_OUTPUT_DIR, "{sample}.Phenotype_Pred.tsv")
    shell:
        "python3 ./bin/5_ML_Pred.py {input} {output}"

