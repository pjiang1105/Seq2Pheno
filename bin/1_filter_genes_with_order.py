import pandas as pd
import os
import argparse

def reorder_genes(input_file, output_file=None, ref_dir="./Ref"):
    gene_order_file = os.path.join(ref_dir, "Gene_names_with_order.tsv")

    # Read input data and gene order
    cmp_df = pd.read_csv(input_file, sep='\t')
    gene_order_df = pd.read_csv(gene_order_file, sep='\t')

    # Prepare gene list and case-insensitive map
    gene_order_list = gene_order_df.iloc[:, 0].str.lower().tolist()
    column_name_map = {col.lower(): col for col in cmp_df.columns}

    # Match and build column list (add placeholders for unmatched)
    matched_genes = []
    unmatched_genes = []

    for gene in gene_order_list:
        if gene in column_name_map:
            matched_genes.append(column_name_map[gene])
        else:
            unmatched_genes.append(gene)
            matched_genes.append(f"__placeholder__::{gene}")  # marker for later

    # Construct the reordered DataFrame
    first_column = cmp_df.columns[0]
    reordered_df = cmp_df[[first_column]].copy()

    for gene in matched_genes:
        if gene.startswith("__placeholder__::"):
            gene_name = gene.split("::")[1]
            reordered_df[gene_name] = 0
        else:
            reordered_df[gene] = cmp_df[gene]

    # Generate output filename if not provided
    if output_file is None:
        base = '.'.join(os.path.basename(input_file).split('.')[:-1])
        output_file = os.path.join(os.path.dirname(input_file), base + ".gene_filter_order.tsv")

    # Save reordered file
    reordered_df.to_csv(output_file, sep='\t', index=False)

    # Generate and write log
    output_dir = os.path.dirname(output_file)
    output_base = os.path.splitext(os.path.basename(output_file))[0]
    log_file = os.path.join(output_dir, f"{output_base}.Unmatched_Genes_Log.txt")
    with open(log_file, 'w') as f:
        f.write(f"Total genes in gene order list: {len(gene_order_list)}\n")
        f.write(f"Number of matched genes: {len(gene_order_list) - len(unmatched_genes)}\n")
        f.write(f"Number of unmatched genes: {len(unmatched_genes)}\n")
        f.write("Unmatched genes:\n")
        for gene in unmatched_genes:
            f.write(f"{gene}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reorder gene columns in a TSV file based on hardcoded reference gene list. Unmatched genes will be filled with 0.")
    parser.add_argument("input_file", help="Input TSV file to process")
    parser.add_argument("-o", "--output", help="Optional output file path for reordered TSV")
    parser.add_argument("--ref-dir", default="./Ref", help="Reference directory containing Gene_names_with_order.tsv")

    args = parser.parse_args()
    reorder_genes(args.input_file, args.output, args.ref_dir)
