import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import argparse
import os

# Load reference rank mapping
def load_ref(ref_path):
    ref_df = pd.read_csv(ref_path, sep='\t')
    ref_cpm = np.round(ref_df['CPM.Ave'].values, 2)
    return np.insert(ref_cpm, 0, 0.0)

# Function to map CPM values to CPM.Ave based on rank
def rank_to_cpm(row, ref_cpm):
    non_zero_idx = row != 0
    ranks = row.argsort()[::-1].argsort() + 1  # Rank in descending order
    cpm_mapped = np.zeros_like(row, dtype=np.float32)
    cpm_mapped[non_zero_idx] = ref_cpm[ranks[non_zero_idx]]
    return np.round(cpm_mapped, 2)

# Process chunk using parallelization
def process_chunk(chunk, ref_cpm, n_jobs):
    chunk_values = chunk.to_numpy(dtype=np.float32)
    results = Parallel(n_jobs=n_jobs)(delayed(rank_to_cpm)(row, ref_cpm) for row in chunk_values)
    return pd.DataFrame(results, index=chunk.index, columns=chunk.columns)

# Main processing function
def main(input_file, output_file, n_jobs=4, chunksize=50000):
    ref_file = './Ref/Reference_Rank_Ave_CPM.tsv'
    ref_cpm = load_ref(ref_file)

    reader = pd.read_csv(input_file, sep='\t', chunksize=chunksize)

    header = True
    for i, chunk in enumerate(reader):
        print(f'Processing chunk {i + 1}...')

        chunk_index = chunk.iloc[:, 0]
        chunk_numeric = chunk.iloc[:, 1:].apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
        chunk_result = process_chunk(chunk_numeric, ref_cpm, n_jobs)
        chunk_result.insert(0, chunk.columns[0], chunk_index)

        chunk_result.to_csv(output_file, sep='\t', mode='w' if header else 'a',
                            header=header, index=False, float_format='%.2f')
        header = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rank-based CPM normalization using reference CPM.Ave values.')
    parser.add_argument('input_file', help='Input TSV file to normalize')
    parser.add_argument('output_file', help='Output file to save normalized results')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs (default: 4)')
    parser.add_argument('--chunksize', type=int, default=50000, help='Rows per chunk for processing (default: 50000)')

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.n_jobs, args.chunksize)
