import pandas as pd
import numpy as np
import argparse

# Log10 transform function for each chunk
def log_transform_chunk(chunk):
    numeric_chunk = chunk.iloc[:, 1:].astype(np.float32)
    transformed_chunk = np.log10(numeric_chunk + 1)
    return pd.concat([chunk.iloc[:, [0]], transformed_chunk], axis=1)

# Main processing function
def main(input_file, output_file, chunksize=50000):
    reader = pd.read_csv(input_file, sep='\t', chunksize=chunksize)

    header = True
    for i, chunk in enumerate(reader):
        print(f'Transforming chunk {i+1}...')
        transformed_chunk = log_transform_chunk(chunk)
        transformed_chunk.to_csv(output_file, sep='\t', mode='w' if header else 'a',
                                 header=header, index=False, float_format='%.4f')
        header = False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Log10 transform CPM matrix by chunk.')
    parser.add_argument('input_file', help='Input TSV file (CPM matrix)')
    parser.add_argument('output_file', help='Output TSV file for log10-transformed data')
    parser.add_argument('--chunksize', type=int, default=50000, help='Number of rows per chunk (default: 50000)')

    args = parser.parse_args()
    main(args.input_file, args.output_file, args.chunksize)
