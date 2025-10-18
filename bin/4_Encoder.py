import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import argparse

def main(input_file, output_file):
    # === Step 1: Load new input dataset ===
    df = pd.read_csv(input_file, sep="\t")
    sample_ids = df.iloc[:, 0]
    X = df.iloc[:, 1:].values

    # === Step 2: Reconstruct StandardScaler from saved parameters ===
    scaler_params = pd.read_csv("./Ref/scaler_parameters.tsv", sep="\t")
    scaler = StandardScaler()
    scaler.mean_ = scaler_params['mean'].values
    scaler.scale_ = scaler_params['scale'].values
    X_scaled = scaler.transform(X)

    # === Step 3: Load trained encoder model ===
    encoder = load_model("./Ref/encoder_model.h5")

    # === Step 4: Encode to latent space ===
    latent_features = encoder.predict(X_scaled)

    # === Step 5: Save latent features ===
    latent_df = pd.DataFrame(latent_features, columns=[f"Latent_{i+1}" for i in range(latent_features.shape[1])])
    latent_df.insert(0, "Sample", sample_ids)
    latent_df.to_csv(output_file, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode input data into latent space using pretrained encoder.")
    parser.add_argument("input_file", help="Path to log10-normalized input file (TSV)")
    parser.add_argument("output_file", help="Path to save encoded latent features (TSV)")
    args = parser.parse_args()

    main(args.input_file, args.output_file)


