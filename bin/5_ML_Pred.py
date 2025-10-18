import os
import pandas as pd
import joblib
import sys

# Parameters
ENCODED_FEATURES_FILE = sys.argv[1]  # User-defined parameter via command-line argument
outfile_name=sys.argv[2]
trunk_name=ENCODED_FEATURES_FILE.split('tsv')[0]
REF_DIR = "./Ref"

subfolders = ["a-SMA", "CD163", "IBA-1", "MAC387", "PAX7", "Semi-fibrosis-score", "TNF-a", "TZ_Percentage_Fibrosis"]

# Load encoded features
encoded_df = pd.read_csv(ENCODED_FEATURES_FILE, sep='\t', index_col=0)

compiled_predictions = pd.DataFrame(index=encoded_df.index)

for subfolder in subfolders:
    best_model_path = os.path.join(REF_DIR, subfolder, "Best_Model.txt")

    # Read best model name from Best_Model.txt
    with open(best_model_path, 'r') as f:
        best_model_name = f.read().strip().lower().replace(" ", "_")

    # Construct the actual model file name
    model_filename = f"final_{best_model_name}_model.pkl"

    # Map to the actual model file
    model_file = os.path.join(REF_DIR, subfolder, model_filename)

    if not os.path.exists(model_file):
        raise ValueError(f"Model file '{model_file}' not found for the best model '{best_model_name}'.")

    # Load the best model
    model = joblib.load(model_file)

    # Predict
    predictions = model.predict(encoded_df)

    # Save predictions (Off)
    #output_filename = f"Pred_value.{subfolder}.tsv"
    #predictions_df = pd.DataFrame(predictions, index=encoded_df.index, columns=[subfolder])
    #predictions_df.to_csv(output_filename, sep='\t')

    # Compile predictions
    compiled_predictions[subfolder] = predictions

# Save compiled predictions
compiled_predictions.reset_index().rename(columns={'index': 'Sample'}).to_csv(outfile_name, sep='\t', index=False)
