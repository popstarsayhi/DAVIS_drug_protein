import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"


# --------------------------------------------------
# Fusion MLP architecture
# --------------------------------------------------

class FusionMLP(nn.Module):
    def __init__(self, input_dim=1248):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.fc3(x)

        return x.squeeze(1)


# --------------------------------------------------
# Load metadata
# --------------------------------------------------

with open(MODEL_DIR / "model_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

input_dim = metadata["input_dim"]


# --------------------------------------------------
# Load shared protein embeddings
# --------------------------------------------------

with open(
    MODEL_DIR / "protein_embeddings_esm2_35m.pkl",
    "rb"
) as f:
    protein_embeddings = pickle.load(f)


# --------------------------------------------------
# Load ChemBERTa drug embeddings
# --------------------------------------------------

with open(
    MODEL_DIR / "drug_embeddings.pkl",
    "rb"
) as f:
    drug_embeddings = pickle.load(f)


# --------------------------------------------------
# Load Morgan fingerprints
# --------------------------------------------------

with open(
    MODEL_DIR / "morgan_fingerprints.pkl",
    "rb"
) as f:
    morgan_fps = pickle.load(f)


# --------------------------------------------------
# Load scaler
# --------------------------------------------------

with open(
    MODEL_DIR / "scaler_final.pkl",
    "rb"
) as f:
    scaler = pickle.load(f)


# --------------------------------------------------
# Load Fusion MLP
# --------------------------------------------------

device = torch.device("cpu")

transformer_model = FusionMLP(
    input_dim=input_dim
)

state_dict = torch.load(
    MODEL_DIR / "fusion_model_final.pt",
    map_location=device
)

transformer_model.load_state_dict(
    state_dict
)

transformer_model.to(device)
transformer_model.eval()


# --------------------------------------------------
# Load XGBoost model
# --------------------------------------------------

xgb_model = xgb.XGBRegressor()

xgb_model.load_model(
    MODEL_DIR / "xgb_model.json"
)

xgb_model.set_params(n_jobs=1)

# --------------------------------------------------
# Load DAVIS observed data
# --------------------------------------------------

davis_df = pd.read_csv(
    DATA_DIR / "davis.csv"
)


# --------------------------------------------------
# Transformer prediction
# ESM-2 + ChemBERTa + Fusion MLP
# --------------------------------------------------

def predict_transformer(protein_id, drug_id):

    if protein_id not in protein_embeddings:
        raise ValueError(
            f"Unknown protein ID: {protein_id}"
        )

    if drug_id not in drug_embeddings:
        raise ValueError(
            f"Unknown drug ID: {drug_id}"
        )

    protein_vector = np.asarray(
        protein_embeddings[protein_id],
        dtype=np.float32
    )

    drug_vector = np.asarray(
        drug_embeddings[drug_id],
        dtype=np.float32
    )

    features = np.concatenate([
        protein_vector,
        drug_vector
    ]).reshape(1, -1)

    features_scaled = scaler.transform(
        features
    )

    features_tensor = torch.tensor(
        features_scaled,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        prediction = transformer_model(
            features_tensor
        ).item()

    return float(prediction)


# --------------------------------------------------
# XGBoost prediction
# ESM-2 + Morgan + XGBoost
# --------------------------------------------------

def predict_xgboost(protein_id, drug_id):

    if protein_id not in protein_embeddings:
        raise ValueError(
            f"Unknown protein ID: {protein_id}"
        )

    if drug_id not in morgan_fps:
        raise ValueError(
            f"Unknown drug ID: {drug_id}"
        )

    protein_vector = np.asarray(
        protein_embeddings[protein_id],
        dtype=np.float32
    )

    drug_vector = np.asarray(
        morgan_fps[drug_id],
        dtype=np.float32
    )

    features = np.concatenate([
        protein_vector,
        drug_vector
    ]).reshape(1, -1)

    prediction = xgb_model.predict(
        features
    )[0]

    return float(prediction)


# --------------------------------------------------
# Compare both models with DAVIS observed values
# --------------------------------------------------

def screen_protein(protein_id):

    if protein_id not in protein_embeddings:
        raise ValueError(
            f"Unknown protein ID: {protein_id}"
        )

    protein_data = davis_df[
        davis_df["Target_ID"] == protein_id
    ]

    results = []

    for _, row in protein_data.iterrows():

        drug_id = row["Drug_ID"]

        transformer_pkd = predict_transformer(
            protein_id,
            drug_id
        )

        xgboost_pkd = predict_xgboost(
            protein_id,
            drug_id
        )

        actual_pkd = float(
            row["Y"]
        )

        results.append({
            "drug_id": int(row["Drug_ID"]),
            "smiles": str(row["Drug"]),
            "transformer_pkd": transformer_pkd,
            "xgboost_pkd": xgboost_pkd,
            "actual_pkd": float(row["Y"]),
            "split": str(row["split"]),
        })

    # Rank by observed DAVIS pKd for comparison
    results.sort(
        key=lambda x: x["transformer_pkd"],
        reverse=True
    )

    return results


# --------------------------------------------------
# Quick test
# --------------------------------------------------

if __name__ == "__main__":

    print("Models loaded successfully.")
    print("Proteins:", len(protein_embeddings))
    print("Drugs:", len(drug_embeddings))
    print("DAVIS pairs:", len(davis_df))

    protein_id = "AAK1"

    results = screen_protein(
        protein_id
    )

    print(
        f"\nComparison for {protein_id}"
    )

    print(
        f"Compounds: {len(results)}\n"
    )

    print(
        f"{'Rank':<6}"
        f"{'Drug':<12}"
        f"{'Transformer':<15}"
        f"{'XGBoost':<12}"
        f"{'DAVIS':<10}"
    )

    print("-" * 55)

    for rank, result in enumerate(
        results[:10],
        start=1
    ):
        print(
            f"{rank:<6}"
            f"{str(result['drug_id']):<12}"
            f"{result['transformer_pkd']:<15.3f}"
            f"{result['xgboost_pkd']:<12.3f}"
            f"{result['actual_pkd']:<10.3f}"
        )