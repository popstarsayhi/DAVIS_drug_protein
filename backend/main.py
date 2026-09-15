from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.model import (
    protein_embeddings,
    screen_protein,
    predict_transformer,
    predict_xgboost,
)
from backend.schemas import ScreeningResponse


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="Protein-Ligand Binding Affinity API",
    version="1.0"
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Home
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Protein-Ligand Binding Affinity API"
    }


# --------------------------------------------------
# Available proteins
# --------------------------------------------------

@app.get("/proteins")
def get_proteins():

    proteins = sorted(
        protein_embeddings.keys()
    )

    return {
        "n_proteins": len(proteins),
        "proteins": proteins
    }


# --------------------------------------------------
# Screen all DAVIS compounds for one protein
# --------------------------------------------------

@app.get(
    "/screen/{protein_id}",
    response_model=ScreeningResponse
)
async def screen(protein_id: str):

    if protein_id not in protein_embeddings:
        raise HTTPException(
            status_code=404,
            detail=f"Protein '{protein_id}' not found."
        )

    results = screen_protein(
        protein_id
    )

    return {
        "protein_id": protein_id,
        "n_compounds": len(results),
        "results": results
    }


# --------------------------------------------------
# Diagnostic test: Transformer only
# --------------------------------------------------

@app.get("/test-transformer")
async def test_transformer():

    prediction = predict_transformer(
        "AAK1",
        11427553
    )

    return {
        "protein_id": "AAK1",
        "drug_id": 11427553,
        "prediction": prediction
    }


# --------------------------------------------------
# Diagnostic test: XGBoost only
# --------------------------------------------------

@app.get("/test-xgboost")
async def test_xgboost():

    prediction = predict_xgboost(
        "AAK1",
        11427553
    )

    return {
        "protein_id": "AAK1",
        "drug_id": 11427553,
        "prediction": prediction
    }