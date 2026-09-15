from pydantic import BaseModel


class ScreeningResult(BaseModel):
    drug_id: int
    smiles: str
    transformer_pkd: float
    xgboost_pkd: float
    actual_pkd: float
    split: str


class ScreeningResponse(BaseModel):
    protein_id: str
    n_compounds: int
    results: list[ScreeningResult]