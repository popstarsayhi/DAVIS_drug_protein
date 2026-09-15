# Transformer-Based Protein--Ligand Binding Affinity Prediction

Predicting drug--target binding affinity using pretrained protein and
molecular language models (ESM-2 + ChemBERTa), benchmarked against a
traditional cheminformatics baseline (Morgan Fingerprint + XGBoost), and
deployed as an API for candidate-ranking (virtual screening) demos.

## Research Question

> Do pretrained protein and molecular Transformer representations
> (ESM-2, ChemBERTa) improve drug--target binding affinity prediction
> over conventional cheminformatics features (Morgan fingerprints), when
> used as frozen feature extractors?

## Architecture at a Glance

    Protein sequence
          ↓
    Protein Transformer
    (ESM-2 35M, frozen)
          ↓
    Protein representation (480-dim) ─────┐
                                           ├── Concatenate → Fusion MLP → Binding Affinity
    Molecule SMILES                       │                              (pKd)
          ↓                               │
    Chemical Transformer ──────────────────┘
    (ChemBERTa, frozen, 768-dim)

## TL;DR

The final experiment uses the full Davis dataset (25,772 drug--target
pairs) with frozen ESM-2 35M protein embeddings and ChemBERTa molecular
embeddings.

The ESM-2 + ChemBERTa + Fusion MLP pipeline achieved a test RMSE of
**0.5227**, R² of **0.6044**, and Pearson correlation of **0.7775**. The
ESM-2 + Morgan fingerprint + XGBoost baseline achieved an RMSE of
**0.5423**, R² of **0.5740**, and Pearson correlation of **0.7593**. The
Fusion MLP pipeline performed better on RMSE, R², and Pearson
correlation, while XGBoost had a slightly lower MAE (0.3425 vs. 0.3453).

Earlier experiments using the smaller ESM-2 8M encoder showed the
opposite result, with the XGBoost baseline performing better on every
metric. After moving to the larger ESM-2 35M encoder, both pipelines
improved --- and notably, **the Fusion MLP's R² improved more (+0.26)
than XGBoost's (+0.10)** --- and the Fusion MLP became the stronger
overall model.

Because the experimental setup changed in more than one way between
stages (encoder size and loss weighting both changed), these results
should not be interpreted as proving that the larger ESM-2 encoder alone
caused the improvement, or attributed to one pipeline benefiting more
"because" of the encoder change in a strict causal sense. The
progression instead shows how sensitive model performance can be to
representation choice, training setup, and dataset scale --- full staged
results are below rather than only the final number.

------------------------------------------------------------------------

## 1. Dataset

-   **Source**: [Davis kinase binding dataset](https://tdcommons.ai/)
    via Therapeutics Data Commons (TDC)
-   **Task**: Regression --- predict pKd (binding affinity) for
    drug--protein pairs
-   **Full dataset**: 25,772 pairs \| 68 unique drugs \| 379 unique
    proteins (complete 68×379 matrix)
-   **Target transform**: Kd → pKd = -log10(Kd in M), standard practice
    for this task, compresses a wide dynamic range into a more learnable
    scale

### Key EDA findings

-   No missing values or duplicate rows
-   **Target variable is highly imbalanced**: \>70% of samples sit at
    the detection floor (pKd ≈ 5, i.e. weak/non-binders), with a long
    sparse tail up to pKd ≈ 9.94 (strong binders)
-   SMILES strings: all 68 valid (RDKit-confirmed), 32--81 characters
    --- well within ChemBERTa's input limits
-   Protein sequences: wide length range (244--2,549 amino acids, median
    632). 76/379 (\~20%) exceed 1,000 residues
-   One protein (TESK1) contains a single non-standard amino acid
    character ('X'), handled natively by ESM-2's tokenizer

### Decision: stratified sampling for early development, full dataset for final experiments

Initial pipeline development (Notebook 02) used a **stratified
4,237-pair subset** (rather than random sampling, which would produce an
almost entirely "non-binder" dataset, or oversampling, which would
duplicate the same \~143 strong-binding examples and risk overfitting)
to validate the pipeline quickly with a more balanced pKd distribution.
The **final experiments** (Notebook 03) scale up to the **complete
25,772-pair dataset**, preserving the real-world class imbalance.

### Note on protein sequence length

ESM-2 embedding extraction used `max_length=1024` (with special tokens
excluded from mean pooling). Protein sequences exceeding this length
were truncated. This is disclosed rather than hidden --- see
Limitations.

------------------------------------------------------------------------

## 2. Model Architecture

    Protein Sequence                    Drug SMILES
          │                                  │
          ▼                                  ▼
       ESM-2                            ChemBERTa
    (esm2_t12_35M_UR50D)             (ChemBERTa-zinc-base-v1)
       FROZEN                              FROZEN
          │                                  │
          ▼                                  ▼
    Protein Embedding                 Ligand Embedding
       (480-dim)                         (768-dim)
          │                                  │
          └────────────────┬─────────────────┘
                            ▼
                      Concatenate
                       (1248-dim)
                            │
                            ▼
                  Fusion MLP (trainable)
                  1248 → 512 → 128 → 1
                            │
                            ▼
                  Predicted pKd

**Why frozen Transformers?**

-   Only 379 unique proteins / 68 unique drugs --- far too few to safely
    fine-tune multi-million-parameter encoders without severe
    overfitting
-   Embeddings are computed **once per unique entity**, not once per
    drug-target pair --- a large reduction in Transformer inference
    calls regardless of how many pairs are used for downstream training
-   Frozen inference is cheap enough to run on a single Colab T4 GPU in
    minutes

``` python
from transformers import AutoTokenizer, AutoModel
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Protein encoder (frozen) — ESM-2 35M
esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
esm_model = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D").to(device)
esm_model.eval()
for p in esm_model.parameters():
    p.requires_grad = False

# Ligand encoder (frozen) — ChemBERTa
chem_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
chem_model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
chem_model.eval()
for p in chem_model.parameters():
    p.requires_grad = False

# Protein embedding: mean pooling over residues, excluding special tokens
def get_protein_embedding(seq):
    inputs = esm_tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        out = esm_model(**inputs)
    residue_embeddings = out.last_hidden_state[:, 1:-1, :]  # drop [CLS]/[EOS]
    return residue_embeddings.mean(dim=1).squeeze(0).cpu().numpy()

# Drug embedding: mean pooling over tokens
def get_drug_embedding(smiles):
    inputs = chem_tokenizer(smiles, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        out = chem_model(**inputs)
    return out.last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
```

### Fusion MLP

``` python
import torch.nn as nn

class FusionMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x.squeeze(-1)
```

Trained with plain MSE loss, Adam optimizer (lr=1e-3) with
`ReduceLROnPlateau` scheduling, mini-batch gradient descent (batch size
128), and early stopping on validation loss (patience=30).

------------------------------------------------------------------------

## 3. Baseline Model

For comparison, we built a second pipeline using the same ESM-2 protein
representation but replacing ChemBERTa molecular embeddings with
1,024-bit Morgan fingerprints, combined with an XGBoost regressor
instead of the Fusion MLP.

This provides a useful conventional cheminformatics baseline for the
complete prediction pipeline. However, because **both** the molecular
representation (ChemBERTa → Morgan) **and** the downstream regressor
(Fusion MLP → XGBoost) change simultaneously, this is a comparison
between two complete pipelines, not a controlled experiment isolating
ChemBERTa vs. Morgan fingerprints alone --- a caveat worth keeping in
mind when interpreting which component drives the difference.

``` python
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import xgboost as xgb

def get_morgan_fingerprint(smiles, n_bits=1024, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

xgb_model = xgb.XGBRegressor(
    n_estimators=800, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, early_stopping_rounds=30, eval_metric='rmse'
)
xgb_model.fit(X_baseline_train, y_baseline_train,
              eval_set=[(X_baseline_val, y_baseline_val)])
```

------------------------------------------------------------------------

## 4. Results --- Full Experimental Progression

Results are reported in the order they were actually obtained, including
reversals, because the progression itself is part of the finding --- a
single final table would hide how sensitive the conclusion is to encoder
capacity and dataset scale.

### Stage 1 --- Development, stratified mini dataset (4,237 pairs), ESM-2 8M

------------------------------------------------------------------------------

  Metric Baseline (Morgan+XGB) Transformer (ESM-2                       
  8M+ChemBERTa)                                                         

-------------------------------------------------------- ------------ --------

  RMSE ↓                                                   **0.8943**   0.9264

  R² ↑                                                     **0.4725**   0.4339

  Pearson r ↑                                              **0.6912**   0.6710
  ------------------------------------------------------------------------------

Baseline wins. Used to validate the pipeline end-to-end before scaling
up.

### Stage 2 --- Full dataset (25,772 pairs), ESM-2 8M, weighted loss

Scaling to the full (imbalanced) dataset required a weighted MSE loss
(capped inverse-frequency weighting by pKd bin) so the model couldn't
trivially predict "everything is \~5".

------------------------------------------------------------------------------

  Metric Baseline (Morgan+XGB) Transformer (ESM-2                       
  8M+ChemBERTa)                                                         

-------------------------------------------------------- ------------ --------

  RMSE ↓                                                   **0.6046**   0.6725

  R² ↑                                                     **0.4707**   0.3451

  Pearson r ↑                                              **0.7204**   0.7042
  ------------------------------------------------------------------------------

Baseline still wins, by a wider margin than Stage 1.

### Stage 3 (Final) --- Full dataset, ESM-2 35M, unweighted loss

Upgrading the protein encoder from 8M → 35M parameters (480-dim vs.
320-dim embeddings) to test whether encoder capacity was the limiting
factor:

-----------------------------------------------------------------------

  Metric                  Baseline (ESM-2 35M +   Fusion MLP (ESM-2 35M +
                              Morgan + XGBoost)                ChemBERTa)

------------------- ------------------------- -------------------------

  MSE ↓                                  0.2941                **0.2732**

  RMSE ↓                                 0.5423                **0.5227**

  MAE ↓                              **0.3425**                    0.3453

  R² ↑                                   0.5740                **0.6044**

  Pearson r ↑                            0.7593                **0.7775**
  -----------------------------------------------------------------------

**The Fusion MLP pipeline wins on 4 of 5 metrics.** This is the
configuration that was exported for deployment.

### What actually changed the outcome

Both models improved substantially from Stage 2 → Stage 3. Looking at
R²: XGBoost improved by +0.10 (0.4707→0.5740), while the Fusion MLP
improved by +0.26 (0.3451→0.6044) --- the Transformer pipeline's R²
improved *more* in absolute terms. This is worth stating precisely
because it would be easy to misread the reversal as "the bigger encoder
mainly helped the baseline" --- the numbers say the opposite for R² (MAE
is the one metric where XGBoost keeps a slim edge in Stage 3).

That said, we're cautious about drawing a strong causal story from this.
Stage 2 → Stage 3 changed **two things at once** --- the protein encoder
(8M→35M) *and* the loss function (weighted → unweighted) --- so we
cannot cleanly attribute the Transformer's larger improvement to encoder
capacity alone; the removed loss weighting could also be doing some of
the work, especially since weighting changes how errors on rare
high-affinity samples are penalized. Disentangling these two factors
(e.g., rerunning ESM-2 35M *with* the weighted loss) is the natural next
step and is listed under Future Work.

------------------------------------------------------------------------

## 5. Discussion & Limitations

-   **The comparison is not fully controlled across stages.** Stage 3
    changed two things at once relative to Stage 2 --- protein encoder
    (8M→35M) *and* removed the weighted loss --- so we cannot cleanly
    attribute the Transformer's improvement to encoder capacity alone. A
    cleaner ablation (35M encoder *with* weighted loss) is natural
    follow-up work.
-   **Protein sequences were truncated at 1,024 residues** for ESM-2
    input. \~20% of proteins exceed 1,000 residues in the raw data, so
    some information loss is possible for those sequences.
-   **Evaluation used a random pair-level split**, not a cold-protein or
    cold-drug split. Because the same protein or drug can appear in both
    train and test, these results measure interpolation within the
    observed Davis chemical/protein space rather than generalization to
    genuinely unseen targets or compounds --- a materially easier task
    than prospective screening. This is the single most important caveat
    for interpreting the reported metrics.
-   **Frozen embeddings, not fine-tuned.** Both encoders were used
    purely as feature extractors; task-specific fine-tuning was not
    attempted here and could change results in either direction.

## 6. Future Work

-   **Cold-protein / cold-drug splits** --- train/test on fully disjoint
    sets of proteins or drugs, to measure genuine generalization rather
    than interpolation. This is the most important next step: a model
    that performs well on a random split may still fail on genuinely
    novel targets, which is the realistic screening scenario.
-   **Isolate the encoder-capacity vs. loss-weighting confound** from
    Stage 2→3 by rerunning ESM-2 35M with the weighted loss.
-   Fine-tuning (or partial fine-tuning / LoRA) of ESM-2/ChemBERTa
-   Attention-based explainability (highlighting which residues/atoms
    drive a prediction)

------------------------------------------------------------------------

## 7. Deployment

The final Stage 3 model (ESM-2 35M + ChemBERTa + Fusion MLP) is served
via a FastAPI backend for candidate-ranking demos. Precomputed
embeddings for all 379 proteins and 68 drugs are cached, so inference is
CPU-only and does not re-run either Transformer at request time.

    GET  /entities        → list of selectable proteins/drugs (demo is
                             restricted to entities seen during training —
                             see note below)
    POST /predict          → { target_id, drug_id } → predicted pKd
    POST /rank              → { target_id, drug_ids: [...] } → candidates
                              ranked by predicted pKd (virtual screening)

**Why the demo is restricted to known entities**: embeddings are only
precomputed for the 379 proteins / 68 drugs in the Davis dataset.
Supporting arbitrary user-supplied protein sequences or SMILES would
require running ESM-2/ChemBERTa inference on-demand, which is a heavier
deployment (larger memory footprint, slower cold starts) and is noted as
future work rather than implemented here.

**Important nuance**: because Davis pairs already have measured pKd
values, the demo's value is in visualizing the model's ranking ability
on **held-out test pairs** (i.e., combinations the model did not see
during training) rather than "predicting" already-known answers --- the
ranking endpoint is most meaningful when restricted to the test split.

------------------------------------------------------------------------

## Repository Structure

    biobindai/
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    ├── data/
    │   └── davis_mini.csv
    ├── notebooks/
    │   ├── 01_eda_and_preprocessing.ipynb
    │   ├── 02_embedding_and_training.ipynb        (Stage 1: mini dataset, ESM-2 8M)
    │   └── 03_full_dataset_and_deployment_prep.ipynb  (Stages 2–3: full dataset, ESM-2 8M→35M)
    ├── assets/
    │   ├── predicted_vs_actual_comparison.png
    │   └── per_bin_rmse_comparison.png
    └── backend/
        ├── main.py
        ├── requirements.txt
        └── model_artifacts/
            ├── fusion_model_final.pt
            ├── scaler_final.pkl
            ├── model_metadata.pkl
            ├── protein_embeddings.pkl      (ESM-2 35M, 480-dim)
            ├── drug_embeddings.pkl         (ChemBERTa, 768-dim)
            └── selectable_entities.json

## Tech Stack

Python, PyTorch, HuggingFace Transformers (ESM-2, ChemBERTa), RDKit,
XGBoost, scikit-learn, FastAPI, pandas, TDC (Therapeutics Data Commons)