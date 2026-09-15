# Protein–Ligand Binding Affinity Prediction

An end-to-end machine learning project for predicting protein–ligand
binding affinity using pretrained protein and molecular representations.

The project compares two modeling approaches on the DAVIS drug–target
affinity dataset:

1. **ESM-2 + ChemBERTa + Fusion MLP**
2. **ESM-2 + Morgan Fingerprints + XGBoost**

For a selected target protein, the application predicts binding affinity
across the DAVIS compound library and compares predictions from both
models with the observed DAVIS binding affinity values.

## Research Question

> Do pretrained protein and molecular Transformer representations
> (ESM-2, ChemBERTa) improve drug–target binding affinity prediction
> over conventional cheminformatics features (Morgan fingerprints), when
> used as frozen feature extractors?

---

## TL;DR

The final experiment uses the full Davis dataset (25,772 drug–target
pairs) with frozen ESM-2 35M protein embeddings and ChemBERTa molecular
embeddings. Results were verified for reproducibility by fixing all
random seeds (Python, NumPy, PyTorch, CUDA) and re-running the full
pipeline twice, producing identical results to four decimal places.

The ESM-2 + ChemBERTa + Fusion MLP pipeline achieved a test RMSE of
**0.5222**, R² of **0.6051**, and Pearson correlation of **0.7780**. The
ESM-2 + Morgan fingerprint + XGBoost baseline achieved an RMSE of
**0.5423**, R² of **0.5740**, and Pearson correlation of **0.7593**. The
Fusion MLP pipeline wins on 4 of 5 metrics (MSE, RMSE, R², Pearson r),
with XGBoost holding a negligible edge on MAE (0.3425 vs. 0.3427).

Earlier experiments using the smaller ESM-2 8M encoder showed a clearer
result in the opposite direction, with the XGBoost baseline
outperforming the Transformer pipeline on every metric. Full staged
results, including this reversal, are documented below rather than only
the final number — see "Model Evaluation."

---

## Project Overview

Drug discovery often requires evaluating large numbers of potential
protein–ligand interactions before experimental validation.

This project explores whether pretrained biological and chemical
representations can be used to predict protein–ligand binding affinity
and prioritize candidate compounds.

The workflow combines:

- protein sequence representations from **ESM-2**
- molecular representations from **ChemBERTa**
- molecular fingerprints generated with **RDKit**
- neural-network regression with **PyTorch**
- gradient-boosted regression with **XGBoost**
- a **FastAPI** backend
- a **React** frontend for interactive model comparison

The goal is not to replace experimental validation, but to demonstrate
how machine learning can support computational prioritization of
candidate protein–ligand interactions.

---

## Dataset

The project uses the **DAVIS kinase–inhibitor binding affinity dataset**
via [Therapeutics Data Commons (TDC)](https://tdcommons.ai/).

The processed dataset contains:

- **25,772 drug–target pairs**
- **68 unique compounds**
- **379 unique target proteins**
- binding affinity represented as **pKd** (= -log10(Kd in M))

The dataset forms a complete matrix of the 68 compounds and 379 proteins.

### Key EDA findings

- No missing values or duplicate rows
- **Target variable is highly imbalanced**: >70% of samples sit at the
  detection floor (pKd ≈ 5, i.e. weak/non-binders), with a long sparse
  tail up to pKd ≈ 9.94 (strong binders)
- SMILES strings: all 68 valid (RDKit-confirmed), 32–81 characters —
  well within ChemBERTa's input limits
- Protein sequences: wide length range (244–2,549 amino acids, median
  632). 76/379 (~20%) exceed 1,000 residues
- One protein (TESK1) contains a single non-standard amino acid
  character ('X'), handled natively by ESM-2's tokenizer

### Decision: stratified sampling for early development, full dataset for final experiments

Initial pipeline development (Notebook 02) used a **stratified
4,237-pair subset** (rather than random sampling, which would produce an
almost entirely "non-binder" dataset) to validate the pipeline quickly
with a more balanced pKd distribution. The **final experiments**
(Notebook 03) scale up to the **complete 25,772-pair dataset**,
preserving the real-world class imbalance.

---

## Modeling Approaches

### Model 1 — ESM-2 + ChemBERTa + Fusion MLP

Protein sequences are represented using the pretrained **ESM-2 35M**
model. Each protein is represented by a 480-dimensional embedding
(mean-pooled over residue tokens, excluding special tokens).

Drug SMILES strings are represented using pretrained **ChemBERTa**
embeddings (768-dimensional, mean-pooled over tokens).

The two representations are concatenated:

```text
ESM-2 Protein Embedding       480
                +
ChemBERTa Drug Embedding      768
                ↓
Combined Feature Vector      1248
                ↓
        Fusion MLP (512 → 128 → 1)
                ↓
          Predicted pKd
```

The fusion model is implemented in PyTorch and trained with an
**unweighted MSE loss**, Adam optimizer with `ReduceLROnPlateau`
scheduling, mini-batch training (batch size 128), and early stopping.

An earlier iteration explored inverse-frequency sample weighting (to
counteract DAVIS's skew toward low-affinity observations), and that
weighted version was also fully trained — but the version that performed
best on the held-out validation set, and the one actually exported for
deployment, uses the plain (unweighted) MSE loss. This is deliberately
disclosed because it factors into how the results below should be
interpreted (see "What actually changed the outcome").

Both protein and molecular encoders are **frozen** (used purely as
feature extractors, not fine-tuned) — with only 379 unique proteins and
68 unique drugs, fine-tuning multi-million-parameter encoders would risk
severe overfitting. Embeddings are computed once per unique entity, not
once per pair, which is why encoding is inexpensive even as the dataset
scales to all 25,772 pairs.

### Model 2 — ESM-2 + Morgan Fingerprints + XGBoost

The second pipeline uses the same ESM-2 protein representations but
replaces ChemBERTa molecular embeddings with 1,024-bit Morgan
fingerprints generated from molecular structure.

```text
ESM-2 Protein Embedding       480
                +
Morgan Fingerprint           1024
                ↓
Combined Feature Vector      1504
                ↓
             XGBoost
                ↓
          Predicted pKd
```

This provides a conventional cheminformatics baseline for the complete
prediction pipeline. Because **both** the molecular representation
(ChemBERTa → Morgan) **and** the downstream regressor (Fusion MLP →
XGBoost) change simultaneously, this is a comparison between two
complete pipelines, not a controlled experiment isolating ChemBERTa vs.
Morgan fingerprints alone.

---

## Interactive Prediction Workflow

The web application is designed around a target-based screening
workflow. The user selects one of the DAVIS target proteins:

```text
                     Target Protein
                          AAK1
                            │
              ┌─────────────┴─────────────┐
              │                           │
              ▼                           ▼
       Transformer Pipeline         XGBoost Pipeline
              │                           │
       ESM-2 + ChemBERTa           ESM-2 + Morgan
              │                           │
          Fusion MLP                   XGBoost
              │                           │
              ▼                           ▼
       Predicted pKd               Predicted pKd
              │                           │
              └─────────────┬─────────────┘
                            │
                            ▼
                    DAVIS Observed pKd
                            │
                            ▼
                     Model Comparison
```

For each of the 68 DAVIS compounds tested against the selected protein,
the application displays:

| Compound | Transformer Prediction | XGBoost Prediction | DAVIS Observed pKd | Split |
|---|---:|---:|---:|---|
| Drug A | predicted | predicted | observed | train |
| Drug B | predicted | predicted | observed | test |
| Drug C | predicted | predicted | observed | validation |

**Why every row shows a "Split" label**: each (protein, drug) pair was
assigned to train/validation/test during model training. A pair labeled
"train" was seen by the model during training, so an accurate prediction
there is not evidence of generalization — it may simply reflect the
model having learned that specific pair. Only "test" pairs are
genuinely held-out. Both models were evaluated with identical split
indices for a fair comparison. The frontend lets the user toggle between
viewing all 68 compounds and viewing test-set-only compounds, so the
screening view and the honest evaluation view are both available without
hiding either one.

**Important nuance**: because Davis pairs already have measured pKd
values, showing "test" pairs specifically is what makes this comparison
meaningful — it demonstrates the model's ability to predict pairs it
never saw, not just its ability to recall training data.

---

## Model Evaluation

The dataset is divided into:

- **70% training**
- **15% validation**
- **15% test**

Both modeling pipelines use identical split indices for a fair
comparison. Performance is evaluated using MSE, RMSE, MAE, R²
(coefficient of determination), and Pearson correlation (r).

The current evaluation uses a **random drug–target pair split**, so the
same drug or target protein may occur across training, validation, and
test sets — only the specific (protein, drug) *combination* is guaranteed
to be held out for test rows. This evaluates interpolation across
observed drug and target identities rather than generalization to
completely unseen proteins or compounds.

### Full experimental progression

Results are reported in the order they were actually obtained, including
reversals, because the progression itself is part of the finding.

**Stage 1 — Development, stratified mini dataset (4,237 pairs), ESM-2 8M**

| Metric | Baseline (Morgan+XGB) | Transformer (ESM-2 8M+ChemBERTa) |
|---|---|---|
| RMSE ↓ | **0.8943** | 0.9264 |
| R² ↑ | **0.4725** | 0.4339 |
| Pearson r ↑ | **0.6912** | 0.6710 |

Baseline wins. Used to validate the pipeline end-to-end before scaling up.

**Stage 2 — Full dataset (25,772 pairs), ESM-2 8M, weighted loss**

| Metric | Baseline (Morgan+XGB) | Transformer (ESM-2 8M+ChemBERTa) |
|---|---|---|
| RMSE ↓ | **0.6046** | 0.6725 |
| R² ↑ | **0.4707** | 0.3451 |
| Pearson r ↑ | **0.7204** | 0.7042 |

Baseline still wins, by a wider margin.

**Stage 3 (Final) — Full dataset, ESM-2 35M, unweighted loss**

Upgrading the protein encoder from 8M → 35M parameters (480-dim vs.
320-dim embeddings) to test whether encoder capacity was the limiting
factor. Verified reproducible by re-running with fixed seeds twice
(identical results to 4 decimal places):

| Metric | Baseline (ESM-2 35M+Morgan+XGB) | Transformer (ESM-2 35M+ChemBERTa) |
|---|---|---|
| MSE ↓ | 0.2941 | **0.2727** |
| RMSE ↓ | 0.5423 | **0.5222** |
| MAE ↓ | **0.3425** | 0.3427 |
| R² ↑ | 0.5740 | **0.6051** |
| Pearson r ↑ | 0.7593 | **0.7780** |

**The Fusion MLP wins on 4 of 5 metrics**, with XGBoost holding a
negligible edge on MAE. This is the configuration exported for deployment.

### What actually changed the outcome

Both models improved substantially from Stage 2 → Stage 3. Looking at
R²: XGBoost improved by +0.10 (0.4707→0.5740), while the Fusion MLP
improved by +0.23 (0.3451→0.6051) — the Transformer pipeline's R²
improved *more* in absolute terms, reversing what had been a clear
baseline advantage into a clear Transformer advantage.

We're cautious about drawing a strong causal story from this. Stage 2 →
Stage 3 changed **two things at once** — the protein encoder (8M→35M)
*and* the loss function (weighted → unweighted) — so we cannot cleanly
attribute the Transformer's larger improvement to encoder capacity
alone. Disentangling these two factors (e.g., rerunning ESM-2 35M *with*
the weighted loss) is listed under Future Work.

---

## Repository Structure

```text
biobindai/
│
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb
│   ├── 02_embedding_and_training.ipynb          (Stage 1: mini dataset, ESM-2 8M)
│   └── 03_full_dataset_and_deployment_prep.ipynb (Stages 2–3: full dataset, ESM-2 8M→35M)
│
├── backend/
│   ├── main.py
│   ├── model.py
│   ├── schemas.py
│   ├── requirements.txt
│   │
│   ├── models/
│   │   ├── protein_embeddings_esm2_35m.pkl   (ESM-2 35M, 480-dim)
│   │   ├── drug_embeddings.pkl                (ChemBERTa, 768-dim)
│   │   ├── fusion_model_final.pt              (deployed model — Stage 3)
│   │   ├── scaler_final.pkl
│   │   ├── model_metadata.pkl
│   │   ├── xgb_model.json                     (baseline, used for comparison)
│   │   └── morgan_fingerprints.pkl
│   │
│   └── data/
│       ├── davis.csv        (all 25,772 pairs + train/val/test split label)
│       ├── drugs.csv
│       └── proteins.csv
│
├── frontend/
│
├── README.md
└── .gitignore
```

---

## Development Workflow

The project was developed in three stages.

### 1. Exploratory Data Analysis and Preprocessing

The first notebook examines the DAVIS dataset, including dataset
dimensions, unique proteins and compounds, pKd distribution, missing
values, duplicate observations, and affinity imbalance.

### 2. Model Development

The second notebook develops the initial protein–ligand prediction
workflow using ESM-2 (8M) protein embeddings, ChemBERTa molecular
embeddings, and a fusion neural network, on a smaller stratified
development sample (4,237 pairs).

### 3. Full Dataset Training and Model Comparison

The third notebook extends the workflow to all 25,772 DAVIS pairs and
includes: ESM-2 35M protein embedding generation, ChemBERTa molecular
representations, Fusion MLP training (both weighted- and unweighted-loss
variants were trained; the unweighted version was selected — see
Modeling Approaches), Morgan fingerprint generation, XGBoost training,
model evaluation with fixed-seed reproducibility verification, and
deployment artifact export.

---

## Deployment Architecture

```text
React Frontend
      │
      │  Target Protein
      ▼
FastAPI Backend
      │
      ├───────────────┐
      │               │
      ▼               ▼
Fusion MLP         XGBoost
      │               │
      ▼               ▼
Transformer       Fingerprint
Prediction        Prediction
      │               │
      └───────┬───────┘
              │
              ▼
   DAVIS Observed pKd + Split Label
              │
              ▼
        JSON Response
              │
              ▼
       React Comparison
```

Precomputed ESM-2, ChemBERTa, and Morgan representations are used for
the DAVIS proteins and compounds so that the deployed application can
perform inference without rerunning the pretrained encoders for every
request — both models run entirely on CPU at request time.

```
GET  /entities   → list of selectable proteins/drugs
POST /predict     → { target_id, drug_id } → predicted pKd (Fusion MLP)
POST /rank         → { target_id, drug_ids: [...] } → candidates ranked
                     by predicted pKd (virtual screening)
```

**Why the demo is restricted to known entities**: embeddings are only
precomputed for the 379 proteins / 68 drugs in the Davis dataset.
Supporting arbitrary user-supplied protein sequences or SMILES would
require running ESM-2/ChemBERTa inference on-demand, which is a heavier
deployment and is noted as future work.

---

## Technology Stack

**Machine Learning**: Python, PyTorch, XGBoost, scikit-learn
**Protein and Molecular Representation**: ESM-2, ChemBERTa, RDKit, Morgan fingerprints
**Data Analysis**: pandas, NumPy, SciPy
**Application**: FastAPI, React, Vite

---

## Limitations

This project is intended as a machine-learning and computational
drug-discovery demonstration.

- **The comparison is not fully controlled across stages.** Stage 2→3
  changed both the protein encoder and the loss function simultaneously
  (see "What actually changed the outcome").
- **Evaluation uses a random drug–target pair split**, not a
  cold-protein or cold-drug split. Drugs and proteins can therefore
  appear in more than one split — only the specific pair is held out for
  test rows. This evaluates interpolation, not generalization to
  genuinely unseen proteins or compounds — the single most important
  caveat for interpreting the reported metrics.
- **ESM-2 sequence encoding uses a maximum token length of 1,024**, so
  long protein sequences (~20% of proteins exceed 1,000 residues) may be
  truncated.
- **Frozen embeddings, not fine-tuned.** Both encoders were used purely
  as feature extractors.
- **The deployed screening library is limited to the 68 DAVIS
  compounds** and 379 DAVIS proteins.
- Predicted binding affinity does not establish biological activity,
  efficacy, selectivity, safety, or clinical utility. Computational
  predictions require downstream experimental validation.

## Future Work

- **Cold-protein / cold-drug splits** — train/test on fully disjoint
  sets of proteins or drugs, to measure genuine generalization rather
  than interpolation. This is the most important next step.
- **Isolate the encoder-capacity vs. loss-weighting confound** by
  rerunning ESM-2 35M with the weighted loss.
- Fine-tuning (or partial fine-tuning / LoRA) of ESM-2/ChemBERTa
- Attention-based explainability (highlighting which residues/atoms
  drive a prediction)
- Support for arbitrary user-supplied protein sequences/SMILES via
  on-demand Transformer inference


---

## Intended Use

This repository is a portfolio and research demonstration exploring
machine-learning approaches for protein–ligand binding affinity
prediction. The application is intended for educational and
computational research purposes and should not be used for clinical or
therapeutic decision-making.