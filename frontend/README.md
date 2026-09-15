# DAVIS Protein–Ligand Virtual Screening Frontend

Interactive React frontend for the **ML-Based Protein–Ligand Binding Affinity Prediction & Virtual Screening** project.

This application allows users to select a protein target from the DAVIS drug–target interaction dataset, screen candidate compounds using two machine learning pipelines, visualize molecular structures, and compare predicted binding affinities with experimentally observed values.

---

## Overview

The application predicts protein–ligand binding affinity, expressed as **pKd**, for protein–compound pairs in the DAVIS dataset.

Two machine learning pipelines are used:

1. **Fusion MLP** — the primary model combining protein and compound transformer embeddings.
2. **XGBoost Baseline** — a baseline pipeline combining protein embeddings with Morgan molecular fingerprints.

The interface ranks candidate compounds by predicted binding affinity to support computational prioritization for downstream experimental validation.

Higher pKd values indicate stronger binding affinity.

---

## Application Workflow

```text
Select Protein Target
        ↓
Screen DAVIS Compounds
        ↓
┌───────────────────────────────┐
│                               │
│        Fusion MLP             │
│                               │
│  Protein → ESM-2              │
│  Compound → ChemBERTa         │
│                               │
└──────────────┬────────────────┘
               │
               ↓
         Predicted pKd


┌───────────────────────────────┐
│                               │
│      XGBoost Baseline         │
│                               │
│  Protein → ESM-2              │
│  Compound → Morgan FP         │
│                               │
└──────────────┬────────────────┘
               │
               ↓
         Predicted pKd


               ↓

Compare Predictions
with DAVIS Observed pKd

               ↓

Rank Candidate Compounds
```

---

## Models

### Model 1 — Fusion MLP

The primary model combines learned representations of the protein sequence and compound SMILES.

**Protein representation**

- ESM-2 35M
- 480-dimensional protein embedding
- Protein sequence encoded using a pretrained protein language model

**Compound representation**

- ChemBERTa
- 768-dimensional compound embedding
- SMILES encoded using a pretrained chemical language model

The two representations are concatenated and passed through a multilayer perceptron.

```text
Protein Sequence
      ↓
ESM-2 35M
      ↓
480d Protein Embedding
                       \
                        \
                         → Fusion MLP → Predicted pKd
                        /
                       /
Compound SMILES
      ↓
ChemBERTa
      ↓
768d Compound Embedding
```

The combined Fusion MLP input contains:

```text
480 + 768 = 1248 features
```

---

### Model 2 — XGBoost Baseline

The baseline pipeline uses the same ESM-2 protein representation but replaces the learned compound representation with a Morgan molecular fingerprint.

**Protein representation**

- ESM-2 35M
- 480-dimensional protein embedding

**Compound representation**

- Morgan fingerprint
- 1024-dimensional molecular fingerprint

```text
Protein Sequence
      ↓
ESM-2 35M
      ↓
480d Protein Embedding
                       \
                        \
                         → XGBoost → Predicted pKd
                        /
                       /
Compound SMILES
      ↓
Morgan Fingerprint
      ↓
1024d Molecular Representation
```

The XGBoost input contains:

```text
480 + 1024 = 1504 features
```

The XGBoost pipeline is used as a baseline for comparison with the Fusion MLP.

Because both the compound representation and downstream prediction model differ between the two pipelines, this comparison should be interpreted as a **pipeline-level comparison**, rather than a controlled ChemBERTa-versus-Morgan ablation.

---

## How to Read the Results

For each protein–compound pair, the application displays three binding-affinity values.

### Fusion MLP pKd

The predicted binding affinity from the primary Fusion MLP model.

The prediction is generated using:

```text
ESM-2 Protein Embedding
+
ChemBERTa Compound Embedding
↓
Fusion MLP
↓
Predicted pKd
```

---

### XGBoost pKd

The predicted binding affinity from the XGBoost baseline.

The prediction is generated using:

```text
ESM-2 Protein Embedding
+
Morgan Fingerprint
↓
XGBoost
↓
Predicted pKd
```

---

### Observed pKd

The experimentally observed binding affinity reported in the DAVIS dataset.

This value serves as the reference against which model predictions are compared.

In general:

```text
Higher pKd
    ↓
Stronger Binding Affinity
```

---

## Prediction Error

For each compound, the interface also calculates the absolute prediction error.

```text
Absolute Error = |Predicted pKd - Observed pKd|
```

A smaller absolute error means that the predicted pKd is closer to the experimentally observed DAVIS value.

The interface calculates the error separately for:

- Fusion MLP
- XGBoost

These compound-level errors are useful for inspecting individual predictions.

Overall model performance should instead be evaluated using the held-out test-set metrics.

---

## Train / Validation / Test Splits

The protein–compound pairs were divided into three subsets.

### Train

Training pairs were used to fit the model parameters.

```text
TRAIN
↓
Used for model training
```

### Validation

Validation pairs were not used to directly fit model parameters.

They were used during model development for validation and model selection.

```text
VALIDATION
↓
Model development
and model selection
```

### Test

Test pairs were held out from training and model selection.

They are used for final model evaluation.

```text
TEST
↓
Final evaluation
```

The frontend displays the split assignment for every protein–compound pair.

---

## Important Evaluation Note

The project uses a **random protein–compound pair split**.

A test protein–compound pair was excluded from model training. However, the individual protein or compound may appear in other training pairs.

For example:

```text
Training:

Protein A + Compound 1
Protein A + Compound 2
Protein B + Compound 3


Test:

Protein A + Compound 3
```

The exact test pair:

```text
Protein A + Compound 3
```

was not used during training.

However, Protein A and Compound 3 may individually have appeared in other training pairs.

Therefore, test performance represents **held-out pair prediction / interpolation**, rather than generalization to completely unseen proteins or completely unseen compounds.

Future work could evaluate more challenging settings such as:

- Cold-protein split
- Cold-compound split
- Cold protein–compound split

---

## Model Performance

Final performance was evaluated on the held-out test set.

| Model | RMSE | R² | Pearson r |
|---|---:|---:|---:|
| Fusion MLP | 0.5222 | 0.6051 | 0.7780 |
| XGBoost Baseline | 0.5423 | 0.5740 | 0.7593 |

These metrics are calculated across the **full held-out test set**, rather than only the compounds associated with the protein currently selected in the frontend.

### Metric Interpretation

**RMSE — Root Mean Squared Error**

Measures the magnitude of prediction error.

Lower values indicate smaller prediction errors.

**R² — Coefficient of Determination**

Measures how much of the variation in observed pKd is captured by the model.

Higher values indicate greater explained variance.

**Pearson r**

Measures the linear correlation between predicted and observed pKd.

Higher values indicate stronger agreement in relative trends between predictions and observations.

---

## Screening Interface

The application allows the user to select a protein target and run both prediction pipelines across the DAVIS compounds associated with that target.

Each result card displays:

- 2D molecular structure
- Compound ID
- Dataset split
- Fusion MLP predicted pKd
- Fusion MLP absolute error
- XGBoost predicted pKd
- XGBoost absolute error
- DAVIS observed pKd

---

## Molecular Visualization

Compound structures are rendered directly in the browser from SMILES strings using **SmilesDrawer**.

```text
SMILES
  ↓
SmilesDrawer
  ↓
2D Molecular Structure
```

This allows the screening results to display both numerical predictions and the corresponding molecular structures.

---

## All Pairs and Test Only

The application provides two result views.

### All Pairs

Displays all DAVIS compounds associated with the selected protein target.

Each card indicates whether the protein–compound pair belongs to:

```text
Train
Validation
Test
```

This view is useful for exploring and ranking the available DAVIS compounds.

### Test Only

Displays only held-out test pairs.

This view is intended for inspecting predictions on pairs excluded from model training and model selection.

---

## Compound Ranking

Screening results can be ranked by:

- Fusion MLP predicted pKd
- XGBoost predicted pKd
- Observed DAVIS pKd

Predicted pKd ranking supports the virtual-screening use case:

```text
Selected Protein
      ↓
Predict Compound Affinity
      ↓
Sort by Predicted pKd
      ↓
Highest Predicted Affinity
      ↓
Candidate Prioritization
```

Observed pKd is provided for retrospective evaluation because DAVIS contains experimentally measured affinities.

In a prospective screening setting, observed affinity would not yet be available and compounds would instead be prioritized using model predictions for subsequent experimental testing.

---

## Frontend Features

The React application currently supports:

- Protein target selection
- FastAPI integration
- Screening of DAVIS compounds
- Fusion MLP predictions
- XGBoost baseline predictions
- Experimental DAVIS pKd comparison
- Absolute prediction errors
- Train / validation / test labels
- All-pairs view
- Test-only view
- Compound ranking
- 2D molecular visualization
- Held-out test-set performance summary
- Responsive compound-card interface

---

## Tech Stack

### Frontend

- React
- Vite
- JavaScript
- CSS
- SmilesDrawer

### Backend

The frontend communicates with a FastAPI inference service containing the trained machine learning pipelines.

Backend technologies include:

- Python
- FastAPI
- PyTorch
- XGBoost
- NumPy
- pandas
- scikit-learn

---

## API Integration

During local development, the frontend connects to:

```javascript
const API_URL = "http://127.0.0.1:8000";
```

The main API endpoints are:

```text
GET /proteins
GET /screen/{protein_id}
```

### `/proteins`

Returns the available DAVIS protein targets.

Example:

```text
GET /proteins
```

The frontend uses this endpoint to populate the protein selection menu.

### `/screen/{protein_id}`

Runs both prediction pipelines for the selected protein.

Example:

```text
GET /screen/AAK1
```

The response contains compound-level information including:

```text
Compound ID
SMILES
Fusion MLP prediction
XGBoost prediction
Observed pKd
Dataset split
```

---

## Local Development

### 1. Install Frontend Dependencies

From the frontend directory:

```bash
cd frontend
npm install
```

---

### 2. Start the React Development Server

```bash
npm run dev
```

Vite will normally start the frontend at:

```text
http://localhost:5173
```

---

### 3. Start the FastAPI Backend

From the project root, activate the Python environment and run:

```bash
python -m uvicorn backend.main:app
```

The backend will normally run at:

```text
http://127.0.0.1:8000
```

Both the frontend and backend must be running for local screening.

---

## Frontend Project Structure

```text
frontend/
│
├── src/
│   ├── App.jsx
│   ├── App.css
│   ├── MoleculeStructure.jsx
│   └── main.jsx
│
├── package.json
├── package-lock.json
├── vite.config.js
└── README.md
```

### `App.jsx`

Contains the main application logic, including:

- Protein loading
- Screening API calls
- Train/test filtering
- Result sorting
- Model performance display
- Compound cards
- Prediction error calculation

### `App.css`

Contains the responsive interface styling for:

- Project overview
- Model descriptions
- Evaluation explanation
- Protein selection
- Performance metrics
- Screening controls
- Molecular result cards

### `MoleculeStructure.jsx`

Converts SMILES strings into 2D molecular visualizations using SmilesDrawer.

---

## Relationship Between Frontend and Backend

The full application architecture is:

```text
┌─────────────────────────────────────────┐
│             React Frontend              │
│                                         │
│  Protein Selection                      │
│  Molecular Visualization                │
│  Ranking                                │
│  Model Comparison                       │
└──────────────────┬──────────────────────┘
                   │
                   │ HTTP API
                   ↓
┌─────────────────────────────────────────┐
│             FastAPI Backend             │
│                                         │
│  Cached Protein Embeddings              │
│  Cached Compound Representations        │
│  Fusion MLP                             │
│  XGBoost                                │
│  DAVIS Data                             │
└──────────────────┬──────────────────────┘
                   │
                   ↓
┌─────────────────────────────────────────┐
│          Screening Predictions          │
│                                         │
│  Fusion MLP pKd                         │
│  XGBoost pKd                            │
│  Observed DAVIS pKd                     │
│  Split                                  │
│  SMILES                                 │
└─────────────────────────────────────────┘
```

---

## Scope

The current interactive application is intentionally limited to the DAVIS dataset.

It supports:

- 379 DAVIS protein targets
- 68 DAVIS compounds

Protein and compound representations are cached for efficient inference.

The application is therefore a demonstration of an end-to-end machine learning workflow rather than a general-purpose drug-discovery platform.

---

## Limitations

Several limitations should be considered when interpreting the results.

### Random Pair Split

The current evaluation uses a random protein–compound pair split.

It does not directly measure generalization to completely unseen proteins or compounds.

### Frozen Encoders

ESM-2 and ChemBERTa are used as pretrained feature extractors rather than being fine-tuned jointly with the prediction model.

### Protein Sequence Length

Protein processing is subject to the token-length constraints of the ESM-2 representation pipeline.

### Dataset Scope

The interactive screening interface is restricted to proteins and compounds represented in DAVIS.

### Experimental Validation

Predicted binding affinities are computational estimates.

Candidate compounds identified by the model would require downstream experimental validation before biological or therapeutic conclusions could be made.

---

## Future Work

Potential extensions include:

- Cold-protein evaluation
- Cold-compound evaluation
- Cold protein–compound evaluation
- Additional drug–target interaction datasets
- Fine-tuning protein and molecular encoders
- Uncertainty estimation
- External validation datasets
- Expanded compound libraries
- Prospective virtual screening
- Additional molecular descriptors
- Model explainability
- Interactive prediction plots
- Protein metadata and annotations

---

## Purpose

This project demonstrates an end-to-end machine learning workflow connecting:

```text
Biological Data
      ↓
Protein Language Models
      +
Chemical Language Models
      ↓
Machine Learning
      ↓
Binding Affinity Prediction
      ↓
FastAPI Inference
      ↓
Interactive React Application
      ↓
Candidate Prioritization
```

The application is designed to demonstrate how machine learning predictions can be translated into an interactive decision-support workflow for computational protein–ligand screening.

Predictions are intended to support **computational prioritization of candidate compounds for downstream experimental validation** and should not be interpreted as experimentally validated drug-discovery results.