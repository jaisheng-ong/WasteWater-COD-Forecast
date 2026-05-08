# wasteWater_model — Pre-trained Extra Trees Pipeline (PyCaret)

Predicts **AT_COD** (Aeration Tank Chemical Oxygen Demand) for wastewater discharge compliance monitoring.  
Trained on Kalman-smoothed + ARIMA-imputed sensor data (866 daily records, 90/10 chronological split).  
Best unseen R² achieved: **0.9548** (tuned Extra Trees Regressor via PyCaret).

---

## What's in this folder

| File | Description |
|---|---|
| `wasteWater_model.pkl` | Complete PyCaret pipeline — preprocessing transforms + trained Extra Trees model, all in one file |

PyCaret bundles the full preprocessing pipeline (imputation, scaling, etc.) together with the model into a single `.pkl` file, so no extra steps are needed before calling `predict_model`.

---

## Requirements

```
pycaret
pandas
```

---

## Steps to use

### 1. Make sure your dataset has the following 18 columns

| # | Column | Description |
|---|---|---|
| 1 | `AT_COD` | **Target** — Aeration tank COD (mg/L). Include for metric comparison; omit for blind prediction. |
| 2 | `AT_Inflow` | Aeration tank inflow rate |
| 3 | `AT_FM_Ratio` | Food-to-microorganism ratio |
| 4 | `AT_MLSS` | Mixed liquor suspended solids |
| 5 | `AT_Temp` | Aeration tank temperature |
| 6 | `AT_Aeration_Rate` | Aeration rate |
| 7 | `AT_Conductivity` | Conductivity in aeration tank |
| 8 | `ET_Phenol` | Effluent phenol concentration |
| 9 | `ET_OP` | Effluent orthophosphate |
| 10 | `ET_NH₄-N` | Effluent ammonium nitrogen |
| 11 | `ET_pH` | Effluent pH |
| 12 | `PWT_COD` | Pre-WAO tank COD |
| 13 | `PostWAO_COD` | Post-WAO tank COD |
| 14 | `PWT_Flow` | Pre-WAO tank flow |
| 15 | `WAO_Flow` | WAO unit flow |
| 16 | `BM_ratio` | Biomass ratio |
| 17 | `AT_CODKSp1` | Kalman-filter smoothed AT_COD prediction (step 1) |
| 18 | `AT_CODKSp2` | Kalman-filter smoothed AT_COD prediction (step 2) |

> **Note on `AT_CODKSp1` / `AT_CODKSp2`:** These are pre-computed Kalman-filter predictions of AT_COD used as engineered features. They must be generated from your AT_COD time series using the same Kalman + ARIMA pipeline before calling `predict_model`.

> **Note on `AT_COD`:** This is the target variable. You can include it (for evaluation/metric reporting) or leave it out entirely (for blind real-world prediction) — PyCaret handles both cases automatically.

### 2. Copy `wasteWater_model.pkl` to your working directory

```
your_project/
├── wasteWater_model.pkl
└── predict.py
```

### 3. Run the following code

```python
import pandas as pd
from pycaret.regression import load_model, predict_model

# ── Load the pipeline ─────────────────────────────────────────────────────────
loaded_model = load_model('wasteWater_model')

# ── Prepare your data ─────────────────────────────────────────────────────────
# CSV must have the columns listed above.
# AT_COD is optional — include it only if you want to compare against actuals.
unseen_data = pd.read_csv('your_new_data.csv')

# ── Predict ───────────────────────────────────────────────────────────────────
predNew = predict_model(loaded_model, data=unseen_data)
print(predNew[['AT_COD', 'prediction_label']].head(10))
```

### Output

`predict_model` returns the original DataFrame with one extra column appended:

| Column | Description |
|---|---|
| `prediction_label` | Model's predicted AT_COD (mg/L) |

All original columns (including `AT_COD` if provided) are preserved as-is.

---

## Model architecture

| Parameter | Value |
|---|---|
| Algorithm | Extra Trees Regressor |
| Optimization | PyCaret `tune_model` (random search) |
| PyCaret preprocessing | Mean imputation, standardisation |
| Train / Test split | 90 / 10 (chronological) |
| Input features | 17 (16 sensor readings + 2 Kalman features) |
| Target | AT_COD |
| Unseen R² | 0.9548 |
