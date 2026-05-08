# wasteWater_lstm — Pre-trained LSTM Pipeline

Predicts **AT_COD** (Aeration Tank Chemical Oxygen Demand) for wastewater discharge compliance monitoring.  
Trained on Kalman-smoothed + ARIMA-imputed sensor data (962 daily records, 90/10 chronological split).  
Best unseen R² achieved: **0.9462** (75-trial Optuna search).

---

## What's in this folder

| File | Description |
|---|---|
| `pipeline.pkl` | Fitted `MinMaxScaler` objects + all pipeline metadata (joblib) |
| `lstm_model.keras` | Trained Keras LSTM weights and architecture |

Both files must be present in the same folder for the pipeline to load correctly.

---

## Requirements

```
tensorflow
keras
scikit-learn
pandas
numpy
joblib
```

---

## Steps to use

### 1. Make sure your dataset has the following 18 columns (in this exact order)

| # | Column | Description |
|---|---|---|
| 1 | `AT_Inflow` | Aeration tank inflow rate |
| 2 | `AT_FM_Ratio` | Food-to-microorganism ratio |
| 3 | `AT_MLSS` | Mixed liquor suspended solids |
| 4 | `AT_Temp` | Aeration tank temperature |
| 5 | `AT_Aeration_Rate` | Aeration rate |
| 6 | `AT_Conductivity` | Conductivity in aeration tank |
| 7 | `ET_Phenol` | Effluent phenol concentration |
| 8 | `ET_OP` | Effluent orthophosphate |
| 9 | `ET_NH₄-N` | Effluent ammonium nitrogen |
| 10 | `ET_pH` | Effluent pH |
| 11 | `PWT_COD` | Pre-WAO tank COD |
| 12 | `PostWAO_COD` | Post-WAO tank COD |
| 13 | `PWT_Flow` | Pre-WAO tank flow |
| 14 | `WAO_Flow` | WAO unit flow |
| 15 | `BM_ratio` | Biomass ratio |
| 16 | `AT_CODKSp1` | Kalman-smoothed AT_COD lag feature 1 |
| 17 | `AT_CODKSp2` | Kalman-smoothed AT_COD lag feature 2 |
| 18 | `AT_COD` | **Target — must be the last column.** Fill with `0` if unknown. |

> **Note:** The pipeline uses a lag window of 1 day (`n_in=1`), so it needs at least **2 rows** to produce a prediction. The first row is always dropped during windowing.

### 2. Copy `pipeline.pkl` and `lstm_model.keras` to your working directory

```
your_project/
├── wasteWater_lstm/
│   ├── pipeline.pkl
│   └── lstm_model.keras
└── predict.py
```

### 3. Run the following code

```python
import pandas as pd
import joblib
from keras.models import load_model as _keras_load

# ── Paste the load_model helper (or import it from LSTM.ipynb) ───────────────
def load_model(name):
    pipeline        = joblib.load(f'{name}/pipeline.pkl')
    pipeline.model_ = _keras_load(f'{name}/lstm_model.keras')
    print(f"Transformation pipeline and model loaded from '{name}/'")
    return pipeline

# ── Load the pipeline ─────────────────────────────────────────────────────────
loaded_model = load_model('wasteWater_lstm')

# ── Prepare your data ─────────────────────────────────────────────────────────
# CSV must have the 18 columns listed above, in order, with AT_COD last.
# If AT_COD is unknown for a row, set it to 0.
unseen_data = pd.read_csv('your_new_data.csv')

# ── Predict ───────────────────────────────────────────────────────────────────
predNew = loaded_model.predict(unseen_data)
print(predNew[['AT_COD', 'prediction_label']].head(10))
```

### Output

The `predict()` call returns a DataFrame with two columns:

| Column | Description |
|---|---|
| `AT_COD` | Actual value from your input (ground truth if known, else `0`) |
| `prediction_label` | Model's predicted AT_COD (mg/L) |

---

## Model architecture

| Parameter | Value |
|---|---|
| Architecture | LSTM → Dense(1) |
| LSTM units | 89 |
| Epochs | 115 |
| Batch size | 32 |
| Loss function | MAE |
| Lag window (`n_in`) | 1 day |
| Train / Test split | 90 / 10 (chronological) |
| Unseen R² | 0.9462 |
