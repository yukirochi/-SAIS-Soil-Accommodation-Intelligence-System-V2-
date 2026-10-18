"""
Step 2 - Train the ANFIS on anfis_train.csv and save anfis_model.json.
(Needs anfis.py in the same folder and anfis_train.csv from clean_dataset.py.)

Run:  python train.py
"""
import numpy as np
import pandas as pd
from anfis import ANFIS

INPUTS = ["deficit_pct_norm", "temp_c_norm", "humidity_pct_norm"]
TARGET = "liters_per_m2"

data = pd.read_csv("anfis_train.csv")
train_all = data[data.split == "train"]
test = data[data.split == "test"]

# Validation = the last 15% (by time) of the training rows, used for early stopping
cut = train_all.row_id.quantile(0.85)
train = train_all[train_all.row_id <= cut]
val = train_all[train_all.row_id > cut]

X_tr, y_tr = train[INPUTS].values, train[TARGET].values
X_va, y_va = val[INPUTS].values, val[TARGET].values
X_te, y_te = test[INPUTS].values, test[TARGET].values

model = ANFIS(n_inputs=3, n_mf=3, init_sigma=0.25)
model.fit(X_tr, y_tr, X_va, y_va, epochs=300, lr=0.01, patience=40)


def report(name, X, y):
    p = np.clip(model.predict(X), 0, None)
    mae = float(np.mean(np.abs(p - y)))
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    ss_res, ss_tot = np.sum((y - p) ** 2), np.sum((y - y.mean()) ** 2)
    print(f"{name:5s}  MAE {mae:.3f}  RMSE {rmse:.3f}  R2 {1 - ss_res / ss_tot:.4f}  (liters per m2)")


print()
report("train", X_tr, y_tr)
report("val", X_va, y_va)
report("test", X_te, y_te)

model.save("anfis_model.json")
print("\nSaved anfis_model.json (predict.py also needs scaler.json from clean_dataset.py)")
