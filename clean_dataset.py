"""
Step 1 - Clean the raw dataset and build the ANFIS training data.

Input : correlated_soil_moisture_weather_300.csv
Output: anfis_train.csv  (normalized inputs + target, with a time-ordered split)
        scaler.json      (min/max used for scaling; reused at prediction time)

Model inputs : deficit_pct, temp_c, humidity_pct
Model target : liters_per_m2   (multiply by a node's area to get liters)

Run:  python clean_dataset.py [path/to/raw.csv]
"""
import json
import sys
import numpy as np
import pandas as pd

# ----------------------------------------------------------------- CONFIG ---
RAW_CSV = sys.argv[1] if len(sys.argv) > 1 else "correlated_soil_moisture_weather_300.csv"
COL_MOISTURE = "Soil_Moisture_%"            # already in percent (no calibration needed)
COL_TEMP = "Atmospheric_Temperature_C"
COL_HUM = "Atmospheric_Humidity_%"

# Soil/crop assumptions - replace with your real values
ROOT_DEPTH_MM = 200.0     # root zone depth
AWC = 0.15                # available water capacity (mm water per mm soil)
EFFICIENCY = 0.85         # drip ~0.85, sprinkler ~0.70
ALPHA_VPD = 0.10          # extra water per kPa of vapor pressure deficit

# Plausible threshold range (% moisture) per node. Your data runs ~11-35 %, so
# thresholds are set inside that band. Real thresholds come from your
# post-deployment automation (calibrate_threshold.py); these only teach the model
# how liters respond to the deficit.
NODE_THRESHOLD_RANGE = {
    "node1": (15, 25),
    "node2": (18, 28),
    "node3": (20, 30),
    "node4": (22, 32),
}
COPIES_PER_NODE = 3       # thresholds sampled per row per node (augmentation)
TRAIN_FRACTION = 0.8      # first 80% of rows (time order) = train, rest = test
SEED = 42
# ------------------------------------------------------------------------- #

rng = np.random.default_rng(SEED)


def vpd_kpa(temp_c, rh_pct):
    """Vapor pressure deficit: how strongly the air pulls water from the soil."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return es * (1 - rh_pct / 100.0)


def liters_per_m2(deficit_pct, temp_c, rh_pct):
    """Target: water to bring the soil back up to the threshold (1 mm = 1 L/m2).
    Equals 0 when deficit is 0."""
    refill_mm = deficit_pct / 100.0 * ROOT_DEPTH_MM * AWC / EFFICIENCY
    return refill_mm * (1 + ALPHA_VPD * vpd_kpa(temp_c, rh_pct))


# 1. Load and clean ----------------------------------------------------------
df = pd.read_csv(RAW_CSV)
df = df.rename(columns={COL_MOISTURE: "moisture_pct", COL_TEMP: "temp_c",
                        COL_HUM: "humidity_pct"})
df = df[["moisture_pct", "temp_c", "humidity_pct"]].dropna()
df = df[df.moisture_pct.between(0, 100) & df.humidity_pct.between(0, 100)
        & df.temp_c.between(-10, 60)].reset_index(drop=True)
df["row_id"] = df.index          # preserves time order for the split

# 2. Simulate node thresholds, then compute deficit and target ---------------
parts = []
for node, (lo, hi) in NODE_THRESHOLD_RANGE.items():
    for _ in range(COPIES_PER_NODE):
        p = df.copy()
        p["node"] = node
        p["threshold_pct"] = rng.uniform(lo, hi, size=len(p))
        p["deficit_pct"] = (p.threshold_pct - p.moisture_pct).clip(lower=0)
        p["liters_per_m2"] = liters_per_m2(p.deficit_pct, p.temp_c, p.humidity_pct)
        parts.append(p)
data = pd.concat(parts).sort_values(["row_id", "node"]).reset_index(drop=True)

# 3. Split by time (by original row, so no sensor row leaks across splits) ---
n_train_rows = int(len(df) * TRAIN_FRACTION)
data["split"] = np.where(data.row_id < n_train_rows, "train", "test")

# 4. Normalize to 0-1 using the TRAIN part only ------------------------------
features = ["deficit_pct", "temp_c", "humidity_pct"]
train_part = data[data.split == "train"]
scaler = {f: {"min": float(train_part[f].min()), "max": float(train_part[f].max())}
          for f in features}
scaler["liters_per_m2"] = {"min": 0.0, "max": float(train_part.liters_per_m2.max())}
for f in features:
    span = (scaler[f]["max"] - scaler[f]["min"]) or 1.0
    data[f + "_norm"] = ((data[f] - scaler[f]["min"]) / span).clip(0, 1)

# 5. Save ----------------------------------------------------------------------
cols = ["row_id", "node", "split", "deficit_pct_norm", "temp_c_norm",
        "humidity_pct_norm", "liters_per_m2", "moisture_pct", "threshold_pct",
        "deficit_pct", "temp_c", "humidity_pct"]
data[cols].to_csv("anfis_train.csv", index=False)
with open("scaler.json", "w") as f:
    json.dump(scaler, f, indent=2)

print(f"Rows: {len(data)}  (train {int((data.split == 'train').sum())}, "
      f"test {int((data.split == 'test').sum())})")
print(data[["deficit_pct", "liters_per_m2"]].describe().round(2))
print("Share of rows with deficit = 0:", round(float((data.deficit_pct == 0).mean()), 3))
print("\nSaved anfis_train.csv and scaler.json")
