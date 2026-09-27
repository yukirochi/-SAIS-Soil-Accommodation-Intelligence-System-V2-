"""
Training Pipeline (Pipeline A) -- Dataset Formation -> Cleaning -> Train -> Validate -> Save
================================================================================================
This is the "calibration" pipeline in the sense we've been using it: it does NOT
run the physical pump-burst/field-capacity hardware routine (that's calibration.py).
Instead, per your instruction, it treats the EXISTING sensor log
(correlated_soil_moisture_weather_300.csv) as if it were the real historical
sensor readings -- moisture, temperature, humidity are genuine columns you
already have; the one thing genuinely missing is a real measured "liters
delivered" outcome, so that part is still formula-derived (same caveat as
every run before this one: the resulting accuracy numbers show the pipeline
works, not that the model is validated against real irrigation outcomes).

STAGES
------
  1. LOAD          raw sensor log
  2. CLEAN         drop physically-impossible readings, duplicates, gaps
  3. FORM DATASET  compute the Liters_Needed target from the calibration formula
  4. TRAIN         fit an ANFISRegressor (anfis-toolbox) on the cleaned, labeled data
  5. VALIDATE      compute RMSE / MAE / R^2 on a held-out test split
  6. QUALITY GATE  only proceed to save if metrics clear defined acceptance
                   thresholds -- a bad model is refused, not silently shipped
  7. SAVE          model.json (only reached if the gate passes)

Run with:
    python3 train_pipeline.py --input correlated_soil_moisture_weather_300.csv
"""

import argparse
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from anfis_toolbox import ANFISRegressor

# ===========================================================================
# 1. CONFIG
# ===========================================================================
# -- Calibration formula constants. In a full deployment these would be
#    loaded from calibrated_constants.json (the output of calibration.py's
#    hardware routine). Here, per your instruction, we're treating the
#    existing dataset as the sensor log directly, so these stay as the
#    literature-informed defaults from anfis_pump_control_liters.py.
HEALTHY_MOISTURE = 30.0
ROOT_ZONE_AREA_M2 = 1.0
ROOT_ZONE_DEPTH_M = 0.30
APPLICATION_EFFICIENCY = 0.85
K_TEMP = 0.020
K_HUM = 0.012
MIN_ET_FACTOR, MAX_ET_FACTOR = 0.6, 1.6
SOIL_VOLUME_LITERS = ROOT_ZONE_AREA_M2 * ROOT_ZONE_DEPTH_M * 1000.0

# -- Data cleaning bounds (physically plausible ranges -- anything outside
#    these is a sensor glitch, not a real reading)
MOISTURE_VALID_RANGE = (0.0, 60.0)     # % VWC -- soil can't exceed ~saturation
TEMP_VALID_RANGE = (-10.0, 55.0)       # deg C -- generous outdoor bounds
HUMIDITY_VALID_RANGE = (0.0, 100.0)    # % RH

# -- Training config
N_MFS_PER_INPUT = 2
N_EPOCHS = 80
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42

# -- QUALITY GATE thresholds. A trained model only gets saved if it clears
#    ALL of these on the held-out test split. Tune these to how much error
#    is actually tolerable for your irrigation use case.
MIN_ACCEPTABLE_R2 = 0.90
MAX_ACCEPTABLE_RMSE_FRACTION = 0.10   # test RMSE must be <=10% of the target's
                                        # observed range (max - min liters)
MAX_ACCEPTABLE_MAE_LITERS = 5.0       # hard cap regardless of scale

MODEL_OUTPUT_PATH = "model.json"
METRICS_OUTPUT_PATH = "training_metrics.json"


# ===========================================================================
# 2. LOAD
# ===========================================================================
def load_sensor_log(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    required = ["Soil_Moisture_%", "Atmospheric_Temperature_C", "Atmospheric_Humidity_%"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Sensor log is missing required column(s): {missing}")
    print(f"[load] {len(df)} rows loaded from {path}")
    return df


# ===========================================================================
# 3. CLEAN
# ===========================================================================
def clean_sensor_log(df):
    """Drop rows a real sensor could never have produced, plus dupes/NaNs.
    Prints exactly what was removed and why, so a bad log is visible, not
    silently fixed."""
    start_n = len(df)
    reasons = {}

    before = len(df)
    df = df.dropna(subset=["Soil_Moisture_%", "Atmospheric_Temperature_C", "Atmospheric_Humidity_%"])
    reasons["missing_values"] = before - len(df)

    if "Record_ID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=["Record_ID"])
        reasons["duplicate_record_id"] = before - len(df)

    before = len(df)
    df = df.drop_duplicates()
    reasons["exact_duplicate_rows"] = before - len(df)

    for col, (lo, hi) in [
        ("Soil_Moisture_%", MOISTURE_VALID_RANGE),
        ("Atmospheric_Temperature_C", TEMP_VALID_RANGE),
        ("Atmospheric_Humidity_%", HUMIDITY_VALID_RANGE),
    ]:
        before = len(df)
        df = df[(df[col] >= lo) & (df[col] <= hi)]
        reasons[f"out_of_range::{col}"] = before - len(df)

    print(f"[clean] {start_n} -> {len(df)} rows after cleaning:")
    for reason, n in reasons.items():
        if n > 0:
            print(f"    dropped {n} row(s) -- {reason}")
    if len(df) < start_n:
        pass
    else:
        print("    no rows dropped -- log was already clean")

    if len(df) < 30:
        raise ValueError(f"Only {len(df)} clean rows remain -- too few to train on reliably.")

    return df.reset_index(drop=True)


# ===========================================================================
# 4. FORM DATASET (compute the Liters_Needed target)
# ===========================================================================
def form_dataset(df):
    moisture = df["Soil_Moisture_%"].values.astype(float)
    temp = df["Atmospheric_Temperature_C"].values.astype(float)
    hum = df["Atmospheric_Humidity_%"].values.astype(float)
    mean_temp, mean_hum = temp.mean(), hum.mean()

    deficit_pct = np.clip(HEALTHY_MOISTURE - moisture, 0.0, None)
    et_factor = np.clip(1.0 + K_TEMP * (temp - mean_temp) - K_HUM * (hum - mean_hum),
                         MIN_ET_FACTOR, MAX_ET_FACTOR)
    liters_in_soil = (deficit_pct / 100.0) * SOIL_VOLUME_LITERS
    liters_needed = np.clip((liters_in_soil * et_factor) / APPLICATION_EFFICIENCY, 0.0, None)

    df = df.copy()
    df["Moisture_Deficit_%"] = deficit_pct
    df["ET_Factor"] = et_factor
    df["Liters_Needed"] = liters_needed

    print(f"[form_dataset] Liters_Needed -> min={liters_needed.min():.2f}, "
          f"max={liters_needed.max():.2f}, mean={liters_needed.mean():.2f}, "
          f"std={liters_needed.std():.2f}")
    if liters_needed.std() < 1e-6:
        raise ValueError("Liters_Needed has ~zero variance -- HEALTHY_MOISTURE is "
                          "likely outside this dataset's moisture range. Fix the "
                          "constants before training.")
    return df


# ===========================================================================
# 5. TRAIN
# ===========================================================================
def train_model(df):
    X = df[["Soil_Moisture_%", "Atmospheric_Temperature_C", "Atmospheric_Humidity_%"]].values
    y = df["Liters_Needed"].values

    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.permutation(len(X))
    n_train = int(len(X) * TRAIN_FRACTION)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    print(f"[train] {len(train_idx)} train rows / {len(test_idx)} test rows, "
          f"{N_MFS_PER_INPUT} MFs/input, {N_EPOCHS} epochs")
    model = ANFISRegressor(n_mfs=N_MFS_PER_INPUT, mf_type="gaussian", optimizer="hybrid",
                            epochs=N_EPOCHS, random_state=RANDOM_STATE, verbose=False)
    model.fit(X[train_idx], y[train_idx])
    return model, X, y, train_idx, test_idx


# ===========================================================================
# 6. VALIDATE
# ===========================================================================
def evaluate(model, X, y, train_idx, test_idx):
    pred_train = model.predict(X[train_idx])
    pred_test = model.predict(X[test_idx])

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    y_test = y[test_idx]
    metrics = {
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "train_rmse_L": rmse(pred_train, y[train_idx]),
        "train_mae_L": mae(pred_train, y[train_idx]),
        "test_rmse_L": rmse(pred_test, y_test),
        "test_mae_L": mae(pred_test, y_test),
        "test_r2": float(model.score(X[test_idx], y_test)),
        "target_range_L": float(y.max() - y.min()),
        "target_min_L": float(y.min()),
        "target_max_L": float(y.max()),
    }
    metrics["test_rmse_fraction_of_range"] = (
        metrics["test_rmse_L"] / metrics["target_range_L"] if metrics["target_range_L"] > 0 else float("inf")
    )

    print("\n[validate] === Performance ===")
    print(f"  Train RMSE: {metrics['train_rmse_L']:.3f} L | MAE: {metrics['train_mae_L']:.3f} L")
    print(f"  Test  RMSE: {metrics['test_rmse_L']:.3f} L "
          f"({metrics['test_rmse_fraction_of_range']*100:.1f}% of target range) | "
          f"MAE: {metrics['test_mae_L']:.3f} L")
    print(f"  Test  R^2 : {metrics['test_r2']:.4f}")
    return metrics


# ===========================================================================
# 7. QUALITY GATE -- decide whether this model is good enough to save
# ===========================================================================
def quality_gate(metrics):
    checks = {
        f"R^2 >= {MIN_ACCEPTABLE_R2}": metrics["test_r2"] >= MIN_ACCEPTABLE_R2,
        f"RMSE <= {MAX_ACCEPTABLE_RMSE_FRACTION*100:.0f}% of target range":
            metrics["test_rmse_fraction_of_range"] <= MAX_ACCEPTABLE_RMSE_FRACTION,
        f"MAE <= {MAX_ACCEPTABLE_MAE_LITERS} L": metrics["test_mae_L"] <= MAX_ACCEPTABLE_MAE_LITERS,
    }
    print("\n[quality_gate] Checking acceptance thresholds:")
    all_passed = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        all_passed = all_passed and passed
    return all_passed, checks


# ===========================================================================
# 8. MAIN
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Dataset formation -> clean -> train -> validate -> save")
    parser.add_argument("--input", default="correlated_soil_moisture_weather_300.csv",
                         help="Sensor log CSV (moisture/temp/humidity columns).")
    parser.add_argument("--model-out", default=MODEL_OUTPUT_PATH)
    parser.add_argument("--force-save", action="store_true",
                         help="Save the model even if it fails the quality gate "
                              "(NOT recommended -- for debugging only).")
    args = parser.parse_args()

    raw_df = load_sensor_log(args.input)
    clean_df = clean_sensor_log(raw_df)
    labeled_df = form_dataset(clean_df)
    model, X, y, train_idx, test_idx = train_model(labeled_df)
    metrics = evaluate(model, X, y, train_idx, test_idx)
    passed, checks = quality_gate(metrics)

    metrics["quality_gate_passed"] = passed
    metrics["quality_gate_checks"] = checks
    metrics["trained_at"] = datetime.now().isoformat()
    with open(METRICS_OUTPUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[metrics] Saved -> {METRICS_OUTPUT_PATH} (kept regardless of pass/fail, for audit trail)")

    if passed or args.force_save:
        model.save(args.model_out)
        if passed:
            print(f"[save] Quality gate PASSED -> saved model -> {args.model_out}")
        else:
            print(f"[save] Quality gate FAILED but --force-save was set -> "
                  f"saved anyway -> {args.model_out} (NOT recommended for production use)")
        return 0
    else:
        print(f"\n[save] Quality gate FAILED -- {args.model_out} was NOT written. "
              f"Fix the failing check(s) above before re-running "
              f"(more/cleaner data, different HEALTHY_MOISTURE, or revisit "
              f"the acceptance thresholds if they're stricter than needed).")
        return 1


if __name__ == "__main__":
    sys.exit(main())
