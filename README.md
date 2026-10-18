# ANFIS Irrigation Model

**Goal:** predict how many liters of water each of the 4 soil nodes needs, using soil moisture, air temperature and humidity. The model is trained once, before deployment. Each node's moisture **threshold** is found separately after deployment by an automation script, and is passed to the model at prediction time (it is not baked into the model).

---

## 1. Files

| File | Role | When you run it |
|---|---|---|
| `clean_dataset.py` | Cleans the raw CSV, builds the target and training data | Once, before training |
| `anfis.py` | The ANFIS model class (pure NumPy). Imported by the other scripts | Never run directly |
| `train.py` | Trains the ANFIS and saves it | Once (and again when you retrain) |
| `predict.py` | Gives liters of water for each node | Every time you need a prediction |
| `calibrate_threshold.py` | Finds each node's threshold by irrigating until moisture stops rising | After deployment, per node |

Files created for you by running the scripts:

| File | Created by | Used by |
|---|---|---|
| `anfis_train.csv` | `clean_dataset.py` | `train.py` |
| `scaler.json` | `clean_dataset.py` | `predict.py` |
| `anfis_model.json` | `train.py` | `predict.py` |
| `thresholds.json` | `calibrate_threshold.py` | `predict.py` |

Keep all scripts and your CSV in **one folder**.

---

## 2. Requirements

- Python 3.8+
- `pip install numpy pandas`

No GPU and no deep learning library are needed, so it also runs on small devices.

---

## 3. Quick start

```bash
python clean_dataset.py correlated_soil_moisture_weather_300.csv
python train.py
python predict.py
```

1. **`clean_dataset.py`** creates `anfis_train.csv` and `scaler.json`.
2. **`train.py`** prints the train/validation/test error and creates `anfis_model.json`.
3. **`predict.py`** prints a demo table of liters for the 4 nodes.

---

## 4. How it works

### Model inputs and output

| | Name | Meaning |
|---|---|---|
| Input 1 | `deficit_pct` | `threshold - current moisture` (0 if already at or above the threshold) |
| Input 2 | `temp_c` | Air temperature (°C) |
| Input 3 | `humidity_pct` | Air relative humidity (%) |
| Output | `liters_per_m2` | Water needed, in liters per square meter |

Total liters = `liters_per_m2 x node area (m2)`.

Because the threshold only enters through the deficit, **one model serves all 4 nodes**. Changing a node's threshold never requires retraining.

### The target

Your dataset has no water-needed column, so `clean_dataset.py` computes the target from a physical formula:

```
refill_mm  = deficit% / 100 x root_depth_mm x AWC / efficiency
liters/m2  = refill_mm x (1 + ALPHA_VPD x vapor_pressure_deficit)
```

The vapor pressure deficit (from temperature and humidity) increases the amount when the air is hot and dry. Liters are 0 when the deficit is 0.

Because training thresholds are not known yet, the script simulates several random thresholds per node (within `NODE_THRESHOLD_RANGE`) for every sensor row, giving 3,600 training rows from your 300.

### The ANFIS

A first-order Sugeno ANFIS with 3 Gaussian membership functions per input (27 rules). Each epoch it uses hybrid learning:
- rule outputs: least squares (closed form)
- membership functions: gradient descent (Adam)

Training stops early when validation error stops improving. Data is split by time (first 80% train, last 20% test), and no sensor row appears in both.

### The threshold automation

`calibrate_threshold.py` irrigates a node in short pulses, waits for the water to soak in, and reads moisture after each pulse. When the rise is smaller than `tol_pct` for `stable_checks` checks in a row, the moisture is "not moving anymore". That plateau becomes the node's threshold.

Safety limits (`max_pulses`, `max_liters`) stop it, and the pump is always switched off, even on a crash.

---

## 5. Things you must edit

### In `clean_dataset.py` (top of the file)

| Setting | What to put |
|---|---|
| `COL_MOISTURE`, `COL_TEMP`, `COL_HUM` | Your CSV column names |
| `ROOT_DEPTH_MM` | Root zone depth of your crop |
| `AWC` | Available water capacity of your soil (about 0.10-0.20) |
| `EFFICIENCY` | Irrigation efficiency (drip about 0.85, sprinkler about 0.70) |
| `NODE_THRESHOLD_RANGE` | Plausible threshold range per node, inside your moisture data range |

The values in the file are placeholders, not measured values.

### In `predict.py`

The demo readings and node areas are placeholders. Set each node's `area_m2` in a `node_config.json`:

```json
{
  "node1": {"area_m2": 2.0},
  "node2": {"area_m2": 2.0},
  "node3": {"area_m2": 3.0},
  "node4": {"area_m2": 1.5}
}
```

### In `calibrate_threshold.py`

1. Fill in the three methods of `HardwareNode` (`read_moisture`, `pump_on`, `pump_off`) for your real sensors and pumps. If you have a flow meter, also implement `liters_delivered`.
2. Set `CONFIG`: `pulse_seconds`, `flow_lpm` (pump liters per minute) and `soak_seconds` (how long your soil takes to spread the water).

---

## 6. Using the threshold automation

```bash
python calibrate_threshold.py node1                         # one node
python calibrate_threshold.py node1 node2 node3 node4       # several nodes
python calibrate_threshold.py node1 --simulate              # test without hardware
python calibrate_threshold.py --show                        # print stored thresholds
```

Results are **appended** to `thresholds.json`. Old runs are never overwritten and other nodes are never touched:

```json
{
  "node1": {
    "threshold_pct": 25.56,
    "updated": "2026-10-18T15:22:27+08:00",
    "history": [
      {
        "timestamp": "2026-10-18T15:22:27+08:00",
        "start_moisture_pct": 13.0,
        "plateau_pct": 25.56,
        "threshold_pct": 25.56,
        "pulses": 8,
        "liters_applied": 5.33,
        "converged": true,
        "reason": "moisture stopped rising"
      }
    ]
  }
}
```

- The stored `threshold_pct` is the **median of the last 5 converged runs**, so one odd run does not swing it.
- Runs that did not converge (hit a safety limit) are logged but ignored for the threshold.
- `threshold_fraction` in `CONFIG` (default 1.0) puts the threshold lower than the plateau if you want irrigation to trigger earlier.
- `liters_applied` is real measured water. Keep it, because it is useful as a training target later (see section 8).

`--simulate` creates fake soil so you can test the file format. **Delete the simulated `thresholds.json` before real use**, or the fake values will be treated as real thresholds.

---

## 7. Using the model in your own code

```python
from predict import IrrigationPredictor

p = IrrigationPredictor()
liters = p.predict_liters(
    moisture_pct=18.2,
    temp_c=30.1,
    humidity_pct=68,
    threshold_pct=24.0,   # from thresholds.json or your automation
    area_m2=2.5,
)
print(liters)
```

`predict_liters` returns 0 when moisture is at or above the threshold. `load_node_config()` in `predict.py` reads each node's threshold from `thresholds.json` and its area from `node_config.json`. A node with no stored threshold yet uses the demo fallback value in `predict.py`, so set real values before relying on it.

---

## 8. Limitations and next steps

- **The target is formula-based.** The ANFIS mostly learns the formula above, so its near-perfect test score does not prove it matches real water needs. The score measures how well it copies the formula.
- **Collect real labels.** For each irrigation, log the moisture before and after, the temperature, the humidity, and the water actually delivered (flow meter, or pump time x flow rate). `calibrate_threshold.py` already records `liters_applied`. After collecting enough real events, retrain with those real values as the target instead of the formula.
- **Check the threshold range.** The model is trained on deficits that come from `NODE_THRESHOLD_RANGE`. If your real plateaus fall well outside that range, update the ranges and retrain. Inputs beyond the trained range are clipped.
- **Sensor calibration.** The dataset is already in percent. If a node's sensor reports raw values, convert them to a 0-100 scale before using them, with a dry (open air) and a wet (in water) reading.

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'anfis'` | Run the scripts from the folder that contains `anfis.py` |
| `FileNotFoundError: anfis_train.csv` | Run `clean_dataset.py` before `train.py` |
| `FileNotFoundError: anfis_model.json` or `scaler.json` | Run `clean_dataset.py` and `train.py` before `predict.py` |
| `KeyError` for a column name in `clean_dataset.py` | Set `COL_MOISTURE`, `COL_TEMP` and `COL_HUM` to match your CSV header |
| `NotImplementedError` in `calibrate_threshold.py` | Fill in the `HardwareNode` methods, or use `--simulate` |
| Calibration never converges | Increase `max_pulses` or `tol_pct`, or lengthen `soak_seconds` so the sensor sees the water |
| Predicted liters seem too high or too low | Re-check `ROOT_DEPTH_MM`, `AWC`, `EFFICIENCY` and the node `area_m2` |
