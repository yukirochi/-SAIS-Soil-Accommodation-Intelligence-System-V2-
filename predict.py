"""
Step 3 - Actual prediction (deployment).

The model is already trained. The per-node moisture threshold is NOT part of the
model: your threshold automation writes it to node_config.json (or passes it in),
and this script turns it into a deficit before calling the ANFIS.

Run the demo:   python predict.py
Use in your own code:
    from predict import IrrigationPredictor
    p = IrrigationPredictor()
    liters = p.predict_liters(moisture_pct=18.2, temp_c=30.1, humidity_pct=68,
                              threshold_pct=24.0, area_m2=2.5)
"""
import json
import os
import numpy as np
from anfis import ANFIS


class IrrigationPredictor:
    def __init__(self, model_path="anfis_model.json", scaler_path="scaler.json"):
        self.model = ANFIS.load(model_path)
        with open(scaler_path) as f:
            self.sc = json.load(f)

    def _norm(self, name, value):
        lo, hi = self.sc[name]["min"], self.sc[name]["max"]
        return float(np.clip((value - lo) / ((hi - lo) or 1.0), 0.0, 1.0))

    def predict_liters(self, moisture_pct, temp_c, humidity_pct, threshold_pct,
                       area_m2=1.0):
        """Liters of water needed for one node right now."""
        deficit = max(threshold_pct - moisture_pct, 0.0)
        if deficit == 0.0:                       # already at/above threshold
            return 0.0
        x = [self._norm("deficit_pct", deficit),
             self._norm("temp_c", temp_c),
             self._norm("humidity_pct", humidity_pct)]
        per_m2 = float(self.model.predict(x)[0])
        return max(per_m2, 0.0) * area_m2


def load_node_config(path="node_config.json", threshold_path="thresholds.json"):
    """Per-node settings: area_m2 comes from node_config.json (or the demo defaults);
    threshold_pct is read from thresholds.json, which calibrate_threshold.py keeps
    up to date. A node with no stored threshold yet keeps its fallback value."""
    cfg = {                                       # demo defaults / fallbacks
        "node1": {"threshold_pct": 20.0, "area_m2": 2.0},
        "node2": {"threshold_pct": 23.0, "area_m2": 2.0},
        "node3": {"threshold_pct": 25.0, "area_m2": 3.0},
        "node4": {"threshold_pct": 27.0, "area_m2": 1.5},
    }
    if os.path.exists(path):
        with open(path) as f:
            for node, vals in json.load(f).items():
                cfg.setdefault(node, {}).update(vals)
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            for node, entry in json.load(f).items():
                if entry.get("threshold_pct") is not None:
                    cfg.setdefault(node, {"area_m2": 1.0})["threshold_pct"] = entry["threshold_pct"]
    return cfg


if __name__ == "__main__":
    predictor = IrrigationPredictor()
    nodes = load_node_config()

    # Latest sensor readings per node (replace with your live readings)
    readings = {
        "node1": {"moisture_pct": 14.0, "temp_c": 31.2, "humidity_pct": 68.0},
        "node2": {"moisture_pct": 17.5, "temp_c": 30.4, "humidity_pct": 70.0},
        "node3": {"moisture_pct": 26.0, "temp_c": 27.0, "humidity_pct": 82.0},
        "node4": {"moisture_pct": 19.0, "temp_c": 29.5, "humidity_pct": 72.0},
    }

    print(f"{'node':6s} {'moist%':>7s} {'thresh%':>8s} {'deficit%':>9s} {'liters':>8s}")
    for name, r in readings.items():
        cfg = nodes[name]
        liters = predictor.predict_liters(r["moisture_pct"], r["temp_c"], r["humidity_pct"],
                                          cfg["threshold_pct"], cfg["area_m2"])
        deficit = max(cfg["threshold_pct"] - r["moisture_pct"], 0)
        print(f"{name:6s} {r['moisture_pct']:7.1f} {cfg['threshold_pct']:8.1f} "
              f"{deficit:9.1f} {liters:8.2f}")
