"""
Step 4 - Automatic threshold discovery (runs AFTER deployment, outside the model).

Goal: find each node's moisture threshold by irrigating in short pulses until the
moisture reading stops rising. The level where it flattens out is that soil's
"full" level, and it becomes the node's threshold.

How one calibration works
  1. read the starting moisture (median of a few samples to ignore noise)
  2. run the pump for one short pulse, then wait for the water to spread (soak)
  3. read moisture again; if it rose by less than TOL for STABLE_CHECKS checks in a
     row, it is "not moving anymore" -> stop, the plateau is the threshold
  4. safety stops: MAX_PULSES / MAX_LITERS, and the pump is always switched off
  5. append the result to thresholds.json (never overwrites old runs)

One script serves all nodes - pass the node names:
    python calibrate_threshold.py node1
    python calibrate_threshold.py node1 node2 node3 node4
    python calibrate_threshold.py node1 --simulate        # test without hardware
    python calibrate_threshold.py --show                  # print stored thresholds

To use real hardware, fill in HardwareNode below (GPIO / MQTT / serial, etc.).
"""
import argparse
import json
import os
import random
import statistics
import time
from datetime import datetime

try:                                   # file lock so 4 nodes can write safely
    import fcntl
except ImportError:                    # e.g. Windows
    fcntl = None

# ----------------------------------------------------------------- CONFIG ---
CONFIG = {
    "pulse_seconds": 20,        # pump run time per pulse
    "flow_lpm": 2.0,            # pump flow in liters/minute (ignored if the node
                                # reports liters itself via NodeIO.liters_delivered)
    "soak_seconds": 300,        # wait after each pulse for water to spread
    "samples": 5,               # readings per check (median is used)
    "sample_gap_seconds": 2,
    "tol_pct": 0.3,             # "not moving": rise smaller than this ...
    "stable_checks": 2,         # ... for this many checks in a row
    "max_pulses": 30,           # safety limit
    "max_liters": 50.0,         # safety limit
    "threshold_fraction": 1.0,  # 1.0 = threshold at the plateau; <1.0 puts it lower
                                # (threshold = start + fraction * (plateau - start))
    "average_last": 5,          # stored threshold = median of the last N good runs
}
THRESHOLD_FILE = "thresholds.json"
# ------------------------------------------------------------------------- #


# ------------------------------------------------------- node hardware API --
class NodeIO:
    """What the script needs from a node. Implement these 3 methods."""

    def read_moisture(self):
        """Return the current soil moisture in percent (one reading)."""
        raise NotImplementedError

    def pump_on(self):
        raise NotImplementedError

    def pump_off(self):
        raise NotImplementedError

    def liters_delivered(self):
        """Optional: return total liters from a flow meter, else None."""
        return None


class HardwareNode(NodeIO):
    """Replace the bodies below with your real node control."""

    def __init__(self, node):
        self.node = node

    def read_moisture(self):
        raise NotImplementedError("read the %s sensor here" % self.node)

    def pump_on(self):
        raise NotImplementedError("switch the %s pump on here" % self.node)

    def pump_off(self):
        raise NotImplementedError("switch the %s pump off here" % self.node)


class SimulatedNode(NodeIO):
    """Fake soil for testing: each pulse closes part of the gap to 'full'."""

    def __init__(self, node, seed=0):
        idx = int("".join(c for c in node if c.isdigit()) or 1)
        self.rng = random.Random(seed + idx)
        self.m = 12.0 + idx                     # starting moisture
        self.full = 24.0 + 2.0 * idx            # this node's true plateau
        self.gain, self.noise, self._on = 0.4, 0.08, False

    def read_moisture(self):
        return self.m + self.rng.gauss(0, self.noise)

    def pump_on(self):
        self._on = True

    def pump_off(self):
        if self._on:
            self.m += self.gain * (self.full - self.m)
            self._on = False


# ------------------------------------------------------------- calibration --
def _median_reading(io, cfg, sleep):
    vals = []
    for i in range(cfg["samples"]):
        vals.append(io.read_moisture())
        if i < cfg["samples"] - 1:
            sleep(cfg["sample_gap_seconds"])
    return statistics.median(vals)


def run_calibration(node, io, cfg=CONFIG, sleep=time.sleep):
    """Irrigate in pulses until moisture stops rising. Returns a result dict."""
    start = _median_reading(io, cfg, sleep)
    checks = [start]                 # moisture after each soak
    liters, stable, pulses = 0.0, 0, 0
    converged, reason = False, "max_pulses reached"

    try:
        while pulses < cfg["max_pulses"]:
            io.pump_on()
            sleep(cfg["pulse_seconds"])
            io.pump_off()
            pulses += 1
            liters += cfg["flow_lpm"] * cfg["pulse_seconds"] / 60.0
            sleep(cfg["soak_seconds"])

            checks.append(_median_reading(io, cfg, sleep))
            rise = checks[-1] - checks[-2]
            stable = stable + 1 if rise < cfg["tol_pct"] else 0
            print(f"  [{node}] pulse {pulses:2d}  moisture {checks[-1]:6.2f}%  "
                  f"rise {rise:+.2f}  liters {liters:.2f}")

            if stable >= cfg["stable_checks"]:
                converged, reason = True, "moisture stopped rising"
                break
            if liters >= cfg["max_liters"]:
                reason = "max_liters reached"
                break
    finally:
        io.pump_off()                # always leave the pump off

    metered = io.liters_delivered()
    if metered is not None:
        liters = float(metered)

    flat = checks[-(cfg["stable_checks"] + 1):] if converged else checks[-1:]
    plateau = float(statistics.mean(flat))
    threshold = start + cfg["threshold_fraction"] * (plateau - start)
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "start_moisture_pct": round(start, 2),
        "plateau_pct": round(plateau, 2),
        "threshold_pct": round(threshold, 2),
        "pulses": pulses,
        "liters_applied": round(liters, 2),
        "converged": converged,
        "reason": reason,
    }


# ----------------------------------------------------------------- storage --
def append_result(node, result, path=THRESHOLD_FILE, average_last=5):
    """Append one run to the node's history and refresh its current threshold.
    Other nodes' entries are left untouched."""
    lock = open(path + ".lock", "w")
    try:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_EX)
        data = {}
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)

        entry = data.setdefault(node, {"threshold_pct": None, "history": []})
        entry["history"].append(result)

        good = [h["threshold_pct"] for h in entry["history"] if h["converged"]]
        if good:                                     # only trust converged runs
            entry["threshold_pct"] = round(statistics.median(good[-average_last:]), 2)
            entry["updated"] = result["timestamp"]

        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)                        # atomic write
    finally:
        if fcntl:
            fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    return data[node]


def show(path=THRESHOLD_FILE):
    if not os.path.exists(path):
        print("No thresholds stored yet.")
        return
    with open(path) as f:
        data = json.load(f)
    for node, e in sorted(data.items()):
        print(f"{node}: threshold {e['threshold_pct']}%  "
              f"({len(e['history'])} run(s), updated {e.get('updated', '-')})")


# --------------------------------------------------------------------- CLI --
def main():
    ap = argparse.ArgumentParser(description="Find node moisture thresholds.")
    ap.add_argument("nodes", nargs="*", help="node names, e.g. node1 node2")
    ap.add_argument("--simulate", action="store_true", help="fake nodes, no waiting")
    ap.add_argument("--file", default=THRESHOLD_FILE)
    ap.add_argument("--show", action="store_true", help="print stored thresholds")
    args = ap.parse_args()

    if args.show or not args.nodes:
        show(args.file)
        return

    for node in args.nodes:
        print(f"Calibrating {node} ...")
        io = SimulatedNode(node, seed=int(time.time()) % 1000) if args.simulate \
            else HardwareNode(node)
        sleep = (lambda s: None) if args.simulate else time.sleep
        result = run_calibration(node, io, CONFIG, sleep)
        entry = append_result(node, result, args.file, CONFIG["average_last"])
        status = "OK" if result["converged"] else "NOT converged (" + result["reason"] + ")"
        print(f"  -> {status}: plateau {result['plateau_pct']}%, "
              f"{result['liters_applied']} L used, stored threshold "
              f"{entry['threshold_pct']}%\n")


if __name__ == "__main__":
    main()
