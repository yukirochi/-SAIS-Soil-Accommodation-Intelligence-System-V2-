import pandas as pd
import numpy as np

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("soil_merged_dataset.csv")

# ── Keep only the 8 target columns ───────────────────────────────────────────
KEEP_COLS = [
    "N_mgkg",            # Nitrogen
    "P_mgkg",            # Phosphorus
    "K_mgkg",            # Potassium
    "pH",                # Soil pH
    "EC_dSm",            # Electrical Conductivity
    "soil_temp_C",       # Soil Temperature
    "soil_moisture_pct", # Soil Moisture
    "atmo_temp_C",       # Atmosphere Temperature
    "atmo_humidity_pct", # Atmosphere Humidity
]

df = df[KEEP_COLS].copy()

# ── Fertility Rate Formula (per row) ─────────────────────────────────────────
#
#  Score breakdown (total = 100 pts):
#    N   → 25 pts  optimal: 60–120 mg/kg
#    P   → 20 pts  optimal: 30–80  mg/kg
#    K   → 20 pts  optimal: 80–200 mg/kg
#    pH  → 25 pts  optimal: 5.8–6.8
#    EC  → 10 pts  optimal: ≤ 1.0 dS/m  (penalty above 2.0 = saline)
#
def score_range(val, lo, hi, mid_lo, mid_hi, weight):
    """Returns partial score for one parameter."""
    if pd.isnull(val):
        return 0
    if mid_lo <= val <= mid_hi:          # inside optimal → full score
        return weight
    elif lo <= val < mid_lo:             # below optimal → scale up
        return weight * (val - lo) / (mid_lo - lo)
    elif mid_hi < val <= hi:             # above optimal → scale down
        return weight * (hi - val) / (hi - mid_hi)
    return 0                             # outside valid range → 0

def compute_fertility_rate(row):
    score = 0
    score += score_range(row["N_mgkg"],  0,   200,  60,  120, 25)
    score += score_range(row["P_mgkg"],  0,   150,  30,   80, 20)
    score += score_range(row["K_mgkg"],  0,   400,  80,  200, 20)
    score += score_range(row["pH"],      3.5,  9.0,  5.8,  6.8, 25)

    ec = row["EC_dSm"]
    if not pd.isnull(ec):
        if ec <= 1.0:
            score += 10                        # ideal
        elif ec <= 2.0:
            score += 10 * (2.0 - ec)          # mild penalty
        # above 2.0 → 0 pts (saline stress)

    return round(min(score, 100), 2)

def fertility_class(score):
    if pd.isnull(score): return "Unknown"
    if score >= 75: return "Very High"
    if score >= 50: return "High"
    if score >= 25: return "Medium"
    return "Low"

# ── Apply per row ─────────────────────────────────────────────────────────────
df["fertility_rate"]  = df.apply(compute_fertility_rate, axis=1)
df["fertility_class"] = df["fertility_rate"].apply(fertility_class)

# ── Preview ───────────────────────────────────────────────────────────────────
print(df.head(10).to_string(index=False))
print(f"\nRows: {len(df)}  |  Columns: {list(df.columns)}")
print("\nFertility class distribution:")
print(df["fertility_class"].value_counts())

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("soil_fertility_clean.csv", index=False)
print("\n✅ Saved → soil_fertility_clean.csv")
