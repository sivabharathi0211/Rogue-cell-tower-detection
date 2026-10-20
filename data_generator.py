import numpy as np
import pandas as pd

def generate_tower_data(n=250, include_anomalies=True, seed=17):
    """Create clearly synthetic cellular tower metadata for local demos."""
    rng = np.random.default_rng(seed)
    n = max(1, int(n))
    start = pd.Timestamp("2026-10-14 08:00:00")
    timestamps = [start + pd.Timedelta(minutes=5*i) for i in range(n)]
    cell_ids = [f"CELL-{1000 + (i % max(10, n // 5))}" for i in range(n)]
    rssi = np.clip(rng.normal(-82, 8, n), -112, -58)
    neighbors = rng.integers(2, 9, n).astype(float)
    lacs = [f"LAC-{int(v)}" for v in rng.integers(10, 30, n)]
    # Clearly fictional demonstration coordinates, not a real tower database.
    latitude = 11.0 + rng.normal(0, 0.12, n)
    longitude = 76.9 + rng.normal(0, 0.12, n)
    labels = np.array(["synthetic_normal"] * n, dtype=object)

    if include_anomalies and n >= 10:
        count = max(2, n // 18)
        indices = rng.choice(np.arange(n), size=min(count, n), replace=False)
        for j, idx in enumerate(indices):
            cell_ids[idx] = f"SIM-NEW-{j+1}"
            rssi[idx] = float(rng.uniform(-52, -40))
            neighbors[idx] = 0 if j % 2 == 0 else float(rng.integers(0, 2))
            labels[idx] = "synthetic_suspicious_scenario"
    df = pd.DataFrame({
        "cell_id": cell_ids,
        "lac": lacs,
        "rssi_dbm": np.round(rssi, 1),
        "neighbor_count": neighbors,
        "timestamp": timestamps,
        "latitude": np.round(latitude, 6),
        "longitude": np.round(longitude, 6),
        "scenario_label": labels,
    })
    return df
