import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

REQUIRED_COLUMNS = ["cell_id", "lac", "rssi_dbm", "neighbor_count", "timestamp"]
OPTIONAL_COLUMNS = ["latitude", "longitude", "scenario_label"]

class TowerDetector:
    """Rule + Isolation Forest detector for tower metadata observations."""

    FEATURE_COLUMNS = [
        "rssi_dbm", "neighbor_count", "is_new_cell",
        "neighbors_missing", "sudden_appearance", "signal_jump_db"
    ]

    def __init__(self, contamination=0.05, random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("isolation_forest", IsolationForest(
                n_estimators=150,
                contamination=contamination,
                random_state=random_state
            ))
        ])
        self.is_fitted = False
        self.training_size = 0
        self.known_cells = set()

    def _normalize(self, data):
        df = data.copy()
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
        df["cell_id"] = df["cell_id"].astype(str)
        df["lac"] = df["lac"].astype(str)
        df["rssi_dbm"] = pd.to_numeric(df["rssi_dbm"], errors="coerce")
        df["neighbor_count"] = pd.to_numeric(df["neighbor_count"], errors="coerce")
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        if "latitude" not in df:
            df["latitude"] = np.nan
        if "longitude" not in df:
            df["longitude"] = np.nan
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["cell_id", "timestamp", "rssi_dbm"]).copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def _features(self, df, baseline_cells=None):
        out = df.copy()
        cells = set(baseline_cells or [])
        out["is_new_cell"] = (~out["cell_id"].isin(cells)).astype(int)
        out["neighbors_missing"] = out["neighbor_count"].isna().astype(int)
        out["sudden_appearance"] = out["is_new_cell"]
        out["signal_jump_db"] = out["rssi_dbm"].diff().abs().fillna(0).clip(upper=100)
        # Feature engineering deliberately avoids treating cell IDs as continuous numbers.
        return out

    def fit_baseline(self, baseline):
        clean = self._normalize(baseline)
        self.known_cells = set(clean["cell_id"].astype(str).unique())
        feat = self._features(clean, baseline_cells=self.known_cells)
        x = feat[self.FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        self.model.fit(x)
        self.is_fitted = True
        self.training_size = len(clean)
        return self

    def analyze(self, data):
        clean = self._normalize(data)
        if not self.is_fitted:
            # Safe first-run fallback: fit on the current data. UI normally supplies a separate baseline.
            self.fit_baseline(clean)
        feat = self._features(clean, baseline_cells=self.known_cells)
        x = feat[self.FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
        predictions = self.model.predict(x)  # -1 = anomaly, 1 = inlier
        raw_scores = self.model.decision_function(x)
        # Lower decision_function values are more anomalous; convert to a relative 0-100 score.
        low, high = np.nanpercentile(raw_scores, [5, 95]) if len(raw_scores) > 1 else (raw_scores[0] - 1, raw_scores[0] + 1)
        scaled_anomaly = np.clip((high - raw_scores) / max(high - low, 1e-9) * 100, 0, 100)

        rows = []
        for i, row in feat.iterrows():
            reasons = []
            rule_points = 0
            rssi = row["rssi_dbm"]
            neighbors = row["neighbor_count"]
            # These are demo heuristics, not universal indicators of rogue towers.
            if pd.notna(rssi) and rssi >= -55:
                reasons.append("Very strong signal (demonstration threshold)")
                rule_points += 25
            if int(row["is_new_cell"]) == 1:
                reasons.append("Cell ID was not present in the baseline")
                rule_points += 15
            if pd.isna(neighbors):
                reasons.append("Neighbor count unavailable")
                rule_points += 5
            elif neighbors == 0:
                reasons.append("No neighboring cells reported")
                rule_points += 15
            if row["signal_jump_db"] >= 25:
                reasons.append("Large signal change between consecutive records")
                rule_points += 20

            ml_anomaly = int(predictions[i] == -1)
            ml_score = float(scaled_anomaly[i])
            combined = int(round(min(100, rule_points * 0.65 + ml_score * 0.35)))
            if combined >= 65:
                status = "High Risk"
            elif combined >= 35 or ml_anomaly:
                status = "Suspicious"
            else:
                status = "Normal"
            if ml_anomaly:
                reasons.append("Isolation Forest marked this observation as anomalous")
            if not reasons:
                reasons = ["No configured rule was triggered; behavior is within the current baseline pattern"]
            rows.append({
                "record_id": f"record-{i+1}",
                "rule_points": min(100, rule_points),
                "ml_anomaly": ml_anomaly,
                "anomaly_score": round(ml_score, 2),
                "risk_score": combined,
                "status": status,
                "reasons": "; ".join(reasons),
            })
        result = pd.concat([clean, pd.DataFrame(rows)], axis=1)
        return result

    @staticmethod
    def rule_counts(df):
        rule_names = [
            "Very strong signal",
            "Cell ID was not present in the baseline",
            "Neighbor count unavailable",
            "No neighboring cells reported",
            "Large signal change",
            "Isolation Forest anomaly",
        ]
        if "reasons" not in df:
            return pd.DataFrame({"Rule": rule_names, "Count": [0] * len(rule_names)})
        reasons = df["reasons"].fillna("")
        counts = [
            int(reasons.str.contains("Very strong signal", regex=False).sum()),
            int(reasons.str.contains("not present in the baseline", regex=False).sum()),
            int(reasons.str.contains("Neighbor count unavailable", regex=False).sum()),
            int(reasons.str.contains("No neighboring cells", regex=False).sum()),
            int(reasons.str.contains("Large signal change", regex=False).sum()),
            int(reasons.str.contains("Isolation Forest", regex=False).sum()),
        ]
        return pd.DataFrame({"Rule": rule_names, "Count": counts})
