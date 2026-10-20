import pandas as pd
from detector import TowerDetector
from data_generator import generate_tower_data

def test_generated_data_has_required_fields():
    df = generate_tower_data(30, seed=1)
    for col in ["cell_id", "lac", "rssi_dbm", "neighbor_count", "timestamp"]:
        assert col in df.columns

def test_detector_returns_status_and_scores():
    baseline = generate_tower_data(100, include_anomalies=False, seed=2)
    incoming = generate_tower_data(40, include_anomalies=True, seed=3)
    detector = TowerDetector().fit_baseline(baseline)
    result = detector.analyze(incoming)
    assert len(result) == len(incoming)
    assert set(result["status"]).issubset({"Normal", "Suspicious", "High Risk"})
    assert result["risk_score"].between(0, 100).all()

def test_missing_required_columns_raises():
    detector = TowerDetector()
    bad = pd.DataFrame({"cell_id": ["x"]})
    try:
        detector.analyze(bad)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
