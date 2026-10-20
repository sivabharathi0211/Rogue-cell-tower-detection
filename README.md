# Rogue Cell Tower Detection

A defensive demonstration app for analyzing cellular tower metadata with configurable rules and an Isolation Forest anomaly detector.

## Features
- Streamlit dashboard
- CSV upload and validation
- Reproducible synthetic tower data
- Rule-based risk indicators
- Scikit-learn Isolation Forest
- Simulated monitoring feed
- Interactive map with status markers
- Tower explorer, alert center, model insights, CSV exports
- Basic automated tests

## Requirements
Python 3.10+ (Python 3.11 recommended).

## Windows setup
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

## Linux/macOS setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Run tests
```bash
pytest -q
```

## CSV schema
Required: `cell_id`, `lac`, `rssi_dbm`, `neighbor_count`, `timestamp`.
Optional: `latitude`, `longitude`, `scenario_label`.

Example:
```csv
cell_id,lac,rssi_dbm,neighbor_count,timestamp,latitude,longitude
CELL-1001,LAC-12,-82,6,2026-10-14 08:00:00,11.01,76.95
```

## Important limitations
- The built-in coordinates and records are synthetic, not verified physical towers.
- Strong signal, a new cell ID, or missing neighbors alone do not establish malicious activity.
- Isolation Forest flags statistical anomalies, not malicious intent, and its anomaly score is not a probability.
- The app does not capture radio signals and does not intercept calls, messages, IMSIs, or subscriber traffic.
- Live SDR integration requires separate authorized hardware/data-source integration and field validation.
