import time
import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_folium import st_folium
import folium

from data_generator import generate_tower_data
from detector import TowerDetector, REQUIRED_COLUMNS

st.set_page_config(page_title="Rogue Cell Tower Detection", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
    div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25); padding: 12px; border-radius: 12px;}
    .small-note {font-size: .88rem; opacity: .8;}
</style>
""", unsafe_allow_html=True)

st.title("📡 Rogue Cell Tower Detection")
st.caption("Defensive metadata-anomaly monitoring • Rule engine + Isolation Forest")

if "source_df" not in st.session_state:
    st.session_state.source_df = generate_tower_data(250, seed=17)
if "processed_df" not in st.session_state:
    st.session_state.processed_df = None
if "alerts" not in st.session_state:
    st.session_state.alerts = pd.DataFrame()
if "monitoring" not in st.session_state:
    st.session_state.monitoring = False
if "stream_index" not in st.session_state:
    st.session_state.stream_index = 0
if "acknowledged" not in st.session_state:
    st.session_state.acknowledged = set()

@st.cache_resource
def get_detector():
    detector = TowerDetector()
    detector.fit_baseline(generate_tower_data(500, include_anomalies=False, seed=41))
    return detector

detector = get_detector()

with st.sidebar:
    st.header("Controls")
    page = st.radio("Navigation", ["Overview", "Live Monitoring", "Tower Explorer", "Tower Map", "Alert Center", "Model Insights", "Dataset"])
    st.divider()
    st.subheader("Data source")
    upload = st.file_uploader("Upload tower metadata CSV", type=["csv"])
    if upload is not None:
        try:
            incoming = pd.read_csv(upload)
            missing = sorted(set(REQUIRED_COLUMNS) - set(incoming.columns))
            if missing:
                st.error("Missing required columns: " + ", ".join(missing))
            else:
                st.session_state.source_df = incoming
                st.session_state.stream_index = 0
                st.success(f"Loaded {len(incoming)} records.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")
    if st.button("Generate new demo dataset", use_container_width=True):
        st.session_state.source_df = generate_tower_data(250, seed=int(time.time()) % 100000)
        st.session_state.processed_df = None
        st.session_state.stream_index = 0
        st.session_state.alerts = pd.DataFrame()
        st.rerun()
    st.caption("Synthetic records are demonstrations, not verified physical towers.")

def process_data(df):
    return detector.analyze(df)

def ensure_processed():
    if st.session_state.processed_df is None:
        st.session_state.processed_df = process_data(st.session_state.source_df)
        st.session_state.alerts = st.session_state.processed_df[
            st.session_state.processed_df["status"].isin(["Suspicious", "High Risk"])
        ].copy()
    return st.session_state.processed_df

def risk_color(status):
    return {"Normal": "green", "Suspicious": "orange", "High Risk": "red"}.get(status, "gray")

def render_map(df):
    valid = df.dropna(subset=["latitude", "longitude"]).copy()
    if valid.empty:
        st.info("No valid coordinates are available for mapping.")
        return
    center = [valid["latitude"].mean(), valid["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=5, control_scale=True)
    colors = {"Normal": "green", "Suspicious": "orange", "High Risk": "red"}
    for _, row in valid.iterrows():
        popup = (
            f"<b>Cell ID:</b> {row.get('cell_id', 'Unknown')}<br>"
            f"<b>Status:</b> {row.get('status', 'Not analyzed')}<br>"
            f"<b>RSSI:</b> {row.get('rssi_dbm', 'N/A')} dBm<br>"
            f"<b>Risk score:</b> {row.get('risk_score', 'N/A')}<br>"
            f"<b>Reasons:</b> {row.get('reasons', 'None')}"
        )
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=7 if row.get("status") != "Normal" else 4,
            color=colors.get(row.get("status"), "blue"),
            fill=True, fill_opacity=0.75, popup=folium.Popup(popup, max_width=350),
            tooltip=f"Cell {row.get('cell_id', 'Unknown')} — {row.get('status', 'Not analyzed')}"
        ).add_to(m)
    st_folium(m, use_container_width=True, height=560, returned_objects=[])

df = ensure_processed()

if page == "Overview":
    st.subheader("Security operations overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric("Unique cell IDs", f"{df['cell_id'].nunique():,}")
    c3.metric("Normal", int((df.status == "Normal").sum()))
    c4.metric("Suspicious", int((df.status == "Suspicious").sum()))
    c5.metric("High Risk", int((df.status == "High Risk").sum()))

    left, right = st.columns(2)
    with left:
        counts = df["status"].value_counts().rename_axis("Status").reset_index(name="Count")
        st.plotly_chart(px.bar(counts, x="Status", y="Count", color="Status", title="Detection outcomes"), use_container_width=True)
    with right:
        st.plotly_chart(px.histogram(df, x="risk_score", color="status", nbins=20, title="Risk score distribution"), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.plotly_chart(px.histogram(df, x="rssi_dbm", color="status", title="Signal strength distribution", labels={"rssi_dbm": "RSSI (dBm)"}), use_container_width=True)
    with right:
        by_time = df.copy()
        by_time["timestamp"] = pd.to_datetime(by_time["timestamp"], errors="coerce")
        timeline = by_time.groupby([pd.Grouper(key="timestamp", freq="h"), "status"]).size().reset_index(name="Observations")
        st.plotly_chart(px.line(timeline, x="timestamp", y="Observations", color="status", title="Observations over time"), use_container_width=True)
    st.markdown('<p class="small-note">A high anomaly score means behavior differs from the baseline. It is not proof that a tower is malicious.</p>', unsafe_allow_html=True)

elif page == "Live Monitoring":
    st.subheader("Simulated live monitoring")
    st.warning("This replays dataset records. It does not capture live cellular radio signals.")
    col1, col2, col3 = st.columns(3)
    if col1.button("▶ Start / process next batch"):
        st.session_state.monitoring = True
        st.session_state.stream_index = min(st.session_state.stream_index + 10, len(df))
        st.rerun()
    if col2.button("⏹ Stop"):
        st.session_state.monitoring = False
        st.info("Monitoring paused.")
    if col3.button("↺ Reset replay"):
        st.session_state.stream_index = 0
        st.session_state.monitoring = False
        st.rerun()
    st.progress(st.session_state.stream_index / max(len(df), 1), text=f"{st.session_state.stream_index} of {len(df)} records replayed")
    visible = df.iloc[:st.session_state.stream_index] if st.session_state.stream_index else df.head(10)
    if not visible.empty:
        st.dataframe(visible.tail(25), use_container_width=True, hide_index=True)
        latest_alerts = visible[visible.status.isin(["Suspicious", "High Risk"])].tail(10)
        if not latest_alerts.empty:
            st.error(f"{len(latest_alerts)} suspicious/high-risk observations in the latest displayed records.")
            st.dataframe(latest_alerts[["timestamp", "cell_id", "rssi_dbm", "status", "risk_score", "reasons"]], use_container_width=True, hide_index=True)
    if st.session_state.stream_index >= len(df):
        st.success("Replay reached the end. Reset to replay again.")

elif page == "Tower Explorer":
    st.subheader("Tower explorer")
    q = st.text_input("Search cell ID or LAC")
    statuses = st.multiselect("Status", ["Normal", "Suspicious", "High Risk"], default=["Normal", "Suspicious", "High Risk"])
    rssi_range = st.slider("RSSI range (dBm)", min_value=-120, max_value=-20, value=(-120, -20))
    filtered = df[df.status.isin(statuses) & df.rssi_dbm.between(rssi_range[0], rssi_range[1])]
    if q:
        filtered = filtered[
            filtered.cell_id.astype(str).str.contains(q, case=False, na=False)
            | filtered.lac.astype(str).str.contains(q, case=False, na=False)
        ]
    st.dataframe(filtered.sort_values("risk_score", ascending=False), use_container_width=True, hide_index=True)
    if not filtered.empty:
        chosen = st.selectbox("Inspect a cell ID", filtered.cell_id.astype(str).unique())
        record = filtered[filtered.cell_id.astype(str) == chosen].iloc[0]
        st.markdown(f"**Status:** {record.status}  |  **Risk:** {record.risk_score}/100")
        st.write("**Explanation:**", record.reasons)

elif page == "Tower Map":
    st.subheader("Tower observation map")
    statuses = st.multiselect("Show statuses", ["Normal", "Suspicious", "High Risk"], default=["Normal", "Suspicious", "High Risk"])
    map_df = df[df.status.isin(statuses)]
    render_map(map_df)

elif page == "Alert Center":
    st.subheader("Alert center")
    alerts = df[df.status.isin(["Suspicious", "High Risk"])].copy()
    if alerts.empty:
        st.success("No alerts in the current dataset.")
    else:
        sev = st.multiselect("Severity", ["Suspicious", "High Risk"], default=["Suspicious", "High Risk"])
        alerts = alerts[alerts.status.isin(sev)].sort_values("risk_score", ascending=False)
        st.dataframe(alerts[["timestamp", "cell_id", "lac", "rssi_dbm", "neighbor_count", "status", "risk_score", "reasons"]], use_container_width=True, hide_index=True)
        cell_options = alerts.cell_id.astype(str).tolist()
        selected = st.selectbox("Select an alert", cell_options)
        selected_row = alerts[alerts.cell_id.astype(str) == selected].iloc[0]
        st.markdown(f"### {selected_row.status} — Cell {selected_row.cell_id}")
        st.write(selected_row.reasons)
        key = str(selected_row.get("record_id", selected_row.name))
        if key in st.session_state.acknowledged:
            st.info("This alert is acknowledged in this session.")
        elif st.button("Acknowledge selected alert"):
            st.session_state.acknowledged.add(key)
            st.success("Alert acknowledged for this session.")
        st.download_button("Download alerts CSV", alerts.to_csv(index=False), "tower_alerts.csv", "text/csv")

elif page == "Model Insights":
    st.subheader("Isolation Forest model insights")
    st.write("Model: Scikit-learn IsolationForest")
    st.write("Training baseline: generated synthetic normal-behavior records.")
    st.write("Features: RSSI, neighbor count, new-cell indicator, missing-neighbor indicator, and sudden-appearance indicator.")
    st.write("Random state: fixed for reproducibility.")
    c1, c2 = st.columns(2)
    c1.metric("Model fitted", "Yes" if detector.is_fitted else "No")
    c2.metric("Training observations", detector.training_size)
    st.info("Isolation Forest is an unsupervised anomaly detector. Its output is not a probability of malicious activity. Use labeled, representative field data to evaluate precision and recall.")
    st.plotly_chart(px.histogram(df, x="anomaly_score", color="status", nbins=25, title="Anomaly score distribution"), use_container_width=True)
    st.write("Rule triggers by type")
    rule_counts = detector.rule_counts(df)
    st.dataframe(rule_counts, use_container_width=True, hide_index=True)

elif page == "Dataset":
    st.subheader("Dataset management")
    st.write(f"Current dataset: {len(st.session_state.source_df)} input records")
    st.dataframe(st.session_state.source_df.head(100), use_container_width=True, hide_index=True)
    st.download_button("Download source CSV", st.session_state.source_df.to_csv(index=False), "tower_source_data.csv", "text/csv")
    st.download_button("Download analyzed results", df.to_csv(index=False), "tower_analysis_results.csv", "text/csv")
    st.caption("Required columns: " + ", ".join(REQUIRED_COLUMNS))

st.divider()
st.caption("Prototype for defensive metadata analysis. Synthetic locations and anomaly labels are illustrative; this system does not intercept calls, messages, IMSIs, or cellular communications.")
