import io
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="QCAUS Sensor Replay & Validation Lab",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

REQUIRED_COLUMNS = ["timestamp", "track_id"]
OPTIONAL_COLUMNS = [
    "latitude", "longitude", "altitude", "speed", "heading",
    "sensor_id", "confidence", "radial_velocity", "signal_strength",
    "frequency", "classification", "ground_truth",
]


def demo_data():
    """Create a reproducible synthetic multi-sensor scenario for first-run demos."""
    rng = np.random.default_rng(42)
    rows = []
    start = pd.Timestamp("2026-01-01 00:00:00")
    n_steps = 120

    # Four tracks: two normal, one maneuvering/anomalous, one intermittent.
    tracks = {
        "T001": dict(lat=18.0, lon=36.0, speed=180.0, heading=72.0, anomaly=False),
        "T002": dict(lat=18.3, lon=36.4, speed=220.0, heading=110.0, anomaly=False),
        "T003": dict(lat=18.8, lon=36.2, speed=260.0, heading=40.0, anomaly=True),
        "T004": dict(lat=19.1, lon=36.7, speed=140.0, heading=250.0, anomaly=False),
    }
    sensors = ["RADAR-A", "RADAR-B", "EW-1", "EO-1"]

    for step in range(n_steps):
        ts = start + pd.Timedelta(seconds=30 * step)
        for tid, base in tracks.items():
            # T004 has intermittent observations.
            if tid == "T004" and step % 5 in (1, 2):
                continue

            progress = step / 10.0
            lat = base["lat"] + 0.008 * progress + rng.normal(0, 0.002)
            lon = base["lon"] + 0.012 * progress + rng.normal(0, 0.002)
            speed = base["speed"] + rng.normal(0, 5)

            if base["anomaly"] and 45 <= step <= 75:
                # Simulated maneuver / kinematic anomaly.
                lat += 0.08 * math.sin(step / 3)
                lon += 0.06 * math.cos(step / 4)
                speed += 90 * math.sin(step / 5)
                heading = (base["heading"] + 70 * math.sin(step / 4)) % 360
                ground_truth = "threat"
            else:
                heading = (base["heading"] + rng.normal(0, 3)) % 360
                ground_truth = "normal"

            sensor = sensors[(step + int(tid[-1])) % len(sensors)]
            confidence = float(np.clip(0.82 + rng.normal(0, 0.06), 0.35, 0.99))

            rows.append({
                "timestamp": ts,
                "track_id": tid,
                "latitude": lat,
                "longitude": lon,
                "altitude": 10000 + rng.normal(0, 150),
                "speed": max(20, speed),
                "heading": heading,
                "sensor_id": sensor,
                "confidence": confidence,
                "classification": "unknown" if base["anomaly"] else "air",
                "ground_truth": ground_truth,
            })

    return pd.DataFrame(rows)


def normalize_dataframe(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    for col in ["track_id", "sensor_id", "classification", "ground_truth"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    numeric_cols = [
        "latitude", "longitude", "altitude", "speed", "heading",
        "confidence", "radial_velocity", "signal_strength", "frequency"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def validate_data(df):
    issues = []
    warnings = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        issues.append(f"Missing required columns: {', '.join(missing)}")

    if "timestamp" in df.columns:
        bad_ts = int(df["timestamp"].isna().sum())
        if bad_ts:
            issues.append(f"{bad_ts} rows have invalid timestamps.")

    if "track_id" in df.columns:
        missing_ids = int(df["track_id"].isna().sum())
        if missing_ids:
            issues.append(f"{missing_ids} rows have missing track IDs.")

    if "latitude" not in df.columns or "longitude" not in df.columns:
        warnings.append("Latitude/longitude are unavailable; geographic replay will be limited.")

    if "sensor_id" not in df.columns:
        warnings.append("sensor_id is unavailable; sensor-agreement analysis will be limited.")

    if "ground_truth" not in df.columns:
        warnings.append("ground_truth is unavailable; supervised validation metrics cannot be calculated.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        warnings.append(f"{duplicate_count} duplicate rows detected.")

    return issues, warnings


def calculate_baseline(df):
    """Simple baseline: use supplied confidence when present, otherwise a neutral score."""
    out = df.copy()
    if "confidence" in out.columns:
        out["baseline_confidence"] = out["confidence"].fillna(out["confidence"].median())
    else:
        out["baseline_confidence"] = 0.50

    out["baseline_threat_score"] = np.clip(
        out["baseline_confidence"].astype(float) * 100, 0, 100
    )
    out["baseline_label"] = pd.cut(
        out["baseline_threat_score"],
        bins=[-np.inf, 39, 69, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    return out


def run_qcaus_analysis(df):
    """
    Transparent MVP scoring layer.

    This is intentionally a validation/demo engine, not a claimed operational
    defense model. Replace this function with the user's validated QCAUS engine
    when available.
    """
    out = df.copy()

    # Per-track kinematic features.
    out = out.sort_values(["track_id", "timestamp"]).reset_index(drop=True)
    out["dt_seconds"] = (
        out.groupby("track_id")["timestamp"].diff().dt.total_seconds().fillna(0)
    )

    if "speed" in out.columns:
        speed_median = out.groupby("track_id")["speed"].transform("median")
        speed_mad = (
            out.groupby("track_id")["speed"]
            .transform(lambda s: np.median(np.abs(s - np.median(s))) + 1e-6)
        )
        out["speed_anomaly"] = np.clip(
            np.abs(out["speed"] - speed_median) / (6 * speed_mad), 0, 1
        )
    else:
        out["speed_anomaly"] = 0.0

    if "heading" in out.columns:
        heading_delta = out.groupby("track_id")["heading"].diff().abs()
        heading_delta = np.minimum(heading_delta, 360 - heading_delta)
        out["heading_anomaly"] = np.clip(heading_delta.fillna(0) / 90, 0, 1)
    else:
        out["heading_anomaly"] = 0.0

    # Track continuity / gaps.
    median_dt = out.loc[out["dt_seconds"] > 0, "dt_seconds"].median()
    median_dt = 30.0 if pd.isna(median_dt) or median_dt <= 0 else median_dt
    out["gap_factor"] = np.clip(
        np.maximum(out["dt_seconds"] - 3 * median_dt, 0) / (10 * median_dt),
        0, 1
    )

    # Sensor contribution.
    if "sensor_id" in out.columns:
        sensor_counts = out.groupby("track_id")["sensor_id"].transform("nunique")
        out["sensor_agreement"] = np.clip(sensor_counts / 4, 0, 1)
    else:
        out["sensor_agreement"] = 0.5

    base_conf = (
        out["confidence"].fillna(0.5).astype(float)
        if "confidence" in out.columns
        else pd.Series(0.5, index=out.index)
    )

    # Transparent weighted MVP score.
    out["qcaus_anomaly_score"] = np.clip(
        0.45 * out["speed_anomaly"]
        + 0.35 * out["heading_anomaly"]
        + 0.20 * out["gap_factor"],
        0, 1
    )

    out["qcaus_threat_score"] = np.clip(
        100 * (
            0.45 * base_conf
            + 0.35 * out["qcaus_anomaly_score"]
            + 0.20 * out["sensor_agreement"]
        ),
        0, 100,
    )

    out["qcaus_label"] = pd.cut(
        out["qcaus_threat_score"],
        bins=[-np.inf, 39, 69, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)

    out["qcaus_anomaly_flag"] = out["qcaus_anomaly_score"] >= 0.45
    out["score_delta"] = out["qcaus_threat_score"] - out["baseline_threat_score"]

    return out


def track_summary(df):
    agg = {
        "qcaus_threat_score": "max",
        "baseline_threat_score": "max",
        "qcaus_anomaly_flag": "max",
    }
    if "ground_truth" in df.columns:
        agg["ground_truth"] = lambda x: x.mode().iloc[0] if not x.mode().empty else "unknown"

    summary = df.groupby("track_id").agg(agg).reset_index()
    summary["qcaus_label"] = pd.cut(
        summary["qcaus_threat_score"],
        bins=[-np.inf, 39, 69, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    summary["baseline_label"] = pd.cut(
        summary["baseline_threat_score"],
        bins=[-np.inf, 39, 69, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    return summary


def validation_metrics(df):
    if "ground_truth" not in df.columns:
        return None

    truth = df["ground_truth"].astype(str).str.lower()
    predicted = df["qcaus_label"].astype(str).str.lower()

    valid = truth.isin(["threat", "normal"]) & predicted.isin(["high", "low", "medium"])
    if valid.sum() == 0:
        return None

    y_true = truth[valid].eq("threat")
    y_pred = predicted[valid].eq("high")

    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    return {
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def make_report(df, scenario_name, notes):
    summary = track_summary(df)
    metrics = validation_metrics(df)

    lines = [
        "# QCAUS Sensor Replay & Validation Report",
        "",
        f"**Scenario:** {scenario_name}",
        f"**Generated:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Dataset",
        f"- Observations: {len(df):,}",
        f"- Unique tracks: {df['track_id'].nunique():,}",
        f"- Sensors: {df['sensor_id'].nunique() if 'sensor_id' in df.columns else 'N/A'}",
        f"- Start: {df['timestamp'].min() if 'timestamp' in df.columns else 'N/A'}",
        f"- End: {df['timestamp'].max() if 'timestamp' in df.columns else 'N/A'}",
        "",
        "## Results",
        f"- High-threat tracks (QCAUS): {(summary['qcaus_label'] == 'HIGH').sum():,}",
        f"- High-threat tracks (baseline): {(summary['baseline_label'] == 'HIGH').sum():,}",
        f"- Tracks flagged with anomaly indicators: {int(summary['qcaus_anomaly_flag'].sum()):,}",
        "",
    ]

    if metrics:
        lines.extend([
            "## Ground-Truth Validation",
            f"- Precision: {metrics['precision']:.3f}",
            f"- Recall: {metrics['recall']:.3f}",
            f"- F1 score: {metrics['f1']:.3f}",
            f"- True positives: {metrics['true_positive']}",
            f"- False positives: {metrics['false_positive']}",
            f"- False negatives: {metrics['false_negative']}",
            "",
        ])
    else:
        lines.extend([
            "## Ground-Truth Validation",
            "Ground-truth labels were not available or were not recognized. "
            "Detection performance metrics were not calculated.",
            "",
        ])

    lines.extend([
        "## Analyst Notes",
        notes or "No analyst notes provided.",
        "",
        "## Important Validation Notice",
        "This MVP uses a transparent demonstration scoring layer. "
        "Operational performance claims should only be made after validation "
        "against controlled datasets, defined baselines, and customer-approved "
        "ground truth.",
    ])

    return "\n".join(lines)


def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def main():
    st.title("🛰️ QCAUS Sensor Replay & Validation Lab")
    st.caption(
        "Replay historical or simulated sensor data, run a transparent QCAUS "
        "analysis layer, compare against a baseline, and generate validation results."
    )

    with st.sidebar:
        st.header("Scenario Setup")
        scenario_name = st.text_input(
            "Scenario name",
            value="QCAUS Demonstration Scenario",
        )
        data_source = st.radio(
            "Data source",
            ["Synthetic Demo Data", "Upload CSV"],
            index=0,
        )

        uploaded = None
        if data_source == "Upload CSV":
            uploaded = st.file_uploader(
                "Upload sensor/track CSV",
                type=["csv"],
                help="Required columns: timestamp, track_id. "
                     "Recommended: latitude, longitude, speed, heading, sensor_id, confidence.",
            )

        st.divider()
        st.header("Replay Controls")
        replay_rows = st.slider(
            "Observations to replay",
            min_value=50,
            max_value=5000,
            value=1000,
            step=50,
        )

        notes = st.text_area(
            "Analyst notes",
            placeholder="Record scenario assumptions, ground-truth context, or evaluation notes.",
        )

    if data_source == "Uploaded CSV" and uploaded is not None:
        raw = pd.read_csv(uploaded)
    elif data_source == "Upload CSV" and uploaded is None:
        st.info("Upload a CSV in the sidebar, or select Synthetic Demo Data.")
        return
    else:
        raw = demo_data()

    df = normalize_dataframe(raw)
    issues, warnings = validate_data(df)

    if issues:
        st.error("Data validation failed.")
        for issue in issues:
            st.error(issue)
        st.stop()

    if warnings:
        with st.expander("Data quality warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)

    # Restrict replay to requested number of chronological observations.
    df = df.sort_values("timestamp").head(replay_rows).copy()
    baseline = calculate_baseline(df)
    analyzed = run_qcaus_analysis(baseline)

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview",
        "Data Validation",
        "Scenario Replay",
        "Track Investigation",
        "Baseline vs QCAUS",
        "Validation Report",
    ])

    with tab1:
        st.subheader(scenario_name)

        summary = track_summary(analyzed)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Observations", f"{len(analyzed):,}")
        c2.metric("Unique Tracks", f"{analyzed['track_id'].nunique():,}")
        c3.metric(
            "QCAUS High-Threat Tracks",
            f"{(summary['qcaus_label'] == 'HIGH').sum():,}",
        )
        c4.metric(
            "Anomaly-Flagged Tracks",
            f"{int(summary['qcaus_anomaly_flag'].sum()):,}",
        )

        if {"latitude", "longitude"}.issubset(analyzed.columns):
            fig = px.scatter_mapbox(
                analyzed,
                lat="latitude",
                lon="longitude",
                color="qcaus_label",
                hover_name="track_id",
                hover_data=["qcaus_threat_score", "baseline_threat_score"],
                animation_frame="timestamp",
                zoom=4,
                height=600,
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add latitude and longitude columns to enable geographic replay.")

    with tab2:
        st.subheader("Data Quality & Schema")
        st.write(f"**Rows loaded:** {len(df):,}")
        st.write(f"**Columns detected:** {len(df.columns)}")

        schema_rows = []
        for col in df.columns:
            schema_rows.append({
                "Column": col,
                "Type": str(df[col].dtype),
                "Non-null": int(df[col].notna().sum()),
                "Missing": int(df[col].isna().sum()),
            })
        st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

        st.subheader("Preview")
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Scenario Replay")

        if "timestamp" in analyzed.columns:
            timestamps = (
                analyzed["timestamp"]
                .dropna()
                .drop_duplicates()
                .sort_values()
            )

            if len(timestamps) > 1:
                min_time = timestamps.iloc[0].to_pydatetime()
                max_time = timestamps.iloc[-1].to_pydatetime()

                selected_time = st.slider(
                    "Replay timestamp",
                    min_value=min_time,
                    max_value=max_time,
                    value=min_time,
                )

                selected_ts = pd.Timestamp(selected_time)
                if selected_ts.tzinfo is None:
                    selected_ts = selected_ts.tz_localize("UTC")
                else:
                    selected_ts = selected_ts.tz_convert("UTC")

                current = analyzed[analyzed["timestamp"] <= selected_ts].copy()
            else:
                current = analyzed.copy()
        else:
            current = analyzed.copy()
        else:
            current = analyzed

        if {"latitude", "longitude"}.issubset(current.columns):
            fig = px.scatter_mapbox(
                current,
                lat="latitude",
                lon="longitude",
                color="qcaus_label",
                hover_name="track_id",
                hover_data=["qcaus_threat_score", "qcaus_anomaly_score"],
                zoom=4,
                height=600,
            )
            fig.update_layout(mapbox_style="open-street-map")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            current.sort_values("qcaus_threat_score", ascending=False).head(100),
            use_container_width=True,
            hide_index=True,
        )

    with tab4:
        st.subheader("Track Investigation")

        summary = track_summary(analyzed)
        selected_track = st.selectbox(
            "Select track",
            options=summary["track_id"].astype(str).tolist(),
        )

        track = analyzed[analyzed["track_id"].astype(str) == selected_track].copy()
        latest = track.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("QCAUS Threat Score", f"{latest['qcaus_threat_score']:.1f}/100")
        c2.metric("Baseline Score", f"{latest['baseline_threat_score']:.1f}/100")
        c3.metric("Score Delta", f"{latest['score_delta']:+.1f}")
        c4.metric("Anomaly", "HIGH" if latest["qcaus_anomaly_flag"] else "LOW")

        st.markdown("### Why did QCAUS raise or lower the score?")

        reasons = []
        if latest["speed_anomaly"] >= 0.45:
            reasons.append("Kinematic speed behavior is inconsistent with the track's historical baseline.")
        if latest["heading_anomaly"] >= 0.45:
            reasons.append("Heading changes exceed the configured demonstration threshold.")
        if latest["gap_factor"] >= 0.45:
            reasons.append("Track continuity contains a significant observation gap.")
        if latest["sensor_agreement"] >= 0.75:
            reasons.append("Multiple sensor sources contribute to the track assessment.")
        if not reasons:
            reasons.append("No strong anomaly indicators were triggered by the demonstration scoring layer.")

        for reason in reasons:
            st.write(f"• {reason}")

        chart_df = track[[
            "timestamp",
            "baseline_threat_score",
            "qcaus_threat_score",
        ]].set_index("timestamp")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df["baseline_threat_score"],
            mode="lines",
            name="Baseline",
        ))
        fig.add_trace(go.Scatter(
            x=chart_df.index,
            y=chart_df["qcaus_threat_score"],
            mode="lines",
            name="QCAUS",
        ))
        fig.update_layout(
            title=f"Track {selected_track}: Baseline vs QCAUS",
            yaxis_title="Threat Score",
            xaxis_title="Time",
            yaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(track, use_container_width=True, hide_index=True)

    with tab5:
        st.subheader("Baseline vs QCAUS")

        summary = track_summary(analyzed)
        comparison = pd.DataFrame({
            "Metric": [
                "Tracks",
                "High-threat tracks",
                "Anomaly-flagged tracks",
                "Average threat score",
            ],
            "Baseline": [
                len(summary),
                int((summary["baseline_label"] == "HIGH").sum()),
                "N/A",
                f"{summary['baseline_threat_score'].mean():.1f}",
            ],
            "QCAUS": [
                len(summary),
                int((summary["qcaus_label"] == "HIGH").sum()),
                int(summary["qcaus_anomaly_flag"].sum()),
                f"{summary['qcaus_threat_score'].mean():.1f}",
            ],
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        fig = px.scatter(
            summary,
            x="baseline_threat_score",
            y="qcaus_threat_score",
            hover_name="track_id",
            labels={
                "baseline_threat_score": "Baseline Threat Score",
                "qcaus_threat_score": "QCAUS Threat Score",
            },
            title="Track-Level Baseline vs QCAUS Scores",
        )
        fig.add_shape(
            type="line",
            x0=0, y0=0, x1=100, y1=100,
            line=dict(dash="dash"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Track Comparison CSV",
            data=to_csv_bytes(summary),
            file_name="qcaus_track_comparison.csv",
            mime="text/csv",
        )

    with tab6:
        st.subheader("Validation Report")

        metrics = validation_metrics(analyzed)
        if metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{metrics['precision']:.3f}")
            c2.metric("Recall", f"{metrics['recall']:.3f}")
            c3.metric("F1", f"{metrics['f1']:.3f}")
        else:
            st.info(
                "No recognized ground-truth labels were available. "
                "Add a ground_truth column containing 'threat' or 'normal' "
                "to calculate validation metrics."
            )

        report = make_report(analyzed, scenario_name, notes)
        st.markdown(report)

        st.download_button(
            "Download Markdown Validation Report",
            data=report.encode("utf-8"),
            file_name="qcaus_validation_report.md",
            mime="text/markdown",
        )

        st.download_button(
            "Download Analyzed Results CSV",
            data=to_csv_bytes(analyzed),
            file_name="qcaus_analyzed_results.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
