"""Anomaly Analysis page (NASA C-MAPSS FD001).

Renders the IsolationForest anomaly model card: threshold, contamination
sweep, score distributions, per-engine anomaly rates and sensor attributions.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl


def _render_threshold(card: dict[str, Any]) -> None:
    ui.section("Detection Threshold")
    test = card.get("test_flagged_summary", {}) or {}
    val = card.get("val_flagged_summary", {}) or {}
    ui.kpi_row(
        [
            {
                "label": "Threshold (score)",
                "value": ui.fmt_num(card.get("default_threshold", 0.0), 4),
                "helper": "decision_function cutoff",
            },
            {
                "label": "Train Contamination (fraction)",
                "value": ui.fmt_num(card.get("default_contamination", 0.0), 4),
                "helper": "target anomaly fraction",
            },
            {
                "label": "Val Anomaly Rate (%)",
                "value": ui.fmt_pct((val.get("anomaly_rate") or 0) * 100, 2),
                "helper": f"{val.get('n_flagged')} flagged",
            },
            {
                "label": "Test Anomaly Rate (%)",
                "value": ui.fmt_pct((test.get("anomaly_rate") or 0) * 100, 2),
                "helper": f"{test.get('n_flagged')} flagged",
            },
        ]
    )


def _render_score_dist(card: dict[str, Any]) -> None:
    ui.section("Score Distributions")
    dist = card.get("score_distributions", {}) or {}
    rows = []
    for split in ("train", "val", "test"):
        d = dist.get(split, {}) or {}
        if not d:
            continue
        pct = d.get("percentiles", {}) or {}
        rows.append(
            {
                "Split": split,
                "Mean": round(d.get("mean", 0.0), 4),
                "Std": round(d.get("std", 0.0), 4),
                "Min": round(d.get("min", 0.0), 4),
                "p1": round(pct.get("1%", 0.0), 4),
                "p50": round(pct.get("50%", 0.0), 4),
                "Max": round(d.get("max", 0.0), 4),
            }
        )
    if rows:
        ui.data_table(rows)
    else:
        st.info("No score distributions recorded.")


def _render_per_engine(card: dict[str, Any]) -> None:
    ui.section("Per-Engine Test Anomaly Rate")
    per_engine = (card.get("test_flagged_summary", {}) or {}).get("per_engine", {}) or {}
    if per_engine:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Engine": list(per_engine.keys()),
                "Anomaly Rate": [round(v * 100, 3) for v in per_engine.values()],
            }
        )
        df = df.sort_values("Anomaly Rate", ascending=False)
        # Chart with text/table companion so chart info is perceivable without colour.
        ui.chart_with_description(
            lambda d, **kw: st.bar_chart(d.set_index("Engine"), height=360),
            df,
            description="Per-engine test anomaly rate (%) — bar chart",
            caption="Each bar shows the percentage of flagged observations for one engine.",
        )
    top = sorted(per_engine.items(), key=lambda kv: kv[1], reverse=True)[:5]
    st.write("**Highest-risk engines (test):**")
    for eng, rate in top:
        st.markdown(f"- Engine {eng}: {rate * 100:.2f}% anomalies")

    # CSV export for per-engine anomaly rates
    if per_engine:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Engine": list(per_engine.keys()),
                "Anomaly Rate (%)": [round(v * 100, 3) for v in per_engine.values()],
            }
        )
        ui.csv_export_button(df, "per_engine_anomaly_rates.csv")
    else:
        st.info("No per-engine anomaly rates recorded.")


def _render_attributions(card: dict[str, Any]) -> None:
    ui.section("Sensor Attributions (Flagged Observations)")
    attributions = card.get("sensor_attributions_test", []) or []
    if not attributions:
        attributions = card.get("sensor_attributions_val", []) or []
    if not attributions:
        st.info("No sensor attributions recorded.")
        return
    rows = []
    for item in attributions:
        top = item.get("top_sensors", []) or []
        top_str = ", ".join(f"{name} ({contrib:.1f})" for name, contrib in top)
        rows.append(
            {
                "Unit": item.get("unit_id"),
                "Cycle": item.get("cycle"),
                "Anomaly Score": round(item.get("anomaly_score", 0.0), 4),
                "Top Sensors": top_str,
            }
        )
    ui.data_table(rows, max_rows=50)
    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        ui.csv_export_button(df, "sensor_attributions.csv")


def _render_sweep(card: dict[str, Any]) -> None:
    sweep = card.get("contamination_sweep", {}) or {}
    if not sweep:
        return
    ui.section("Contamination Sweep (threshold sensitivity)")
    rows = []
    for cont, r in sweep.items():
        if not isinstance(r, dict):
            continue
        rows.append(
            {
                "Contamination": cont,
                "Threshold": round(r.get("threshold", 0.0), 4),
                "Train Rate": round(r.get("train_anomaly_rate", 0.0), 4),
                "Val Rate": round(r.get("val_anomaly_rate", 0.0), 4),
                "Test Rate": round(r.get("test_anomaly_rate", 0.0), 4),
            }
        )
    if rows:
        ui.data_table(rows)


def render() -> None:
    """Render the Anomaly Detection page."""
    st.set_page_config(page_title="Anomaly Analysis", page_icon="🚨", layout="wide")
    ui.page_header(
        "Anomaly Analysis",
        "Unsupervised IsolationForest scoring of engine sensor data",
        "nasa",
    )
    ui.freshness_badge(dl.REPORTS_DIR / "anomaly_model_card.json", "Anomaly model card freshness")
    st.divider()

    card = dl.load_anomaly_model_card()
    if card is None:
        ui.missing_artifact("anomaly model card", "reports/anomaly_model_card.json")
        return

    _render_threshold(card)
    _render_score_dist(card)
    _render_per_engine(card)
    _render_sweep(card)
    _render_attributions(card)

    # Connect to the live artefact for confirmation.
    ui.section("Saved Model Artefact")
    detector = dl.load_anomaly_detector()
    if detector is None:
        st.warning("anomaly_model.pkl present but could not be loaded.")
    else:
        st.success(
            f"Loaded IsolationForest with {len(detector.feature_names)} features "
            f"and threshold {detector.threshold_:.4f}."
        )
