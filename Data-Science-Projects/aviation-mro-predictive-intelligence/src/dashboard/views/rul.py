"""Engine Health & RUL page (NASA C-MAPSS FD001).

Renders the RUL model card: model comparison, test metrics, per-engine error
distribution, and the documented root-cause/findings. Optionally connects to
the saved ``baseline_rul_model_v2.pkl`` artefact to report its metadata.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl


def _render_metrics(card: dict[str, Any]) -> None:
    test = card.get("test_metrics", {}) or {}
    ui.section("Test Set Metrics (GradientBoosting)")
    ui.kpi_row(
        [
            {"label": "Test MAE (cycles)", "value": ui.fmt_num(test.get("mae", 0.0), 2)},
            {"label": "Test RMSE (cycles)", "value": ui.fmt_num(test.get("rmse", 0.0), 2)},
            {"label": "Test R2 (dimensionless)", "value": ui.fmt_num(test.get("r2", 0.0), 3)},
            {
                "label": "Best Model",
                "value": card.get("best_model", "n/a"),
                "helper": f"{card.get('n_features')} features",
            },
        ]
    )


def _render_comparison(card: dict[str, Any]) -> None:
    ui.section("Model Comparison (Validation)")
    comparison = card.get("model_comparison", []) or []
    if not comparison:
        st.info("No model comparison recorded in the model card.")
        return
    # Bar chart of validation R2 by model.
    labels = [m.get("name", "") for m in comparison]
    r2_vals = [round(m.get("val_r2", 0.0), 4) for m in comparison]
    try:
        import pandas as pd

        df = pd.DataFrame({"Model": labels, "Validation R2": r2_vals})
        st.caption("Bar chart: model name on the x-axis, validation R2 on the y-axis.")
        st.bar_chart(df.set_index("Model"))
    except Exception:
        st.write("Model comparison:")
    rows = [
        {
            "Model": m.get("name"),
            "Type": m.get("model_type"),
            "Val MAE": round(m.get("val_mae", 0.0), 3),
            "Val RMSE": round(m.get("val_rmse", 0.0), 3),
            "Val R2": round(m.get("val_r2", 0.0), 3),
        }
        for m in comparison
    ]
    ui.data_table(rows)


def _render_per_engine(card: dict[str, Any]) -> None:
    ui.section("Per-Engine Test Error Distribution")
    errors = card.get("test_per_engine_errors", {}) or {}
    summary = card.get("test_error_summary", {}) or {}
    if errors:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Engine": list(errors.keys()),
                "MAE": [round(v, 3) for v in errors.values()],
            }
        )
        # Chart with text/table companion so chart info is perceivable without colour.
        ui.chart_with_description(
            lambda d, **kw: st.bar_chart(d.set_index("Engine"), height=360),
            df,
            description="Per-engine test MAE (cycles) — bar chart",
            caption="Each bar shows the mean absolute error for one engine in cycles.",
        )
        ui.kpi_row(
            [
                {"label": "Mean (cycles)", "value": ui.fmt_num(summary.get("mean", 0.0), 2)},
                {"label": "Median (cycles)", "value": ui.fmt_num(summary.get("median", 0.0), 2)},
                {"label": "Min (cycles)", "value": ui.fmt_num(summary.get("min", 0.0), 2)},
                {"label": "Max (cycles)", "value": ui.fmt_num(summary.get("max", 0.0), 2)},
            ]
        )
    else:
        st.info("No per-engine errors recorded.")

    # CSV export for per-engine errors
    if errors:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Engine": list(errors.keys()),
                "MAE (cycles)": [round(v, 3) for v in errors.values()],
            }
        )
        ui.csv_export_button(df, "per_engine_errors.csv")


def _render_artefact() -> None:
    ui.section("Saved Model Artefact")
    artifact = dl.load_rul_artifact()
    if artifact is None:
        st.warning(
            "RUL model artefact present on disk but could not be loaded directly "
            "in this context (it was pickled from a __main__ training session). "
            "Metrics above are read from the saved model card instead."
        )
        return
    st.success(
        f"Loaded `baseline_rul_model_v2.pkl` - best model "
        f"**{getattr(artifact, 'best_model_name', 'n/a')}** with "
        f"{len(getattr(artifact, 'feature_names', []))} features."
    )
    cfg = getattr(artifact, "feature_config", {}) or {}
    if cfg:
        st.json(cfg)


def _render_explainability() -> None:
    """Render the SHAP explainability section for the RUL model."""
    st.divider()
    ui.section("Explainability (SHAP)")

    expl = dl.load_rul_explanation()
    if expl is None:
        st.info(
            "No SHAP explanations found. Generate them offline with "
            "`python -m aviation_mro.explainability.rul_explainer` "
            "(writes reports/explainability/rul_shap_summary.json)."
        )
        return

    consistency = expl.get("consistency", {}) or {}
    ui.kpi_row(
        [
            {"label": "SHAP Version", "value": expl.get("shap_version", "n/a")},
            {"label": "Explainer", "value": expl.get("explainer_type", "n/a")},
            {
                "label": "Background Samples",
                "value": expl.get("background_size", "n/a"),
            },
            {
                "label": "Consistency",
                "value": "Passed" if consistency.get("passed") else "Failed",
                "helper": f"max abs err {consistency.get('max_abs_error')}",
            },
        ]
    )
    st.caption(
        f"Model version: {expl.get('model_version')} | "
        f"Algorithm: {expl.get('model_algorithm')} | "
        f"Features: {expl.get('n_features')}"
    )

    # ---- Global feature importance ---------------------------------------
    st.markdown("**Global feature importance (mean |SHAP| over test engines)**")
    importance = expl.get("global_importance", []) or []
    top_global = importance[:15]
    if top_global:
        try:
            import pandas as pd

            df = pd.DataFrame(
                {
                    "Feature": [g["feature"] for g in top_global][::-1],
                    "mean_abs_shap": [g["mean_abs_shap"] for g in top_global][::-1],
                }
            )
            st.bar_chart(df.set_index("Feature"), height=360)
        except Exception:
            pass
        ui.data_table(
            [
                {
                    "Rank": g["rank"],
                    "Feature": g["feature"],
                    "Mean |SHAP|": g["mean_abs_shap"],
                }
                for g in top_global
            ]
        )

    # ---- Local explanation explorer --------------------------------------
    locals_ = expl.get("local_explanations", []) or []
    if locals_:
        st.markdown("**Individual prediction explanation**")
        options = {
            f"Engine {le['unit_id']} (cycle {le['cycle']}, "
            f"pred {le['prediction']:.1f}, |err| {le['absolute_error']:.2f})": idx
            for idx, le in enumerate(locals_)
        }
        choice = st.selectbox(
            "Select an engine to explain", list(options.keys()), key="shap_engine"
        )
        le = locals_[options[choice]]
        ui.kpi_row(
            [
                {"label": "Predicted RUL", "value": ui.fmt_num(le["prediction"], 2)},
                {"label": "Base Value (mean)", "value": ui.fmt_num(le["base_value"], 2)},
                {"label": "Absolute Error", "value": ui.fmt_num(le["absolute_error"], 3)},
                {
                    "label": "Consistency",
                    "value": f"{le['consistency_max_abs_error']:.2e}",
                },
            ]
        )
        top = le.get("top_contributions", []) or []
        if top:
            try:
                import pandas as pd

                df = pd.DataFrame(
                    {
                        "Feature": [t["feature"] for t in top][::-1],
                        "SHAP value": [t["shap_value"] for t in top][::-1],
                    }
                )
                st.bar_chart(df.set_index("Feature"), height=320)
            except Exception:
                pass
            ui.data_table(
                [
                    {
                        "Feature": t["feature"],
                        "SHAP value": round(t["shap_value"], 4),
                        "Effect": "Increases RUL" if t["shap_value"] >= 0 else "Decreases RUL",
                        "Scaled value": round(
                            le["feature_values_scaled"].get(t["feature"], 0.0), 4
                        ),
                    }
                    for t in top
                ]
            )
        st.caption(
            f"How to read: the model's baseline prediction is "
            f"{le['base_value']:.1f} cycles. Each SHAP value pushes the "
            f"prediction up (+) or down (-) from that baseline to reach "
            f"{le['prediction']:.1f}."
        )

    # ---- Caveats ----------------------------------------------------------
    st.warning(
        "SHAP contributions show how features push the prediction for a "
        "specific engine. They reflect correlations in the NASA FD001 simulated "
        "data and DO NOT establish causation."
    )
    st.caption(expl.get("feature_space_note", ""))
    ui.limitations_box(expl.get("limitations", []) or [], title="Explainability limitations")


def render() -> None:
    """Render the RUL Prediction & Explainability page."""
    st.set_page_config(page_title="Engine Health & RUL", page_icon="🔧", layout="wide")
    ui.page_header(
        "Engine Health & RUL",
        "Engine-level GradientBoosting model with leak-free validation and SHAP explanations",
        "nasa",
    )
    ui.freshness_badge(dl.REPORTS_DIR / "model_card_baseline_v2.json", "RUL model card freshness")
    st.divider()

    card = dl.load_rul_model_card()
    if card is None:
        ui.missing_artifact("RUL model card", "reports/model_card_baseline_v2.json")
        return

    _render_metrics(card)
    _render_comparison(card)
    _render_per_engine(card)
    _render_artefact()

    findings = card.get("findings", {}) or {}
    if findings:
        ui.section("Root-Cause & Fixes (v1 -> v2)")
        st.markdown(f"**Root cause:** {findings.get('previous_negative_r2_root_cause', '')}")
        fixes = findings.get("fixes_applied", []) or []
        if fixes:
            st.markdown("**Fixes applied:**")
            for fix in fixes:
                st.markdown(f"- {fix}")

    ui.limitations_box(card.get("findings", {}).get("limitations", []) or [])

    _render_explainability()
