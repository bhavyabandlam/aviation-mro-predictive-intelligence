"""Model Performance & Explainability page.

Full SHAP explanation view for the Section 9.2 RUL model. Reads the
precomputed ``reports/explainability/rul_shap_summary.json`` (generated
offline) and presents:

- methodology and model compatibility summary,
- global feature importance (mean |SHAP|),
- an individual-prediction explorer (per-engine top contributions),
- feature-space and causation caveats,
- how to regenerate the explanations.

No SHAP values are computed in the dashboard and none are fabricated.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl


def _render_methodology(expl: dict[str, Any]) -> None:
    ui.section("Methodology & Compatibility")
    consistency = expl.get("consistency", {}) or {}
    ui.kpi_row(
        [
            {"label": "SHAP Version", "value": expl.get("shap_version", "n/a")},
            {"label": "Explainer", "value": expl.get("explainer_type", "n/a")},
            {"label": "Model", "value": expl.get("model_algorithm", "n/a")},
            {
                "label": "Consistency",
                "value": "Passed" if consistency.get("passed") else "Failed",
                "helper": f"max abs err {consistency.get('max_abs_error')}",
            },
        ]
    )
    st.markdown(
        f"- **Model version:** `{expl.get('model_version')}` "
        f"(reused as-is; not retrained)\n"
        f"- **Features explained:** {expl.get('n_features')} "
        f"(exact training order)\n"
        f"- **Background samples:** {expl.get('background_size')}\n"
        f"- **Additivity check:** `base_value + sum(SHAP) == prediction` — "
        f"{'PASS' if consistency.get('passed') else 'FAIL'} "
        f"(tolerance {consistency.get('tolerance')})"
    )


def _render_global(expl: dict[str, Any]) -> None:
    ui.section("Global Feature Importance")
    st.caption("Mean |SHAP| across all test engines — which sensors drive predictions most.")
    importance = expl.get("global_importance", []) or []
    top = importance[:20]
    if not top:
        st.info("No global importance recorded.")
        return
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "Feature": [g["feature"] for g in top][::-1],
                "mean_abs_shap": [g["mean_abs_shap"] for g in top][::-1],
            }
        )
        st.bar_chart(df.set_index("Feature"), height=420)
    except Exception:
        pass
    ui.data_table(
        [
            {"Rank": g["rank"], "Feature": g["feature"], "Mean |SHAP|": g["mean_abs_shap"]}
            for g in top
        ]
    )


def _render_local(expl: dict[str, Any]) -> None:
    ui.section("Individual Prediction Explorer")
    locals_ = expl.get("local_explanations", []) or []
    if not locals_:
        st.info("No per-engine explanations recorded.")
        return
    options = {
        f"Engine {le['unit_id']} (cycle {le['cycle']}, pred {le['prediction']:.1f}, "
        f"|err| {le['absolute_error']:.2f})": idx
        for idx, le in enumerate(locals_)
    }
    choice = st.selectbox("Select an engine to explain", list(options.keys()), key="xai_engine")
    le = locals_[options[choice]]
    ui.kpi_row(
        [
            {"label": "Predicted RUL", "value": ui.fmt_num(le["prediction"], 2)},
            {"label": "Base Value (mean)", "value": ui.fmt_num(le["base_value"], 2)},
            {"label": "Absolute Error", "value": ui.fmt_num(le["absolute_error"], 3)},
            {"label": "Additivity Error", "value": f"{le['consistency_max_abs_error']:.2e}"},
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
            st.bar_chart(df.set_index("Feature"), height=340)
        except Exception:
            pass
        ui.data_table(
            [
                {
                    "Feature": t["feature"],
                    "SHAP value": round(t["shap_value"], 4),
                    "Effect": (
                        "Increases RUL (+)" if t["shap_value"] >= 0 else "Decreases RUL (-)"
                    ),
                    "Scaled value": round(le["feature_values_scaled"].get(t["feature"], 0.0), 4),
                    "Raw value": round(le["feature_values_raw"].get(t["feature"], 0.0), 4),
                }
                for t in top
            ]
        )
    st.info(
        f"How to read: baseline prediction {le['base_value']:.1f} cycles. "
        f"Each SHAP value pushes the prediction up (+) or down (-) to reach "
        f"{le['prediction']:.1f}. Green = increases RUL, red = decreases RUL."
    )


def _render_regeneration() -> None:
    with st.expander("How to regenerate these explanations", expanded=False):
        st.code("python -m aviation_mro.explainability.rul_explainer")
        st.caption(
            "Requires shap==0.54.1. Writes reports/explainability/rul_shap_summary.json. "
            "Reuses the existing RUL model artefact; does not retrain."
        )


def render() -> None:
    """Render the Explainability (SHAP) page."""
    st.set_page_config(
        page_title="Model Performance & Explainability",
        page_icon="🔎",
        layout="wide",
    )
    ui.page_header(
        "Model Performance & Explainability",
        "Global and per-prediction feature attributions for the RUL model",
        "nasa",
    )
    ui.freshness_badge(
        dl.REPORTS_DIR / "explainability" / "rul_shap_summary.json",
        "SHAP explanation freshness",
    )
    st.divider()

    expl = dl.load_rul_explanation()
    if expl is None:
        ui.missing_artifact("SHAP summary", "reports/explainability/rul_shap_summary.json")
        _render_regeneration()
        return

    _render_methodology(expl)
    _render_global(expl)
    _render_local(expl)

    st.warning(
        "SHAP contributions describe how features push a prediction for a specific "
        "engine. They reflect correlations in the NASA FD001 simulated data and DO "
        "NOT establish causation."
    )
    st.caption(expl.get("feature_space_note", ""))
    ui.limitations_box(expl.get("limitations", []) or [], title="Explainability limitations")
    _render_regeneration()
