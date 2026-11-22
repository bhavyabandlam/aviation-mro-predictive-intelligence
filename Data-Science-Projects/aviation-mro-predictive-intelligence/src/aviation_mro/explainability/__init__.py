"""Explainability (XAI) for the Aviation MRO platform.

Currently provides SHAP-based explanations for the Section 9.2 RUL prediction
model (scikit-learn ``GradientBoostingRegressor``).

All explanations are precomputed offline and persisted to
``reports/explainability/`` so the Streamlit dashboard never runs expensive
SHAP computation at page-load time.

Important: SHAP contributions describe how each feature *pushes* a model
prediction up or down for a specific input. They reflect learned correlations
in the training data and DO NOT establish causation.
"""

from __future__ import annotations

__all__ = ["rul_explainer"]
