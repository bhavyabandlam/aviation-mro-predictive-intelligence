"""Streamlit entry point for the Aviation MRO Predictive Intelligence dashboard.

Run from the project root:

    streamlit run src/dashboard/app.py

Pages (Sections 9.1-9.6):
    - Executive Overview
    - Data Explorer & Quality
    - Engine Health & RUL
    - Anomaly Analysis
    - Maintenance Forecasting
    - Inventory Intelligence
    - Financial Scenarios
    - Model Performance & Explainability
    - Data Sources & Model Registry

Every page reads only saved artefacts (JSON reports, Parquet datasets, model
cards). Missing artefacts surface a warning instead of crashing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``src`` is importable regardless of the current working directory.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from aviation_mro.services.logging import setup_logging  # noqa: E402

# Configure structured application logging (NFR-07) before any page imports.
setup_logging()

import streamlit as st  # noqa: E402

from dashboard.views import (  # noqa: E402
    anomaly,
    eda,
    executive,
    explainability,
    financial,
    forecasting,
    inventory,
    provenance,
    rul,
)


def main() -> None:
    """Configure navigation and run the selected page."""
    st.set_page_config(
        page_title="Aviation MRO Predictive Intelligence",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = [
        st.Page(executive.render, title="Executive Overview", url_path="home", default=True),
        st.Page(eda.render, title="Data Explorer & Quality", url_path="eda"),
        st.Page(
            rul.render,
            title="Engine Health & RUL",
            url_path="rul",
        ),
        st.Page(anomaly.render, title="Anomaly Analysis", url_path="anomaly"),
        st.Page(forecasting.render, title="Maintenance Forecasting", url_path="forecasting"),
        st.Page(inventory.render, title="Inventory Intelligence", url_path="inventory"),
        st.Page(financial.render, title="Financial Scenarios", url_path="financial"),
        st.Page(
            explainability.render,
            title="Model Performance & Explainability",
            url_path="explainability",
        ),
        st.Page(
            provenance.render,
            title="Data Sources & Model Registry",
            url_path="provenance",
        ),
    ]

    navigation = st.navigation(pages, position="sidebar")
    navigation.run()


if __name__ == "__main__":
    main()
