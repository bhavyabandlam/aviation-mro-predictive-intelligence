"""Data Sources & Model Registry page.

Renders data-source provenance for the whole platform: the sources documented
in ``docs/DATA_SOURCES.md`` (summarised here), the provenance legend, saved
model artefacts and their dataset fingerprints, and per-report reproducibility
metadata (random seed + fingerprint). Read-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from dashboard import components as ui
from dashboard import data_loader as dl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_SOURCES_MD = PROJECT_ROOT / "docs" / "DATA_SOURCES.md"


def _render_legend() -> None:
    ui.section("Provenance Legend")
    st.markdown(
        "- **NASA C-MAPSS FD001** — turbofan run-to-failure *simulation* published "
        "by NASA PCoE. Powers EDA, RUL prediction and anomaly detection. Simulated "
        "benchmark, not real fleet records.\n"
        "- **Synthetic MRO** — weekly maintenance demand, inventory and shop "
        "parameters generated locally (seed=42). Powers forecasting and inventory.\n"
        "- **Synthetic Financial** — labour, downtime, holding and stockout cost "
        "parameters generated locally (seed=42). Powers financial analytics.\n"
        "- **Mixed** — the Executive Overview combines NASA-simulated and "
        "synthetic MRO sources."
    )


def _render_sources_table() -> None:
    ui.section("Documented Data Sources")
    rows = [
        {
            "ID": "nasa_cmapss_fd001",
            "Title": "NASA C-MAPSS FD001 Turbofan Degradation (simulated)",
            "Type": "NASA simulated benchmark",
            "Used by": "EDA, RUL, Anomaly",
        },
        {
            "ID": "synthetic_mro_v1",
            "Title": "Synthetic Weekly Maintenance Demand",
            "Type": "Synthetic (seed=42)",
            "Used by": "Forecasting, Inventory",
        },
        {
            "ID": "synthetic_inventory_v1",
            "Title": "Synthetic Inventory Part Master (150 parts)",
            "Type": "Synthetic (seed=42)",
            "Used by": "Inventory, Financial",
        },
        {
            "ID": "synthetic_financial_v1",
            "Title": "Synthetic MRO Cost Parameters",
            "Type": "Synthetic (seed=42)",
            "Used by": "Financial",
        },
    ]
    ui.data_table(rows)
    if DATA_SOURCES_MD.exists():
        st.caption("Full onboarding detail (checksums, citations, schemas): docs/DATA_SOURCES.md")
    else:
        st.caption("docs/DATA_SOURCES.md not found.")


def _render_reproducibility() -> None:
    ui.section("Reproducibility (seeds & fingerprints)")
    reports = {
        "RUL model card": dl.load_rul_model_card(),
        "Anomaly model card": dl.load_anomaly_model_card(),
        "Forecasting": dl.load_forecast_results(),
        "Inventory": dl.load_inventory_analysis(),
        "Financial": dl.load_financial_analysis(),
        "Explainability (SHAP)": dl.load_rul_explanation(),
    }

    def _extract(report: dict[str, Any] | None) -> dict[str, str]:
        if not report:
            return {"status": "missing"}
        seed = (
            report.get("random_seed")
            or (report.get("feature_config") or {}).get("random_seed")
            or (report.get("metadata") or {}).get("random_seed")
        )
        meta = report.get("metadata") or {}
        fp = (
            report.get("dataset_fingerprint")
            or meta.get("data_fingerprint")
            or meta.get("inventory_fingerprint")
        )
        if isinstance(fp, dict):
            fp = next(iter(fp.values()), None)
        return {
            "random_seed": str(seed) if seed is not None else "—",
            "fingerprint": str(fp) if fp is not None else "—",
        }

    rows = []
    for name, report in reports.items():
        info = _extract(report)
        rows.append(
            {
                "Report": name,
                "Random seed": info.get("random_seed", "—"),
                "Data fingerprint": info.get("fingerprint", "—"),
                "Status": "present" if report else "missing",
            }
        )
    ui.data_table(rows)
    st.caption(
        "All synthetic generators use seed=42 for reproducibility. Fingerprints are "
        "short SHA-256 hashes of the underlying data."
    )


def _render_artifacts() -> None:
    ui.section("Saved Model Artefacts")
    status = dl.model_artifact_status()
    ui.data_table(
        [
            {
                "Artefact": name,
                "Present": "Yes" if info["exists"] else "No",
                "Size (KB)": info["size_kb"],
                "Path": info["path"],
            }
            for name, info in status.items()
        ]
    )


def _render_sources_md() -> None:
    if not DATA_SOURCES_MD.exists():
        return
    with st.expander("docs/DATA_SOURCES.md (raw)", expanded=False):
        text = DATA_SOURCES_MD.read_text(encoding="utf-8")
        # Show section headers only, to keep it navigable.
        headers = [ln for ln in text.splitlines() if ln.startswith("## ")]
        st.markdown("\n".join(headers) if headers else "_No section headers found._")


def _render_model_registry() -> None:
    """Render model registry entries from saved artefacts."""
    ui.section("Model Registry (FR-14)")
    try:
        from aviation_mro.persistence.database import MRODatabase
        from aviation_mro.services.model_registry import register_saved_artefacts

        db = MRODatabase()
        db.create_schema()
        entries = register_saved_artefacts(db)
        if entries:
            rows = []
            for entry in entries:
                rows.append(
                    {
                        "Model ID": entry.model_id,
                        "Model Name": entry.model_name,
                        "Version": entry.version,
                        "Algorithm": entry.algorithm,
                        "Artifact Path": entry.artifact_path,
                        "Training Date": entry.training_date or "—",
                        "Random Seed": entry.random_seed,
                    }
                )
            ui.data_table(rows)
            import pandas as pd

            df = pd.DataFrame(rows)
            ui.csv_export_button(df, "model_registry.csv")
            ui.export_section(
                {"model_registry": df},
                "provenance_model_registry",
                source_lineage="NASA FD001 + synthetic MRO business data",
                synthetic=True,
                label="Export model registry (CSV + HTML)",
            )
        else:
            st.info("No model registry entries found.")
    except Exception as exc:  # pragma: no cover - defensive
        st.warning(f"Model registry unavailable: {exc}")


def render() -> None:
    """Render the Data Provenance & Sources page."""
    st.set_page_config(page_title="Data Sources & Model Registry", page_icon="🗂️", layout="wide")
    ui.page_header(
        "Data Sources & Model Registry",
        "Where every dataset and model artefact comes from, and how it is reproduced",
        "mixed",
    )
    ui.freshness_badge(dl.REPORTS_DIR / "model_card_baseline_v2.json", "Model registry freshness")
    st.divider()

    _render_legend()
    _render_sources_table()
    _render_reproducibility()
    _render_artifacts()
    _render_model_registry()
    _render_sources_md()
