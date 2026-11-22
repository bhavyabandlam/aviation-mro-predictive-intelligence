"""Report export helpers (SRS v1.1 FR-13).

Exports dashboard/report artefacts to CSV, HTML and PNG using only already-
pinned dependencies (pandas, matplotlib). No new heavy libraries are added;
PDF is intentionally omitted because SRS marks it optional.

Every exported artefact carries: a generated UTC timestamp, source lineage and
a synthetic-data warning where relevant.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPORTS_DIR = PROJECT_ROOT / "data" / "exports"

SYNTHETIC_WARNING = (
    "This report contains synthetic MRO business data generated locally "
    "(seed=42). It is not real airline records and must not be used for "
    "operational decisions."
)

__all__ = [
    "EXPORTS_DIR",
    "SYNTHETIC_WARNING",
    "export_dataframe_csv",
    "export_dataframe_html",
    "export_figure_png",
    "export_report",
]


def _timestamp() -> str:
    """Return the current time as an ISO-8601 UTC string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def export_dataframe_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a DataFrame to CSV (no index). Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def export_dataframe_html(
    df: pd.DataFrame,
    path: str | Path,
    title: str = "Aviation MRO Report",
    source_lineage: str | None = None,
    synthetic: bool = False,
) -> Path:
    """Write a DataFrame to a self-contained HTML report.

    The HTML includes the generation timestamp, optional source lineage and an
    optional synthetic-data warning (FR-13).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lineage = f"<p><b>Source lineage:</b> {source_lineage}</p>" if source_lineage else ""
    warning = (
        f'<p style="color:#B23B00;"><b>Warning:</b> {SYNTHETIC_WARNING}</p>' if synthetic else ""
    )
    html = (
        "<!DOCTYPE html>\n"
        f'<html><head><meta charset="utf-8"><title>{title}</title></head>\n'
        "<body>\n"
        f"<h1>{title}</h1>\n"
        f"<p><b>Generated:</b> {_timestamp()}</p>\n"
        f"{lineage}"
        f"{warning}"
        f"{df.to_html(index=False, border=1)}"
        "</body></html>"
    )
    path.write_text(html, encoding="utf-8")
    return path


def export_figure_png(fig: Any, path: str | Path, dpi: int = 150) -> Path:
    """Save a matplotlib figure to PNG. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def export_report(
    report_name: str,
    frames: dict[str, pd.DataFrame],
    out_dir: str | Path = EXPORTS_DIR,
    source_lineage: str | None = None,
    synthetic: bool = False,
) -> dict[str, Path]:
    """Export named frames as CSV + HTML + a metadata sidecar.

    Returns a mapping of logical name -> written path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _timestamp()
    written: dict[str, Path] = {}
    for name, df in frames.items():
        base = out_dir / f"{report_name}_{name}"
        written[f"{name}_csv"] = export_dataframe_csv(df, base.with_suffix(".csv"))
        written[f"{name}_html"] = export_dataframe_html(
            df,
            base.with_suffix(".html"),
            title=f"{report_name} — {name}",
            source_lineage=source_lineage,
            synthetic=synthetic,
        )
    metadata: dict[str, Any] = {
        "report_name": report_name,
        "generated_utc": stamp,
        "source_lineage": source_lineage,
        "synthetic": synthetic,
        "frames": list(frames.keys()),
        "files": {k: str(v) for k, v in written.items()},
    }
    meta_path = out_dir / f"{report_name}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written["metadata"] = meta_path
    return written
