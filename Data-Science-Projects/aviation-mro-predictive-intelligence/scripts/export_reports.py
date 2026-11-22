"""Command-line report exporter (SRS v1.1 FR-13).

Reads a JSON report artefact and exports its tabular contents to CSV and HTML
in ``data/exports/`` (default), including a timestamp, source lineage and
synthetic-data warning where relevant.

Example:

    python scripts/export_reports.py --report reports/financial/financial_analysis.json \
        --out-dir data/exports --synthetic

Only standard-library modules plus already-pinned pandas are used; no new
heavy dependencies are introduced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd  # noqa: E402

from aviation_mro.reporting import export_report  # noqa: E402


def _frame_for_value(key: str, value: Any) -> pd.DataFrame | None:
    """Best-effort conversion of a JSON value into a single-row/multi-row frame."""
    if value is None:
        return pd.DataFrame({key: [None]})
    if isinstance(value, (bool, int, float, str)):
        return pd.DataFrame({key: [value]})
    if isinstance(value, list):
        if not value:
            return pd.DataFrame({key: []})
        if all(isinstance(v, dict) for v in value):
            return pd.DataFrame(value)
        return pd.DataFrame({key: value})
    if isinstance(value, dict):
        # Flatten one level: {k: v} -> row with column f"{key}.{k}"
        return pd.DataFrame([{f"{key}.{sub_k}": sub_v for sub_k, sub_v in value.items()}])
    return None


def _load_report_frames(report_path: Path) -> dict[str, pd.DataFrame]:
    """Load a JSON report and return name -> DataFrame for each top-level key."""
    data: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
    frames: dict[str, pd.DataFrame] = {}
    for key, value in data.items():
        frame = _frame_for_value(key, value)
        if frame is not None and len(frame.columns) > 0:
            frames[key] = frame
    return frames


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a JSON report artefact to CSV and HTML (FR-13)."
    )
    parser.add_argument(
        "--report",
        required=True,
        type=Path,
        help="Path to the JSON report artefact to export.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Export directory (default: data/exports/).",
    )
    parser.add_argument(
        "--title",
        default="Aviation MRO Report",
        help="Report title embedded in the HTML export.",
    )
    parser.add_argument(
        "--source-lineage",
        default=None,
        help="Source lineage string embedded in the export metadata.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Add the synthetic-data warning to the export.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report_path: Path = args.report
    if not report_path.exists():
        sys.stderr.write(f"ERROR: report not found: {report_path}\n")
        return 2

    frames = _load_report_frames(report_path)
    if not frames:
        sys.stderr.write(f"ERROR: no exportable frames in {report_path}\n")
        return 2

    out_dir = args.out_dir
    report_name = report_path.stem

    written = export_report(
        report_name=report_name,
        frames=frames,
        out_dir=out_dir,
        source_lineage=args.source_lineage,
        synthetic=args.synthetic,
    )

    sys.stdout.write(f"Exported '{report_name}' to {out_dir}:\n")
    for name, path in written.items():
        sys.stdout.write(f"  - {name}: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
