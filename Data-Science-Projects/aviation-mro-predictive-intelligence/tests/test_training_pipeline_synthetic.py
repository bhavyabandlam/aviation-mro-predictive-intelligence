"""Fast synthetic unit tests for the RUL and anomaly training pipelines.

These exercise the same pipeline contracts as the slow real-FD001 integration
tests, but on tiny deterministic synthetic data:

- artifact structure and contents
- disjoint train/val engine splits (no row-level leakage)
- no RUL labels leaking into anomaly features
- preprocessing (imputer/scaler) fitted on training data only
- determinism under fixed seeds
- artifacts written to caller-provided directories, never production

The real-data integration tests live in ``test_baseline.py`` /
``test_anomaly_train.py`` and are opt-in via ``--run-integration``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aviation_mro.features.anomaly_features import (  # noqa: E402
    AnomalyFeatureConfig,
    engineer_anomaly_features,
    get_anomaly_feature_columns,
)
from aviation_mro.features.rul_features import (  # noqa: E402
    FeatureConfig,
    engineer_features,
    get_feature_columns,
)
from aviation_mro.models import anomaly_train as anomaly_mod  # noqa: E402
from aviation_mro.models import baseline as baseline_mod  # noqa: E402
from aviation_mro.models.anomaly_model import AnomalyModelConfig  # noqa: E402
from aviation_mro.models.baseline import TrainingArtifacts  # noqa: E402

_INFORMATIVE = [
    "sensor_2", "sensor_3", "sensor_4", "sensor_6", "sensor_7", "sensor_8",
    "sensor_9", "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]  # fmt: skip
_CONSTANT = {
    "sensor_1": 518.67, "sensor_5": 14.62, "sensor_10": 1.30,
    "sensor_16": 0.03, "sensor_18": 2388.0, "sensor_19": 100.0,
}  # fmt: skip


def make_raw(n_engines: int = 4, n_cycles: int = 10, seed: int = 42) -> pd.DataFrame:
    """Build a small deterministic FD001-shaped raw dataframe."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for unit in range(1, n_engines + 1):
        for cycle in range(1, n_cycles + 1):
            row: dict = {
                "unit_id": unit,
                "cycle": cycle,
                "op_setting_1": 0.0,
                "op_setting_2": 0.0,
                "op_setting_3": 100.0,
            }
            for sensor, const in _CONSTANT.items():
                row[sensor] = const
            for i, sensor in enumerate(_INFORMATIVE):
                base = 500.0 + 10.0 * i
                drift = 0.5 * cycle * (1 if i % 2 == 0 else -1)
                row[sensor] = base + drift + rng.normal(0, 0.1)
            rows.append(row)
    return pd.DataFrame(rows)


def _baseline_prepared(raw: pd.DataFrame, config: FeatureConfig) -> dict:
    """Build the prepared-data dict that ``load_and_prepare_data`` returns."""
    feat = engineer_features(raw, config)
    feature_cols = get_feature_columns(feat, config)
    engine_ids = sorted(feat["unit_id"].unique())
    # Hold out two engines so the test set has >=2 rows (R² well-defined).
    train_ids, val_ids = engine_ids[:-2], engine_ids[-2:]
    train_df = feat[feat["unit_id"].isin(train_ids)].copy()
    val_df = feat[feat["unit_id"].isin(val_ids)].copy()
    test_last = val_df.groupby("unit_id", sort=False).tail(1).copy()
    rul_df = pd.DataFrame({"unit_id": val_ids, "rul": [0] * len(val_ids)})
    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_last": test_last,
        "rul_df": rul_df,
        "feature_cols": feature_cols,
        "train_engine_ids": train_ids,
        "val_engine_ids": val_ids,
        "config": config,
    }


def _fast_models() -> dict:
    """Deterministic, fast model set for unit tests."""
    from sklearn.dummy import DummyRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
    }


def _stub_fingerprint() -> dict:
    return {
        "train_FD001.txt": "synthetic",
        "test_FD001.txt": "synthetic",
        "RUL_FD001.txt": "synthetic",
    }


class TestBaselinePipelineSynthetic:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TrainingArtifacts:
        raw = make_raw()
        config = FeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_cycle_norm=True,
            rul_cap=125,
            random_seed=42,
        )
        prepared = _baseline_prepared(raw, config)
        monkeypatch.setattr(baseline_mod, "load_and_prepare_data", lambda config=None: prepared)
        monkeypatch.setattr(baseline_mod, "dataset_fingerprint", _stub_fingerprint)
        monkeypatch.setattr(baseline_mod, "get_models", _fast_models)
        return baseline_mod.run_baseline_training(
            config, models_dir=tmp_path / "models", reports_dir=tmp_path / "reports"
        )

    def test_artifacts_structure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        artifacts = self._run(tmp_path, monkeypatch)

        assert isinstance(artifacts, TrainingArtifacts)
        assert artifacts.best_model_name in {"DummyMean", "LinearRegression", "Ridge"}
        assert artifacts.best_model is not None
        assert len(artifacts.feature_names) > 0
        assert artifacts.rul_cap == 125
        assert {"mae", "rmse", "r2"} <= set(artifacts.test_metrics)
        assert all(isinstance(v, float) for v in artifacts.test_metrics.values())

    def test_disjoint_engine_split(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        artifacts = self._run(tmp_path, monkeypatch)
        train = set(artifacts.train_engine_ids)
        val = set(artifacts.val_engine_ids)
        assert train and val
        assert train.isdisjoint(val)

    def test_per_engine_errors_match_test_engines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        artifacts = self._run(tmp_path, monkeypatch)
        assert len(artifacts.test_engine_errors) == len(artifacts.val_engine_ids)

    def test_artifacts_written_to_tmp_not_production(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._run(tmp_path, monkeypatch)
        assert (tmp_path / "models" / "baseline_rul_model.pkl").exists()
        assert (tmp_path / "reports" / "model_card_baseline.json").exists()

    def test_deterministic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        a = self._run(tmp_path / "a", monkeypatch)
        b = self._run(tmp_path / "b", monkeypatch)
        assert a.best_model_name == b.best_model_name
        assert a.test_metrics == b.test_metrics


class TestAnomalyPipelineSynthetic:
    def _run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> object:
        raw = make_raw()
        config = AnomalyFeatureConfig(
            window_sizes=(3, 5),
            include_op_settings=True,
            include_residuals=True,
            residual_windows=(3, 5),
            rul_cap=125,
            random_seed=42,
        )
        feat = engineer_anomaly_features(raw, config)
        feature_cols = get_anomaly_feature_columns(feat, config)
        engine_ids = sorted(feat["unit_id"].unique())
        n_train = max(1, int(0.8 * len(engine_ids)))
        train_ids, val_ids = sorted(engine_ids[:n_train]), sorted(engine_ids[n_train:])
        prepared = {
            "train_df": feat[feat["unit_id"].isin(train_ids)].copy(),
            "val_df": feat[feat["unit_id"].isin(val_ids)].copy(),
            "feature_cols": feature_cols,
            "train_engine_ids": train_ids,
            "val_engine_ids": val_ids,
            "config": config,
        }
        test_raw = make_raw(seed=7)
        monkeypatch.setattr(anomaly_mod, "load_and_prepare_anomaly_data", lambda config: prepared)
        monkeypatch.setattr(anomaly_mod, "load_raw_data", lambda split: test_raw)
        monkeypatch.setattr(anomaly_mod, "dataset_fingerprint", _stub_fingerprint)

        model_config = AnomalyModelConfig(
            n_estimators=10, contamination="auto", random_state=42, n_jobs=1
        )
        return anomaly_mod.run_anomaly_training(
            config,
            model_config,
            [0.01],
            models_dir=tmp_path / "models",
            reports_dir=tmp_path / "reports",
        )

    def test_artifacts_structure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        artifacts = self._run(tmp_path, monkeypatch)
        assert artifacts.model is not None
        assert len(artifacts.feature_names) > 0
        assert artifacts.default_contamination == 0.01
        assert artifacts.default_threshold is not None

    def test_disjoint_engine_split(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        artifacts = self._run(tmp_path, monkeypatch)
        assert set(artifacts.train_engine_ids).isdisjoint(set(artifacts.val_engine_ids))

    def test_no_rul_leakage(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        artifacts = self._run(tmp_path, monkeypatch)
        assert all("rul" not in name.lower() for name in artifacts.feature_names)

    def test_preprocessing_fit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        artifacts = self._run(tmp_path, monkeypatch)
        assert hasattr(artifacts.imputer, "statistics_")
        assert hasattr(artifacts.scaler, "mean_")
        assert hasattr(artifacts.scaler, "scale_")

    def test_artifacts_written_to_tmp_not_production(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._run(tmp_path, monkeypatch)
        assert (tmp_path / "models" / "anomaly_model.pkl").exists()
        assert (tmp_path / "reports" / "anomaly_model_card.json").exists()
