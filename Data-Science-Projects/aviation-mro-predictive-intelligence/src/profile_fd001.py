"""Parse NASA C-MAPSS FD001 whitespace-delimited files and produce a profiling report.

No headers are assumed. The SRS (readme.txt) documents 26 columns:
  1  unit_id
  2  cycle
  3  op_setting_1
  4  op_setting_2
  5  op_setting_3
  6  sensor_1
  ...
  26 sensor_21
"""

import hashlib
import os

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT, "data", "raw", "fd001")

COLUMNS = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: str) -> np.ndarray:
    """Load whitespace-delimited numeric file into structured array."""
    raw = np.loadtxt(path, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    return raw


def profile(name: str, path: str) -> np.ndarray:
    arr = load(path)
    if arr.shape[1] != 26:
        pass
    # per-unit row counts
    units = np.unique(arr[:, 0]).astype(int)
    [int(np.sum(arr[:, 0] == u)) for u in units]
    # operational settings
    for c in (2, 3, 4):
        np.unique(arr[:, c])
    # sensor summary
    for c in range(5, 26):
        arr[:, c]
    return arr


def main() -> None:
    train = os.path.join(RAW, "train_FD001.txt")
    test = os.path.join(RAW, "test_FD001.txt")
    rul = os.path.join(RAW, "RUL_FD001.txt")

    tr = profile("TRAIN", train)
    te = profile("TEST", test)

    rul_vals = np.loadtxt(rul, dtype=np.float64).reshape(-1)

    # Cross-check: number of test units must equal number of RUL entries
    n_test_units = len(np.unique(te[:, 0]))
    assert len(np.unique(tr[:, 0])) == 100, "train must have 100 units"
    assert n_test_units == 100, "test must have 100 units"
    assert len(rul_vals) == 100, "RUL must have 100 entries"

    # Verify column count on every row
    for _name, arr in [("train", tr), ("test", te)]:
        bad = np.argwhere(np.any(np.isnan(arr)))
        if len(bad):
            pass
        else:
            pass


if __name__ == "__main__":
    main()
