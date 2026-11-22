# Data Profiling Report — NASA C-MAPSS FD001

Generated: 2026-11-11
Source: NASA PCoE Prognostics Data Repository, entry #6
Parser: `src/profile_fd001.py` (numpy.loadtxt, no header assumption)

## 1. Ingestion summary

| Artifact | Path | Bytes | SHA-256 |
|---|---|---|---|
| Outer archive | `data/raw/6. Turbofan Engine Degradation Simulation Data Set.zip` | 12,429,152 | `C9C5DEC12A945A82E8BB4446589D7FB3CC057B5E5D81FA1A12E25EE9912AD3B2` |
| Train set | `data/raw/fd001/train_FD001.txt` | 3,515,356 | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| Test set | `data/raw/fd001/test_FD001.txt` | 2,228,855 | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| RUL truth | `data/raw/fd001/RUL_FD001.txt` | 429 | — |

## 2. Schema validation

Expected columns per `readme.txt`: 26 (unit, cycle, 3 op settings, 21 sensors).

| Check | Expected | Observed | Result |
|---|---|---|---|
| Columns | 26 | 26 | PASS |
| Train units | 100 | 100 | PASS |
| Test units | 100 | 100 | PASS |
| RUL entries | 100 | 100 | PASS |
| NaN cells | 0 | 0 | PASS |

## 3. Train set (`train_FD001.txt`)

- Shape: (20,631 rows, 26 cols)
- Units: 1..100
- Cycles per unit: 128–362 (mean 206.3)
- op_setting_3: constant 100.0 (Sea Level)
- op_setting_1 / op_setting_2: small variation around zero

## 4. Test set (`test_FD001.txt`)

- Shape: (13,096 rows, 26 cols)
- Units: 1..100
- Cycles per unit: 31–303 (mean 131.0)
- Trajectories truncated before failure.

## 5. RUL truth (`RUL_FD001.txt`)

- 100 values
- Range: 7–145 cycles, mean 75.5

## 6. Sensor observations

Constant / near-constant sensors (zero variance) — non-informative, candidates for
feature removal: sensor_1, sensor_5, sensor_10, sensor_16, sensor_18, sensor_19.

Informative sensors (varying with degradation): sensor_2, 3, 4, 6, 7, 8, 9, 11, 12,
13, 14, 15, 17, 20, 21.

## 7. Quality gate

All ingestion and schema checks passed. FD001 is approved for downstream modeling.
FD002–FD004 remain queued pending this gate.