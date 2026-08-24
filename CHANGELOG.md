# Changelog

All notable changes to the FPL Transfer Model project are logged here, in reverse chronological order (newest first).

Each entry should include: date, files changed, what changed, and why.

---

## [Unreleased]

## 2026-08-24

**Files changed:**
- src/modeling/data.py
- src/modeling/cv.py (new)
- src/compare_models.py (new)

**What changed:**
Added `modeling.data.sort_chronologically()` (orders rows by season+GW across players, rather than the per-player `(season, element, GW)` order used for feature engineering) and a new `modeling/cv.py::build_time_series_cv()` wrapping scikit-learn's `TimeSeriesSplit`. Added `src/compare_models.py`, which cross-validates the linear baseline, tunes `RandomForestRegressor` hyperparameters with `RandomizedSearchCV` (10 candidates x 5 folds) over that time-series CV, then fits both the linear model and the best random forest on the full training seasons and reports both on the untouched 2025-26 test set for a final apples-to-apples comparison.

**Why:** Both `train_baseline_model.py` and `train_random_forest.py` only ever checked performance on a single train/test split, and the random forest's hyperparameters (`n_estimators=300, max_depth=10, min_samples_leaf=5`) were a guess, not a tuned choice. This adds real model selection: cross-validated MAE (linear: 1.042, tuned RF: 0.985) confirmed on the 2025-26 hold-out (linear: MAE 1.012/RMSE 1.931; tuned RF: MAE 0.961/RMSE 1.925) that the tuned random forest genuinely beats both the linear baseline and the untuned random forest from the previous entry (MAE 0.970). Tuned params: `n_estimators=300, min_samples_leaf=5, max_features='log2', max_depth=15`.

**Data-leakage note (per CLAUDE.md):** This is the important one to flag. Plain K-Fold CV on this dataset would let a model be *fit* on rows from later gameweeks while being *validated* on earlier ones -- not a feature-level leak (each row's lag/rolling features already only look backward via `shift(1)`), but still an unrealistic advantage the model would never get in actual use, since real predictions can only ever be made forward in time from what's already happened. `sort_chronologically()` + `TimeSeriesSplit` avoids this: every validation fold is strictly later in time than the training fold it's evaluated against. The 2025-26 test season is never touched during CV or tuning -- it's only used once, at the end, for the final comparison.

**Runtime note:** The hyperparameter search takes ~6 minutes on the full training set (50 random-forest fits). Not a concern for occasional use, but worth knowing before running it repeatedly.

## 2026-08-23 (2)

**Files changed:**
- src/modeling/preprocessing.py (new)
- src/train_baseline_model.py
- src/train_random_forest.py

**What changed:**
Extracted the preprocessing pipeline (median-impute numeric features; most-frequent-impute + one-hot encode categorical features) out of `train_baseline_model.py` into a new shared `modeling/preprocessing.py::build_preprocessor()`. `train_baseline_model.py` now calls this instead of building its own `ColumnTransformer` inline. Implemented `train_random_forest.py` (previously an empty stub) as a `RandomForestRegressor` on top of the same shared `modeling.data`, `modeling.evaluate`, `modeling.features`, and new `modeling.preprocessing` modules, following the same structure as `train_baseline_model.py` (train/test split, all-players + likely-starters metrics, sample predictions), plus a feature-importance printout as a sanity check.

**Why:**
Continuing the Phase 1 deduplication: adding a second model (random forest) on top of the linear baseline would otherwise mean copy-pasting the same `ColumnTransformer`/imputer/one-hot setup a second time. Centralizing it means both models (and any future ones) share one preprocessing definition. Random forest hyperparameters (`n_estimators=300, max_depth=10, min_samples_leaf=5, random_state=42`) are a reasonable starting point, not tuned — worth revisiting later.

**Verification:** Ran both `train_baseline_model.py` (output unchanged — MAE/RMSE identical to before this refactor) and `train_random_forest.py` end-to-end against real training data; results are in a sensible range (comparable to the linear baseline) and feature importances rank recent points/minutes highest, which matches expectations rather than looking like noise.

**Data-leakage note (per CLAUDE.md):** No feature construction or train/test split logic changed here (only where the existing preprocessing code lives, and a new model was added) — `split_by_season` and the feature definitions from the earlier entry are untouched.

## 2026-08-23

**Files changed:**
- src/modeling/features.py
- src/build_training_data.py
- src/evaluate_baselines.py

**What changed:**
Phase 1 of the modeling refactor: centralized feature definitions in `src/modeling/features.py` (new `LAG_SOURCE_COLUMNS`, `ROLLING_SOURCE_COLUMNS`, `ROLLING_WINDOWS`, `IDENTIFICATION_COLUMNS`, `FIXTURE_KNOWN_COLUMNS`, `TARGET_COLUMN`, and a derived `TRAINING_COLUMNS`), and updated `build_training_data.py` and `evaluate_baselines.py` to use the shared `modeling/` package instead of their own hardcoded copies. `build_training_data.py` now builds its lag/rolling features and selects its output columns from the shared column lists rather than three separately hardcoded lists. `evaluate_baselines.py` now loads and splits data via `modeling.data.load_training_data`/`split_by_season` and computes MAE/RMSE via `modeling.evaluate.calculate_metrics`, instead of re-reading the CSV and re-implementing the metric calculation. `train_baseline_model.py` was already using the shared `modeling/` package from earlier work and did not need changes.

**Why:**
`src/modeling/` (data loading, train/test splitting, feature definitions, evaluation) existed but was only partially adopted — the feature column names were duplicated between `modeling/features.py` and `build_training_data.py`, and `evaluate_baselines.py` had its own copy of the CSV loading, season-filtering, and MAE/RMSE logic. This duplication risked the feature set silently drifting out of sync between the data-building step and the modeling step. Consolidating removes that risk and is prep for adding `train_random_forest.py` (currently an empty stub) on top of the same shared modules.

**Data-leakage note (per CLAUDE.md):** This change touches feature construction column lists in `src/modeling/features.py`, but does not change any leakage-prevention logic — the `shift(1)` semantics in `build_training_data.py`'s rolling/lag calculations are untouched, and the derived feature names/values were verified to match the pre-refactor output exactly (same `NUMERIC_FEATURES`/`TRAINING_COLUMNS` lists, and `evaluate_baselines.py`'s MAE/RMSE numbers were cross-checked against the original logic and are identical).

<!-- New entries go above this line -->

---

## Format for new entries

```
## YYYY-MM-DD

**Files changed:**
- path/to/file1.py
- path/to/file2.py

**What changed:**
Short description of the change.

**Why:**
Reasoning / goal behind the change.
```
