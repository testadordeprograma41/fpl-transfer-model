# Changelog

All notable changes to the FPL Transfer Model project are logged here, in reverse chronological order (newest first).

Each entry should include: date, files changed, what changed, and why.

---

## [Unreleased]

## 2026-08-25 (2)

**Files changed:**
- src/modeling/features.py
- src/build_training_data.py
- src/build_live_features.py
- src/predict.py
- src/train_random_forest.py
- src/fetch_fixtures.py
- data/processed/training_data.csv (regenerated)
- models/random_forest.joblib (regenerated, gitignored)

**What changed:**
Replaced FPL's raw, per-season numeric `opponent_team` id with a stable `opponent_name` (and renamed the already-name-based `team` to `team_name`, for a consistent, unambiguous pair) as the model's categorical team features, everywhere: `modeling/features.py` (`IDENTIFICATION_COLUMNS`, `FIXTURE_KNOWN_COLUMNS`, `CATEGORICAL_FEATURES`), `build_training_data.py` (new `build_season_team_name_maps()` + `add_stable_team_names()`), `build_live_features.py` (live players' own team and next opponent are now both read from this season's bootstrap-static team names, kept in `team_id`/`opponent_team_id` only as non-feature reference columns), `predict.py`, and `train_random_forest.py`'s sample-predictions display. Regenerated `training_data.csv` and retrained/resaved the production model with the existing tuned hyperparameters (unchanged: `n_estimators=300, max_depth=15, min_samples_leaf=5, max_features='log2'`).

`build_training_data.py`'s new `build_season_team_name_maps()` derives each season's `{numeric id: team name}` mapping directly from `data/historical/historical_gws.csv` itself -- no external reference file, no guessing. Every played fixture pairs exactly two teams, and one side's already-name-based `team` column paired with the OTHER side's numeric `opponent_team` id (grouped by `(season, fixture)`) hands us both directions of the mapping directly from data the model was already trained on.

Also updated `fetch_fixtures.py`: its `verify_team_id_mapping()` cross-check existed only because the model used to consume `training_data.csv`'s numeric `opponent_team` column directly -- with that column gone (replaced by `opponent_name`), the check would raise a `KeyError` and, more fundamentally, had nothing left to verify (there's no id<->name mapping for the model to get wrong anymore). Removed it and `infer_current_season()` (now unused), replaced with a much smaller `report_unseen_team_names()` that's purely informational: which of this fetch's team names have never appeared in `training_data.csv` (e.g. a promoted club) -- not a correctness gate.

**Why:**
`opponent_team` is FPL's raw, per-season numeric team id, and that id is NOT the same category across seasons. Confirmed empirically by `build_season_team_name_maps()` against this project's own historical data: id `3` was Bournemouth in both 2023-24 and 2024-25, but Burnley in 2025-26; id `11` was Liverpool, then Leicester, then Leeds across the same three seasons. `training_data.csv` pooled all three seasons together and one-hot encoded `opponent_team` as if id `3` meant one consistent club throughout -- silently teaching the model that unrelated clubs were the same opponent, and (per the fetch_fixtures.py entries earlier in this file) putting live 2026-27 predictions at risk of the SAME numeric id silently pointing at the wrong club. Team names don't have this problem: a club keeps its name across seasons regardless of which numeric id the API assigns it that year.

**Data-leakage / correctness notes (per CLAUDE.md -- flagging as instructed):**
1. **Leakage protections preserved, verified not just asserted:** `create_features()`'s `shift(1)`/rolling logic, `split_by_season()`'s chronological train/test split, and `modeling/cv.py`'s `TimeSeriesSplit` were NOT touched by this change. Verified directly: the regenerated `training_data.csv` has the exact same shape (84,577 rows x 26 columns) as before, and all 24 columns unrelated to team/opponent are byte-identical to the pre-change file, row for row. The only columns that changed are the renamed/re-derived `team_name`/`opponent_name` themselves.
2. **New mapping verified two ways, not assumed correct:** (a) zero fixture-pairing problems found across all 3 seasons x 380 fixtures each (every fixture cleanly resolved to exactly two sides, no id/name conflicts within a season); (b) independently cross-checked every one of the 84,577 regenerated rows' new `opponent_name` against what the OLD numeric `opponent_team` id resolves to via the same derived per-season map -- 84,577/84,577 matched.
3. **Unseen-category behavior is now correctly scoped:** a genuinely new club (Coventry City, Hull City, Ipswich Town this season -- still 3 teams, same as before this change) is still, correctly, an unseen category to `OneHotEncoder(handle_unknown="ignore")` -- there's no history for a club that's never been in the Premier League, and zeroing that feature rather than guessing is right. What's fixed is that an EXISTING club's numeric id can no longer be silently mistaken for a different club that happened to hold the same id in a past season.
4. **Not assumed better just because it's more correct (per instruction):** retrained with unchanged hyperparameters and compared against the recorded 2025-26 hold-out benchmark rather than assuming an improvement:
   - Old (numeric `opponent_team`): MAE 0.961, RMSE 1.925-1.926
   - New (`team_name`/`opponent_name`): MAE 0.958, RMSE 1.926
   Essentially a wash, with a small, plausible improvement in MAE (cleaner, non-conflated categories for returning clubs) and RMSE flat. Not a dramatic change -- expected, since `team_name`/`opponent_name` replace only one categorical feature's semantics among several, and most of the model's signal comes from the numeric form features (see feature-importance printout).

**Verification:** Ran the full pipeline end-to-end: `build_training_data.py` (0 fixture-pairing problems, 100% opponent_name match against the old mapping, identical row/column counts otherwise), `train_random_forest.py` (metrics above, saved production model), and a `build_live_features.py` + `predict.py` dry run using already-fetched `data/raw/fixtures.csv` and `data/raw/players.csv` (network access to the live FPL API isn't available from this side) -- all 21 `FEATURE_COLUMNS` present, 606 players matched to GW1 fixtures, 112 with no history (unchanged from before, expected), 3 teams correctly flagged as unseen, and `next_gw_predictions.csv`'s top 20 shows correct, sensible GW1 opponents by name (e.g. all three Man Utd players correctly show `Hull City` as their opponent) with no obvious mapping problems. **The user still needs to re-run this same sequence on their own machine against the live API and real model file for final confirmation**, matching the pattern from the previous session.

**New limitations discovered:**
- Removing `fetch_fixtures.py`'s numeric-id cross-check was not on the original file list for this task, but was an unavoidable consequence of removing `opponent_team` from `training_data.csv` (the old check would otherwise crash with `KeyError`) -- flagging this explicitly since it's an incidental scope expansion, not requested directly.
- `build_season_team_name_maps()`'s fixture-pairing derivation depends on `historical_gws.csv`'s `fixture`/`team`/`opponent_team` columns being internally consistent per season (clean for all 3 seasons currently in this project) -- if a future season's data has a postponed/voided/data-quality-issue fixture, `build_training_data.py` now reports it as a "problem" rather than silently mismapping, but this hasn't been tested against real messy data.
- Pre-existing, unchanged by this fix: GW1-of-a-season feature vectors were still never present in training (see the 2026-08-25 (1) entry above); promoted-team players still get zero match history.

**Not yet built (explicitly out of scope for this task):** multi-Gameweek expected-points forecasting and the transfer optimizer.

## 2026-08-25

**Files changed:**
- src/build_live_features.py (new)
- src/predict.py
- src/modeling/data.py

**What changed:**
Added `build_live_features.py`, which builds one row per live player for the next unplayed 2026-27 Gameweek (610 players for GW1). It fetches current `bootstrap-static` data plus `data/raw/fixtures.csv`, computes the same lag/rolling features used in training (`minutes_last`, `total_points_last`, `expected_goals_last`, `expected_assists_last`, `expected_goal_involvements_last`, `starts_last`, plus 3- and 5-match rolling averages) from each player's most recent PAST matches in `data/historical/` (any season, matched by player name since FPL's numeric element ids aren't stable across seasons), and joins each player's real upcoming `opponent_team`/`was_home`/`value` from `fixtures.csv`. Saves `data/processed/live_features.csv` containing every `modeling.features.FEATURE_COLUMNS` entry plus display/identification columns (name, team, position, price, opponent name, kickoff time, match count on record). Validates and reports (without silently failing): missing `FEATURE_COLUMNS`, players with no next-GW fixture, players with zero match history, and teams unseen in `training_data.csv`. Added `LIVE_FEATURES_FILE`/`NEXT_GW_PREDICTIONS_FILE` path constants to `modeling/data.py`, matching the existing `MODEL_FILE` constant pattern. Rewrote `predict.py`'s `main()` to load `live_features.csv` and produce `data/processed/next_gw_predictions.csv` (name/team/position/price/opponent/home_away/predicted_points), fully replacing the old "reuse each player's last played fixture" demo.

**Why:**
This closes the gap flagged in `predict.py`'s previous docstring and in the `fetch_fixtures.py` entry above: the project could fetch upcoming fixtures but had no way to actually turn them into real prediction inputs. This is the first genuinely forward-looking (not retrospective) prediction this model has produced.

**Data-leakage / correctness notes (per CLAUDE.md -- flagging as instructed):**
1. `build_live_features.py` deliberately computes lag/rolling features ACROSS season boundaries (grouped by player name only, sorted by season+GW), unlike `build_training_data.py`'s per-`(season, element)` grouping. This is necessary: under the training-time rule, every player's first match of a season has no lag features and is dropped outright (`dropna(subset=["minutes_last"])`) -- applied to a live GW1 prediction, that rule would leave every single player with empty features and nothing to predict. Only past, already-played matches are ever read (`data/historical/*.csv`) -- nothing about the unplayed upcoming fixture feeds into a lag/rolling feature, so the no-future-leakage property build_training_data.py's `shift(1)` was written to guarantee is preserved, just applied across a season boundary instead of within one. This does mean GW1-shaped feature vectors were never present in `training_data.csv` (they're dropped there) -- so this is inherently a slightly out-of-distribution input for the model, which is flagged in `build_live_features.py`'s module docstring rather than hidden.
2. Players with zero prior Premier League history (113 of 610 for the real 2026-27 GW1 run) get NaN lag/rolling features, left for the model's existing `SimpleImputer(strategy="median")` to handle -- the same treatment as any other missing value, not a new code path.
3. **Known limitation, NOT fixed here:** `team`/`opponent_team` are one-hot encoded against whatever categories appeared in `training_data.csv` (2023-24/2024-25/2025-26 pooled). Coventry City, Hull City, and Ipswich Town never appear there, so their players' `team` feature is silently zeroed by `OneHotEncoder(handle_unknown="ignore")`, and `opponent_team` (a raw per-season numeric id, already flagged as unstable across seasons in the `fetch_fixtures.py` entry above) may silently collide with an unrelated team's id from a past season for ANY team, promoted or not. `build_live_features.py` reports which teams are unseen in training data at runtime so affected players are visible; redesigning the categorical encoding itself would be a feature-construction change out of scope for this task and needs its own sign-off first.

**Verification:** Ran `build_live_features.py` against the live FPL API: 610 players fetched for GW1 2026-27, 0 players without a fixture, 113 without match history, the 3 promoted teams correctly flagged as unseen in training data. Ran `train_random_forest.py` to regenerate the local (gitignored) model -- MAE 0.961 on the 2025-26 hold-out, matching the number already recorded in this CHANGELOG, confirming the retrain reproduced the same model. Ran `predict.py` against the real model and real `live_features.csv`: loaded without error, all `FEATURE_COLUMNS` present, produced `next_gw_predictions.csv` with a plausible top 20 (recognisable in-form players -- Bruno Fernandes, Szoboszlai, Watkins, Cole Palmer, Virgil van Dijk, etc. -- at reasonable predicted-points values, with correct GW1 opponents and home/away).

**Not yet built (explicitly out of scope for this task):** the transfer optimizer. This only produces single-Gameweek point predictions.

## 2026-08-24 (4)

**Files changed:**
- src/fetch_fixtures.py

**What changed:**
Replaced the hardcoded `CURRENT_SEASON = "2025-26"` constant in `verify_team_id_mapping()` with a new `infer_current_season()` helper, which derives the season string directly from the fetched fixtures' own kickoff dates (a Jul-Dec kickoff belongs to the season starting that calendar year; Jan-Jun belongs to the season that started the previous year). `verify_team_id_mapping()` now looks up `training_data.csv` rows for that inferred season instead of the hardcoded one.

**Why:**
Running `fetch_fixtures.py` today showed `CURRENT_SEASON = "2025-26"` had gone stale: the fetched fixtures are actually 2026-27 season fixtures (new promoted teams -- Coventry City, Hull City, Ipswich Town -- confirm this), but the hardcoded constant still pointed at last season. All 380 fetched fixtures currently show `finished == False`, so today's run happened to fail safely (0 fixture-sides to check -> the existing "unverified" message). But this was about to become an active correctness bug, not just a cosmetic one: once real 2026-27 fixtures start finishing, the old code would have cross-checked them against **2025-26's different fixtures** for the same team+GW number -- and per this file's own docstring, team ids are not guaranteed stable across a promotion/relegation season boundary. That would have produced spurious pass/fail verdicts that say nothing real about whether *this season's* id<->name mapping is correct, defeating the point of the check.

**Data-leakage / correctness note (per CLAUDE.md -- flagging as instructed):** This touches the mapping-verification logic flagged as a known area of concern in the `fetch_fixtures.py (new)` entry above. The fix doesn't change any feature-construction or leakage-prevention code in `modeling/` -- it only corrects which season's rows the cross-check compares `fixtures.csv` against. Verified without hitting the network: `infer_current_season()` correctly returns `"2026-27"` for the real fetched fixtures' Aug 2026 kickoff dates, and correctly handles the Jan-Jun season-boundary case against synthetic Jan/May 2027 dates. The user then re-ran the script end-to-end against the live API: it now prints `"(no 2026-27 rows in training_data.csv yet -- skipping team id/name cross-check...)"` instead of the old, less accurate `"could not find any matching played fixtures"` message.

**Still unresolved -- flagging, not fixing here:** The team id<->name mapping for 2026-27 is still NOT verified against real data (`training_data.csv` has no 2026-27 rows to check against yet, since nothing in this project ingests live in-season results). This fix only stops the cross-check from silently comparing against the *wrong* season once 2026-27 games start finishing -- it doesn't add real verification. Per discussion with the user, we're proceeding to build on top of the unverified `bootstrap-static` mapping for now (it's FPL's own official API, judged low risk) rather than blocking on adding a live-results-ingestion step first.

## 2026-08-24 (3)

**Files changed:**
- src/fetch_fixtures.py (new)

**What changed:**
Added `fetch_fixtures.py`, which fetches upcoming fixtures from the FPL API (`bootstrap-static` for team id<->name, `fixtures` for the schedule), saves them to `data/raw/fixtures.csv`, and prints the next unplayed gameweek's matchups.

**Why:** `predict.py` (previous entry) can currently only reuse a player's *last played* fixture's opponent/was_home, because nothing in this project fetches upcoming fixtures -- this closes that gap, as a step toward real next-gameweek predictions and eventually a transfer recommender.

**Data-leakage / correctness note (per CLAUDE.md -- important, please read):** `opponent_team` in `training_data.csv` is the FPL API's raw, per-season numeric team id (1-20), not a name -- and that id assignment is NOT simply alphabetical. I tested the "alphabetical" hypothesis offline against your real `training_data.csv` (assign id 1..20 to the 2025-26 teams in alphabetical order, then check whether that's internally consistent with the `opponent_team` ids already recorded) and it failed on 197 of 740 fixture-sides checked -- i.e. a plausible-looking guess would have been wrong roughly 27% of the time. Getting this mapping wrong would silently produce bad predictions (`OneHotEncoder(handle_unknown="ignore")` just zeroes out an unrecognised id instead of erroring), which matters more here than elsewhere in this project since it would eventually feed transfer advice.

Because of that, `fetch_fixtures.py` does NOT guess -- it fetches the real id<->name mapping from the live `bootstrap-static` endpoint, and then cross-checks it against already-played 2025-26 fixtures in `training_data.csv` before printing OK or a WARNING. The cross-check logic itself (`verify_team_id_mapping`) was unit-tested with synthetic data to confirm it correctly flags mismatches and passes correct mappings.

**Not yet verified against the live API:** this environment has no network access to fantasy.premierleague.com, so I could not run this script end-to-end the way I ran and verified everything else this session. It follows the same request/parse pattern as the already-working `fetch_fpl_data.py`, but please run it and check for "Team id<->name mapping verified... OK" in the output before I build anything (real predict.py wiring, a recommender) on top of its output.


## 2026-08-24 (2)

**Files changed:**
- src/modeling/data.py
- src/train_random_forest.py
- src/predict.py (new)
- models/.gitkeep (new)
- .gitignore

**What changed:**
`train_random_forest.py` now uses the hyperparameters found by `compare_models.py`'s tuning run (`n_estimators=300, max_depth=15, min_samples_leaf=5, max_features='log2'`, replacing the earlier untuned guess) and, after the usual train/test evaluation, refits on the *entire* dataset (train + test seasons) and saves that fitted pipeline to `models/random_forest.joblib` via `joblib.dump`. Added `MODEL_DIR`/`MODEL_FILE` path constants to `modeling/data.py` so the save location is defined once. Added `src/predict.py`, a small reusable `load_model()` / `predict(model, df)` inference entrypoint intended for downstream use (e.g. a future transfer recommender). `.gitignore` now excludes `models/*.joblib` (the trained binary is a generated artifact, same treatment as the existing `data/*/​*.csv` rules); `models/.gitkeep` keeps the empty directory tracked in git the same way `data/processed/.gitkeep` already does.

**Why:**
Every training script so far retrained from scratch every time it ran, with no way to reuse a fitted model. This makes the tuned random forest a persisted artifact that can be loaded once and reused, which is a prerequisite for building an actual transfer recommender on top of it later.

**Important limitation, called out in `predict.py`'s docstring:** `predict.py`'s demo predicts using each player's most recent *played* fixture (their last known opponent/was_home/value), not a real upcoming gameweek — there's no `fetch_fixtures.py` yet to supply the next fixture's opponent/home-away for each player. The demo exists to sanity-check that the saved model loads and predicts sensibly (verified: the top of the list is recognisable, currently-in-form players like Bruno Fernandes and Ollie Watkins), not to produce real transfer advice yet.

**Data-leakage note (per CLAUDE.md):** The saved *production* model is deliberately fit on the 2025-26 test season too (not just train) — this is fine because the test-season MAE/RMSE numbers used to judge the model were already computed beforehand, from a model fit ONLY on train. Once that honest estimate exists, refitting on all available data before saving is standard practice (more data → a better model for real future predictions) and isn't circular, since no future information leaks backward into any feature (`shift(1)` in `build_training_data.py` is unchanged).

**Not yet committed:** `models/random_forest.joblib` itself isn't included in this delivery — it's gitignored, and at ~58MB is also too large for the file-sync tool that writes files onto your machine. Run `python src/train_random_forest.py` from the project root after pulling these changes to generate your own local copy.

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
