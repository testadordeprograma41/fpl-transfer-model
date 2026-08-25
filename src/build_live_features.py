"""Build one row per player for the next unplayed 2026-27 Gameweek, with
the same historical features the model was trained on, ready for
predict.py.

This is the live-inference counterpart to build_training_data.py. It does
NOT touch training_data.csv or any of build_training_data.py's shift(1)
logic -- the data-leakage protections there are untouched by this script.

WHY THIS EXISTS: predict.py used to reuse each player's most recent PLAYED
fixture as a stand-in for the next one -- that's not a real forecast (see
predict.py's previous docstring, and the CHANGELOG entry that added it).
This script builds the real thing: for each player, their next ACTUAL
upcoming fixture (opponent/was_home/value) from data/raw/fixtures.csv,
plus lag/rolling features computed from their most recent PAST matches.

DELIBERATE DIFFERENCE FROM build_training_data.py's shift(1) LOGIC:
build_training_data.py groups by (season, element) -- lag/rolling features
never look past a season boundary, and rows with no prior match THAT
SEASON (i.e. every player's first match of every season) are dropped
entirely from training_data.csv via dropna(subset=["minutes_last"]).
That's fine for a training set, where plenty of other rows exist per
player. It is not workable for live GW1 inference: literally no 2026-27
match has been played yet, so under that same season-scoped rule every
single player would have empty lag features and -- per
build_training_data.py's own dropna step -- would have to be dropped
outright. Zero predictions is not a useful outcome for "predict the next
Gameweek".

Instead, this script computes each player's lag/rolling features from
their most recent PAST matches regardless of season, matched by player
NAME rather than FPL's numeric "element" id (element ids are NOT stable
across seasons -- same reasoning already documented in fetch_fixtures.py
for team ids). A player's most recent known form genuinely IS their last
2025-26 match if they have not played a 2026-27 match yet. This only ever
reads matches that have already been played (data/historical/*.csv) --
nothing about the upcoming, unplayed fixture is used to build lag/rolling
features, so this preserves the no-future-data-leakage property
build_training_data.py's shift(1) was written to guarantee, just applied
across a season boundary instead of within one.

Players with NO historical match at all (e.g. genuinely new to the
Premier League, or on a newly-promoted team with no top-flight history in
data/historical/) get NaN lag/rolling features. Those are left as NaN
here on purpose -- the model's existing preprocessing pipeline
(modeling.preprocessing.build_preprocessor's SimpleImputer(strategy=
"median")) already handles missing feature values for exactly this
reason, so an "insufficient history" player is handled the same way any
other missing value already is, not silently guessed at by this script.

KNOWN LIMITATIONS (flagged per CLAUDE.md, not fixed here -- out of scope
for "produce real predictions with the existing pipeline"):

1. GW1-of-a-season feature vectors like these were never present in
   training_data.csv (build_training_data.py drops every player's first
   match of every season). The model has never explicitly trained on a
   row shaped like "first match of the season" -- treat these predictions
   as the best available forward-looking estimate, not a guarantee of the
   MAE/RMSE numbers in CHANGELOG.md, which were measured on GW2+ rows.
2. "team" and "opponent_team" are both categorical features one-hot
   encoded against whatever categories appeared in training_data.csv
   (2023-24 + 2024-25 + 2025-26, pooled). A team that never appears in
   that pooled set (e.g. a newly-promoted team) is an unseen category:
   OneHotEncoder(handle_unknown="ignore") silently zeroes out the "team"
   feature for its players rather than erroring. Separately,
   "opponent_team" is FPL's raw numeric per-season id, which is already
   documented (fetch_fixtures.py) as NOT stable across seasons -- so even
   for a team that DID appear before, this season's id for it may
   silently collide with a different team's id from a past season. This
   script reports which live teams are unseen in training data so you can
   see the affected players, but does not change how the feature is
   encoded -- that would be a feature-construction change and needs its
   own sign-off, not a side effect of building live inputs.
3. Assumes exactly one fixture per team in the next Gameweek (true for a
   normal, non-blank/non-double Gameweek; GW1 always qualifies). Not
   handled: blank/double Gameweeks later in the season.
"""

from pathlib import Path

import pandas as pd
import requests
requests.packages.urllib3.util.connection.HAS_IPV6 = False

from modeling.data import LIVE_FEATURES_FILE
from modeling.features import (
    FEATURE_COLUMNS,
    LAG_SOURCE_COLUMNS,
    ROLLING_SOURCE_COLUMNS,
    ROLLING_WINDOWS,
)


BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

FIXTURES_FILE = Path("data/raw/fixtures.csv")
HISTORICAL_FILE = Path("data/historical/historical_gws.csv")

POSITION_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Columns carried through to live_features.csv beyond what FEATURE_COLUMNS
# strictly requires -- needed for predict.py's next_gw_predictions.csv
# (player name, price display, opponent name, etc.) and for spot-checking
# this file by eye.
IDENTIFICATION_OUTPUT_COLUMNS = [
    "id",
    "code",
    "name",
    "web_name",
    "GW",
    "kickoff_time",
    "opponent_name",
    "matches_on_record",
]


def fetch_json(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def load_live_players(bootstrap):
    """Every current player, with position/team names and current price.

    "value" is now_cost straight from the API (tenths of a million, e.g.
    45 = GBP4.5m) -- the SAME scale as training_data.csv's "value" column
    (also raw FPL tenths, see CHANGELOG.md), not the /10 "price" used
    elsewhere in this project for display.
    """

    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])

    team_map = teams.set_index("id")["name"].to_dict()

    players["team"] = players["team"].map(team_map)
    players["position"] = players["element_type"].map(POSITION_MAP)
    players["name"] = players["first_name"] + " " + players["second_name"]
    players["value"] = players["now_cost"]

    return players[
        ["id", "code", "name", "web_name", "team", "position", "value"]
    ].copy()


def load_next_gw_fixtures():
    """The earliest Gameweek with any unplayed fixture, and its fixtures.

    Same "next unplayed GW" definition as fetch_fixtures.py, so the two
    scripts always agree on which Gameweek "next" means.
    """

    if not FIXTURES_FILE.exists():
        raise FileNotFoundError(
            f"{FIXTURES_FILE} not found -- run fetch_fixtures.py first."
        )

    fixtures = pd.read_csv(FIXTURES_FILE)
    upcoming = fixtures[fixtures["finished"] == False]

    if upcoming.empty:
        raise ValueError(
            f"No unplayed fixtures found in {FIXTURES_FILE} -- nothing to "
            f"build live features for. Re-run fetch_fixtures.py."
        )

    next_gw = upcoming["event"].min()
    next_gw_fixtures = upcoming[upcoming["event"] == next_gw].copy()

    duplicated_teams = pd.concat(
        [next_gw_fixtures["team_h_name"], next_gw_fixtures["team_a_name"]]
    )
    duplicated_teams = duplicated_teams[duplicated_teams.duplicated()]

    if len(duplicated_teams):
        print(
            f"WARNING: GW{int(next_gw)} has a team appearing in more than "
            f"one fixture (blank/double Gameweek) -- this script assumes "
            f"one fixture per team and will only keep the first match "
            f"per team: {sorted(duplicated_teams.unique())}"
        )

    return int(next_gw), next_gw_fixtures


def build_next_fixture_lookup(next_gw_fixtures):
    """One row per (team, upcoming opponent id/name, was_home) for the
    next Gameweek -- every player on a team shares that team's fixture.
    """

    home_side = next_gw_fixtures[
        ["team_h_name", "team_a", "team_a_name", "kickoff_time"]
    ].rename(
        columns={
            "team_h_name": "team",
            "team_a": "opponent_team",
            "team_a_name": "opponent_name",
        }
    )
    home_side["was_home"] = True

    away_side = next_gw_fixtures[
        ["team_a_name", "team_h", "team_h_name", "kickoff_time"]
    ].rename(
        columns={
            "team_a_name": "team",
            "team_h": "opponent_team",
            "team_h_name": "opponent_name",
        }
    )
    away_side["was_home"] = False

    lookup = pd.concat([home_side, away_side], ignore_index=True)

    # If a blank/double Gameweek slipped through, keep only the first
    # fixture per team rather than silently duplicating that team's
    # players into two rows each.
    return lookup.drop_duplicates(subset=["team"], keep="first")


def load_player_history():
    """Every played match on record, any season, oldest first.

    Chronological order ACROSS seasons (not per-season, unlike
    build_training_data.py's load_data()) is the one deliberate departure
    explained in the module docstring.
    """

    if not HISTORICAL_FILE.exists():
        raise FileNotFoundError(
            f"{HISTORICAL_FILE} not found -- run fetch_historical_data.py "
            f"first."
        )

    history = pd.read_csv(HISTORICAL_FILE)

    numeric_columns = sorted(set(LAG_SOURCE_COLUMNS) | set(ROLLING_SOURCE_COLUMNS))
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    return history.sort_values(["name", "season", "GW"]).reset_index(drop=True)


def compute_player_form(history):
    """One row per player name: lag features from their single most
    recent played match, and 3/5-match rolling averages -- built ONLY
    from matches already in data/historical (i.e. already played, never
    the upcoming fixture itself). Players with fewer than a window's
    worth of matches get an average over whatever they do have (matching
    build_training_data.py's rolling(window, min_periods=1) behaviour).
    """

    grouped = history.groupby("name", group_keys=False, sort=False)

    form = pd.DataFrame({"matches_on_record": grouped.size()})

    last_match = grouped.tail(1).set_index("name")
    for column in LAG_SOURCE_COLUMNS:
        form[f"{column}_last"] = last_match[column]

    for window in ROLLING_WINDOWS:
        window_avg = grouped[ROLLING_SOURCE_COLUMNS].apply(
            lambda group, w=window: group.tail(w).mean()
        )
        window_avg.columns = [
            f"{column}_avg_{window}" for column in ROLLING_SOURCE_COLUMNS
        ]
        form = form.join(window_avg)

    return form.reset_index()


def assemble_live_dataset(live_players, fixture_lookup, form):
    live = live_players.merge(form, on="name", how="left")
    live = live.merge(fixture_lookup, on="team", how="left")

    live["matches_on_record"] = live["matches_on_record"].fillna(0)

    return live


def validate(live, next_gw, known_teams):
    """Print the checks required before this file is trusted: missing
    FEATURE_COLUMNS, players with no upcoming fixture, players with
    insufficient (or zero) history, and teams unseen during training.
    Does not raise -- reports so a human decides whether to proceed.
    """

    print()
    print(f"Live players fetched: {len(live):,}")
    print(f"Target Gameweek: GW{next_gw}")

    missing_columns = [c for c in FEATURE_COLUMNS if c not in live.columns]
    if missing_columns:
        print(f"ERROR: missing required FEATURE_COLUMNS: {missing_columns}")
    else:
        print(f"All {len(FEATURE_COLUMNS)} FEATURE_COLUMNS present. OK.")

    no_fixture = live[live["opponent_team"].isna()]
    print()
    if len(no_fixture):
        print(
            f"WARNING: {len(no_fixture)} player(s) have no GW{next_gw} "
            f"fixture (team not found in fixtures.csv for this "
            f"Gameweek) -- they will be EXCLUDED from live_features.csv:"
        )
        print(
            no_fixture[["name", "team"]]
            .drop_duplicates()
            .to_string(index=False)
        )
    else:
        print(f"All players matched to a GW{next_gw} fixture. OK.")

    has_fixture = live[live["opponent_team"].notna()]
    no_history = has_fixture[has_fixture["matches_on_record"] == 0]
    print()
    print(
        f"{len(no_history):,} of {len(has_fixture):,} player(s) with a "
        f"GW{next_gw} fixture have NO match history in data/historical "
        f"(new to the Premier League, or on a newly-promoted team). "
        f"Their lag/rolling features are left as NaN for the model's "
        f"existing median-imputation step to handle -- not an error, but "
        f"expect these predictions to be the least reliable in the file."
    )

    unseen_teams = sorted(set(has_fixture["team"].dropna()) - known_teams)
    if unseen_teams:
        print()
        print(
            f"NOTE (known limitation -- see module docstring / "
            f"CHANGELOG, not fixed here): {len(unseen_teams)} team(s) "
            f"never appear in training_data.csv: {unseen_teams}. Their "
            f"players' 'team' feature will be silently zeroed by the "
            f"trained OneHotEncoder, and matches against these teams' "
            f"'opponent_team' id may silently collide with an unrelated "
            f"team from a past season. Treat predictions for, and "
            f"against, {unseen_teams} with extra caution."
        )

    return missing_columns, no_fixture, no_history, unseen_teams


def main():
    bootstrap = fetch_json(BOOTSTRAP_URL)
    live_players = load_live_players(bootstrap)

    next_gw, next_gw_fixtures = load_next_gw_fixtures()
    fixture_lookup = build_next_fixture_lookup(next_gw_fixtures)

    history = load_player_history()
    form = compute_player_form(history)
    known_teams = set(history["team"].unique())

    live = assemble_live_dataset(live_players, fixture_lookup, form)

    missing_columns, no_fixture, no_history, unseen_teams = validate(
        live, next_gw, known_teams
    )

    if missing_columns:
        raise ValueError(
            f"Refusing to write {LIVE_FEATURES_FILE}: missing required "
            f"FEATURE_COLUMNS {missing_columns}. This means "
            f"modeling/features.py and build_live_features.py have "
            f"drifted out of sync -- fix the mismatch before predicting."
        )

    live = live[live["opponent_team"].notna()].copy()

    live["GW"] = next_gw
    live["opponent_team"] = live["opponent_team"].astype(int)
    live["matches_on_record"] = live["matches_on_record"].astype(int)

    output_columns = IDENTIFICATION_OUTPUT_COLUMNS + [
        column for column in FEATURE_COLUMNS
        if column not in IDENTIFICATION_OUTPUT_COLUMNS
    ]

    LIVE_FEATURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    live[output_columns].to_csv(LIVE_FEATURES_FILE, index=False)

    print()
    print(f"Live features saved: {len(live):,} players -> {LIVE_FEATURES_FILE}")


if __name__ == "__main__":
    main()
