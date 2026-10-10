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
across seasons -- same reasoning as team ids, see below). A player's most
recent known form genuinely IS their last 2025-26 match if they have not
played a 2026-27 match yet. This only ever reads matches that have
already been played (data/historical/*.csv) -- nothing about the
upcoming, unplayed fixture is used to build lag/rolling features, so this
preserves the no-future-data-leakage property build_training_data.py's
shift(1) was written to guarantee, just applied across a season boundary
instead of within one.

Players with NO historical match at all (e.g. genuinely new to the
Premier League, or on a newly-promoted team with no top-flight history in
data/historical/) get NaN lag/rolling features. Those are left as NaN
here on purpose -- the model's existing preprocessing pipeline
(modeling.preprocessing.build_preprocessor's SimpleImputer(strategy=
"median")) already handles missing feature values for exactly this
reason, so an "insufficient history" player is handled the same way any
other missing value already is, not silently guessed at by this script.

TEAM/OPPONENT REPRESENTATION (fixed 2026-08-25, see CHANGELOG.md):
"team_name" and "opponent_name" are the player's own club and their next
opponent's club, BY NAME -- both come straight from this season's live
bootstrap-static "teams" table via fetch_json(BOOTSTRAP_URL), which is
the single authoritative source for what a numeric id means THIS season.
This project previously fed the model FPL's raw numeric opponent_team id
directly. That id is NOT a stable category across seasons (confirmed
empirically -- see build_training_data.py's
build_season_team_name_maps(), which found e.g. id 3 was Bournemouth in
2023-24/2024-25 but Burnley in 2025-26 within this project's own training
data), so training_data.csv and modeling/features.py now use team_name/
opponent_name throughout instead. The raw numeric ids are still carried
in this file as "team_id"/"opponent_team_id" for reference/debugging, but
are NOT model features.

A genuinely new club (Coventry City, Hull City, Ipswich Town this season)
is still, correctly, an unseen category to the trained
OneHotEncoder(handle_unknown="ignore") -- there is no training history
for a club that's never been in the Premier League before, and zeroing
that feature rather than guessing is the right behaviour. What's fixed is
that an EXISTING club's numeric id can no longer be silently mistaken for
a different club that happened to hold the same id in a past season.

KNOWN LIMITATIONS (flagged per CLAUDE.md, not fixed here -- out of scope
for "produce real predictions with the existing pipeline"):

1. GW1-of-a-season feature vectors like these were never present in
   training_data.csv (build_training_data.py drops every player's first
   match of every season). The model has never explicitly trained on a
   row shaped like "first match of the season" -- treat these predictions
   as the best available forward-looking estimate, not a guarantee of the
   MAE/RMSE numbers in CHANGELOG.md, which were measured on GW2+ rows.
2. Assumes exactly one fixture per team in the next Gameweek (true for a
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
# (player name, price display, etc.), and "team_id"/"opponent_team_id"
# are kept purely for reference/debugging (NOT model features -- see
# module docstring).
IDENTIFICATION_OUTPUT_COLUMNS = [
    "id",
    "code",
    "name",
    "web_name",
    "GW",
    "kickoff_time",
    "team_id",
    "opponent_team_id",
    "matches_on_record",
]


def fetch_json(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def load_live_players(bootstrap):
    """Every current player, with position/team name and current price.

    "team_id" is FPL's raw numeric id for the player's own club THIS
    season, kept for reference only. "team_name" (derived from the same
    bootstrap-static "teams" table) is the actual model feature -- see
    module docstring for why the numeric id isn't used directly anymore.

    "value" is now_cost straight from the API (tenths of a million, e.g.
    45 = GBP4.5m) -- the SAME scale as training_data.csv's "value" column
    (also raw FPL tenths, see CHANGELOG.md), not the /10 "price" used
    elsewhere in this project for display.
    """

    players = pd.DataFrame(bootstrap["elements"])
    teams = pd.DataFrame(bootstrap["teams"])

    team_map = teams.set_index("id")["name"].to_dict()

    players["team_id"] = players["team"]
    players["team_name"] = players["team"].map(team_map)
    players["position"] = players["element_type"].map(POSITION_MAP)
    players["name"] = players["first_name"] + " " + players["second_name"]
    players["value"] = players["now_cost"]

    return players[
        [
            "id", "code", "name", "web_name",
            "team_id", "team_name", "position", "value",
        ]
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
    """One row per (team_name, upcoming opponent_name/opponent_team_id,
    was_home) for the next Gameweek -- every player on a team shares that
    team's fixture.

    team_h_name/team_a_name/team_h/team_a in fixtures.csv all come from
    THIS season's live bootstrap-static "teams" table (see
    fetch_fixtures.py) -- the same authoritative source used for
    team_name in load_live_players(), so names line up by construction.
    """

    home_side = next_gw_fixtures[
        ["team_h_name", "team_a", "team_a_name", "kickoff_time"]
    ].rename(
        columns={
            "team_h_name": "team_name",
            "team_a": "opponent_team_id",
            "team_a_name": "opponent_name",
        }
    )
    home_side["was_home"] = True

    away_side = next_gw_fixtures[
        ["team_a_name", "team_h", "team_h_name", "kickoff_time"]
    ].rename(
        columns={
            "team_a_name": "team_name",
            "team_h": "opponent_team_id",
            "team_h_name": "opponent_name",
        }
    )
    away_side["was_home"] = False

    lookup = pd.concat([home_side, away_side], ignore_index=True)

    # If a blank/double Gameweek slipped through, keep only the first
    # fixture per team rather than silently duplicating that team's
    # players into two rows each.
    return lookup.drop_duplicates(subset=["team_name"], keep="first")


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

    history = pd.read_csv(HISTORICAL_FILE, low_memory=False)

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
    live = live.merge(fixture_lookup, on="team_name", how="left")

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

    no_fixture = live[live["opponent_name"].isna()]
    print()
    if len(no_fixture):
        print(
            f"WARNING: {len(no_fixture)} player(s) have no GW{next_gw} "
            f"fixture (team not found in fixtures.csv for this "
            f"Gameweek) -- they will be EXCLUDED from live_features.csv:"
        )
        print(
            no_fixture[["name", "team_name"]]
            .drop_duplicates()
            .to_string(index=False)
        )
    else:
        print(f"All players matched to a GW{next_gw} fixture. OK.")

    has_fixture = live[live["opponent_name"].notna()]
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

    unseen_teams = sorted(set(has_fixture["team_name"].dropna()) - known_teams)
    if unseen_teams:
        print()
        print(
            f"NOTE: {len(unseen_teams)} team(s) never appear in "
            f"training_data.csv: {unseen_teams}. This IS expected and "
            f"acceptable for a genuinely new-to-the-top-flight club -- "
            f"their 'team_name' feature is correctly treated as an "
            f"unseen category by OneHotEncoder(handle_unknown='ignore') "
            f"rather than guessed at. This is no longer the numeric-id "
            f"collision risk flagged before 2026-08-25 (see CHANGELOG.md) "
            f"-- an unseen NAME cannot silently be mistaken for a "
            f"different, unrelated club the way a reused numeric id "
            f"could."
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

    live = live[live["opponent_name"].notna()].copy()

    live["GW"] = next_gw
    live["team_id"] = live["team_id"].astype(int)
    live["opponent_team_id"] = live["opponent_team_id"].astype(int)
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
