"""Fetch upcoming Premier League fixtures from the FPL API.

This is a prerequisite for real (not demo) predictions in predict.py: right
now predict.py can only reuse a player's LAST PLAYED fixture's
opponent/was_home, because nothing in this project fetches upcoming
fixtures. This script closes that gap.

IMPORTANT: modeling/features.py's `opponent_team` is the FPL API's raw,
per-season numeric team id (1-20) -- NOT a name, and NOT necessarily stable
across seasons (a promoted/relegated season can shuffle which club gets
which id). Getting the id<->name mapping wrong here would silently corrupt
future predictions -- OneHotEncoder(handle_unknown="ignore") just zeroes
out an unrecognised id rather than raising an error. So this script
cross-checks the mapping it fetches against already-played fixtures in
data/processed/training_data.csv (which the model was trained on) before
you should trust its output for anything downstream.
"""

from pathlib import Path

import pandas as pd
import requests
requests.packages.urllib3.util.connection.HAS_IPV6 = False

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

OUTPUT_FILE = Path("data/raw/fixtures.csv")
TRAINING_DATA_FILE = Path("data/processed/training_data.csv")


def fetch_json(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def infer_current_season(fixtures_df):
    """Infer the season string (e.g. "2026-27") from fixture kickoff dates.

    The Premier League season runs roughly August through May. A fixture
    kicking off Jul-Dec belongs to the season starting that calendar year;
    Jan-Jun belongs to the season that started the previous calendar year.

    This used to be a hardcoded CURRENT_SEASON = "2025-26" constant, which
    silently went stale the moment the season rolled over: once 2026-27
    fixtures start finishing, comparing them against 2025-26 rows in
    training_data.csv (same team name + GW number, but a completely
    different fixture) would very likely fail the cross-check with a false
    "WARNING: mismatch" -- or worse, coincidentally "pass" -- neither of
    which says anything real about whether *this season's* id<->name
    mapping is correct. Deriving the season from the fetched fixtures
    themselves keeps this correct across season rollovers.
    """

    kickoffs = pd.to_datetime(fixtures_df["kickoff_time"], errors="coerce").dropna()

    if kickoffs.empty:
        return None

    reference = kickoffs.min()
    start_year = reference.year if reference.month >= 7 else reference.year - 1

    return f"{start_year}-{str(start_year + 1)[-2:]}"


def build_team_map(bootstrap):
    teams = pd.DataFrame(bootstrap["teams"])
    return teams.set_index("id")["name"].to_dict()


def build_fixtures_dataframe(fixtures, team_map):
    df = pd.DataFrame(fixtures)

    df["team_h_name"] = df["team_h"].map(team_map)
    df["team_a_name"] = df["team_a"].map(team_map)

    return df


def verify_team_id_mapping(fixtures_df, team_map):
    """Cross-check the id<->name mapping against fixtures the training
    data has already seen played this season.

    See the module docstring for why this matters: a silently wrong
    mapping here would degrade any prediction built on top of it, rather
    than erroring loudly.
    """

    if not TRAINING_DATA_FILE.exists():
        print(
            "(training_data.csv not found -- "
            "skipping team id/name cross-check)"
        )
        return

    season = infer_current_season(fixtures_df)

    if season is None:
        print(
            "(could not determine the current season from fixture "
            "kickoff times -- skipping team id/name cross-check)"
        )
        return

    training = pd.read_csv(TRAINING_DATA_FILE)
    season_rows = training[training["season"] == season]

    if season_rows.empty:
        print(
            f"(no {season} rows in training_data.csv yet -- skipping "
            f"team id/name cross-check; this is expected before any "
            f"{season} results have been ingested into training data, "
            f"e.g. at the very start of a season)"
        )
        return

    played = fixtures_df[fixtures_df["finished"] == True]

    checked = 0
    mismatches = 0

    for _, fixture in played.iterrows():
        gw = fixture["event"]

        if pd.isna(gw):
            continue

        home_name = fixture["team_h_name"]
        away_name = fixture["team_a_name"]
        home_id = fixture["team_h"]
        away_id = fixture["team_a"]

        # A home-side player's opponent_team should be the away team's id
        # (and vice versa) for that same gameweek.
        home_rows = season_rows[
            (season_rows["team"] == home_name)
            & (season_rows["GW"] == gw)
        ]

        away_rows = season_rows[
            (season_rows["team"] == away_name)
            & (season_rows["GW"] == gw)
        ]

        if len(home_rows):
            checked += 1
            if not (home_rows["opponent_team"] == away_id).all():
                mismatches += 1

        if len(away_rows):
            checked += 1
            if not (away_rows["opponent_team"] == home_id).all():
                mismatches += 1

    print()

    if checked == 0:
        print(
            "Could not find any matching played fixtures to cross-check "
            "against training_data.csv -- proceed with caution, this "
            "mapping is unverified."
        )
    elif mismatches == 0:
        print(
            f"Team id<->name mapping verified against {checked} "
            f"already-played fixture side(s) in training_data.csv. OK."
        )
    else:
        print(
            f"WARNING: {mismatches}/{checked} fixture side(s) did NOT "
            f"match training_data.csv's opponent_team ids. Do NOT trust "
            f"fixtures.csv's team ids for predictions until this is "
            f"resolved."
        )


def main():
    bootstrap = fetch_json(BOOTSTRAP_URL)
    fixtures = fetch_json(FIXTURES_URL)

    team_map = build_team_map(bootstrap)
    fixtures_df = build_fixtures_dataframe(fixtures, team_map)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fixtures_df.to_csv(OUTPUT_FILE, index=False)

    print(f"Fixtures fetched: {len(fixtures_df):,}")
    print(f"Saved to: {OUTPUT_FILE}")

    verify_team_id_mapping(fixtures_df, team_map)

    upcoming = fixtures_df[fixtures_df["finished"] == False].copy()

    print()
    print(f"Upcoming (unplayed) fixtures: {len(upcoming):,}")

    if len(upcoming):
        next_gw = upcoming["event"].min()
        next_gw_fixtures = upcoming[upcoming["event"] == next_gw]

        print(f"Next gameweek: GW{int(next_gw)}")
        print(
            next_gw_fixtures[
                ["event", "team_h_name", "team_a_name", "kickoff_time"]
            ]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
