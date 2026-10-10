"""Fetch upcoming Premier League fixtures from the FPL API.

This is a prerequisite for real (not demo) predictions in predict.py: right
now predict.py can only reuse a player's LAST PLAYED fixture's
opponent/was_home, because nothing in this project fetches upcoming
fixtures. This script closes that gap.

TEAM NAMES, NOT NUMERIC IDS (updated 2026-08-25, see CHANGELOG.md): this
project used to feed the model FPL's raw, per-season numeric team id
(1-20) directly, and this script used to carry a whole
verify_team_id_mapping() cross-check because that numeric id is NOT
guaranteed to mean the same club across seasons (confirmed empirically --
id 3 was Bournemouth in 2023-24/2024-25 but Burnley in 2025-26 in this
project's own training data). That verification existed only because the
model consumed the numeric id directly; now that training_data.csv,
modeling/features.py, and build_live_features.py all use team_name/
opponent_name instead, there is no id<->name mapping left for the MODEL to
get wrong -- team_h_name/team_a_name below already ARE the stable
representation, straight from this season's live bootstrap-static "teams"
table, which is authoritative for what a name means this season by
definition. What's still worth checking (report_unseen_team_names(),
below) is only informational: which of this season's team names have
never appeared in training_data.csv at all (e.g. a newly-promoted club) --
not a mapping-correctness gate, just a heads-up. build_live_features.py
reports this again, and acts on it, when it actually builds prediction
inputs.
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


def build_team_map(bootstrap):
    teams = pd.DataFrame(bootstrap["teams"])
    return teams.set_index("id")["name"].to_dict()


def build_fixtures_dataframe(fixtures, team_map):
    df = pd.DataFrame(fixtures)

    df["team_h_name"] = df["team_h"].map(team_map)
    df["team_a_name"] = df["team_a"].map(team_map)

    return df


def report_unseen_team_names(fixtures_df):
    """Which of this fetch's team names have never appeared in
    training_data.csv. Informational only -- see module docstring for why
    this is no longer a correctness gate the way the old numeric-id
    cross-check was.
    """

    if not TRAINING_DATA_FILE.exists():
        print("(training_data.csv not found -- skipping team name check)")
        return

    training = pd.read_csv(TRAINING_DATA_FILE)
    known_teams = set(training["team_name"].unique())

    live_teams = set(fixtures_df["team_h_name"]) | set(fixtures_df["team_a_name"])
    unseen = sorted(live_teams - known_teams)

    print()
    if unseen:
        print(
            f"{len(unseen)} team name(s) in this fetch never appear in "
            f"training_data.csv (new to the Premier League this season, "
            f"e.g. promoted): {unseen}. Not an error -- "
            f"OneHotEncoder(handle_unknown='ignore') correctly treats "
            f"these as unseen categories now that team/opponent are "
            f"represented by name rather than a reused numeric id."
        )
    else:
        print(
            "All team names in this fetch already appear in "
            "training_data.csv. OK."
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

    report_unseen_team_names(fixtures_df)

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
