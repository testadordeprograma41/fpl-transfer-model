import pandas as pd

from modeling.features import (
    LAG_SOURCE_COLUMNS,
    ROLLING_SOURCE_COLUMNS,
    ROLLING_WINDOWS,
    TRAINING_COLUMNS,
)


INPUT_FILE = "data/historical/historical_gws.csv"
OUTPUT_FILE = "data/processed/training_data.csv"


def load_data():
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Make sure observations are chronologically ordered
    df = df.sort_values(
        ["season", "element", "GW"]
    ).reset_index(drop=True)

    return df


def convert_numeric_columns(df):
    numeric_columns = [
        "minutes",
        "total_points",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "starts",
        "value",
        "selected",
        "transfers_in",
        "transfers_out",
        "transfers_balance",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df


def create_features(df):
    # Identify the same player within the same season
    player_group = df.groupby(
        ["season", "element"],
        group_keys=False
    )

    # What happened in the player's PREVIOUS fixture?
    for column in LAG_SOURCE_COLUMNS:
        df[f"{column}_last"] = (
            player_group[column].shift(1)
        )

    # Rolling averages based ONLY on previous matches.
    #
    # shift(1) is crucial:
    # when predicting GW10, GW10 itself must not be
    # included in the average.
    for column in ROLLING_SOURCE_COLUMNS:
        for window in ROLLING_WINDOWS:
            df[f"{column}_avg_{window}"] = (
                player_group[column]
                .transform(
                    lambda x, window=window:
                        x.shift(1)
                        .rolling(window, min_periods=1)
                        .mean()
                )
            )

    return df


def build_season_team_name_maps(df):
    """Derive each season's {numeric opponent_team id: team name} mapping
    directly from historical_gws.csv itself -- no external reference
    file, no guessing.

    WHY THIS WORKS WITHOUT AN EXTERNAL SOURCE: every played fixture pairs
    exactly two teams. A row's "team" is already a name (unchanged from
    the raw data); its "opponent_team" is the OTHER side's raw numeric
    id. So for any given "fixture" id within a season, the two rows'
    (team name, opponent_team id) pairs directly hand us both mappings at
    once: team A's recorded opponent id IS team B's own id, and vice
    versa. This is scoped per (season, fixture), so it can never mix up
    two different seasons' id spaces.

    Returns (maps, problems):
      maps: {season: {id: name}}
      problems: list of human-readable strings for any fixture that
        didn't cleanly resolve to exactly two sides, or any id/name that
        resolved to conflicting values within the same season. Expected
        to be empty for clean data -- reported by main(), not silently
        swallowed.
    """

    maps = {}
    problems = []

    for season, season_df in df.groupby("season"):
        id_to_name = {}
        name_to_id = {}

        for fixture_id, fixture_df in season_df.groupby("fixture"):
            sides = fixture_df[["team", "opponent_team"]].drop_duplicates()

            if len(sides) != 2:
                problems.append(
                    f"{season} fixture {fixture_id}: expected 2 sides, "
                    f"found {len(sides)} distinct (team, opponent_team) "
                    f"pairs -- skipped, opponent_name left unresolved "
                    f"for this fixture."
                )
                continue

            (name_a, opponent_id_of_a), (name_b, opponent_id_of_b) = (
                sides.itertuples(index=False)
            )

            # name_a's recorded opponent id IS name_b's own id, and
            # vice versa.
            resolved = [(name_b, opponent_id_of_a), (name_a, opponent_id_of_b)]

            for name, own_id in resolved:
                if name in name_to_id and name_to_id[name] != own_id:
                    problems.append(
                        f"{season} fixture {fixture_id}: {name} previously "
                        f"resolved to id {name_to_id[name]}, now {own_id} "
                        f"-- keeping the first value seen."
                    )
                    continue

                if own_id in id_to_name and id_to_name[own_id] != name:
                    problems.append(
                        f"{season} fixture {fixture_id}: id {own_id} "
                        f"previously resolved to {id_to_name[own_id]}, "
                        f"now {name} -- keeping the first value seen."
                    )
                    continue

                id_to_name[own_id] = name
                name_to_id[name] = own_id

        maps[season] = id_to_name

    return maps, problems


def add_stable_team_names(df, season_maps):
    """Rename "team" -> "team_name" (already a name, unchanged values)
    and add "opponent_name" (derived via season_maps) alongside the
    existing raw "opponent_team" id column.

    Nothing here touches LAG_SOURCE_COLUMNS/ROLLING_SOURCE_COLUMNS or the
    shift(1)/rolling logic in create_features() -- this only concerns the
    team/opponent identity columns.
    """

    df = df.rename(columns={"team": "team_name"})

    lookup_rows = [
        {"season": season, "opponent_team": team_id, "opponent_name": name}
        for season, id_to_name in season_maps.items()
        for team_id, name in id_to_name.items()
    ]
    lookup = pd.DataFrame(lookup_rows)

    df = df.merge(lookup, on=["season", "opponent_team"], how="left")

    return df


def select_training_columns(df):
    return df[TRAINING_COLUMNS]


def report_team_name_validation(df, problems):
    """Print the checks required before trusting the derived
    opponent_name column: any fixture-pairing problems found while
    building the map, and how many rows ended up without a resolved
    opponent_name (should be zero for clean data).
    """

    print()
    print("Team/opponent name mapping validation:")
    print("-" * 50)

    if problems:
        print(f"{len(problems)} problem(s) found while deriving id<->name maps:")
        for problem in problems:
            print(f"  - {problem}")
    else:
        print("No fixture-pairing problems found across all seasons. OK.")

    unresolved = df["opponent_name"].isna().sum()
    if unresolved:
        print(
            f"WARNING: {unresolved:,} row(s) have no resolved "
            f"opponent_name (opponent_team id not found in that season's "
            f"derived map) -- these will have a missing opponent_name "
            f"feature, left for the model's imputer to handle."
        )
    else:
        print(f"Every row resolved to a real opponent_name. OK.")

    for season, season_df in df.groupby("season"):
        n_teams = season_df["team_name"].nunique()
        n_opponents = season_df["opponent_name"].dropna().nunique()
        print(
            f"  {season}: {n_teams} distinct team_name value(s), "
            f"{n_opponents} distinct opponent_name value(s) "
            f"(expect 20 and 20 for a normal season)."
        )


def main():
    df = load_data()
    df = convert_numeric_columns(df)
    df = create_features(df)

    season_maps, problems = build_season_team_name_maps(df)
    df = add_stable_team_names(df, season_maps)

    report_team_name_validation(df, problems)

    training = select_training_columns(df)

    # First appearance has no historical information,
    # so remove observations without previous minutes.
    training = training.dropna(
        subset=["minutes_last"]
    )

    training.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Training rows: {len(training):,}")
    print(f"Training columns: {len(training.columns)}")

    print("\nExample:")
    print(training.head().to_string(index=False))


if __name__ == "__main__":
    main()
