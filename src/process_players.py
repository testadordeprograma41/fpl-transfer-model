import pandas as pd


RAW_FILE = "data/raw/players.csv"
OUTPUT_FILE = "data/processed/players_processed.csv"


def load_players():
    return pd.read_csv(RAW_FILE)


def clean_players(players):
    # Convert FPL string fields into numbers
    numeric_columns = [
        "form",
        "points_per_game",
        "selected_by_percent",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    ]

    for column in numeric_columns:
        players[column] = pd.to_numeric(
            players[column],
            errors="coerce"
        )

    # FPL stores price as tenths:
    # 105 = £10.5m
    players["price"] = players["now_cost"] / 10

    # Convert element_type into readable positions
    position_map = {
        1: "GK",
        2: "DEF",
        3: "MID",
        4: "FWD",
    }

    players["position"] = players["element_type"].map(position_map)

    # Create full player name
    players["player_name"] = (
        players["first_name"] + " " + players["second_name"]
    )

    return players


def create_features(players):
    # Simple points-per-million measure
    players["value_season"] = (
        players["total_points"] / players["price"]
    )

    # Expected attacking involvement per 90 minutes
    players["xgi_per_90"] = (
        players["expected_goal_involvements"]
        / players["minutes"]
        * 90
    )

    # Avoid misleading infinity values for players with 0 minutes
    players["xgi_per_90"] = players["xgi_per_90"].fillna(0)

    return players


def select_columns(players):
    columns = [
        "player_name",
        "team_name",
        "position",
        "price",
        "total_points",
        "points_per_game",
        "form",
        "minutes",
        "selected_by_percent",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "xgi_per_90",
        "value_season",
    ]

    return players[columns]


def main():
    players = load_players()

    players = clean_players(players)
    players = create_features(players)
    players = select_columns(players)

    players.to_csv(OUTPUT_FILE, index=False)

    print("\nTop 20 players by total points:\n")

    print(
        players
        .sort_values("total_points", ascending=False)
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()