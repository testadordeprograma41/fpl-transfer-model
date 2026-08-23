import requests
import pandas as pd

URL = "https://fantasy.premierleague.com/api/bootstrap-static/"


def fetch_fpl_data():
    response = requests.get(URL, timeout=30)
    response.raise_for_status()
    return response.json()


def build_players_dataframe(data):
    players = pd.DataFrame(data["elements"])
    teams = pd.DataFrame(data["teams"])

    team_map = teams.set_index("id")["name"].to_dict()

    players["team_name"] = players["team"].map(team_map)
    players["price"] = players["now_cost"] / 10

    return players


def main():
    data = fetch_fpl_data()
    players = build_players_dataframe(data)

    columns = [
        "first_name",
        "second_name",
        "team_name",
        "total_points",
        "price",
        "minutes",
    ]

    top_players = (
        players[columns]
        .sort_values("total_points", ascending=False)
        .head(20)
    )
    players.to_csv("data/raw/players.csv", index=False)
    print(top_players.to_string(index=False))


if __name__ == "__main__":
    main()