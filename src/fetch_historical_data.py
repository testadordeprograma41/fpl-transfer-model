from pathlib import Path

import pandas as pd


BASE_URL = (
    "https://raw.githubusercontent.com/"
    "vaastav/Fantasy-Premier-League/master/data"
)

SEASONS = [
    "2023-24",
    "2024-25",
    "2025-26",
]

OUTPUT_DIR = Path("data/historical")


def fetch_season(season):
    url = f"{BASE_URL}/{season}/gws/merged_gw.csv"

    print(f"Downloading {season}...")

    df = pd.read_csv(url)

    df["season"] = season

    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_seasons = []

    for season in SEASONS:
        df = fetch_season(season)

        output_file = OUTPUT_DIR / f"{season}.csv"
        df.to_csv(output_file, index=False)

        print(
            f"Saved {season}: "
            f"{len(df):,} rows, {len(df.columns)} columns"
        )

        all_seasons.append(df)

    combined = pd.concat(
        all_seasons,
        ignore_index=True
    )

    combined_file = OUTPUT_DIR / "historical_gws.csv"

    combined.to_csv(
        combined_file,
        index=False
    )

    print()
    print(
        f"Combined dataset: "
        f"{len(combined):,} rows"
    )

    print(
        f"Saved to: {combined_file}"
    )


if __name__ == "__main__":
    main()