# Source columns (from the raw historical gameweek data) that feed the
# "previous fixture" lag features. Used by build_training_data.py to build
# these features, and reflected below in the derived feature names so the
# two stay in sync.
LAG_SOURCE_COLUMNS = [
    "minutes",
    "total_points",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "starts",
]

# Source columns that feed the rolling-average features.
ROLLING_SOURCE_COLUMNS = [
    "minutes",
    "total_points",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
]

# Window sizes (in matches) used for the rolling averages.
ROLLING_WINDOWS = [3, 5]

# Columns that identify an observation rather than describe it.
IDENTIFICATION_COLUMNS = [
    "season",
    "GW",
    "element",
    "name",
    "position",
    "team",
]

# Information that is known ahead of the upcoming fixture (as opposed to
# results, which are only known after it has been played).
FIXTURE_KNOWN_COLUMNS = [
    "value",
    "opponent_team",
    "was_home",
]

TARGET_COLUMN = "total_points"


def lag_feature_names():
    """Feature names produced from LAG_SOURCE_COLUMNS (e.g. 'minutes_last')."""

    return [f"{column}_last" for column in LAG_SOURCE_COLUMNS]


def rolling_feature_names():
    """Feature names produced from ROLLING_SOURCE_COLUMNS, grouped by window
    (e.g. 'minutes_avg_3', ..., 'minutes_avg_5', ...)."""

    return [
        f"{column}_avg_{window}"
        for window in ROLLING_WINDOWS
        for column in ROLLING_SOURCE_COLUMNS
    ]


NUMERIC_FEATURES = (
    ["value"]
    + lag_feature_names()
    + rolling_feature_names()
)

CATEGORICAL_FEATURES = [
    "position",
    "team",
    "opponent_team",
    "was_home",
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

# Full set of columns kept when building the training dataset:
# identification, fixture-known info, derived features, and the target.
TRAINING_COLUMNS = (
    IDENTIFICATION_COLUMNS
    + FIXTURE_KNOWN_COLUMNS
    + lag_feature_names()
    + rolling_feature_names()
    + [TARGET_COLUMN]
)
