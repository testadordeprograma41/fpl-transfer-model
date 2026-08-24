from sklearn.model_selection import TimeSeriesSplit


def build_time_series_cv(n_splits=5):
    """Chronological cross-validation splitter for model comparison/tuning.

    Plain K-Fold CV shuffles rows randomly, which for this dataset would let
    a model be FIT on rows from later gameweeks while being VALIDATED on
    earlier ones -- something that can never happen when the model is
    actually used (it only ever predicts forward from what has already
    happened). TimeSeriesSplit instead grows the training window forward in
    time, so every validation fold comes strictly after its training fold,
    matching how the model is used in practice.

    IMPORTANT: the data passed to this CV splitter must already be sorted
    chronologically (see modeling.data.sort_chronologically) -- TimeSeriesSplit
    only looks at row order/position, so an unsorted frame silently produces
    meaningless folds instead of an error.
    """

    return TimeSeriesSplit(n_splits=n_splits)
