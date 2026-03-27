import numpy as np


def count_evaluations(history_price, horizon, interval):
    ts_values = [ts for ts, _ in history_price]
    count = 0
    prev_ts = ts_values[0]
    for ts in ts_values[1:]:
        if ts - prev_ts >= interval:
            if ts - ts_values[0] >= horizon:
                count += 1
            prev_ts = ts
    return count


def build_events(history_test_prices, df_trackers_history, asset, horizon, interval):
    """
    Build prediction and evaluation event timelines for a given asset.

    Two types of timestamps are generated:
    - performed_at: when predictions should be issued
    - resolvable_at: when predictions can be evaluated (i.e., enough future data is available)

    The function supports two modes:
    1. Replay mode (historical):
       If `df_trackers_history` is provided, reuse the exact event timestamps
       from past tracker runs to ensure fair comparison.
    2. Simulation mode (dynamic):
       If no history is available, generate events at a fixed interval.

    Parameters
    ----------
    history_test_prices : list of (timestamp, price)
        Time-ordered test price series for the asset.
    df_trackers_history : pd.DataFrame or None
        Historical tracker results containing 'performed_at' and 'resolvable_at'.
    asset : str
        Asset identifier.
    horizon : int
        Forecast horizon (in seconds). Determines when predictions become evaluable.
    interval : int
        Time interval (in seconds) between consecutive predictions.

    Returns
    -------
    performed : np.ndarray
        Array of timestamps when predictions should be triggered.
    resolvable : np.ndarray
        Array of timestamps when predictions can be evaluated.
    """

    # 1. Replay historical event timeline (if available)
    if df_trackers_history is not None and len(df_trackers_history) > 0:

        # Filter events for the given asset
        df_asset_events = (
            df_trackers_history[df_trackers_history['asset'] == asset][
                ['performed_at', 'resolvable_at']
            ]
            .drop_duplicates()
            .sort_values(['performed_at', 'resolvable_at'])
            .reset_index(drop=True)
        )

        # Convert datetime to integer timestamps
        performed = df_asset_events['performed_at'].apply(lambda dt: int(dt.timestamp()))
        resolvable = df_asset_events['resolvable_at'].apply(lambda dt: int(dt.timestamp()))

    # 2. Dynamically generate event timeline
    else:
        # Extract first and last timestamps from test data
        first_timestamp = history_test_prices[0][0]
        last_timestamp = history_test_prices[-1][0]

        # Generate prediction timestamps:
        # Start at first_timestamp and step forward every `interval`
        # Stop early enough so that a full horizon can still be evaluated
        performed = np.arange(
            first_timestamp,
            last_timestamp + 1 - horizon,
            interval
        )

        # Each prediction becomes evaluable after `horizon` seconds
        resolvable = performed + horizon

    return np.asarray(performed), np.asarray(resolvable)


def compute_ranks(df_trackers_history, tracker_evaluator, asset):
    """
    Compute ranking of the current tracker against historical trackers for a given asset.
    Ranking is based on the average score per tracker (lower is better).
    """

    # 1. Aggregate historical scores per tracker
    df_asset_history = df_trackers_history[df_trackers_history['asset'] == asset]

    # Compute mean score per tracker
    mean_scores_by_tracker = (
        df_asset_history
        .groupby('tracker')['score']
        .mean()
        .sort_values()
    )

    # 2. Current tracker score
    my_score = tracker_evaluator.overall_score_asset(asset)

    # Rank
    my_rank = (mean_scores_by_tracker < my_score).sum() + 1

    # Include current tracker in total count
    total_trackers = len(mean_scores_by_tracker) + 1

    # 3. Benchmark comparison
    benchmark_score = mean_scores_by_tracker.get('benchmark', np.nan)

    if 'benchmark' in mean_scores_by_tracker.index:
        benchmark_rank = (mean_scores_by_tracker < benchmark_score).sum() + 1
    else:
        benchmark_rank = np.nan

    return my_score, my_rank, benchmark_score, benchmark_rank, total_trackers