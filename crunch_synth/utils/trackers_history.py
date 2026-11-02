import io
import pandas as pd
import requests
from datetime import datetime, timedelta, timezone


ROUND_SCORES_URL = "http://synth.coordinator.crunchdao.com:8000/reports/rounds/scores"
ROUND_SCORES_MAX_RANGE = timedelta(days=10)

COLUMNS = ["tracker", "horizon", "asset", "performed_at", "resolvable_at", "score"]


def _fetch_round_scores(asset: str, horizon: int, start: datetime, end: datetime, timeout: int = 120) -> pd.DataFrame:
    response = requests.get(
        ROUND_SCORES_URL,
        params={"asset": asset, "horizon": horizon, "start": start.isoformat(), "end": end.isoformat()},
        timeout=timeout,
    )
    response.raise_for_status()

    df = pd.read_csv(io.StringIO(response.text), dtype={"tracker": str})
    df["performed_at"] = pd.to_datetime(df["performed_at"], unit="s", utc=True)
    df["resolvable_at"] = df["performed_at"] + pd.Timedelta(seconds=horizon)
    df["horizon"] = horizon
    df["asset"] = asset
    return df[COLUMNS]


def load_and_prepare_trackers_history(horizon: int, assets: list[str], evaluation_end: datetime, days: int):
    """ Load the live scores of every tracker for a specific evaluation setup """

    # Live leaderboard scores (raw CRPS, as computed by the coordinator, including the volatility
    # term on 1 hour) of every tracker on every round of the evaluation window; only trackers with
    # at least 80% of the rounds of an asset, so their mean scores are comparable.
    # (The df_trackers_history*.csv files of the repository stay for older versions of the package.)
    if evaluation_end.tzinfo is None:
        evaluation_end = evaluation_end.replace(tzinfo=timezone.utc)
    start_time = evaluation_end - timedelta(days=days)
    end_time = evaluation_end

    frames = []
    for asset in assets:
        # longer windows are fetched in parts (the 80% rule then applies to each part)
        part_start = start_time
        while part_start < end_time:
            part_end = min(part_start + ROUND_SCORES_MAX_RANGE, end_time)
            try:
                frames.append(_fetch_round_scores(asset, horizon, part_start, part_end))
            except Exception as error:
                print(f"[WARNING] Could not load the live scores of {asset} from {part_start} to {part_end}: {error}")
            part_start = part_end

    df_filtered = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    df_filtered = df_filtered[(df_filtered["performed_at"] > start_time) & (df_filtered["performed_at"] < end_time)].copy()

    if df_filtered.empty:
        print(
            "[WARNING] No live scores for this evaluation window (they are kept 120 days).\n"
            f"Test window: [{start_time}, {end_time}]\n"
            "Tracker history will NOT be used."
        )

    return df_filtered
