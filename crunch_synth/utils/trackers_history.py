import pandas as pd
from datetime import datetime, timedelta


def load_and_prepare_trackers_history(horizon: int, assets: list[str], evaluation_end: datetime, days: int):
    """ Load and filter trackers history for a specific evaluation setup from a .csv endpoint """

    source_path = "https://raw.githubusercontent.com/crunchdao/crunch-synth/master/crunch_synth/examples/trackers_history/df_trackers_history.csv"

    df = pd.read_csv(source_path)

    df["performed_at"] = pd.to_datetime(df["performed_at"], utc=True)
    df["resolvable_at"] = pd.to_datetime(df["resolvable_at"], utc=True)

    # Filter by horizon and assets
    df_filtered = df[(df["horizon"] == horizon) & (df["asset"].isin(assets))].copy()

    # Capture full history time range (before time filtering)
    if not df_filtered.empty:
        df_min_time = df_filtered["performed_at"].min()
        df_max_time = df_filtered["performed_at"].max()
    else:
        df_min_time, df_max_time = None, None

    # Filter by evaluation window
    start_time = evaluation_end - timedelta(days=days)
    end_time = evaluation_end

    if df_min_time is not None and start_time < df_min_time:
        print(
            "[WARNING] Evaluation start_time is earlier than available tracker history.\n"
            f"Test window: [{start_time}, {end_time}]\n"
            f"Tracker history window: [{df_min_time}, {df_max_time}]\n"
            "Tracker history will NOT be used."
        )
        return df_filtered.iloc[0:0]

    df_filtered = df_filtered[(df_filtered["performed_at"] > start_time) & (df_filtered["performed_at"] < end_time)].copy()

    if df_filtered.empty:
        print(
            "[WARNING] No overlapping events. Tracker history does not intersect "
            f"Test window: [{start_time}, {end_time}]\n"
            f"Tracker history window: [{df_min_time}, {df_max_time}]"
        )

    return df_filtered