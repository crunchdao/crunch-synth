from datetime import datetime, timezone, timedelta
import os
import glob
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def load_test_prices_once(assets, pricedb, evaluation_end, days=30):
    """
    Load the test price data

    Returns
    -------
    dict[str, list[(timestamp, price)]]
    """
    to = evaluation_end if evaluation_end else datetime.now(timezone.utc)
    from_ = to - timedelta(days=days)

    test_asset_prices = {}

    for asset in assets:
        test_asset_prices[asset] = pricedb.get_price_history(
            asset=asset,
            from_=from_,
            to=to,
        )

    return test_asset_prices


def load_initial_price_histories_once(assets, pricedb, evaluation_end, days_history=30, days_offset=30):
    """
    Load initial historical data for the tracker (e.g., the 30 days BEFORE the test window)

    Parameters
    ----------
    days_history : int
        Amount of warm-up history to load
    days_offset : int
        Gap between last warm-up history and evaluation_end

    Returns
    -------
    dict[str, list[(timestamp, price)]]
    """
    to_test = evaluation_end if evaluation_end else datetime.now(timezone.utc)
    from_test = to_test - timedelta(days=days_offset)

    histories = {}

    for asset in assets:
        histories[asset] = pricedb.get_price_history(
            asset=asset,
            from_=from_test - timedelta(days=days_history),
            to=from_test,
        )

    return histories


def visualize_price_data(
    history_data: dict,
    test_data: dict,
    selected_assets: list | None = None,
    show_graph: bool = True,
) -> pd.DataFrame:
    """ Visualize historical and test price data side by side for each asset. """
    rows = []

    def append_rows(data: dict, split: str):
        for asset, records in data.items():
            if selected_assets is not None and asset not in selected_assets:
                continue
            for ts, price in records:
                rows.append({
                    "asset": asset,
                    "ts": int(ts),
                    "price": float(price),
                    "split": split,
                })

    append_rows(history_data, split="history")
    append_rows(test_data, split="test")

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    if show_graph:
        print("Dataset:")
        for asset, g in df.groupby("asset", sort=False):
            g = g.sort_values("time")

            g_hist = g[g["split"] == "history"]
            g_test = g[g["split"] == "test"]

            fig = go.Figure()

            # --- History ---
            if not g_hist.empty:
                fig.add_trace(
                    go.Scatter(
                        x=g_hist["time"],
                        y=g_hist["price"],
                        mode="lines",
                        line=dict(color="rgba(120,120,120,0.8)", width=2),
                        name="History",
                    )
                )

            # --- Test ---
            if not g_test.empty:
                fig.add_trace(
                    go.Scatter(
                        x=g_test["time"],
                        y=g_test["price"],
                        mode="lines",
                        line=dict(color="#1f77b4", width=2),
                        name="Test",
                    )
                )

                # Vertical separator at test start
                test_start_time = g_test["time"].iloc[0].timestamp()*1000 - 3600*1000
                fig.add_vline(
                    x=test_start_time,
                    line_dash="dash",
                    line_color="black",
                    annotation_text="Test start",
                    annotation_position="top left",
                )

            fig.update_layout(
                title=dict(
                    text=f"{asset} — History & Test Prices",
                    x=0.5,
                    xanchor="center",
                    font=dict(size=18),
                ),
                xaxis_title="Time (UTC)",
                yaxis_title="Price",
                hovermode="x unified",
                plot_bgcolor="white",
                xaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
                yaxis=dict(showgrid=True, gridcolor="rgba(200,200,200,0.4)"),
                legend=dict(
                    orientation="h",
                    y=-0.15,
                ),
                margin=dict(l=60, r=30, t=70, b=50),
            )

            fig.show()

    return df


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


##################################################
# Tracker Comparison
##################################################


def load_scores_json(path: str):
    
    with open(path, "r") as f:
        return json.load(f)
    

def scores_json_to_df(scores_json):
    # start = scores_json["period"]["start"]
    # interval = scores_json["interval"]
    horizon = scores_json["horizon"]

    rows = []
    tracker_name = scores_json["tracker"]

    for asset, score_list in scores_json["asset_scores"].items():
        for i, score_data in enumerate(score_list):
            rows.append({
                "tracker": tracker_name,
                "asset": asset,
                "horizon": horizon,
                "ts": score_data["ts"],
                "score": score_data["score"],
            })

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df


def load_all_results(current_results_directory, horizon=None):
    """
    Load all JSON results files matching *_h{horizon}.json
    and return a concatenated DataFrame.
    If horizon is None then take all JSON results
    """

    if horizon is None:
        pattern = "*.json"
    else:
        pattern = f"*h{horizon}.json"
    search_path = os.path.join(current_results_directory, pattern)
    print(f"Directory: {search_path}")

    # Find matching files
    files = glob.glob(search_path)

    if not files:
        print(f"[!] No result files found matching {pattern}")
        return pd.DataFrame()

    print(f"[✔] Found {len(files)} files:")
    for f in files:
        print("   -", os.path.basename(f))

    # Load all files
    dfs = [scores_json_to_df(load_scores_json(f)) for f in files]

    # Combine
    df_all = pd.concat(dfs, ignore_index=True)

    return df_all


def plot_tracker_comparison(df_all, asset=None):
    """
    df_all must contain: columns ['time', 'asset', 'tracker', 'score']
    """
    df_plot = df_all.copy()

    if asset is None:
        asset = df_plot["asset"].unique().tolist()
    else:
        if isinstance(asset, str):
            df_plot = df_plot[df_plot["asset"] == asset]
        else:
            df_plot = df_plot[df_plot["asset"].isin(asset)]

    df_plot = df_plot.groupby(["time", "tracker"])["score"].mean().reset_index()

    # ---- Compute stats per tracker ----
    tracker_means = df_plot.groupby("tracker")["score"].mean()

    # Count best-times: each timestamp → who had lowest score
    best_counts = (
        df_plot.loc[df_plot.groupby("time")["score"].idxmin()]
        .groupby("tracker")
        .size()
    )

    # ---- Build custom legend labels ----
    legend_names = {}
    for tracker in df_plot["tracker"].unique():
        mean_val = tracker_means.get(tracker, float("nan"))
        best_val = best_counts.get(tracker, 0)

        legend_names[tracker] = (
            f"{tracker} (mean={mean_val:.3f} | best {best_val} times)"
        )

    # ---- Replace tracker column with custom label ----
    df_plot["tracker"] = df_plot["tracker"].map(legend_names)

    fig = px.line(
        df_plot,
        x="time",
        y="score",
        color="tracker",
        title=f"Tracker Comparison {asset} — Normalized CRPS Over Time",
    )

    fig.update_traces(mode="lines+markers")
    fig.update_layout(hovermode="x unified")

    # fig.update_layout(
    #     legend=dict(
    #         orientation="v",
    #         yanchor="bottom",
    #         y=-0.6,
    #         xanchor="left",
    #         x=0.0,
    #         bgcolor="rgba(0,0,0,0)",
    #     ),
    #     margin=dict(t=150)
    # )


    fig.show()
