import os
import glob
import json
import pandas as pd
import plotly.express as px

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
