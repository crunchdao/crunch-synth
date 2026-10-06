import pandas as pd
import numpy as np

from crunch_synth.utils.densitytosimulations import simulate_paths
from crunch_synth.prices import PriceStore

def plot_quarantine(asset, quarantine_entry, step: int, prices: PriceStore, mode="direct", title="", lookback_seconds=3600):
    """
    Plot the predicted price/return distribution (quarantine) for a given asset.

    Parameters
    ----------
    asset : str
        Asset symbol, e.g., "BTC".
    quarantine_entry : tuple
        Tuple of (timestamp, predictions, step) representing a single quarantine entry.
        - timestamp (int): the reference time of the quarantine
        - predictions (list): predicted distributions for each future step
        - step (int): time interval in seconds between prediction steps
    prices : PriceStore
        Object that holds historical price data for the asset.
    mode : str, default "direct"
        - "direct": predictions are returns
        - otherwise: predictions are in price space
    title : str
        Plot title.
    """

    ts, predictions, steps = quarantine_entry

    if step not in predictions:
        return

    predictions = predictions[step]
    horizon = step * len(predictions)

    # Skip if the reference timestamp is after the last known price
    if ts > prices.get_last_price(asset)[0]:
        return

    # Determine starting price for simulation
    # If returns ("direct") mode, use 0.0 as the starting point
    start_price = prices.get_closest_price(asset, ts - step * len(predictions))[1]

    # Simulate multiple paths based on the predictions
    # This creates a Monte Carlo distribution of potential price trajectories   
    simulations = simulate_paths(
        predictions,
        start_point=0.0 if mode=="direct" else start_price,
        num_paths=10000,
        step_minutes=None,
        start_time=None,
        mode=mode
    )

    # Create a DataFrame to store simulated mean and confidence intervals
    scales_df = pd.DataFrame({
        "mean": simulations["mean"],
        "q_low_paths": simulations["q_low_paths"],
        "q_high_paths": simulations["q_high_paths"],
    })

    # Map timestamps for each prediction step
    scales_df["ts"] = [ts - step * i for i in range(len(scales_df) - 1, -1, -1)]
    scales_df["time"] = pd.to_datetime(scales_df["ts"], unit="s", utc=True)

    # Attach the historical price for each timestamp
    scales_df["price"] = [prices.get_closest_price(asset, ts)[1] for ts in scales_df["ts"]]
    scales_df["return"] = scales_df["price"].diff().fillna(0.0)
    # print(scales_df)

    # Build historical price context before forecast
    forecast_origin_ts = scales_df["ts"].iloc[0]

    hist_ts = []
    hist_price = []

    t = forecast_origin_ts - lookback_seconds
    while t < forecast_origin_ts:
        p = prices.get_closest_price(asset, t)
        if p:
            hist_ts.append(p[0])
            hist_price.append(p[1])
        t += step

    history_df = pd.DataFrame({
        "ts": hist_ts + [scales_df["ts"].iloc[0]],
        "price": hist_price + [scales_df["price"].iloc[0]],
    })
    history_df["time"] = pd.to_datetime(history_df["ts"], unit="s", utc=True)


    import plotly.graph_objects as go

    title=f"Predicted {asset} {'return price' if mode=="direct" else "price"} distribution (horizon={horizon}s | step={step}s) at {scales_df["time"].iloc[0]}"

    # Create a filled band between q05 and q95
    fig = go.Figure()

    # Lower bound (q05)
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["q_low_paths"],
        mode='lines',
        line=dict(width=0),
        name=f'{round(simulations["quantile_range"][0]*100)}th percentile',
        showlegend=False
    ))

    # Upper bound (q95) with fill to previous trace
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["q_high_paths"],
        mode='lines',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.3)',
        name=f"Predicted range ({round(100*(simulations["quantile_range"][1] - simulations["quantile_range"][0]))}%)"
    ))

    # Mean prediction line with shadow effect
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["mean"],
        mode='lines',
        line=dict(color='firebrick', width=3),
        name='Predicted mean'
    ))

    # Actual price / returns line with markers
    fig.add_trace(go.Scatter(
        x=scales_df["time"],
        y=scales_df["return"] if mode=="direct" else scales_df["price"],
        mode='lines',  # 'lines+markers'
        line=dict(color='royalblue', width=2),
        marker=dict(size=4, opacity=0.7),
        name='Observed relative price' if mode=="direct" else "Observed price"
    ))

    if not mode=="direct":
        # Historical price (pre-forecast context)
        fig.add_trace(go.Scatter(
            x=history_df["time"],
            y=history_df["price"],
            mode="lines",
            line=dict(color="grey", width=2),#, dash="dot"),
            name="Historical price",
        ))

    if not mode=="direct":
        fig.add_vline(
            x=scales_df["time"].iloc[0].timestamp()*1000 - 3600*1000,
            line_dash="dash",
            line_color="black",
            annotation_text="Forecast origin",
            annotation_position="top"
        )

        fig.add_vrect(
            x0=scales_df["time"].iloc[0],
            x1=scales_df["time"].iloc[-1],
            fillcolor="rgba(255,165,0,0.05)",
            layer="below",
            line_width=0,
        )

    # Layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        hovermode='x unified',
        xaxis=dict(title='Time', showgrid=True, gridcolor='lightgrey'),
        yaxis=dict(title='Return' if mode=="direct" else 'Price', showgrid=True, gridcolor='lightgrey'),
        plot_bgcolor='white',
        legend=dict(x=1.02, y=1, bordercolor='Black', borderwidth=1)
    )

    fig.show()


def plot_prices(data, title):
    import plotly.express as px

    df = pd.DataFrame(data, columns=["ts", "price"])

    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Create a line graph using Plotly
    fig = px.line(
        df,
        x="time",
        y="price",
        title=title,
    )

    # Show the graph
    fig.show()


def plot_log_return_prices(data, title):
    import plotly.express as px

    df = pd.DataFrame(data, columns=["ts", "price"])

    df["log-return"] = np.log(df["price"]).diff().fillna(0.0)

    # print(df.describe())

    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Create a line graph using Plotly
    fig = px.line(
        df,
        x="time",
        y="log-return",
        title=title,
    )

    # Show the graph
    fig.show()

def plot_scores(data):
    import plotly.express as px

    df = pd.DataFrame([
        {"asset": asset, "ts": ts, "score": score}
        for asset, records in data.items()
        for ts, score in records
    ])

    df["time"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # average score accross asset
    # df_avg = df.groupby("time", as_index=False)["score"].mean()

    start_scores = df['time'].iloc[0]
    end_scores = df['time'].iloc[-1]
    assets = df["asset"].unique().tolist()
    title = f"{assets} crps scores from {start_scores} to {end_scores}"
    
    # Create a line graph using Plotly
    fig = px.line(
        df,
        x="time",
        y="score",
        color="asset",
        markers=True,
        title=title,
    )

    # Show the graph
    fig.show()

def plot_volatility(asset, quarantine_entry, prices: PriceStore):
    """
    Plot the volatility CRPS breakdown of one 1h prediction (live 1h scoring
    adds this term, see `crunch_synth.volatility`).

    For every scored block (the full hour, four 15-minute blocks and twelve
    5-minute blocks), the box shows the volatility of the paths simulated from
    the predicted densities and the marker the realized volatility.

    Parameters
    ----------
    asset : str
        Asset symbol, e.g., "BTC".
    quarantine_entry : tuple
        Tuple of (timestamp, predictions, steps), as returned by
        `TrackerEvaluator.evaluate_quarantine()`.
    prices : PriceStore
        Object that holds historical price data for the asset.
    """
    import plotly.graph_objects as go
    from crunch_synth.volatility import (
        VOL_CRPS_ASSETS, VOL_SCORING_BLOCKS, block_volatilities, crps_ensemble, get_interval_steps, score_volatility,
    )

    result = score_volatility(asset, quarantine_entry, prices)
    if result is None:
        print(f"No volatility score for this prediction: only 1h predictions on {sorted(VOL_CRPS_ASSETS)} "
              "with realized prices over the hour are volatility-scored.")
        return

    sim_paths = result["simulated_paths"]
    real_path = result["realized_path"]
    time_increment = result["time_increment"]
    start_time = pd.to_datetime(quarantine_entry[0] - time_increment * (len(real_path) - 1), unit="s", utc=True)

    fig = go.Figure()
    for name, (block_seconds, weight) in VOL_SCORING_BLOCKS.items():
        block_steps = get_interval_steps(block_seconds, time_increment)
        simulated_vol = block_volatilities(sim_paths, block_steps)
        real_vol = block_volatilities(real_path.reshape(1, -1), block_steps)[0]
        minutes = block_seconds // 60

        for block in range(real_vol.size):
            block_start = start_time + pd.Timedelta(seconds=block * block_seconds)
            label = f"{minutes}m {block_start:%H:%M}"
            crps = crps_ensemble(real_vol[block], simulated_vol[:, block])

            fig.add_trace(go.Box(
                y=simulated_vol[:, block],
                name=label,
                marker_color="orange",
                boxpoints=False,
                showlegend=False,
                hovertemplate=f"{label}<br>Simulated vol: %{{y:.2f}} bps<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=[label],
                y=[real_vol[block]],
                mode="markers",
                marker=dict(color="royalblue", size=10, symbol="diamond"),
                name="Realized volatility",
                legendgroup="realized",
                showlegend=(name == "vol_60min" and block == 0),
                hovertemplate=f"{label}<br>Realized vol: %{{y:.2f}} bps<br>CRPS: {crps:.3f} (weight {weight:.3f})<extra></extra>",
            ))

    title = (
        f"{asset} 1h volatility CRPS from {start_time:%Y-%m-%d %H:%M} UTC: "
        f"{result['vol_crps']:.3f} (adds {result['vol_score']:.4f} to the score)"
    )
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=18)),
        xaxis=dict(title="Block (size and start time)", showgrid=True, gridcolor="lightgrey"),
        yaxis=dict(title="Volatility of 1-minute returns (bps)", showgrid=True, gridcolor="lightgrey"),
        plot_bgcolor="white",
        legend=dict(x=1.02, y=1, bordercolor="Black", borderwidth=1),
    )

    fig.show()
