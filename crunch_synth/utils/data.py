from datetime import datetime, timezone, timedelta
import pandas as pd
import plotly.graph_objects as go

from crunch_synth.price_provider import pricedb

def load_test_prices_once(assets, evaluation_end, days=30):
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


def load_initial_price_histories_once(assets, evaluation_end, days_history=30, days_offset=30):
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