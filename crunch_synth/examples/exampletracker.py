from datetime import datetime, timezone, timedelta
import numpy as np
from tqdm.auto import tqdm

from crunch_synth import TrackerBase, TrackerEvaluator


class GaussianStepTracker(TrackerBase):
    """
    A benchmark tracker that models *future incremental returns* as Gaussian-distributed.

    For each forecast step, the tracker returns a normal distribution
    r_{t,step} ~ N(a · mu, √a · sigma) where:
        - mu    = mean historical return
        - sigma = std historical return
        - a = (step / 300) represents the ratio of the forecast step duration to the historical 5-minute return interval.

    Multi-resolution forecasts (5min, 1h, 6h, 24h, ...)
    are automatically handled by `TrackerBase.predict_all()`,
    which calls the `predict()` method once per step size.

    /!/ This is not a price-distribution; it is a distribution over 
    incremental returns between consecutive steps /!/
    """
    def __init__(self):
        super().__init__()

    def predict(self, asset: str, horizon: int, step: int):
        """
        Produce a sequence of incremental return distributions
        for a single (asset, horizon, step) configuration.

        This method is called automatically by `TrackerBase.predict_all()`
        for each step resolution requested by the game.
        """

        # Retrieve recent historical prices (up to 30 days) sampled at 5-minute resolution
        resolution = 5 * 60 # Use a multiple of 60s because the original data has 60s resolution
        price_points = self.prices.get_prices(asset, days=3, resolution=resolution)
        if not price_points:
            return []

        past_times, past_prices = zip(*price_points)

        # Latest observed price (Use it to handle time-dependent logic (e.g. market hours, weekends))
        current_price = past_prices[-1]
        current_time = datetime.fromtimestamp(past_times[-1], tz=timezone.utc)

        # Compute historical incremental returns (price differences)
        returns = np.diff(past_prices)

        # Estimate drift (mean return) and volatility (std dev of returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))

        if sigma <= 0:
            return []

        num_segments = horizon // step

        # Construct one predictive distribution per future time step.
        # Each distribution models the incremental return over a `step`-second interval.
        #
        # IMPORTANT:
        # - The returned objects must strictly follow the `density_pdf` specification.
        # - Each entry corresponds to the return between t + (k−1)·step and t + k·step.
        #
        # We use a single-component Gaussian mixture for simplicity:
        #   r_{t,k} ~ N( (step / 300) · μ , sqrt(step / 300) · σ )
        #
        # where μ and σ are estimated from historical 5-minute returns.
        distributions = []
        for k in range(1, num_segments + 1):
            distributions.append({
                "step": k * step,                      # Time offset (in seconds) from forecast origin
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "builtin",             # Note: use 'builtin' distributions instead of 'scipy' for speed
                        "name": "norm",  
                        "params": {
                            "loc": (step/resolution) * mu, 
                            "scale": np.sqrt(step/resolution) * sigma
                            }
                    },
                    "weight": 1                        # Mixture weight — multiple densities with different weights can be combined
                                                       # total components capped for runtime safety to constants.MAX_DISTRIBUTION_COMPONENTS
                }]
            })

        return distributions


if __name__ == "__main__":

    from crunch_synth import (
        FORECAST_PROFILES,
        SUPPORTED_ASSETS,
        pricedb,
        load_test_prices_once,
        load_initial_price_histories_once,
        count_evaluations,
        plot_quarantine,
        plot_scores,
    )

    # Setup tracker + evaluator
    tracker = GaussianStepTracker()
    tracker_evaluator = TrackerEvaluator(tracker)

    # For each asset and historical timestamp, generate density forecasts
    # over a fixed forecast horizon (e.g. 24h or 1h) at multiple temporal
    # resolutions and evaluate them against realized outcomes.

    # Assets to evaluate
    assets = ["BTC", "SOL"] # Supported assets: "BTC", "SOL", "ETH", "XAUT", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"
    print("Supported assets:", ", ".join(SUPPORTED_ASSETS))
    print("Selected assets:", ", ".join(assets))

    # Select which forecast profile to evaluate
    ACTIVE_HORIZON = "24h"  # options: "24h", "1h"

    HORIZON = FORECAST_PROFILES[ACTIVE_HORIZON]["horizon"]
    STEPS = FORECAST_PROFILES[ACTIVE_HORIZON]["steps"]
    INTERVAL = FORECAST_PROFILES[ACTIVE_HORIZON]["interval"]

    # End timestamp for the test data
    # evaluation_end: datetime = datetime.now(timezone.utc)
    evaluation_end: datetime = datetime(2025, 11, 15, 00, 00, 00, tzinfo=timezone.utc)

    # Number of days of test data to load
    # Note: the last `horizon` seconds of the time series will not be scored
    days = 5

    # Number of days of historical data used as warm-up before evaluation.
    # This history is used only to initialize the tracker and is not scored.
    days_history = 30

    ## Load the last N days of price data (test period)
    test_asset_prices = load_test_prices_once(
        assets, evaluation_end, days=days
    )
    # test_asset_prices : dict : {asset -> [(timestamp, price), ...]} used for evaluation.

    ## Provide the tracker with initial historical data (for the first tick):
    ## load prices from the last H days up to N days ago
    initial_histories = load_initial_price_histories_once(
        assets, evaluation_end, days_history=days_history, days_offset=days
    )
    # initial_histories : dict : {asset -> [(timestamp, price), ...]} used as warm-up history.

    # Run live simulation on historic data
    show_first_plot = True

    for asset, history_price in test_asset_prices.items():

        # First tick: initialize the full historical data (prices before test prices)
        # This initializes the tracker state before evaluation begins.
        tracker_evaluator.tick({asset: initial_histories[asset]})

        prev_ts = 0
        predict_count = 0
        pbar = tqdm(desc=f"Evaluating {asset}", total=count_evaluations(history_price, HORIZON, INTERVAL), unit="eval")
        for ts, price in history_price:
            # Feed the new test price tick
            tracker_evaluator.tick({asset: [(ts, price)]})

            # Trigger a prediction round at the configured interval (ts is in second)
            if ts - prev_ts >= INTERVAL:
                prev_ts = ts
                predictions_evaluated = tracker_evaluator.predict(asset, HORIZON, STEPS)

                # Quarantine mechanism:
                # - Predictions are not scored immediately. Each prediction is placed in a quarantine 
                #   until sufficient future price data (up to the full horizon ticks) becomes available.
                # - Predictions issued within the final `horizon` seconds of the
                #   time series cannot be scored, as future observations are unavailable.

                if predictions_evaluated:
                    pbar.update(1)

                # Periodically display results
                if predictions_evaluated and predict_count % 10 == 0:

                    if show_first_plot:
                        ## Return forecast mapped into price space
                        plot_quarantine(asset, predictions_evaluated[0], step=STEPS[0], prices=tracker_evaluator.tracker.prices, mode="incremental", lookback_seconds=HORIZON/4)
                        ## density forecast over returns
                        plot_quarantine(asset, predictions_evaluated[0], step=STEPS[0], prices=tracker_evaluator.tracker.prices, mode="direct")
                        show_first_plot = False

                    pbar.write(
                        f"[{asset}] avg norm CRPS={tracker_evaluator.overall_score_asset(asset):.4f} | "
                        f"recent={tracker_evaluator.recent_score_asset(asset):.4f}"
                    )
                predict_count += 1
        
        # Final summary for this asset
        pbar.write(
                f"[{asset}] avg norm CRPS={tracker_evaluator.overall_score_asset(asset):.4f} | "
                f"recent={tracker_evaluator.recent_score_asset(asset):.4f}"
            )
        
        pbar.close()
        print()

    # Global summary across all assets
    tracker_name = tracker_evaluator.tracker.__class__.__name__
    print(f"\nTracker {tracker_name}:"
        f"\nFinal average normalized crps score: {tracker_evaluator.overall_score():.4f}")

    # Plot scoring timeline
    timestamped_scores = tracker_evaluator.scores
    print("\n(Note - Scores appear after quarantine: a score at time t evaluates a forecast issued at (t - horizon))")
    plot_scores(timestamped_scores)
