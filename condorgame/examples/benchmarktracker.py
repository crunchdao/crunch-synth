from datetime import datetime, timezone, timedelta
import numpy as np

from condorgame.price_provider import shared_pricedb
from condorgame.tracker import TrackerBase
from condorgame.tracker_evaluator import TrackerEvaluator


class GaussianStepTracker(TrackerBase):
    """
    A benchmark tracker that models *future log-returns* as Gaussian-distributed.

    For each forecast step k, the tracker returns a normal distribution
    N(mu, sigma) where:
        - mu    = mean historical log-return
        - sigma = std historical log-return

    This is NOT a price-distribution; it is a distribution over log-returns
    between consecutive steps.
    """
    def __init__(self):
        super().__init__()

    # The `tick` method is inherited from TrackerBase.
    # override it if your tracker requires custom update logic

    def predict(self, asset: str, horizon: int, step: int):

        # Retrieve past prices with sampling resolution equal to the prediction step.
        pairs = self.prices.get_prices(asset, days=5, resolution=step)
        if not pairs:
            return []

        _, past_prices = zip(*pairs)

        if len(past_prices) < 3:
            return []

        # Compute historical log-returns
        log_prices = np.log(past_prices)
        returns = np.diff(log_prices)

        # Estimate drift (mean log-return) and volatility (std dev of log-returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))

        if sigma <= 0:
            return []

        num_segments = horizon // step

        # Produce one Gaussian for each future time step
        # The returned list must be compatible with the `density_pdf` library.
        distributions = []
        for k in range(1, num_segments + 1):
            distributions.append({
                "step": k * step,
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "builtin",
                        "name": "norm",  
                        "params": {"loc": mu, "scale": sigma}
                    },
                    "weight": 1
                }]
            })

        return distributions


if __name__ == "__main__":
    from condorgame.debug.plots import plot_quarantine, plot_prices, plot_log_return_prices, plot_scores
    from condorgame.examples.utils import load_test_prices_once, load_initial_price_histories_once

    # Setup tracker + evaluator
    tracker_evaluator = TrackerEvaluator(GaussianStepTracker())

    # For each asset and historical timestamp, compute a 24-hour density forecast 
    # at 5-minute intervals and evaluate the tracker against actual outcomes.
    assets = ["SOL", "BTC"]
    
    # Prediction horizon = 24h (in seconds)
    HORIZON = 86400
    # Prediction step = 5 minutes (in seconds)
    STEP = 300
    # How often we evaluate the tracker (in seconds)
    INTERVAL = 3600

    # End timestamp for the test data
    evaluation_end: datetime = datetime.now(timezone.utc)

    # Number of days of test data to load
    days = 30
    # Amount of warm-up history to load
    days_history = 30

    ## Load the last N days of price data (test period)
    test_asset_prices = load_test_prices_once(
        assets, shared_pricedb, evaluation_end, days=days
    )

    ## Provide the tracker with initial historical data (for the first tick):
    ## load prices from the last H days up to N days ago
    initial_histories = load_initial_price_histories_once(
        assets, shared_pricedb, evaluation_end, days_history=days_history, days_offset=days
    )

    # Run live simulation on historic data
    show_first_plot = True

    for asset, history_price in test_asset_prices.items():

        # First tick: initialize historical data
        tracker_evaluator.tick({asset: initial_histories[asset]})

        prev_ts = 0
        predict_count = 0
        for ts, price in history_price:

            # Feed the new tick
            tracker_evaluator.tick({asset: [(ts, price)]})

            # Evaluate prediction every hour (ts is in second)
            if ts - prev_ts >= INTERVAL:
                prev_ts = ts
                predictions_evaluated = tracker_evaluator.predict(asset, HORIZON, STEP)

                # Periodically display results
                if predictions_evaluated and predict_count % 300 == 0:
                    if show_first_plot:
                        ## log-return forecast mapped into price space
                        plot_quarantine(asset, predictions_evaluated[0], tracker_evaluator.tracker.prices, mode="incremental")
                        ## density forecast over log-returns
                        plot_quarantine(asset, predictions_evaluated[0], tracker_evaluator.tracker.prices, mode="direct")
                        show_first_plot = False
                    print(f"My average likelihood score {asset}: {tracker_evaluator.overall_likelihood_score_asset(asset):.4f}")
                    print(f"My recent average likelihood score {asset}: {tracker_evaluator.recent_likelihood_score_asset(asset):.4f}")
                predict_count += 1

    tracker_name = tracker_evaluator.tracker.__class__.__name__
    print(f"\nTracker {tracker_name}:"
        f"\nFinal average likelihood score: {tracker_evaluator.overall_likelihood_score():.4f}")
    
    # Plot scoring timeline
    timestamped_scores = tracker_evaluator.scores
    plot_scores(timestamped_scores)
