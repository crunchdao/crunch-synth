# test.py
from datetime import datetime, timezone, timedelta
import numpy as np

from crunch_synth.tracker import PriceData
from crunch_synth.examples.exampletracker import GaussianStepTracker  # replace with actual import path


# -------------------------------
# Helper: generate synthetic price data
# -------------------------------
def generate_synthetic_prices(start_ts: int, n: int, step_sec: int, start_price: float = 100.0):
    """
    Generate synthetic price series using a simple random walk.

    Returns:
        List of (timestamp, price) tuples
    """
    prices = []
    price = start_price
    ts = start_ts
    for _ in range(n):
        # Simple log-normal-like step
        price *= np.exp(np.random.normal(0, 0.01))
        prices.append((ts, float(price)))
        ts += step_sec
    return prices


# -------------------------------
# Test function
# -------------------------------
def test_gaussian_step_tracker():
    # Initialize tracker
    tracker = GaussianStepTracker()

    # Generate 5 days of synthetic 5-min data
    start_ts = int(datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp())
    step_sec = 300  # 5 min
    n_points = 5 * 24 * 12  # 5 days of 5-minute points
    synthetic_prices = generate_synthetic_prices(start_ts, n_points, step_sec)

    # Feed historical data
    tracker.tick({"FAKE": synthetic_prices})

    # Make a prediction for next 24h at 5-min steps
    horizon = 86400  # 24h
    predictions = tracker.predict("FAKE", horizon, step_sec)

    # Print basic sanity info
    print(f"Number of prediction steps: {len(predictions)}")
    if predictions:
        print("First prediction step:", predictions[0])
        print("Last prediction step:", predictions[-1])

    # Check that each prediction is a Gaussian mixture
    for pred in predictions[:3]:  # just check first 3 steps
        assert pred["type"] == "mixture"
        assert "components" in pred
        component = pred["components"][0]
        assert component["weight"] == 1
        assert component["density"]["name"] == "norm"

    print("Sanity check passed: GaussianStepTracker predictions valid.")


# -------------------------------
# Run test
# -------------------------------
if __name__ == "__main__":
    test_gaussian_step_tracker()
