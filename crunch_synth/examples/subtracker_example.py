"""
Example: using SubTracker + cron for background training.

Key pattern: train() builds a NEW model, then swaps the reference atomically.
This avoids any concurrency issue between predict() and train() — no lock needed.

    cron thread                         main thread
    ──────────                          ───────────
    train() starts                      predict() reads self._model  ← old model, safe
    new_model = fit(...)                predict() reads self._model  ← still old, safe
    self._model = new_model  ← atomic  predict() reads self._model  ← new model, safe

In Python, assigning a reference (self._model = x) is atomic thanks to the GIL.
As long as train() never mutates the existing model in-place, predict() is always
reading a consistent object.

DON'T do this:
    def train(self):
        self._model.weights = new_weights   # ← mutates in-place, predict() may
        self._model.bias = new_bias          #   see weights from new + bias from old

DO this:
    def train(self):
        new_model = fit_new_model(...)       # ← build entirely new object
        self._model = new_model              # ← atomic swap, predict() sees old or new, never half
"""

from datetime import datetime, timezone, timedelta
import pickle
import logging
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from crunch_synth import TrackerBase, SubTracker, FORECAST_PROFILES, SUPPORTED_ASSETS

logger = logging.getLogger(__name__)

RESOURCES_DIR = "/workspace/submission/code/resources"


class MySubTracker(SubTracker):

    def __init__(self):
        super().__init__()
        self.config = {
            "window_days": 10,
            "window_size": 288,
            "resolution": 300
        }

        self.fitted = False
        self.scales = {}

        self.tracker_assets = SUPPORTED_ASSETS

        # Warm-start: in live mode, load any pre-existing models from the 'resources/' directory.
        # This allows predictions to continue while first training loop isn't finished.
        # Also enables recovery of previously trained models if the online system stops unexpectedly.
        self._load_existing_models()

    def _model_path(self, horizon: int, asset: str) -> str:
        """Return path for a given (horizon, asset) model."""
        os.makedirs("resources", exist_ok=True) # On CrunchDAO platform, you can store files into the directory named "resources/"
        return f"resources/model_{horizon}_{asset}.pkl"

    def _save_model(self, model, horizon: int, asset: str):
        """Persist model to disk."""
        path = self._model_path(horizon, asset)
        with open(path, "wb") as f:
            pickle.dump(model, f)

    def _load_model(self, horizon: int, asset: str):
        """Load model from disk if it exists."""
        path = self._model_path(horizon, asset)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"[Model Load Warning] Failed to load {path}: {e}")
        return None
    
    def _load_existing_models(self):
        """
        Load all available models from disk.

        This enables warm-start:
        - skip initial training if models already exist
        """
        self.models = {}

        # loop per horizon
        for forecast_profile in FORECAST_PROFILES.values():
            horizon = forecast_profile["horizon"]
            self.models[horizon] = {}

            for asset in self.tracker_assets:
                model = self._load_model(horizon, asset)
                if model is not None:
                    print(f"Horizon [{horizon}] Asset [{asset}] Loaded existing models from 'resources/'")
                    self.models[horizon][asset] = model

    def build_feature_dataframe(self, pairs, resolution):
        timestamps, prices = zip(*pairs)
        return pd.DataFrame({
            "timestamp": timestamps,
            "price": prices
        })

    def train_model(self, horizon, asset, histories):
        model = LinearRegression()

        X = []
        y = []

        prices = histories[asset]
        prices = np.array(prices)

        returns = np.diff(prices)

        window = self.config["window_size"]
        future = 5  # small future window for target

        for i in range(window, len(returns) - future):
            past = returns[i - window:i]

            # simple features
            feat_mean = np.mean(past)
            feat_std = np.std(past)

            X.append([feat_mean, feat_std])

            # target = future volatility
            future_returns = returns[i:i + future]
            y.append(np.log(np.std(future_returns) + 1e-8))

        if len(X) == 0:
            raise ValueError("Not enough data to train")

        X = np.array(X)
        y = np.array(y)

        model.fit(X, y)
        return model

    def train(self):
        # Train one model per (horizon, asset)
        # Models are stored as `self.models[horizon][asset]`

        new_models = {
            forecast_profile["horizon"]: {} for forecast_profile in FORECAST_PROFILES.values()  # get all supported horizon
        }

        # Use rolling window data for both primary
        initial_histories = {asset: self.prices.get_prices(asset, days=self.config["window_days"], resolution=self.config["resolution"]) for asset in self.tracker_assets}

        for horizon in new_models.keys():
            print("\n-----------------------------------------------------")
            print(f"Training tracker for horizon {horizon}")
            print("-----------------------------------------------------")

            for asset in self.tracker_assets:
                try:
                    model = self.train_model(horizon, asset, initial_histories)

                    new_models[horizon][asset] = model
                    # Save
                    self._save_model(model, horizon, asset)
                except Exception as e:
                    print(f"Error: {e}")
                    # If training fails, fallback to previous model (if available)
                    if hasattr(self, 'models'):
                        new_models[horizon][asset] = self.models[horizon][asset]

        self.models = new_models
        self.fitted = True

    def predict(self, asset: str, horizon: int, step: int):
        try:
            # Build feature vector from:
            # - recent asset returns
            # - aligned exogenous returns

            # Retrieve recent historical prices
            pairs = self.prices.get_prices(asset, days=5, resolution=self.config["resolution"])

            if not pairs:
                return []

            df = self.build_feature_dataframe(pairs, self.config["resolution"])

            prices = df["price"].values
            timestamps = df["timestamp"].values

            returns = np.diff(prices)

            past = returns[-self.config["window_size"]:]
            current_time = pd.Timestamp(timestamps[-1])

            feat_mean = np.mean(past)
            feat_std = np.std(past)

            feats = [feat_mean, feat_std]

            X = np.array([feats])

            if asset not in self.models[horizon]:
                raise ValueError(f"Missing model horizon {horizon}, asset {asset}")

            # Predict volatility scale using trained model
            log_scale = float(self.models[horizon][asset].predict(X)[0])
            scale = np.exp(log_scale)

            # Save
            self.scales[asset] = scale

        except Exception as e:
            print(
                f"[Tracker Warning] Feature computation or model prediction "
                f"failed for asset={asset}, horizon={horizon}. "
                f"Falling back to historical volatility. Error: {e}"
            )

            # Retrieve recent historical prices (up to 30 days)
            pairs = self.prices.get_prices(asset, days=10, resolution=300)
            _, past_prices = zip(*pairs)

            # Compute historical incremental returns (price differences)
            returns = np.diff(past_prices)

            # Estimate volatility (std dev of returns)
            scale = float(np.std(returns))

        num_segments = horizon // step

        # Convert predicted scale into step-wise Gaussian distributions
        # using Brownian scaling: σ_step = √(step / resolution) × scale
        distributions = []
        for k in range(1, num_segments + 1):
            distributions.append({
                "step": k * step,  # Time offset (in seconds) from forecast origin
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "builtin",  # Note: use 'builtin' distributions instead of 'scipy' for speed
                        "name": "norm",
                        "params": {
                            "loc": 0.0,  # Assume zero drift
                            "scale": np.sqrt(step / self.config["resolution"]) * scale
                        }
                    },
                    "weight": 1  # Mixture weight — multiple densities with different weights can be combined
                    # total components capped for runtime safety to constants.MAX_DISTRIBUTION_COMPONENTS
                }]
            })

        return distributions


HOUR = 3600


class MyTracker(TrackerBase):
    def __init__(self):
        super().__init__()

        self.subtracker = MySubTracker()

        # Route predictions to the sub-tracker
        self.track(24 * HOUR, '*', self.subtracker)  # 24h horizon, all assets
        self.track(1 * HOUR, '*', self.subtracker)   # 1h horizon, all assets

        # Train every 5 days in background — first run after 5 days
        self.schedule('subtracker train', self.subtracker.train, timedelta(days=5), immediate=True)