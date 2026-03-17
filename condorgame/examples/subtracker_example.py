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

import datetime
import json
import logging
import os

import numpy as np

from condorgame.tracker import TrackerBase, SubTracker
from condorgame.prices import Asset, PriceData
from condorgame.price_provider import shared_pricedb

logger = logging.getLogger(__name__)

RESOURCES_DIR = "/workspace/submission/code/resources"


class CryptoTracker(SubTracker):
    """SubTracker that trains a model in background via cron."""

    MODEL_PATH = os.path.join(RESOURCES_DIR, "crypto_model.json")
    ASSETS = {"BTC", "ETH", "SOL"}

    def __init__(self):
        super().__init__()
        self._model = self._load_model()

    def tick(self, data: PriceData):
        """Only ingest crypto assets."""
        filtered = {asset: prices for asset, prices in data.items() if asset in self.ASSETS}
        if filtered:
            self.prices.add_bulk(filtered)

    def _load_model(self) -> dict | None:
        """Load model from disk if it exists."""
        if not os.path.exists(self.MODEL_PATH):
            return None
        try:
            with open(self.MODEL_PATH) as f:
                model = json.load(f)
            logger.info("Loaded model from %s", self.MODEL_PATH)
            return model
        except Exception:
            logger.exception("Failed to load model from %s", self.MODEL_PATH)
            return None

    def _save_model(self, model: dict):
        """Persist model to disk."""
        os.makedirs(os.path.dirname(self.MODEL_PATH), exist_ok=True)
        with open(self.MODEL_PATH, "w") as f:
            json.dump(model, f)
        logger.info("Saved model to %s", self.MODEL_PATH)

    def train(self):
        """
        Called by cron every N days in a background thread.
        Builds a new model from scratch, saves it, then swaps it in atomically.
        """
        now = datetime.datetime.now(datetime.timezone.utc)
        prices = shared_pricedb.get_price_history(
            asset="GMCI30",
            from_=now - datetime.timedelta(days=30),
            to=now,
        )
        if not prices:
            return

        _, values = zip(*prices)
        returns = np.diff(values)

        new_model = {
            "mu": float(np.mean(returns)),
            "sigma": float(np.std(returns)),
        }

        # Save to disk so next restart picks it up
        self._save_model(new_model)

        # Atomic swap — predict() always sees a consistent model
        self._model = new_model

    def predict(self, asset: Asset, horizon: int, step: int) -> list:
        model = self._model  # snapshot the reference once

        if model is None:
            # No model on disk and not trained yet — fall back to a simple prior
            model = {"mu": 0.0, "sigma": 1.0}

        num_segments = horizon // step
        return [
            {
                "step": k * step,
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "builtin",
                        "name": "norm",
                        "params": {
                            "loc": model["mu"],
                            "scale": model["sigma"],
                        }
                    },
                    "weight": 1,
                }]
            }
            for k in range(1, num_segments + 1)
        ]


HOUR = 3600


class MyTracker(TrackerBase):
    def __init__(self):
        super().__init__()

        crypto = CryptoTracker()

        # Route predictions to the sub-tracker
        self.track(24 * HOUR, '*', crypto)  # 24h horizon, all assets
        self.track(1 * HOUR, '*', crypto)   # 1h horizon, all assets

        # Train every 5 days in background — first run after 5 days
        self.schedule('Crypto train', crypto.train, datetime.timedelta(days=5))
