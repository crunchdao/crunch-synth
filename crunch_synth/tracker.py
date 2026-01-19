import abc

from crunch_synth.prices import PriceStore, Asset, PriceEntry, PriceData
from crunch_synth.utils.distributions import validate_distribution, round_distribution_digits


class TrackerBase(abc.ABC):
    """
    Base class for all trackers.

    A tracker is a model that processes real-time asset prices and outputs 
    probabilistic forecasts of future returns.

    You must implement `predict()` for a *single*
    (asset, horizon, step) configuration.

    The framework will automatically call `predict()` multiple times
    via `predict_all()` to obtain multi-resolution forecasts.
    """
    def __init__(self):
        """
        Initialize the tracker with a `PriceStore`.

        `self.prices` provides:
        - Storage of recent historical prices per asset (rolling window of 30 days)
        - Convenient accessors:
            - get_last_price(asset)
            - get_prices(asset, days, resolution)
            - get_closest_price(asset, timestamp)
        """
        self.prices = PriceStore()

    def tick(self, data: PriceData):
        """
        The framework automatically updates the `PriceStore` by calling `tick()`

        data : dict[Asset, list[PriceEntry]]
            Example:
            {
                "BTC": [(ts1, p1), (ts2, p2)],
                "SOL": [(ts1, p1)],
            }

        The tick() method is called whenever new market data arrives:
        When it's called:
        - To store historical data
        - Typically every minute or when new data is available
        - Before any prediction request
        - Can be called multiple times before a predict
        """
        self.prices.add_bulk(data)

    @abc.abstractmethod
    def predict(self, asset: Asset, horizon: int, step: int) -> list[dict]:
        """
        Generate a sequence of return price density predictions for a given asset.

        This method produces a list of predictive distributions (densities)
        for the future return price of a given asset (e.g., BTC, SOL, etc.)
        starting from the current timestamp.

        Each distribution corresponds to a prediction at a specific time offset,
        spaced by `step` seconds, up to the total prediction horizon `horizon`.

        The returned list is directly compatible with the `density_pdf` library.

        Example:
            >>> model.predict(asset="SOL", horizon=86400, step=300)
            [
                {
                    "step": (k+1)*step,
                    "prediction": {
                        "type": "builtin",
                        "name": "norm",
                        "params": {"loc": -0.01, "scale": 0.4}
                    }
                }
                for k in range(0, horizon // step)
            ]

        :param asset: Asset symbol to predict (e.g. "BTC", "SOL").
        :param horizon: Prediction horizon in seconds (e.g. 86400 for 24h ahead).
        :param step: Interval between each prediction in seconds (e.g. 300 for 5 minutes).
        :return: List of predictive density objects, each representing a probability
                 distribution for the return price at a given time step.
        """
        pass

    def predict_all(self, asset: Asset, horizon: int, steps: list[int]) -> dict[int, list[dict]]:
        """
        Generate predictive distributions at multiple time resolutions
        for a fixed prediction horizon.

        :param asset: Asset symbol to predict (e.g. "BTC", "SOL").
        :param horizon: Prediction horizon in seconds (e.g. 86400 for 24h ahead).
        :param steps: List of step sizes (in seconds) at which to generate predictions.

        :return predictions: dict[int, list[dict]]
            Mapping from step size to the list of density predictions.

        Example:
            >>> model.predict_all(asset="SOL", horizon=86400, steps=[300, 3600, 21600, 86400])
            {
                300:   [...],
                3600:  [...],
                21600:  [...],
                86400: [...]
            }
        """
        all_predictions = {}

        for step in steps:
            if step > horizon:
                continue

            predictions = self.predict(asset=asset, horizon=horizon, step=step)
            if not predictions:
                all_predictions[step] = []
                continue

            predictions_ready = []
            for dist in predictions:
                # Validate mixture distribution: enforce MAX_DISTRIBUTION_COMPONENTS recursively
                validate_distribution(dist)
                # Round all numeric parameters to a fixed number of significant digits
                dist = round_distribution_digits(dist, digits=6)
                predictions_ready.append(dist)

            all_predictions[step] = predictions_ready

        return all_predictions


