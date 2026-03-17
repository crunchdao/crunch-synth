import abc
import datetime
import logging
import threading
import time

from condorgame.prices import PriceStore, Asset, PriceEntry, PriceData

logger = logging.getLogger(__name__)


class SubTracker(abc.ABC):
    """
    Specialized tracker for a specific (horizon, asset) combination.

    """

    def __init__(self):
        self.prices = PriceStore()

    def tick(self, data: PriceData):
        """Override to filter which assets are ingested. Default: ingest all."""
        self.prices.add_bulk(data)

    @abc.abstractmethod
    def predict(self, asset: Asset, horizon: int, step: int) -> list:
        """Return a list of density predictions."""
        pass


class TrackerBase(abc.ABC):
    """
    Base class for all trackers.

    You can either:
    - Override `predict()` directly for a single strategy, or
    - Use `track(horizon, asset, sub_tracker)` to route predictions
      to specialized SubTracker instances.

    The framework will automatically call `predict()` multiple times
    via `predict_all()` to obtain multi-resolution forecasts.
    """

    def __init__(self):
        self.prices = PriceStore()
        self._routes: list[tuple[int, str, SubTracker]] = []
        self._sub_trackers: set[SubTracker] = set()
        self._cron_threads: list[threading.Thread] = []

    def track(self, horizon: int, asset: str, tracker: SubTracker):
        """
        Register a SubTracker for a given (horizon, asset) pair.

        :param horizon: Prediction horizon in seconds (e.g. 86400 for 24h).
        :param asset: Asset symbol (e.g. "BTC") or "*" for all assets.
        :param tracker: A SubTracker instance to handle predictions.
        """
        self._routes.append((horizon, asset, tracker))
        self._sub_trackers.add(tracker)

    def schedule(self, name: str, func, interval: datetime.timedelta, immediate: bool = False):
        """
        Schedule a recurring background task.

        :param name: Human-readable name for logging.
        :param func: Callable to execute periodically.
        :param interval: Time between executions.
        :param immediate: If True, run func() once before the first interval.
        """
        interval_sec = interval.total_seconds()

        def _loop():
            if not immediate:
                time.sleep(interval_sec)
            while True:
                try:
                    logger.info("[cron] '%s' running", name)
                    func()
                    next_run = datetime.datetime.now() + interval
                    logger.info("[cron] '%s' done — next run at %s", name, next_run.strftime("%Y-%m-%d %H:%M:%S"))
                except Exception:
                    next_run = datetime.datetime.now() + interval
                    logger.exception("[cron] '%s' failed — next run at %s", name, next_run.strftime("%Y-%m-%d %H:%M:%S"))

        t = threading.Thread(target=_loop, name=f"cron-{name}", daemon=True)
        t.start()
        self._cron_threads.append(t)

    def tick(self, data: PriceData):
        """
        The first tick is the initial state and send you the last 30 days of data.
        The resolution of the data is 1 minute.

        data = {
            "BTC": [(ts1, p1), (ts2, p2)],
            "SOL": [(ts1, p1)],
        }

        The tick() method is called whenever new market data arrives:
        When it's called:
        - Typically every minute or when new data is available
        - Before any prediction request
        - Can be called multiple times before a predict
        """
        self.prices.add_bulk(data)
        for tracker in self._sub_trackers:
            tracker.tick(data)

    def predict(self, asset: Asset, horizon: int, step: int):
        """
        Generate a sequence of return price density predictions for a given asset.

        If a SubTracker is registered via `track()` for this (horizon, asset),
        the call is routed to it. Otherwise, subclasses must override this method.

        :param asset: Asset symbol to predict (e.g. "BTC", "SOL").
        :param horizon: Total prediction horizon in seconds (e.g. 86400 for 24h ahead).
        :param step: Interval between each prediction in seconds (e.g. 300 for 5 minutes).
        :return: List of predictive density objects.
        """
        tracker = self._resolve_tracker(horizon, asset)
        if tracker is not None:
            return tracker.predict(asset, horizon, step)

        raise NotImplementedError(
            f"{self.__class__.__name__}.predict() not implemented "
            f"and no SubTracker registered for horizon={horizon}, asset='{asset}'. "
            f"Either override predict() or use self.track() to register a SubTracker."
        )

    def _resolve_tracker(self, horizon: int, asset: str) -> SubTracker | None:
        """Find the best matching SubTracker: exact asset match > wildcard '*'."""
        wildcard_match = None
        for h, pattern, tracker in self._routes:
            if h != horizon:
                continue
            if pattern == asset:
                return tracker
            if pattern == '*' and wildcard_match is None:
                wildcard_match = tracker
        return wildcard_match

    def predict_all(self, asset: Asset, horizon: int, steps: list[int]):
        """
        Generate predictive distributions at multiple time resolutions
        for a fixed prediction horizon.

        Returns:
            dict[str, list[dict]]:
                {
                    300:   [...],
                    3600:  [...],
                    21600:  [...],
                    86400: [...]
                }
        """
        predictions = {}

        for step in steps:
            if step > horizon:
                continue

            predictions[step] = self.predict(asset=asset, horizon=horizon, step=step)

        return predictions
