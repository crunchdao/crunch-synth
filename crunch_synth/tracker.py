import abc
import datetime
import logging
import threading
import time
from typing import Union, List

from crunch_synth.prices import PriceStore, Asset, PriceEntry, PriceData
from crunch_synth.utils.distributions import validate_distribution, round_distribution_digits

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
    def __init__(self, data_clock: datetime.datetime = datetime.datetime.now(), async_mode: bool = True):
        """
        :param async_mode: If True, run in a background thread. If False, run synchronously.
        """
        self.prices = PriceStore()
        self.data_clock = int(datetime.datetime.timestamp(data_clock))  # current data timestamp by default, override for backtesting
        self.async_mode = async_mode
        self._tick_event = threading.Event()
        self._routes: list[tuple[int, str, SubTracker]] = []
        self._sub_trackers: set[SubTracker] = set()
        self._cron_threads: list[threading.Thread] = []
        self._sync_steps: list[callable] = []

    def track(self, horizon: int, asset: Union[str, List[str]], tracker: SubTracker):
        """
        Register a SubTracker for a given (horizon, asset) pair.

        :param horizon: Prediction horizon in seconds (e.g. 86400 for 24h).
        :param asset: Asset symbol (e.g. "BTC"), "*" for all assets, or a list of assets.
        :param tracker: A SubTracker instance to handle predictions.
        """
        # Normalize asset to a list
        assets = [asset] if isinstance(asset, str) else asset

        for asset in assets:
            self._routes.append((horizon, asset, tracker))

        self._sub_trackers.add(tracker)

    def schedule(self, name: str, func, interval: datetime.timedelta, immediate: bool = False):
        """
        Schedule a recurring background task driven by data time (self.data_clock).

        The thread polls self.data_clock every second and triggers func() when
        enough data-time has elapsed. Works correctly in both live and
        backtesting modes.

        :param name: Human-readable name for logging.
        :param func: Callable to execute periodically.
        :param interval: Time between executions (in data-time).
        :param immediate: If True, run func() immediately on first check.
        """
        interval_sec = int(interval.total_seconds())
        last_run = 0 if immediate else self.data_clock

        def _step():
            nonlocal last_run
            now = self.data_clock
            if now - last_run >= interval_sec:
                start = time.monotonic()
                try:
                    logger.info("[schedule] '%s' running (data-time %s)", name,
                                datetime.datetime.fromtimestamp(now, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
                    func()
                    elapsed_s = time.monotonic() - start
                    last_run = now
                    next_at = now + interval_sec
                    logger.info("[schedule] '%s' done in %.1fs — next at data-time %s", name, elapsed_s,
                                datetime.datetime.fromtimestamp(next_at, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
                except Exception:
                    last_run = now
                    logger.exception("[schedule] '%s' failed", name)

        if not self.async_mode:
            if immediate:
                _step()
            self._sync_steps.append(_step)
            return

        # Async mode (default): background thread
        def _loop():
            while True:
                self._tick_event.wait()
                self._tick_event.clear()
                _step()

        t = threading.Thread(target=_loop, name=f"schedule-{name}", daemon=True)
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

        self._sync_clock()

    def _sync_clock(self, timestamp: int | None = None):
        """Update data_clock and wake up schedule threads. Call this after ingesting data."""
        if timestamp is not None:
            self.data_clock = timestamp
        elif self.prices.last_timestamp is not None:
            self.data_clock = self.prices.last_timestamp

        if self.async_mode:
            self._tick_event.set()
        else:
            for step in self._sync_steps:
                step()

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


