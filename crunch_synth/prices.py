from __future__ import annotations

from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import TypeAlias

from sortedcontainers import SortedDict

PriceEntry: TypeAlias = tuple[int, float]
Asset: TypeAlias = str
PriceData: TypeAlias = dict[Asset, list[PriceEntry]]


class PriceStore:
    """
    PriceStore caches prices for multiple assets and allows fast updates / queries.
    Uses sortedcontainers.SortedDict: timestamp -> price (kept sorted, updates overwrite).
    """

    def __init__(self, window_days: int = 30):
        # per asset: SortedDict[int, float]
        self.data: dict[Asset, SortedDict[int, float]] = defaultdict(SortedDict)
        # Maximum number of days of historical data to keep per asset.
        self.window_days = window_days

    def add_price(self, symbol: Asset, price: float, timestamp: int):
        """Add a single (timestamp, price) entry for an asset."""
        self.add_prices(symbol, [(timestamp, price)])

    def add_prices(self, symbol: Asset, entries: list[PriceEntry]):
        """Add multiple (timestamp, price) pairs for a single asset (insert/update)."""
        if not entries:
            return

        series = self.data[symbol]

        # Update/insert (if same ts exists, it's overwritten)
        # (If entries may contain duplicates, last one wins naturally.)
        for t, p in entries:
            series[t] = p

        # Keep only last window_days relative to newest timestamp
        if series:
            last_ts = series.peekitem(-1)[0]
            cutoff = int(
                (datetime.fromtimestamp(last_ts, tz=timezone.utc) - timedelta(days=self.window_days)).timestamp()
            )

            i = series.bisect_left(cutoff)
            if i:
                del series.iloc[:i]

    def add_bulk(self, data: PriceData):
        """
        Add prices for multiple assets at once.
        Example:
          data = {
            "BTC": [(ts1, p1), (ts2, p2)],
            "ETH": [(ts1, p1)],
          }
        """
        for symbol, entries in data.items():
            self.add_prices(symbol, entries)

    def get_prices(self, asset: str, days: int | None = None, resolution: int = 60) -> list[PriceEntry]:
        """
        Quickly retrieve (timestamp, price) pairs spaced by `resolution` seconds.

        days : int | None
            Limit to the last `days` days of data. None returns all available data.
        resolution : int
            Minimum time difference between consecutive returned points in seconds.
        """
        series = self.data.get(asset)
        if not series:
            return []

        if days is not None:
            last_ts = series.peekitem(-1)[0]
            cutoff = int((datetime.fromtimestamp(last_ts, tz=timezone.utc) - timedelta(days=days)).timestamp())
            it = series.irange(minimum=cutoff, inclusive=(True, True))
        else:
            it = iter(series.keys())

        result: list[PriceEntry] = []
        target_next: int | None = None

        for t in it:
            if target_next is None:
                result.append((t, series[t]))
                target_next = t + resolution
            elif t >= target_next:
                result.append((t, series[t]))
                target_next = t + resolution

        return result

    def get_last_price(self, asset: str) -> PriceEntry | None:
        """Retrieve the last (timestamp, price) pair for a given asset."""
        series = self.data.get(asset)
        if not series:
            return None
        t, p = series.peekitem(-1)
        return t, p

    def get_closest_price(self, asset: str, time: int) -> PriceEntry | None:
        """
        Retrieve the (timestamp, price) pair closest to the given timestamp for a specific asset.
        Returns None if no data is available.
        """
        series = self.data.get(asset)
        if not series:
            return None

        pos = series.bisect_left(time)

        if pos == 0:
            t, p = series.peekitem(0)
            return t, p
        if pos == len(series):
            t, p = series.peekitem(-1)
            return t, p

        t_after = series.iloc[pos]
        t_before = series.iloc[pos - 1]

        if time - t_before <= t_after - time:
            return t_before, series[t_before]
        else:
            return t_after, series[t_after]