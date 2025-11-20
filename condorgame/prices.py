from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from collections import defaultdict
from typing import TypeAlias

PriceEntry: TypeAlias = tuple[int, float]
Asset: TypeAlias = str
PriceData: TypeAlias = dict[Asset, list[PriceEntry]]

class PriceStore:
    """
    PriceStore caches prices for multiple assets,
    allows you to access and retrieve these prices, and update them.
    Everything is designed to maintain performance and provide convenient functions.
    """

    def __init__(self, window_days: int = 30):
        self.data = defaultdict(lambda: {"ts": [], "price": []})
        self.window_days = window_days

    def add_price(self, symbol: Asset, price: float, timestamp: int):
        self.add_prices(symbol, [(timestamp, price)])

    def add_prices(self, symbol: Asset, entries: list[PriceEntry]):
        """Add multiple (timestamp, price) pairs for a single asset."""
        if not entries:
            return

        d = self.data[symbol]
        ts_new, price_new = zip(*entries)

        # Case 1: All new data is strictly after the last known timestamp → fast path
        if not d["ts"] or ts_new[0] > d["ts"][-1]:
            d["ts"].extend(ts_new)
            d["price"].extend(price_new)
        else:
            # Case 2: Potential overlap → append only strictly newer points, skip duplicates
            last_ts = d["ts"][-1] if d["ts"] else None
            for t, p in zip(ts_new, price_new):
                if last_ts is None or t > last_ts:
                    d["ts"].append(t)
                    d["price"].append(p)
                    last_ts = t
                elif t == last_ts:
                    # Update existing last value instead of adding a duplicate
                    d["price"][-1] = p

        # Drop old points
        if len(d["ts"]) > 0:
            cutoff = int((datetime.fromtimestamp(d["ts"][0], tz=timezone.utc) - timedelta(days=5)).timestamp())
            i = bisect_left(d["ts"], cutoff)
            if i > 0:
                d["ts"] = d["ts"][i:]
                d["price"] = d["price"][i:]

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

    def get_prices(self, asset: str, days: int | None = None, resolution: int = 60):
        """
        Quickly retrieve (timestamp, price) pairs spaced by `resolution` seconds.
        Stored data has a 60-second granularity.
        """
        d = self.data.get(asset)
        if not d:
            return []

        ts = d["ts"]
        prices = d["price"]
        if not ts:
            return []

        # Precompute cutoff timestamp if needed
        if days:
            cutoff = int((datetime.fromtimestamp(ts[-1], tz=timezone.utc) - timedelta(days=days)).timestamp())
            start_idx = bisect_left(ts, cutoff)
        else:
            start_idx = 0

        n = len(ts)
        if start_idx >= n:
            return []

        result = []
        last_t = ts[start_idx]

        result.append((last_t, prices[start_idx]))
        target_next = last_t + resolution

        for i in range(start_idx + 1, n):
            t = ts[i]
            if t >= target_next:
                result.append((t, prices[i]))
                target_next = t + resolution  # Maintain spacing

        return result

    def get_last_price(self, asset: str) -> tuple[int, float] | None:
        """
        Retrieve the last (timestamp, price) pair for a given asset.
        Returns None if no data is available.
        """
        d = self.data.get(asset)
        if not d or not d["ts"]:
            return None

        return d["ts"][-1], d["price"][-1]

    def get_closest_price(self, asset: str, time: int) -> tuple[int, float] | None:
        """
        Retrieve the (timestamp, price) pair closest to the given timestamp for a specific asset.
        Returns None if no data is available.
        """
        d = self.data.get(asset)
        if not d or not d["ts"]:
            return None

        ts = d["ts"]
        prices = d["price"]
        pos = bisect_left(ts, time)

        if pos == 0:
            return ts[0], prices[0]
        if pos == len(ts):
            return ts[-1], prices[-1]

        before = pos - 1
        after = pos
        if time - ts[before] <= ts[after] - time:
            return ts[before], prices[before]
        else:
            return ts[after], prices[after]

