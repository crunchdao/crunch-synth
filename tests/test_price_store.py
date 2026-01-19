# tests/test_price_store.py

import pytest
from datetime import datetime, timedelta, timezone
from bisect import bisect_left

from crunch_synth.prices import PriceStore, Asset, PriceEntry


# ---------------------------
# Helper function
# ---------------------------
def generate_price_series(start_ts: int, n: int, step_sec: int, start_price: float = 100.0):
    """
    Generate synthetic price data as a list of (timestamp, price) tuples.
    """
    series = []
    price = start_price
    ts = start_ts
    for _ in range(n):
        series.append((ts, price))
        price *= 1 + 0.01  # simple +1% per step
        ts += step_sec
    return series


# ---------------------------
# Tests
# ---------------------------
def test_add_and_get_last_price():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 5, 60)

    store.add_prices("BTC", series)

    # Check last price
    last_ts, last_price = store.get_last_price("BTC")
    assert last_ts == series[-1][0]
    assert last_price == series[-1][1]


def test_get_closest_price():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 5, 60)
    store.add_prices("BTC", series)

    # Exact timestamp
    closest_ts, closest_price = store.get_closest_price("BTC", series[2][0])
    assert closest_ts == series[2][0]
    assert closest_price == series[2][1]

    # Between timestamps
    mid_ts = (series[2][0] + series[3][0]) // 2
    closest_ts, closest_price = store.get_closest_price("BTC", mid_ts)
    # Should be closer to series[2]
    assert closest_ts == series[2][0]


def test_get_prices_with_resolution():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 10, 60)  # every 1 min
    store.add_prices("BTC", series)

    # Request prices every 2 minutes
    prices_resampled = store.get_prices("BTC", resolution=120)
    # Should skip every other point
    assert all(prices_resampled[i + 1][0] - prices_resampled[i][0] >= 120 for i in range(len(prices_resampled) - 1))
    assert len(prices_resampled) == 5


def test_get_prices_with_days():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 10, 60 * 60 * 24)  # every 1 min
    store.add_prices("BTC", series)

    prices_resampled = store.get_prices("BTC", 5)
    assert len(prices_resampled) == 6  # because inclusive


def test_add_older_prices():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 10, 60)  # every 1 min

    series2 = generate_price_series(series[-1][0] + 60, 1, 60)  # every 1 min
    store.add_prices("BTC", series2)

    store.add_prices("BTC", series)

    # Request prices every 2 minutes
    prices_resampled = store.get_prices("BTC", resolution=60)
    # Should skip every other point
    assert len(prices_resampled) == len(series) + len(series2)


def test_add_bulk_and_deduplication():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())

    # Add first batch
    series1 = generate_price_series(ts, 5, 60)
    store.add_bulk({"BTC": series1})

    # Add overlapping batch with updated last price
    series2 = series1[-3:] + [(series1[-1][0] + 60, 200.0)]
    store.add_bulk({"BTC": series2})

    # Last price should be from series2
    last_ts, last_price = store.get_last_price("BTC")
    assert last_price == 200.0
    assert len(store.data["BTC"]) == len(series1) + 1


def test_add_duplicate_prices():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 5, 60)
    store.add_prices("BTC", series)

    # Add duplicate prices
    store.add_prices("BTC", series)

    # Fetch all prices to ensure no duplicates exist
    prices = store.get_prices("BTC")
    assert len(prices) == len(series)
    assert prices == series

def test_update_existing_timestamp_middle():
    store = PriceStore()
    ts = int(datetime.now(timezone.utc).timestamp())
    series = generate_price_series(ts, 5, 60)
    store.add_prices("BTC", series)

    # update middle point
    mid_t = series[2][0]
    store.add_prices("BTC", [(mid_t, 999.0)])

    closest_ts, closest_price = store.get_closest_price("BTC", mid_t)
    assert closest_ts == mid_t
    assert closest_price == 999.0

    # length unchanged
    assert len(store.data["BTC"]) == len(series)

def test_duplicates_inside_same_batch_last_wins():
    store = PriceStore()
    t = int(datetime.now(timezone.utc).timestamp())
    store.add_prices("BTC", [(t, 100.0), (t, 200.0), (t, 300.0)])
    assert store.get_last_price("BTC") == (t, 300.0)
    assert len(store.data["BTC"]) == 1

def test_add_unsorted_entries():
    store = PriceStore()
    t = int(datetime.now(timezone.utc).timestamp())
    entries = [(t+120, 3.0), (t, 1.0), (t+60, 2.0)]
    store.add_prices("BTC", entries)

    prices = store.get_prices("BTC", resolution=60)
    assert [x[0] for x in prices] == [t, t+60, t+120]

def test_window_days_trimming():
    store = PriceStore(window_days=5)
    now = int(datetime.now(timezone.utc).timestamp())

    # create points spanning 10 days (1/day)
    series = generate_price_series(now - 9*86400, 10, 86400)
    store.add_prices("BTC", series)

    # should keep only last 5 days => days 0..5 inclusive = 6 points
    # (si ton cutoff est ts_last - 5days, inclusif)
    assert len(store.data["BTC"]) == 6
    last_ts, _ = store.get_last_price("BTC")
    oldest_kept = min(store.data["BTC"].keys())
    assert oldest_kept >= last_ts - 5*86400
# ---------------------------
# Run pytest directly
# ---------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
