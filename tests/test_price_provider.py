from datetime import datetime

from crunch_synth.price_provider import shared_pricedb

# ---------------------------
# Tests
# ---------------------------


def test_get_price_history():
    prices = shared_pricedb.get_price_history(
        asset="BTC",
        from_=datetime(2024, 1, 1),
        to=datetime(2024, 1, 3),
    )

    assert len(prices) > 1
