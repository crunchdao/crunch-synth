import pytest
from unittest.mock import MagicMock

from crunch_synth.tracker_evaluator import TrackerEvaluator
from crunch_synth.constants import ASSET_WEIGHTS


def _make_evaluator(scores_by_asset: dict[str, list[float]]) -> TrackerEvaluator:
    """Create a TrackerEvaluator with pre-filled scores (bypass predictions)."""
    tracker = MagicMock()
    ev = TrackerEvaluator(tracker)
    for asset, scores in scores_by_asset.items():
        for i, s in enumerate(scores):
            ev.scores[asset].append((i, s))
    return ev


def test_weighted_overall_score():
    """overall_score uses weighted average across assets."""
    ev = _make_evaluator({"BTC": [0.8], "SOL": [0.5]})

    w_btc = ASSET_WEIGHTS["BTC"]
    w_sol = ASSET_WEIGHTS["SOL"]
    expected = (w_btc * 0.8 + w_sol * 0.5) / (w_btc + w_sol)

    assert ev.overall_score() == pytest.approx(expected)


def test_weighted_overall_score_multiple_scores():
    """Weighted average uses per-asset mean, then weighted across assets."""
    ev = _make_evaluator({"BTC": [0.6, 0.8], "ETH": [0.4, 0.6]})

    w_btc = ASSET_WEIGHTS["BTC"]
    w_eth = ASSET_WEIGHTS["ETH"]
    expected = (w_btc * 0.7 + w_eth * 0.5) / (w_btc + w_eth)

    assert ev.overall_score() == pytest.approx(expected)


def test_overall_score_unknown_asset_returns_none():
    """If any asset is not in ASSET_WEIGHTS, overall_score returns None."""
    ev = _make_evaluator({"BTC": [0.8], "UNKNOWN": [0.5]})

    assert ev.overall_score() is None


def test_overall_score_empty():
    """No scores at all returns 0.0."""
    ev = _make_evaluator({})

    assert ev.overall_score() == 0.0