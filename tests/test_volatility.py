import random
from unittest.mock import MagicMock

import numpy as np
import pytest

from crunch_synth.prices import PriceStore
from crunch_synth.tracker_evaluator import TrackerEvaluator
from crunch_synth.utils.densitytosimulations import simulate_points, combine_multiscale_simulations, condition_sum
from crunch_synth.volatility import (
    VOL_CRPS_K,
    block_volatilities,
    crps_ensemble,
    score_volatility,
)

START_TS = 1_759_276_800  # 2025-10-01 00:00 UTC
STEPS_1H = [60, 300, 900, 1800, 3600]


def _norm(scale):
    return {"type": "builtin", "name": "norm", "params": {"loc": 0.0, "scale": scale}}


def _predictions(horizon, steps, scale_per_minute=10.0):
    return {
        step: [_norm(scale_per_minute * np.sqrt(step / 60)) for _ in range(horizon // step)]
        for step in steps
    }


def _price_store(asset="BTC", minutes=180, seed=0):
    rng = np.random.default_rng(seed)
    prices = 60_000 + np.cumsum(rng.normal(0, 10, minutes))
    store = PriceStore()
    store.add_prices(asset, [(START_TS + 60 * i, float(p)) for i, p in enumerate(prices)])
    return store


def test_crps_ensemble_matches_definition():
    rng = np.random.default_rng(1)
    x = rng.normal(size=200)
    y = 0.3
    expected = np.mean(np.abs(x - y)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert crps_ensemble(y, x) == pytest.approx(expected, rel=1e-12)


def test_crps_ensemble_matches_properscoring():
    properscoring = pytest.importorskip("properscoring")
    rng = np.random.default_rng(2)
    x = rng.lognormal(size=1000)
    assert crps_ensemble(1.2, x) == pytest.approx(properscoring.crps_ensemble(1.2, x), rel=1e-10)


def test_block_volatilities_skips_blocks_without_data():
    path = np.array([[100.0, 101.0, 100.0, np.nan, np.nan, np.nan, 102.0]])
    vols = block_volatilities(path, 3)
    assert np.isfinite(vols[0, 0])
    assert np.isnan(vols[0, 1])


def test_combine_multiscale_matches_loop():
    rng = np.random.default_rng(3)
    paths = {"3600": rng.normal(size=(20, 1)), "900": rng.normal(size=(20, 4)), "60": rng.normal(size=(20, 60))}
    config = {"3600": 3600, "900": 900, "60": 60}

    # Reference: the per-path loop used by the live scorer
    constrained = {k: v.copy() for k, v in paths.items()}
    for parent, child, ratio in (("3600", "900", 4), ("900", "60", 15)):
        for i in range(20):
            for k in range(constrained[parent].shape[1]):
                sl = slice(k * ratio, (k + 1) * ratio)
                constrained[child][i, sl] = condition_sum(constrained[child][i, sl], constrained[parent][i, k])
    expected = np.zeros((20, 61))
    expected[:, 1:] = np.cumsum(constrained["60"], axis=1)

    np.testing.assert_allclose(combine_multiscale_simulations(paths, config), expected, atol=1e-12)


def test_simulate_points_ignores_unknown_params():
    spec = {"type": "builtin", "name": "norm", "params": {"loc": 0.0, "scale": 1.0, "df": 5}}
    samples, _ = simulate_points(spec, num_simulations=10)
    assert samples.shape == (10,)


def test_score_volatility_is_deterministic_and_keeps_rng_state():
    store = _price_store()
    entry = (START_TS + 3600 + 600, _predictions(3600, STEPS_1H), STEPS_1H)

    np.random.seed(42)
    random.seed(42)
    first = score_volatility("BTC", entry, store)
    np_after, py_after = np.random.random(), random.random()

    np.random.seed(42)
    random.seed(42)
    second = score_volatility("BTC", entry, store)

    assert first["vol_crps"] > 0
    assert first["vol_crps"] == second["vol_crps"]
    assert first["vol_score"] == pytest.approx(VOL_CRPS_K["BTC"] * first["vol_crps"])
    assert (np_after, py_after) == (np.random.random(), random.random())


def test_score_volatility_not_applicable():
    store = _price_store("XAUT")
    entry = (START_TS + 3600 + 600, _predictions(3600, STEPS_1H), STEPS_1H)
    assert score_volatility("XAUT", entry, store) is None

    store = _price_store("BTC", minutes=60 * 30)
    steps_24h = [300, 3600, 6 * 3600, 24 * 3600]
    entry = (START_TS + 86400 + 600, _predictions(86400, steps_24h), steps_24h)
    assert score_volatility("BTC", entry, store) is None


def _evaluator(store, score_volatility=True):
    tracker = MagicMock()
    tracker.prices = store
    return TrackerEvaluator(tracker, score_volatility=score_volatility)


def test_evaluator_adds_volatility_on_1h():
    store = _price_store()
    entry = (START_TS + 3600 + 600, _predictions(3600, STEPS_1H), STEPS_1H)

    with_vol = _evaluator(store)
    without_vol = _evaluator(store, score_volatility=False)

    price_only, none = without_vol._score_quarantines_with_volatility("BTC", [entry])
    total, vol = with_vol._score_quarantines_with_volatility("BTC", [entry])

    assert none is None
    assert vol > 0
    assert total == pytest.approx(price_only + vol)


def test_evaluator_falls_back_to_price_on_volatility_failure():
    store = _price_store()
    predictions = _predictions(3600, STEPS_1H)
    predictions[60][0] = {"type": "builtin", "name": "not_a_distribution", "params": {}}
    entry = (START_TS + 3600 + 600, predictions, STEPS_1H)

    evaluator = _evaluator(store)
    evaluator._score_quarantines = MagicMock(return_value=1.0)

    with pytest.warns(RuntimeWarning, match="volatility scoring failed"):
        total, vol = evaluator._score_quarantines_with_volatility("BTC", [entry])

    assert (total, vol) == (1.0, None)
