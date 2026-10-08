import math
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

import crunch_synth.synth_score as synth_score
from crunch_synth.synth_score import (
    SYNTH_GAP_SCALE,
    calculate_crps_for_miner,
    evaluate_synth_score,
    fetch_synth_prompts,
    field_reference,
    gap_to_score,
    synth_total_score,
)
from crunch_synth.tracker import TrackerBase

START = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
STEPS_1H = [60, 300, 900, 1800, 3600]


def test_field_reference_like_synth_compute_prompt_scores():
    # median 102 -> cap 1020: 5000 is clipped and left out of the miss fill (p95)
    ref = field_reference([100.0, 101.0, 103.0, 5000.0, -1.0])
    assert ref.best == 100.0
    assert ref.cap == 1020.0
    assert ref.p95 == pytest.approx(np.percentile([100.0, 101.0, 103.0], 95))
    assert ref.gap(101.0) == pytest.approx(1.0)
    assert ref.gap(5000.0) == pytest.approx(920.0)
    assert ref.gap(99.0) == 0.0  # beating every miner is the best possible score
    assert ref.gap(None) == pytest.approx(ref.p95 - 100.0)  # a miss gets the p95 fill
    assert field_reference([-1.0, None]) is None


def test_gap_to_score():
    assert gap_to_score(0.0, 86400) == 1.0
    assert gap_to_score(SYNTH_GAP_SCALE[3600], 3600) == pytest.approx(math.exp(-1))
    assert gap_to_score(None, 3600) is None


def _real_path(n, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(np.r_[0.0, rng.normal(0, 0.001, n - 1)]))


def test_perfect_paths_score_zero_and_zero_prices_are_rejected():
    real = _real_path(289)
    paths = np.tile(real, (50, 1))
    assert calculate_crps_for_miner(paths, real, 300, synth_score.SCORING_INTERVALS[300]) == pytest.approx(0.0, abs=1e-9)

    paths[0, 10] = 0.0
    assert synth_total_score(paths, real, "ETH", 86400) == -1.0


def test_volatility_term_only_on_crypto_1h():
    real = _real_path(61, seed=1)
    rng = np.random.default_rng(2)
    paths = real[0] * np.exp(np.cumsum(np.c_[np.zeros(200), rng.normal(0, 0.001, (200, 60))], axis=1))
    price_only = calculate_crps_for_miner(paths, real, 60, synth_score.SCORING_INTERVALS[60])
    assert synth_total_score(paths, real, "BTC", 3600) > price_only  # + volatility CRPS
    assert synth_total_score(paths, real, "NVDAX", 3600) == pytest.approx(price_only)
    # before Synth added the volatility term, its published 1h CRPS is price only
    assert synth_total_score(paths, real, "BTC", 3600, scored_at=datetime(2026, 9, 1, tzinfo=timezone.utc)) == pytest.approx(price_only)
    assert synth_total_score(paths, real, "BTC", 3600, scored_at=datetime(2026, 9, 23, tzinfo=timezone.utc)) > price_only


def _api_rows(scored, crps_by_uid):
    return [{"miner_uid": uid, "asset": "BTC", "prompt_score": 0, "scored_time": scored, "crps": crps, "time_length": 3600}
            for uid, crps in crps_by_uid.items()]


def test_fetch_synth_prompts_pages_and_filters(monkeypatch):
    pages = [
        _api_rows("2026-10-01T13:00:00Z", {1: 10.0, 2: -1.0}) + _api_rows("2026-10-01T13:10:00Z", {1: 11.0, 2: 12.0}),
        _api_rows("2026-10-01T13:20:00Z", {1: 9.0}) + _api_rows("2026-10-01T15:00:00Z", {1: 8.0}),  # last one out of the window
        [],
    ]
    calls = []

    def fake_get(url, params, timeout):
        calls.append(params)
        response = MagicMock(status_code=200)
        response.json.return_value = pages[len(calls) - 1]
        return response

    monkeypatch.setattr(synth_score.requests, "get", fake_get)
    prompts = fetch_synth_prompts("BTC", 3600, START, START + timedelta(minutes=30))

    assert [p[0] for p in prompts] == [START, START + timedelta(minutes=10), START + timedelta(minutes=20)]
    assert prompts[0][1] == {"1": 10.0}  # the missed answer (-1) is not part of the field
    assert calls[1]["from"] == "2026-10-01T13:10:01Z"  # next page starts after the latest row seen


class _ConstantTracker(TrackerBase):
    """Predicts N(0, 0.1 x sqrt(step in minutes)) price changes and records what it saw."""

    def __init__(self):
        super().__init__(async_mode=False)
        self.seen_last_ts = []

    def predict(self, asset, horizon, step):
        self.seen_last_ts.append(self.prices.get_last_price(asset)[0])
        scale = 0.1 * math.sqrt(step / 60)
        return [{"type": "builtin", "name": "norm", "params": {"loc": 0.0, "scale": scale}}] * (horizon // step)


def test_evaluate_synth_score_end_to_end(monkeypatch):
    t0 = int(START.timestamp())
    rng = np.random.default_rng(3)
    minutes = 3 * 24 * 60
    closes = 100.0 + np.cumsum(rng.normal(0, 0.1, minutes))
    history = [(t0 - 2 * 86400 + 60 * i, float(p)) for i, p in enumerate(closes)]
    monkeypatch.setattr(synth_score.pricedb, "get_price_history", lambda **kwargs: history)

    prompt_starts = [START + timedelta(minutes=10 * k) for k in range(3)]
    field = {str(uid): 50.0 + uid for uid in range(20)}
    monkeypatch.setattr(synth_score, "fetch_synth_prompts", lambda *args, **kwargs: [(p, dict(field)) for p in prompt_starts])

    tracker = _ConstantTracker()
    result = evaluate_synth_score(tracker, "BTC", 3600, STEPS_1H, prompt_starts[0], prompt_starts[-1], history_days=1)

    assert len(result.prompts) == 3
    # the tracker only ever sees prices closed before the validator's request (1 min before the start)
    for prompt_start, last_seen in zip(prompt_starts, tracker.seen_last_ts[::len(STEPS_1H)]):
        assert last_seen + 60 <= int(prompt_start.timestamp()) - 60
    ref = field_reference(field.values())
    for _, row in result.prompts.iterrows():
        assert row["gap"] == pytest.approx(ref.gap(row["crps"]))
    assert 0.0 <= result.score <= 1.0
    assert result.score == pytest.approx(gap_to_score(result.prompts["gap"].mean(), 3600))
    assert 1 <= result.rank <= result.num_miners + 1


def test_extra_series_are_fed_causally(monkeypatch):
    t0 = int(START.timestamp())
    history = [(t0 - 86400 + 60 * i, 100.0 + 0.01 * i) for i in range(3 * 1440)]
    monkeypatch.setattr(synth_score.pricedb, "get_price_history", lambda **kwargs: history)
    monkeypatch.setattr(synth_score, "fetch_synth_prompts", lambda *args, **kwargs: [(START, {"1": 50.0, "2": 60.0})])

    seen = {}

    class Tracker(_ConstantTracker):
        def tick(self, data):
            super().tick(data)
            for name, points in data.items():
                seen.setdefault(name, []).extend(points)

    exogenous = [(t0 - 3600 + 60 * i, float(i)) for i in range(120)]  # one hour before to one hour after the start
    evaluate_synth_score(Tracker(), "BTC", 3600, STEPS_1H, START, START, history_days=1, extra_series={"NASDAQ": exogenous})

    assert seen["NASDAQ"]
    assert max(ts for ts, _ in seen["NASDAQ"]) + 60 <= t0 - 60  # nothing after the request time
