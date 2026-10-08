"""
Synth score: how a tracker would score on Synth itself.

Our leaderboard ranks trackers against each other, but the ensemble is mined on
Synth (Bittensor subnet 50), which compares it with every Synth miner on Synth's
own prompts. The Synth score measures that directly, the same way as the
"Synth score" columns of the leaderboard:

1. For each Synth prompt, the tracker predicts as our miner would answer it: with
   the prices known when the validator's request arrives (~1 min before a 1h
   prompt starts, ~2 min before a 24h one).
2. 1,000 price paths are simulated from the predicted densities, all step
   resolutions combined (exactly like the live miner).
3. The paths are scored with Synth's own CRPS on Synth's realized prices
   (5 min / 30 min / 3 h / 24 h changes for 24h; 1 to 60 min changes, gaps and
   the volatility term for 1h).
4. The tracker's gap is its CRPS minus the best Synth miner's CRPS on the same
   prompt -- the prompt score Synth would have given it.
5. Gaps are averaged over the prompts, then mapped to 0-1:
   `score = exp(-mean_gap / scale)`, 1 = best of all Synth miners on every prompt.

Synth's per-prompt scores of every miner come from its public API
(https://api.synthdata.co/validation/scores/historical).

The CRPS functions below are ported from synth-subnet
(synth/validator/crps_calculation.py, MIT License, Copyright (c) 2023 Opentensor).
"""

import math
import zlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import numpy as np
import pandas
import requests

from crunch_synth.price_provider import pricedb
from crunch_synth.volatility import (
    VOL_SCORING_BLOCKS,
    calculate_vol_crps_for_miner,
    crps_ensemble,
    get_interval_steps,
    is_vol_crps_applicable,
    simulate_price_paths,
)

SYNTH_SCORES_URL = "https://api.synthdata.co/validation/scores/historical"

# Synth's scoring intervals per time increment (synth/validator/competition_config.py)
SCORING_INTERVALS = {
    300: {
        "5min": 300,
        "30min": 1800,
        "3hour": 10800,
        "24hour_abs": 86400,
    },
    60: {
        "1min": 60,
        "2min": 120,
        "5min": 300,
        "15min": 900,
        "30min": 1800,
        "60min_abs": 3600,
        "0_5min_gaps": 300,
        "0_10min_gaps": 600,
        "0_15min_gaps": 900,
        "0_20min_gaps": 1200,
        "0_25min_gaps": 1500,
        "0_30min_gaps": 1800,
        "0_35min_gaps": 2100,
        "0_40min_gaps": 2400,
        "0_45min_gaps": 2700,
        "0_50min_gaps": 3000,
        "0_55min_gaps": 3300,
        "0_60min_gaps": 3600,
    },
}

# Synth's path resolution per horizon
TIME_INCREMENT = {3600: 60, 86400: 300}

# How long before a Synth prompt starts its validator request reaches a miner
REQUEST_LEAD_SECONDS = {3600: 60, 86400: 120}

# synth/validator/reward.py OUTLIER_SCORE_MEDIAN_MULTIPLE
OUTLIER_SCORE_MEDIAN_MULTIPLE = 10

# Median gap of a Synth miner per prompt (2026-09-28 to 2026-10-04): a typical Synth miner scores ~0.37
SYNTH_GAP_SCALE = {3600: 25.5, 86400: 42.5}

# Our asset names -> Synth's
ASSET_TO_SYNTH = {"XAUT": "XAU"}

# Synth added the volatility term to its crypto 1h score on this date: before it, its published 1h
# CRPS is price only, so ours must be too to be compared with it
SYNTH_VOL_CRPS_ACTIVE_FROM = datetime(2026, 9, 22, tzinfo=timezone.utc)


# ------------------------------------------------------------------
# Synth's CRPS (ported from synth-subnet)
# ------------------------------------------------------------------

def label_observed_blocks(arr: np.ndarray) -> np.ndarray:
    """
    Groups blocks of consecutive observed data together.
    Example input/output:
    [1.0, 2.0, np.nan, 4.0, np.nan, np.nan, 7.0, 8.0] -> [0, 0, -1, 1, -1, -1, 2, 2]
    """
    not_nan = ~np.isnan(arr)
    block_start = not_nan & np.concatenate(([True], ~not_nan[:-1]))
    group_numbers = np.cumsum(block_start) - 1
    return np.where(not_nan, group_numbers, -1)


def calculate_price_changes_over_intervals(
    price_paths: np.ndarray,
    interval_steps: int,
    absolute_price=False,
    is_gap=False,
) -> np.ndarray:
    """Price changes (in bps) over intervals, or absolute prices for `_abs` intervals."""
    interval_prices = price_paths[:, ::interval_steps]
    if is_gap:
        interval_prices = interval_prices[:, :2]

    if absolute_price:
        return interval_prices[:, 1:]

    return (np.diff(interval_prices, axis=1) / interval_prices[:, :-1]) * 10_000


def sum_crps_over_blocks(
    simulated_changes: np.ndarray,
    real_changes: np.ndarray,
    data_blocks: np.ndarray,
    absolute_price: bool,
    real_price_path: np.ndarray,
) -> float:
    """Sum the CRPS over the observed (non-missing) blocks of a single interval."""
    crps_values = 0.0
    for block in np.unique(data_blocks):
        if block == -1:
            continue

        mask = data_blocks == block
        simulated_changes_block = simulated_changes[:, mask]
        real_changes_block = real_changes[0, mask]
        crps_values_block = np.array([
            crps_ensemble(real_changes_block[t], simulated_changes_block[:, t])
            for t in range(simulated_changes_block.shape[1])
        ])

        if absolute_price:
            crps_values_block = crps_values_block / real_price_path[-1] * 10_000

        crps_values += float(crps_values_block.sum())

    return float(crps_values)


def calculate_crps_for_miner(
    simulation_runs: np.ndarray,
    real_price_path: np.ndarray,
    time_increment: int,
    scoring_intervals: dict[str, int],
) -> float:
    """Synth's price CRPS of a set of price paths (sum over the scoring intervals); -1 if rejected."""
    sum_all_scores = 0.0

    for interval_name, interval_seconds in scoring_intervals.items():
        interval_steps = get_interval_steps(interval_seconds, time_increment)
        absolute_price = interval_name.endswith("_abs")
        is_gap = interval_name.endswith("_gaps")

        if absolute_price:
            while real_price_path[::interval_steps].shape[0] == 1 and interval_steps > 1:
                interval_steps -= 1

        # Synth rejects the answer when a path has a zero price
        if np.any(simulation_runs == 0):
            return -1.0

        simulated_changes = calculate_price_changes_over_intervals(simulation_runs, interval_steps, absolute_price, is_gap)
        real_changes = calculate_price_changes_over_intervals(real_price_path.reshape(1, -1), interval_steps, absolute_price, is_gap)
        data_blocks = label_observed_blocks(real_changes[0])
        if len(data_blocks) == 0:
            continue

        sum_all_scores += sum_crps_over_blocks(simulated_changes, real_changes, data_blocks, absolute_price, real_price_path)

    return sum_all_scores


def synth_total_score(price_paths: np.ndarray, real_price_path: np.ndarray, asset: str, horizon: int,
                      scored_at: Optional[datetime] = None) -> float:
    """Synth's score of one answer: price CRPS, plus the volatility CRPS on crypto 1h (for prompts
    scored since SYNTH_VOL_CRPS_ACTIVE_FROM; `scored_at` None = now). -1 if rejected."""
    time_increment = TIME_INCREMENT[horizon]
    price_score = calculate_crps_for_miner(price_paths, real_price_path, time_increment, SCORING_INTERVALS[time_increment])
    vol_active = scored_at is None or scored_at >= SYNTH_VOL_CRPS_ACTIVE_FROM
    if price_score == -1.0 or not is_vol_crps_applicable(asset, horizon) or not vol_active:
        return price_score
    vol_score, _ = calculate_vol_crps_for_miner(price_paths, real_price_path, time_increment, VOL_SCORING_BLOCKS)
    return price_score + vol_score


# ------------------------------------------------------------------
# Synth prompt scores
# ------------------------------------------------------------------

@dataclass(frozen=True)
class FieldReference:
    """Best / outlier cap / miss fill of one Synth prompt, like Synth's compute_prompt_scores."""
    best: float
    cap: float
    p95: float

    def gap(self, crps: Optional[float]) -> float:
        """Prompt score of an answer joining this field (0 when it beats every miner); a miss gets the p95 fill."""
        if crps is None or not np.isfinite(crps) or crps <= 0:
            return max(0.0, self.p95 - self.best)
        return max(0.0, min(crps, self.cap) - self.best)


def field_reference(crps_values: Iterable[float]) -> Optional[FieldReference]:
    values = np.asarray([v for v in crps_values if v is not None and np.isfinite(v) and v > 0], dtype=float)
    if values.size == 0:
        return None
    cap = float(np.median(values)) * OUTLIER_SCORE_MEDIAN_MULTIPLE
    uncapped = values[values <= cap]
    p95 = float(np.percentile(uncapped if uncapped.size else np.minimum(values, cap), 95))
    return FieldReference(best=float(np.min(np.minimum(values, cap))), cap=cap, p95=p95)


def gap_to_score(mean_gap: Optional[float], horizon: int) -> Optional[float]:
    """0-1 Synth score of a mean gap (higher is better)."""
    if mean_gap is None or not np.isfinite(mean_gap):
        return None
    return math.exp(-max(0.0, mean_gap) / SYNTH_GAP_SCALE[horizon])


def fetch_synth_prompts(asset: str, horizon: int, start: datetime, end: datetime, timeout: int = 60) -> list[tuple[datetime, dict[str, float]]]:
    """
    Synth prompts of an asset/horizon starting in [start, end], with every miner's CRPS:
    [(prompt_start, {miner_uid: crps}), ...] (missed answers excluded).
    """
    synth_asset = ASSET_TO_SYNTH.get(asset, asset)
    first_scored = start + timedelta(seconds=horizon)
    last_scored = end + timedelta(seconds=horizon)
    by_prompt: dict[datetime, dict[str, float]] = {}

    cursor = first_scored
    while cursor <= last_scored:
        # the API returns rows from `from` onwards (`to` is required but not applied): page on scored_time
        response = requests.get(SYNTH_SCORES_URL, params={
            "from": cursor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": min(cursor + timedelta(days=2), last_scored).strftime("%Y-%m-%dT%H:%M:%SZ"),  # the API refuses > 3 days
            "asset": synth_asset,
            "time_increment": TIME_INCREMENT[horizon],
            "time_length": horizon,
        }, timeout=timeout)
        if response.status_code == 404:
            break
        response.raise_for_status()
        rows = response.json()
        if not rows:
            break

        latest = cursor
        for row in rows:
            scored = datetime.fromisoformat(row["scored_time"].replace("Z", "+00:00"))
            latest = max(latest, scored)
            if first_scored <= scored <= last_scored and row["crps"] > 0:
                by_prompt.setdefault(scored, {})[str(row["miner_uid"])] = float(row["crps"])
        if latest <= cursor:
            break
        cursor = latest + timedelta(seconds=1)

    return sorted((scored - timedelta(seconds=horizon), field) for scored, field in by_prompt.items())


@dataclass
class SynthScoreResult:
    asset: str
    horizon: int
    prompts: pandas.DataFrame  # one row per Synth prompt
    mean_gap: Optional[float]
    score: Optional[float]  # 0-1, higher is better
    rank: Optional[int]  # where the tracker's mean gap would rank among the Synth miners of these prompts
    num_miners: int

    def __repr__(self):
        score = f"{self.score:.3f}" if self.score is not None else "n/a"
        gap = f"{self.mean_gap:.1f}" if self.mean_gap is not None else "n/a"
        return (f"SynthScoreResult({self.asset} {self.horizon // 3600}h: score={score}, mean gap={gap}, "
                f"rank #{self.rank} of {self.num_miners + 1} on {len(self.prompts)} prompts)")


def evaluate_synth_score(
    tracker,
    asset: str,
    horizon: int,
    steps: list[int],
    start: datetime,
    end: datetime,
    max_prompts: int = 30,
    history_days: int = 30,
    seed: int = 0,
    extra_series: Optional[dict[str, list[tuple[int, float]]]] = None,
) -> SynthScoreResult:
    """
    Score a tracker on Synth's own prompts, like the leaderboard's Synth score.

    Parameters
    ----------
    tracker : TrackerBase
        A fresh tracker: it receives `history_days` of 1-minute prices before the
        first prompt, then the prices up to each prompt's request time.
    asset, horizon, steps
        What to predict, e.g. "BTC", 86400, [300, 3600, 21600, 86400].
    start, end : datetime (UTC)
        Window of Synth prompt start times (they must be resolved: end + horizon <= now).
    max_prompts : int
        Prompts evaluated, evenly spread over the window (each takes a few seconds).
    extra_series : dict, optional
        Other series the tracker reads through `tick()` (e.g. exogenous data preloaded for
        backtesting): {name: [(timestamp, value), ...]}, fed with the same rule as prices
        (only values known at the request time).
    """
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    prompts = fetch_synth_prompts(asset, horizon, start, end)
    if not prompts:
        raise ValueError(f"No resolved Synth prompt for {asset} h={horizon} between {start} and {end}")
    if len(prompts) > max_prompts:
        prompts = [prompts[int(i)] for i in np.linspace(0, len(prompts) - 1, max_prompts)]

    time_increment = TIME_INCREMENT[horizon]
    lead = REQUEST_LEAD_SECONDS[horizon]
    history = pricedb.get_price_history(
        asset=asset,
        from_=prompts[0][0] - timedelta(days=history_days),
        to=prompts[-1][0] + timedelta(seconds=horizon + 120),
    )
    closes = {int(ts): float(price) for ts, price in history}
    bars = sorted(closes)

    extra = {name: sorted((int(ts), value) for ts, value in points) for name, points in (extra_series or {}).items()}
    extra_fed_until = {name: None for name in extra}

    rows, fed_until = [], None
    for prompt_start, field in prompts:
        t0 = int(prompt_start.timestamp())
        request_ts = t0 - lead
        # prices a live tracker has at the request: 1-minute bars closed by then
        # (a bar's timestamp is its open, its close is the price one minute later)
        known = [ts for ts in bars if ts + 60 <= request_ts and (fed_until is None or ts > fed_until)]
        update = {asset: [(ts, closes[ts]) for ts in known]} if known else {}
        for name, points in extra.items():
            new = [(ts, v) for ts, v in points if ts + 60 <= request_ts and (extra_fed_until[name] is None or ts > extra_fed_until[name])]
            if new:
                update[name] = new
                extra_fed_until[name] = new[-1][0]
        if update:
            tracker.tick(update)
        if known:
            fed_until = known[-1]
        if fed_until is None:
            continue

        # Synth's realized path: the price at t is the close of the bar opening at t
        real = np.array([closes.get(t0 + i * time_increment, np.nan) for i in range(horizon // time_increment + 1)])
        reference = field_reference(field.values())
        if reference is None or np.isnan(real[0]):
            continue

        try:
            predictions = tracker.predict_all(asset, horizon, steps)
            paths = simulate_price_paths(predictions, closes[fed_until], seed=zlib.crc32(f"{asset}{t0}{seed}".encode()))
            paths = np.where(paths < 0, 0.0, paths)  # like the miner; Synth then rejects a zero price
            crps = synth_total_score(paths, real, asset, horizon, scored_at=prompt_start + timedelta(seconds=horizon))
        except Exception as error:
            crps, failure = None, repr(error)
        else:
            failure = None if crps > 0 else "rejected by Synth (zero price in a path)"
            crps = crps if crps > 0 else None

        field_values = np.array(list(field.values()))
        rows.append({
            "prompt_start": prompt_start,
            "crps": crps,
            "synth_best": reference.best,
            "synth_median": float(np.median(field_values)),
            "gap": reference.gap(crps),
            "rank_on_prompt": int((field_values < crps).sum()) + 1 if crps is not None else None,
            "num_miners": int(field_values.size),
            "failure": failure,
            "field": field,
        })

    table = pandas.DataFrame(rows)
    if table.empty:
        return SynthScoreResult(asset, horizon, table, None, None, None, 0)

    # Where the tracker would rank: mean gap of every Synth miner over the same prompts
    miner_gaps: dict[str, list[float]] = {}
    for row in rows:
        reference = field_reference(row["field"].values())
        for uid, value in row["field"].items():
            miner_gaps.setdefault(uid, []).append(reference.gap(value))
    miner_means = [float(np.mean(g)) for g in miner_gaps.values() if len(g) >= 0.8 * len(rows)]

    mean_gap = float(table["gap"].mean())
    return SynthScoreResult(
        asset=asset,
        horizon=horizon,
        prompts=table.drop(columns=["field"]),
        mean_gap=mean_gap,
        score=gap_to_score(mean_gap, horizon),
        rank=int(sum(m < mean_gap for m in miner_means)) + 1,
        num_miners=len(miner_means),
    )
