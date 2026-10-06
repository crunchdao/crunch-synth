"""
Volatility CRPS: the second term of the live 1-hour score.

Since 22 September 2026, Synth scores its crypto 1h competition with
`price CRPS + volatility CRPS` (see
https://synthdata.co/research/introducing-volatility-crps-updated), and the
Crunch coordinator does the same for the 1h horizon:

1. 1,000 price paths are simulated from the submitted densities (all step
   resolutions combined, finest = 1 minute).
2. For every path, the realized volatility (std of 1-minute returns, in bps) is
   computed over the full hour, four 15-minute blocks and twelve 5-minute blocks.
3. Each block's realized volatility is scored with CRPS against the ensemble of
   simulated volatilities, weighted by `VOL_SCORING_BLOCKS` (lambda = 5.25).

The volatility CRPS is in Synth's units, so the coordinator converts it into the
units of our price CRPS with a factor `k` (median, over every participant of the
round, of `price CRPS / Synth price CRPS`). A single local tracker has no round
to take that median over, so `VOL_CRPS_K` holds the typical live value per
asset: local 1h scores are close to, but not exactly, the live ones.
"""

import random
import warnings
import zlib

import numpy as np

from crunch_synth.utils.densitytosimulations import simulate_paths, combine_multiscale_simulations


# Synth's crypto 1h competition: horizon and assets that get the volatility term
VOL_CRPS_HORIZON = 3600
VOL_CRPS_ASSETS = frozenset({"BTC", "ETH", "SOL", "XRP", "HYPE"})

# 1h score = price CRPS + VOL_CRPS_LAMBDA x the mean, over the block sizes, of the
# mean per-block volatility CRPS -- so each weight is
# VOL_CRPS_LAMBDA / (3 block sizes x the number of blocks of that size).
VOL_CRPS_LAMBDA = 5.25
VOL_SCORING_BLOCKS = {
    "vol_60min": (3600, VOL_CRPS_LAMBDA / 3),   # 1 block
    "vol_15min": (900, VOL_CRPS_LAMBDA / 12),   # 4 blocks
    "vol_5min": (300, VOL_CRPS_LAMBDA / 36),    # 12 blocks
}

# Same ensemble size as Synth miners submit
VOL_CRPS_NUM_PATHS = 1000

# Converts the volatility CRPS into the units of the (normalized) price CRPS.
# Median of the live coordinator's per-round factor (Oct 2026); it varies by
# about +/-15% from one round to another.
VOL_CRPS_K = {
    "BTC": 0.00310804,
    "ETH": 0.00183436,
    "SOL": 0.00161268,
    "XRP": 0.00204925,
    "HYPE": 0.00389259,
}


def is_vol_crps_applicable(asset: str, horizon: int) -> bool:
    return horizon == VOL_CRPS_HORIZON and asset in VOL_CRPS_ASSETS


def crps_ensemble(observation: float, forecasts: np.ndarray) -> float:
    """
    CRPS of an ensemble forecast against one observation:
    E|X - y| - 0.5 * E|X - X'|

    Same value as `properscoring.crps_ensemble` (unweighted); NaN forecasts are
    ignored.
    """
    x = np.sort(np.asarray(forecasts, dtype=float).ravel())
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0 or np.isnan(observation):
        return float("nan")

    # sum_{i,j} |x_i - x_j| = 2 * sum_i (2i - n + 1) * x_i  for sorted x (0-indexed)
    spread = 2.0 * np.sum((2.0 * np.arange(n) - n + 1.0) * x) / (n * n)
    return float(np.mean(np.abs(x - observation)) - 0.5 * spread)


def get_interval_steps(interval_seconds: int, time_increment: int) -> int:
    return int(interval_seconds / time_increment)


def block_volatilities(price_paths: np.ndarray, block_steps: int) -> np.ndarray:
    """
    Calculate the volatility of each consecutive block of a price path.

    Parameters:
        price_paths (numpy.ndarray): Array of price paths.
        block_steps (int): Number of steps that make up a block.

    Returns:
        numpy.ndarray: Standard deviation, in basis points, of the step
        returns inside each block. A block with fewer than two observed
        returns is NaN so that it can be skipped; the steps left over after
        the last whole block are dropped.
    """
    returns = (np.diff(price_paths, axis=1) / price_paths[:, :-1]) * 10_000

    n_blocks = returns.shape[1] // block_steps
    blocks = returns[:, : n_blocks * block_steps].reshape(
        returns.shape[0], n_blocks, block_steps
    )

    # Only the scorable blocks go through nanstd: it warns on a block with
    # fewer than two observed returns, which is the tolerated case here.
    scorable = np.sum(~np.isnan(blocks), axis=2) >= 2
    volatilities: np.ndarray = np.full(scorable.shape, np.nan)
    volatilities[scorable] = np.nanstd(blocks[scorable], axis=1, ddof=1)

    return volatilities


def calculate_vol_crps_for_miner(
    simulation_runs: np.ndarray,
    real_price_path: np.ndarray,
    time_increment: int,
    vol_scoring_blocks: dict[str, tuple[int, float]] = VOL_SCORING_BLOCKS,
) -> tuple[float, list[dict]]:
    """
    Calculate the weighted volatility CRPS for a set of simulated price paths.

    The path is cut into consecutive blocks of each configured size, the
    realized volatility of every block is scored against the ensemble of
    simulated block volatilities, and the per-block CRPS values are summed
    and scaled by that block size's weight.

    Parameters:
        simulation_runs (numpy.ndarray): Simulated price paths.
        real_price_path (numpy.ndarray): The real price path.
        time_increment (int): Time increment in seconds.
        vol_scoring_blocks (dict): Dictionary of block sizes with their names,
            durations in seconds and weights.

    Returns:
        float: Sum of the weighted volatility CRPS over the block sizes.
        list[dict]: Per-block-size detail rows (name, raw CRPS sum, weight).
    """
    detailed_crps_data: list[dict] = []
    sum_all_scores = 0.0

    for name, (block_seconds, weight) in vol_scoring_blocks.items():
        block_steps = get_interval_steps(block_seconds, time_increment)

        simulated_vol = block_volatilities(simulation_runs, block_steps)
        real_vol = block_volatilities(
            real_price_path.reshape(1, -1), block_steps
        )[0]
        observed_blocks = np.flatnonzero(~np.isnan(real_vol))

        # Not enough observed data -> skip this block size
        if observed_blocks.size == 0:
            continue

        crps_sum = float(
            sum(
                crps_ensemble(real_vol[block], simulated_vol[:, block])
                for block in observed_blocks
            )
        )
        sum_all_scores += weight * crps_sum

        detailed_crps_data.append(
            {
                "Interval": name,
                "Increment": "Total",
                "CRPS": crps_sum,
                "Weight": weight,
            }
        )

    detailed_crps_data.append(
        {"Interval": "Vol", "Increment": "Total", "CRPS": sum_all_scores}
    )

    return sum_all_scores, detailed_crps_data


def simulate_price_paths(
    predictions: dict,
    start_price: float,
    seed: int,
    num_paths: int = VOL_CRPS_NUM_PATHS,
) -> np.ndarray:
    """
    Simulate price paths from multi-resolution predictions, like the live scorer.

    Parameters
    ----------
    predictions : dict[int, list[dict]]
        Mapping step (seconds) -> list of densities over price increments.
    start_price : float
        Price at the start of the horizon.
    seed : int
        Seed of the simulation. The global `numpy` and `random` states are
        restored afterwards, so the tracker's own randomness is not affected.

    Returns
    -------
    np.ndarray
        Shape (num_paths, horizon // finest_step + 1), absolute prices.
    """
    np_state = np.random.get_state()
    py_state = random.getstate()
    try:
        # simulate_points samples through both numpy's and the stdlib's global RNG
        np.random.seed(seed)
        random.seed(seed)

        dict_paths = {}
        for step, densities in predictions.items():
            simulations = simulate_paths(
                densities,
                start_point=0.0,
                num_paths=num_paths,
                step_minutes=None,
                start_time=None,
                mode="point",
            )
            dict_paths[str(step)] = simulations["paths"][:, 1:]
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)

    step_config = {str(step): int(step) for step in predictions.keys()}
    return start_price + combine_multiscale_simulations(dict_paths, step_config)


def realized_price_path(prices, asset: str, end_ts: int, horizon: int, time_increment: int) -> np.ndarray:
    """
    Realized prices on the prediction's grid (end - horizon, ..., end).

    Each grid point takes the closest known price, as long as it is within half
    an increment; otherwise the point is a gap (NaN), which is skipped.
    """
    start_ts = end_ts - horizon
    path = np.full(horizon // time_increment + 1, np.nan)
    for i in range(path.size):
        t = start_ts + i * time_increment
        closest = prices.get_closest_price(asset, t)
        if closest is not None and abs(closest[0] - t) <= time_increment / 2:
            path[i] = closest[1]
    return path


def score_volatility(asset: str, quarantine_entry: tuple, prices) -> dict | None:
    """
    Volatility CRPS of one quarantined 1h prediction, as scored live.

    Parameters
    ----------
    asset : str
    quarantine_entry : tuple
        (resolution_ts, predictions, steps), as returned by
        `TrackerEvaluator.evaluate_quarantine()`.
    prices : PriceStore
        Prices covering the prediction's horizon.

    Returns
    -------
    dict or None
        None when the prediction is not volatility-scored (not 1h, not a
        volatility asset, or no realized prices over the hour). Otherwise:
        - "vol_crps": weighted volatility CRPS (Synth units)
        - "vol_score": vol_crps converted into price-CRPS units (VOL_CRPS_K),
          i.e. what is added to the tracker's score
        - "details": per-block-size CRPS rows
        - "simulated_paths", "realized_path", "time_increment"

    Raises
    ------
    ValueError
        If the densities cannot be turned into valid price paths (live scoring
        counts such a prediction as failed).
    """
    resolution_ts, predictions, steps = quarantine_entry

    time_increment = min(int(step) for step in predictions.keys())
    horizon = time_increment * len(predictions[time_increment])
    if not is_vol_crps_applicable(asset, horizon):
        return None

    real_path = realized_price_path(prices, asset, resolution_ts, horizon, time_increment)
    if np.isnan(real_path[0]) or np.count_nonzero(~np.isnan(real_path)) < 2:
        return None

    try:
        seed = zlib.crc32(f"{asset}:{resolution_ts}".encode())
        sim_paths = simulate_price_paths(predictions, real_path[0], seed)
    except Exception as error:
        raise ValueError(f"cannot simulate price paths from the densities: {error}") from error

    if sim_paths.shape[1] != horizon // time_increment + 1:
        raise ValueError(f"number of price points invalid {sim_paths.shape[1]} != {horizon // time_increment + 1}")
    if np.any(sim_paths == 0):
        raise ValueError("zero price encountered in simulated paths")

    vol_crps, details = calculate_vol_crps_for_miner(sim_paths, real_path, time_increment)
    if not np.isfinite(vol_crps):
        raise ValueError("non-finite volatility CRPS")

    return {
        "vol_crps": float(vol_crps),
        "vol_score": VOL_CRPS_K[asset] * float(vol_crps),
        "details": details,
        "simulated_paths": sim_paths,
        "realized_path": real_path,
        "time_increment": time_increment,
    }


def warn_volatility_failure(asset: str, resolution_ts: int, error: Exception):
    warnings.warn(
        f"[{asset}] volatility scoring failed for the prediction resolved at {resolution_ts} ({error}). "
        "It is scored on price CRPS only here, but live scoring counts such a prediction as failed.",
        RuntimeWarning,
        stacklevel=3,
    )
