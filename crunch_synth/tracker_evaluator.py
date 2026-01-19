import numpy as np
import json
import os
from collections import defaultdict, deque
from datetime import datetime, timezone

from densitypdf import density_pdf

from crunch_synth.prices import Asset
from crunch_synth.quarantine import Quarantine, QuarantineGroup
from crunch_synth.tracker import TrackerBase, PriceData
from crunch_synth.constants import CRPS_BOUNDS


class TrackerEvaluator:
    """
    Evaluates a tracker by comparing its predictions to realized asset returns
    using scoring: CRPS (Continuous Ranked Probability Score).

    This class also handles:
    - Quarantine: delaying scoring until sufficient future data is available.
    - Rolling evaluation windows for recent CRPS computation.
    - Storing timestamped scores for per-asset and global evaluation.
    """

    def __init__(self, tracker: TrackerBase, score_window_size: int = 100):
        """
        Parameters
        ----------
        tracker : TrackerBase
            The tracker instance to evaluate.
        score_window_size : int, optional
            The number of most recent scores to retain for computing a
            rolling recent CRPS per asset.
        """

        super().__init__()
        self.tracker = tracker
        self.quarantine_group = QuarantineGroup()
        # Store (timestamp, score)
        # Store per-asset: {asset → [(timestamp, score), ...]}
        self.scores = defaultdict(list)
        # Store recent scores per asset: {asset → deque([(timestamp, score)])}
        self.latest_scores = defaultdict(lambda: deque(maxlen=score_window_size))

    def tick(self, data: PriceData):
        """ Feed new market data to the tracker. 
        data : PriceData
            Mapping from asset symbol to a list of (timestamp, price) pairs.
        """
        self.tracker.tick(data)

    def predict(self, asset: Asset, horizon: int, steps: list[int]):
        """
        Request multi-resolution predictions, place them
        in quarantine and evaluate them when realized.

        Returns: list or None
            List of quarantined predictions that were scored (or None if none ready yet).
        """
        predictions = self.tracker.predict_all(asset, horizon, steps)

        # Validate predictions
        for step in steps:
            if step > horizon:
                continue
            expected_len = horizon // step
            if step not in predictions:
                raise ValueError(f"Missing predictions for step {step}")
            if len(predictions[step]) != expected_len:
                raise ValueError(
                    f"Prediction length mismatch for step {step}: "
                    f"{len(predictions[step])} != {expected_len}"
                )

        ts, _ = self.tracker.prices.get_last_price(asset)
        self.quarantine_group.add(asset, ts, predictions, horizon, steps)
        quarantines_predictions = self.quarantine_group.pop(asset, ts)

        if not quarantines_predictions:
            return

        # Compute score for quarantined predictions
        score = self._score_quarantines(asset, quarantines_predictions)
        
        # Store timestamped scores
        self.scores[asset].append((ts, score))
        self.latest_scores[asset].append((ts, score))  # Maintain a rolling window of recent scores

        return quarantines_predictions
    
    def _score_quarantines(self, asset: Asset, quarantines_predictions: list):
        """
        Compute CRPS scores for all quarantined predictions of a given asset.

        Parameters
        ----------
        asset : Asset
        quarantines_predictions : list
            Each element is a tuple: (quarantine_ts, predictions, steps)
            In most cases, only a single quarantine is resolved per asset
            at a given evaluation time.

        Returns
        -------
        float
            Mean normalized CRPS score across all quarantines.
        """

        quarantines_scores = []

        # ------------------------------------------------------------------
        # Iterate over all quarantines
        # quar_ts             : reference timestamp of the quarantine
        # quar_predictions    : dict of predictions per step (5m, 1h, 6h, ...)
        # quar_steps    : mapping {step_name -> step_seconds}
        # ------------------------------------------------------------------
        for quar_ts, quar_predictions, quar_steps in quarantines_predictions:

            total_score = 0.0

            # --------------------------------------------------------------
            # Score predictions at each temporal resolution independently
            # (e.g. 5min, 1hour, 6hour, 24hour)
            # --------------------------------------------------------------
            for step in quar_steps:

                preds = quar_predictions[step]

                # Get timestamp of the first prediction step
                ts_rolling = quar_ts - step * (len(preds) - 1)

                scores_step = []

                # Evaluate each forecasted increment for this step
                for i in range(len(preds)):

                    current_price_data  = self.tracker.prices.get_closest_price(asset, ts_rolling)
                    previous_price_data = self.tracker.prices.get_closest_price(asset, ts_rolling - step)

                    ts_rolling += step

                    if not current_price_data or not previous_price_data:
                        continue

                    ts_current, price_current = current_price_data
                    ts_prev, price_prev = previous_price_data

                    if ts_current != ts_prev:
                        # Observed price increment over this step
                        delta = (price_current - price_prev)

                        # Step-dependent scaling coefficient for CRPS bounds
                        # K adjusts the CRPS integration range to the time resolution of the forecast
                        # For the base step (and finer), no scaling is applied
                        K = np.sqrt(step / CRPS_BOUNDS["base_step"]) if step > CRPS_BOUNDS["base_step"] else 1

                        crps_value = crps_integral(
                            density_dict=preds[i],
                            x=delta,
                            t_min=-K * CRPS_BOUNDS["t"][asset],
                            t_max=K * CRPS_BOUNDS["t"][asset],
                            num_points=CRPS_BOUNDS["num_points"]
                        )
                        scores_step.append(crps_value)

                # Accumulate score across all steps of this resolution
                total_score += np.sum(scores_step)

            # Normalize by asset-specific scale (keep scores comparable across asset)
            total_score = total_score / CRPS_BOUNDS["t"][asset]
            quarantines_scores.append(total_score)

        return np.mean(quarantines_scores)

    
    def recent_score_asset(self, asset: Asset):
        """
        Return the mean crps score of the most recent `score_window_size` scores.
        """
        if not self.latest_scores[asset]:
            return 0.0
        values = [s for _, s in self.latest_scores[asset]]
        return float(np.mean(values))
    
    def overall_score_asset(self, asset: Asset):
        """
        Return the mean crps score over all recorded scores.
        """
        if not self.scores[asset]:
            return 0.0
        values = [s for _, s in self.scores[asset]]
        return float(np.mean(values))

    def overall_score(self):
        """
        Return the mean crps score across all assets together.
        """
        all_scores = []

        for asset_scores in self.scores.values():
            all_scores.extend(s for _, s in asset_scores)

        if not all_scores:
            return 0.0

        return float(np.mean(all_scores))

    
    def to_json(self, horizon: int, steps: list[int], interval: int, base_dir="results"):
        """Save crps scores and metadata to a JSON file."""
        tracker_name = self.tracker.__class__.__name__

        # Serialize scores
        assets_json = {}
        for asset, records in self.scores.items():
            assets_json[asset] = [
                    {"ts": ts, "score": float(score)}
                    for ts, score in records
                ]

        start_ts = min(ts for asset in self.scores for ts, _ in self.scores[asset])
        end_ts   = max(ts for asset in self.scores for ts, _ in self.scores[asset])

        data = {
            "tracker": tracker_name,
            "assets": list(self.scores.keys()),
            "period": {
                "start": start_ts,
                "end": end_ts,
            },
            "horizon": horizon,
            "steps": steps,
            "interval": interval,
            "asset_scores": assets_json,
        }
        
        # Format directory name: "results/2025-02-05T12-00-00_to_2025-02-12T12-00-00/"
        def fmt(ts):
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")

        directory = os.path.join(base_dir, f"{fmt(start_ts)}_to_{fmt(end_ts)}")
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{tracker_name}_h{horizon}.json")

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        print(f"[✔] Tracker results saved to {path}")

        return directory


# NumPy 2.0 removed np.trapz → replaced by np.trapezoid
if hasattr(np, "trapezoid"):
    trapezoid = np.trapezoid
else:
    trapezoid = np.trapz
    
def crps_integral(density_dict, x, t_min=-4000, t_max=4000, num_points=256):
    """
    CRPS score (Integrated Quadratic Score) using:
    - single PDF evaluation per grid point
    - cumulative sum to get CDF

    CRPS quantifies the accuracy of probabilistic forecasts by measuring the squared distance 
    between the forecast CDF and the observed indicator function.
    """
    ts = np.linspace(t_min, t_max, num_points)
    dt = ts[1] - ts[0]

    # Vectorized PDF computation
    pdfs = np.array([density_pdf(density_dict, t) for t in ts], dtype=float)

    # Build CDF by cumulative integration
    cdfs = np.cumsum(pdfs) * dt
    cdfs = np.clip(cdfs, 0.0, 1.0)

    # Indicator
    indicators = (ts >= x).astype(float)

    # Integrate squared error
    integrand = (cdfs - indicators)**2
    return float(trapezoid(integrand, ts))