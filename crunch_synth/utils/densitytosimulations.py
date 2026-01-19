import numpy as np
from datetime import datetime, timedelta
from scipy import stats as st
from statistics import NormalDist


def simulate_points(
    density_dict: dict,
    current_point: float = 0.0,
    num_simulations: int = 1,
    max_depth: int = 3,
    current_depth: int = 0,
    max_mixtures: int = 5,
    mixture_count: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate 'next point' samples based on a density specification.
    Returns both the sampled values and the 'loc' (mean) values used.

    Supports the same formats as density_pdf:
      1) Scipy distribution
      2) Statistics (NormalDist)
      3) Builtin distribution (via scipy)
      4) Mixture distribution (recursive)

    Parameters
    ----------
    density_dict : dict
        Density specification dictionary.
    current_point : float
        Current point used as a reference (optional).
    num_simulations : int
        Number of samples to draw.
    max_depth : int
        Maximum recursion depth for mixtures.
    current_depth : int
        Current recursion level (internal usage).
    max_mixtures : int
        Maximum total number of mixtures allowed.
    mixture_count : int
        Current count of mixtures encountered.

    Returns
    -------
    samples : np.ndarray
        Simulated values.
    locs : np.ndarray
        Loc (mean) values used in each simulation.
    """

    # --- Check recursion depth
    if current_depth > max_depth:
        raise RecursionError(
            f"Exceeded maximum recursion depth of {max_depth}. "
            "Possible nested mixtures beyond allowed depth."
        )

    dist_type = density_dict.get("type")

    # --- 1) Mixture distribution
    if dist_type == "mixture":
        mixture_count += 1
        if mixture_count > max_mixtures:
            raise ValueError(f"Exceeded maximum mixture count {max_mixtures}")

        components = density_dict["components"]
        weights = np.array([abs(c["weight"]) for c in components], dtype=float)
        weights /= weights.sum()

        # Choose which component each sample comes from
        chosen_idx = np.random.choice(len(components), size=num_simulations, p=weights)

        samples = np.empty(num_simulations)
        locs = np.empty(num_simulations)

        # --- Vectorize: process all samples for each component in a batch
        for j, comp in enumerate(components):
            idx = np.where(chosen_idx == j)[0]
            if len(idx) == 0:
                continue
            sub_spec = comp["density"]
            sub_samples, sub_locs = simulate_points(
                sub_spec,
                current_point=current_point,
                num_simulations=len(idx),
                max_depth=max_depth,
                current_depth=current_depth + 1,
                max_mixtures=max_mixtures,
                mixture_count=mixture_count,
            )
            samples[idx] = sub_samples
            locs[idx] = sub_locs

        return samples, locs

    # --- 2) Scipy distribution
    elif dist_type == "scipy":
        dist_name = density_dict["name"]
        params = density_dict["params"]
        dist_class = getattr(st, dist_name, None)
        if dist_class is None:
            raise ValueError(f"Unknown scipy distribution '{dist_name}'.")
        dist_obj = dist_class(**params)

        loc_val = params.get("loc", 0.0)
        samples = dist_obj.rvs(size=num_simulations)
        locs = np.full(num_simulations, loc_val)
        return samples, locs

    # --- 3) Statistics distribution
    elif dist_type == "statistics":
        bname = density_dict["name"]
        bparams = density_dict["params"]
        if bname == "normal":
            mu = bparams.get("mu", bparams.get("loc", 0.0))
            sigma = bparams.get("sigma", bparams.get("scale", 1.0))
            dist_obj = NormalDist(mu=mu, sigma=sigma)
            samples = np.array(dist_obj.samples(num_simulations))
            locs = np.full(num_simulations, mu)
            return samples, locs
        else:
            raise NotImplementedError(f"Unsupported statistics distribution '{bname}'.")

    # --- 4) Builtin (using scipy fallback)
    elif dist_type == "builtin":
        dist_name = density_dict["name"]
        params = density_dict["params"]
        dist_class = getattr(st, dist_name, None)
        if dist_class is None:
            raise ValueError(f"Unknown builtin distribution '{dist_name}'.")
        dist_obj = dist_class(**params)
        samples = dist_obj.rvs(size=num_simulations)
        locs = np.full(num_simulations, params.get("loc", 0.0))
        return samples, locs

    else:
        raise ValueError(f"Unknown or missing 'type' in density_dict: {density_dict}")
    

def simulate_paths(
    mixture_specs: list,
    start_point: float,
    num_paths: int = 100,
    step_minutes: int = 5,
    start_time: datetime | None = None,
    mode: str = "incremental",  # "absolute", "incremental", "relative"
    quantile_range: list = [0.05, 0.95],
    **simulate_kwargs,
):
    """
    Simulate multiple paths forward given a list of mixture specs for each step.

    Parameters
    ----------
    mixture_specs : list of dict
        Each dict is a valid density spec (mixture, scipy, builtin...), one per step.
    start_point : float
        Initial value at time step 0.
    num_paths : int
        Number of independent paths to simulate.
    step_minutes : int
        Minutes between consecutive steps.
    start_time : datetime, optional
        If provided, returns timestamps instead of integer steps.
    mode : {"absolute", "incremental", "relative"}, default="incremental"
        Determines how simulated values are applied:
        - "absolute" : draw represents an absolute target value
        - "incremental" : draw represents a change (Δ) added to previous value
        - "relative" : draw represents a fractional change
        - "direct" : draw represents the next absolute value directly
    quantile_range : list[float, float], default=[0.05, 0.95]
        Quantile interval to compute for uncertainty bands.
    **simulate_kwargs :
        Extra arguments to pass to simulate_points() (e.g., max_depth, max_mixtures)

    Returns
    -------
    dict
        Dictionary containing:
            "times"       : list of timestamps or integer step indices
            "paths"       : np.ndarray, shape (num_paths, num_steps + 1)
            "mean"        : np.ndarray, mean path value at each step
            "q_low_paths"  : np.ndarray, lower quantile path (quantile_range[0])
            "q_high_paths" : np.ndarray, upper quantile path (quantile_range[1])
    """
    num_steps = len(mixture_specs)
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = start_point

    current_points = np.full(num_paths, start_point)

    for t, spec in enumerate(mixture_specs):
        # Simulate all paths for this step
        draws, locs = simulate_points(spec, num_simulations=num_paths, **simulate_kwargs)

        if mode == "absolute":
            # The mixture gives absolute value around loc, so use deviation from loc
            increment = draws - locs
            next_values = current_points + increment
        elif mode == "incremental":
            # The mixture directly represents a change (Δ)
            next_values = current_points + draws
        elif mode == "relative":
            # The mixture represents a fractional change
            next_values = current_points * (1 + draws)
        elif mode == "direct":
            # The mixture draws represent the next absolute value directly
            next_values = draws
        elif mode == "point":
            next_values = draws
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'absolute', 'incremental', or 'relative'.")

        paths[:, t + 1] = next_values
        current_points = next_values

    # Build timestamps if requested
    if start_time is not None:
        times = [start_time + timedelta(minutes=step_minutes * i)
                 for i in range(num_steps + 1)]
    else:
        times = list(range(num_steps + 1))

    # --- Compute per-step statistics
    mean_path  = np.mean(paths, axis=0)
    q_low = np.quantile(paths, quantile_range[0], axis=0)
    q_high = np.quantile(paths, quantile_range[1], axis=0)

    return {"times": times, "paths": paths, "mean": mean_path, "q_low_paths": q_low, "q_high_paths": q_high, "quantile_range": quantile_range}


if __name__ == "__main__":

    # Example mixture with one scipy normal and one builtin normal
    mixture_spec = {
        "type": "mixture",
        "components": [
            {
                "density": {
                    "type": "scipy",
                    "name": "norm",
                    "params": {"loc": 0, "scale": 1}
                },
                "weight": 0.1
            },
            {
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {"loc": 0, "scale": 1}
                },
                "weight": 0.8
            },
            {
                "density": {
                    "type": "builtin",
                    "name": "t",
                    "params": {"df":4, "loc": 0, "scale": 1}
                },
                "weight": 0.5
            },
            {
                "density": {
                    "type": "builtin",
                    "name": "pareto",
                    "params": {"b":4, "loc": 10, "scale": 1}
                },
                "weight": 0.5
            }
        ]
    }

    # mixture_spec = {
    #     "type": "builtin",
    #     "name": "pareto",
    #     "params": {"b":4, "loc": 10, "scale": 1}
    # }

    current_point = 0.0
    simulated, locs = simulate_points(mixture_spec, current_point, num_simulations=100)

    print(f"Mean of simulated points: {simulated.mean():.4f}")
    print(f"Std of simulated points: {simulated.std():.4f}")


    #####################################################################################################

    start_point = 100_000
    dict_mode = {
        "absolute": {"loc": start_point},
        "direct": {"loc": start_point},
        "point": {"loc": start_point},
        "incremental": {"loc": 0.0},
    }
    MODE = "incremental"

    # --- 12 mixture specs, each slightly different
    mixture_specs = []
    for i in range(12):
        mixture_specs.append({
            "type": "mixture",
            "components": [
                {
                    "density": {
                        "type": "scipy",
                        "name": "norm",
                        "params": {"loc": dict_mode[MODE]["loc"], "scale": 5}
                    },
                    "weight": 1.0
                },
            # {
            #     "density": {
            #         "type": "builtin",
            #         "name": "t",
            #         "params": {"df":4, "loc": dict_mode[MODE]["loc"], "scale": 1}
            #     },
            #     "weight": 0.5
            # },
            # {
            #     "density": {
            #         "type": "builtin",
            #         "name": "pareto",
            #         "params": {"b":4, "loc": dict_mode[MODE]["loc"], "scale": 1}
            #     },
            #     "weight": 0.5
            # }
            ]
        })

    # --- Simulate multiple paths starting at 0.0
    result = simulate_paths(
        mixture_specs,
        start_point=start_point,
        num_paths=1000,
        step_minutes=5,
        start_time=datetime.now(),
        max_depth=3,
        max_mixtures=5,
        mode=MODE
    )

    times = result["times"]
    paths = result["paths"]
    mean_vals = result["mean"]
    q_low_paths = result["q_low_paths"]
    q_high_paths = result["q_high_paths"]

    print(f"Simulated {paths.shape[0]} paths for {len(times)-1} steps.")
    print(f"Example path:\n{paths[0]}")

    # Optional plot
    import matplotlib.pyplot as plt
    if MODE != "point":
        for p in paths:
            plt.plot(times, p, color="gray", alpha=0.2, linewidth=0.7)
    else:
        num_paths, num_steps = paths.shape
        for j in range(num_steps):
            # Scatter all simulated points for step j
            plt.scatter(
                np.full(num_paths, times[j]),
                paths[:, j],
                s=8,
                alpha=0.15,
                color="gray"
            )

    # Mean line
    plt.plot(times, mean_vals, color="blue", label="Mean", linewidth=2)

    # Shaded quantile band
    plt.fill_between(times, q_low_paths, q_high_paths, color="blue", alpha=0.2, label="5–95% range")

    plt.title("Simulated 5-minute paths for next 1 hour")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()