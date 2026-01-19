from math import log10, floor
from crunch_synth.constants import MAX_DISTRIBUTION_COMPONENTS


def count_distribution_components(dist: dict) -> int:
    """
    Recursively count the number of leaf components in a predictive distribution.

    - A non-mixture distribution counts as 1 component.
    - A mixture distribution counts as the sum of its leaf components.
    - Nested mixtures are fully expanded.

    """
    if dist.get("type") != "mixture":
        return 1

    components = dist.get("components", [])
    total = 0

    for comp in components:
        density = comp.get("density", {})
        total += count_distribution_components(density)

    return total


def validate_distribution(dist: dict):
    """
    Validate structural constraints on a predictive distribution.

    Constraints
    -----------
    - The total number of leaf components (including nested mixtures)
      must not exceed `MAX_DISTRIBUTION_COMPONENTS`.

    This limit is enforced to ensure:
        - Fast CRPS evaluation
        - Bounded memory usage

    The limit may be increased in the future.

    Raises
    ------
    ValueError
        If the distribution violates the component limit.
    """
    n_components = count_distribution_components(dist)

    if n_components > MAX_DISTRIBUTION_COMPONENTS:
        raise ValueError(
            f"Distribution contains {n_components} total components "
            f"(including nested mixtures), but the maximum allowed is "
            f"{MAX_DISTRIBUTION_COMPONENTS}."
        )


def round_significant(value: float, sig_digits: int = 10) -> float:
    """
    Round a number to a specified number of significant digits.

    Parameters
    ----------
    value : float
        The number to round.
    sig_digits : int
        Number of significant digits to retain.

    Returns
    -------
    float
        The value rounded to `sig_digits` significant digits.
    """
    if value == 0.0:
        return 0.0

    order = floor(log10(abs(value)))
    decimals = sig_digits - order - 1
    return round(value, decimals)


def round_distribution_digits(dist: dict, digits: int = 10) -> dict:
    """ Round weights and params values of a distribution 
        to a specified number of significant digits. """
    if dist.get("type") == "mixture":
        for comp in dist["components"]:
            comp["weight"] = round_significant(comp["weight"], digits)
            round_distribution_digits(comp["density"], digits)

    else:
        params = dist.get("params", {})
        for k, v in params.items():
            if isinstance(v, (int, float)):
                params[k] = round_significant(v, digits)

    return dist