from condorgame.constants import MAX_DISTRIBUTION_COMPONENTS


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
