import math

import pytest

from crunch_synth.constants import ASSET_WEIGHTS, CRPS_BOUNDS, SUPPORTED_ASSETS
from crunch_synth.tracker_evaluator import crps_integral


def gaussian_crps(x, loc, scale):
    """Closed form CRPS of N(loc, scale) at x."""
    z = (x - loc) / scale
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return scale * (z * (2 * cdf - 1) + 2 * pdf - 1 / math.sqrt(math.pi))


def norm(loc, scale):
    return {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": scale}}


def test_every_supported_asset_can_be_scored():
    for asset in SUPPORTED_ASSETS:
        assert asset in CRPS_BOUNDS["t"], asset
        assert asset in ASSET_WEIGHTS, asset


@pytest.mark.parametrize("x", [-1.5, 0.0, 0.4, 2.0])
def test_crps_integral_matches_closed_form(x):
    # density inside the integration bounds: the numerical CRPS is the exact one, up to the
    # discretization of the grid (same method and number of points as the live scoring)
    assert crps_integral(norm(0.1, 0.5), x, -4, 4) == pytest.approx(gaussian_crps(x, 0.1, 0.5), rel=1e-2)


def test_crps_integral_renormalizes_truncated_densities():
    # half of the mass lies outside the bounds: like the live scoring, the PDF is renormalized
    # over [t_min, t_max], so the score is the one of the truncated distribution (finite, CDF ending at 1)
    wide = crps_integral(norm(0.0, 5.0), 0.0, -2, 2)
    assert 0.0 < wide < 2.0
    mixture = {"type": "mixture", "components": [
        {"weight": 0.5, "density": norm(-0.2, 0.3)},
        {"weight": 0.5, "density": norm(0.3, 0.4)},
    ]}
    both = crps_integral(mixture, 0.1, -10, 10)
    assert 0.0 < both < min(gaussian_crps(0.1, -0.2, 0.3), gaussian_crps(0.1, 0.3, 0.4)) + 0.2
