"""
Constants used across the Crunch-Synth game package.
"""

# Supported assets
SUPPORTED_ASSETS = [
    "BTC",      # BTC/USD
    "ETH",      # ETH/USD
    "XAUT",     # XAUT/USD
    "SOL",      # SOL/USD
    "SPYX",     # SPYX/USD
    "NVDAX",    # NVDAX/USD
    "TSLAX",    # TSLAX/USD
    "AAPLX",    # AAPLX/USD
    "GOOGLX",   # GOOGLX/USD
    "XRP",      # XRP/USD
    "HYPE",     # HYPE/USD
    "WTIOIL",   # WTIOIL/USDC
]

# ------------------------------------------------------------------
# Forecast configuration (in seconds)
# Each profile defines:
# - a forecast horizon
# - a set of step resolutions
# - how often predictions are triggered

FORECAST_PROFILES = {
    "24h": {
        "horizon": 24 * 3600,  # 24 hours
        # Multi-resolution forecast grid
        # All forecasts span the same horizon but differ in temporal granularity.
        "steps": [
                    300,       # "5min"
                    3600,      # "1hour"
                    6 * 3600,  # "6hour"
                    24 * 3600, # "24hour"
        ],
        "interval": 3600,  # triggered every hour
    },
    "1h": {
        "horizon": 1 * 3600,  # 1 hour
        "steps": [
                    60,       # "1min"
                    60 * 5,   # "5min"
                    60 * 15,  # "15min"
                    60 * 30,  # "30min"
                    3600,     # "1hour"
        ],
        "interval": 60 * 12,  # triggered every 12 minutes
    },
}
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# CRPS configuration
#
# CRPS is computed as:
#
#     CRPS = ∫ (F(z) − 1[z ≥ x])² dz ,  z ∈ [t_min, t_max]
#
# where F(z) is the forecast CDF and x is the realized return.
#
# - `base_step` (seconds) defines the reference forecast resolution.
#   CRPS integration bounds are scaled relative to this step so that
#   scores remain comparable across different temporal resolutions.
#
# - `t[asset]` specifies the base half-width of the CRPS integration
#   range for each asset at the reference resolution. This value
#   represents a typical maximum price move to cover most of the
#   predictive mass while keeping integration finite and stable.
#
# - `num_points` is the number of discretization points used to 
#   numerically approximate the CRPS integral. Higher values improve 
#   accuracy but increase computation time.
#
# For steps larger than `base_step`, integration bounds are expanded
# by sqrt(step / base_step) to reflect increased uncertainty over
# longer time intervals.
#
# Check `crps_integral` in tracker_evaluator.py for more information
CRPS_BOUNDS = {
    "base_step": 300,
    "t":{
        "BTC": 1500,
        "SOL": 4,
        "ETH": 80,
        "XAUT": 33,

        "SPYX": 3.2,
        "NVDAX": 2.3,
        "TSLAX": 5.9,
        "AAPLX": 2.1,
        "GOOGLX": 3.4,

        "XRP": 0.04,
        "HYPE": 1.1,
        "WTIOIL": 1.2,
    },
    "num_points": 256
}
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Per-asset weighting coefficients for score aggregation.
# Higher weight = asset contributes more to the overall score.
# BTC is the baseline (1.0).
ASSET_WEIGHTS = {
    "BTC": 1.0,
    "ETH": 0.7064366394033871,
    "XAUT": 1.7370922597118699,
    "SOL": 0.6310037175639559,
    "SPYX": 3.437935601155441,
    "NVDAX": 1.6028217601617174,
    "TSLAX": 1.6068755936957768,
    "AAPLX": 2.0916380815843123,
    "GOOGLX": 1.6827392777257926,
    "XRP": 0.5658394110809131,
    "HYPE": 0.4784547133706857,
    "WTIOIL": 0.8475062847978935,
}
# ------------------------------------------------------------------

# Maximum number of mixture components allowed per predictive distribution
# The limit may be increased in the future.
MAX_DISTRIBUTION_COMPONENTS = 3
