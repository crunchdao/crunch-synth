"""
Constants used across the condor game package.
"""

# ------------------------------------------------------------------
# CRPS integration bounds configuration
#
# crps_bounds["t"][asset] defines the BASE integration half-width
# used when computing the CRPS integral:
#
#     CRPS = ∫ (F(z) - 1[z ≥ x])² dz ,  z ∈ [t_min, t_max]
#
# This value represents a reference maximum price-move scale for the
# asset at the given horizon. It is a truncation range ensuring
# enough mass is covered for meaningful CRPS evaluation.
CRPS_BOUNDS = {
    "base_step": 300,
    "t":{
        "BTC": 1500,
        "SOL": 4,
        "ETH": 80,
        "XAU": 28,
    }
}
# ------------------------------------------------------------------