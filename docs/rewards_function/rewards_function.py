"""
Reward Function Demonstration

This script demonstrates how rewards are distributed on the crunch-synth leaderboard.

Process:
- Rewards are distributed every week.
- Total reward pool is fixed.
- Rewards are split across prediction horizons (1H / 24H).
- Only the best model per participant is rewarded.
- Only models beating the benchmark receive rewards.
- Rewards follow an exponential distribution based on rank.
"""

import pandas as pd
import numpy as np


# Utility Functions
def keep_best_model_per_player(df: pd.DataFrame, col_score="anchor", col_rank="rank") -> pd.DataFrame:
    """Keep only the best model per participant"""
    
    # sort so best anchor is first per player
    df_sorted = df.sort_values(col_score, ascending=False)
    
    # keep best model per player
    df_best = df_sorted.groupby("player_name", as_index=False).first()

    df_best = df_best.sort_values(col_score, ascending=False)

    # Recompute rank (1 = best)
    df_best[col_rank] = df_best[col_score].rank(
        method="first", ascending=False
    ).fillna(len(df_best)).astype(int)
    
    return df_best


def explode_ranks_by_horizon(df: pd.DataFrame, config_rewards) -> pd.DataFrame:
    """Convert leaderboard horizon dictionary into a flat table"""
    rows = []

    for _, row in df.iterrows():
        model_id = row["model_id"]
        player = row["player_name"]
        model = row["model_name"]

        ranks_dict = row["ranks_by_horizon"]

        for horizon, value in ranks_dict.items():

            if value["score"] and value["score"][config_rewards["col_score"]]:
                rows.append({
                    "model_id": model_id,
                    "player_name": player,
                    "model_name": model,
                    "horizon": horizon,
                    "rank": value["rank"],
                    config_rewards["col_score"]: value["score"][config_rewards["col_score"]],
                })

    return pd.DataFrame(rows)


# Reward Function
def compute_exponential_rewards(
    df: pd.DataFrame,
    config_rewards: dict,
) -> pd.DataFrame:
    """
    Exponential reward function with benchmark cutoff.

    - Fixed reward pool is allocated for each payout period.
    - The top 10 participants receive 100% of the pot.
    - Only models outperforming the benchmark model (synth:benchmarktracker) are rewarded.
    - Models with a score below or equal to the benchmark at payout time are included in
      calculations but receive no payout, leaving any fraction of the pool tied to lower-performing models undistributed.

    # Payout logic:
    - All valid anchor scores participate in normalization
    - Only top k models strictly above benchmark receive payout
    - Undistributed mass is left unallocated
    """

    df = df.copy()

    # keep only valid anchors
    valid = df[config_rewards["col_score"]].notna()
    df = df[valid].reset_index(drop=True)

    # for one participant, we reward only the best of the 2 models
    df = keep_best_model_per_player(df, col_score=config_rewards["col_score"], col_rank=config_rewards["col_rank"])

    # Find benchmark model
    benchmark_mask = (
        (df["player_name"] == config_rewards["benchmark_player_name"]) &
        (df["model_name"] == config_rewards["benchmark_model_name"])
    )

    if not benchmark_mask.any():
        raise ValueError("Benchmark model not found")

    # Extract the benchmark anchor score
    benchmark_score = df.loc[benchmark_mask, config_rewards["col_score"]].iloc[0]

    # Keep only models above benchmark
    df = df[df[config_rewards["col_score"]] > benchmark_score]

    # keep only top K participants
    if config_rewards["apply_top_k_participants"]:
        df = df[df[config_rewards["col_rank"]] <= config_rewards["top_k_participants"]]

    # Cumalitive reward distribution is exponential based on rank
    # exponential weights
    df["weight"] = np.exp(config_rewards["alpha"] / df[config_rewards["col_rank"]])

    # Normalize over all K positions (1..K), not just those present
    k = config_rewards["top_k_participants"] if config_rewards["apply_top_k_participants"] else len(df)
    total_weight = sum(np.exp(config_rewards["alpha"] / r) for r in range(1, k + 1))

    # Raw reward fraction before eligibility filtering
    df["reward_fraction"] = df["weight"] / total_weight

    # Distribute reward pool
    # Convert reward fractions into absolute payouts
    df["rewards"] = df["reward_fraction"] * config_rewards["week_reward_pool"]

    return df


def compute_rewards_per_horizon(df_leaderboard, config_rewards):
    """ Compute Rewards per Horizon """

    # explode ranks_by_horizon
    df_long = explode_ranks_by_horizon(df_leaderboard, config_rewards)

    results = []

    # split total pool equally by horizon
    horizons = df_long["horizon"].unique()
    n_horizons = len(horizons)

    horizon_pool = config_rewards["week_reward_pool"] / n_horizons

    # loop per horizon
    for horizon, df_group in df_long.groupby("horizon"):

        config_local = config_rewards.copy()
        config_local["week_reward_pool"] = horizon_pool

        df_rewards = compute_exponential_rewards(
            df_group,
            config_local
        )

        df_rewards["horizon"] = horizon

        results.append(df_rewards)

    return pd.concat(results, ignore_index=True)


# Configs

# - $30K over the first 4 months followed by real mining rewards from Synth Miners (currently up to 50K / months)
# - Rewards are distributed based on Anchor score every 7 days (payout on Monday)
# - Fixed reward pool is allocated for each payout period.
# - Rewards are split per horizon (1H and 24H separated 50/50)
# - For each horizon, only the top 10 participants are rewarded (for one participant, we reward only the best of the 2 models).
# - Only models outperforming the benchmark model (synth:benchmarktracker) are rewarded.
# - Models with a score below or equal to the benchmark at payout time are included in calculations but receive no payout, leaving any fraction of the pool tied to lower-performing models undistributed.

config_rewards = {
    # the benchmark model
    "benchmark_player_name": "synth",
    "benchmark_model_name": "benchmarktracker",

    # Column score
    "col_score": "anchor",
    # Column rank
    "col_rank": "rank",

    # weight computation
    # "score" → exp(alpha * score)
    # "rank"  → exp(alpha / rank)
    "weight_mode": "rank",   # "score" or "rank"

    # Controls steepness of the exponential curve (higher = more top-heavy)
    "alpha": 1.0,

    # Whether to restrict rewards to the top K ranked participants
    "apply_top_k_participants": True,
    "top_k_participants": 10,

    # Total reward pool to distribute for the week
    "week_reward_pool": 2500
}


# Example Leaderboard Data
df_leaderboard = pd.DataFrame([
    {
        "model_id": "m1",
        "player_name": "A",
        "model_name": "A1",
        "ranks_by_horizon": {
            3600: {"rank": 1, "score": {"anchor": 0.72}},
            86400: {"rank": 2, "score": {"anchor": 0.68}},
        },
    },
    {
        "model_id": "m2",
        "player_name": "B",
        "model_name": "B2",
        "ranks_by_horizon": {
            3600: {"rank": 4, "score": {"anchor": 0.60}},
            86400: {"rank": 1, "score": {"anchor": 0.71}},
        },
    },
    {
        "model_id": "m3",
        "player_name": "C",
        "model_name": "C1",
        "ranks_by_horizon": {
            3600: {"rank": 3, "score": {"anchor": 0.66}},
            86400: {"rank": 3, "score": {"anchor": 0.64}},
        },
    },
    {
        "model_id": "m4",
        "player_name": "synth",
        "model_name": "benchmarktracker",
        "ranks_by_horizon": {
            3600: {"rank": 2, "score": {"anchor": 0.70}},
            86400: {"rank": 4, "score": {"anchor": 0.60}},
        },
    },
])

# Run Example

if __name__ == "__main__":

    rewards = compute_rewards_per_horizon(
        df_leaderboard,
        config_rewards
    )

    print("\nReward Distribution\n")
    print(rewards[[
        "player_name",
        "model_name",
        "horizon",
        "rank",
        "anchor",
        "rewards"
    ]])