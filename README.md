# Synth Game

Synth Game is a real-time probabilistic forecasting challenge hosted by CrunchDAO at [crunchdao.com](https://crunchdao.com)

The goal is to anticipate how asset prices will evolve by providing not a single forecasted value, but a full probability distribution over the future price change at multiple forecast horizons and steps.

**The current crypto assets to model are:**
- **Bitcoin (BTC)**
- **Ethereum (ETH)**
- **Solana (SOL)**
- **Tether Gold (XAUT)**
- **SP500 tokenized ETF (SPYX)**
- **NVIDIA tokenized stock (NVDAX)**
- **Tesla tokenized stock (TSLAX)**
- **Apple tokenized stock (AAPLX)**
- **Alphabet tokenized stock (GOOGLX)**

## Install

```bash
pip install crunch-synth
```

## What You Must Predict

Trackers must predict the **probability distribution of price changes**, defined as:

$$
r_{t,k} = P_t - P_{t-k}
$$

For each defined step **$k$** (e.g., 5 minutes, 1 hour, …), your tracker must return a full **probability density function (PDF)** over the future price change **$r_{t,k}$**.


## Visualize the challenge

The Synth game is evaluated on **incremental return predictions**, not raw prices.  
Incremental returns capture the *relative* change in price and produce a stationary series that is easier to model and compare across assets.

Below is an example of a **density forecast over incremental returns for the next 24h at 5-minute intervals**:

![](docs/density_prediction_return_24h.png)

Below is a minimal example showing what your tracker might return:
```python
>>> model.predict(asset="SOL", horizon=86400, step=300)
[
    {
        "step": (k + 1) * step,
        "prediction": {
            "type": "builtin",
            "name": "norm",
            "params": {
                "loc": -0.01,       # mean return
                "scale": 0.4        # standard deviation of return
            }
        }
    }
    for k in range(0, horizon // step)
]
```

Here is the **return forecast mapped into price space**:

![](docs/density_prediction_price_24h.png)


## Create your Tracker

A **tracker** is a model that processes real-time asset data to **predict future price changes**. It uses past prices to generate a probabilistic forecast of incremental returns. You can use the data provided by the challenge or any other datasets to improve your predictions.

It operates incrementally: prices are pushed to the tracker as they arrive and predictions are requested at specific times by the framework.

**To create your tracker, you need to define a class that implements the `TrackerBase` interface, which already handles:**

- price storage and alignment via `PriceStore`
- multi-resolution forecasting through `predict_all()`

As a participant, you only need to implement **one method**: `predict()`.

1. **Price data handling (already provided)**  

    Each tracker instance contains a `PriceStore` (`self.prices`) that:
    - stores recent historical prices per asset
    - maintains a rolling time window (30 days)
    - provides convenient accessors such as:
        - `get_last_price()`
        - `get_prices(asset, days, resolution)`
        - `get_closest_price(asset, timestamp)`

    The framework automatically updates the `PriceStore` by calling `tick(self, data: PriceData)` before any prediction request.

    Data example:
     ```python
      data = {
            "BTC": [(timestamp1, price1), (timestamp2, price2)],
            "SOL": [(timestamp1, price1)],
        }
     ```

    When it's called:
    - Typically every minute or when new data is available
    - Before any prediction request
    - Can be called multiple times before a predict

2. **Required method: `predict(self, asset: str, horizon: int, step: int)`**  
    This is the **only method you must implement.**

    It must return a sequence of **predictive density distributions** for the **incremental price change** of an asset:

    - Forecast horizon: horizon seconds into the future
    - Temporal resolution: one density every step seconds
    - Output length: horizon // step
    
    Each density prediction must comply with the [density_pdf](https://github.com/microprediction/densitypdf/blob/main/densitypdf/__init__.py) specification.
    
3. **Multi-step forecasts (handled automatically)**

    You **do not** need to implement multi-step logic.

    The framework will automatically call your `predict()` method multiple times via `predict_all(asset, horizon, steps)` to construct forecasts at different temporal resolutions.

You can refer to the [Tracker examples](crunch_synth/examples) for guidance.

```python
class GaussianStepTracker(TrackerBase):
    """
    An example tracker that models *future incremental returns* as Gaussian-distributed.

    For each forecast step, the tracker returns a normal distribution
    r_{t,step} ~ N(a · mu, √a · sigma) where:
        - mu    = mean historical return
        - sigma = std historical return
        - a = (step / 300) represents the ratio of the forecast step duration to the historical 5-minute return interval.

    Multi-resolution forecasts (5min, 1h, 6h, 24h, ...)
    are automatically handled by `TrackerBase.predict_all()`,
    which calls the `predict()` method once per step size.

    /!/ This is not a price-distribution; it is a distribution over 
    incremental returns between consecutive steps /!/
    """
    def __init__(self):
        super().__init__()

    def predict(self, asset: str, horizon: int, step: int):

        # Retrieve recent historical prices sampled at 5-minute resolution
        resolution=300
        pairs = self.prices.get_prices(asset, days=5, resolution=300)
        if not pairs:
            return []

        _, past_prices = zip(*pairs)

        if len(past_prices) < 3:
            return []

        # Compute historical incremental returns (price differences)
        returns = np.diff(past_prices)

        # Estimate drift (mean return) and volatility (std dev of returns)
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))

        if sigma <= 0:
            return []

        num_segments = horizon // step

        # Construct one predictive distribution per future time step.
        # Each distribution models the incremental return over a `step`-second interval.
        #
        # IMPORTANT:
        # - The returned objects must strictly follow the `density_pdf` specification.
        # - Each entry corresponds to the return between t + (k−1)·step and t + k·step.
        #
        # We use a single-component Gaussian mixture for simplicity:
        #   r_{t,k} ~ N( (step / 300) · μ , sqrt(step / 300) · σ )
        #
        # where μ and σ are estimated from historical 5-minute returns.
        distributions = []
        for k in range(1, num_segments + 1):
            distributions.append({
                "step": k * step,                      # Time offset (in seconds) from forecast origin
                "type": "mixture",
                "components": [{
                    "density": {
                        "type": "builtin",             # Note: use 'builtin' instead of 'scipy' for speed
                        "name": "norm",  
                        "params": {
                            "loc": (step/resolution) * mu, 
                            "scale": np.sqrt(step/resolution) * sigma}
                    },
                    "weight": 1                        # Mixture weight — multiple densities with different weights can be combined
                }]
            })

        return distributions
```

## Prediction Phase

In each prediction round, players must submit **a set of density forecasts.**

A **prediction round** is defined by **one asset, one forecast horizon** and **one or more step resolutions.**
- A **24-hour horizon** forecast
    - Triggered **hourly** for each asset
    - Step resolutions: {5-minute, 1-hour, 6-hour, 24-hour}
    - Supported assets:
        ```["BTC", "SOL", "ETH", "XAUT", "SPYX", "NVDAX", "TSLAX", "AAPLX", "GOOGLX"]```
- A **1-hour horizon** forecast
    - Triggered **every 12 minutes** for each asset
    - Step resolutions: {1-minute, 5-minute, 15-minute, 30-minute, 1-hour}
    - Supported assets:
        ```["BTC", "SOL", "ETH", "XAUT"]```

All required forecasts for a prediction round must be generated within **40 seconds.**

## Scoring
- Once the full horizon has passed, each prediction is scored using a **[CRPS](https://en.wikipedia.org/wiki/Scoring_rule#:~:text=%5B8%5D-,Continuous%20ranked%20probability%20score,-%5Bedit%5D) scoring function**.
- A lower **CRPS score** reflects more accurate predictions.
- Leaderboard ranking is based on a **7-day rolling average** of CRPS scores across **all assets and horizons**, evaluated **relative to other participants**:
  - for each prediction round, the **best CRPS score receives a normalized score of 1**
  - the **worst 5% of CRPS scores receive a score of 0**

## Check your Tracker performance

TrackerEvaluator allows you to track your model's performance over time locally before participating in the live game. It maintains:

- Overall CRPS score
- Recent CRPS score
- Quarantine predictions (predictions stored and evaluated at a later time)

**A lower CRPS score reflects more accurate predictions.**

```python
from crunch_synth.tracker_evaluator import TrackerEvaluator
from crunch_synth.examples.benchmarktracker import GaussianStepTracker  # Your custom tracker

# Initialize the tracker evaluator with your custom GaussianStepTracker
tracker_evaluator = TrackerEvaluator(GaussianStepTracker())
# Feed a new price tick for SOL
tracker_evaluator.tick({"SOL": [(ts, price)]})
# You will generate predictive densities for SOL over a 24-hour period (86400s) 
# at multiple step resolutions: 5 minutes, 1 hour, 6 hours and 24 hours
predictions = tracker_evaluator.predict("SOL", horizon=3600*24,
                                        steps=[300, 3600, 3600*6, 3600*24])

print(f"My overall normalized CRPS score: {tracker_evaluator.overall_score("SOL"):.4f}")
```


## Tracker examples 
See [Tracker examples](crunch_synth/examples). There are:

- Quickstarter Notebooks
- Self-contained examples


## General Synth Game Advice 

The Synth game challenges you to predict the asset location using probabilistic forecasting.

### Probabilistic Forecasting

Probabilistic forecasting provides **a distribution of possible future values** rather than a single point estimate, allowing for uncertainty quantification. Instead of predicting only the most likely outcome, it estimates a range of potential outcomes along with their probabilities by outputting a **probability distribution**.

A probabilistic forecast models the conditional probability distribution of a future value $(Y_t)$ given past observations $(\mathcal{H}_{t-1})$. This can be expressed as:  

$$P(Y_t \mid \mathcal{H}_{t-1})$$

where $(\mathcal{H}_{t-1})$ represents the historical data up to time $(t-1)$. Instead of a single prediction $(\hat{Y}_t)$, the model estimates a full probability distribution $(f(Y_t \mid \mathcal{H}_{t-1}))$, which can take different parametric forms, such as a Gaussian:

$$Y_t \mid \mathcal{H}_{t-1} \sim \mathcal{N}(\mu_t, \sigma_t^2)$$

where $(\mu_t)$ is the predicted mean and $(\sigma_t^2)$ represents the uncertainty in the forecast.

Probabilistic forecasting can be handled through various approaches, including **variance forecasters**, **quantile forecasters**, **interval forecasters** or **distribution forecasters**, each capturing uncertainty differently.

For example, you can try to forecast the target location by a gaussian density function (or a mixture), thus the model output follows the form:

```python
{
   "density": {
      "type": "builtin",
      "name": "normal",
      "params": {"loc": y_mean, "scale": y_var}
   },
   "weight": weight
}
```

A **mixture density**, such as the gaussion mixture $\sum_{i=1}^{K} w_i \mathcal{N}(Y_t | \mu_i, \sigma_i^2)$ allows for capturing multi-modal distributions and approximate more complex distributions.

![](docs/proba_forecast.png)

### Additional Resources

- [Literature](LITERATURE.md) 
- Useful Python [packages](PACKAGES.md)