# Implementation Plan - Day 3: Thompson Sampling & Bayesian Bandits

Implement Thompson Sampling dynamic pricing agent using Beta-Bernoulli conjugate priors and set up a 3-way agent comparison (Epsilon-Greedy, UCB1, Thompson Sampling).

## User Review Required

> [!NOTE]
> The Thompson Sampling implementation utilizes a price-scaled selection rule to maximize expected revenue (modeled as `price * conversion_probability_sample`) rather than raw conversion probability. This ensures structural alignment with the corrected UCB1 selection logic.

## Open Questions

None.

## Proposed Changes

We will introduce the `ThompsonSampling` class, export it, add unit tests, and create a 3-way evaluation script.

---

### Bandit Agents Module

#### [NEW] [thompson_sampling.py](file:///c:/Users/dwive/OneDrive/Desktop/Vantage/src/vantage/agents/thompson_sampling.py)
* Add a `ThompsonSampling` agent subclassing `BanditAgent`.
* Initialize `alpha` and `beta` parameters to `1.0` (Uniform prior Beta(1,1)).
* Implement `select_arm()` to draw samples from Beta distribution and scale by price.
* Implement `update()` to increment `alpha` on purchase and `beta` on no-purchase, while updating the rolling mean estimate `q_estimates`.
* Implement `posterior_means()` convenience helper for diagnostic reporting.

#### [MODIFY] [__init__.py](file:///c:/Users/dwive/OneDrive/Desktop/Vantage/src/vantage/agents/__init__.py)
* Export `ThompsonSampling` from `vantage.agents`.

---

### Evaluation and Scripts

#### [NEW] [evaluate_all_agents.py](file:///c:/Users/dwive/OneDrive/Desktop/Vantage/scripts/evaluate_all_agents.py)
* Create evaluation script running Epsilon-Greedy, UCB1, and Thompson Sampling across multiple seeds (e.g. 30 seeds, 10,000 rounds).
* Speed up execution by pre-calculating purchase probabilities and expected revenues outside the simulation loop.
* Compute and plot cumulative regrets for all three agents and save the comparison chart as `runs/day3_regret_comparison.png`.

---

### Tests

#### [NEW] [test_thompson_sampling.py](file:///c:/Users/dwive/OneDrive/Desktop/Vantage/tests/test_thompson_sampling.py)
* Add unit tests:
  - `test_uniform_prior_at_init`: asserts alphas/betas are set to 1.
  - `test_posterior_update_on_purchase`: verifies alpha increments by 1.
  - `test_posterior_update_on_no_purchase`: verifies beta increments by 1.
  - `test_no_explicit_cold_start_required`: verifies arm selection works immediately without prior pulls.
  - `test_thompson_sampling_sublinear_regret`: simulates a 2000-round run, asserting sub-linear regret growth (second-half regret < first-half regret * 0.7).

---

## Verification Plan

### Automated Tests
* Run pytest to verify all existing and new tests pass:
  ```bash
  uv run pytest
  ```

### Manual Verification
* Run the evaluation script:
  ```bash
  uv run python scripts/evaluate_all_agents.py
  ```
* Verify `runs/day3_regret_comparison.png` is generated correctly.
* Inspect that Thompson Sampling displays sublinear cumulative regret and converges to the correct optimal pricing arms.
