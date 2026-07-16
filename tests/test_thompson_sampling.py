import numpy as np
from vantage.agents.thompson_sampling import ThompsonSampling


def test_uniform_prior_at_init():
    agent = ThompsonSampling(prices=[10.0, 20.0, 30.0])
    assert np.all(agent.alpha == 1.0)
    assert np.all(agent.beta == 1.0)


def test_posterior_update_on_purchase():
    agent = ThompsonSampling(prices=[10.0, 20.0])
    agent.update(arm_idx=0, reward=10.0)  # purchase at price $10
    assert agent.alpha[0] == 2.0
    assert agent.beta[0] == 1.0
    assert agent.alpha[1] == 1.0  # untouched arm unaffected


def test_posterior_update_on_no_purchase():
    agent = ThompsonSampling(prices=[10.0, 20.0])
    agent.update(arm_idx=1, reward=0.0)  # no purchase at price $20
    assert agent.beta[1] == 2.0
    assert agent.alpha[1] == 1.0


def test_no_explicit_cold_start_required():
    # Unlike UCB1, select_arm() must not raise or special-case round 1 --
    # the uniform Beta(1,1) prior already supports sampling immediately.
    rng = np.random.default_rng(42)
    agent = ThompsonSampling(prices=[10.0, 20.0, 30.0], rng=rng)
    arm = agent.select_arm()
    assert 0 <= arm < 3


def test_thompson_sampling_sublinear_regret():
    # Mirrors Day 2's test_ucb1_sublinear_regret pattern.
    from vantage.tools.simulator import MarketSimulator
    from vantage.schemas import CustomerContext

    context = CustomerContext(
        segment="default", day_type="weekday", competitor_price=20.0
    )
    prices = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0]
    sim = MarketSimulator(seed=42)
    arm_probs = [sim.purchase_probability(p, context) for p in prices]
    arm_expected_rewards = [p * prob for p, prob in zip(prices, arm_probs)]
    optimal_rev = max(arm_expected_rewards)

    agent = ThompsonSampling(prices, rng=np.random.default_rng(42))
    regrets, cumulative_regret = [], 0.0

    for _ in range(2000):
        arm_idx = agent.select_arm()
        prob = arm_probs[arm_idx]
        outcome = int(sim.rng.random() < prob)
        reward = prices[arm_idx] * outcome

        cumulative_regret += optimal_rev - arm_expected_rewards[arm_idx]
        regrets.append(cumulative_regret)
        agent.update(arm_idx, reward)

    regret_first_half = regrets[999]
    regret_second_half = regrets[1999] - regrets[999]
    assert regret_second_half < regret_first_half * 0.7
