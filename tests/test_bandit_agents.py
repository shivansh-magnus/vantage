import pytest
import numpy as np
from vantage.agents.bandit_agents import EpsilonGreedy, UCB1

def test_epsilon_greedy_cold_start():
    prices = [10.0, 20.0, 30.0]
    agent = EpsilonGreedy(prices, epsilon=0.1)
    
    # First K rounds must pull each untried arm once (calling update in between)
    first_draws = []
    for _ in range(3):
        arm = agent.select_arm()
        first_draws.append(arm)
        agent.update(arm, 1.0)
    assert first_draws == [0, 1, 2]

def test_epsilon_greedy_exploitation():
    prices = [10.0, 20.0, 30.0]
    agent = EpsilonGreedy(prices, epsilon=0.0) # Pure exploitation
    
    # Simulate updates for cold start to set the estimates
    agent.update(0, 5.0)  # Arm 0: Q = 5.0
    agent.update(1, 15.0) # Arm 1: Q = 15.0
    agent.update(2, 8.0)  # Arm 2: Q = 8.0
    
    # Under epsilon=0, it must always choose the arm with the highest Q value (Arm 1)
    assert agent.select_arm() == 1

def test_incremental_update():
    prices = [10.0]
    agent = EpsilonGreedy(prices)
    
    # Test first update
    agent.update(0, 10.0)
    assert agent.q_estimates[0] == 10.0
    assert agent.counts[0] == 1
    assert agent.total_rounds == 1
    
    # Test second update
    agent.update(0, 0.0)
    assert agent.q_estimates[0] == 5.0
    assert agent.counts[0] == 2
    assert agent.total_rounds == 2
    
    # Test third update
    agent.update(0, 14.0)
    assert pytest.approx(agent.q_estimates[0]) == 8.0
    assert agent.counts[0] == 3
    assert agent.total_rounds == 3

def test_ucb1_cold_start():
    prices = [10.0, 20.0, 30.0, 40.0]
    agent = UCB1(prices)
    
    # First 4 rounds must be round-robin sequential cold start
    first_draws = []
    for _ in range(4):
        arm = agent.select_arm()
        first_draws.append(arm)
        agent.update(arm, 1.0)
    assert first_draws == [0, 1, 2, 3]

def test_ucb1_selection_logic():
    prices = [10.0, 20.0]
    agent = UCB1(prices)
    
    # Trigger cold start updates
    agent.update(0, 1.0) # Q(0) = 1.0, N(0) = 1 (reward > 0 -> purchase_outcome = 1.0)
    agent.update(1, 0.0) # Q(1) = 0.0, N(1) = 1 (reward = 0 -> purchase_outcome = 0.0)
    
    # Current total_rounds (t) = 2
    # UCB score for arm 0: price * (Q(0) + sqrt(2 * ln(t) / N(0))) = 10.0 * (1.0 + sqrt(2 * ln(2) / 1)) = 10.0 * 2.1774 = 21.774
    # UCB score for arm 1: price * (Q(1) + sqrt(2 * ln(t) / N(1))) = 20.0 * (0.0 + sqrt(2 * ln(2) / 1)) = 20.0 * 1.1774 = 23.548
    # Since 23.548 > 21.774, it should choose arm 1
    assert agent.select_arm() == 1
    
    # Now pull arm 1 again and observe 0 reward
    agent.update(1, 0.0) # Q(1) = 0.0, N(1) = 2
    # Now t = 3
    # UCB score for arm 0: 10.0 * (1.0 + sqrt(2 * ln(3) / 1)) = 10.0 * (1.0 + 1.4823) = 24.823
    # UCB score for arm 1: 20.0 * (0.0 + sqrt(2 * ln(3) / 2)) = 20.0 * (0.0 + 1.0481) = 20.962
    # Since 24.823 > 20.962, it should now choose arm 0
    assert agent.select_arm() == 0
    
    # Update arm 0 again with 0 reward
    agent.update(0, 0.0) # Q(0) = 0.5, N(0) = 2
    # Now t = 4
    # UCB score for arm 0: 10.0 * (0.5 + sqrt(2 * ln(4) / 2)) = 10.0 * (0.5 + 1.1774) = 16.774
    # UCB score for arm 1: 20.0 * (0.0 + sqrt(2 * ln(4) / 2)) = 20.0 * (0.0 + 1.1774) = 23.548
    # Since 23.548 > 16.774, it should choose arm 1
    assert agent.select_arm() == 1


def test_ucb1_sublinear_regret():
    from vantage.schemas import CustomerContext
    from vantage.tools.simulator import MarketSimulator
    from vantage.tools.optimization import find_optimal_price
    
    prices = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0]
    agent = UCB1(prices, rng=np.random.default_rng(42))
    sim = MarketSimulator(seed=42)
    context = CustomerContext(segment="default", day_type="weekday", competitor_price=20.0)
    _, optimal_rev = find_optimal_price(context, sim)
    
    regrets = []
    cumulative_regret = 0.0
    
    for _ in range(2000):
        arm_idx = agent.select_arm()
        chosen_price = prices[arm_idx]
        outcome = sim.step(chosen_price, context)
        reward = chosen_price * outcome
        
        prob = sim.purchase_probability(chosen_price, context)
        expected_reward = chosen_price * prob
        instant_regret = optimal_rev - expected_reward
        cumulative_regret += instant_regret
        regrets.append(cumulative_regret)
        
        agent.update(arm_idx, reward)
        
    regret_first_half = regrets[999]
    regret_second_half = regrets[1999] - regrets[999]
    
    # Regret in the second half must be significantly smaller than in the first half
    # demonstrating learning and sub-linear regret growth
    assert regret_second_half < regret_first_half * 0.7


