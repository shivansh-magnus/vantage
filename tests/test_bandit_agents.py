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
    agent.update(0, 1.0) # Q(0) = 1.0, N(0) = 1
    agent.update(1, 0.0) # Q(1) = 0.0, N(1) = 1
    
    # Current total_rounds (t) = 2
    # UCB score for arm 0: Q(0) + sqrt(2 * ln(t) / N(0)) = 1.0 + sqrt(2 * ln(2) / 1) = 1.0 + 1.1774 = 2.1774
    # UCB score for arm 1: Q(1) + sqrt(2 * ln(t) / N(1)) = 0.0 + sqrt(2 * ln(2) / 1) = 0.0 + 1.1774 = 1.1774
    # Since 2.1774 > 1.1774, it should choose arm 0
    assert agent.select_arm() == 0
    
    # Now pull arm 0 again and observe 0 reward
    agent.update(0, 0.0) # Q(0) = (1.0 + 0.0) / 2 = 0.5, N(0) = 2
    # Now t = 3
    # UCB score for arm 0: 0.5 + sqrt(2 * ln(3) / 2) = 0.5 + 1.0481 = 1.5481
    # UCB score for arm 1: 0.0 + sqrt(2 * ln(3) / 1) = 0.0 + 1.4823 = 1.4823
    # 1.5481 > 1.4823, still arm 0
    assert agent.select_arm() == 0
    
    # Update arm 0 again with 0 reward
    agent.update(0, 0.0) # Q(0) = (1.0 + 0.0 + 0.0) / 3 = 0.3333, N(0) = 3
    # Now t = 4
    # UCB score for arm 0: 0.3333 + sqrt(2 * ln(4) / 3) = 0.3333 + 0.9612 = 1.2945
    # UCB score for arm 1: 0.0 + sqrt(2 * ln(4) / 1) = 0.0 + 1.6651 = 1.6651
    # 1.6651 > 1.2945, so now it should switch to arm 1!
    assert agent.select_arm() == 1
