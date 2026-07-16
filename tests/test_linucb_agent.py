import numpy as np
import pytest
from vantage.agents.linucb_agent import LinUCB
from vantage.tools.simulator import MarketSimulator
from vantage.schemas import CustomerContext
from vantage.tools.optimization import find_optimal_price


def test_linucb_initial_state():
    prices = [10.0, 20.0, 30.0]
    d = 5
    agent = LinUCB(prices, d=d, lmbda=2.0)

    assert agent.d == 5
    assert agent.lmbda == 2.0
    assert len(agent.A) == 3
    assert len(agent.b) == 3

    for a in range(3):
        assert np.array_equal(agent.A[a], 2.0 * np.eye(5))
        assert np.array_equal(agent.b[a], np.zeros(5))


def test_linucb_worked_example():
    # Worked example from Section 1.7 of study guide
    # Context x = [1.0, is_weekend], lambda = 1.0
    prices = [10.0]
    agent = LinUCB(prices, d=2, lmbda=1.0)

    # 1. Weekday x = [1.0, 0.0], reward = 0.6
    agent.update(arm_idx=0, reward=0.6, context=np.array([1.0, 0.0]))

    # 2. Weekend x = [1.0, 1.0], reward = 0.9
    agent.update(arm_idx=0, reward=0.9, context=np.array([1.0, 1.0]))

    # Expected design matrix A_0 = [[3.0, 1.0], [1.0, 2.0]]
    # Expected b_0 = [1.5, 0.9]
    # Expected theta_hat = [0.42, 0.24]
    assert np.allclose(agent.A[0], np.array([[3.0, 1.0], [1.0, 2.0]]))
    assert np.allclose(agent.b[0], np.array([1.5, 0.9]))

    A_inv = np.linalg.inv(agent.A[0])
    theta_hat = A_inv @ agent.b[0]
    assert np.allclose(theta_hat, np.array([0.42, 0.24]))


def test_linucb_convergence():
    # End-to-end convergence check over 2000 rounds
    prices = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 80.0]
    d = 5
    agent = LinUCB(prices, d=d, lmbda=1.0, alpha=1.0, rng=np.random.default_rng(42))
    sim = MarketSimulator(seed=42)

    contexts = [
        CustomerContext(segment="student", day_type="weekday", competitor_price=15.0),
        CustomerContext(segment="professional", day_type="weekend", competitor_price=25.0),
    ]

    # Precalculate optimal expected revenue and optimal arms
    opt_arms = []
    for ctx in contexts:
        opt_price, _ = find_optimal_price(ctx, sim)
        # Find closest price arm
        closest_arm = np.argmin(np.abs(np.array(prices) - opt_price))
        opt_arms.append(closest_arm)

    # Simulation loop
    for _ in range(2500):
        # Sample context stochastically from our 2 choices
        ctx = sim.rng.choice(contexts)
        x = ctx.to_vector()

        arm_idx = agent.select_arm(x)
        chosen_price = prices[arm_idx]

        outcome = sim.step(chosen_price, ctx)
        reward = chosen_price * outcome

        agent.update(arm_idx, reward, x)

    # Verify that greedy selection for student points to student optimal arm
    # (Student should prefer lower price than Professional)
    x_student = contexts[0].to_vector()
    x_professional = contexts[1].to_vector()

    # Compute greedy choice argmax_a x^T * theta_hat
    student_preds = []
    professional_preds = []
    for a in range(agent.k_arms):
        A_inv = np.linalg.inv(agent.A[a])
        theta_hat = A_inv @ agent.b[a]
        student_preds.append(theta_hat @ x_student)
        professional_preds.append(theta_hat @ x_professional)

    student_greedy = np.argmax(student_preds)
    professional_greedy = np.argmax(professional_preds)

    # Student optimal price is lower ($15 or $20), professional is higher ($60 or $80)
    assert prices[student_greedy] < prices[professional_greedy]
    # Check that they match or are adjacent to their optimal arms
    assert np.abs(student_greedy - opt_arms[0]) <= 1
    assert np.abs(professional_greedy - opt_arms[1]) <= 1
