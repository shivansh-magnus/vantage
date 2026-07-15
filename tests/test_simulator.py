import numpy as np
from vantage.schemas import CustomerContext
from vantage.tools.simulator import MarketSimulator

def test_purchase_probability_monotonicity():
    """
    Checks that purchase probability strictly decreases as price increases.
    """
    sim = MarketSimulator(seed=42)
    context = CustomerContext(segment="default", day_type="weekday", competitor_price=20.0)
    
    prices = [10.0, 20.0, 30.0, 40.0, 50.0]
    probs = [sim.purchase_probability(p, context) for p in prices]
    
    # Assert each probability is strictly less than the previous one
    for i in range(1, len(probs)):
        assert probs[i] < probs[i-1], f"Probability did not decrease: {probs[i]} >= {probs[i-1]}"

def test_probability_boundaries():
    """
    Ensures purchase probability stays strictly inside (0, 1) even for extreme prices.
    """
    sim = MarketSimulator(seed=42)
    context = CustomerContext(segment="student", day_type="weekend", competitor_price=10.0)
    
    # Extreme low price
    low_prob = sim.purchase_probability(-1000.0, context)
    assert 0.0 < low_prob <= 1.0
    
    # Extreme high price
    high_prob = sim.purchase_probability(1000.0, context)
    assert 0.0 <= high_prob < 1.0

def test_segment_price_sensitivities():
    """
    Verifies that students are more price sensitive (probability drops faster)
    than professionals.
    """
    sim = MarketSimulator(seed=42)
    
    student_ctx = CustomerContext(segment="student", day_type="weekday", competitor_price=20.0)
    prof_ctx = CustomerContext(segment="professional", day_type="weekday", competitor_price=20.0)
    
    # Compare probability drop from price 10 to 30
    student_drop = sim.purchase_probability(10.0, student_ctx) - sim.purchase_probability(30.0, student_ctx)
    prof_drop = sim.purchase_probability(10.0, prof_ctx) - sim.purchase_probability(30.0, prof_ctx)
    
    assert student_drop > prof_drop, f"Expected student drop ({student_drop}) to be higher than professional drop ({prof_drop})"

def test_stochastic_draws_law_of_large_numbers():
    """
    Verifies that the average of stochastic Bernoulli draws converges
    to the expected purchase probability over a large number of trials.
    """
    sim = MarketSimulator(seed=42)
    context = CustomerContext(segment="default", day_type="weekend", competitor_price=15.0)
    price = 25.0
    
    expected_prob = sim.purchase_probability(price, context)
    
    # Draw 10,000 samples
    n_samples = 10000
    draws = [sim.step(price, context) for _ in range(n_samples)]
    empirical_prob = sum(draws) / n_samples
    
    # Check if empirical probability is close to expected (within 2% margin)
    assert np.isclose(empirical_prob, expected_prob, atol=0.02), f"Empirical ({empirical_prob}) was not close to expected ({expected_prob})"
