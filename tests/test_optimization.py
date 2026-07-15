import numpy as np
import pytest
from vantage.schemas import CustomerContext
from vantage.tools.optimization import (
    find_optimal_price,
    expected_revenue,
    analytical_linear_optimum,
    grid_search_optimal_price
)

class MockLinearSimulator:
    """
    Mock simulator representing a linear demand curve:
        q(p) = a - b * p
    Let a = 1.0, b = 0.02, so that probability is strictly within [0, 1] for p in [0, 50].
    The analytical optimum should be: p* = a / (2b) = 1.0 / 0.04 = 25.0.
    """
    def purchase_probability(self, price: float, context: CustomerContext) -> float:
        a = 1.0
        b = 0.02
        return float(np.clip(a - b * price, 0.0, 1.0))

def test_analytical_linear_optimum():
    """
    Verifies that the closed-form analytical linear formula is correct.
    """
    assert analytical_linear_optimum(1.0, 0.02) == 25.0
    with pytest.raises(ValueError):
        analytical_linear_optimum(1.0, -0.01)

def test_optimizer_against_closed_form():
    """
    Unit tests the numerical optimizer (SciPy bounded and grid search)
    against the known closed-form analytical linear optimum.
    """
    simulator = MockLinearSimulator()
    context = CustomerContext(segment="default", day_type="weekday", competitor_price=20.0)
    
    # Bounded numerical optimizer search
    opt_price_num, _ = find_optimal_price(context, simulator, price_bounds=(0.0, 50.0))
    
    # Grid search optimizer
    opt_price_grid, _ = grid_search_optimal_price(context, simulator, min_price=0.0, max_price=50.0, step=0.001)
    
    # Expected analytical optimum is 25.0
    expected_opt_price = 25.0
    
    assert np.isclose(opt_price_num, expected_opt_price, atol=1e-4), f"Numerical optimum ({opt_price_num}) did not match expected ({expected_opt_price})"
    assert np.isclose(opt_price_grid, expected_opt_price, atol=1e-3), f"Grid search optimum ({opt_price_grid}) did not match expected ({expected_opt_price})"

def test_expected_revenue_calculation():
    """
    Validates that the expected revenue is correctly computed as price * purchase_probability.
    """
    class SimpleSimulator:
        def purchase_probability(self, price: float, context: CustomerContext) -> float:
            return 0.5

    sim = SimpleSimulator()
    context = CustomerContext(segment="default", day_type="weekday", competitor_price=20.0)
    
    assert expected_revenue(10.0, context, sim) == 5.0
    assert expected_revenue(0.0, context, sim) == 0.0
