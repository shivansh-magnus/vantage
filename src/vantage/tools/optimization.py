from typing import Tuple, Callable
import numpy as np
from scipy.optimize import minimize_scalar
from vantage.schemas import CustomerContext
from vantage.tools.simulator import MarketSimulator

def expected_revenue(price: float, context: CustomerContext, simulator: MarketSimulator) -> float:
    """
    Computes the expected revenue for a given price and customer context:
        R(price, context) = price * P(purchase | price, context)
    """
    prob = simulator.purchase_probability(price, context)
    return price * prob

def find_optimal_price(
    context: CustomerContext,
    simulator: MarketSimulator,
    price_bounds: Tuple[float, float] = (0.0, 100.0)
) -> Tuple[float, float]:
    """
    Finds the revenue-maximizing price and the maximum expected revenue for a context
    using SciPy's bounded numerical optimizer.
    
    Returns:
        Tuple[optimal_price, max_expected_revenue]
    """
    # Objective function is negative expected revenue since minimize_scalar minimizes
    def objective(price: float) -> float:
        return -expected_revenue(price, context, simulator)

    res = minimize_scalar(
        objective,
        bounds=price_bounds,
        method="bounded"
    )
    
    if not res.success:
        # Fallback to a fine grid search if numerical optimization fails
        prices = np.linspace(price_bounds[0], price_bounds[1], 10000)
        revenues = [expected_revenue(p, context, simulator) for p in prices]
        best_idx = np.argmax(revenues)
        return float(prices[best_idx]), float(revenues[best_idx])

    optimal_price = float(res.x)
    max_rev = -float(res.fun)
    return optimal_price, max_rev

def analytical_linear_optimum(a: float, b: float) -> float:
    """
    Calculates the exact revenue-optimal price for a linear demand curve q(p) = a - b * p.
    Closed-form solution: p* = a / (2b).
    """
    if b <= 0:
        raise ValueError("Price sensitivity parameter 'b' must be greater than zero.")
    return a / (2.0 * b)

def grid_search_optimal_price(
    context: CustomerContext,
    simulator: MarketSimulator,
    min_price: float = 0.0,
    max_price: float = 100.0,
    step: float = 0.01
) -> Tuple[float, float]:
    """
    Runs a deterministic grid search to find the optimal price.
    Useful for verifying the SciPy solver.
    """
    prices = np.arange(min_price, max_price + step, step)
    revenues = np.array([expected_revenue(p, context, simulator) for p in prices])
    best_idx = np.argmax(revenues)
    return float(prices[best_idx]), float(revenues[best_idx])
