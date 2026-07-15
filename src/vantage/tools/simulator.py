import numpy as np
from typing import Optional
from vantage.schemas import CustomerContext

class MarketSimulator:
    """
    A simulator of customer purchase behavior.
    Models purchase probability as a sigmoid function of price and customer context:
        P(purchase | price, context) = sigmoid(beta_0 - beta_price * price)
    where beta_0 and beta_price are context-dependent parameters.
    """
    # True hidden baseline parameters (intercept, is_student, is_professional, is_weekend, competitor_price)
    _V0 = np.array([3.0, -1.0, 1.5, 0.5, 0.05])
    
    # True hidden price sensitivity parameters (intercept, is_student, is_professional, is_weekend, competitor_price)
    # Price sensitivity is always positive to ensure demand monotonically decreases with price.
    _V_PRICE = np.array([0.10, 0.08, -0.05, 0.0, 0.0])

    def __init__(self, seed: Optional[int] = None):
        """
        Initializes the simulator with an optional random seed.
        """
        self.rng = np.random.default_rng(seed)

    def sample_context(self) -> CustomerContext:
        """
        Stochastically generates a CustomerContext representing a single customer interaction.
        """
        # Segment probabilities: 30% student, 30% professional, 40% default
        segments = ["student", "professional", "default"]
        segment = self.rng.choice(segments, p=[0.3, 0.3, 0.4])

        # Day type probabilities: 70% weekday, 30% weekend
        day_types = ["weekday", "weekend"]
        day_type = self.rng.choice(day_types, p=[0.7, 0.3])

        # Competitor price: normal distribution mean=20, std=3, clipped to minimum of 5.0
        competitor_price = float(np.clip(self.rng.normal(20.0, 3.0), 5.0, None))

        return CustomerContext(
            segment=segment,
            day_type=day_type,
            competitor_price=competitor_price
        )

    def purchase_probability(self, price: float, context: CustomerContext) -> float:
        """
        Calculates the deterministic purchase probability for a given price and context.
        """
        x = context.to_vector()
        
        # Calculate beta parameters by linear combination of context vector
        beta_0 = np.dot(self._V0, x)
        beta_price = np.dot(self._V_PRICE, x)
        
        # Ensure beta_price is positive (physics constraint of demand curves)
        beta_price = max(beta_price, 0.001)

        z = beta_0 - beta_price * price
        
        # Sigmoid function: 1 / (1 + exp(-z))
        # Protect against overflow by clipping z
        z_clipped = np.clip(z, -20.0, 20.0)
        return float(1.0 / (1.0 + np.exp(-z_clipped)))

    def step(self, price: float, context: CustomerContext) -> int:
        """
        Generates a stochastic purchase outcome (1 for purchase, 0 for no purchase)
        via a Bernoulli draw using the calculated purchase probability.
        """
        prob = self.purchase_probability(price, context)
        return int(self.rng.random() < prob)
