import numpy as np
from typing import Optional
from vantage.agents.bandit_agents import BanditAgent


class LinUCB(BanditAgent):
    """
    Disjoint LinUCB (Linear Upper Confidence Bound) Bandit Agent.
    """

    def __init__(
        self,
        prices: list[float],
        d: int,
        lmbda: float = 1.0,
        alpha: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initializes the LinUCB agent.
        Args:
            prices: A list of candidate prices.
            d: The dimension of the context vector.
            lmbda: Regularization parameter lambda.
            alpha: Exploration parameter alpha.
            rng: NumPy random generator.
        """
        super().__init__(prices, rng=rng)
        self.d = d
        self.lmbda = lmbda
        self.alpha = alpha

        # Per arm, initialize Design Matrix A to lambda * I
        self.A = [lmbda * np.eye(d) for _ in range(self.k_arms)]
        # Per arm, initialize reward accumulator b to zeros
        self.b = [np.zeros(d) for _ in range(self.k_arms)]

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        """
        Selects which arm index to pull next based on the current context vector x.
        """
        if context is None:
            raise ValueError("LinUCB requires a context vector for arm selection.")

        x = np.asarray(context, dtype=np.float64)

        ucb_values = np.zeros(self.k_arms)
        for a in range(self.k_arms):
            # Compute inverse design matrix
            A_inv = np.linalg.inv(self.A[a])

            # Ridge regression coefficient estimate theta_hat = A_inv * b
            theta_hat = A_inv @ self.b[a]

            # Expected reward estimate (expected revenue directly, as model fits revenue)
            pred_reward = theta_hat @ x

            # Context-aware exploration bonus: alpha * sqrt(x^T * A_inv * x)
            bonus = self.alpha * np.sqrt(x @ A_inv @ x)

            ucb_values[a] = pred_reward + bonus

        # Resolve ties randomly
        max_ucb = np.max(ucb_values)
        best_arms = np.where(np.abs(ucb_values - max_ucb) < 1e-9)[0]
        return int(self.rng.choice(best_arms))

    def update(
        self, arm_idx: int, reward: float, context: Optional[np.ndarray] = None
    ) -> None:
        """
        Updates the ridge regression matrices using the observed context and reward.
        """
        if context is None:
            raise ValueError("LinUCB requires a context vector for updates.")

        x = np.asarray(context, dtype=np.float64)

        # Update accumulators
        self.A[arm_idx] += np.outer(x, x)
        self.b[arm_idx] += reward * x

        # Update base agent stats
        self.total_rounds += 1
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]

        # q_estimates tracking for parity
        self.q_estimates[arm_idx] += (reward - self.q_estimates[arm_idx]) / n
