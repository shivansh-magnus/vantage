import numpy as np
from typing import Optional, Union, Dict, Tuple
from vantage.agents.bandit_agents import BanditAgent
from vantage.schemas import CustomerContext


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
        scaled_bonus: bool = True,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Initializes the LinUCB agent.
        Args:
            prices: A list of candidate prices.
            d: The dimension of the context vector.
            lmbda: Regularization parameter lambda.
            alpha: Exploration parameter alpha.
            scaled_bonus: If True, scales the exploration bonus by the arm price.
            rng: NumPy random generator.
        """
        super().__init__(prices, rng=rng)
        self.d = d
        self.lmbda = lmbda
        self.alpha = alpha
        self.scaled_bonus = scaled_bonus

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

            # Expected reward estimate
            pred_reward = theta_hat @ x

            # Context-aware exploration bonus: alpha * sqrt(x^T * A_inv * x)
            # If scaled_bonus is True, scale by the price of the arm to match reward scale
            if self.scaled_bonus:
                bonus = self.alpha * self.prices[a] * np.sqrt(x @ A_inv @ x)
            else:
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


class SegmentSeparatedLinUCB(BanditAgent):
    """
    LinUCB Wrapper that maintains separate independent LinUCB instances per customer segment
    to eliminate cross-segment prediction poisoning.
    """

    def __init__(
        self,
        prices: list[float],
        lmbda: float = 1.0,
        alpha: float = 1.0,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__(prices, rng=rng)
        self.lmbda = lmbda
        self.alpha = alpha

        # 3 independent LinUCB agents with dimension d=3
        # Sub-context features: [intercept, is_weekend, competitor_price / 20.0]
        self.agents: Dict[str, LinUCB] = {
            "student": LinUCB(
                prices,
                d=3,
                lmbda=lmbda,
                alpha=alpha,
                scaled_bonus=True,
                rng=rng if rng is not None else np.random.default_rng(),
            ),
            "professional": LinUCB(
                prices,
                d=3,
                lmbda=lmbda,
                alpha=alpha,
                scaled_bonus=True,
                rng=rng if rng is not None else np.random.default_rng(),
            ),
            "default": LinUCB(
                prices,
                d=3,
                lmbda=lmbda,
                alpha=alpha,
                scaled_bonus=True,
                rng=rng if rng is not None else np.random.default_rng(),
            ),
        }

    def _get_segment_and_sub_context(
        self, context: Union[CustomerContext, np.ndarray]
    ) -> Tuple[str, np.ndarray]:
        """
        Parses the context input (either CustomerContext or numpy vector) and returns
        the target segment and the normalized 3-dimensional sub-context vector.
        """
        if isinstance(context, CustomerContext):
            segment = context.segment
            is_weekend = 1.0 if context.day_type == "weekend" else 0.0
            comp_price = context.competitor_price
        elif isinstance(context, dict):
            segment = context.get("segment", "default")
            is_weekend = 1.0 if context.get("day_type") == "weekend" else 0.0
            comp_price = context.get("competitor_price", 20.0)
        else:
            # Assume it's a 5-element numpy vector:
            # [intercept, is_student, is_professional, is_weekend, competitor_price]
            x = np.asarray(context, dtype=np.float64)
            if len(x) == 5:
                if x[1] == 1.0:
                    segment = "student"
                elif x[2] == 1.0:
                    segment = "professional"
                else:
                    segment = "default"
                is_weekend = x[3]
                comp_price = x[4]
            else:
                # Fallback if dimension is already 3
                segment = "default"
                is_weekend = x[1] if len(x) > 1 else 0.0
                comp_price = (x[2] * 20.0) if len(x) > 2 else 20.0

        # sub-context vector: [intercept, is_weekend, competitor_price / 20.0]
        sub_context = np.array(
            [1.0, is_weekend, comp_price / 20.0], dtype=np.float64
        )
        return segment, sub_context

    def select_arm(self, context: Optional[np.ndarray] = None) -> int:
        if context is None:
            raise ValueError("SegmentSeparatedLinUCB requires context.")
        segment, sub_context = self._get_segment_and_sub_context(context)
        return self.agents[segment].select_arm(sub_context)

    def update(
        self, arm_idx: int, reward: float, context: Optional[np.ndarray] = None
    ) -> None:
        if context is None:
            raise ValueError("SegmentSeparatedLinUCB requires context.")
        segment, sub_context = self._get_segment_and_sub_context(context)
        self.agents[segment].update(arm_idx, reward, sub_context)

        # Update wrapper statistics
        self.total_rounds += 1
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.q_estimates[arm_idx] += (reward - self.q_estimates[arm_idx]) / n
