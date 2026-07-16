import numpy as np
from typing import Optional
from vantage.agents.bandit_agents import BanditAgent


class ThompsonSampling(BanditAgent):
    """
    Thompson Sampling Bandit Agent using Beta-Bernoulli conjugate priors.
    """

    def __init__(self, prices: list[float], rng: Optional[np.random.Generator] = None):
        super().__init__(prices, rng=rng)
        # Beta(1, 1) = Uniform(0, 1) prior for every arm.
        # Positive density everywhere on (0,1) from round 1 -- no cold-start
        # round-robin phase is required, unlike EpsilonGreedy/UCB1.
        self.alpha = np.ones(self.k_arms)
        self.beta = np.ones(self.k_arms)

    def select_arm(self) -> int:
        # Draw one posterior sample per arm this round.
        sampled_probs = self.rng.beta(self.alpha, self.beta)

        # Price-scale the samples: we're maximizing expected REVENUE,
        # not raw purchase probability (see Day 3 README, Section 1.5 --
        # this is the Thompson-Sampling analog of the Day 2 UCB1 fix).
        sampled_revenue = self.prices * sampled_probs

        return int(np.argmax(sampled_revenue))

    def update(self, arm_idx: int, reward: float) -> None:
        self.total_rounds += 1
        self.counts[arm_idx] += 1

        # Thompson Sampling's posterior tracks purchase probability
        # (binary outcome), exactly like the corrected UCB1 from Day 2.
        purchase_outcome = 1.0 if reward > 0.0 else 0.0
        if purchase_outcome == 1.0:
            self.alpha[arm_idx] += 1.0
        else:
            self.beta[arm_idx] += 1.0

        # Keep q_estimates in sync too (inherited incremental mean),
        # purely for inspection/plotting parity with the other two agents.
        n = self.counts[arm_idx]
        self.q_estimates[arm_idx] += (purchase_outcome - self.q_estimates[arm_idx]) / n

    def posterior_means(self) -> np.ndarray:
        """
        Convenience accessor: current E[theta_a] per arm, for diagnostics/plots.
        """
        return self.alpha / (self.alpha + self.beta)
