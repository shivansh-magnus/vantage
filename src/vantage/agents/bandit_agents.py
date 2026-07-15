import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class BanditAgent(ABC):
    """Abstract Base Class for Multi-Armed Bandit Agents."""
    
    def __init__(self, prices: list[float], rng: Optional[np.random.Generator] = None):
        """
        Initializes the agent with a list of candidate prices.
        """
        self.prices = np.array(prices)
        self.k_arms = len(prices)
        self.q_estimates = np.zeros(self.k_arms)
        self.counts = np.zeros(self.k_arms, dtype=int)
        self.total_rounds = 0
        self.rng = rng if rng is not None else np.random.default_rng()

    @abstractmethod
    def select_arm(self) -> int:
        """
        Selects which arm index to pull next.
        Returns:
            int: The index of the selected arm (0 to k_arms-1)
        """
        pass

    def update(self, arm_idx: int, reward: float) -> None:
        """
        Updates the empirical expected reward (Q) for the pulled arm using the incremental mean rule.
        """
        self.total_rounds += 1
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        
        # Incremental running mean update
        self.q_estimates[arm_idx] += (reward - self.q_estimates[arm_idx]) / n


class EpsilonGreedy(BanditAgent):
    """
    Epsilon-Greedy Bandit Agent.
    """
    
    def __init__(self, prices: list[float], epsilon: float = 0.1, rng: Optional[np.random.Generator] = None):
        super().__init__(prices, rng=rng)
        self.epsilon = epsilon

    def select_arm(self) -> int:
        # Cold start phase: pull any untried arm first
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return int(untried[0])
            
        if self.rng.random() < self.epsilon:
            # Explore: pick an arm uniformly at random
            return int(self.rng.integers(self.k_arms))
        else:
            # Exploit: choose arm with highest Q-estimate, resolve ties randomly
            max_q = np.max(self.q_estimates)
            best_arms = np.where(self.q_estimates == max_q)[0]
            return int(self.rng.choice(best_arms))


class UCB1(BanditAgent):
    """
    UCB1 (Upper Confidence Bound) Bandit Agent.
    """
    
    def select_arm(self) -> int:
        # Cold start: pull every arm once in round-robin order
        untried = np.where(self.counts == 0)[0]
        if len(untried) > 0:
            return int(untried[0])
            
        # UCB1 decision rule: argmax_a [ Q(a) + sqrt(2 * ln(t) / N(a)) ]
        ucb_values = np.zeros(self.k_arms)
        for a in range(self.k_arms):
            bonus = np.sqrt((2.0 * np.log(self.total_rounds)) / self.counts[a])
            ucb_values[a] = self.q_estimates[a] + bonus
            
        # Resolve ties randomly
        max_ucb = np.max(ucb_values)
        best_arms = np.where(ucb_values == max_ucb)[0]
        return int(self.rng.choice(best_arms))

