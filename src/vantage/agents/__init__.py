# Agents module init
from vantage.agents.bandit_agents import BanditAgent, EpsilonGreedy, UCB1
from vantage.agents.thompson_sampling import ThompsonSampling

__all__ = ["BanditAgent", "EpsilonGreedy", "UCB1", "ThompsonSampling"]

