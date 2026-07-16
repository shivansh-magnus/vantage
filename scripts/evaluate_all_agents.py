import os
import numpy as np
import matplotlib.pyplot as plt
from vantage.schemas import CustomerContext
from vantage.tools.simulator import MarketSimulator
from vantage.tools.optimization import find_optimal_price
from vantage.agents.bandit_agents import EpsilonGreedy, UCB1
from vantage.agents.thompson_sampling import ThompsonSampling

AGENTS = {
    "Epsilon-Greedy": (EpsilonGreedy, {"epsilon": 0.1}),
    "UCB1": (UCB1, {}),
    "Thompson Sampling": (ThompsonSampling, {}),
}


def run_simulation(
    agent_class, agent_kwargs, context, prices, optimal_rev, n_rounds, seed
):
    sim = MarketSimulator(seed=seed)

    # Pre-calculate purchase probabilities and expected rewards (Day 2 speedup)
    arm_probs = [sim.purchase_probability(p, context) for p in prices]
    arm_expected_rewards = [p * prob for p, prob in zip(prices, arm_probs)]

    agent_rng = np.random.default_rng(seed)
    agent = agent_class(prices, rng=agent_rng, **agent_kwargs)

    regrets = []
    cumulative_regret = 0.0

    for _ in range(n_rounds):
        arm_idx = agent.select_arm()
        chosen_price = prices[arm_idx]

        outcome = int(sim.rng.random() < arm_probs[arm_idx])
        reward = chosen_price * outcome

        instant_regret = optimal_rev - arm_expected_rewards[arm_idx]
        cumulative_regret += instant_regret
        regrets.append(cumulative_regret)

        agent.update(arm_idx, reward)

    return np.array(regrets), agent.counts


def main():
    # Context translation from legacy/pseudocode is_professional=True to schema format:
    context = CustomerContext(
        segment="professional", day_type="weekday", competitor_price=25.0
    )
    prices = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 65.0, 80.0, 85.0, 90.0, 95.0]

    sim = MarketSimulator(seed=0)
    optimal_price, optimal_rev = find_optimal_price(context, sim)

    print("=" * 60)
    print("MAB Pricing Evaluation Setup (All Agents):")
    print(
        f"Context: {context.segment} | {context.day_type} | Comp Price: ${context.competitor_price:.2f}"
    )
    print(f"Continuous Optimal Price (p*): ${optimal_price:.2f}")
    print(f"Optimal Expected Revenue: ${optimal_rev:.2f}")
    print(f"Candidate Price Arms: {prices}")
    print("=" * 60)

    n_rounds, n_seeds = 10000, 30
    plt.figure(figsize=(10, 6))

    # Standard styling and colors
    colors = {
        "Epsilon-Greedy": "#ff8a3d",
        "UCB1": "#6ec8e8",
        "Thompson Sampling": "#6fcf97",
    }

    for name, (agent_class, kwargs) in AGENTS.items():
        all_runs = []
        counts_sum = np.zeros(len(prices))

        for seed in range(n_seeds):
            regrets, counts = run_simulation(
                agent_class, kwargs, context, prices, optimal_rev, n_rounds, seed
            )
            all_runs.append(regrets)
            counts_sum += counts

        mean_regret = np.mean(all_runs, axis=0)
        plt.plot(mean_regret, label=name, color=colors[name], linewidth=2.5)

        # Print pull distributions for diagnostics
        avg_counts = counts_sum / n_seeds
        print(f"\nPull Distribution for {name} (Averaged over {n_seeds} seeds):")
        for p, count in zip(prices, avg_counts):
            print(f"  Price ${p:<4.1f}: {count:>6.1f} pulls")

    plt.xlabel("Round (t)", fontsize=12)
    plt.ylabel("Cumulative Regret ($)", fontsize=12)
    plt.title(
        f"Cumulative Regret Comparison (n_seeds={n_seeds}, optimal_price=${optimal_price:.2f})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs("runs", exist_ok=True)
    out_path = "runs/day3_regret_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nRegret curves comparison plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
