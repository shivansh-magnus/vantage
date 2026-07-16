import os
import numpy as np
import matplotlib.pyplot as plt
from vantage.schemas import CustomerContext
from vantage.tools.simulator import MarketSimulator
from vantage.tools.optimization import find_optimal_price
from vantage.agents.bandit_agents import EpsilonGreedy, UCB1
from vantage.agents.thompson_sampling import ThompsonSampling
from vantage.agents.linucb_agent import LinUCB, SegmentSeparatedLinUCB

# Global cache for optimal revenue to avoid slow SciPy optimization inside the loop
_OPTIMAL_REV_CACHE = {}


def get_optimal_revenue(ctx: CustomerContext, sim: MarketSimulator) -> float:
    # Key: (segment, day_type, rounded competitor price to 2 decimal places)
    key = (ctx.segment, ctx.day_type, round(ctx.competitor_price, 2))
    if key not in _OPTIMAL_REV_CACHE:
        _, opt_rev = find_optimal_price(ctx, sim)
        _OPTIMAL_REV_CACHE[key] = opt_rev
    return _OPTIMAL_REV_CACHE[key]


def run_simulation(
    agent_class, agent_kwargs, prices, n_rounds, seed, sim_for_opt
):
    sim = MarketSimulator(seed=seed)
    agent_rng = np.random.default_rng(seed)

    # Initialize agent
    if agent_class == LinUCB:
        agent = agent_class(prices, d=5, rng=agent_rng, **agent_kwargs)
    elif agent_class == SegmentSeparatedLinUCB:
        agent = agent_class(prices, rng=agent_rng, **agent_kwargs)
    else:
        agent = agent_class(prices, rng=agent_rng, **agent_kwargs)

    regrets = []
    cumulative_regret = 0.0

    for _ in range(n_rounds):
        ctx = sim.sample_context()
        x = ctx.to_vector()

        # Prepare normalized vector for Joint LinUCB
        if agent_class == LinUCB:
            x_agent = x.copy()
            x_agent[4] /= 20.0
        else:
            x_agent = x

        # Select arm
        if agent_class in [LinUCB, SegmentSeparatedLinUCB]:
            # LinUCB agents require context input (either array or CustomerContext object)
            if agent_class == SegmentSeparatedLinUCB:
                arm_idx = agent.select_arm(ctx)
            else:
                arm_idx = agent.select_arm(x_agent)
        else:
            arm_idx = agent.select_arm()

        chosen_price = prices[arm_idx]

        # Simulator step
        outcome = sim.step(chosen_price, ctx)
        reward = chosen_price * outcome

        # Calculate optimal expected revenue and actual expected reward
        opt_rev = get_optimal_revenue(ctx, sim_for_opt)
        prob = sim.purchase_probability(chosen_price, ctx)
        expected_reward = chosen_price * prob

        # Regret = Optimal expected revenue - Actual expected reward
        instant_regret = opt_rev - expected_reward
        cumulative_regret += instant_regret
        regrets.append(cumulative_regret)

        # Update agent
        if agent_class in [LinUCB, SegmentSeparatedLinUCB]:
            if agent_class == SegmentSeparatedLinUCB:
                agent.update(arm_idx, reward, ctx)
            else:
                agent.update(arm_idx, reward, x_agent)
        else:
            agent.update(arm_idx, reward)

    return np.array(regrets), agent.counts


def main():
    prices = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 65.0, 80.0]
    n_rounds, n_seeds = 15000, 30

    sim_opt = MarketSimulator(seed=42)

    agents_config = {
        "Epsilon-Greedy (e=0.1)": (EpsilonGreedy, {"epsilon": 0.1}),
        "UCB1": (UCB1, {}),
        "Thompson Sampling": (ThompsonSampling, {}),
        "Joint LinUCB (scaled)": (LinUCB, {"scaled_bonus": True}),
        "Segment-Separated LinUCB": (SegmentSeparatedLinUCB, {}),
    }

    colors = {
        "Epsilon-Greedy (e=0.1)": "#ff8a3d",
        "UCB1": "#6ec8e8",
        "Thompson Sampling": "#b39ddb",
        "Joint LinUCB (scaled)": "#6fcf97",
        "Segment-Separated LinUCB": "#d32f2f",
    }

    plt.figure(figsize=(11, 6))

    print("=" * 60)
    print(f"5-Way Monte Carlo Dynamic Pricing Simulation:")
    print(f"Rounds (T): {n_rounds} | Seeds: {n_seeds}")
    print(f"Prices: {prices}")
    print("=" * 60)

    for name, (agent_class, kwargs) in agents_config.items():
        print(f"Simulating {name}...")
        all_runs = []
        counts_sum = np.zeros(len(prices))

        for seed in range(n_seeds):
            regrets, counts = run_simulation(
                agent_class, kwargs, prices, n_rounds, seed, sim_opt
            )
            all_runs.append(regrets)
            counts_sum += counts

        mean_regret = np.mean(all_runs, axis=0)
        plt.plot(mean_regret, label=name, color=colors[name], linewidth=2.5)

        avg_counts = counts_sum / n_seeds
        print(f"  Final Mean Regret: ${mean_regret[-1]:.2f}")
        print(
            f"  Average Pulls for $80 (best arm for Professional): {avg_counts[-1]:.1f}"
        )

    plt.xlabel("Round (t)", fontsize=12)
    plt.ylabel("Cumulative Regret ($)", fontsize=12)
    plt.title(
        f"5-Way Multi-Armed & Contextual Bandit Comparison (n_seeds={n_seeds})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    os.makedirs("runs", exist_ok=True)
    out_path = "runs/day4_5way_regret_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"\n5-Way comparison plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
