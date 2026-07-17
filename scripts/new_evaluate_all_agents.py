import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
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


def find_grid_optimal_price(prices: list[float], context: CustomerContext, simulator: MarketSimulator) -> float:
    """Finds the price on the discrete grid that maximizes true expected revenue."""
    revenues = [p * simulator.purchase_probability(p, context) for p in prices]
    best_idx = np.argmax(revenues)
    return prices[best_idx]


def get_greedy_price(agent, agent_class, context: CustomerContext, prices: list[float]) -> float:
    """Computes the current believed-best price for an agent given a context."""
    if agent_class == LinUCB:
        x = context.to_vector()
        x_agent = x.copy()
        x_agent[4] /= 20.0
        preds = []
        for a in range(agent.k_arms):
            A_inv = np.linalg.inv(agent.A[a])
            theta_hat = A_inv @ agent.b[a]
            preds.append(theta_hat @ x_agent)
        greedy_arm = np.argmax(preds)
        return prices[greedy_arm]

    elif agent_class == SegmentSeparatedLinUCB:
        sub_agent = agent.agents[context.segment]
        _, sub_x = agent._get_segment_and_sub_context(context)
        preds = []
        for a in range(sub_agent.k_arms):
            A_inv = np.linalg.inv(sub_agent.A[a])
            theta_hat = A_inv @ sub_agent.b[a]
            preds.append(theta_hat @ sub_x)
        greedy_arm = np.argmax(preds)
        return prices[greedy_arm]

    elif agent_class == ThompsonSampling:
        # Expected revenue = price * posterior_mean
        expected_revenue = np.array(prices) * (agent.alpha / (agent.alpha + agent.beta))
        greedy_arm = np.argmax(expected_revenue)
        return prices[greedy_arm]

    elif agent_class == UCB1:
        # q_estimates tracks conversion probability. Expected revenue = price * conversion_prob
        expected_revenue = np.array(prices) * agent.q_estimates
        greedy_arm = np.argmax(expected_revenue)
        return prices[greedy_arm]

    else:  # EpsilonGreedy
        # q_estimates tracks running average revenue directly
        greedy_arm = np.argmax(agent.q_estimates)
        return prices[greedy_arm]


def run_simulation(
    agent_class, agent_kwargs, prices, n_rounds, seed, sim_for_opt
) -> Tuple[np.ndarray, np.ndarray]:
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


def run_convergence_simulation(
    agent_class, agent_kwargs, prices, n_rounds, seed, eval_interval, target_contexts, noise_level=0.0
) -> Tuple[List[int], Dict[str, List[float]]]:
    sim = MarketSimulator(seed=seed)
    agent_rng = np.random.default_rng(seed)

    # Initialize agent
    if agent_class == LinUCB:
        agent = agent_class(prices, d=5, rng=agent_rng, **agent_kwargs)
    elif agent_class == SegmentSeparatedLinUCB:
        agent = agent_class(prices, rng=agent_rng, **agent_kwargs)
    else:
        agent = agent_class(prices, rng=agent_rng, **agent_kwargs)

    history = {name: [] for name in target_contexts}
    rounds = []

    for r in range(1, n_rounds + 1):
        ctx = sim.sample_context()
        x = ctx.to_vector()

        if agent_class == LinUCB:
            x_agent = x.copy()
            x_agent[4] /= 20.0
        else:
            x_agent = x

        # Select arm
        if agent_class in [LinUCB, SegmentSeparatedLinUCB]:
            if agent_class == SegmentSeparatedLinUCB:
                arm_idx = agent.select_arm(ctx)
            else:
                arm_idx = agent.select_arm(x_agent)
        else:
            arm_idx = agent.select_arm()

        chosen_price = prices[arm_idx]

        # Step
        prob = sim.purchase_probability(chosen_price, ctx)
        if noise_level > 0.0:
            if sim.rng.random() < noise_level:
                outcome = int(sim.rng.choice([0, 1]))
            else:
                outcome = int(sim.rng.random() < prob)
        else:
            outcome = int(sim.rng.random() < prob)

        reward = chosen_price * outcome

        # Update agent
        if agent_class in [LinUCB, SegmentSeparatedLinUCB]:
            if agent_class == SegmentSeparatedLinUCB:
                agent.update(arm_idx, reward, ctx)
            else:
                agent.update(arm_idx, reward, x_agent)
        else:
            agent.update(arm_idx, reward)

        # Record believed-best price at intervals
        if r % eval_interval == 0:
            rounds.append(r)
            for name, eval_ctx in target_contexts.items():
                price = get_greedy_price(agent, agent_class, eval_ctx, prices)
                history[name].append(price)

    return rounds, history


def get_convergence_round(history: List[float], opt_price: float, eval_interval: int, total_rounds: int) -> int:
    """Computes the exact round after which the price stays at the true grid optimum."""
    last_incorrect = -1
    for i, p in enumerate(history):
        if abs(p - opt_price) > 1e-5:
            last_incorrect = i
    if last_incorrect == len(history) - 1:
        return total_rounds  # did not converge by the end
    return (last_incorrect + 1) * eval_interval


def main():
    prices = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 65.0, 80.0]
    n_rounds, n_seeds = 15000, 30
    eval_interval = 50

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

    target_contexts = {
        "Student (Weekday, Comp=$15)": CustomerContext(
            segment="student", day_type="weekday", competitor_price=15.0
        ),
        "Professional (Weekend, Comp=$25)": CustomerContext(
            segment="professional", day_type="weekend", competitor_price=25.0
        ),
        "Default (Weekday, Comp=$20)": CustomerContext(
            segment="default", day_type="weekday", competitor_price=20.0
        ),
    }

    # Calculate optimal prices on the 10-arm grid
    optimal_prices_10 = {}
    for name, ctx in target_contexts.items():
        opt_price = find_grid_optimal_price(prices, ctx, sim_opt)
        optimal_prices_10[name] = opt_price

    os.makedirs("runs", exist_ok=True)

    # =========================================================================
    # Evaluation 1: 5-Way Cumulative Regret Comparison
    # =========================================================================
    print("=" * 70)
    print("EVALUATION 1: Cumulative Regret Comparison (30 Seeds, 15,000 Rounds)")
    print("=" * 70)

    plt.figure(figsize=(11, 6))
    summary_regret = {}

    for name, (agent_class, kwargs) in agents_config.items():
        print(f"Simulating regret for {name}...")
        all_runs = []
        counts_sum = np.zeros(len(prices))

        for seed in range(n_seeds):
            regrets, counts = run_simulation(
                agent_class, kwargs, prices, n_rounds, seed, sim_opt
            )
            all_runs.append(regrets)
            counts_sum += counts

        mean_regret = np.mean(all_runs, axis=0)
        summary_regret[name] = mean_regret[-1]
        plt.plot(mean_regret, label=name, color=colors[name], linewidth=2.5)

        avg_counts = counts_sum / n_seeds
        print(f"  Final Mean Regret: ${mean_regret[-1]:.2f}")
        print(f"  Average Pulls for $80 (best arm for Professional): {avg_counts[-1]:.1f}")

    plt.xlabel("Round (t)", fontsize=12)
    plt.ylabel("Cumulative Regret ($)", fontsize=12)
    plt.title(
        f"5-Way Multi-Armed & Contextual Bandit Regret Comparison (n_seeds={n_seeds})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    out_path_regret = "runs/day4_5way_regret_comparison.png"
    plt.savefig(out_path_regret, dpi=150)
    print(f"Regret comparison plot saved to: {os.path.abspath(out_path_regret)}\n")

    # =========================================================================
    # Evaluation 2: Per-Context Price Convergence Plot
    # =========================================================================
    print("=" * 70)
    print("EVALUATION 2: Price Convergence Diagnostics (Seed 100)")
    print("=" * 70)

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    convergence_data = {}

    for i, (ctx_name, eval_ctx) in enumerate(target_contexts.items()):
        ax = axes[i]
        opt_price = optimal_prices_10[ctx_name]
        ax.axhline(y=opt_price, color="#666666", linestyle="--", alpha=0.8, linewidth=2, label="True Optimum")

        print(f"Tracking convergence for context: {ctx_name} (Optimum: ${opt_price:.2f})...")

        for name, (agent_class, kwargs) in agents_config.items():
            rounds, history = run_convergence_simulation(
                agent_class, kwargs, prices, n_rounds, seed=100,
                eval_interval=eval_interval, target_contexts={ctx_name: eval_ctx}
            )
            price_history = history[ctx_name]
            ax.plot(rounds, price_history, label=name, color=colors[name], linewidth=2, alpha=0.9)

            # Store convergence round for summary table
            conv_round = get_convergence_round(price_history, opt_price, eval_interval, n_rounds)
            if name not in convergence_data:
                convergence_data[name] = {}
            convergence_data[name][ctx_name] = conv_round

        ax.set_ylabel("Believed Price ($)", fontsize=11)
        ax.set_title(f"Price Convergence - {ctx_name}", fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylim(5, 95)
        if i == 0:
            ax.legend(loc="upper right", ncol=3, fontsize=9)

    axes[-1].set_xlabel("Round (t)", fontsize=12)
    plt.suptitle("5-Agent Price Convergence by Segment", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path_conv = "runs/day5_price_convergence.png"
    plt.savefig(out_path_conv, dpi=150)
    print(f"Price convergence plot saved to: {os.path.abspath(out_path_conv)}\n")

    # =========================================================================
    # Evaluation 3: Sensitivity Sweeps (Noise Level and Arm Count)
    # =========================================================================
    print("=" * 70)
    print("EVALUATION 3: Sensitivity Sweeps")
    print("=" * 70)

    # 3a. Noise Sweep
    noise_levels = [0.0, 0.1, 0.25]
    noise_results = {name: [] for name in agents_config}

    for noise in noise_levels:
        print(f"Running Noise Sweep (p_noise = {noise})...")
        for name, (agent_class, kwargs) in agents_config.items():
            avg_conv_rounds = []
            # Run convergence over all 3 contexts
            rounds, history = run_convergence_simulation(
                agent_class, kwargs, prices, n_rounds, seed=100,
                eval_interval=eval_interval, target_contexts=target_contexts, noise_level=noise
            )
            for ctx_name, opt_price in optimal_prices_10.items():
                conv_round = get_convergence_round(history[ctx_name], opt_price, eval_interval, n_rounds)
                avg_conv_rounds.append(conv_round)
            noise_results[name].append(np.mean(avg_conv_rounds))

    # 3b. Arm Grid Sweep (10 vs 20 Arms)
    prices_20 = sorted(list(set(prices + [12.0, 18.0, 22.0, 28.0, 32.0, 38.0, 45.0, 55.0, 60.0, 72.0])))
    optimal_prices_20 = {}
    for name, ctx in target_contexts.items():
        opt_price = find_grid_optimal_price(prices_20, ctx, sim_opt)
        optimal_prices_20[name] = opt_price

    arm_sweep_configs = {
        10: (prices, optimal_prices_10),
        20: (prices_20, optimal_prices_20),
    }
    arm_results = {name: [] for name in agents_config}

    for arm_count, (sweep_prices, sweep_opt_prices) in arm_sweep_configs.items():
        print(f"Running Arm Count Sweep (k_arms = {arm_count})...")
        for name, (agent_class, kwargs) in agents_config.items():
            avg_conv_rounds = []
            rounds, history = run_convergence_simulation(
                agent_class, kwargs, sweep_prices, n_rounds, seed=100,
                eval_interval=eval_interval, target_contexts=target_contexts
            )
            for ctx_name, opt_price in sweep_opt_prices.items():
                conv_round = get_convergence_round(history[ctx_name], opt_price, eval_interval, n_rounds)
                avg_conv_rounds.append(conv_round)
            arm_results[name].append(np.mean(avg_conv_rounds))

    # Plot Sensitivity Sweeps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Noise Sweep Plot
    for name in agents_config:
        axes[0].plot(noise_levels, noise_results[name], marker="o", color=colors[name], linewidth=2, label=name)
    axes[0].set_xlabel("Label Noise Level (p_noise)", fontsize=12)
    axes[0].set_ylabel("Avg Rounds to Convergence (3 contexts)", fontsize=12)
    axes[0].set_title("Noise Sensitivity Sweep", fontsize=13, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].set_xticks(noise_levels)

    # Arm Count Sweep Plot
    for name in agents_config:
        axes[1].plot([10, 20], arm_results[name], marker="s", color=colors[name], linewidth=2, label=name)
    axes[1].set_xlabel("Number of Price Arms (k)", fontsize=12)
    axes[1].set_ylabel("Avg Rounds to Convergence (3 contexts)", fontsize=12)
    axes[1].set_title("Price Grid Density Sweep", fontsize=13, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].set_xticks([10, 20])
    axes[1].legend(fontsize=9, loc="upper left")

    plt.suptitle("Agent Convergence Sensitivity Sweeps", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_path_sensitivity = "runs/day5_sensitivity_sweep.png"
    plt.savefig(out_path_sensitivity, dpi=150)
    print(f"Sensitivity sweeps plot saved to: {os.path.abspath(out_path_sensitivity)}\n")

    # =========================================================================
    # Report & Summary Tables
    # =========================================================================
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Agent':<30} | {'Mean Cum Regret':<15}")
    print("-" * 50)
    for name, regret_val in summary_regret.items():
        print(f"{name:<30} | ${regret_val:<14.2f}")
    print("\n")

    print(f"{'Agent':<30} | {'Student (rounds)':<18} | {'Prof (rounds)':<15} | {'Default (rounds)':<15}")
    print("-" * 88)
    for name in agents_config:
        s_conv = convergence_data[name]["Student (Weekday, Comp=$15)"]
        p_conv = convergence_data[name]["Professional (Weekend, Comp=$25)"]
        d_conv = convergence_data[name]["Default (Weekday, Comp=$20)"]
        s_str = f"{s_conv}" if s_conv < n_rounds else "DNQ"
        p_str = f"{p_conv}" if p_conv < n_rounds else "DNQ"
        d_str = f"{d_conv}" if d_conv < n_rounds else "DNQ"
        print(f"{name:<30} | {s_str:<18} | {p_str:<15} | {d_str:<15}")
    print("\n* DNQ = Did Not Converge within 15,000 rounds.")
    print("=" * 80)


if __name__ == "__main__":
    main()
