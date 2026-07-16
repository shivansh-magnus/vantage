import os
import numpy as np
import matplotlib.pyplot as plt
from vantage.schemas import CustomerContext
from vantage.tools.simulator import MarketSimulator
from vantage.tools.optimization import find_optimal_price
from vantage.agents.linucb_agent import LinUCB, SegmentSeparatedLinUCB


def main():
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

    prices = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 65.0, 80.0]
    n_rounds = 15000
    eval_interval = 50

    # Calculate optimal prices
    sim_opt = MarketSimulator(seed=42)
    optimal_prices = {}
    for name, ctx in target_contexts.items():
        opt_price, _ = find_optimal_price(ctx, sim_opt)
        optimal_prices[name] = opt_price

    print("=" * 60)
    print("Evaluating Both LinUCB Agents:")
    for name, opt in optimal_prices.items():
        print(f"  True Optimal Price for {name}: ${opt:.2f}")
    print("=" * 60)

    # 1. Run Joint LinUCB (scaled & normalized)
    print("Running Joint LinUCB simulation...")
    sim_joint = MarketSimulator(seed=100)
    agent_joint = LinUCB(
        prices, d=5, lmbda=1.0, alpha=1.0, scaled_bonus=True, rng=np.random.default_rng(100)
    )
    history_joint = {name: [] for name in target_contexts}
    rounds = []

    for r in range(1, n_rounds + 1):
        ctx = sim_joint.sample_context()
        x = ctx.to_vector()
        
        # Normalize competitor price
        x_agent = x.copy()
        x_agent[4] /= 20.0

        arm_idx = agent_joint.select_arm(x_agent)
        chosen_price = prices[arm_idx]
        outcome = sim_joint.step(chosen_price, ctx)
        reward = chosen_price * outcome
        agent_joint.update(arm_idx, reward, x_agent)

        if r % eval_interval == 0:
            rounds.append(r)
            for name, eval_ctx in target_contexts.items():
                x_eval = eval_ctx.to_vector()
                x_eval_agent = x_eval.copy()
                x_eval_agent[4] /= 20.0

                preds = []
                for a in range(agent_joint.k_arms):
                    A_inv = np.linalg.inv(agent_joint.A[a])
                    theta_hat = A_inv @ agent_joint.b[a]
                    preds.append(theta_hat @ x_eval_agent)

                greedy_arm = np.argmax(preds)
                history_joint[name].append(prices[greedy_arm])

    # 2. Run Segment-Separated LinUCB
    print("Running Segment-Separated LinUCB simulation...")
    sim_sep = MarketSimulator(seed=100)
    agent_sep = SegmentSeparatedLinUCB(
        prices, lmbda=1.0, alpha=1.0, rng=np.random.default_rng(100)
    )
    history_sep = {name: [] for name in target_contexts}

    for r in range(1, n_rounds + 1):
        ctx = sim_sep.sample_context()
        
        arm_idx = agent_sep.select_arm(ctx)
        chosen_price = prices[arm_idx]
        outcome = sim_sep.step(chosen_price, ctx)
        reward = chosen_price * outcome
        agent_sep.update(arm_idx, reward, ctx)

        if r % eval_interval == 0:
            for name, eval_ctx in target_contexts.items():
                eval_agent = agent_sep.agents[eval_ctx.segment]
                _, x_eval_sub = agent_sep._get_segment_and_sub_context(eval_ctx)

                preds = []
                for a in range(eval_agent.k_arms):
                    A_inv = np.linalg.inv(eval_agent.A[a])
                    theta_hat = A_inv @ eval_agent.b[a]
                    preds.append(theta_hat @ x_eval_sub)

                greedy_arm = np.argmax(preds)
                history_sep[name].append(prices[greedy_arm])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    colors = {
        "Student (Weekday, Comp=$15)": "#ff8a3d",
        "Professional (Weekend, Comp=$25)": "#6ec8e8",
        "Default (Weekday, Comp=$20)": "#b39ddb",
    }

    # Plot Joint LinUCB
    ax_j = axes[0]
    for name, hist in history_joint.items():
        color = colors[name]
        ax_j.plot(rounds, hist, label=f"Learned: {name}", color=color, linewidth=2.5)
        ax_j.axhline(y=optimal_prices[name], color=color, linestyle="--", alpha=0.7)
    ax_j.set_xlabel("Round (t)", fontsize=12)
    ax_j.set_ylabel("Price ($)", fontsize=12)
    ax_j.set_title("Joint LinUCB (Scaled & Normalized)", fontsize=13, fontweight="bold")
    ax_j.grid(True, linestyle="--", alpha=0.5)
    ax_j.legend(loc="upper left")

    # Plot Segment-Separated LinUCB
    ax_s = axes[1]
    for name, hist in history_sep.items():
        color = colors[name]
        ax_s.plot(rounds, hist, label=f"Learned: {name}", color=color, linewidth=2.5)
        ax_s.axhline(y=optimal_prices[name], color=color, linestyle="--", alpha=0.7)
    ax_s.set_xlabel("Round (t)", fontsize=12)
    ax_s.set_title("Segment-Separated LinUCB (Scaled)", fontsize=13, fontweight="bold")
    ax_s.grid(True, linestyle="--", alpha=0.5)
    ax_s.legend(loc="upper left")

    plt.suptitle("LinUCB Convergence Comparison", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    os.makedirs("runs", exist_ok=True)
    out_path = "runs/day4_linucb_convergence.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nConvergence plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
