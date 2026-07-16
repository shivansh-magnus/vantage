import os
import numpy as np
import matplotlib.pyplot as plt
from vantage.schemas import CustomerContext
from vantage.tools.simulator import MarketSimulator
from vantage.tools.optimization import find_optimal_price
from vantage.agents.linucb_agent import LinUCB


def main():
    # 1. Define target contexts to track
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
    d = 5  # intercept, is_student, is_professional, is_weekend, competitor_price

    # Initialize simulator to compute analytical optima
    sim_opt = MarketSimulator(seed=42)
    optimal_prices = {}
    for name, ctx in target_contexts.items():
        opt_price, _ = find_optimal_price(ctx, sim_opt)
        optimal_prices[name] = opt_price

    print("=" * 60)
    print("LinUCB Evaluation Setup:")
    for name, opt in optimal_prices.items():
        print(f"  True Optimal Price for {name}: ${opt:.2f}")
    print(f"Candidate Price Arms: {prices}")
    print("=" * 60)

    # Initialize simulator and LinUCB agent for the simulation run
    sim = MarketSimulator(seed=100)
    agent = LinUCB(
        prices, d=d, lmbda=1.0, alpha=1.0, rng=np.random.default_rng(100)
    )

    n_rounds = 15000
    eval_interval = 50

    history = {name: [] for name in target_contexts}
    rounds = []

    for r in range(1, n_rounds + 1):
        # Sample context stochastically from the simulator (varying context)
        ctx = sim.sample_context()
        x = ctx.to_vector()

        # Select price and step
        arm_idx = agent.select_arm(x)
        chosen_price = prices[arm_idx]

        outcome = sim.step(chosen_price, ctx)
        reward = chosen_price * outcome

        # Update agent
        agent.update(arm_idx, reward, x)

        # Periodically evaluate greedy selection for the target contexts
        if r % eval_interval == 0:
            rounds.append(r)
            for name, eval_ctx in target_contexts.items():
                x_eval = eval_ctx.to_vector()

                # Find greedy price: argmax_a x^T * theta_hat
                preds = []
                for a in range(agent.k_arms):
                    A_inv = np.linalg.inv(agent.A[a])
                    theta_hat = A_inv @ agent.b[a]
                    preds.append(theta_hat @ x_eval)

                greedy_arm = np.argmax(preds)
                history[name].append(prices[greedy_arm])

    # Plot results
    plt.figure(figsize=(12, 7))

    colors = {
        "Student (Weekday, Comp=$15)": "#ff8a3d",
        "Professional (Weekend, Comp=$25)": "#6ec8e8",
        "Default (Weekday, Comp=$20)": "#b39ddb",
    }

    for name, history_prices in history.items():
        color = colors[name]
        # Plot learned prices over time
        plt.plot(
            rounds,
            history_prices,
            label=f"Learned Price: {name}",
            color=color,
            linewidth=2.5,
        )
        # Plot horizontal line for true optimum
        plt.axhline(
            y=optimal_prices[name],
            color=color,
            linestyle="--",
            alpha=0.7,
            label=f"True Optimum: {name}",
        )

    plt.xlabel("Round (t)", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.title(
        "LinUCB Contextual Pricing Convergence Over Time",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)
    plt.tight_layout()

    os.makedirs("runs", exist_ok=True)
    out_path = "runs/day4_linucb_convergence.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nConvergence plot saved to: {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()
