import os
import numpy as np
import matplotlib.pyplot as plt
from vantage.schemas import CustomerContext
from vantage.tools.simulator import MarketSimulator
from vantage.tools.optimization import find_optimal_price
from vantage.agents.bandit_agents import EpsilonGreedy, UCB1

def run_simulation(agent_class, agent_kwargs, context, prices, optimal_rev, n_rounds, seed):
    # Re-initialize simulator with the specific seed for reproducibility
    sim = MarketSimulator(seed=seed)
    # Create an isolated numpy random Generator for the agent to ensure fully independent and reproducible logic
    agent_rng = np.random.default_rng(seed)
    agent = agent_class(prices, rng=agent_rng, **agent_kwargs)
    
    regrets = []
    cumulative_regret = 0.0
    
    for _ in range(n_rounds):
        arm_idx = agent.select_arm()
        chosen_price = prices[arm_idx]
        
        # Pull the arm: simulate customer interaction (stochastic Bernoulli outcome)
        outcome = sim.step(chosen_price, context)
        reward = chosen_price * outcome
        
        # Calculate expected reward (revenue) for the chosen price to compute regret
        prob = sim.purchase_probability(chosen_price, context)
        expected_reward = chosen_price * prob
        
        # Regret = Optimal Expected Revenue - Expected Reward of Chosen Arm
        instant_regret = optimal_rev - expected_reward
        cumulative_regret += instant_regret
        regrets.append(cumulative_regret)
        
        # Update agent with the observed feedback
        agent.update(arm_idx, reward)
        
    return np.array(regrets), agent.counts

def main():
    # 1. Configuration
    n_rounds = 50000
    n_seeds = 20
    prices = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0, 80.0]
    
    # Freeze one context: Default segment on a Weekday with Competitor Price of $20
    context = CustomerContext(segment="default", day_type="weekday", competitor_price=20.0)
    
    # Calculate Ground-Truth Continuous Optimum for this context
    sim_meta = MarketSimulator(seed=42)
    opt_price, optimal_rev = find_optimal_price(context, sim_meta)
    
    print("=" * 60)
    print("MAB Pricing Evaluation Setup:")
    print(f"Context: {context.segment} | {context.day_type} | Comp Price: ${context.competitor_price:.2f}")
    print(f"Continuous Optimal Price (p*): ${opt_price:.2f}")
    print(f"Optimal Expected Revenue: ${optimal_rev:.2f}")
    print(f"Candidate Price Arms: {prices}")
    print(f"Rounds (T): {n_rounds} | Seeds: {n_seeds}")
    print("=" * 60)
    
    egreedy_regrets_all = []
    ucb1_regrets_all = []
    
    egreedy_counts_sum = np.zeros(len(prices))
    ucb1_counts_sum = np.zeros(len(prices))
    
    # 2. Run Simulations
    for seed in range(n_seeds):
        # Run Epsilon-Greedy (epsilon = 0.1)
        eg_regret, eg_counts = run_simulation(
            EpsilonGreedy, {"epsilon": 0.1}, context, prices, optimal_rev, n_rounds, seed
        )
        egreedy_regrets_all.append(eg_regret)
        egreedy_counts_sum += eg_counts
        
        # Run UCB1
        ucb_regret, ucb_counts = run_simulation(
            UCB1, {}, context, prices, optimal_rev, n_rounds, seed
        )
        ucb1_regrets_all.append(ucb_regret)
        ucb1_counts_sum += ucb_counts
        
    # Calculate Mean Regrets over all seeds
    egreedy_mean_regrets = np.mean(egreedy_regrets_all, axis=0)
    ucb1_mean_regrets = np.mean(ucb1_regrets_all, axis=0)
    
    # 3. Print Pull Distributions
    print("\nArm Pull Distribution (Averaged over all seeds):")
    print(f"{'Price Arm':<10} | {'Epsilon-Greedy (e=0.1)':<22} | {'UCB1':<10}")
    print("-" * 50)
    for i, p in enumerate(prices):
        eg_avg = egreedy_counts_sum[i] / n_seeds
        ucb_avg = ucb1_counts_sum[i] / n_seeds
        print(f"${p:<9.1f} | {eg_avg:<22.1f} | {ucb_avg:<10.1f}")
    print("-" * 50)
    
    # 4. Save and Plot Regret Curves
    os.makedirs("runs", exist_ok=True)
    plt.figure(figsize=(11, 6))
    
    # Set premium aesthetic styling
    plt.plot(egreedy_mean_regrets, label="Epsilon-Greedy (e=0.1)", color="#ff8a3d", linewidth=2.5)
    plt.plot(ucb1_mean_regrets, label="UCB1", color="#6ec8e8", linewidth=2.5)
    
    plt.title("Multi-Armed Bandit Comparison: Cumulative Regret", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Round (t)", fontsize=12)
    plt.ylabel("Cumulative Regret ($)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    out_path = "runs/regret_curves_egreedy_vs_ucb1.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nRegret curves plot saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
