import os
import matplotlib.pyplot as plt
import numpy as np
from vantage.schemas import CustomerContext
from vantage.tools.simulator import MarketSimulator
from vantage.tools.optimization import expected_revenue

def main():
    # Initialize simulator
    sim = MarketSimulator(seed=42)
    
    # Define contexts to compare
    contexts = {
        "Student (Weekday, Comp=$20)": CustomerContext(segment="student", day_type="weekday", competitor_price=20.0),
        "Professional (Weekday, Comp=$20)": CustomerContext(segment="professional", day_type="weekday", competitor_price=20.0),
        "Default (Weekday, Comp=$20)": CustomerContext(segment="default", day_type="weekday", competitor_price=20.0),
    }
    
    prices = np.linspace(1.0, 160, 1000)
    
    # Set up matplotlib figure with 2 subplots (1 row, 2 columns)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = {"student": "#ff8a3d", "professional": "#6ec8e8", "default": "#b39ddb"}
    
    for label, ctx in contexts.items():
        color = colors[ctx.segment]
        probs = [sim.purchase_probability(p, ctx) for p in prices]
        revenues = [expected_revenue(p, ctx, sim) for p in prices]
        
        # Subplot 1: Demand curves (Purchase probability vs Price)
        ax1.plot(prices, probs, label=label, color=color, linewidth=2)
        
        # Subplot 2: Expected Revenue curves
        ax2.plot(prices, revenues, label=label, color=color, linewidth=2)
        
    # Formatting Subplot 1
    ax1.set_title("Demand Curves (Purchase Probability)", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Price ($)", fontsize=12)
    ax1.set_ylabel("Purchase Probability", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    ax1.set_ylim(0, 1.05)
    
    # Formatting Subplot 2
    ax2.set_title("Expected Revenue Curves", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Price ($)", fontsize=12)
    ax2.set_ylabel("Expected Revenue ($)", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)
    
    # Save the figure
    os.makedirs("runs", exist_ok=True)
    out_path = "runs/day1_demand_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Demand curves visualized and saved to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
