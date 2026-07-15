import os
import json
from vantage.schemas import CustomerContext, GroundTruthEntry
from vantage.tools.simulator import MarketSimulator
from vantage.tools.optimization import find_optimal_price

def main():
    # Initialize simulator
    sim = MarketSimulator(seed=42)
    
    segments = ["student", "professional", "default"]
    day_types = ["weekday", "weekend"]
    competitor_prices = [15.0, 20.0, 25.0]
    
    entries = []
    
    print("=" * 70)
    print(f"{'Segment':<15} | {'Day Type':<10} | {'Comp Price':<10} | {'Opt Price':<10} | {'Expected Rev':<12}")
    print("=" * 70)
    
    for segment in segments:
        for day_type in day_types:
            for comp_price in competitor_prices:
                ctx = CustomerContext(
                    segment=segment,
                    day_type=day_type,
                    competitor_price=comp_price
                )
                
                # Find optimal price and maximum expected revenue
                opt_price, max_rev = find_optimal_price(ctx, sim)
                
                # Format to GroundTruthEntry Pydantic model
                entry = GroundTruthEntry(
                    segment=segment,
                    day_type=day_type,
                    competitor_price=comp_price,
                    optimal_price=round(opt_price, 2),
                    expected_revenue=round(max_rev, 2)
                )
                
                entries.append(entry.model_dump())
                
                print(f"{segment:<15} | {day_type:<10} | ${comp_price:<9.2f} | ${opt_price:<9.2f} | ${max_rev:<11.2f}")
                
    print("=" * 70)
    
    # Save to data/raw/ground_truth.json
    os.makedirs(os.path.join("data", "raw"), exist_ok=True)
    out_path = os.path.join("data", "raw", "ground_truth.json")
    
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=4)
        
    print(f"Ground-truth table saved successfully to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
