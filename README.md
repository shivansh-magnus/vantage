# Vantage: Personalized Dynamic Pricing Engine

Vantage is a production-grade personalized dynamic pricing engine designed to optimize product pricing in real-time. It leverages advanced reinforcement learning and bandit algorithms to personalize pricing strategies for distinct customer segments under varying context conditions (such as weekday vs. weekend patterns and competitor pricing).

The system transitions from standard point-estimate baseline models to context-aware linear reward models, achieving optimal revenue generation while minimizing cumulative regret.

---

## 🚀 System Architecture & Agent Roster

The core engine supports five different pricing bandit agents under a unified interface:

1. **Epsilon-Greedy** (`EpsilonGreedy`): Evaluates prices based on a running average reward (revenue) estimate, exploring randomly with probability $\epsilon$.
2. **UCB1** (`UCB1`): A standard multi-armed bandit that balances exploration and exploitation using an upper confidence bound based on Chernoff-Hoeffding bounds. Corrected to track conversion probabilities and price-scale updates.
3. **Thompson Sampling** (`ThompsonSampling`): A Bayesian agent using Beta-Bernoulli conjugate priors to maintain a full probability distribution over purchase conversions, selecting prices using price-scaled posterior sampling.
4. **Joint LinUCB** (`LinUCB`): A contextual linear bandit that models expected revenue as a linear function of a 5-dimensional context vector, utilizing Ridge regression and a price-scaled exploration confidence ellipsoid.
5. **Segment-Separated LinUCB** (`SegmentSeparatedLinUCB`): A contextual agent that instantiates separate LinUCB instances per customer segment (Student, Professional, Default) to eliminate cross-segment parameter poisoning.

---

## 📂 Repository Layout

```filepath
Vantage/
├── src/vantage/
│   ├── schemas.py              # Customer context & data model schemas
│   ├── tools/
│   │   ├── simulator.py        # Bernoulli market simulator & context sampler
│   │   └── optimization.py    # SciPy numerical bounds solver & optimal revenue calculations
│   └── agents/
│       ├── __init__.py         # Module exports
│       ├── bandit_agents.py    # Epsilon-Greedy, UCB1 baseline classes
│       ├── thompson_sampling.py# Thompson Sampling Bayesian implementation
│       └── linucb_agent.py     # Disjoint and Segment-Separated LinUCB implementations
├── scripts/
│   ├── new_evaluate_all_agents.py # 5-way regret comparison, convergence, & sensitivity sweeps
│   ├── evaluate_bandit_agents.py  # Regret evaluation script
│   ├── evaluate_linucb.py         # Convergence evaluation script
│   └── generate_ground_truth.py   # Ground-truth table compiler
└── tests/
    ├── test_simulator.py       # Simulator validation tests
    ├── test_optimization.py    # SciPy optimizer tests
    ├── test_bandit_agents.py   # EpsilonGreedy & UCB1 test suite
    ├── test_thompson_sampling.py# Thompson Sampling test suite
    └── test_linucb_agent.py    # LinUCB & Segment-Separated LinUCB test suite
```

---

## 📈 Evaluation Results

### 1. Cumulative Regret Comparison (30 Seeds, 15,000 Rounds)
Non-contextual algorithms (Epsilon-Greedy, UCB1, Thompson Sampling) show **linear regret growth** because they must charge a compromise price across all segments. Contextual models (Joint LinUCB and Segment-Separated LinUCB) exhibit **flat, logarithmic (sub-linear) regret curves**, saving nearly **10x** in lost revenue over 15,000 rounds:
* **Thompson Sampling (Best Non-Contextual)**: \$181,726.47
* **Segment-Separated LinUCB (Best Contextual)**: \$21,725.43

### 2. Price Convergence by Segment
The price convergence diagnostics track whether the agent's greedy pricing converges to the true optimal price for specific contexts:
* **Student Segment (Optimal: \$15)**: Contextual models converge to the vicinity of \$15. Non-contextual models select compromise prices (such as \$35 or \$50) for students, leaving revenue on the table.
* **Professional Segment (Optimal: \$80)**: Joint LinUCB and Segment-Separated LinUCB successfully converge to the optimal \$80 price. (Early cross-segment poisoning in standard LinUCB, which locked professionals out of \$80, was resolved via sub-agent isolation).

---

## 🔧 Installation & Verification

### Prerequisites
Make sure Python is installed. We recommend using `uv` or standard `pip` inside a virtual environment.

### Setup
```bash
# Clone the repository
git clone https://github.com/shivansh-magnus/vantage.git
cd vantage

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Running Tests
Execute the unit and integration test suite to verify the mathematical updates, routing logic, and regret boundaries:
```bash
pytest
```

### Running Evaluations
Run the full 5-way simulation harness to perform regret analysis, convergence diagnostics, and sensitivity sweeps, generating the comparison plots under `runs/`:
```bash
python scripts/new_evaluate_all_agents.py
```
