import os
import pickle
import asyncio
from typing import Dict, Tuple
import numpy as np
from vantage.agents.linucb_agent import LinUCB
from vantage.schemas import CustomerContext

STATE_FILE = "data/agent_state.pkl"
PRICES = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 65.0, 80.0]


class AppState:
    """Manages global thread-safe state, request correlation, and snapshot persistence."""

    def __init__(self):
        self.agent: LinUCB = None
        self.lock = asyncio.Lock()
        # Maps request_id -> (arm_idx, context_vector, CustomerContext)
        self.pending_requests: Dict[str, Tuple[int, np.ndarray, CustomerContext]] = {}
        self.update_count = 0
        self.save_interval = 10  # Save snapshot every 10 updates

    def initialize(self) -> None:
        """Loads agent from disk snapshot or initializes a fresh Joint LinUCB agent."""
        os.makedirs("data", exist_ok=True)
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "rb") as f:
                    state_dict = pickle.load(f)

                # Initialize fresh agent and load parameters
                self.agent = LinUCB(PRICES, d=5, scaled_bonus=True)
                self.agent.A = state_dict["A"]
                self.agent.b = state_dict["b"]
                self.agent.counts = state_dict["counts"]
                self.agent.total_rounds = state_dict["total_rounds"]
                self.agent.q_estimates = state_dict["q_estimates"]
                print(f"Loaded existing agent state from {STATE_FILE}. Total rounds: {self.agent.total_rounds}")
            except Exception as e:
                print(f"Error loading state file: {e}. Initializing fresh Joint LinUCB agent.")
                self.agent = LinUCB(PRICES, d=5, scaled_bonus=True)
        else:
            print("No state snapshot found. Initializing fresh Joint LinUCB agent.")
            self.agent = LinUCB(PRICES, d=5, scaled_bonus=True)

    def save_snapshot(self) -> None:
        """Serializes current agent parameters to disk."""
        try:
            state_dict = {
                "A": self.agent.A,
                "b": self.agent.b,
                "counts": self.agent.counts,
                "total_rounds": self.agent.total_rounds,
                "q_estimates": self.agent.q_estimates
            }
            # Save to a temporary file first, then rename, for atomic writes
            tmp_file = STATE_FILE + ".tmp"
            with open(tmp_file, "wb") as f:
                pickle.dump(state_dict, f)
            os.replace(tmp_file, STATE_FILE)
            print(f"Saved agent state snapshot. Total rounds: {self.agent.total_rounds}")
        except Exception as e:
            print(f"Error saving state snapshot: {e}")

    def add_pending_request(self, request_id: str, arm_idx: int, x: np.ndarray, ctx: CustomerContext) -> None:
        """Registers a price offer context, with a simple memory overflow guard."""
        if len(self.pending_requests) > 10000:
            # Evict oldest 1000 requests to prevent leaks from abandoned transactions
            oldest_keys = list(self.pending_requests.keys())[:1000]
            for key in oldest_keys:
                self.pending_requests.pop(key, None)
        self.pending_requests[request_id] = (arm_idx, x, ctx)


# Global singleton instance of AppState
state = AppState()
