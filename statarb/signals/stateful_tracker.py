from typing import Dict, Tuple
import numpy as np
import pandas as pd


class StatefulPairSignalTracker:
    def __init__(self, z_entry: float = 2.0, z_exit: float = 0.5, tau_decay: float = 15.0):
        self.z_entry = z_entry
        self.z_exit = z_exit
        self.tau_decay = tau_decay
        self.pair_states: Dict[Tuple[str, str], int] = {}

    def compute_pair_signal(
        self,
        asset_A: str,
        asset_B: str,
        z_score: float,
        beta: float,
        half_life: float,
    ) -> Tuple[float, float]:
        pair_key = (asset_A, asset_B)
        current_state = self.pair_states.get(pair_key, 0)

        # Updated hysteresis logic with stop-loss guardrail
        if abs(z_score) >= 3.5:
            new_state = 0  # Force exit on spread divergence / cointegration break
        elif z_score >= self.z_entry:
            new_state = -1
        elif z_score <= -self.z_entry:
            new_state = 1
        elif abs(z_score) <= self.z_exit:
            new_state = 0
        else:
            new_state = current_state

        self.pair_states[pair_key] = new_state

        if new_state == 0 or not (0.5 <= half_life <= 60.0):
            return 0.0, 0.0

        decay_weight = np.exp(-half_life / self.tau_decay)
        effective_z = abs(z_score) - self.z_exit
        intensity = new_state * np.arctan(effective_z) * decay_weight

        # Dollar-neutral allocation per pair
        denom = 1.0 + abs(beta)
        weight_A = (1.0 / denom) * intensity
        weight_B = (-beta / denom) * intensity

        return weight_A, weight_B