"""
BS Mode Selection Algorithms for Cell-Free ISAC Networks.

Implements greedy and heuristic-based mode selection to decide which BSs
operate in communication mode vs. sensing mode. The mode selection problem
is NP-hard; we provide practical suboptimal algorithms.
"""

import numpy as np
from system_model import CellFreeISACSystem, BSMode


class ModeSelector:
    """
    Base station mode selector for cooperative cell-free ISAC.

    Each BS independently selects between communication and sensing modes.
    The selection is optimized to maximize network-wide ISAC performance.
    """

    def __init__(self, system: CellFreeISACSystem):
        self.system = system

    def greedy_selection(self, min_comm_bs: int = 1, min_sens_bs: int = 1) -> np.ndarray:
        """
        Greedy mode selection based on channel quality.

        Assigns each BS to the mode where it has stronger average channel gain.

        Args:
            min_comm_bs: Minimum number of communication BSs.
            min_sens_bs: Minimum number of sensing BSs.

        Returns:
            Binary mode vector (n_bs,): 0=communication, 1=sensing.
        """
        n_bs = self.system.n_bs
        H_comm = self.system.get_communication_channels()

        # Compute average communication channel gain per BS
        comm_gain = np.zeros(n_bs)
        for m in range(n_bs):
            for k in range(self.system.n_users):
                comm_gain[m] += np.linalg.norm(H_comm[k, m]) ** 2
            comm_gain[m] /= self.system.n_users

        # Compute average sensing channel gain per BS
        sens_channels = self.system.get_sensing_channels()
        sens_gain = np.zeros(n_bs)
        for m in range(n_bs):
            for t in range(self.system.n_targets):
                a, pl = sens_channels[t][m]
                sens_gain[m] += pl * np.linalg.norm(a) ** 2
            sens_gain[m] /= self.system.n_targets

        # Normalize gains
        comm_norm = comm_gain / (np.max(comm_gain) + 1e-12)
        sens_norm = sens_gain / (np.max(sens_gain) + 1e-12)

        # Greedy: assign to mode with higher relative gain
        modes = np.zeros(n_bs, dtype=int)
        for m in range(n_bs):
            modes[m] = 1 if sens_norm[m] > comm_norm[m] else 0

        # Ensure minimum counts
        if np.sum(modes == 0) < min_comm_bs:
            # Force some BSs to communication
            sorted_by_sens = np.argsort(sens_norm - comm_norm)
            for idx in sorted_by_sens:
                if np.sum(modes == 0) >= min_comm_bs:
                    break
                modes[idx] = 0

        if np.sum(modes == 1) < min_sens_bs:
            # Force some BSs to sensing
            sorted_by_comm = np.argsort(comm_norm - sens_norm)
            for idx in sorted_by_comm:
                if np.sum(modes == 1) >= min_sens_bs:
                    break
                modes[idx] = 1

        return modes

    def channel_norm_selection(self) -> np.ndarray:
        """
        Mode selection based on Frobenius norm of channel matrices.

        BS with stronger communication channels serve users;
        BS with stronger sensing channels perform sensing.

        Returns:
            Binary mode vector (n_bs,).
        """
        n_bs = self.system.n_bs
        H_comm = self.system.get_communication_channels()
        sens_channels = self.system.get_sensing_channels()

        # Communication norm per BS
        comm_norm = np.array([
            np.sum(np.abs(H_comm[:, m, :]) ** 2) for m in range(n_bs)
        ])

        # Sensing norm per BS
        sens_norm = np.zeros(n_bs)
        for m in range(n_bs):
            for t in range(self.system.n_targets):
                a, pl = sens_channels[t][m]
                sens_norm[m] += pl ** 2 * np.linalg.norm(a) ** 2

        # Decision
        modes = (sens_norm > comm_norm).astype(int)

        # Ensure diversity
        if np.all(modes == 0):
            modes[np.argmax(sens_norm)] = 1
        elif np.all(modes == 1):
            modes[np.argmax(comm_norm)] = 0

        return modes

    def distance_based_selection(
        self,
        comm_radius: float = 200.0,
        sens_radius: float = 300.0,
    ) -> np.ndarray:
        """
        Mode selection based on distance to users and targets.

        BSs closer to users serve communication; BSs closer to targets
        perform sensing.

        Args:
            comm_radius: Maximum distance to consider BS for communication.
            sens_radius: Maximum distance to consider BS for sensing.

        Returns:
            Binary mode vector (n_bs,).
        """
        n_bs = self.system.n_bs
        modes = np.zeros(n_bs, dtype=int)

        for m in range(n_bs):
            bs_pos = self.system.bs_configs[m].position

            # Average distance to users
            user_dists = [
                np.linalg.norm(bs_pos - u.position)
                for u in self.system.user_configs
            ]
            avg_user_dist = np.mean(user_dists)

            # Average distance to targets
            if self.system.n_targets > 0:
                target_dists = [
                    np.linalg.norm(bs_pos - t.position)
                    for t in self.system.target_configs
                ]
                avg_target_dist = np.mean(target_dists)
            else:
                avg_target_dist = float('inf')

            # Select mode based on proximity
            if avg_target_dist < avg_user_dist and avg_target_dist < sens_radius:
                modes[m] = 1
            else:
                modes[m] = 0

        # Ensure at least one BS in each mode
        if np.sum(modes == 0) == 0:
            modes[0] = 0
        if np.sum(modes == 1) == 0:
            modes[-1] = 1

        return modes

    def equal_split_selection(self, sens_fraction: float = 0.5) -> np.ndarray:
        """
        Simple equal split: assign a fraction of BSs to sensing.

        Args:
            sens_fraction: Fraction of BSs in sensing mode.

        Returns:
            Binary mode vector (n_bs,).
        """
        n_bs = self.system.n_bs
        n_sens = max(1, int(np.round(n_bs * sens_fraction)))
        n_sens = min(n_sens, n_bs - 1)  # At least one comm BS

        modes = np.zeros(n_bs, dtype=int)

        # Sort BSs by sensing suitability (distance to nearest target)
        sens_scores = np.zeros(n_bs)
        for m in range(n_bs):
            bs_pos = self.system.bs_configs[m].position
            if self.system.n_targets > 0:
                min_dist = min(
                    np.linalg.norm(bs_pos - t.position)
                    for t in self.system.target_configs
                )
                sens_scores[m] = -min_dist  # Closer = better for sensing
            else:
                sens_scores[m] = 0

        sens_bs = np.argsort(sens_scores)[:n_sens]
        modes[sens_bs] = 1

        return modes

    def optimize(
        self,
        method: str = "greedy",
        **kwargs,
    ) -> np.ndarray:
        """
        Run mode selection optimization.

        Args:
            method: Selection method ('greedy', 'channel_norm', 'distance', 'equal_split').
            **kwargs: Additional arguments for the selected method.

        Returns:
            Optimized mode vector.
        """
        if method == "greedy":
            modes = self.greedy_selection(**kwargs)
        elif method == "channel_norm":
            modes = self.channel_norm_selection()
        elif method == "distance":
            modes = self.distance_based_selection(**kwargs)
        elif method == "equal_split":
            modes = self.equal_split_selection(**kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply to system
        self.system.set_mode_vector(modes)
        return modes
