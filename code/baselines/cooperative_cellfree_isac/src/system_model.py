"""
Cell-Free ISAC Network System Model.

Models a cell-free network with multiple BSs, each operating in either
communication or sensing mode. Includes channel models, geometry, and
configuration management.
"""

import numpy as np
from enum import IntEnum
from dataclasses import dataclass, field
from typing import Optional


class BSMode(IntEnum):
    """Base station operating mode."""
    COMMUNICATION = 0
    SENSING = 1


@dataclass
class BSConfig:
    """Configuration for a single base station."""
    position: np.ndarray  # (2,) or (3,) position in meters
    n_antennas: int = 4
    p_max_dbm: float = 30.0  # Maximum transmit power in dBm
    mode: BSMode = BSMode.COMMUNICATION
    beamforming_vector: Optional[np.ndarray] = None

    @property
    def p_max(self) -> float:
        """Maximum transmit power in linear scale (Watts)."""
        return 10 ** (self.p_max_dbm / 10) / 1000


@dataclass
class UserConfig:
    """Configuration for a communication user."""
    position: np.ndarray  # (2,) or (3,) position in meters
    min_rate: float = 0.1  # Minimum rate requirement in bps/Hz


@dataclass
class TargetConfig:
    """Configuration for a sensing target."""
    position: np.ndarray  # (2,) or (3,) position in meters
    radar_cross_section: float = 1.0  # RCS in m^2
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))


class CellFreeISACSystem:
    """
    Cell-Free ISAC Network System Model.

    Models a cooperative cell-free network where multiple distributed BSs
    serve communication users and sense targets. Each BS can independently
    choose between communication and sensing modes.

    Attributes:
        n_bs: Number of base stations
        n_users: Number of communication users
        n_targets: Number of sensing targets
        bandwidth: System bandwidth in Hz
        carrier_freq: Carrier frequency in Hz
        noise_power_dbm: Noise power in dBm
        bs_configs: List of BS configurations
        user_configs: List of user configurations
        target_configs: List of target configurations
    """

    # Physical constants
    SPEED_OF_LIGHT = 3e8  # m/s

    def __init__(
        self,
        n_bs: int = 8,
        n_users: int = 4,
        n_targets: int = 2,
        area_size: float = 500.0,
        n_antennas: int = 4,
        bandwidth: float = 10e6,
        carrier_freq: float = 3.5e9,
        noise_power_dbm: float = -174 + 10 * np.log10(10e6),
        p_max_dbm: float = 30.0,
        seed: Optional[int] = None,
    ):
        self.n_bs = n_bs
        self.n_users = n_users
        self.n_targets = n_targets
        self.area_size = area_size
        self.n_antennas = n_antennas
        self.bandwidth = bandwidth
        self.carrier_freq = carrier_freq
        self.noise_power_dbm = noise_power_dbm
        self.p_max_dbm = p_max_dbm
        self.rng = np.random.default_rng(seed)

        # Wavelength
        self.wavelength = self.SPEED_OF_LIGHT / carrier_freq

        # Generate positions and configurations
        self.bs_configs = self._generate_bs_configs()
        self.user_configs = self._generate_user_configs()
        self.target_configs = self._generate_target_configs()

        # Pre-compute channels
        self._H_comm = None  # Communication channels (n_users, n_bs, n_antennas)
        self._H_sens = None  # Sensing channels (n_targets, n_bs, n_antennas)

    def _generate_bs_configs(self) -> list[BSConfig]:
        """Generate BS positions uniformly in the area."""
        configs = []
        for _ in range(self.n_bs):
            pos = self.rng.uniform(0, self.area_size, size=2)
            configs.append(BSConfig(
                position=pos,
                n_antennas=self.n_antennas,
                p_max_dbm=self.p_max_dbm,
            ))
        return configs

    def _generate_user_configs(self) -> list[UserConfig]:
        """Generate user positions uniformly in the area."""
        configs = []
        for _ in range(self.n_users):
            pos = self.rng.uniform(0, self.area_size, size=2)
            configs.append(UserConfig(position=pos))
        return configs

    def _generate_target_configs(self) -> list[TargetConfig]:
        """Generate target positions uniformly in the area."""
        configs = []
        for _ in range(self.n_targets):
            pos = self.rng.uniform(0, self.area_size, size=2)
            vel = self.rng.uniform(-10, 10, size=2)
            configs.append(TargetConfig(position=pos, velocity=vel))
        return configs

    def path_loss(self, distance: float, d0: float = 1.0) -> float:
        """
        Compute path loss using 3GPP UMi model.

        Args:
            distance: Distance in meters.
            d0: Reference distance in meters.

        Returns:
            Linear path loss coefficient (PL = d^{-alpha} * PL_0).
        """
        if distance < d0:
            distance = d0

        # 3GPP UMi path loss: PL(dB) = 36.7*log10(d) + 22.7 + 26*log10(fc)
        fc_ghz = self.carrier_freq / 1e9
        pl_db = 36.7 * np.log10(distance) + 22.7 + 26 * np.log10(fc_ghz)
        return 10 ** (-pl_db / 10)

    def compute_channel_bs_user(self, bs_idx: int, user_idx: int) -> np.ndarray:
        """
        Compute channel vector between BS and user.

        Models: LoS + NLoS components with array response.

        Args:
            bs_idx: BS index.
            user_idx: User index.

        Returns:
            Channel vector of shape (n_antennas,).
        """
        bs = self.bs_configs[bs_idx]
        user = self.user_configs[user_idx]

        distance = np.linalg.norm(bs.position - user.position)
        pl = self.path_loss(distance)

        # AoA (Angle of Arrival) at BS
        delta = user.position - bs.position
        theta = np.arctan2(delta[1], delta[0])

        # ULA array response
        d_spacing = self.wavelength / 2
        n = np.arange(self.n_antennas)
        array_response = np.exp(1j * 2 * np.pi * d_spacing / self.wavelength * n * np.sin(theta))

        # Rician fading (K = 10 dB)
        K_dB = 10
        K = 10 ** (K_dB / 10)
        h_los = np.sqrt(K / (K + 1)) * array_response
        h_nlos = np.sqrt(1 / (K + 1)) * (self.rng.standard_normal(self.n_antennas) +
                                           1j * self.rng.standard_normal(self.n_antennas)) / np.sqrt(2)
        h = np.sqrt(pl) * (h_los + h_nlos)

        return h

    def compute_channel_bs_target(self, bs_idx: int, target_idx: int) -> tuple[np.ndarray, float]:
        """
        Compute sensing channel (array response + path loss) between BS and target.

        Args:
            bs_idx: BS index.
            target_idx: Target index.

        Returns:
            Tuple of (array_response_vector, path_loss_scalar).
        """
        bs = self.bs_configs[bs_idx]
        target = self.target_configs[target_idx]

        distance = np.linalg.norm(bs.position - target.position)
        pl = self.path_loss(distance)

        # AoA/AoD at BS
        delta = target.position - bs.position
        theta = np.arctan2(delta[1], delta[0])

        # ULA array response
        d_spacing = self.wavelength / 2
        n = np.arange(self.n_antennas)
        a = np.exp(1j * 2 * np.pi * d_spacing / self.wavelength * n * np.sin(theta))

        return a, pl

    def get_communication_channels(self) -> np.ndarray:
        """
        Get all communication channels.

        Returns:
            Channel matrix of shape (n_users, n_bs, n_antennas).
        """
        if self._H_comm is None:
            H = np.zeros((self.n_users, self.n_bs, self.n_antennas), dtype=complex)
            for k in range(self.n_users):
                for m in range(self.n_bs):
                    H[k, m] = self.compute_channel_bs_user(m, k)
            self._H_comm = H
        return self._H_comm

    def get_sensing_channels(self) -> list[list[tuple[np.ndarray, float]]]:
        """
        Get all sensing channels.

        Returns:
            List of list of (array_response, path_loss) for each (target, BS).
        """
        channels = []
        for t in range(self.n_targets):
            target_channels = []
            for m in range(self.n_bs):
                a, pl = self.compute_channel_bs_target(m, t)
                target_channels.append((a, pl))
            channels.append(target_channels)
        return channels

    @property
    def noise_power(self) -> float:
        """Noise power in linear scale (Watts)."""
        return 10 ** (self.noise_power_dbm / 10) / 1000

    def get_mode_vector(self) -> np.ndarray:
        """
        Get current mode selection vector.

        Returns:
            Binary vector of shape (n_bs,): 1 for sensing, 0 for communication.
        """
        return np.array([bs.mode for bs in self.bs_configs], dtype=int)

    def set_mode_vector(self, modes: np.ndarray):
        """
        Set BS modes.

        Args:
            modes: Binary vector of shape (n_bs,).
        """
        for m in range(self.n_bs):
            self.bs_configs[m].mode = BSMode(int(modes[m]))

    def reset_channels(self):
        """Clear cached channels to force re-computation."""
        self._H_comm = None
        self._H_sens = None

    def __repr__(self) -> str:
        comm = sum(1 for bs in self.bs_configs if bs.mode == BSMode.COMMUNICATION)
        sens = sum(1 for bs in self.bs_configs if bs.mode == BSMode.SENSING)
        return (
            f"CellFreeISACSystem(n_bs={self.n_bs}, users={self.n_users}, "
            f"targets={self.n_targets}, comm_bs={comm}, sens_bs={sens})"
        )
