"""
Module: envs.channel_model
Implementation of WirelessChannel providing Air-to-Ground (A2G) path loss
with 3GPP probabilistic LoS/NLoS modeling, directional jamming interference,
and Shannon capacity data rates.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class ChannelParameters:
    f_c: float = 2.0e9               # Carrier frequency (2 GHz)
    bandwidth: float = 1.0e6         # Bandwidth (1 MHz)
    noise_power_dbm: float = -104.0  # Noise power in dBm
    p_iot_dbm: float = 20.0          # IoT transmit power in dBm (0.1 W)
    p_jam_dbm: float = 30.0          # Jammer power in dBm (1.0 W)
    
    # 3GPP Urban Macro environment constants
    a_los: float = 9.61              # Sigmoid parameter a
    b_los: float = 0.16              # Sigmoid parameter b
    eta_los_db: float = 1.0          # Additional attenuation for LoS (dB)
    eta_nlos_db: float = 20.0        # Additional attenuation for NLoS (dB)
    
    # Directional Jammer Antenna
    jam_beamwidth_deg: float = 45.0  # Half-power beamwidth
    sidelobe_gain_db: float = -20.0  # Sidelobe attenuation


class WirelessChannel:
    """
    Simulates the wireless propagation environment, directional jamming, and throughput.
    """
    def __init__(self, params: ChannelParameters = None):
        self.params = params if params is not None else ChannelParameters()
        self.c = 3.0e8  # Speed of light (m/s)
        
        # Precompute linear power quantities
        self.noise_power = 10.0 ** ((self.params.noise_power_dbm - 30.0) / 10.0)
        self.p_iot = 10.0 ** ((self.params.p_iot_dbm - 30.0) / 10.0)
        self.p_jam = 10.0 ** ((self.params.p_jam_dbm - 30.0) / 10.0)

    def compute_los_probability(self, p_uav: np.ndarray, p_ground: np.ndarray) -> float:
        """
        Calculates 3GPP LoS probability based on elevation angle:
        P_LoS = 1 / (1 + a * exp(-b * (theta - a)))
        """
        rel = p_uav - p_ground
        dist_horizontal = max(1e-3, float(np.hypot(rel[0], rel[1])))
        dist_vertical = abs(float(rel[2]))
        
        elevation_angle_deg = np.degrees(np.arctan2(dist_vertical, dist_horizontal))
        p_los = 1.0 / (1.0 + self.params.a_los * np.exp(-self.params.b_los * (elevation_angle_deg - self.params.a_los)))
        return float(np.clip(p_los, 0.0, 1.0))

    def compute_channel_gain(self, p_uav: np.ndarray, p_ground: np.ndarray) -> float:
        """
        Calculates expected channel power gain:
        g(d) = P_LoS * g_LoS + (1 - P_LoS) * g_NLoS
        where Free Space Path Loss (FSPL) = (4 * pi * f_c * d / c)^2
        """
        d = max(1.0, float(np.linalg.norm(p_uav - p_ground)))
        fspl = (4.0 * np.pi * self.params.f_c * d / self.c) ** 2
        
        # Path loss with excess atmospheric/urban losses
        pl_los = fspl * (10.0 ** (self.params.eta_los_db / 10.0))
        pl_nlos = fspl * (10.0 ** (self.params.eta_nlos_db / 10.0))
        
        p_los = self.compute_los_probability(p_uav, p_ground)
        expected_pl = p_los * pl_los + (1.0 - p_los) * pl_nlos
        gain = 1.0 / expected_pl
        return gain

    def compute_jammer_gain(self, p_uav: np.ndarray, p_jam: np.ndarray) -> float:
        """
        Computes channel gain from a directional jammer with vertical conical beam.
        """
        rel = p_uav - p_jam
        dist = max(1.0, float(np.linalg.norm(rel)))
        
        # Directional antenna gain pattern
        cos_theta = rel[2] / dist
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        if angle_deg <= (self.params.jam_beamwidth_deg / 2.0):
            antenna_gain_linear = 1.0  # Main beam (0 dB)
        else:
            antenna_gain_linear = 10.0 ** (self.params.sidelobe_gain_db / 10.0)  # Sidelobe
            
        g_channel = self.compute_channel_gain(p_uav, p_jam)
        return g_channel * antenna_gain_linear

    def compute_sinr(self, p_uav: np.ndarray, p_iot: np.ndarray, p_jammers: list[np.ndarray]) -> float:
        """
        Computes SINR:
        SINR = (P_iot * g_iot) / (sigma^2 + sum_m P_jam * g_jam_m)
        """
        g_iot = self.compute_channel_gain(p_uav, p_iot)
        signal_power = self.p_iot * g_iot
        
        interference_power = 0.0
        for jam_pos in p_jammers:
            g_jam = self.compute_jammer_gain(p_uav, jam_pos)
            interference_power += self.p_jam * g_jam
            
        sinr = signal_power / (self.noise_power + interference_power)
        return float(sinr)

    def compute_data_rate(self, p_uav: np.ndarray, p_iot: np.ndarray, p_jammers: list[np.ndarray]) -> tuple[float, float]:
        """
        Computes instantaneous data rate R(t) in bits/second using Shannon capacity:
        R(t) = B * log2(1 + SINR)
        Returns: (data_rate_bps, sinr_linear)
        """
        sinr = self.compute_sinr(p_uav, p_iot, p_jammers)
        rate_bps = self.params.bandwidth * np.log2(1.0 + sinr)
        return rate_bps, sinr
