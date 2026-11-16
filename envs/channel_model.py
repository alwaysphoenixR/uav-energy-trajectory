"""
Air-to-Ground Wireless Channel Model, SINR Calculation, and Directional Jamming.
"""
import numpy as np
from configs.settings import SimulationConfig

class AirToGroundChannel:
    def __init__(self, sim_cfg: SimulationConfig):
        self.sim_cfg = sim_cfg
        
    def compute_channel_gain(self, uav_pos: np.ndarray, ground_pos: np.ndarray) -> float:
        """
        g_i(t) = rho_0 / (d_i(t)^eta)
        """
        dist = np.linalg.norm(uav_pos - ground_pos)
        dist = max(1.0, float(dist))  # Avoid singularity
        gain = self.sim_cfg.path_loss_ref / (dist ** self.sim_cfg.path_loss_exp)
        return gain

    def compute_directional_jammer_gain(self, uav_pos: np.ndarray) -> float:
        """
        Directional jammer with conical main beam pointing upward / toward center.
        """
        rel_vec = uav_pos - self.sim_cfg.jammer_pos
        dist = np.linalg.norm(rel_vec)
        if dist < 1e-3:
            return 1.0
            
        # Jammer points towards +z with 45 degree beamwidth
        cos_theta = rel_vec[2] / dist
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        if angle_deg <= self.sim_cfg.jam_beamwidth_deg:
            # Inside main lobe
            antenna_gain = 1.0
        else:
            # Sidelobe attenuation (-20 dB)
            antenna_gain = 0.01
            
        g_jam = self.compute_channel_gain(uav_pos, self.sim_cfg.jammer_pos)
        return g_jam * antenna_gain

    def compute_sinr_and_rate(self, uav_pos: np.ndarray, dt: float) -> tuple[float, float, float]:
        """
        Calculates instantaneous SINR, data rate R(t), and collected data volume in Mbits.
        R(t) = B * log2(1 + (P_iot * g_iot) / (sigma^2 + P_jam * g_jam * G_beam))
        """
        g_iot = self.compute_channel_gain(uav_pos, self.sim_cfg.iot_pos)
        s_power = self.sim_cfg.p_iot_tx * g_iot
        
        g_jam_effective = self.compute_directional_jammer_gain(uav_pos)
        i_power = self.sim_cfg.p_jam_tx * g_jam_effective
        
        noise = self.sim_cfg.noise_power
        sinr = s_power / (noise + i_power)
        
        # Shannon Capacity (bits/sec)
        rate_bps = self.sim_cfg.bandwidth * np.log2(1.0 + sinr)
        
        # Mbits collected over duration dt
        collected_mbits = (rate_bps * dt) / 1.0e6
        
        return sinr, rate_bps, collected_mbits
