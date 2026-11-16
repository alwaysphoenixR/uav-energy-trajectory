"""
Baseline Models for Benchmark Comparison:
1. Standalone Pure NMPC (Greedy direct tracking towards IoT and Destination)
2. Standalone PPO (Direct kinematic integration without low-level NMPC constraints)
3. Standard Hierarchical DRL-NMPC (Without energy depletion awareness)
4. Proposed Energy-Aware Hierarchical DRL-NMPC (With aerodynamic power model)
"""
import numpy as np
from configs.settings import SimulationConfig, NMPCConfig
from controllers.nmpc_controller import NMPCTracker

class GreedyNMPCBaseline:
    def __init__(self, sim_cfg: SimulationConfig):
        self.sim_cfg = sim_cfg
        self.tracker = NMPCTracker(sim_cfg, NMPCConfig(N_horizon=5))
        
    def get_action(self, current_pos: np.ndarray, current_vel: np.ndarray, total_data: float):
        # If required data collected, fly to destination, else loiter around IoT
        if total_data >= self.sim_cfg.p_req_data:
            target = self.sim_cfg.dest_pos
        else:
            target = self.sim_cfg.iot_pos + np.array([0.0, 0.0, 100.0])
            
        direction = target - current_pos
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-3:
            u_dir = direction / norm_dir
        else:
            u_dir = np.zeros(3)
            
        return u_dir
