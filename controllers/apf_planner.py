"""
Module: controllers.apf_planner
Artificial Potential Field (APF) Collision Avoidance Safety Filter.
Computes smooth repulsive potential forces and safety-modified reference
trajectories for decentralized multi-UAV swarms.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class APFParams:
    d_safe: float = 30.0        # Safe separation threshold (m)
    k_rep: float = 50.0         # Repulsive field gain constant
    a_rep_max: float = 4.0      # Maximum repulsive acceleration (m/s^2)


class APFCollisionFilter:
    """
    Artificial Potential Field (APF) safety filter for multi-UAV deconfliction.
    """
    def __init__(self, params: APFParams = None):
        self.params = params if params is not None else APFParams()

    def compute_repulsive_acc(self, p_own: np.ndarray, p_other: np.ndarray) -> np.ndarray:
        """
        Computes repulsive acceleration vector:
        If d_12 <= d_safe:
            a_rep = k_rep * (1/d_12 - 1/d_safe) * (1 / d_12^2) * ((p_own - p_other) / d_12)
        Else:
            a_rep = 0
        """
        rel_vec = p_own - p_other
        d_12 = float(np.linalg.norm(rel_vec))
        
        if d_12 < 1e-4:
            # Random perturb if exactly collocated
            direction = np.array([0.0, 1.0, 0.0])
            d_12 = 1e-3
        else:
            direction = rel_vec / d_12
            
        if d_12 <= self.params.d_safe:
            repulsive_mag = self.params.k_rep * (1.0 / d_12 - 1.0 / self.params.d_safe) * (1.0 / (d_12 ** 2))
            a_rep = repulsive_mag * direction
            norm_a = np.linalg.norm(a_rep)
            if norm_a > self.params.a_rep_max:
                a_rep = (a_rep / norm_a) * self.params.a_rep_max
            return a_rep
        else:
            return np.zeros(3, dtype=np.float64)

    def filter_reference_horizon(self, p_ref_horizon: np.ndarray, p_other: np.ndarray, dt: float = 0.5) -> np.ndarray:
        """
        Modifies tactical waypoint reference horizon using APF repulsive field:
        p_ref_horizon_safe = p_ref_horizon + 0.5 * a_rep * dt^2
        """
        p_own_current = p_ref_horizon[:, 0]
        a_rep = self.compute_repulsive_acc(p_own_current, p_other)
        
        p_ref_safe = np.copy(p_ref_horizon)
        if np.linalg.norm(a_rep) > 1e-5:
            N = p_ref_horizon.shape[1]
            for k in range(1, N):
                # Repulsive offset applied progressively over prediction horizon
                p_ref_safe[:, k] += 0.5 * a_rep * ((k * dt) ** 2)
                
        return p_ref_safe
