"""
Module: envs.multi_uav_env
Dual-UAV Cooperative Swarm Environment for IoT data harvesting
with inter-UAV collision checks and co-channel mutual interference.
"""
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.uav_dynamics import UAVDynamics, UAVPhysicalParams
from envs.channel_model import WirelessChannel, ChannelParameters

@dataclass
class MultiAirspaceConfig:
    x_bounds: tuple = (0.0, 1000.0)
    y_bounds: tuple = (0.0, 1000.0)
    z_bounds: tuple = (20.0, 200.0)
    
    # UAV 1: Start, Dest, IoT Node 1
    uav1_start: np.ndarray = field(default_factory=lambda: np.array([0.0, 300.0, 50.0]))
    uav1_dest: np.ndarray = field(default_factory=lambda: np.array([1000.0, 300.0, 50.0]))
    iot1_pos: np.ndarray = field(default_factory=lambda: np.array([400.0, 300.0, 0.0]))
    
    # UAV 2: Start, Dest, IoT Node 2
    uav2_start: np.ndarray = field(default_factory=lambda: np.array([0.0, 700.0, 50.0]))
    uav2_dest: np.ndarray = field(default_factory=lambda: np.array([1000.0, 700.0, 50.0]))
    iot2_pos: np.ndarray = field(default_factory=lambda: np.array([600.0, 700.0, 0.0]))
    
    # Shared Jammer
    jammer_pos: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0, 0.0]))
    
    # Safety and Swarm Parameters
    d_safe: float = 30.0          # Minimum safe separation (m)
    w_col: float = 100.0          # Heavy collision penalty weight
    dest_tolerance: float = 30.0  # m
    dt: float = 0.5               # time step per macro action (s)
    max_steps: int = 100
    r_norm: float = 5.0e6
    p_norm: float = 300.0


class MultiUAVTrajectoryEnv(gym.Env):
    """
    Decentralized Dual-UAV Swarm Environment.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, cfg: MultiAirspaceConfig = None):
        super(MultiUAVTrajectoryEnv, self).__init__()
        self.cfg = cfg if cfg is not None else MultiAirspaceConfig()
        
        self.uav_params = UAVPhysicalParams(
            v_max=20.0, a_max=5.0,
            z_min=self.cfg.z_bounds[0],
            z_max=self.cfg.z_bounds[1]
        )
        
        # Dual-agent kinematics
        self.uav1 = UAVDynamics(params=self.uav_params, initial_pos=self.cfg.uav1_start)
        self.uav2 = UAVDynamics(params=self.uav_params, initial_pos=self.cfg.uav2_start)
        
        self.channel = WirelessChannel()
        self.d_init = float(np.linalg.norm(self.cfg.uav1_dest - self.cfg.uav1_start))
        
        # Action space: 3D continuous heading displacement for each UAV
        self.action_space = spaces.Dict({
            'uav_1': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
            'uav_2': spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        })
        
        # Observation space per agent (15 dims):
        # [vector_to_dest (3), vector_to_own_iot (3), vector_to_jammer (3), vel (3), vector_to_neighbor (3)]
        self.observation_space = spaces.Dict({
            'uav_1': spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32),
            'uav_2': spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        })
        
        self.reset()

    def _get_agent_obs(self, uav_own: UAVDynamics, dest: np.ndarray, iot: np.ndarray, uav_other: UAVDynamics) -> np.ndarray:
        rel_dest = (uav_own.pos - dest) / self.d_init
        rel_iot = (uav_own.pos - iot) / self.d_init
        rel_jam = (uav_own.pos - self.cfg.jammer_pos) / self.d_init
        norm_vel = uav_own.vel / self.uav_params.v_max
        rel_neighbor = (uav_own.pos - uav_other.pos) / self.d_init
        
        return np.concatenate([rel_dest, rel_iot, rel_jam, norm_vel, rel_neighbor]).astype(np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            'uav_1': self._get_agent_obs(self.uav1, self.cfg.uav1_dest, self.cfg.iot1_pos, self.uav2),
            'uav_2': self._get_agent_obs(self.uav2, self.cfg.uav2_dest, self.cfg.iot2_pos, self.uav1)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav1.reset(initial_pos=self.cfg.uav1_start)
        self.uav2.reset(initial_pos=self.cfg.uav2_start)
        
        self.current_step = 0
        self.total_bits_uav1 = 0.0
        self.total_bits_uav2 = 0.0
        self.traj_uav1 = [np.copy(self.uav1.pos)]
        self.traj_uav2 = [np.copy(self.uav2.pos)]
        self.min_dist_separation = float('inf')
        
        return self._get_obs(), {}

    def step(self, actions: dict[str, np.ndarray]):
        """
        Step with high-level actions.
        """
        dt = self.cfg.dt
        a1 = np.array(actions['uav_1'], dtype=np.float64)
        a2 = np.array(actions['uav_2'], dtype=np.float64)
        
        target_v1 = (a1 / max(1.0, np.linalg.norm(a1))) * self.uav_params.v_max
        target_v2 = (a2 / max(1.0, np.linalg.norm(a2))) * self.uav_params.v_max
        
        acc1 = (target_v1 - self.uav1.vel) / dt
        acc2 = (target_v2 - self.uav2.vel) / dt
        
        return self.step_with_acc(acc1, acc2)

    def step_with_acc(self, acc1: np.ndarray, acc2: np.ndarray):
        """
        Physical step with explicit acceleration commands for both drones.
        """
        self.current_step += 1
        dt = self.cfg.dt
        
        # Kinematic state integration
        pos1, vel1, eng1 = self.uav1.step(acc1, dt)
        pos2, vel2, eng2 = self.uav2.step(acc2, dt)
        
        self.traj_uav1.append(np.copy(pos1))
        self.traj_uav2.append(np.copy(pos2))
        
        # 1. Inter-UAV Separation Distance
        d_12 = float(np.linalg.norm(pos1 - pos2))
        if d_12 < self.min_dist_separation:
            self.min_dist_separation = d_12
            
        # 2. Wireless Communications & Co-Channel Mutual Interference
        # UAV 1 communicates with IoT 1, UAV 2 with IoT 2
        # Co-channel mutual interference added when both transmit
        dist_inter_uav = max(1.0, d_12)
        mutual_interference_gain = 1.0 / (dist_inter_uav ** 2.0)
        p_interf_uav = self.channel.p_iot * mutual_interference_gain * 0.1  # Leakage factor
        
        # Channel rate 1
        g_iot1 = self.channel.compute_channel_gain(pos1, self.cfg.iot1_pos)
        s_power1 = self.channel.p_iot * g_iot1
        g_jam1 = self.channel.compute_jammer_gain(pos1, self.cfg.jammer_pos)
        i_power1 = self.channel.p_jam * g_jam1 + p_interf_uav
        sinr1 = s_power1 / (self.channel.noise_power + i_power1)
        rate1_bps = self.channel.params.bandwidth * np.log2(1.0 + sinr1)
        self.total_bits_uav1 += rate1_bps * dt
        
        # Channel rate 2
        g_iot2 = self.channel.compute_channel_gain(pos2, self.cfg.iot2_pos)
        s_power2 = self.channel.p_iot * g_iot2
        g_jam2 = self.channel.compute_jammer_gain(pos2, self.cfg.jammer_pos)
        i_power2 = self.channel.p_jam * g_jam2 + p_interf_uav
        sinr2 = s_power2 / (self.channel.noise_power + i_power2)
        rate2_bps = self.channel.params.bandwidth * np.log2(1.0 + sinr2)
        self.total_bits_uav2 += rate2_bps * dt
        
        # 3. Collision Avoidance Penalty
        r_collision = 0.0
        collision_event = False
        if d_12 < self.cfg.d_safe:
            r_collision = -self.cfg.w_col * ((1.0 - d_12 / self.cfg.d_safe) ** 2)
            if d_12 < 10.0:  # Physical structural collision
                collision_event = True
                
        # 4. Individual and Team Rewards
        dist_dest1 = float(np.linalg.norm(pos1 - self.cfg.uav1_dest))
        dist_dest2 = float(np.linalg.norm(pos2 - self.cfg.uav2_dest))
        
        r1 = (1.5 * (rate1_bps / self.cfg.r_norm)
              - 0.8 * (eng1 / dt / self.cfg.p_norm)
              - 0.5 * (dist_dest1 / self.d_init)
              + r_collision)
              
        r2 = (1.5 * (rate2_bps / self.cfg.r_norm)
              - 0.8 * (eng2 / dt / self.cfg.p_norm)
              - 0.5 * (dist_dest2 / self.d_init)
              + r_collision)
              
        rewards = {'uav_1': r1, 'uav_2': r2}
        
        # Terminations
        terminated = False
        truncated = False
        
        succ1 = dist_dest1 <= self.cfg.dest_tolerance
        succ2 = dist_dest2 <= self.cfg.dest_tolerance
        if succ1 and succ2:
            rewards['uav_1'] += 100.0
            rewards['uav_2'] += 100.0
            terminated = True
        elif collision_event:
            terminated = True
            
        if self.current_step >= self.cfg.max_steps:
            truncated = True
            
        info = {
            'pos_uav1': pos1,
            'pos_uav2': pos2,
            'd_12': d_12,
            'min_separation': self.min_dist_separation,
            'total_bits_uav1_mbit': self.total_bits_uav1 / 1e6,
            'total_bits_uav2_mbit': self.total_bits_uav2 / 1e6,
            'total_data_joint_mbit': (self.total_bits_uav1 + self.total_bits_uav2) / 1e6,
            'total_energy_joint_kj': (self.uav1.total_energy_joules + self.uav2.total_energy_joules) / 1000.0,
            'collision_event': collision_event
        }
        
        return self._get_obs(), rewards, terminated, truncated, info
