"""
Module: envs.uav_env
Gymnasium Environment wrapper for 3D UAV Trajectory Planning
integrating UAVDynamics and WirelessChannel.
"""
from dataclasses import dataclass, field
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from envs.uav_dynamics import UAVDynamics, UAVPhysicalParams
from envs.channel_model import WirelessChannel, ChannelParameters

@dataclass
class EnvRewardWeights:
    w_data: float = 1.5
    w_energy: float = 0.8
    w_dist: float = 0.5
    w_jam: float = 1.0
    w_crash: float = 50.0

@dataclass
class AirspaceConfig:
    x_bounds: tuple = (0.0, 1000.0)
    y_bounds: tuple = (0.0, 1000.0)
    z_bounds: tuple = (20.0, 200.0)
    
    start_pos: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0, 50.0]))
    dest_pos: np.ndarray = field(default_factory=lambda: np.array([900.0, 900.0, 100.0]))
    iot_pos: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0, 0.0]))
    jammer_pos: np.ndarray = field(default_factory=lambda: np.array([450.0, 550.0, 0.0]))
    
    # Cylindrical static obstacles: (x, y, radius, height)
    obstacles: list = field(default_factory=lambda: [
        (350.0, 350.0, 50.0, 180.0),
        (650.0, 650.0, 60.0, 190.0)
    ])
    
    dest_tolerance: float = 30.0  # meters
    dt: float = 0.5               # time step per macro action (seconds)
    max_steps: int = 100          # episode truncation limit
    r_norm: float = 5.0e6         # 5 Mbps normalization for data rate
    p_norm: float = 300.0         # 300 W normalization for power


class UAVTrajectoryEnv(gym.Env):
    """
    3D UAV Airspace Navigation Environment with Contested RF Spectrum.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, air_cfg: AirspaceConfig = None, reward_weights: EnvRewardWeights = None):
        super(UAVTrajectoryEnv, self).__init__()
        
        self.air_cfg = air_cfg if air_cfg is not None else AirspaceConfig()
        self.reward_weights = reward_weights if reward_weights is not None else EnvRewardWeights()
        
        self.uav_params = UAVPhysicalParams(
            v_max=20.0, a_max=5.0,
            z_min=self.air_cfg.z_bounds[0],
            z_max=self.air_cfg.z_bounds[1]
        )
        self.uav = UAVDynamics(params=self.uav_params, initial_pos=self.air_cfg.start_pos)
        self.channel = WirelessChannel()
        
        self.d_init = float(np.linalg.norm(self.air_cfg.dest_pos - self.air_cfg.start_pos))
        
        # Action space: 3D continuous heading displacement / tactical vector in [-1.0, 1.0]^3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Observation space (dim 12):
        # [(p - p_dest)/D_init (3), (p - p_iot)/D_init (3), (p - p_jam)/D_init (3), v/V_max (3)]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        
        self.reset()

    def _get_obs(self) -> np.ndarray:
        rel_dest = (self.uav.pos - self.air_cfg.dest_pos) / self.d_init
        rel_iot = (self.uav.pos - self.air_cfg.iot_pos) / self.d_init
        rel_jam = (self.uav.pos - self.air_cfg.jammer_pos) / self.d_init
        norm_vel = self.uav.vel / self.uav_params.v_max
        
        obs = np.concatenate([rel_dest, rel_iot, rel_jam, norm_vel]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.uav.reset(initial_pos=self.air_cfg.start_pos)
        self.current_step = 0
        self.total_bits_collected = 0.0
        self.trajectory_history = [np.copy(self.uav.pos)]
        return self._get_obs(), {}

    def _check_collision_or_oob(self, pos: np.ndarray) -> bool:
        # Check out of bounds
        if (pos[0] < self.air_cfg.x_bounds[0] or pos[0] > self.air_cfg.x_bounds[1] or
            pos[1] < self.air_cfg.y_bounds[0] or pos[1] > self.air_cfg.y_bounds[1] or
            pos[2] < self.air_cfg.z_bounds[0] or pos[2] > self.air_cfg.z_bounds[1]):
            return True
            
        # Check static obstacles
        for ox, oy, r, h in self.air_cfg.obstacles:
            if np.hypot(pos[0] - ox, pos[1] - oy) <= r and pos[2] <= h:
                return True
        return False

    def step(self, action: np.ndarray):
        """
        Executes one step in the environment.
        Action can be a 3D tactical heading displacement vector u_ppo or direct acceleration a_cmd.
        """
        self.current_step += 1
        dt = self.air_cfg.dt
        
        action = np.array(action, dtype=np.float64)
        # If action represents a unit heading/tactical displacement vector, scale to acceleration command:
        norm_act = np.linalg.norm(action)
        if norm_act > 1.0:
            action = action / norm_act
            
        # Target velocity based on tactical action:
        target_vel = action * self.uav_params.v_max
        commanded_acc = (target_vel - self.uav.vel) / dt
        
        return self.step_with_acc(commanded_acc)

    def step_with_acc(self, commanded_acc: np.ndarray):
        """
        Step environment directly using physical acceleration command (e.g. from NMPC controller).
        """
        dt = self.air_cfg.dt
        pos, vel, step_energy = self.uav.step(commanded_acc, dt)
        self.trajectory_history.append(np.copy(pos))
        
        # Wireless channel observation
        rate_bps, sinr = self.channel.compute_data_rate(pos, self.air_cfg.iot_pos, [self.air_cfg.jammer_pos])
        bits_collected = rate_bps * dt
        self.total_bits_collected += bits_collected
        
        # Jammer exposure indicator (in main beam of directional jammer)
        rel_jam = pos - self.air_cfg.jammer_pos
        dist_jam = max(1.0, float(np.linalg.norm(rel_jam)))
        cos_theta = rel_jam[2] / dist_jam
        angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        in_jam_beam = 1.0 if angle_deg <= (self.channel.params.jam_beamwidth_deg / 2.0) else 0.0
        
        # Check collision or out of bounds
        is_crash = self._check_collision_or_oob(pos)
        
        # Compute Reward:
        # r_t = w_data * (R_t / R_norm) - w_energy * (P_t / P_norm) - w_dist * (dist_dest / D_init)
        #       - w_jam * in_jam_beam - w_crash * is_crash
        p_instantaneous = step_energy / dt
        dist_dest = float(np.linalg.norm(pos - self.air_cfg.dest_pos))
        
        reward = (
            self.reward_weights.w_data * (rate_bps / self.air_cfg.r_norm)
            - self.reward_weights.w_energy * (p_instantaneous / self.air_cfg.p_norm)
            - self.reward_weights.w_dist * (dist_dest / self.d_init)
            - self.reward_weights.w_jam * in_jam_beam
        )
        
        if is_crash:
            reward -= self.reward_weights.w_crash
            
        # Termination check
        terminated = False
        truncated = False
        
        if dist_dest <= self.air_cfg.dest_tolerance:
            reward += 100.0  # Mission success bonus
            terminated = True
        elif is_crash:
            terminated = True
            
        if self.current_step >= self.air_cfg.max_steps:
            truncated = True
            
        info = {
            'pos': pos,
            'vel': vel,
            'rate_bps': rate_bps,
            'sinr': sinr,
            'step_energy_joules': step_energy,
            'total_energy_kj': self.uav.total_energy_joules / 1000.0,
            'total_data_mbits': self.total_bits_collected / 1.0e6,
            'dist_to_dest': dist_dest,
            'is_crash': is_crash,
            'in_jam_beam': in_jam_beam
        }
        
        return self._get_obs(), reward, terminated, truncated, info
