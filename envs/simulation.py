"""
Custom Gymnasium Environment for UAV 3D Trajectory Planning with Directional Jamming,
Obstacle Avoidance, and Rotary-Wing Propulsion Energy Constraints.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from configs.settings import SimulationConfig, UAVRotaryEnergyConfig, PPORewardWeights, NMPCConfig
from envs.uav_dynamics import UAVKinematics
from envs.channel_model import AirToGroundChannel
from controllers.nmpc_controller import NMPCTracker

class UAVAirspaceEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, use_nmpc: bool = True, energy_aware: bool = True):
        super(UAVAirspaceEnv, self).__init__()
        
        self.sim_cfg = SimulationConfig()
        self.energy_cfg = UAVRotaryEnergyConfig()
        self.reward_weights = PPORewardWeights()
        self.nmpc_cfg = NMPCConfig()
        
        self.use_nmpc = use_nmpc
        self.energy_aware = energy_aware
        
        self.kinematics = UAVKinematics(self.sim_cfg, self.energy_cfg)
        self.channel = AirToGroundChannel(self.sim_cfg)
        
        if self.use_nmpc:
            self.nmpc = NMPCTracker(self.sim_cfg, self.nmpc_cfg)
            
        # Action space: normalized 3D tactical displacement vector u_ppo in [-1, 1]^3
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # State observation:
        # [rel_iot(3), rel_dest(3), rel_jammer(3), vel(3), normalized_battery(1), total_collected_ratio(1)] -> Total 14 dims
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32)
        
        self.d_init = np.linalg.norm(self.sim_cfg.dest_pos - self.sim_cfg.start_pos)
        self.reset()
        
    def _get_obs(self):
        rel_iot = (self.kinematics.pos - self.sim_cfg.iot_pos) / 1000.0
        rel_dest = (self.kinematics.pos - self.sim_cfg.dest_pos) / 1000.0
        rel_jammer = (self.kinematics.pos - self.sim_cfg.jammer_pos) / 1000.0
        norm_vel = self.kinematics.vel / self.sim_cfg.v_max
        norm_battery = np.array([self.kinematics.remaining_battery / self.energy_cfg.battery_capacity_joules])
        data_ratio = np.array([self.total_collected_data / self.sim_cfg.p_req_data])
        
        obs = np.concatenate([rel_iot, rel_dest, rel_jammer, norm_vel, norm_battery, data_ratio]).astype(np.float32)
        return obs
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.kinematics.reset()
        self.current_step = 0
        self.total_collected_data = 0.0
        self.trajectory_history = [np.copy(self.kinematics.pos)]
        self.sinr_history = []
        self.power_history = []
        
        return self._get_obs(), {}

    def _check_collision(self, pos: np.ndarray) -> bool:
        # Check boundary violation
        if (pos[0] < self.sim_cfg.x_bounds[0] or pos[0] > self.sim_cfg.x_bounds[1] or
            pos[1] < self.sim_cfg.y_bounds[0] or pos[1] > self.sim_cfg.y_bounds[1] or
            pos[2] < self.sim_cfg.z_bounds[0] or pos[2] > self.sim_cfg.z_bounds[1]):
            return True
            
        # Check static cylinder obstacles: (ox, oy, r, h)
        for ox, oy, r, h in self.sim_cfg.obstacles:
            dist_xy = np.hypot(pos[0] - ox, pos[1] - oy)
            if dist_xy <= r and pos[2] <= h:
                return True
        return False

    def step(self, action: np.ndarray):
        self.current_step += 1
        action = np.clip(action, -1.0, 1.0)
        norm_action = np.linalg.norm(action)
        u_ppo = action if norm_action < 1e-4 else (action / norm_action)
        
        # Micro-stepping execution (macro step dt_macro divided into micro steps dt_micro)
        n_micro = int(round(self.sim_cfg.dt_macro / self.sim_cfg.dt_micro))
        dt = self.sim_cfg.dt_micro
        
        step_energy = 0.0
        step_data = 0.0
        step_violation = False
        
        if self.use_nmpc:
            # Generate reference waypoints for NMPC prediction horizon N
            N = self.nmpc_cfg.N_horizon
            p_ref_horizon = np.zeros((3, N + 1))
            p_ref_horizon[:, 0] = self.kinematics.pos
            for k in range(1, N + 1):
                p_ref_horizon[:, k] = self.kinematics.pos + k * (self.sim_cfg.l_step / N) * u_ppo
                
            for _ in range(n_micro):
                a_cmd = self.nmpc.solve(self.kinematics.pos, self.kinematics.vel, p_ref_horizon)
                pos, vel, de, bat = self.kinematics.step(a_cmd, dt)
                step_energy += de
                
                # Wireless channel update
                sinr, rate, collected = self.channel.compute_sinr_and_rate(pos, dt)
                step_data += collected
                self.sinr_history.append(sinr)
                
                if self._check_collision(pos):
                    step_violation = True
        else:
            # Standalone DRL without NMPC: direct velocity assignment
            desired_vel = u_ppo * self.sim_cfg.v_max
            for _ in range(n_micro):
                acc_direct = (desired_vel - self.kinematics.vel) / dt
                pos, vel, de, bat = self.kinematics.step(acc_direct, dt)
                step_energy += de
                sinr, rate, collected = self.channel.compute_sinr_and_rate(pos, dt)
                step_data += collected
                self.sinr_history.append(sinr)
                if self._check_collision(pos):
                    step_violation = True

        self.total_collected_data += step_data
        self.trajectory_history.append(np.copy(self.kinematics.pos))
        
        # Reward Shaping:
        # r = w_d * R_norm - w_u * J_unpred - w_n * (dist / D_init) - w_c * I_violation - w_e * Energy
        dist_dest = np.linalg.norm(self.kinematics.pos - self.sim_cfg.dest_pos)
        norm_data_reward = step_data / 1.0  # normalized
        dist_penalty = dist_dest / self.d_init
        
        # Jammer unpredictability incentive: distance to jammer
        dist_jammer = np.linalg.norm(self.kinematics.pos - self.sim_cfg.jammer_pos)
        jammer_penalty = 1.0 / (1.0 + (dist_jammer / 100.0) ** 2)
        
        reward = (
            self.reward_weights.omega_d * norm_data_reward
            - self.reward_weights.omega_u * jammer_penalty
            - self.reward_weights.omega_n * dist_penalty
        )
        
        if step_violation:
            reward -= self.reward_weights.omega_c
            
        if self.energy_aware:
            # Novelty: Energy depletion penalty
            reward -= self.reward_weights.omega_e * (step_energy / 100.0)
            
        # Termination conditions
        terminated = False
        truncated = False
        
        reached_dest = dist_dest <= self.sim_cfg.dest_tolerance
        data_satisfied = self.total_collected_data >= self.sim_cfg.p_req_data
        out_of_battery = self.kinematics.remaining_battery <= 0.0
        
        if reached_dest:
            if data_satisfied:
                reward += 100.0  # Mission success bonus
            else:
                reward += 20.0   # Reached destination without enough data
            terminated = True
            
        if step_violation or out_of_battery:
            terminated = True
            
        if self.current_step >= self.sim_cfg.max_steps:
            truncated = True
            
        info = {
            'total_data': self.total_collected_data,
            'total_energy': self.kinematics.total_energy_expended,
            'dist_to_dest': dist_dest,
            'violation': step_violation,
            'battery_pct': (self.kinematics.remaining_battery / self.energy_cfg.battery_capacity_joules) * 100.0
        }
        
        return self._get_obs(), reward, terminated, truncated, info
