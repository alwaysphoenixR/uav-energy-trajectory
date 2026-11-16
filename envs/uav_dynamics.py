"""
UAV Kinematics & Aerodynamic Rotary-Wing Propulsion Energy Consumption.
"""
import numpy as np
from configs.settings import UAVRotaryEnergyConfig, SimulationConfig

class UAVKinematics:
    def __init__(self, sim_cfg: SimulationConfig, energy_cfg: UAVRotaryEnergyConfig):
        self.sim_cfg = sim_cfg
        self.energy_cfg = energy_cfg
        self.reset()
        
    def reset(self, start_pos: np.ndarray = None):
        if start_pos is None:
            self.pos = np.copy(self.sim_cfg.start_pos).astype(np.float64)
        else:
            self.pos = np.copy(start_pos).astype(np.float64)
            
        self.vel = np.zeros(3, dtype=np.float64)
        self.acc = np.zeros(3, dtype=np.float64)
        self.remaining_battery = self.energy_cfg.battery_capacity_joules
        self.total_energy_expended = 0.0
        return self.pos, self.vel, self.remaining_battery

    def compute_propulsion_power(self, speed: float) -> float:
        """
        Rotary-wing propulsion power consumption model:
        P(v) = P_0 * (1 + 3*v^2 / U_tip^2) + P_i * (sqrt(1 + v^4 / (4*v_0^4)) - v^2 / (2*v_0^2))^0.5 + 0.5 * d_0 * rho * s * A * v^3
        """
        cfg = self.energy_cfg
        v = max(0.0, float(speed))
        
        # Blade profile power
        p_profile = cfg.P_0 * (1.0 + (3.0 * (v ** 2)) / (cfg.U_tip ** 2))
        
        # Induced power
        v_induced_term = np.sqrt(1.0 + (v ** 4) / (4.0 * (cfg.v_0 ** 4))) - (v ** 2) / (2.0 * (cfg.v_0 ** 2))
        v_induced_term = max(0.0, v_induced_term)
        p_induced = cfg.P_i * np.sqrt(v_induced_term)
        
        # Parasite drag power
        p_parasite = 0.5 * cfg.d_0 * cfg.rho * cfg.s * cfg.A * (v ** 3)
        
        # Total power = propulsion + avionics/comm
        p_total = p_profile + p_induced + p_parasite + cfg.P_comm
        return p_total

    def step(self, acc_cmd: np.ndarray, dt: float):
        """
        Kinematic update:
        p(t+dt) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
        v(t+dt) = v(t) + a(t)*dt
        Clip to v_max and a_max
        """
        # Clip acceleration
        norm_a = np.linalg.norm(acc_cmd)
        if norm_a > self.sim_cfg.a_max:
            acc_cmd = (acc_cmd / norm_a) * self.sim_cfg.a_max
        self.acc = acc_cmd
        
        # Integrate position and velocity
        new_pos = self.pos + self.vel * dt + 0.5 * self.acc * (dt ** 2)
        new_vel = self.vel + self.acc * dt
        
        # Clip horizontal velocity magnitude
        speed = np.linalg.norm(new_vel)
        if speed > self.sim_cfg.v_max:
            new_vel = (new_vel / speed) * self.sim_cfg.v_max
            speed = self.sim_cfg.v_max
            
        self.pos = new_pos
        self.vel = new_vel
        
        # Energy consumption
        power = self.compute_propulsion_power(speed)
        delta_energy = power * dt
        self.remaining_battery = max(0.0, self.remaining_battery - delta_energy)
        self.total_energy_expended += delta_energy
        
        return self.pos, self.vel, delta_energy, self.remaining_battery
