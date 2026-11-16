"""
Module: envs.uav_dynamics
Implementation of UAVDynamics modeling a rotary-wing quadrotor with
kinematic updates, physical limits, and aerodynamic propulsion power consumption.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class UAVPhysicalParams:
    # Kinematic limits
    v_max: float = 20.0             # m/s
    a_max: float = 5.0              # m/s^2
    z_min: float = 20.0             # m
    z_max: float = 200.0            # m
    
    # Aerodynamic power constants
    P_0: float = 79.86              # Blade profile power (W)
    P_i: float = 120.3              # Induced power in hover (W)
    U_tip: float = 120.0            # Tip speed of the rotor blade (m/s)
    v_0: float = 4.03               # Mean rotor induced velocity in hover (m/s)
    d_0: float = 0.6                # Fuselage drag ratio
    rho: float = 1.225              # Air density (kg/m^3)
    s: float = 0.05                 # Rotor solidity
    A: float = 0.503                # Rotor disc area (m^2)
    P_avionics: float = 5.0         # Avionics & onboard sensor circuitry power (W)


class UAVDynamics:
    """
    Simulates rotary-wing quadrotor kinematics and power consumption.
    """
    def __init__(self, params: UAVPhysicalParams = None, initial_pos: np.ndarray = None):
        self.params = params if params is not None else UAVPhysicalParams()
        self.pos = np.array([0.0, 0.0, self.params.z_min] if initial_pos is None else initial_pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.acc = np.zeros(3, dtype=np.float64)
        self.total_energy_joules = 0.0

    def reset(self, initial_pos: np.ndarray = None, initial_vel: np.ndarray = None):
        """Resets the state of the UAV."""
        if initial_pos is not None:
            self.pos = np.array(initial_pos, dtype=np.float64)
        else:
            self.pos = np.array([0.0, 0.0, self.params.z_min], dtype=np.float64)
            
        if initial_vel is not None:
            self.vel = np.array(initial_vel, dtype=np.float64)
        else:
            self.vel = np.zeros(3, dtype=np.float64)
            
        self.acc = np.zeros(3, dtype=np.float64)
        self.total_energy_joules = 0.0
        return self.pos, self.vel

    def compute_power(self, speed: float) -> float:
        """
        Calculates instantaneous aerodynamic propulsion power P(v):
        P(v) = P_0 * (1 + 3 * ||v||^2 / U_tip^2)
             + P_i * ((1 + ||v||^4 / (4 * v_0^4))^0.5 - ||v||^2 / (2 * v_0^2))^0.5
             + 0.5 * d_0 * rho * s * A * ||v||^3 + P_avionics
        """
        p = self.params
        v = max(0.0, float(speed))

        # 1. Blade profile power
        p_profile = p.P_0 * (1.0 + (3.0 * (v ** 2)) / (p.U_tip ** 2))

        # 2. Induced power
        inner_term = np.sqrt(1.0 + (v ** 4) / (4.0 * (p.v_0 ** 4))) - (v ** 2) / (2.0 * (p.v_0 ** 2))
        inner_term = max(0.0, inner_term)
        p_induced = p.P_i * np.sqrt(inner_term)

        # 3. Parasite drag power
        p_parasite = 0.5 * p.d_0 * p.rho * p.s * p.A * (v ** 3)

        return p_profile + p_induced + p_parasite + p.P_avionics

    def compute_energy(self, velocity: np.ndarray, dt: float) -> float:
        """Computes energy expended in Joules over duration dt."""
        speed = float(np.linalg.norm(velocity))
        power = self.compute_power(speed)
        return power * dt

    def step(self, commanded_acc: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Performs discrete-time kinematic update:
            p(t+1) = p(t) + v(t)*dt + 0.5*a(t)*dt^2
            v(t+1) = v(t) + a(t)*dt
        Subject to hard physical saturation:
            ||a(t)|| <= A_max
            ||v(t)|| <= V_max
            z_min <= z <= z_max
        """
        a_cmd = np.array(commanded_acc, dtype=np.float64)
        norm_a = np.linalg.norm(a_cmd)
        if norm_a > self.params.a_max:
            a_cmd = (a_cmd / norm_a) * self.params.a_max
        self.acc = a_cmd

        # State integration
        new_pos = self.pos + self.vel * dt + 0.5 * self.acc * (dt ** 2)
        new_vel = self.vel + self.acc * dt

        # Velocity magnitude clipping
        speed = np.linalg.norm(new_vel)
        if speed > self.params.v_max:
            new_vel = (new_vel / speed) * self.params.v_max
            speed = self.params.v_max

        # Altitude bounds clipping
        if new_pos[2] < self.params.z_min:
            new_pos[2] = self.params.z_min
            new_vel[2] = max(0.0, new_vel[2])
        elif new_pos[2] > self.params.z_max:
            new_pos[2] = self.params.z_max
            new_vel[2] = min(0.0, new_vel[2])

        self.pos = new_pos
        self.vel = new_vel

        # Energy consumption
        step_energy = self.compute_energy(self.vel, dt)
        self.total_energy_joules += step_energy

        return self.pos, self.vel, step_energy
