"""
Global configuration and hyperparameters for Hierarchical DRL-NMPC UAV Trajectory Planning.
"""
from dataclasses import dataclass, field
import numpy as np

@dataclass
class SimulationConfig:
    # 3D Bounding volume [X, Y, Z] in meters
    x_bounds: tuple = (0.0, 1000.0)
    y_bounds: tuple = (0.0, 1000.0)
    z_bounds: tuple = (50.0, 300.0)
    
    # Mission start & destination
    start_pos: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0, 100.0]))
    dest_pos: np.ndarray = field(default_factory=lambda: np.array([900.0, 900.0, 100.0]))
    dest_tolerance: float = 30.0  # meters
    
    # IoT Ground Device
    iot_pos: np.ndarray = field(default_factory=lambda: np.array([500.0, 500.0, 0.0]))
    p_iot_tx: float = 0.1          # 100 mW (20 dBm)
    p_req_data: float = 15.0       # Required data collection in Mbits
    bandwidth: float = 1.0e6       # 1 MHz
    noise_power: float = 1.0e-13   # -100 dBm (sigma^2)
    path_loss_ref: float = 1.42e-4 # Reference path loss rho_0 at 1m (-38.4 dB)
    path_loss_exp: float = 2.0     # Free-space / LoS exponent
    
    # Directional Jammer
    jammer_pos: np.ndarray = field(default_factory=lambda: np.array([450.0, 550.0, 0.0]))
    p_jam_tx: float = 2.0          # 2 W (33 dBm)
    jam_beamwidth_deg: float = 45.0
    
    # Static Obstacle Cylinders: (x, y, radius, height)
    obstacles: list = field(default_factory=lambda: [
        (300.0, 350.0, 60.0, 250.0),
        (700.0, 650.0, 70.0, 280.0),
        (500.0, 300.0, 50.0, 200.0)
    ])
    
    # Kinematic Limits
    v_max: float = 20.0            # Max horizontal velocity (m/s)
    a_max: float = 5.0             # Max horizontal acceleration (m/s^2)
    dt_macro: float = 1.0          # High-level PPO tactical step (seconds)
    dt_micro: float = 0.2          # Low-level NMPC control step (seconds)
    l_step: float = 15.0           # Tactical waypoint distance per step
    max_steps: int = 150           # Max steps per episode

@dataclass
class UAVRotaryEnergyConfig:
    """
    Standard rotary-wing aerodynamic power consumption parameters.
    P(v) = P_0 * (1 + 3*v^2 / U_tip^2) + P_i * sqrt(sqrt(1 + v^4 / (4*v_0^4)) - v^2 / (2*v_0^2)) + 0.5 * d_0 * rho * s * A * v^3
    """
    P_0: float = 79.86             # Blade profile power in hover (Watts)
    P_i: float = 88.63             # Induced power in hover (Watts)
    U_tip: float = 120.0           # Tip speed of the rotor blade (m/s)
    v_0: float = 4.03              # Mean rotor induced velocity in hover (m/s)
    d_0: float = 0.012             # Fuselage drag ratio
    rho: float = 1.225             # Air density (kg/m^3)
    s: float = 0.05                # Rotor solidity
    A: float = 0.503               # Rotor disc area (m^2)
    P_comm: float = 5.0            # Onboard avionics and communication circuitry power (Watts)
    battery_capacity_joules: float = 100000.0  # 100 kJ (~28 Wh)

@dataclass
class NMPCConfig:
    N_horizon: int = 5             # Prediction horizon steps
    q_pos: float = 10.0            # Weight for tracking error
    r_acc: float = 0.1             # Weight for acceleration effort
    dt: float = 0.2                # Control discretization step (s)

@dataclass
class PPORewardWeights:
    omega_d: float = 2.0           # Normalized data harvesting reward
    omega_u: float = 0.5           # Jammer evasion / unpredictability reward
    omega_n: float = 1.0           # Distance-to-destination penalty
    omega_c: float = 50.0          # Obstacle / boundary collision penalty
    omega_e: float = 0.005         # Rotary energy depletion penalty (novelty)
