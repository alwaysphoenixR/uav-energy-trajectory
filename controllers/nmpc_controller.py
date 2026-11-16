"""
Module: controllers.nmpc_controller
CasADi-based Nonlinear Model Predictive Control (NMPC) Tracker.
Solves constrained quadratic program within <15 ms to enforce physical safety,
kinematic acceleration and velocity limits, and altitude bounds.
"""
from dataclasses import dataclass
import numpy as np
import casadi as ca
import time

@dataclass
class NMPCParams:
    N_mpc: int = 5                  # Prediction horizon
    dt: float = 0.5                 # Sampling time (s)
    V_max: float = 20.0             # Max speed (m/s)
    A_max: float = 5.0              # Max acceleration (m/s^2)
    Z_min: float = 20.0             # Min altitude (m)
    Z_max: float = 200.0            # Max altitude (m)
    q_pos: float = 10.0             # Tracking error weight
    r_acc: float = 0.1              # Control effort weight


class NMPCTracker:
    """
    CasADi finite-horizon quadratic tracking controller.
    """
    def __init__(self, params: NMPCParams = None):
        self.params = params if params is not None else NMPCParams()
        self._build_opti_solver()

    def _build_opti_solver(self):
        """
        Formulates the constrained NLP tracking formulation using CasADi Opti:
        min sum_{k=0}^{N_mpc-1} ||p(k) - p_ref(k)||_Q^2 + ||a(k)||_R^2
        s.t.
            p(k+1) = p(k) + v(k)*dt + 0.5*a(k)*dt^2
            v(k+1) = v(k) + a(k)*dt
            ||v(k)||_2 <= V_max
            ||a(k)||_2 <= A_max
            Z_min <= p_z(k) <= Z_max
        """
        N = self.params.N_mpc
        dt = self.params.dt
        
        opti = ca.Opti()
        
        # Decision variables
        P = opti.variable(3, N + 1)  # Positions
        V = opti.variable(3, N + 1)  # Velocities
        A = opti.variable(3, N)      # Accelerations
        
        # Parameters (Inputs to solver at each step)
        P_init = opti.parameter(3)
        V_init = opti.parameter(3)
        P_ref = opti.parameter(3, N + 1)
        
        # Initial boundary conditions
        opti.subject_to(P[:, 0] == P_init)
        opti.subject_to(V[:, 0] == V_init)
        
        cost = 0
        for k in range(N):
            pos_err = P[:, k + 1] - P_ref[:, k + 1]
            cost += self.params.q_pos * ca.dot(pos_err, pos_err)
            cost += self.params.r_acc * ca.dot(A[:, k], A[:, k])
            
            # Kinematic state transition constraints
            opti.subject_to(P[:, k + 1] == P[:, k] + V[:, k] * dt + 0.5 * A[:, k] * (dt ** 2))
            opti.subject_to(V[:, k + 1] == V[:, k] + A[:, k] * dt)
            
            # Velocity constraint: ||v||^2 <= V_max^2
            opti.subject_to(ca.dot(V[:, k + 1], V[:, k + 1]) <= self.params.V_max ** 2)
            
            # Acceleration constraint: ||a||^2 <= A_max^2
            opti.subject_to(ca.dot(A[:, k], A[:, k]) <= self.params.A_max ** 2)
            
            # Altitude limits: Z_min <= z <= Z_max
            opti.subject_to(P[2, k + 1] >= self.params.Z_min)
            opti.subject_to(P[2, k + 1] <= self.params.Z_max)
            
        opti.minimize(cost)
        
        # Solver options: Fast IPOPT configuration for <15ms online latency
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 40,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
            'ipopt.warm_start_init_point': 'yes'
        }
        opti.solver('ipopt', opts)
        
        self.opti = opti
        self.P = P
        self.V = V
        self.A = A
        self.P_init = P_init
        self.V_init = V_init
        self.P_ref = P_ref

    def solve(self, current_p: np.ndarray, current_v: np.ndarray, p_ref_horizon: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Solves the tracking optimization problem.
        Args:
            current_p: (3,) current UAV coordinates
            current_v: (3,) current UAV velocity
            p_ref_horizon: (3, N+1) reference trajectory over horizon
        Returns:
            a_opt_0: (3,) optimal immediate acceleration command
            p_pred: (3, N+1) predicted position trajectory
            solve_time_ms: solve latency in milliseconds
        """
        t0 = time.perf_counter()
        
        # Set parameter values
        self.opti.set_value(self.P_init, np.array(current_p, dtype=np.float64))
        self.opti.set_value(self.V_init, np.array(current_v, dtype=np.float64))
        self.opti.set_value(self.P_ref, np.array(p_ref_horizon, dtype=np.float64))
        
        try:
            sol = self.opti.solve()
            solve_time_ms = (time.perf_counter() - t0) * 1000.0
            
            a_opt = sol.value(self.A)
            a_0 = a_opt[:, 0] if a_opt.ndim > 1 else a_opt
            p_pred = sol.value(self.P)
            
            return np.array(a_0, dtype=np.float64), np.array(p_pred, dtype=np.float64), solve_time_ms
        except Exception:
            # Fallback PD control if solver fails
            solve_time_ms = (time.perf_counter() - t0) * 1000.0
            target_p = p_ref_horizon[:, 1] if p_ref_horizon.shape[1] > 1 else p_ref_horizon[:, 0]
            err = target_p - current_p
            a_fallback = np.clip(err, -self.params.A_max, self.params.A_max)
            p_pred = np.tile(current_p[:, None], (1, self.params.N_mpc + 1))
            return a_fallback, p_pred, solve_time_ms
