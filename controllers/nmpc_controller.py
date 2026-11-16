"""
Nonlinear Model Predictive Control (NMPC) Tracker using CasADi optimization solver.
Decouples high-level strategic tactical goals from low-level kinematic physical feasibility.
"""
import numpy as np
import casadi as ca
from configs.settings import SimulationConfig, NMPCConfig

class NMPCTracker:
    def __init__(self, sim_cfg: SimulationConfig, mpc_cfg: NMPCConfig = None):
        self.sim_cfg = sim_cfg
        self.mpc_cfg = mpc_cfg if mpc_cfg is not None else NMPCConfig()
        self._build_solver()
        
    def _build_solver(self):
        """
        Build parametric CasADi NLP solver for reference trajectory tracking:
        min sum_{k=0}^{N-1} ||p(k) - p_ref(k)||^2_Q + ||a(k)||^2_R
        s.t. 
            p(k+1) = p(k) + v(k)*dt + 0.5*a(k)*dt^2
            v(k+1) = v(k) + a(k)*dt
            ||v(k)||^2 <= v_max^2
            ||a(k)||^2 <= a_max^2
            z_min <= z(k) <= z_max
        """
        N = self.mpc_cfg.N_horizon
        dt = self.mpc_cfg.dt
        
        # State variables: p in R^3, v in R^3 (total 6 per step)
        P = ca.SX.sym('P', 3, N + 1)
        V = ca.SX.sym('V', 3, N + 1)
        
        # Control inputs: a in R^3 (total 3 per step)
        A = ca.SX.sym('A', 3, N)
        
        # Parameters: Initial state (p0, v0) and Reference trajectory P_ref (3 x (N+1))
        P_init = ca.SX.sym('P_init', 3)
        V_init = ca.SX.sym('V_init', 3)
        P_ref = ca.SX.sym('P_ref', 3, N + 1)
        
        cost = 0
        g = []
        lbg = []
        ubg = []
        
        # Initial condition constraints
        g.append(P[:, 0] - P_init)
        lbg.extend([0.0, 0.0, 0.0])
        ubg.extend([0.0, 0.0, 0.0])
        
        g.append(V[:, 0] - V_init)
        lbg.extend([0.0, 0.0, 0.0])
        ubg.extend([0.0, 0.0, 0.0])
        
        for k in range(N):
            # Tracking cost
            pos_err = P[:, k+1] - P_ref[:, k+1]
            cost += self.mpc_cfg.q_pos * ca.dot(pos_err, pos_err)
            
            # Control effort cost
            cost += self.mpc_cfg.r_acc * ca.dot(A[:, k], A[:, k])
            
            # Kinematic integration
            p_next = P[:, k] + V[:, k] * dt + 0.5 * A[:, k] * (dt ** 2)
            v_next = V[:, k] + A[:, k] * dt
            
            g.append(P[:, k+1] - p_next)
            lbg.extend([0.0, 0.0, 0.0])
            ubg.extend([0.0, 0.0, 0.0])
            
            g.append(V[:, k+1] - v_next)
            lbg.extend([0.0, 0.0, 0.0])
            ubg.extend([0.0, 0.0, 0.0])
            
            # Speed bound: ||v||^2 <= v_max^2
            g.append(ca.dot(V[:, k+1], V[:, k+1]))
            lbg.append(0.0)
            ubg.append(self.sim_cfg.v_max ** 2)
            
            # Acc bound: ||a||^2 <= a_max^2
            g.append(ca.dot(A[:, k], A[:, k]))
            lbg.append(0.0)
            ubg.append(self.sim_cfg.a_max ** 2)
            
            # Altitude bounds
            g.append(P[2, k+1])
            lbg.append(self.sim_cfg.z_bounds[0])
            ubg.append(self.sim_cfg.z_bounds[1])
            
        # Decision variable vector
        opt_vars = ca.vertcat(ca.reshape(P, -1, 1), ca.reshape(V, -1, 1), ca.reshape(A, -1, 1))
        params = ca.vertcat(P_init, V_init, ca.reshape(P_ref, -1, 1))
        
        nlp = {
            'x': opt_vars,
            'f': cost,
            'g': ca.vertcat(*g),
            'p': params
        }
        
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 50
        }
        self.solver = ca.nlpsol('nmpc_solver', 'ipopt', nlp, opts)
        self.lbg = lbg
        self.ubg = ubg
        self.n_vars = opt_vars.size1()
        
    def solve(self, p_current: np.ndarray, v_current: np.ndarray, p_ref_horizon: np.ndarray) -> np.ndarray:
        """
        Solves the NMPC optimization and returns the immediate control input a*_v(0).
        p_ref_horizon: shape (3, N+1)
        """
        N = self.mpc_cfg.N_horizon
        p_init_val = p_current.reshape(3, 1)
        v_init_val = v_current.reshape(3, 1)
        p_ref_val = p_ref_horizon.reshape(-1, 1)
        
        param_val = np.vstack([p_init_val, v_init_val, p_ref_val])
        x0 = np.zeros(self.n_vars)
        
        try:
            sol = self.solver(x0=x0, p=param_val, lbg=self.lbg, ubg=self.ubg)
            x_opt = np.array(sol['x']).flatten()
            
            # Acceleration starts after (N+1)*3 pos and (N+1)*3 vel
            acc_offset = 2 * (N + 1) * 3
            a_opt_0 = x_opt[acc_offset:acc_offset + 3]
            return a_opt_0
        except Exception:
            # Fallback proportional acceleration if solver is infeasible
            target = p_ref_horizon[:, 1] if p_ref_horizon.shape[1] > 1 else p_ref_horizon[:, 0]
            err = target - p_current
            acc_fallback = np.clip(err, -self.sim_cfg.a_max, self.sim_cfg.a_max)
            return acc_fallback
