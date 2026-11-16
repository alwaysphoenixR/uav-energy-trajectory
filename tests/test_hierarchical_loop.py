"""
Module: tests.test_hierarchical_loop
Integration test for the full hierarchical DRL-NMPC execution loop:
1. PPO Tactical Planner outputs u_ppo
2. Multi-step reference horizon p_ref(k) = p_current + k * L_step * u_ppo is generated
3. CasADi NMPC solves constrained quadratic program for a*_0 in <15 ms
4. Environment executes step with a*_0
5. Kinematic invariants and statistics are verified
"""
import numpy as np
import time

from envs.uav_env import UAVTrajectoryEnv, AirspaceConfig
from controllers.nmpc_controller import NMPCTracker, NMPCParams
from agents.ppo_agent import ActorCritic

def test_closed_loop_hierarchical_rollout():
    print("=" * 75)
    print("PHASE 2 INTEGRATION TEST: Closed-Loop Hierarchical DRL-NMPC Rollout")
    print("=" * 75)
    
    # 1. Initialize environment and controller
    air_cfg = AirspaceConfig(max_steps=50, dt=0.5)
    env = UAVTrajectoryEnv(air_cfg=air_cfg)
    nmpc_params = NMPCParams(N_mpc=5, dt=0.5, V_max=20.0, A_max=5.0)
    tracker = NMPCTracker(params=nmpc_params)
    agent = ActorCritic(state_dim=12, action_dim=3)
    
    obs, _ = env.reset()
    L_step = 5.0  # Tactical waypoint step size (m)
    N_horizon = nmpc_params.N_mpc
    
    solve_times = []
    rewards = []
    velocities = []
    accelerations = []
    
    done = False
    step_count = 0
    
    print("Starting 50-step simulation rollout...")
    while not done and step_count < air_cfg.max_steps:
        step_count += 1
        
        # 1. High-level agent selects tactical direction u_ppo
        u_ppo, _, _ = agent.get_action(obs, deterministic=True)
        assert not np.isnan(u_ppo).any(), "PPO tactical direction contains NaN!"
        assert (np.abs(u_ppo) <= 1.0 + 1e-5).all(), "PPO output exceeds [-1, 1] bounds!"
        
        # 2. Generate reference horizon waypoints
        p_current = env.uav.pos
        v_current = env.uav.vel
        p_ref_horizon = np.zeros((3, N_horizon + 1), dtype=np.float64)
        p_ref_horizon[:, 0] = p_current
        for k in range(1, N_horizon + 1):
            p_ref_horizon[:, k] = p_current + k * L_step * u_ppo
            
        # 3. NMPC solves for optimal acceleration command a*_0
        a_opt_0, p_pred, solve_time_ms = tracker.solve(p_current, v_current, p_ref_horizon)
        solve_times.append(solve_time_ms)
        accelerations.append(np.linalg.norm(a_opt_0))
        
        # Assertions on NMPC outputs
        assert not np.isnan(a_opt_0).any(), f"NMPC acceleration has NaN at step {step_count}!"
        assert np.linalg.norm(a_opt_0) <= nmpc_params.A_max + 1e-2, f"Acceleration {np.linalg.norm(a_opt_0):.2f} exceeded A_max {nmpc_params.A_max}!"
        
        # 4. Low-level physical step in environment using optimal acceleration
        obs, reward, terminated, truncated, info = env.step_with_acc(a_opt_0)
        rewards.append(reward)
        velocities.append(np.linalg.norm(env.uav.vel))
        
        # Kinematic safety checks
        speed = np.linalg.norm(env.uav.vel)
        altitude = env.uav.pos[2]
        assert speed <= nmpc_params.V_max + 1e-2, f"Speed {speed:.2f} exceeded V_max {nmpc_params.V_max}!"
        assert nmpc_params.Z_min - 1e-2 <= altitude <= nmpc_params.Z_max + 1e-2, f"Altitude {altitude:.2f} violated bounds [{nmpc_params.Z_min}, {nmpc_params.Z_max}]!"
        
        done = terminated or truncated

    # Summary Statistics
    avg_solve_time = np.mean(solve_times)
    max_solve_time = np.max(solve_times)
    p95_solve_time = np.percentile(solve_times, 95)
    total_reward = np.sum(rewards)
    final_dist = info['dist_to_dest']
    total_energy_kj = info['total_energy_kj']
    total_data_mbits = info['total_data_mbits']
    
    print("\n" + "-" * 75)
    print("ROLLOUT EXECUTION SUMMARY:")
    print("-" * 75)
    print(f"Total Steps Taken:            {step_count}")
    print(f"Cumulative Episode Reward:    {total_reward:.2f}")
    print(f"Final Distance to Target:     {final_dist:.2f} m")
    print(f"Total Data Harvested:         {total_data_mbits:.2f} Mbits")
    print(f"Total Energy Expended:        {total_energy_kj:.3f} kJ")
    print(f"Average NMPC Solve Time:      {avg_solve_time:.2f} ms per step")
    print(f"95th Percentile Solve Time:   {p95_solve_time:.2f} ms")
    print(f"Max NMPC Solve Time:          {max_solve_time:.2f} ms")
    print(f"Max Observed Speed:           {np.max(velocities):.2f} m/s (Limit: {nmpc_params.V_max} m/s)")
    print(f"Max Observed Acceleration:    {np.max(accelerations):.2f} m/s^2 (Limit: {nmpc_params.A_max} m/s^2)")
    print("-" * 75)
    
    assert avg_solve_time < 15.0, f"Average NMPC solve time ({avg_solve_time:.2f} ms) must be under 15 ms!"
    print(">>> PHASE 2 HIERARCHICAL INTEGRATION TEST PASSED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    test_closed_loop_hierarchical_rollout()
