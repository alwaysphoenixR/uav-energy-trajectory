"""
Module: tests.test_multi_uav_swarm
Closed-loop verification of Decentralized Multi-UAV Swarm with
Decentralized CasADi NMPC and APF Collision Avoidance Filter.
Introduces intentional crossing trajectories to test active deconfliction.
"""
import numpy as np
import time

from envs.multi_uav_env import MultiUAVTrajectoryEnv, MultiAirspaceConfig
from controllers.nmpc_controller import NMPCTracker, NMPCParams
from controllers.apf_planner import APFCollisionFilter, APFParams

def test_cooperative_multi_uav_rollout():
    print("=" * 80)
    print("PHASE 4 VALIDATION: Cooperative Dual-UAV Swarm with APF Safety Filter")
    print("=" * 80)
    
    # Configure intentional diagonal crossing trajectory to test deconfliction
    cfg = MultiAirspaceConfig(
        uav1_start=np.array([100.0, 300.0, 50.0]),
        uav1_dest=np.array([900.0, 700.0, 50.0]),   # Path: (100, 300) -> (900, 700)
        uav2_start=np.array([100.0, 700.0, 50.0]),
        uav2_dest=np.array([900.0, 300.0, 50.0]),   # Path: (100, 700) -> (900, 300) (cross at midpoint (500, 500))
        max_steps=50,
        dt=0.5
    )
    
    env = MultiUAVTrajectoryEnv(cfg=cfg)
    
    # Two independent decentralized NMPC trackers
    nmpc_params = NMPCParams(N_mpc=5, dt=0.5, V_max=20.0, A_max=5.0)
    tracker1 = NMPCTracker(params=nmpc_params)
    tracker2 = NMPCTracker(params=nmpc_params)
    
    # APF collision safety filter
    apf_filter = APFCollisionFilter(APFParams(d_safe=40.0, k_rep=80.0, a_rep_max=4.5))
    
    obs, _ = env.reset()
    L_step = 6.0
    N = nmpc_params.N_mpc
    
    solve_times_1 = []
    solve_times_2 = []
    separation_distances = []
    
    step_count = 0
    done = False
    
    print("Running 50-step intentional crossing trajectory rollout...")
    while not done and step_count < cfg.max_steps:
        step_count += 1
        
        p1 = env.uav1.pos
        v1 = env.uav1.vel
        p2 = env.uav2.pos
        v2 = env.uav2.vel
        
        # 1. Tactical Heading Guidance towards destinations
        d1 = cfg.uav1_dest - p1
        d2 = cfg.uav2_dest - p2
        u1 = d1 / max(1e-3, np.linalg.norm(d1))
        u2 = d2 / max(1e-3, np.linalg.norm(d2))
        
        # 2. Build Nominal Reference Horizons
        p_ref1 = np.zeros((3, N + 1))
        p_ref2 = np.zeros((3, N + 1))
        p_ref1[:, 0] = p1
        p_ref2[:, 0] = p2
        for k in range(1, N + 1):
            p_ref1[:, k] = p1 + k * L_step * u1
            p_ref2[:, k] = p2 + k * L_step * u2
            
        # 3. Apply APF Safety Filter to decouple and repel drones
        p_ref1_safe = apf_filter.filter_reference_horizon(p_ref1, p2, dt=cfg.dt)
        p_ref2_safe = apf_filter.filter_reference_horizon(p_ref2, p1, dt=cfg.dt)
        
        # 4. Independent Decentralized NMPC Solves
        a_cmd1, _, t1 = tracker1.solve(p1, v1, p_ref1_safe)
        a_cmd2, _, t2 = tracker2.solve(p2, v2, p_ref2_safe)
        
        solve_times_1.append(t1)
        solve_times_2.append(t2)
        
        # 5. Environment Step
        obs, rewards, terminated, truncated, info = env.step_with_acc(a_cmd1, a_cmd2)
        d_12 = info['d_12']
        separation_distances.append(d_12)
        
        # Invariant Assertions
        assert not np.isnan(a_cmd1).any() and not np.isnan(a_cmd2).any(), "NaN found in accelerations!"
        assert np.linalg.norm(a_cmd1) <= nmpc_params.A_max + 1e-2, f"UAV1 Acc {np.linalg.norm(a_cmd1):.2f} exceeded A_max!"
        assert np.linalg.norm(a_cmd2) <= nmpc_params.A_max + 1e-2, f"UAV2 Acc {np.linalg.norm(a_cmd2):.2f} exceeded A_max!"
        assert np.linalg.norm(env.uav1.vel) <= nmpc_params.V_max + 1e-2, f"UAV1 Speed exceeded V_max!"
        assert np.linalg.norm(env.uav2.vel) <= nmpc_params.V_max + 1e-2, f"UAV2 Speed exceeded V_max!"
        
        done = terminated or truncated

    min_sep = np.min(separation_distances)
    avg_solve = (np.mean(solve_times_1) + np.mean(solve_times_2)) / 2.0
    
    print("\n" + "-" * 80)
    print("DUAL-UAV SWARM EXECUTION METRICS:")
    print("-" * 80)
    print(f"Total Steps Executed:             {step_count}")
    print(f"Minimum Separation Distance:      {min_sep:.2f} meters (Safety Threshold: 15.0 m)")
    print(f"Total Joint Data Harvested:       {info['total_data_joint_mbit']:.2f} Mbits")
    print(f"  - UAV 1 Throughput:             {info['total_bits_uav1_mbit']:.2f} Mbits")
    print(f"  - UAV 2 Throughput:             {info['total_bits_uav2_mbit']:.2f} Mbits")
    print(f"Total Joint Energy Expended:      {info['total_energy_joint_kj']:.3f} kJ")
    print(f"Average NMPC Tracker Solve Time:  {avg_solve:.2f} ms")
    print(f"Structural Collisions:            {'NONE (Passed)' if not info['collision_event'] else 'COLLISION DETECTED!'}")
    print("-" * 80)
    
    # Assert safety guarantee
    assert min_sep > 15.0, f"Minimum separation {min_sep:.2f}m was below 15.0m threshold!"
    assert not info['collision_event'], "A structural collision occurred!"
    print(">>> DUAL-UAV APF SWARM VALIDATION TEST PASSED SUCCESSFULLY!")
    print("=" * 80)
    
    # Save trajectories for plotting
    np.save("experiments/swarm_traj_uav1.npy", np.array(env.traj_uav1))
    np.save("experiments/swarm_traj_uav2.npy", np.array(env.traj_uav2))
    return min_sep, info

if __name__ == "__main__":
    test_cooperative_multi_uav_rollout()
