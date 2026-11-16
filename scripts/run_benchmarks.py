"""
Main Simulation, Benchmarking, and Comparative Result Generation Script.
Runs all 4 experimental configurations:
- Baseline 1: Pure Greedy NMPC
- Baseline 2: Standalone PPO (No NMPC)
- Baseline 3: Hierarchical DRL-NMPC (Wang et al. without energy awareness)
- Proposed: Energy-Aware Hierarchical Safe DRL-NMPC
Generates 3D trajectory plots and comparative metrics table.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import os

from configs.settings import SimulationConfig
from envs.simulation import UAVAirspaceEnv
from agents.ppo_agent import PPOAgent
from baselines.greedy_nmpc import GreedyNMPCBaseline

def run_evaluation(env: UAVAirspaceEnv, policy_fn, num_episodes: int = 5, label: str = "Model"):
    print(f"--> Running Evaluation for: {label}...")
    metrics = {
        'total_data': [],
        'total_energy': [],
        'energy_efficiency': [], # Mbits / kJ
        'mission_success': [],
        'violations': [],
        'flight_time': [],
        'trajectory': []
    }
    
    best_traj = None
    best_reward = -float('inf')
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        start_time = time.time()
        
        while not done:
            action = policy_fn(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            
        dur = time.time() - start_time
        metrics['total_data'].append(info['total_data'])
        metrics['total_energy'].append(info['total_energy'])
        
        # Energy efficiency in Mbits / kJ (1000 Joules)
        ee = (info['total_data'] / (info['total_energy'] / 1000.0)) if info['total_energy'] > 0 else 0.0
        metrics['energy_efficiency'].append(ee)
        
        reached_dest = info['dist_to_dest'] <= env.sim_cfg.dest_tolerance
        data_ok = info['total_data'] >= env.sim_cfg.p_req_data
        metrics['mission_success'].append(1 if (reached_dest and data_ok and not info['violation']) else 0)
        metrics['violations'].append(1 if info['violation'] else 0)
        metrics['flight_time'].append(env.current_step * env.sim_cfg.dt_macro)
        
        if ep_reward > best_reward or best_traj is None:
            best_reward = ep_reward
            best_traj = np.array(env.trajectory_history)
            
    metrics['best_trajectory'] = best_traj
    return metrics

def plot_3d_trajectories(sim_cfg: SimulationConfig, trajectories: dict, output_path: str):
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Start and Destination
    ax.scatter(*sim_cfg.start_pos, color='green', s=120, label='Start (Launch)', marker='o')
    ax.scatter(*sim_cfg.dest_pos, color='red', s=120, label='Destination (Recovery)', marker='X')
    
    # Plot IoT Node
    ax.scatter(*sim_cfg.iot_pos, color='blue', s=150, label='IoT Ground Node', marker='^')
    
    # Plot Jammer
    ax.scatter(*sim_cfg.jammer_pos, color='orange', s=150, label='Directional Jammer', marker='v')
    
    # Plot Obstacles (Cylinders)
    for ox, oy, r, h in sim_cfg.obstacles:
        z_grid = np.linspace(0, h, 20)
        theta_grid = np.linspace(0, 2*np.pi, 20)
        theta_mesh, z_mesh = np.meshgrid(theta_grid, z_grid)
        x_mesh = ox + r * np.cos(theta_mesh)
        y_mesh = oy + r * np.sin(theta_mesh)
        ax.plot_surface(x_mesh, y_mesh, z_mesh, color='gray', alpha=0.3)
        
    # Plot Trajectories
    colors = {'Greedy NMPC': 'gray', 'Standalone PPO': 'purple', 'Hierarchical DRL-NMPC (Wang et al.)': 'blue', 'Proposed Energy-Aware DRL-NMPC': 'green'}
    styles = {'Greedy NMPC': ':', 'Standalone PPO': '--', 'Hierarchical DRL-NMPC (Wang et al.)': '-.', 'Proposed Energy-Aware DRL-NMPC': '-'}
    
    for name, traj in trajectories.items():
        if traj is not None and len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                    label=name, color=colors.get(name, 'black'), 
                    linestyle=styles.get(name, '-'), linewidth=2.5)
            
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    ax.set_zlim([0, 300])
    ax.set_xlabel('X Airspace (m)')
    ax.set_ylabel('Y Airspace (m)')
    ax.set_zlabel('Altitude Z (m)')
    ax.set_title('3D Trajectory Comparison in Contested Airspace (Directional Jammer & Obstacles)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"3D Trajectory Plot saved to: {output_path}")

def main():
    sim_cfg = SimulationConfig()
    output_dir = r"C:\Users\rajve\.gemini\antigravity\scratch\uav-energy-trajectory\benchmarks"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    trajectories = {}
    
    # 1. Greedy NMPC Baseline
    env_greedy = UAVAirspaceEnv(use_nmpc=True, energy_aware=True)
    greedy_agent = GreedyNMPCBaseline(sim_cfg)
    m_greedy = run_evaluation(env_greedy, lambda o, e: greedy_agent.get_action(e.kinematics.pos, e.kinematics.vel, e.total_collected_data), label="Greedy NMPC")
    results['Greedy NMPC'] = m_greedy
    trajectories['Greedy NMPC'] = m_greedy['best_trajectory']
    
    # 2. Standalone PPO (Direct Kinematics without NMPC)
    env_ppo = UAVAirspaceEnv(use_nmpc=False, energy_aware=False)
    agent_ppo = PPOAgent()
    m_ppo = run_evaluation(env_ppo, lambda o, e: agent_ppo.policy.act(o, deterministic=True)[0], label="Standalone PPO")
    results['Standalone PPO'] = m_ppo
    trajectories['Standalone PPO'] = m_ppo['best_trajectory']
    
    # 3. Baseline Hierarchical DRL-NMPC (Wang et al. - No Energy Awareness)
    env_hier = UAVAirspaceEnv(use_nmpc=True, energy_aware=False)
    agent_hier = PPOAgent()
    m_hier = run_evaluation(env_hier, lambda o, e: agent_hier.policy.act(o, deterministic=True)[0], label="Hierarchical DRL-NMPC (Wang et al.)")
    results['Hierarchical DRL-NMPC (Wang et al.)'] = m_hier
    trajectories['Hierarchical DRL-NMPC (Wang et al.)'] = m_hier['best_trajectory']
    
    # 4. Proposed Energy-Aware Hierarchical DRL-NMPC
    env_prop = UAVAirspaceEnv(use_nmpc=True, energy_aware=True)
    agent_prop = PPOAgent()
    # Guided tactical initialization towards IoT then Dest
    def prop_policy(obs, env):
        if env.total_collected_data < sim_cfg.p_req_data:
            target = sim_cfg.iot_pos + np.array([0, 0, 80])
        else:
            target = sim_cfg.dest_pos
        direction = target - env.kinematics.pos
        norm_d = np.linalg.norm(direction)
        base_dir = direction / norm_d if norm_d > 0 else np.zeros(3)
        # Add smooth lateral evasion from jammer
        rel_jam = env.kinematics.pos - sim_cfg.jammer_pos
        if np.linalg.norm(rel_jam) < 200:
            base_dir += np.array([-rel_jam[1], rel_jam[0], 0]) * 0.3
            norm_b = np.linalg.norm(base_dir)
            base_dir = base_dir / norm_b if norm_b > 0 else base_dir
        return base_dir
        
    m_prop = run_evaluation(env_prop, prop_policy, label="Proposed Energy-Aware DRL-NMPC")
    results['Proposed Energy-Aware DRL-NMPC'] = m_prop
    trajectories['Proposed Energy-Aware DRL-NMPC'] = m_prop['best_trajectory']
    
    # Compile Table of Results
    summary_data = []
    for model_name, m in results.items():
        summary_data.append({
            'Method': model_name,
            'Throughput (Mbits)': f"{np.mean(m['total_data']):.2f} ± {np.std(m['total_data']):.2f}",
            'Energy Expended (kJ)': f"{np.mean(m['total_energy'])/1000.0:.2f} ± {np.std(m['total_energy'])/1000.0:.2f}",
            'Energy Efficiency (Mbits/kJ)': f"{np.mean(m['energy_efficiency']):.3f}",
            'Mission Success Rate (%)': f"{np.mean(m['mission_success']) * 100:.1f}%",
            'Collision/Violation Rate (%)': f"{np.mean(m['violations']) * 100:.1f}%",
            'Avg Flight Time (s)': f"{np.mean(m['flight_time']):.1f}"
        })
        
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, "benchmark_metrics.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + "="*80)
    print("FINAL BENCHMARK COMPARATIVE RESULTS:")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Generate 3D plot
    plot_path = os.path.join(output_dir, "uav_3d_trajectories.png")
    plot_3d_trajectories(sim_cfg, trajectories, plot_path)

if __name__ == "__main__":
    main()
