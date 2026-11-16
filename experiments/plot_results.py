"""
Module: experiments.plot_results
Generates publication-quality figures and a Markdown summary table:
- Figure 1: experiments/training_curves.png
- Figure 2: experiments/trajectory_3d_comparison.png
- Figure 3: experiments/metrics_bar_chart.png
- Summary Table: experiments/summary_table.md
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_training_curves(history_file="experiments/training_history.npy"):
    if not os.path.exists(history_file):
        print("Training history file not found, skipping training curves plot.")
        return
        
    hist = np.load(history_file, allow_pickle=True).item()
    timesteps = hist.get('timesteps', [])
    rewards = hist.get('episode_rewards', [])
    ee = hist.get('energy_efficiency', [])
    
    if len(timesteps) == 0:
        return
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Cumulative Episode Reward
    ax1.plot(timesteps, rewards, color='#1f77b4', linewidth=2.0, label='Episodic Reward')
    ax1.set_xlabel('Timesteps', fontsize=12)
    ax1.set_ylabel('Reward', fontsize=12)
    ax1.set_title('PPO Policy Convergence with QiER Buffer', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()
    
    # 2. Energy Efficiency (Mbits/kJ)
    ax2.plot(timesteps, ee, color='#2ca02c', linewidth=2.0, label='Energy Efficiency (Mbits/kJ)')
    ax2.set_xlabel('Timesteps', fontsize=12)
    ax2.set_ylabel('Mbits / kJ', fontsize=12)
    ax2.set_title('Harvested Throughput per Energy Unit', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()
    
    plt.tight_layout()
    out_path = os.path.join("experiments", "training_curves.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_3d_trajectories(results_file="experiments/benchmark_results.json"):
    with open(results_file, "r") as f:
        data = json.load(f)
        
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Key Landmarks
    start_pos = np.array([50.0, 50.0, 50.0])
    dest_pos = np.array([900.0, 900.0, 100.0])
    iot_pos = np.array([500.0, 500.0, 0.0])
    jammer_pos = np.array([450.0, 550.0, 0.0])
    
    ax.scatter(*start_pos, color='green', s=130, marker='o', label='Start (Launch)')
    ax.scatter(*dest_pos, color='red', s=130, marker='X', label='Destination (Recovery)')
    ax.scatter(*iot_pos, color='blue', s=150, marker='^', label='IoT Ground Node')
    ax.scatter(*jammer_pos, color='magenta', s=160, marker='v', label='Directional Jammer')
    
    # Obstacle Cylinders
    obstacles = [(350.0, 350.0, 50.0, 180.0), (650.0, 650.0, 60.0, 190.0)]
    for ox, oy, r, h in obstacles:
        z_grid = np.linspace(0, h, 20)
        theta_grid = np.linspace(0, 2*np.pi, 20)
        theta_mesh, z_mesh = np.meshgrid(theta_grid, z_grid)
        x_mesh = ox + r * np.cos(theta_mesh)
        y_mesh = oy + r * np.sin(theta_mesh)
        ax.plot_surface(x_mesh, y_mesh, z_mesh, color='gray', alpha=0.3)
        
    colors = {
        "Baseline 1 (Direct DRL)": "purple",
        "Baseline 2 (Hierarchical DRL-NMPC - Wang et al.)": "orange",
        "Proposed Method (Energy-Aware QiER-DRL-NMPC)": "green"
    }
    styles = {
        "Baseline 1 (Direct DRL)": ":",
        "Baseline 2 (Hierarchical DRL-NMPC - Wang et al.)": "--",
        "Proposed Method (Energy-Aware QiER-DRL-NMPC)": "-"
    }
    
    for name, m_data in data.items():
        traj = np.array(m_data.get('best_trajectory', []))
        if len(traj) > 0:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
                    color=colors.get(name, 'black'),
                    linestyle=styles.get(name, '-'),
                    linewidth=2.5, label=name)
            
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    ax.set_zlim([0, 200])
    ax.set_xlabel('X Airspace (m)', fontsize=11)
    ax.set_ylabel('Y Airspace (m)', fontsize=11)
    ax.set_zlabel('Altitude Z (m)', fontsize=11)
    ax.set_title('3D UAV Flight Trajectories in Jammer & Obstacle Contested Airspace', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
    plt.tight_layout()
    out_path = os.path.join("experiments", "trajectory_3d_comparison.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def plot_metrics_bar_chart(results_file="experiments/benchmark_results.json"):
    with open(results_file, "r") as f:
        data = json.load(f)
        
    labels = ["Direct DRL", "Hierarchical DRL-NMPC", "Proposed QiER-DRL-NMPC"]
    keys = list(data.keys())
    
    data_means = [np.mean(data[k]['data_harvested_mbits']) for k in keys]
    energy_means = [np.mean(data[k]['energy_expended_kj']) for k in keys]
    ee_means = [np.mean(data[k]['energy_efficiency_mbits_per_kj']) for k in keys]
    
    x = np.arange(len(labels))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    r1 = ax.bar(x - width, data_means, width, label='Harvested Data (Mbits)', color='#1f77b4')
    r2 = ax.bar(x, energy_means, width, label='Energy Expended (kJ)', color='#d62728')
    r3 = ax.bar(x + width, ee_means, width, label='Energy Efficiency (Mbits/kJ)', color='#2ca02c')
    
    ax.set_ylabel('Metric Magnitude', fontsize=12)
    ax.set_title('Comparative Benchmark Across Evaluated Frameworks (20 Episodes)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_path = os.path.join("experiments", "metrics_bar_chart.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")

def generate_summary_table(results_file="experiments/benchmark_results.json"):
    with open(results_file, "r") as f:
        data = json.load(f)
        
    rows = []
    for k, v in data.items():
        succ = np.mean(v['success']) * 100.0
        data_m = np.mean(v['data_harvested_mbits'])
        data_s = np.std(v['data_harvested_mbits'])
        eng_m = np.mean(v['energy_expended_kj'])
        eng_s = np.std(v['energy_expended_kj'])
        ee_m = np.mean(v['energy_efficiency_mbits_per_kj'])
        viols = np.sum(v['kinematic_violations'])
        
        rows.append({
            'Method': k,
            'Success Rate (%)': f"{succ:.1f}%",
            'Harvested Data (Mbits)': f"{data_m:.2f} ± {data_s:.2f}",
            'Energy Expended (kJ)': f"{eng_m:.2f} ± {eng_s:.2f}",
            'Energy Efficiency (Mbits/kJ)': f"{ee_m:.3f}",
            'Kinematic Violations': f"{viols}"
        })
        
    md = "# Performance Comparison Summary (20 Evaluation Episodes)\n\n"
    md += "| Method | Success Rate (%) | Harvested Data (Mbits) | Energy Expended (kJ) | Energy Efficiency (Mbits/kJ) | Kinematic Violations |\n"
    md += "|---|---|---|---|---|---|\n"
    for r in rows:
        md += f"| {r['Method']} | {r['Success Rate (%)']} | {r['Harvested Data (Mbits)']} | {r['Energy Expended (kJ)']} | {r['Energy Efficiency (Mbits/kJ)']} | {r['Kinematic Violations']} |\n"
        
    table_path = os.path.join("experiments", "summary_table.md")
    with open(table_path, "w") as f:
        f.write(md)
        
    print("\n" + "=" * 90)
    print("BENCHMARK SUMMARY TABLE:")
    print("=" * 90)
    print(md)
    print("=" * 90)
    print(f"Saved to: {table_path}")

def main():
    plot_training_curves()
    plot_3d_trajectories()
    plot_metrics_bar_chart()
    generate_summary_table()

if __name__ == "__main__":
    main()
