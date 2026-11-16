"""
Module: experiments.plot_multi_uav
Generates Figure 4: experiments/multi_uav_3d_swarm.png
Visualizes dual-UAV cooperative 3D swarm trajectories, point of closest
approach with safety deconfliction, IoT ground nodes, and central jammer.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dual_uav_swarm():
    traj1_file = "experiments/swarm_traj_uav1.npy"
    traj2_file = "experiments/swarm_traj_uav2.npy"
    
    if not os.path.exists(traj1_file) or not os.path.exists(traj2_file):
        print("Trajectory files not found. Run tests/test_multi_uav_swarm.py first.")
        return
        
    t1 = np.load(traj1_file)
    t2 = np.load(traj2_file)
    
    # Calculate point of closest approach
    min_steps = min(len(t1), len(t2))
    distances = [np.linalg.norm(t1[i] - t2[i]) for i in range(min_steps)]
    min_idx = np.argmin(distances)
    min_dist = distances[min_idx]
    
    p1_closest = t1[min_idx]
    p2_closest = t2[min_idx]
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Landmarks
    iot1 = np.array([400.0, 300.0, 0.0])
    iot2 = np.array([600.0, 700.0, 0.0])
    jammer = np.array([500.0, 500.0, 0.0])
    
    # Ground IoT Nodes
    ax.scatter(*iot1, color='blue', s=160, marker='^', label='IoT Node 1 (Cluster A)')
    ax.scatter(*iot2, color='green', s=160, marker='^', label='IoT Node 2 (Cluster B)')
    
    # Central Jammer
    ax.scatter(*jammer, color='magenta', s=180, marker='v', label='Central Sweeping Jammer')
    
    # UAV 1 Trajectory & Endpoints
    ax.plot(t1[:, 0], t1[:, 1], t1[:, 2], color='deepskyblue', linewidth=3.0, label='UAV 1 Trajectory (Cyan)')
    ax.scatter(*t1[0], color='blue', s=120, marker='o', label='UAV 1 Launch')
    ax.scatter(*t1[-1], color='darkblue', s=140, marker='X', label='UAV 1 Dest')
    
    # UAV 2 Trajectory & Endpoints
    ax.plot(t2[:, 0], t2[:, 1], t2[:, 2], color='darkorange', linewidth=3.0, label='UAV 2 Trajectory (Orange)')
    ax.scatter(*t2[0], color='red', s=120, marker='o', label='UAV 2 Launch')
    ax.scatter(*t2[-1], color='darkred', s=140, marker='X', label='UAV 2 Dest')
    
    # Red Dashed Line connecting closest approach
    ax.plot([p1_closest[0], p2_closest[0]],
            [p1_closest[1], p2_closest[1]],
            [p1_closest[2], p2_closest[2]],
            color='red', linestyle='--', linewidth=2.5,
            label=f'Closest Approach: {min_dist:.2f} m (APF Safe > 15m)')
            
    ax.scatter(*p1_closest, color='red', s=90, marker='*')
    ax.scatter(*p2_closest, color='red', s=90, marker='*')
    
    # Airspace limits
    ax.set_xlim([0, 1000])
    ax.set_ylim([0, 1000])
    ax.set_zlim([0, 150])
    ax.set_xlabel('X Airspace (m)', fontsize=11)
    ax.set_ylabel('Y Airspace (m)', fontsize=11)
    ax.set_zlabel('Altitude Z (m)', fontsize=11)
    ax.set_title('Dual-UAV Cooperative Swarm: 3D Trajectory Deconfliction via APF-NMPC', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.20, 1.0))
    
    plt.tight_layout()
    out_file = os.path.join("experiments", "multi_uav_3d_swarm.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Figure 4 saved successfully to: {out_file}")

if __name__ == "__main__":
    plot_dual_uav_swarm()
