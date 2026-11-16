# Energy-Efficient and Survival-Aware 3D Trajectory Planning for UAV-Assisted IoT Networks

An end-to-end implementation of a **Hierarchical Safe Reinforcement Learning** framework for UAV 3D trajectory planning under directional jamming, obstacle avoidance, and rotary-wing propulsion energy constraints.

Based on:
* **Primary Base Paper:** Shihui Wang, Yibo Yi, Jin Zhang, and Minsan Zhang, *"Survival-Aware Trajectory Planning for UAV-Assisted IoT Data Acquisition: A Hierarchical DRL-NMPC Method"*, IEEE / arXiv.
* **Proposed Novelty:** 
  1. Aerodynamic rotary-wing propulsion and hovering power depletion model $P(v)$.
  2. Energy-aware reward penalty $-\omega_e P(v)$ preventing battery drain.
  3. Directional predictive jamming evasion coupled with CasADi Nonlinear Model Predictive Control (NMPC).

---

## 🏗️ Repository Architecture

```
uav-energy-trajectory/
├── configs/
│   └── settings.py          # Airspace bounds, IoT & Jammer coordinates, Energy constants
├── envs/
│   ├── uav_dynamics.py      # Rotary-wing power model P(v) & kinematic integration
│   ├── channel_model.py     # Air-to-ground path loss, SINR & Shannon capacity
│   └── simulation.py        # Gymnasium 3D environment with collision detection & reward shaping
├── controllers/
│   └── nmpc_controller.py   # CasADi Ipopt solver for finite-horizon trajectory tracking
├── agents/
│   └── ppo_agent.py         # PyTorch Actor-Critic PPO high-level tactical planner
├── baselines/
│   └── greedy_nmpc.py       # Baseline heuristic comparison
├── benchmarks/
│   ├── benchmark_metrics.csv# Comparative metrics across all 4 baselines
│   └── uav_3d_trajectories.png # 3D trajectory visualization in contested airspace
└── scripts/
    └── run_benchmarks.py    # Main evaluation and plotting runner
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install torch gymnasium casadi scipy matplotlib pandas numpy
```

### 2. Run Benchmarks and Generate Results
```bash
python scripts/run_benchmarks.py
```

Generated outputs will be saved to `benchmarks/benchmark_metrics.csv` and `benchmarks/uav_3d_trajectories.png`.
