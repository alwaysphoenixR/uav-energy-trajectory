"""
Unit test and physical simulation verification for UAVDynamics and WirelessChannel.
"""
import numpy as np
from envs.uav_dynamics import UAVDynamics, UAVPhysicalParams
from envs.channel_model import WirelessChannel, ChannelParameters

def test_10s_straight_flight():
    print("=" * 70)
    print("TEST 1: UAV Dynamics - 10-second straight flight at 10 m/s")
    print("=" * 70)
    
    uav = UAVDynamics(initial_pos=np.array([0.0, 0.0, 50.0]))
    dt = 0.1
    total_time = 10.0
    steps = int(total_time / dt)
    
    # Accelerate to 10 m/s along x-axis, then maintain constant speed
    for s in range(steps):
        if uav.vel[0] < 10.0:
            a_cmd = np.array([5.0, 0.0, 0.0])
        else:
            a_cmd = np.array([0.0, 0.0, 0.0])
        pos, vel, step_energy = uav.step(a_cmd, dt)
        
    print(f"Final UAV Position:   {uav.pos}")
    print(f"Final UAV Velocity:   {uav.vel} (speed: {np.linalg.norm(uav.vel):.2f} m/s)")
    print(f"Total Flight Time:    {total_time:.1f} s")
    print(f"Total Energy Expended:{uav.total_energy_joules:.2f} Joules ({uav.total_energy_joules/1000.0:.3f} kJ)")
    
    assert uav.pos[0] >= 90.0, "UAV should have traveled at least 90 meters."
    assert uav.total_energy_joules > 0.0, "Energy expenditure must be positive."
    print(">>> Test 1 PASSED.")

def test_air_to_ground_channel_flyby():
    print("\n" + "=" * 70)
    print("TEST 2: Wireless Channel - Overhead Flyby with Directional Jammer")
    print("=" * 70)
    
    channel = WirelessChannel()
    p_iot = np.array([500.0, 500.0, 0.0])
    p_jammer = np.array([400.0, 400.0, 0.0])
    
    # UAV flies from (0, 500, 50) to (1000, 500, 50) at constant 10 m/s
    start_pos = np.array([0.0, 500.0, 50.0])
    end_pos = np.array([1000.0, 500.0, 50.0])
    speed = 10.0  # m/s
    total_distance = np.linalg.norm(end_pos - start_pos)
    flight_time = total_distance / speed
    dt = 0.5
    num_steps = int(flight_time / dt)
    
    uav = UAVDynamics(initial_pos=start_pos)
    # Set initial velocity directly along X
    uav.vel = np.array([speed, 0.0, 0.0])
    
    total_bits_collected = 0.0
    sinr_history = []
    rate_history = []
    
    for _ in range(num_steps):
        pos, vel, _ = uav.step(commanded_acc=np.array([0.0, 0.0, 0.0]), dt=dt)
        rate_bps, sinr = channel.compute_data_rate(pos, p_iot, [p_jammer])
        bits = rate_bps * dt
        total_bits_collected += bits
        sinr_history.append(sinr)
        rate_history.append(rate_bps)
        
    total_mbits = total_bits_collected / 1.0e6
    avg_rate_mbps = (np.mean(rate_history)) / 1.0e6
    peak_sinr_db = 10.0 * np.log10(np.max(sinr_history))
    min_sinr_db = 10.0 * np.log10(np.min(sinr_history))
    
    print(f"Flight Duration:      {flight_time:.1f} s (Trajectory: {start_pos} -> {end_pos})")
    print(f"Total Data Harvested: {total_mbits:.2f} Mbits")
    print(f"Average Rate:         {avg_rate_mbps:.2f} Mbps")
    print(f"Peak SINR:            {peak_sinr_db:.2f} dB")
    print(f"Min SINR:             {min_sinr_db:.2f} dB")
    print(f"Total Propulsion Energy:{uav.total_energy_joules:.2f} Joules ({uav.total_energy_joules/1000.0:.2f} kJ)")
    print(f"Overall Efficiency:   {total_mbits / (uav.total_energy_joules / 1000.0):.3f} Mbits/kJ")
    
    assert total_mbits > 1.0, "UAV should harvest significant data when passing directly over IoT node."
    print(">>> Test 2 PASSED.")

if __name__ == "__main__":
    test_10s_straight_flight()
    test_air_to_ground_channel_flyby()
    print("\n" + "=" * 70)
    print("ALL ENVIRONMENT CORE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
