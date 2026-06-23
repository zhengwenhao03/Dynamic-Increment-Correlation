import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import core streaming processing solvers
from simulation_strain import simulate_bridge_strain
from streaming_pipeline import RealTimeBridgeMonitor

# Set academic plotting style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

def run_sudden_damage_simulation(total_sim_hours=2.0, damage_time_hours=1.0, target_joint=3, damage_alpha=0.05):
    """
    Executes a continuous streaming simulation modeling a sudden, brittle joint failure
    by stitching time-domain physical strain responses.
    
    Args:
        total_sim_hours (float): Total continuous monitoring duration (hours).
        damage_time_hours (float): The exact operational timestamp when the structural brittle failure occurs (hours).
        target_joint (int): Index of the joint undergoing sudden structural snapping.
        damage_alpha (float): Remaining stiffness reduction ratio post-failure (e.g., 0.05 means 95% loss).
        
    Returns:
        RealTimeBridgeMonitor: The fully executed edge monitoring instance containing processed metrics.
    """
    print(f"Launching sudden brittle damage simulation: Total duration {total_sim_hours}h, structural snapping at {damage_time_hours}h...")
    
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6  
    num_joints = MM - 1
    
    # Enforce global seed synchronization to secure stable traffic flow reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    
    # --- Phase 1: Intact Baseline Health Simulation (0 -> damage_time) ---
    print("\n[Phase 1] Generating intact baseline health strain data...")
    k_hinge_healthy = np.full((num_joints, 1), k_hinge_baseline)
    strain_phase1, _, freq_Hz_healthy = simulate_bridge_strain(
        HOUR=damage_time_hours, Fs=Fs, DT=20,
        LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
        E=E, I=I, k_hinge=k_hinge_healthy, y_sensor=y_sensor, 
        add_noise=False, SNR=None
    )
    
    # --- Phase 2: Post-Failure Catastrophic Simulation (damage_time -> total_time) ---
    print("\n[Phase 2] Simulating sudden brittle structural snapping! Generating damaged strain data...")
    k_hinge_damaged = np.full((num_joints, 1), k_hinge_baseline)
    k_hinge_damaged[target_joint, 0] = k_hinge_baseline * damage_alpha
    
    strain_phase2, _, _ = simulate_bridge_strain(
        HOUR=total_sim_hours - damage_time_hours, Fs=Fs, DT=20,
        LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
        E=E, I=I, k_hinge=k_hinge_damaged, y_sensor=y_sensor, 
        add_noise=False, SNR=None
    )
    
    # Vertically stack matrices to reconstruct a seamless continuous streaming data timeline
    strain_combined = np.vstack((strain_phase1, strain_phase2))
    
    # Initialize the real-time edge monitoring state-machine engine
    monitor = RealTimeBridgeMonitor(
        bridge_freq_hz=freq_Hz_healthy, 
        num_channels=strain_combined.shape[1], 
        Fs=Fs, ma_threshold=2, snr_threshold=5, indicator_smooth_size=200
    )
    
    # Feed continuous data line-by-line into the real-time stream processing loops
    for sample in tqdm(strain_combined, desc="Streaming data processing"):
        monitor.process_sample(sample)
        
    return monitor


def plot_emergency_alert_timeseries(monitor, damage_time_hours, target_joint):
    """
    Plots the time-evolving real-time monitoring trajectories to demonstrate 
    the distinct cliff-like catastrophic alert capability.
    """
    print("Rendering localized sudden event emergency alert timeline...")
    
    if len(monitor.single_events_indices) == 0:
        print("Warning: No single-vehicle event intercepted. Plotting aborted.")
        return
        
    # Extract absolute event timestamps converted to operational hours
    event_times_hours = np.array([idx[1] for idx in monitor.single_events_indices]) / monitor.Fs / 3600.0
    
    # Extract corresponding converged indicator sequences for the damaged joint
    dyn_indicator = np.array(monitor.smoothed_indicators)[:, target_joint]
    tot_indicator = np.array(monitor.smoothed_indicators_total)[:, target_joint]
    
    plt.figure(figsize=(8, 4))
    
    # Benchmark Curve: Traditional Total Strain (Showing high structural desensitization and alert blindness)
    plt.plot(
        event_times_hours, tot_indicator, 
        color='gray', linewidth=2.5, linestyle='--', alpha=0.8,
        label=f'Joint {target_joint+1} (Total Strain)'
    )
    
    # Proposed Curve: Pure Dynamic Strain Increment (Showing immediate precipitous responsive drop)
    plt.plot(
        event_times_hours, dyn_indicator, 
        color='#d62728', linewidth=2.5, linestyle='-', 
        label=f'Joint {target_joint+1} (Dynamic Increment)'
    )
    
    # Draw vertical reference line pinpointing the exact physical moment of catastrophic failure
    plt.axvline(
        x=damage_time_hours, color='black', linestyle=':', linewidth=2, 
        label='Moment of Sudden Brittle Failure'
    )

    plt.xlim(0.0, max(event_times_hours) + 0.1)
    plt.ylim(0.0, 1.05)
    
    plt.xlabel('Time (h)', fontsize=13, fontweight='bold')
    plt.ylabel('Indicator', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('sudden_damage_alert.pdf', dpi=600, bbox_inches='tight')
    print("Catastrophic alert plot successfully saved as 'sudden_damage_alert.pdf'.")
    # plt.show()


def plot_all_joints_timeseries(monitor, damage_time_hours, target_joint):
    """
    Plots the concurrent real-time indicator paths across all cross-sectional joints,
    proving excellent spatial topology isolation and complete false-alarm immunity.
    """
    print("Rendering multi-joint cross-talk check and spatial isolation curves...")
    
    if len(monitor.single_events_indices) == 0:
        return
        
    event_times_hours = np.array([idx[1] for idx in monitor.single_events_indices]) / monitor.Fs / 3600.0
    
    # Extract indicators across all cross-sectional joint topologies (Shape: N_events x num_joints)
    dyn_indicators = np.array(monitor.smoothed_indicators)
    num_joints = dyn_indicators.shape[1]
    
    plt.figure(figsize=(8, 4))
    
    # Establish a desaturated cool-tone color pool for intact interfaces to emphasize focus
    healthy_colors = plt.cm.Blues(np.linspace(0.4, 0.9, num_joints))
    
    for j in range(num_joints):
        is_damaged = (j == target_joint)
        
        if is_damaged:
            # Damaged Target: High-contrast thick solid red line
            label_str = f'Joint {j+1} (Sudden Failure)'
            plt.plot(
                event_times_hours, dyn_indicators[:, j], 
                color='#d62728', linewidth=3.0, linestyle='-', alpha=1.0, zorder=10,
                label=label_str
            )
        else:
            # Intact Neighbors: Subtle thin dashed cool-tone lines
            label_str = f'Joint {j+1} (Healthy)'
            plt.plot(
                event_times_hours, dyn_indicators[:, j], 
                color=healthy_colors[j], linewidth=1.5, linestyle='--', alpha=0.7, zorder=1,
                label=label_str
            )
            
    # Mark the physical moment of sudden failure
    plt.axvline(
        x=damage_time_hours, color='black', linestyle=':', linewidth=2, zorder=0,
        label='Moment of Sudden Brittle Failure'
    )
    
    plt.xlim(0.0, max(event_times_hours) + 0.1)
    plt.ylim(0.0, 1.05)
    
    plt.xlabel('Time (h)', fontsize=13, fontweight='bold')
    plt.ylabel('Indicator', fontsize=13, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Adjust legend geometry to avoid crowding and overlapping
    plt.legend(loc='lower left', ncol=1, fontsize=10, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('all_joints_isolation.pdf', dpi=600, bbox_inches='tight')
    print("Spatial cross-talk verification plot saved as 'all_joints_isolation.pdf'.")
    # plt.show()


if __name__ == '__main__':
    # =========================================================================
    # Parametric Control Setup for Real-time Continuous Structural Health Monitoring
    # =========================================================================
    TOTAL_HOURS = 12       # Total operational continuous simulation length (Hours)
    DAMAGE_TIME = 4        # The exact scheduled operational hour for sudden structural brittle snapping
    
    # Target joint index configuration 
    # Tip: TARGET_JOINT = 1 denotes Joint 2 (index 1), often representative of 
    # interfaces mapping between the heavy truck crawling lane and the emergency shoulder.
    TARGET_JOINT = 1       
    
    # 1. Execute full continuous streaming FSM simulation loop
    monitor_instance = run_sudden_damage_simulation(
        total_sim_hours=TOTAL_HOURS, 
        damage_time_hours=DAMAGE_TIME, 
        target_joint=TARGET_JOINT,
        damage_alpha=0.2   # Stiffness instantaneously plunges down to 20% of baseline health
    )
    
    # 2. Export Figure 1: Single-interface catastrophic alert gap comparison
    plot_emergency_alert_timeseries(
        monitor=monitor_instance, 
        damage_time_hours=DAMAGE_TIME, 
        target_joint=TARGET_JOINT
    )

    # 3. Export Figure 2: Multi-interface structural network topology isolation validation
    plot_all_joints_timeseries(
        monitor=monitor_instance, 
        damage_time_hours=DAMAGE_TIME, 
        target_joint=TARGET_JOINT
    )