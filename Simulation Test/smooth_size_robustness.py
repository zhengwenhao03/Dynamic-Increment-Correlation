import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import core streaming processing solvers from internal modules
from simulation_strain import simulate_bridge_strain
from streaming_pipeline import RealTimeBridgeMonitor

# Set academic publication-quality standard fonts and typesetting styles
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

def analyze_smooth_size_convergence(smooth_sizes, sim_hours=12, target_joint=0):
    """
    [Computation Core] Evaluates indicator convergence and statistical fluctuations 
    under a constant healthy structural state using different smooth sizes.
    """
    print(f"\n[Task 1 Core] Simulating stationary traffic flow under healthy state...")
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6
    
    # Secure deterministic reproducibility for stochastic traffic flow
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    
    # Generate long-term stationary healthy bridge strain history
    strain_history, _, freq_Hz = simulate_bridge_strain(
        HOUR=sim_hours, Fs=Fs, DT=20,
        LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
        E=E, I=I, k_hinge=k_hinge_baseline, y_sensor=y_sensor, 
        add_noise=False, SNR=None
    )
    
    results_curves = {}
    raw_time_axis = None
    raw_pcc_scatters = None
    
    # Scan through different smoothing window capacities
    for size in tqdm(smooth_sizes, desc="Task 1 Size Scanning"):
        monitor = RealTimeBridgeMonitor(
            bridge_freq_hz=freq_Hz, 
            num_channels=MM, 
            Fs=Fs, 
            ma_threshold=2.0, 
            ma_length_multiple=2.0,
            low_cutoff_multiplier=0.7,
            snr_threshold=5, 
            indicator_smooth_size=size
        )
        
        for sample in strain_history:
            monitor.process_sample(sample)
            
        if len(monitor.single_events_indices) > 0:
            end_indices = np.array([idx[1] for idx in monitor.single_events_indices])
            time_axis = end_indices / Fs / 3600.0  # Map samples to absolute timeline (Hours)
            smooth_inds = np.array(monitor.smoothed_indicators)[:, target_joint]
            
            results_curves[size] = (time_axis, smooth_inds)
            
            # Capture the raw discrete event PCC scatters once (shared across all sizes)
            if raw_pcc_scatters is None:
                raw_time_axis = time_axis
                raw_pcc_scatters = np.array(monitor.correlations_history)[:, target_joint]

    return raw_time_axis, raw_pcc_scatters, results_curves


def plot_smooth_size_convergence(smooth_sizes, raw_time_axis, raw_pcc_scatters, results_curves, sim_hours=12):
    """
    [Visualization Engine] Renders Figure 1: Superimposed Raw Scatters and Smoothed Continuous Curves.
    """
    print("--> Rendering Figure 1: Smooth size convergence comparison...")
    plt.figure(figsize=(8, 4))
    
    # 1. Plot raw discrete single-event correlation coefficients as background scatters
    if raw_pcc_scatters is not None:
        plt.scatter(
            raw_time_axis, raw_pcc_scatters, 
            color='#1f77b4', s=12, alpha=0.25, marker='o', edgecolors='none',
            label='Raw Single-Event PCC'
        )
        
    # 2. Superimpose continuous indicator curves under different smoothing windows
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['--', '-.', '-', '-', '-']
    
    for idx, size in enumerate(smooth_sizes):
        if size in results_curves:
            t_ax, curve = results_curves[size]
            plt.plot(
                t_ax, curve, 
                color=colors[idx % len(colors)], 
                linewidth=2.5 if idx >= 2 else 1.0, 
                linestyle=line_styles[idx % len(line_styles)], 
                label=rf'Smooth Size $N_s = {size}$'  # Used raw string 'rf' to prevent syntax warning
            )
            
    plt.xlabel('Time (Hours)', fontsize=12, fontweight='bold')
    plt.ylabel('Dynamic Correlation Indicator', fontsize=12, fontweight='bold')
    plt.xlim(0.0, sim_hours)
    plt.ylim(0.9, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9, edgecolor='black', ncol=2)
    plt.tight_layout()
    
    plt.savefig('smooth_size_convergence.pdf', dpi=600, bbox_inches='tight')
    print("--> Figure saved successfully as 'smooth_size_convergence.pdf'")


def analyze_smooth_size_sudden_damage(smooth_sizes, hours=[6.0, 10.0], target_joint=0):
    """
    [Computation Core] Simulates an abrupt structural stiffness step drop at mid-timeline 
    to evaluate the transient tracking delay and slope distortions.
    """
    print(f"\n[Task 2] Simulating sudden damage tracking latency (Stiffness Step-Drop)...")
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6
    num_joints = MM - 1
    
    # Segment 1: 4 hours under intact healthy condition
    SEED1 = 42
    random.seed(SEED1)
    np.random.seed(SEED1)
    k_healthy = np.full((num_joints, 1), k_hinge_baseline)
    strain_healthy, _, freq_Hz_healthy = simulate_bridge_strain(
        HOUR=hours[0], Fs=Fs, DT=20, LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
        E=E, I=I, k_hinge=k_healthy, y_sensor=y_sensor, add_noise=False
    )
    
    # Segment 2: 14 hours under sudden localized damage (50% stiffness step drop)
    SEED2 = 84
    random.seed(SEED2)
    np.random.seed(SEED2)
    k_damaged = np.full((num_joints, 1), k_hinge_baseline)
    k_damaged[target_joint, 0] = k_hinge_baseline * 0.5  
    strain_damaged, _, _ = simulate_bridge_strain(
        HOUR=hours[1], Fs=Fs, DT=20, LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
        E=E, I=I, k_hinge=k_damaged, y_sensor=y_sensor, add_noise=False
    )
    
    # Concatenate profiles along temporal axis to generate sudden shock anomaly at exactly t = 4.0 hours
    strain_combined = np.vstack([strain_healthy, strain_damaged])
    
    results_curves = {}
    
    for size in tqdm(smooth_sizes, desc="Task 2 Size Scanning"):
        monitor = RealTimeBridgeMonitor(
            bridge_freq_hz=freq_Hz_healthy, 
            num_channels=MM, 
            Fs=Fs, 
            ma_threshold=2.0, 
            ma_length_multiple=2.0,
            low_cutoff_multiplier=0.7,
            snr_threshold=5, 
            indicator_smooth_size=size
        )
        
        for sample in strain_combined:
            monitor.process_sample(sample)
            
        if len(monitor.single_events_indices) > 0:
            end_indices = np.array([idx[1] for idx in monitor.single_events_indices])
            time_axis = end_indices / Fs / 3600.0
            smooth_inds = np.array(monitor.smoothed_indicators)[:, target_joint]
            results_curves[size] = (time_axis, smooth_inds)

    return results_curves


def plot_smooth_size_sudden_damage(smooth_sizes, results_curves, hours):
    """
    [Visualization Engine] Renders Figure 2: Transient Sudden Damage Tracking Trajectories.
    """
    print("--> Rendering Figure 2: Sudden damage response tracking...")
    plt.figure(figsize=(8, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['--', '-.', '-', '-', '-']
    
    for idx, size in enumerate(smooth_sizes):
        if size in results_curves:
            t_ax, curve = results_curves[size]
            plt.plot(
                t_ax, curve, 
                color=colors[idx % len(colors)], 
                linewidth=2.5 if idx >= 2 else 1.0,
                linestyle=line_styles[idx % len(line_styles)], 
                label=rf'Smooth Size $N_s = {size}$'  # Used raw string 'rf' to prevent syntax warning
            )
    
    damage_time = hours[0]  # The exact time of sudden damage induction
    total_hours = sum(hours)  # Total simulation timeline

    # Add explicit vertical boundary indicating sudden crack/damage induction
    plt.axvline(x=damage_time, color='black', linestyle=':', linewidth=2.0, label='Sudden Damage Anomaly')
    
    plt.xlabel('Time (Hours)', fontsize=12, fontweight='bold')
    plt.ylabel('Smoothed Dynamic Indicator', fontsize=12, fontweight='bold')
    plt.xlim(0.0, total_hours)
    plt.ylim(0.9, 1.0)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='black', ncol=2)
    plt.tight_layout()
    
    plt.savefig('smooth_size_damage_tracking.pdf', dpi=600, bbox_inches='tight')
    print("--> Figure saved successfully as 'smooth_size_damage_tracking.pdf'\n")


if __name__ == '__main__':
    # 1. Define the parameters and configurations
    SMOOTH_SIZES_ARRAY = [10, 50, 100, 200, 300]
    TARGET_JOINT_INDEX = 0
    TOTAL_SIM_HOURS = 12
    DAMAGE_INJECTION_TIME = 4.0

    HOURs = [6.0, 10.0]
    
    # =========================================================================
    # Task 1 Execution: Convergence Behavior Under Stationary State
    # =========================================================================
    # Step A: Run simulation calculation core
    t_axis, scatters, conv_curves = analyze_smooth_size_convergence(
        smooth_sizes=SMOOTH_SIZES_ARRAY, 
        sim_hours=TOTAL_SIM_HOURS, 
        target_joint=TARGET_JOINT_INDEX
    )
    # Step B: Render plot (You can re-run this line instantly with modified plot parameters)
    plot_smooth_size_convergence(
        smooth_sizes=SMOOTH_SIZES_ARRAY, 
        raw_time_axis=t_axis, 
        raw_pcc_scatters=scatters, 
        results_curves=conv_curves, 
        sim_hours=TOTAL_SIM_HOURS
    )
    
    # =========================================================================
    # Task 2 Execution: Dynamic Response Tracking Under Sudden Shock Drop
    # =========================================================================
    # Step A: Run simulation calculation core
    dmg_curves = analyze_smooth_size_sudden_damage(
        smooth_sizes=SMOOTH_SIZES_ARRAY, 
        hours=HOURs,
        target_joint=TARGET_JOINT_INDEX
    )
    # Step B: Render plot (You can re-run this line instantly with modified plot parameters)
    plot_smooth_size_sudden_damage(
        smooth_sizes=SMOOTH_SIZES_ARRAY, 
        results_curves=dmg_curves, 
        hours=HOURs
    )