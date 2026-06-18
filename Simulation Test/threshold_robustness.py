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

def compute_threshold_robustness(threshold_list, alpha_list, target_joint, sim_hours=6):
    """
    Task 1: Compute and extract the evolution curves of the dynamic-increment indicator 
    with respect to structural stiffness degradation (alpha) under different triggering thresholds.
    """
    print(f"\n[Task 1] Launching full-cycle threshold sensitivity scan for stiffness degradation (Target Joint: {target_joint+1})...")
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6
    num_joints = MM - 1
    
    results = {th: [] for th in threshold_list}
    
    for alpha in tqdm(alpha_list, desc="Stiffness Scanning"):
        # Force lock the random seed to ensure the spatial distribution of stochastic traffic is absolutely identical under each alpha
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        
        k_hinge_array = np.full((num_joints, 1), k_hinge_baseline)
        k_hinge_array[target_joint, 0] = k_hinge_baseline * (1 - alpha)
        
        strain_history, _, freq_Hz = simulate_bridge_strain(
            HOUR=sim_hours, Fs=Fs, DT=20,
            LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
            E=E, I=I, k_hinge=k_hinge_array, y_sensor=y_sensor, 
            add_noise=False, SNR=None
        )
        
        # Allow all configured thresholds to share the same set of physical strain data to strictly control variables
        for thresh in threshold_list:
            monitor = RealTimeBridgeMonitor(
                bridge_freq_hz=freq_Hz, 
                num_channels=MM, 
                Fs=Fs, 
                ma_threshold=thresh, 
                snr_threshold=5, 
                indicator_smooth_size=2000 
            )
            for sample in strain_history:
                monitor.process_sample(sample)
                
            if len(monitor.smoothed_indicators) > 0:
                results[thresh].append(monitor.smoothed_indicators[-1][target_joint])
            else:
                results[thresh].append(np.nan)
                
    return results

def plot_threshold_robustness(threshold_list, alpha_list, results, target_joint):
    """Render the stiffness evolution curve plot for Task 1."""
    plt.figure(figsize=(8, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, thresh in enumerate(threshold_list):
        plt.plot(
            alpha_list, results[thresh], 
            color=colors[idx % len(colors)], linewidth=2.0, 
            marker=markers[idx % len(markers)], markersize=5, alpha=0.85,
            label=f'Threshold $\\gamma$ = {thresh}'
        )
        
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel(f'Local Stiffness Reduction Factor of Joint {target_joint+1} ($\\alpha$)', fontsize=12, fontweight='bold')
    plt.ylabel('Converged Indicator Value', fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('threshold_robustness.pdf', dpi=600, bbox_inches='tight')
    print("Stiffness evolution comparison plot saved as 'threshold_robustness.pdf'")


def compute_threshold_convergence(threshold_list, target_joint, fixed_k_hinge=3e6, sim_hours=12):
    """
    Task 2: Compute and extract the convergence trajectories over time/events under different thresholds at a specific damage state (k=3e6).
    """
    print(f"\n[Task 2] Launching long-term convergence analysis under a fixed stiffness state (Fixed Hinge Joint Stiffness = {fixed_k_hinge})...")
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6
    num_joints = MM - 1
    
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    
    k_hinge_array = np.full((num_joints, 1), k_hinge_baseline)
    k_hinge_array[target_joint, 0] = fixed_k_hinge
    
    print(f"Generating {sim_hours} hours of structural dynamic response data source...")
    strain_history, _, freq_Hz = simulate_bridge_strain(
        HOUR=sim_hours, Fs=Fs, DT=20,
        LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
        E=E, I=I, k_hinge=k_hinge_array, y_sensor=y_sensor, 
        add_noise=False, SNR=None
    )
    
    results_convergence = {}
    
    for thresh in threshold_list:
        monitor = RealTimeBridgeMonitor(
            bridge_freq_hz=freq_Hz, 
            num_channels=MM, 
            Fs=Fs, 
            ma_threshold=thresh, 
            snr_threshold=5, 
            indicator_smooth_size=200000
        )
        
        for sample in strain_history:
            monitor.process_sample(sample)
            
        if len(monitor.single_events_indices) > 0:
            # Extract the absolute physical end times of valid captured events and map them to the "Hours" axis
            end_indices = np.array([idx[1] for idx in monitor.single_events_indices])
            time_axis = end_indices / Fs / 3600.0  
            smooth_inds = np.array(monitor.smoothed_indicators)[:, target_joint]
            results_convergence[thresh] = (time_axis, smooth_inds)
            print(f"Threshold = {thresh:2d} | Captured valid vehicle events: {len(end_indices):4d}")
        else:
            results_convergence[thresh] = (np.array([]), np.array([]))
            print(f"Threshold = {thresh:2d} | Captured valid vehicle events:    0")
            
    return results_convergence

def plot_threshold_convergence(threshold_list, results_convergence):
    """Render the convergence trajectory plot over time for Task 2, adhering to the style of the plot_indicators_over_time function."""
    plt.figure(figsize=(8, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':', '-']
    
    for idx, thresh in enumerate(threshold_list):
        time_axis, smooth_inds = results_convergence[thresh]
        if len(time_axis) > 0:
            # Filter out NaN values and align data slices
            valid_mask = ~np.isnan(smooth_inds)
            plt.plot(
                time_axis[valid_mask], smooth_inds[valid_mask], 
                color=colors[idx % len(colors)], linewidth=2.5, linestyle=line_styles[idx % len(line_styles)], 
                alpha=0.9, label=f'Threshold $\\gamma$ = {thresh}'
            )
            
    plt.xlabel('Time (Hours)', fontsize=12, fontweight='bold')
    plt.ylabel('Indicator', fontsize=12, fontweight='bold')
    
    # Due to the presence of damage (k=3e6), the indicator convergence value will fall between 0.8 and 0.9. Dynamically adjust the Y-axis range to highlight the comparison.
    plt.ylim(0.8, 1.0) 
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('threshold_convergence.pdf', dpi=600, bbox_inches='tight')
    print("Time convergence comparison plot saved as 'threshold_convergence.pdf'\n")


if __name__ == '__main__':
    # Define the core threshold array for comparison
    THRESHOLD_LIST = [2, 5, 10, 20, 30]
    
    # Lock the target damage interface
    TARGET_JOINT = 0
    
    # ==========================================
    # Execute Task 1: Stiffness degradation trajectory comparison
    # ==========================================
    # Utilize 11 calculation nodes to ensure physical waveform accuracy while controlling script execution time
    alpha_list = np.linspace(1.0, 0.0, 11) 
    res_stiffness = compute_threshold_robustness(THRESHOLD_LIST, alpha_list, TARGET_JOINT, sim_hours=4)
    plot_threshold_robustness(THRESHOLD_LIST, alpha_list, res_stiffness, TARGET_JOINT)
    
    # ==========================================
    # Execute Task 2: Time/event convergence comparison under a quantitative stiffness condition
    # ==========================================
    FIXED_STIFFNESS = 3e6
    res_convergence = compute_threshold_convergence(THRESHOLD_LIST, TARGET_JOINT, fixed_k_hinge=FIXED_STIFFNESS, sim_hours=12)
    plot_threshold_convergence(THRESHOLD_LIST, res_convergence)