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

def compute_window_multiple_robustness(multiple_list, alpha_list, target_joint, sim_hours=6):
    """
    Task 1: Compute and extract the evolution curves of the dynamic-increment indicator 
    with respect to structural stiffness degradation (alpha) under different MWA window length multiples.
    """
    print(f"\n[Task 1] Launching full-cycle window multiple sensitivity scan for stiffness degradation (Target Joint: {target_joint+1})...")
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6
    num_joints = MM - 1
    
    results = {mult: [] for mult in multiple_list}
    
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
        
        # Allow all configured window multipliers to share the same set of physical strain data to strictly control variables
        for multiple in multiple_list:
            monitor = RealTimeBridgeMonitor(
                bridge_freq_hz=freq_Hz, 
                num_channels=MM, 
                Fs=Fs, 
                ma_threshold=2, 
                ma_length_multiple=multiple,  # Inject the parametric sensitive variable
                snr_threshold=5, 
                indicator_smooth_size=2000 
            )
            for sample in strain_history:
                monitor.process_sample(sample)
                
            if len(monitor.smoothed_indicators) > 0:
                results[multiple].append(monitor.smoothed_indicators[-1][target_joint])
            else:
                results[multiple].append(np.nan)
                
    return results

def plot_window_multiple_robustness(multiple_list, alpha_list, results, target_joint):
    """Render the stiffness evolution curve plot for window multiples."""
    plt.figure(figsize=(8, 4))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    
    for idx, multiple in enumerate(multiple_list):
        plt.plot(
            alpha_list, results[multiple], 
            color=colors[idx % len(colors)], linewidth=2.0, 
            marker=markers[idx % len(markers)], markersize=5, alpha=0.85,
            label=f'$n$ = {multiple}'
        )
        
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel(f'Stiffness Reduction Factor of Joint {target_joint+1}', fontsize=12, fontweight='bold')
    plt.ylabel('Indicator', fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('window_multiple_robustness.pdf', dpi=600, bbox_inches='tight')
    print("Window multiple evolution comparison plot saved as 'window_multiple_robustness.pdf'")


if __name__ == '__main__':
    # Define the core window multiplier array for comparison
    MULTIPLE_LIST = [0.2, 0.5, 1.0, 2.0, 4.0]
    
    # Lock the target damage interface
    TARGET_JOINT = 0
    
    # Define parametric resolution array for structural stiffness reduction
    alpha_list = np.linspace(1.0, 0.0, 21) 
    
    # Execute the core robustness simulation loop
    res_stiffness = compute_window_multiple_robustness(MULTIPLE_LIST, alpha_list, TARGET_JOINT, sim_hours=4)
    
    # Generate and export publication-ready high-fidelity vector graphics
    plot_window_multiple_robustness(MULTIPLE_LIST, alpha_list, res_stiffness, TARGET_JOINT)