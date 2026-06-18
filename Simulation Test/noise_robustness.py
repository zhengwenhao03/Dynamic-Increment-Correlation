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

def compute_noise_robustness(alpha_list, snr_levels, target_joint, sim_hours=6):
    """
    Scans and evaluates the tracking trajectories of the proposed localized joint 
    degradation indicators under varying environmental signal-to-noise ratios (SNR).
    
    Args:
        alpha_list (iterable): Array of stiffness reduction factors (1.0 down to 0.0).
        snr_levels (list): List of evaluated SNR values (including None for Clean).
        target_joint (int): Index of the joint interface undergoing damage scanning.
        sim_hours (float): Duration of traffic flow simulation per case (hours).
        
    Returns:
        dict: Hierarchical index maps grouping converged index tracks under each SNR tier.
    """
    print(f"Initializing noise robustness parametric simulation for Joint {target_joint+1}...")
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6  
    num_joints = MM - 1
    
    # Initialize dictionary mapping indicators across different SNR levels
    results_dyn = {snr: [] for snr in snr_levels}
    
    for snr in snr_levels:
        snr_label = f"{snr} dB" if snr is not None else "Clean"
        print(f"\nEvaluating operational environmental noise level: {snr_label}")
        
        for alpha in tqdm(alpha_list, desc=f"Stiffness scanning ({snr_label})"):
            
            # Construct localized multi-joint stiffness topology matrix array
            k_hinge_array = np.full((num_joints, 1), k_hinge_baseline)
            k_hinge_array[target_joint, 0] = k_hinge_baseline * (1 - alpha)
            
            # Enforce seed synchronization to secure identical random traffic configurations
            SEED = 42
            random.seed(SEED)
            np.random.seed(SEED)
            
            add_noise_flag = (snr is not None)
            
            strain_history, _, freq_Hz = simulate_bridge_strain(
                HOUR=sim_hours, Fs=Fs, DT=20,
                LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
                E=E, I=I, k_hinge=k_hinge_array, y_sensor=y_sensor, 
                add_noise=add_noise_flag, SNR=snr
            )
            
            monitor = RealTimeBridgeMonitor(
                bridge_freq_hz=freq_Hz, 
                num_channels=strain_history.shape[1], 
                Fs=Fs, ma_threshold=2, snr_threshold=5, indicator_smooth_size=2000
            )
            
            for sample in strain_history:
                monitor.process_sample(sample)
                
            if len(monitor.smoothed_indicators) > 0:
                results_dyn[snr].append(monitor.smoothed_indicators[-1][target_joint])
            else:
                results_dyn[snr].append(np.nan)
                
    print("Noise robustness batch simulation completed successfully.")
    return results_dyn


def plot_noise_robustness(alpha_list, results_dyn, snr_levels, target_joint):
    """
    Plots concurrent multi-tier trajectories to visually demonstrate the strong 
    immunity and high stability convergence of the proposed indicators under sensor noise.
    """
    print("Rendering noise robustness verification curves...")
    plt.figure(figsize=(8, 4))
    
    # Establish sequential color schemes and marker profiles to manifest gradual increments
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#e377c2', '#8c564b', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    for idx, snr in enumerate(snr_levels):
        label_str = 'Clean Signal (No Noise)' if snr is None else f'Noisy Signal (SNR = {snr} dB)'
        line_style = '-' if snr is None else '--'
        alpha_val = 1.0 if snr is None else 0.8
        
        plt.plot(
            alpha_list, results_dyn[snr], 
            color=colors[idx % len(colors)], linewidth=2.5, linestyle=line_style, 
            marker=markers[idx % len(markers)], markersize=6, alpha=alpha_val,
            label=label_str
        )
        
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel(f'Local Stiffness Reduction Factor of Joint {target_joint+1} ($\\alpha$)', fontsize=12, fontweight='bold')
    plt.ylabel('Proposed Indicator Value', fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('noise_robustness.pdf', dpi=600, bbox_inches='tight')
    print("Noise robustness verification plot successfully saved as 'noise_robustness.pdf'.")
    # plt.show()


if __name__ == '__main__':
    # 1. Define parametric resolution array (Stiffness reduction factor alpha)
    alpha_list = np.linspace(1.0, 0.0, 21)
    
    # Define damage target interface: Joint 4 (index 3)
    TARGET_JOINT = 3
    
    # 2. Establish assessed signal-to-noise ratio (SNR) spectrum tiers
    # Includes: None (Clean), 60dB (Slight), 50dB (Moderate), 40dB (Heavy), 30dB/20dB/10dB (Severe noise interferences)
    SNR_LEVELS = [None, 60, 50, 40, 30, 20, 10]
    
    # 3. Execute core noise-resilient simulation loop
    # Tip: sim_hours can be scaled down locally from 3 to 0.5 to perform quick runtime debugging
    results_dyn = compute_noise_robustness(
        alpha_list=alpha_list, 
        snr_levels=SNR_LEVELS, 
        target_joint=TARGET_JOINT, 
        sim_hours=3
    )
    
    # 4. Generate high-fidelity publication-quality figures
    plot_noise_robustness(
        alpha_list=alpha_list, 
        results_dyn=results_dyn, 
        snr_levels=SNR_LEVELS, 
        target_joint=TARGET_JOINT
    )