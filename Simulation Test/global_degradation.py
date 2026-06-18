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

def compute_global_degradation_indicators(alpha_list, sim_hours=6):
    """
    Executes parametric batch simulations for global uniform joint degradation
    and extracts health monitoring indicators for all connection joints.
    
    Args:
        alpha_list (iterable): Array of stiffness reduction factors (1.0 down to 0.0).
        sim_hours (float): Duration of traffic flow simulation per case (hours).
        
    Returns:
        tuple: (dyn_indicators_all, tot_indicators_all) dicts containing historical 
               converged index paths for all joints.
    """
    print(f"Initializing global degradation batch simulation across {len(alpha_list)} cases...")
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6  
    
    num_joints = MM - 1
    
    # Initialize storage pools for tracking all adjacent joint interfaces
    dyn_indicators_all = {j: [] for j in range(num_joints)}
    tot_indicators_all = {j: [] for j in range(num_joints)}
    
    for alpha in tqdm(alpha_list, desc="Global stiffness scanning"):
        current_k_hinge = k_hinge_baseline * (1 - alpha)
        
        # Enforce seed synchronization to eliminate stochastic traffic variations
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        
        strain_history, _, freq_Hz = simulate_bridge_strain(
            HOUR=sim_hours, Fs=Fs, DT=20,
            LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
            E=E, I=I, k_hinge=current_k_hinge, y_sensor=y_sensor,
            add_noise=False, SNR=None
        )
        
        monitor = RealTimeBridgeMonitor(
            bridge_freq_hz=freq_Hz, 
            num_channels=strain_history.shape[1], 
            Fs=Fs, ma_threshold=2, snr_threshold=5, indicator_smooth_size=2000
        )
        
        for sample in strain_history:
            monitor.process_sample(sample)
            
        # Distribute and log final converged indicators for each channel pair
        if len(monitor.smoothed_indicators) > 0:
            for j in range(num_joints):
                dyn_indicators_all[j].append(monitor.smoothed_indicators[-1][j])
                tot_indicators_all[j].append(monitor.smoothed_indicators_total[-1][j])
        else:
            for j in range(num_joints):
                dyn_indicators_all[j].append(np.nan)
                tot_indicators_all[j].append(np.nan)
                
    print("✅ Global degradation batch computation completed successfully.")
    return dyn_indicators_all, tot_indicators_all


def plot_indicator_comparison(alpha_list, dyn_indicators_all, tot_indicators_all, target_joint):
    """
    Plots sensitivity trajectories comparing the traditional total strain indicator 
    against the proposed dynamic increment indicator for a target joint.
    """
    print(f"Rendering indicator sensitivity comparison for Joint {target_joint+1}...")
    plt.figure(figsize=(6, 4))
    
    dyn_data = dyn_indicators_all[target_joint]
    tot_data = tot_indicators_all[target_joint]
    
    plt.plot(
        alpha_list, tot_data, 
        color='gray', linewidth=2.5, linestyle='--', marker='s', markersize=6,
        label='Total Strain Indicator'
    )
    plt.plot(
        alpha_list, dyn_data, 
        color='#d62728', linewidth=3, linestyle='-', marker='o', markersize=7,
        label='Dynamic Increment Indicator'
    )
    
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel(r'Global Stiffness Reduction Factor ($\alpha$)', fontsize=12, fontweight='bold')
    plt.ylabel('Converged Indicator Value', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('global_compare.pdf', dpi=600, bbox_inches='tight')
    print("Sensitivity comparison plot saved as 'global_compare.pdf'.")
    # plt.show()


def plot_multi_joint_degradation(alpha_list, dyn_indicators_all, target_joints_list):
    """
    Plots the synchronous co-evolution trajectories of the proposed dynamic indicators 
    across multiple selected joints to confirm spatial robustness under uniform degradation.
    """
    print("Rendering multi-joint co-evolution trajectory plot...")
    plt.figure(figsize=(6, 4))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(target_joints_list)))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    for idx, j in enumerate(target_joints_list):
        plt.plot(
            alpha_list, dyn_indicators_all[j], 
            color=colors[idx], linewidth=2, linestyle='-', 
            marker=markers[idx % len(markers)], markersize=5,
            label=f'Joint {j+1} (Girder {j+1} & {j+2})'
        )
        
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel(r'Global Stiffness Reduction Factor ($\alpha$)', fontsize=13, fontweight='bold')
    plt.ylabel('Proposed Indicator Value', fontsize=13, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', ncol=1, fontsize=10, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('global_multi.pdf', dpi=600, bbox_inches='tight')
    print("Multi-joint evolution plot saved as 'global_multi.pdf'.")
    # plt.show()


if __name__ == '__main__':
    # 1. Define parametric resolution array (Stiffness reduction factor alpha)
    alpha_list = np.linspace(1.0, 0.0, 21)
    
    # 2. Execute core degradation evaluation 
    # Tip: For fast local validation, you can reduce sim_hours from 6 to 0.5
    dyn_all, tot_all = compute_global_degradation_indicators(alpha_list, sim_hours=6)
    
    # 3. Generate verification graphics for publication
    # Figure 1: Performance gap check on Joint 4 (Interface between Girder 4 & 5)
    plot_indicator_comparison(
        alpha_list=alpha_list, 
        dyn_indicators_all=dyn_all, 
        tot_indicators_all=tot_all, 
        target_joint=3
    )
    
    # Figure 2: Co-evolution pattern across all 7 cross-sectional joint interfaces
    plot_multi_joint_degradation(
        alpha_list=alpha_list, 
        dyn_indicators_all=dyn_all, 
        target_joints_list=[0, 1, 2, 3, 4, 5, 6]
    )