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

def compute_local_degradation_indicators(alpha_list, damaged_joints_list, sim_hours=6):
    """
    Executes parametric batch simulations for localized multiple joint degradation 
    to extract and track health monitoring indicators across all interfaces.
    
    Args:
        alpha_list (iterable): Array of stiffness reduction factors (1.0 down to 0.0).
        damaged_joints_list (list): Indices of targeted joints undergoing severe localized failure.
        sim_hours (float): Duration of traffic flow simulation per case (hours).
        
    Returns:
        tuple: (dyn_indicators_all, tot_indicators_all) dicts containing historical 
               converged index paths for all joints.
    """
    joints_str = ", ".join([str(j + 1) for j in damaged_joints_list])
    print(f"Initializing localized joint degradation simulation. Targeted Damage Profile: Joint {joints_str}...")
    
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6  
    
    num_joints = MM - 1
    
    # Initialize storage pools for tracking all adjacent joint interfaces
    dyn_indicators_all = {j: [] for j in range(num_joints)}
    tot_indicators_all = {j: [] for j in range(num_joints)}
    
    for alpha in tqdm(alpha_list, desc="Local stiffness scanning"):
        
        # Assemble localized stiffness matrix vector (折减矩阵列向量化构造)
        k_hinge_array = np.full((num_joints, 1), k_hinge_baseline)
        for j in damaged_joints_list:
            k_hinge_array[j, 0] = k_hinge_baseline * (1 - alpha)
        
        # Enforce seed synchronization to eliminate stochastic traffic variations
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        
        strain_history, _, freq_Hz = simulate_bridge_strain(
            HOUR=sim_hours, Fs=Fs, DT=20,
            LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
            E=E, I=I, k_hinge=k_hinge_array, y_sensor=y_sensor, 
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
                
    print("Localized degradation batch computation completed successfully.")
    return dyn_indicators_all, tot_indicators_all


def plot_damaged_joints_comparison(alpha_list, dyn_indicators_all, tot_indicators_all, damaged_joints_list):
    """
    Plots sensitivity trajectories comparing the traditional total strain indicator 
    against the proposed dynamic increment indicator specifically for the damaged joints.
    """
    print("Rendering indicator sensitivity comparison for damaged joint group...")
    plt.figure(figsize=(6, 4))
    
    colors_dyn = ['#d62728', '#ff7f0e']  # Warm red colors for proposed metrics
    colors_tot = ['gray', 'black']       # Desaturated colors for traditional benchmarks
    markers = ['o', 's']
    
    for idx, j in enumerate(damaged_joints_list):
        # Traditional Benchmark Curve
        plt.plot(
            alpha_list, tot_indicators_all[j], 
            color=colors_tot[idx % 2], linewidth=2.5, linestyle='--', 
            marker=markers[idx % 2], markersize=6,
            label=f'Total Strain Indicator (Joint {j+1})'
        )
        # Proposed Dynamic Curve
        plt.plot(
            alpha_list, dyn_indicators_all[j], 
            color=colors_dyn[idx % 2], linewidth=3, linestyle='-', 
            marker=markers[idx % 2], markersize=7,
            label=f'Dynamic Increment Indicator (Joint {j+1})'
        )
    
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.xlabel(r'Local Stiffness Reduction Factor ($\alpha$)', fontsize=12, fontweight='bold')
    plt.ylabel('Converged Indicator Value', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=11, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('local_compare.pdf', dpi=600, bbox_inches='tight')
    print("Damage sensitivity comparison plot saved as 'local_compare.pdf'.")
    # plt.show()


def plot_multi_joint_degradation(alpha_list, dyn_indicators_all, target_joints_list):
    """
    Plots trajectories across all cross-sectional joint interfaces under localized damage.
    This graph effectively proves the SPATIAL ROBUSTNESS and FALSE-ALARM IMMUNITY 
    of the proposed method (healthy joints remain undisturbed near 1.0).
    """
    print("Rendering multi-joint spatial localization and robustness plot...")
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
    plt.xlabel(r'Local Stiffness Reduction Factor ($\alpha$)', fontsize=13, fontweight='bold')
    plt.ylabel('Proposed Indicator Value', fontsize=13, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', ncol=1, fontsize=10, framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('local_spatial_localization.pdf', dpi=600, bbox_inches='tight')
    print("Spatial localization plot saved as 'local_spatial_localization.pdf'.")
    # plt.show()


if __name__ == '__main__':
    # 1. Define parametric resolution array (Stiffness reduction factor alpha)
    alpha_list = np.linspace(1.0, 0.0, 21)
    
    # Define damage scenario: Assume joint 4 (index 3) and joint 5 (index 4) degrade locally
    target_damaged_joints = [3, 4]
    
    # 2. Execute localized multi-joint computation
    dyn_all, tot_all = compute_local_degradation_indicators(
        alpha_list=alpha_list, 
        damaged_joints_list=target_damaged_joints, 
        sim_hours=6  # Can be scaled down to 0.5 hours for fast local verification testing
    )
    
    # 3. Generate verification graphics for publication
    # Figure 1: Comparison plot for the targeted damaged joint subset
    plot_damaged_joints_comparison(
        alpha_list=alpha_list, 
        dyn_indicators_all=dyn_all, 
        tot_indicators_all=tot_all, 
        damaged_joints_list=target_damaged_joints
    )
    
    # Figure 2: Spatial isolation check across all 7 joints (Proving zero cross-talk)
    plot_multi_joint_degradation(
        alpha_list=alpha_list, 
        dyn_indicators_all=dyn_all, 
        target_joints_list=[0, 1, 2, 3, 4, 5, 6]
    )