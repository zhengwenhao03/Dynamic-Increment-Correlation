import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from response_history import get_response

def plot_pcc_velocity_evolution():
    # =========================================================================
    # 1. Define Parameter Range and Operational Velocities
    # =========================================================================
    # Generate log-scale stiffness array k (from 10^2 to 10^9 N/m^2)
    k_values = np.logspace(4, 8, 100)  

    # Select representative vehicle velocities (km/h) for sensitivity analysis
    v_kmh_list = [20, 60, 100, 140]   
    
    # Define distinct colors and line styles for each velocity curve
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']

    # =========================================================================
    # 2. Core Loop: Compute PCC Evolution for Each Velocity Curve
    # =========================================================================
    plt.figure(figsize=(7.5, 5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    for i, v_kmh in enumerate(v_kmh_list):
        v_ms = v_kmh / 3.6  # Convert velocity from km/h to m/s for the model
        pcc_dyn_list = []
        
        print(f"Calculating PCC evolution curve for velocity v = {v_kmh} km/h...")
        for k in k_values:
            # Call response function with varying stiffness (k) and current velocity (v)
            t, (_, _, strain1_dyn, strain2_dyn, _, _) = get_response(v=v_ms, k=k)
            
            # Calculate the Pearson Correlation Coefficient (PCC) for dynamic increments
            pcc_dyn, _ = pearsonr(strain1_dyn, strain2_dyn)
            pcc_dyn_list.append(pcc_dyn)
            
        # Plot the PCC evolution curve for the current velocity condition
        plt.semilogx(k_values, pcc_dyn_list, color=colors[i], linestyle=linestyles[i], 
                     linewidth=2.5, label=f'$v$ = {v_kmh} km/h')

    # =========================================================================
    # 3. Post-processing and Figure Export Settings
    # =========================================================================
    plt.xlabel('Transverse Connection Stiffness $k$ (N/m$^2$) - Log Scale', fontsize=12, weight='bold')
    plt.ylabel('Pearson Correlation Coefficient', fontsize=12, weight='bold')

    plt.xlim(10**4, 10**8)
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()

    plt.savefig('pcc_velocity.pdf', dpi=600)
    print("Evolution plot successfully generated and saved as 'pcc_velocity.pdf'.")
    plt.show()

if __name__ == '__main__':
    plot_pcc_velocity_evolution()