import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from response_history import get_response

def plot_pcc_weight_evolution():
    # =========================================================================
    # 1. Define Parameter Range and Vehicle Weights
    # =========================================================================
    # Generate log-scale stiffness array k (from 10^2 to 10^9 N/m^2)
    k_values = np.logspace(4, 8, 100)  

    # Select representative vehicle weights (in metric tons, t)
    # Corresponding to light trucks, medium trucks, heavy trucks, and overloaded vehicles
    P_tons_list = [10, 20, 40, 60]   
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = ['-', '--', '-.', ':']

    # =========================================================================
    # 2. Core Loop: Compute PCC Evolution under Varying Vehicle Weights
    # =========================================================================
    plt.figure(figsize=(7.5, 5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    for i, P_tons in enumerate(P_tons_list):
        # Convert weight from metric tons to Newtons (N)
        P_N = P_tons * 1e3 * 9.8
        pcc_dyn_list = []
        
        print(f"Calculating PCC evolution curve for vehicle weight P = {P_tons} t...")
        for k in k_values:
            # Call response function with varying stiffness (k) and current weight (P)
            t, (_, _, strain1_dyn, strain2_dyn, _, _) = get_response(P=P_N, k=k)
            
            # Calculate the Pearson Correlation Coefficient (PCC) for dynamic increments
            pcc_dyn, _ = pearsonr(strain1_dyn, strain2_dyn)
            pcc_dyn_list.append(pcc_dyn)
            
        # Plot the PCC evolution curve for the current weight condition.
        # Note: Because the load amplitude scales the analytical responses linearly, 
        # the normalized PCC curves will overlap perfectly. Distinct line widths (lw)
        # are deliberately implemented here to ensure all curves remain visually distinguishable.
        lw = 4.0 - i * 0.8 
        plt.semilogx(k_values, pcc_dyn_list, color=colors[i], linestyle=linestyles[i], 
                     linewidth=lw, label=f'$P$ = {P_tons} t')

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

    plt.savefig('pcc_weight.pdf', dpi=600)
    print("Evolution plot successfully generated and saved as 'pcc_weight.pdf'.")
    plt.show()

if __name__ == '__main__':
    plot_pcc_weight_evolution()