import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from response_history import get_response

def plot_pcc_stiffness_evolution():
    # =========================================================================
    # 1. Define Parameter Range (Log-scale stiffness from 10^4 to 10^8 N/m^2)
    # =========================================================================
    k_values = np.logspace(4, 8, 100)
    pcc_total_list, pcc_dyn_list = [], []

    # =========================================================================
    # 2. Core Loop: Compute Strain Responses and PCC for Each Stiffness Value
    # =========================================================================
    print("Calculating PCC evolution across varying joint stiffness values...")
    for k in k_values:
        # Fetch responses using the unified structural parameters from response_history
        t, (strain1_qs, strain2_qs, strain1_dyn, strain2_dyn, strain1, strain2) = get_response(k=k)
            
        # Calculate Pearson Correlation Coefficients (PCC)
        pcc_tot, _ = pearsonr(strain1, strain2)
        pcc_dyn, _ = pearsonr(strain1_dyn, strain2_dyn)
        
        pcc_total_list.append(pcc_tot)
        pcc_dyn_list.append(pcc_dyn)
    print("Computation completed successfully.")

    # =========================================================================
    # 3. Plotting Configuration (Log-scale X-axis)
    # =========================================================================
    plt.figure(figsize=(7.5, 5))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Plot the Dynamic Increment Correlation curve 
    # Set high zorder to ensure it sits on top of the background color blocks
    plt.semilogx(k_values, pcc_dyn_list, '#d62728', linewidth=2.5, linestyle='-', 
                 label='Dynamic Increment Correlation', zorder=3)
    plt.semilogx(k_values, pcc_total_list, '#1f77b4', linewidth=2.5, linestyle='--', 
                 label='Total Strain Correlation', zorder=3)

    plt.xlabel('Transverse Connection Stiffness $k$ (N/m$^2$) - Log Scale', fontsize=12, weight='bold')
    plt.ylabel('Pearson Correlation Coefficient', fontsize=12, weight='bold')

    # =========================================================================
    # 4. Structural Health Regions & Background Shading
    # =========================================================================
    # Boundaries correspond to the stiffness reduction coefficient (alpha) defined in the paper
    health_regions = [
        # 1. Complete Failure: alpha < 0.01 (k < 5e4)
        (1e4, 5e4, '#b30000', 'Complete\nFailure'),       
        
        # 2. Severe Damage: alpha between 0.01 and 0.20 (5e4 ~ 1e6)
        (5e4, 1e6, '#e34a33', 'Severe\nDamage'),          
        
        # 3. Moderate Damage: alpha between 0.20 and 0.50 (1e6 ~ 2.5e6)
        (1e6, 2.5e6, '#fc8d59', 'Moderate\nDamage'),        
        
        # 4. Slight Degradation: alpha between 0.50 and 0.80 (2.5e6 ~ 4e6)
        (2.5e6, 4e6, '#fdcc8a', 'Slight\nDegradation'),     
        
        # 5. Healthy or Intact: alpha between 0.80 and 3.0 (4e6 ~ 1.5e7)
        (4e6, 1.5e7, '#74c476', 'Healthy or\nIntact'),      
        
        # 6. Nearly Rigid: alpha > 3.0 (k > 1.5e7)
        (1.5e7, 1e8, '#c6dbef', 'Nearly\nRigid')            
    ]

    for xmin, xmax, color, label in health_regions:
        # Fill background blocks with alpha=0.15 for transparency
        plt.axvspan(xmin, xmax, color=color, alpha=0.15, zorder=0)
        
        # Calculate text placement (geometric mean for log scale representation)
        text_x = np.sqrt(xmin * xmax) 
        
        # Add labels rotated 90 degrees to avoid horizontal crowding
        # Centered at y=0.4 to prevent overlap with the curve lines
        plt.text(text_x, 0.4, label, rotation=90, 
                 va='center', ha='center', fontsize=10.5, 
                 color='#4d4d4d', weight='bold', alpha=0.8, zorder=1)

    # =========================================================================
    # 5. Baseline Health Reference Indicator (K_0 = 5e6 N/m^2)
    # =========================================================================
    k0_baseline = 5e6

    # Draw vertical reference line
    plt.axvline(x=k0_baseline, color='#4d4d4d', linestyle='--', linewidth=1.5, alpha=0.8, zorder=2)

    # Add text annotation and directional arrow at an upper clear zone (y=0.90)
    plt.annotate(
        r'Baseline Health ($K_0$)', 
        xy=(k0_baseline, 0.90),            
        xytext=(2e6, 0.90), 
        arrowprops=dict(
            facecolor='#4d4d4d', edgecolor='#4d4d4d', 
            arrowstyle='->', lw=1.5
        ),
        fontsize=11, color='black', weight='bold', 
        va='center', ha='right', zorder=4
    )

    # =========================================================================
    # 6. Post-processing & Figure Export
    # =========================================================================
    plt.xlim(10**4, 10**8)
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", linestyle=':', alpha=0.6, zorder=0)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.tight_layout()

    plt.savefig('pcc_stiffness.pdf', dpi=600, bbox_inches='tight')
    print("Evolution plot successfully generated and saved as 'pcc_stiffness.pdf'.")
    plt.show()

if __name__ == '__main__':
    plot_pcc_stiffness_evolution()