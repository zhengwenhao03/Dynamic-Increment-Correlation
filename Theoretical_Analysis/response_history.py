import numpy as np
import matplotlib.pyplot as plt

def get_response(
    L=25.0,          # Span length (m)
    d=2500,          # Material density (kg/m^3)
    A=0.9220,        # Cross-sectional area (m^2)
    E=3.25e10,       # Elastic modulus of concrete (N/m^2)
    I=0.4361,        # Moment of inertia for a single girder (m^4)
    k=5e6,           # Equivalent shear stiffness of the hinge joint (N/m^2)
    y=1.25,          # Distance from neutral axis to beam bottom (m)
    P=10e4,          # Single-axle heavy vehicle load (N), approx. 10 tons
    v=20,            # Vehicle velocity (m/s)
    fs=1000          # Sampling frequency (Hz)
    ):
    """
    Calculates the mid-span strain response of the analytical dual-beam model 
    during the vehicle crossing period (forced vibration phase).
    
    This function utilizes the mathematical properties of the analytical solution 
    to explicitly decouple the quasi-static and dynamic increment components.

    Returns:
    t : numpy array
        Time vector.
    tuple : 6 numpy arrays
        (strain1_qs, strain2_qs, strain1_dyn, strain2_dyn, strain1_tot, strain2_tot)
        All strain values are converted to micro-strain (με).
    """
    # ================= 1. Structural & Load Parameters =================
    m = d * A
    EI = E * I

    # ================= 2. Time Window Definition =================
    t_end = L / v     # The exact moment the vehicle leaves the bridge
    t = np.arange(0, t_end, 1/fs)
    
    # Initialize output arrays
    strain1_qs, strain2_qs = np.zeros_like(t), np.zeros_like(t)
    strain1_dyn, strain2_dyn = np.zeros_like(t), np.zeros_like(t)

    # Excitation amplitude coefficient
    F0 = P / (m * L)

    # ================= 3. Modal Superposition Computation =================
    # Calculate the first 100 odd modes (n=1, 3, 5...). 
    # Even modes have zero mode shape values at the mid-span (x=L/2).
    n_modes = 100
    
    for n in range(1, n_modes + 1, 2):
        # -- A. Natural Frequencies --
        # Symmetric mode (governed solely by the beam's flexural rigidity)
        omega_sn = np.sqrt((EI / m) * (n * np.pi / L)**4)
        # Antisymmetric mode (governed by flexural rigidity and joint shear stiffness)
        omega_an = np.sqrt((EI / m) * (n * np.pi / L)**4 + 2 * k / m)

        # -- B. Moving Load Driving Frequency --
        Omega_n = n * np.pi * v / L

        # -- C. Absolute Mathematical Decoupling of the Analytical Solution --
        # (1) Symmetric Components
        coeff_sn = F0 / (omega_sn**2 - Omega_n**2)
        q_sn_qs = coeff_sn * np.sin(Omega_n * t)
        q_sn_dyn = -coeff_sn * (Omega_n / omega_sn) * np.sin(omega_sn * t)

        # (2) Antisymmetric Components
        coeff_an = F0 / (omega_an**2 - Omega_n**2)
        q_an_qs = coeff_an * np.sin(Omega_n * t)
        q_an_dyn = -coeff_an * (Omega_n / omega_an) * np.sin(omega_an * t)

        # -- D. Coordinate Transformation: Generalized to Physical (Beam 1 & 2) --
        q1_qs = q_sn_qs + q_an_qs
        q2_qs = q_sn_qs - q_an_qs

        q1_dyn = q_sn_dyn + q_an_dyn
        q2_dyn = q_sn_dyn - q_an_dyn

        # -- E. Strain Conversion Factor --
        # Deflection w(x) = sum(q_n * sin(n*pi*x/L))
        # Curvature w''(x) = sum(q_n * -(n*pi/L)^2 * sin(n*pi*x/L))
        # Strain = -y * w''(L/2)
        curvature_factor = -((n * np.pi / L)**2) * np.sin(n * np.pi / 2)
        strain_factor = -y * curvature_factor

        # Accumulate strain contributions from each mode
        strain1_qs += q1_qs * strain_factor
        strain2_qs += q2_qs * strain_factor

        strain1_dyn += q1_dyn * strain_factor
        strain2_dyn += q2_dyn * strain_factor

    # ================= 4. Total Strain Reconstruction & Output =================
    strain1 = strain1_qs + strain1_dyn
    strain2 = strain2_qs + strain2_dyn

    # Convert to micro-strain (με) to prevent float precision loss and aid plotting
    to_mu = 1e6
    
    return t, (strain1_qs * to_mu, strain2_qs * to_mu, 
               strain1_dyn * to_mu, strain2_dyn * to_mu, 
               strain1 * to_mu, strain2 * to_mu)

if __name__ == '__main__':
    # ================= 1. Get Responses =================
    t, (s1_qs, s2_qs, s1_dyn, s2_dyn, s1_tot, s2_tot) = get_response(k=2e6, P=2e4)

    # ================= 2. Plot Configuration =================
    plt.figure(figsize=(7.5, 6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    # Subplot 1: Total Strain vs. Quasi-Static Baseline
    plt.subplot(2, 1, 1)
    plt.plot(t, s1_qs, color='#1f77b4', linewidth=2.5, linestyle='--', label='Quasi-static Strain (Beam 1)')
    plt.plot(t, s2_qs, color='#d62728', linewidth=2.5, linestyle='--', label='Quasi-static Strain (Beam 2)')
    plt.plot(t, s1_tot, color='#1f77b4', linewidth=2.5, label='Total Strain (Beam 1)')
    plt.plot(t, s2_tot, color='#d62728', linewidth=2.5, label='Total Strain (Beam 2)')

    plt.ylabel(r'Mid-span Strain ($\mu\epsilon$)', fontsize=12, weight='bold')
    plt.legend(loc='upper right', fontsize=11, ncol=1) 
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.ylim([-0.4, 10.5])

    # Subplot 2: Dynamic Increment
    plt.subplot(2, 1, 2)
    plt.plot(t, s1_dyn, color='#1f77b4', linewidth=2.5, label='Dynamic Increment (Beam 1)')
    plt.plot(t, s2_dyn, color='#d62728', linewidth=2.5, linestyle='--', label='Dynamic Increment (Beam 2)')
    plt.xlabel('Time (s)', fontsize=12, weight='bold')
    plt.ylabel(r'Mid-span Strain ($\mu\epsilon$)', fontsize=12, weight='bold')
    plt.legend(loc='upper right', fontsize=11)
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.ylim([-0.45, 0.7])

    plt.tight_layout()
    plt.savefig('strain_response.pdf', dpi=600, bbox_inches='tight')
    print("Plot saved as 'strain_response.pdf'")
    plt.show()