import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
import scipy.fft as fft

def get_stiffness_matrix(m, n, E, I, LL, Spring_Stiffness):
    """Assembles the global structural stiffness matrix for multi-girder system."""
    L = LL / (n - 1)
    c12 = 12 * E * I / L**3
    c6  = 6 * E * I / L**2
    c4  = 4 * E * I / L
    c2  = 2 * E * I / L
    c24 = 24 * E * I / L**3
    c8  = 8 * E * I / L
    K = np.zeros((2 * m * n, 2 * m * n))
    
    for j in range(n):
        base_j = 2 * m * j
        for i in range(m):
            base = base_j + 2 * i
            if j == 0:
                K[base, base] += c12; K[base+1, base] += c6; K[base+2*m, base] -= c12; K[base+1+2*m, base] += c6
                K[base, base+1] += c6; K[base+1, base+1] += c4; K[base+2*m, base+1] -= c6; K[base+1+2*m, base+1] += c2
            elif j == n - 1:
                K[base-2*m, base] -= c12; K[base-2*m+1, base] -= c6; K[base, base] += c12; K[base+1, base] -= c6
                K[base-2*m, base+1] += c6; K[base-2*m+1, base+1] += c2; K[base, base+1] -= c6; K[base+1, base+1] += c4
            else:
                K[base-2*m, base] -= c12; K[base-2*m+1, base] -= c6; K[base, base] += c24; K[base+2*m, base] -= c12; K[base+2*m+1, base] += c6
                K[base-2*m, base+1] += c6; K[base-2*m+1, base+1] += c2; K[base+1, base+1] += c8; K[base+2*m, base+1] -= c6; K[base+2*m+1, base+1] += c2
            
            if i == 0:
                K[base, base] += Spring_Stiffness[0, j]; K[base+2, base] -= Spring_Stiffness[0, j]
            elif i == m - 1:
                K[base-2, base] -= Spring_Stiffness[m-2, j]; K[base, base] += Spring_Stiffness[m-2, j]
            else:
                K[base-2, base] -= Spring_Stiffness[i-1, j]; K[base, base] += Spring_Stiffness[i-1, j] + Spring_Stiffness[i, j]; K[base+2, base] -= Spring_Stiffness[i, j]
    return K

def get_vehicle_weight(VW_mu1, VW_sigma1, VW_mu2, VW_sigma2, VW_mu3, VW_sigma3, P1, P2, num):
    """Generates stochastic vehicle weights using a mixed distribution model."""
    weights = np.zeros(num)
    rand_probs = np.random.rand(num)
    idx1 = rand_probs < P1
    idx2 = (rand_probs >= P1) & (rand_probs < P1 + P2)
    idx3 = rand_probs >= P1 + P2
    weights[idx1] = np.random.normal(VW_mu1, VW_sigma1, np.sum(idx1))
    weights[idx2] = np.random.normal(VW_mu2, VW_sigma2, np.sum(idx2))
    weights[idx3] = np.random.gumbel(VW_mu3, VW_sigma3, np.sum(idx3))
    return np.maximum(weights, 0.5)

def get_traffic_load(hour, LL, MM, NN, fs, DT, VW_mu1, VW_sigma1, VW_mu2, VW_sigma2, VW_mu3, VW_sigma3, P1, P2, VV_mu, VV_sigma):
    """Generates space-time coupled stochastic traffic load matrices."""
    T = hour * 3600; dt = 1.0 / fs; L = LL / (NN - 1); total_pts = int(T * fs)
    padding = int((LL / 1.0) * fs) + 500
    p = np.zeros((MM * (NN - 2), total_pts + padding))
    vehicle_records = []
    
    for k in range(MM):
        prob = 1.0 / (DT * fs * MM)
        vehicle_or_not = np.random.binomial(1, prob, total_pts)
        A = np.where(vehicle_or_not == 1)[0]
        vehicle_num = len(A)
        if vehicle_num == 0: continue
        
        weights_t = get_vehicle_weight(VW_mu1, VW_sigma1, VW_mu2, VW_sigma2, VW_mu3, VW_sigma3, P1, P2, vehicle_num)
        weights_N = 1000 * 9.8 * weights_t
        speeds = np.maximum(np.random.normal(VV_mu, VV_sigma, vehicle_num) / 3.6, 1.0)
        
        for i in range(vehicle_num):
            w = weights_N[i]; v = speeds[i]; entry_idx = A[i]
            vehicle_records.append({'girder': k + 1, 'time_s': entry_idx / fs, 'weight_t': weights_t[i], 'speed_kmh': speeds[i] * 3.6})
            for j in range(1, NN - 1):
                row = (j - 1) * MM + k
                s1 = int(np.ceil((j - 1) * L / v * fs)); e1 = int(np.floor(j * L / v * fs))
                if e1 >= s1: p[row, entry_idx+s1 : entry_idx+e1+1] += w * (v * (np.arange(s1, e1 + 1) * dt) - (j - 1) * L) / L
                s2 = int(np.ceil(j * L / v * fs)); e2 = int(np.floor((j + 1) * L / v * fs))
                if e2 >= s2: p[row, entry_idx+s2 : entry_idx+e2+1] += w * ((j + 1) * L - v * (np.arange(s2, e2 + 1) * dt)) / L
                
    return p[:, :total_pts], sorted(vehicle_records, key=lambda x: x['time_s'])

def apply_boundary_constraints(matrix, del_indices):
    """Applies simply-supported boundary conditions via the striking-out method."""
    return np.delete(np.delete(matrix, del_indices, axis=0), del_indices, axis=1)

def simulate_bridge_strain(
    HOUR=24, Fs=50, LL=20, MM=8, NN=21, Density=2500, Area=0.51, E=3.45e10, I=0.05, k_hinge=5e6, y_sensor=0.45,
    DT=20, vehicleWeight_mu1=2.0, vehicleWeight_sigma1=0.5, vehicleWeight_mu2=15, vehicleWeight_sigma2=4, 
    vehicleWeight_mu3=42, vehicleWeight_sigma3=5, P1=0.65, P2=0.2, vechileSpeed_mu=80, vehicleSpeed_sigma=10,
    add_noise=False, SNR=30
):
    """Main solver executing high-efficiency strain modal superposition simulation."""
    Cols = int(HOUR * 3600 * Fs); L_elem = LL / (NN - 1); mass = (Density * LL * Area) / (NN - 1); J = mass * (L_elem)**2 / 12
    Spring_Stiffness = np.atleast_2d(k_hinge).repeat(NN, axis=1) if isinstance(k_hinge, np.ndarray) else np.full((MM - 1, NN), k_hinge)
    Spring_Stiffness[:, 0] /= 2; Spring_Stiffness[:, -1] /= 2
    
    K = get_stiffness_matrix(MM, NN, E, I, LL, Spring_Stiffness)
    M_diag = np.zeros(2 * MM * NN)
    M_diag[2*MM : 2*MM*(NN-1) : 2] = mass; M_diag[2*MM+1 : 2*MM*(NN-1) : 2] = J
    M_diag[0:2*MM:2] = mass/2; M_diag[1:2*MM:2] = J/2; M_diag[2*MM*(NN-1)::2] = mass/2; M_diag[2*MM*(NN-1)+1::2] = J/2
    M = np.diag(M_diag)
    
    del_idx = np.concatenate([np.arange(0, 2 * MM, 2), np.arange(2 * MM * (NN - 1), 2 * MM * NN, 2)])
    K = apply_boundary_constraints(K, del_idx); M = apply_boundary_constraints(M, del_idx)
    
    w2, _ = la.eigh(K, M); omega_eig = np.sqrt(np.abs(w2))
    a0 = 2 * 0.03 / (omega_eig[0] + omega_eig[2]) * omega_eig[0] * omega_eig[2]
    a1 = 2 * 0.03 / (omega_eig[0] + omega_eig[2])
    C = a0 * M + a1 * K
    
    P_traffic, vehicle_records = get_traffic_load(
        HOUR, LL, MM, NN, Fs, DT, 
        vehicleWeight_mu1, vehicleWeight_sigma1, 
        vehicleWeight_mu2, vehicleWeight_sigma2, 
        vehicleWeight_mu3, vehicleWeight_sigma3, 
        P1, P2, vechileSpeed_mu, vehicleSpeed_sigma)
    P_fft = fft.fft(P_traffic, n=Cols, axis=1)
    P_all = np.zeros((2 * MM * (NN - 1), Cols), dtype=complex)
    P_all[MM : -MM : 2, :] = P_fft
    
    N_half = Cols // 2; omega = 2 * np.pi * Fs * np.arange(0, N_half + 1) / Cols
    num_modes = min(40, M.shape[0])
    W2, Phi = spla.eigsh(K, M=M, k=num_modes, which='SM')
    omega_n = np.sqrt(np.abs(W2))
    
    for ii in range(num_modes): Phi[:, ii] /= np.sqrt(Phi[:, ii].T @ M @ Phi[:, ii])
    C_modal = np.diag(Phi.T @ C @ Phi)
    Q_modal = (Phi.T @ P_all[:, :N_half+1]) / ((omega_n[:, None]**2 - omega[None, :]**2) + 1j * (C_modal[:, None] * omega[None, :]))
    
    j_mid = (NN - 1) // 2; j_prev = j_mid - 1
    Phi_strain_mid = np.zeros((MM, num_modes))
    for k in range(MM):
        wp = 2 * (MM * j_prev + k) - MM; wm = 2 * (MM * j_mid + k) - MM
        Curvature_mode = (6/L_elem**2)*Phi[wp,:] + (2/L_elem)*Phi[wp+1,:] - (6/L_elem**2)*Phi[wm,:] + (4/L_elem)*Phi[wm+1,:]
        Phi_strain_mid[k, :] = np.abs(y_sensor) * -Curvature_mode
        
    Strain_mid_freq = np.zeros((MM, Cols), dtype=complex)
    Strain_mid_freq[:, :N_half+1] = Phi_strain_mid @ Q_modal
    Strain_mid_freq[:, N_half+1:] = np.conj(Strain_mid_freq[:, 1:N_half][:, ::-1])
    strain = fft.ifft(Strain_mid_freq, axis=1).real
    
    if add_noise:
        noise = np.sqrt(np.var(strain, axis=1, keepdims=True) / (10**(SNR / 10))) * np.random.randn(*strain.shape)
        strain += noise
        
    return strain.T * 1e6, vehicle_records, omega_eig / (2 * np.pi)

if __name__ == '__main__':
    print("Solver kernel verification...")
    # Standard check pipeline to verify dimensional consistency
    strain_history, vehicle_records, freq_Hz = simulate_bridge_strain(HOUR=0.1, Fs=50, k_hinge=5e6)
    print(f"Success. Matrix shape output: {strain_history.shape}, Fundamental frequency: {freq_Hz[0]:.2f} Hz")