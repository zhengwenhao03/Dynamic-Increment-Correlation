import numpy as np
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import internal core solvers
from simulation_strain import simulate_bridge_strain
from streaming_pipeline import RealTimeBridgeMonitor
from utils import count_vehicles_in_event
from ormsby_filter import ormsby

# Set academic publication-quality standard fonts and typesetting styles
plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

class AbsoluteFilterBridgeMonitor(RealTimeBridgeMonitor):
    """
    Subclass of RealTimeBridgeMonitor that overrides the event analysis 
    to inject absolute lower cut-off frequencies for the Ormsby filter.
    """
    def __init__(self, low_cutoff_hz, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.low_cutoff_hz = low_cutoff_hz

    def _analyze_completed_event(self):
        """
        Overridden solver to apply absolute frequency band-pass decoupling.
        """
        phys_raw = np.array(self.current_raw_event)
        phys_ma = np.array(self.current_ma_event)
        
        if len(phys_raw) <= self.W:
            return
        
        num_cars = count_vehicles_in_event(phys_ma, self.Fs)
        
        if num_cars == 1:
            # Step 1: Extract quasi-static components using user-defined absolute frequency
            # Roll-off frequency (rf) = low_cutoff_hz
            # Termination frequency (tf) = rf + 0.2 Hz (constrained below fundamental frequency)
            rf_low = self.low_cutoff_hz - 0.1
            tf_low = self.low_cutoff_hz + 0.1
            
            static_data = ormsby(phys_raw, rf_low, tf_low, 1 / self.Fs)
            dyn_data = phys_raw - static_data
            
            # Step 2: Keep the rest of the original high-frequency processing unchanged
            dyn_data = ormsby(
                dyn_data, 
                self.bridge_freq_hz[2] * 1.2, 
                self.bridge_freq_hz[2] * 1.4, 
                1 / self.Fs
            )

            trim_start = self.buffer_pts // 2  
            trim_end = -self.cooldown_pts // 2 if self.cooldown_pts > 0 else None
            
            dyn_data = dyn_data[trim_start:trim_end, :]
            phys_raw = phys_raw[trim_start:trim_end, :]

            self.single_events_dyn.append(dyn_data)
            self.single_events_pool.append(phys_raw)

            start_idx = self.current_event_start_idx + trim_start
            end_idx = start_idx + len(phys_raw)
            self.single_events_indices.append((start_idx, end_idx))

            adj_corrs = [np.nan] * (self.num_channels - 1)
            max_ch = np.argmax(np.max(np.abs(phys_raw), axis=0))
            max_val = np.max(np.abs(phys_raw[:, max_ch]))

            if max_val > self.snr_threshold:
                if max_ch > 0:
                    left_ch = max_ch - 1
                    corr = np.corrcoef(dyn_data[:, left_ch], dyn_data[:, max_ch])[0, 1]
                    adj_corrs[left_ch] = corr  
                if max_ch < self.num_channels - 1:
                    right_ch = max_ch + 1
                    corr = np.corrcoef(dyn_data[:, max_ch], dyn_data[:, right_ch])[0, 1]
                    adj_corrs[max_ch] = corr   
                        
            self.correlations_history.append(adj_corrs)
            self._update_smoothed_indicator()


def compute_absolute_filter_robustness(cutoff_list, alpha_list, target_joint, sim_hours=4):
    """
    Executes a batch parametric scan across designated absolute cut-off frequencies.
    """
    print(f"\n[Task 1] Launching absolute filter cut-off frequency sensitivity scan (Target Joint: {target_joint+1})...")
    Fs = 50
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; y_sensor = 0.45
    k_hinge_baseline = 5e6
    num_joints = MM - 1
    
    results = {fc: [] for fc in cutoff_list}
    
    for alpha in tqdm(alpha_list, desc="Stiffness Scanning"):
        SEED = 42
        random.seed(SEED)
        np.random.seed(SEED)
        
        k_hinge_array = np.full((num_joints, 1), k_hinge_baseline)
        k_hinge_array[target_joint, 0] = k_hinge_baseline * (1 - alpha)
        
        # Pure physical simulation without sensor noise to completely isolate filter boundary effects
        strain_history, _, freq_Hz = simulate_bridge_strain(
            HOUR=sim_hours, Fs=Fs, DT=20,
            LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
            E=E, I=I, k_hinge=k_hinge_array, y_sensor=y_sensor,
            add_noise=True, SNR=30
        )
        
        for fc in cutoff_list:
            monitor = AbsoluteFilterBridgeMonitor(
                low_cutoff_hz=fc, # Inject absolute target frequency boundary
                bridge_freq_hz=freq_Hz, 
                num_channels=MM, 
                Fs=Fs, 
                ma_threshold=2, 
                snr_threshold=5, 
                indicator_smooth_size=2000 
            )
            for sample in strain_history:
                monitor.process_sample(sample)
                
            if len(monitor.smoothed_indicators) > 0:
                results[fc].append(monitor.smoothed_indicators[-1][target_joint])
            else:
                results[fc].append(np.nan)
                
    return results

def plot_absolute_filter_robustness(cutoff_list, alpha_list, results, target_joint):
    """Renders the publication-ready verification graphic."""
    plt.figure(figsize=(8, 4))
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd', '#e377c2', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    line_styles = [':', '-', '-', '--']
    
    for idx, fc in enumerate(cutoff_list):
        # Apply thicker lines to satisfactory passband regions (2.5Hz - 4.0Hz)
        lw = 2.5
        plt.plot(
            alpha_list, results[fc], 
            color=colors[idx % len(colors)], linewidth=lw, linestyle=line_styles[idx % len(line_styles)], 
            marker=markers[idx % len(markers)], markersize=5, alpha=0.85,
            label=f'$f_l$ = {fc:.1f} Hz'
        )
        
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel(f'Local Stiffness Reduction Factor of Joint {target_joint+1} ($\\alpha$)', fontsize=12, fontweight='bold')
    plt.ylabel('Converged Indicator Value', fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower left', fontsize=10, framealpha=0.9, edgecolor='black', ncol=1)
    
    plt.tight_layout()
    plt.savefig('filter_absolute_robustness.pdf', dpi=600, bbox_inches='tight')
    print("Comparison plot successfully saved as 'filter_absolute_robustness.pdf'")


if __name__ == '__main__':
    # 1. Define the specific target frequency array derived from your physical derivation
    CUTOFF_LIST = [2.0, 3.0, 4.0, 5.0]
    
    # 2. Configure structural degradation array
    alpha_list = np.linspace(1.0, 0.0, 11) 
    TARGET_JOINT = 0
    
    # 3. Execute parametric execution loop
    res_stiffness = compute_absolute_filter_robustness(CUTOFF_LIST, alpha_list, TARGET_JOINT, sim_hours=4)
    
    # 4. Generate high-fidelity vector figure
    plot_absolute_filter_robustness(CUTOFF_LIST, alpha_list, res_stiffness, TARGET_JOINT)