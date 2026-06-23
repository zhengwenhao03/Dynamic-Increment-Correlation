import numpy as np
from collections import deque
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'stix'

from utils import count_vehicles_in_event
from ormsby_filter import ormsby

class RealTimeBridgeMonitor:
    def __init__(
            self, 
            bridge_freq_hz, 
            num_channels, 
            Fs=50, 
            ma_threshold=2, 
            ma_length_multiple=2.0,  # Moving average window length as a multiple of the 1st-order modal period
            low_cutoff_multiplier=0.7,  # Ormsby filter low cut-off frequency as a multiple of the 1st-order modal frequency
            snr_threshold=5, 
            indicator_smooth_size=50
        ):
        """
        Edge computing engine for real-time bridge health monitoring. It concurrently 
        computes the Pearson Correlation Coefficient (PCC) indicators using both 
        total dynamic strains and dynamic strain increments for comparative validation.
        """
        self.Fs = Fs
        self.bridge_freq_hz = bridge_freq_hz
        self.ma_threshold = ma_threshold
        self.snr_threshold = snr_threshold  # Threshold for verifying channel loading validity
        self.num_channels = num_channels
        self.low_cutoff_multiplier = low_cutoff_multiplier
        # Core physical parameter: number of samples within a 2-period moving window
        self.W = int((ma_length_multiple / bridge_freq_hz[0]) * Fs)
        
        # O(1) circular buffer for real-time rolling moving average (MA) computation
        self.rolling_window = deque(maxlen=self.W)
        self.current_sum = np.zeros(num_channels)
        
        # Finite State Machine (FSM) state variables
        self.is_active = False
        self.global_sample_count = 0          # Global sample counter acting as the absolute timeline
        self.current_event_start_idx = 0      # Global physical start index of the ongoing event

        self.current_raw_event = []
        self.current_ma_event = []

        # Elastic buffers to counteract boundary effects
        self.buffer_pts = int(1.0 * Fs)      # Retain 1 second of pre-trigger history data
        self.cooldown_pts = int(1.0 * Fs)    # Continue recording for 1 second post-threshold drop

        self.history_buffer = deque(maxlen=self.buffer_pts)
        self.history_ma_buffer = deque(maxlen=self.buffer_pts)
        self.current_cooldown = 0
        
        # Data storage pools for processed events
        self.single_events_pool = []
        self.single_events_indices = []
        self.single_events_dyn = []
        self.single_events_static = []

        # Historical lists for raw correlation coefficients
        self.correlations_history = []         # Proposed indicators (Dynamic increment PCC)
        self.smoothed_indicators = []

        self.correlations_history_total = []    # Benchmark indicators (Total strain PCC)
        self.smoothed_indicators_total = []

        self.indicator_smooth_size = indicator_smooth_size
        
    def process_sample(self, sample):
        """
        Simulates the Interrupt Service Routine (ISR) of an edge gateway, 
        processing exactly one multi-channel sample per time step.
        """
        sample = np.array(sample, dtype=np.float64)

        # Append to historical buffer to ensure seamless stitching for rising edge triggers
        self.history_buffer.append(sample)
        
        # 1. Update rolling moving average (O(1) time complexity)
        if len(self.rolling_window) == self.W:
            oldest = self.rolling_window[0]
            self.current_sum -= oldest
            
        self.rolling_window.append(sample)
        self.current_sum += sample
        
        current_ma = self.current_sum / len(self.rolling_window)
        self.history_ma_buffer.append(current_ma)

        global_ma_val = np.max(np.abs(current_ma))
        
        # 2. Edge-triggered FSM (adapts inherently to physical vehicle entries/exits)
        if global_ma_val > self.ma_threshold:
            if not self.is_active:
                # Rising edge triggered: A vehicle enters the bridge!
                self.is_active = True

                # Prepend pre-trigger historical buffer to recover the complete waveform onset
                self.current_raw_event = list(self.history_buffer)
                self.current_ma_event = list(self.history_ma_buffer)

                # Track absolute global start index of the event
                self.current_event_start_idx = self.global_sample_count - len(self.history_buffer)
                
            else:
                # Vehicle remains on-bridge; continue real-time data logging
                self.current_raw_event.append(sample)
                self.current_ma_event.append(current_ma)

            # Reset cooldown counter as long as active signal levels persist
            self.current_cooldown = self.cooldown_pts
            
        else:
            if self.is_active:
                # Falling edge encountered: Vehicle is leaving the bridge
                self.current_raw_event.append(sample)
                self.current_ma_event.append(current_ma)
                self.current_cooldown -= 1

                # Conclude recording only when trailing cooldown window fully expires
                if self.current_cooldown <= 0:
                    self.is_active = False
                    self._analyze_completed_event()
                
        # Time step incrementation
        self.global_sample_count += 1
                
    def _analyze_completed_event(self):
        """
        Executes closed-loop analysis immediately upon capturing a completed event.
        """
        phys_raw = np.array(self.current_raw_event)
        phys_ma = np.array(self.current_ma_event)
        
        # Filter out transient electromagnetic spikes or high-frequency glitches
        if len(phys_raw) <= self.W:
            return
        
        # 3. Verify single-vehicle crossing condition
        num_cars = count_vehicles_in_event(phys_ma, self.Fs)
        
        if num_cars == 1:
            # 4. Extract dynamic strain increments using cascaded zero-phase Ormsby filtering
            static_data = ormsby(
                phys_raw, 
                self.bridge_freq_hz[0] * (self.low_cutoff_multiplier - 0.1),
                self.bridge_freq_hz[0] * (self.low_cutoff_multiplier + 0.1),
                1 / self.Fs
            )

            dyn_data = phys_raw - static_data
            dyn_data = ormsby(
                dyn_data, 
                self.bridge_freq_hz[2] * 1.2, 
                self.bridge_freq_hz[2] * 1.4, 
                1 / self.Fs
            )

            # Trim trailing edges to eliminate filter boundary distortion effects
            trim_start = self.buffer_pts // 2  
            trim_end = -self.cooldown_pts // 2 if self.cooldown_pts > 0 else None
            
            dyn_data = dyn_data[trim_start:trim_end, :]
            static_data = static_data[trim_start:trim_end, :]
            phys_raw = phys_raw[trim_start:trim_end, :]

            self.single_events_dyn.append(dyn_data)
            self.single_events_static.append(static_data)
            self.single_events_pool.append(phys_raw)

            # Compute absolute global timeline parameters
            start_idx = self.current_event_start_idx + trim_start
            end_idx = start_idx + len(phys_raw)
            self.single_events_indices.append((start_idx, end_idx))

            # 5. Concurrently compute adjacent channel correlations for both methods
            adj_corrs = [np.nan] * (self.num_channels - 1)
            adj_corrs_total = [np.nan] * (self.num_channels - 1)

            # Identify the dominant controlling channel with maximum response magnitude
            max_ch = np.argmax(np.max(np.abs(phys_raw), axis=0))
            max_val = np.max(np.abs(phys_raw[:, max_ch]))

            # Verify if the controlling channel meets the signal-to-noise ratio requirements
            if max_val > self.snr_threshold:
                # Check left adjacent connection pair (if applicable)
                if max_ch > 0:
                    left_ch = max_ch - 1
                    # Proposed method (Pure dynamic increment correlation)
                    corr = np.corrcoef(dyn_data[:, left_ch], dyn_data[:, max_ch])[0, 1]
                    adj_corrs[left_ch] = corr  
                    # Traditional baseline (Total strain correlation)
                    corr_total = np.corrcoef(phys_raw[:, left_ch], phys_raw[:, max_ch])[0, 1]
                    adj_corrs_total[left_ch] = corr_total
                
                # Check right adjacent connection pair (if applicable)
                if max_ch < self.num_channels - 1:
                    right_ch = max_ch + 1
                    # Proposed method
                    corr = np.corrcoef(dyn_data[:, max_ch], dyn_data[:, right_ch])[0, 1]
                    adj_corrs[max_ch] = corr   
                    # Traditional baseline
                    corr_total = np.corrcoef(phys_raw[:, max_ch], phys_raw[:, right_ch])[0, 1]
                    adj_corrs_total[max_ch] = corr_total
                        
            self.correlations_history.append(adj_corrs)
            self.correlations_history_total.append(adj_corrs_total)

            # 6. Update moving-window smoothed health evaluation indicators
            self._update_smoothed_indicator()
            
    def _update_smoothed_indicator(self):
        """
        Updates moving average indicators to monitor structural health status in real time.
        """
        current_smoothed = []
        current_smoothed_total = [] 

        for ch in range(self.num_channels - 1):
            # 1. Proposed dynamic increment indicator smoothing
            history_for_this_ch = [record[ch] for record in self.correlations_history]
            valid_history = [val for val in history_for_this_ch if not np.isnan(val)]
            
            if len(valid_history) == 0:
                current_smoothed.append(np.nan)
            else:
                recent_valid_data = valid_history[-self.indicator_smooth_size:]
                current_smoothed.append(np.mean(recent_valid_data))

            # 2. Traditional total strain indicator smoothing
            history_for_this_ch_tot = [record[ch] for record in self.correlations_history_total]
            valid_history_tot = [val for val in history_for_this_ch_tot if not np.isnan(val)]
            
            if len(valid_history_tot) == 0:
                current_smoothed_total.append(np.nan)
            else:
                recent_valid_data_tot = valid_history_tot[-self.indicator_smooth_size:]
                current_smoothed_total.append(np.mean(recent_valid_data_tot))
                
        self.smoothed_indicators.append(current_smoothed)
        self.smoothed_indicators_total.append(current_smoothed_total)


def plot_single_vehicle_events_overlay(monitor, strain_data, ch_idx=0, plot_minutes=5, xlabel=True, out_name=None):
    """
    Plots the continuous raw strain time history superimposed with high-contrast red highlights
    denoting single-vehicle events captured adaptively by the FSM.
    """
    Fs = monitor.Fs
    if isinstance(plot_minutes, (list, tuple)) and len(plot_minutes) == 2:
        start_min, end_min = plot_minutes
    else:
        start_min = 0
        end_min = plot_minutes
        
    start_idx = int(start_min * 60 * Fs)
    end_idx = int(end_min * 60 * Fs)
    
    end_idx = min(end_idx, len(strain_data))
    start_idx = min(max(0, start_idx), end_idx)
    time_axis = np.arange(start_idx, end_idx) / Fs

    # plt.figure(figsize=(8, 4))
    plt.plot(
        time_axis, strain_data[start_idx:end_idx, ch_idx], 
        color='lightgray', linewidth=1.0, label='Raw Strain'
    )
    
    show_legend = True
    for s_idx, e_idx in monitor.single_events_indices:
        if e_idx <= start_idx: continue
        if s_idx >= end_idx: break
            
        plot_s_idx = max(s_idx, start_idx)
        plot_e_idx = min(e_idx, end_idx)
        event_time_axis = np.arange(plot_s_idx, plot_e_idx) / Fs
        
        plt.plot(
            event_time_axis, 
            strain_data[plot_s_idx:plot_e_idx, ch_idx], 
            color='red', linewidth=1.5, 
            label='Extracted Single-Vehicle Events' if show_legend else "_nolegend_"
        )
        show_legend = False  
        
    if xlabel:
        plt.xlabel(f'Time (s)', fontsize=12, weight='bold')
        
    plt.ylabel(r'Strain ($\mu\epsilon$)', fontsize=12, weight='bold')
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(start_idx / Fs, end_idx / Fs)
    plt.tight_layout()
    if out_name:
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
    # plt.show()


def plot_single_vehicle_events_overlay_concat(monitor, strain_data, ch_idx, plot_minutes, out_name=None):

    plt.figure(figsize=(8, 8))

    # Subplot 1
    plt.subplot(2, 1, 1)
    plot_single_vehicle_events_overlay(monitor, strain_data, ch_idx, plot_minutes=plot_minutes[0], xlabel=False)
    
    # Subplot 1
    plt.subplot(2, 1, 2)
    plot_single_vehicle_events_overlay(monitor, strain_data, ch_idx, plot_minutes=plot_minutes[1])

    if out_name:
        plt.savefig(out_name, dpi=600, bbox_inches='tight')


def plot_single_event_decomposition(monitor, target_event_idx, ch_idx=None, out_name=None):
    """
    Plots the multi-component decomposition (Total, Quasi-Static, and Dynamic Increment)
    for an isolated vehicle crossing event.
    """
    target_raw = monitor.single_events_pool[target_event_idx]
    target_dyn = monitor.single_events_dyn[target_event_idx]
    target_static = monitor.single_events_static[target_event_idx]
    start_idx, end_idx = monitor.single_events_indices[target_event_idx]

    if not ch_idx:
        ch_idx = np.argmax(np.max(np.abs(target_raw), axis=0))
    
    time_axis = np.arange(start_idx, end_idx) / monitor.Fs  

    plt.figure(figsize=(7.5, 5))
    plt.plot(time_axis, target_raw[:, ch_idx], color='black', linewidth=2, linestyle='-', label='Total Strain')
    plt.plot(time_axis, target_static[:, ch_idx], color='red', linewidth=2, linestyle='--', label='Quasi-Static Strain')
    plt.plot(time_axis, target_dyn[:, ch_idx], color='blue', linewidth=1.5, linestyle='-', label='Dynamic Increment')
    
    plt.xlabel('Time (s)', fontsize=12, weight='bold')
    plt.ylabel(r'Strain ($\mu\epsilon$)', fontsize=12, weight='bold')
    plt.legend(loc='upper right', fontsize=11) 
    plt.grid(True, which="both", linestyle=':', alpha=0.6)
    plt.tight_layout()
    if out_name:
        plt.savefig(out_name, dpi=600, bbox_inches='tight')
    # plt.show()


def plot_indicators_over_time(monitor, target_joint_idx=0, time_unit='hours', out_name_raw=None, out_name_smooth=None):
    """
    Generates isolated plots of statistical metrics over operational timelines,
    tailored specifically for LaTeX subfigure compilation.
    """
    if len(monitor.single_events_indices) == 0:
        print("No single-vehicle events captured. Plotting cancelled.")
        return
        
    end_indices = np.array([idx[1] for idx in monitor.single_events_indices])
    Fs = monitor.Fs
    
    if time_unit == 'hours':
        time_axis = end_indices / Fs / 3600.0
        x_label = 'Time (h)'
    elif time_unit == 'minutes':
        time_axis = end_indices / Fs / 60.0
        x_label = 'Time (min)'
    else:
        time_axis = end_indices / Fs
        x_label = 'Time (s)'
        
    raw_corrs = np.array(monitor.correlations_history)[:, target_joint_idx]
    smooth_inds = np.array(monitor.smoothed_indicators)[:, target_joint_idx]
    pair_name = f'Girder {target_joint_idx + 1} & {target_joint_idx + 2}'
    fig_size = (8, 4)
    
    # --- Plot 1: Discrete event scatter correlations ---
    plt.figure(figsize=fig_size)
    plt.scatter(
        time_axis, raw_corrs, 
        color='#1f77b4', s=25, alpha=0.5, marker='o', 
        label=f'Joint {target_joint_idx + 1} ({pair_name})'
    )
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
    plt.ylim(0.0, 1.05) 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    if out_name_raw:
        plt.savefig(out_name_raw, dpi=600, bbox_inches='tight')
    # plt.show()
    
    # --- Plot 2: Window-averaged continuous trend indicators ---
    plt.figure(figsize=fig_size)
    plt.plot(
        time_axis, smooth_inds, 
        color='#d62728', linewidth=2.5, 
        label=f'Joint {target_joint_idx + 1} ({pair_name})'
    )
    plt.xlabel(x_label, fontsize=12, fontweight='bold')
    plt.ylabel('Indicator', fontsize=12, fontweight='bold')
    plt.ylim(0.0, 1.05) 
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.tight_layout()
    if out_name_smooth:
        plt.savefig(out_name_smooth, dpi=600, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    import random
    
    # Fix random seed for traffic reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    from simulation_strain import simulate_bridge_strain

    # Unified structural configuration (Assembled precast hollow slab bridge)
    LL = 20; MM = 8; NN = 21; Density = 2500; Area = 0.51 
    E = 3.45e10; I = 0.05; k_hinge = 5e6; y_sensor = 0.45
    Fs = 50
    
    print("Executing baseline streaming processing across 12 monitoring hours...")
    strain_history, vehicle_records, freq_Hz = simulate_bridge_strain(
        HOUR=12, Fs=Fs, DT=20,
        LL=LL, MM=MM, NN=NN, Density=Density, Area=Area,
        E=E, I=I, k_hinge=k_hinge, y_sensor=y_sensor,
        add_noise=False, SNR=40
    )

    # Initialize rolling state machine monitor
    monitor = RealTimeBridgeMonitor(
        bridge_freq_hz=freq_Hz, 
        num_channels=strain_history.shape[1], 
        Fs=Fs, 
        ma_threshold=2, 
        ma_length_multiple=2.0,
        low_cutoff_multiplier=0.7,
        snr_threshold=5, 
        indicator_smooth_size=200
    )

    # Stream raw inputs continuously line by line
    for sample in strain_history:
        monitor.process_sample(sample)
    print("Streaming simulation successfully completed.")

    print("\n" + "="*50)
    print(f"1. Number of vehicles generated: {len(vehicle_records)}")
    print(f"2. Number of bridge crossing events detected: {len(monitor.correlations_history)}")

    discard_rate = (1 - len(monitor.correlations_history) / len(vehicle_records)) * 100 if len(vehicle_records) > 0 else 0
    print(f"--> Discard Rate: {discard_rate:.2f}%")
    print("="*50 + "\n")

    # Execute verification plotting pipeline
    plot_single_vehicle_events_overlay_concat(
        monitor, strain_history, ch_idx=0, plot_minutes=[[20, 30], [24, 26]], out_name='event_extraction.pdf')
    plot_single_event_decomposition(monitor, target_event_idx=4, ch_idx=5, out_name='event_decomposition.pdf')
    plot_indicators_over_time(monitor, target_joint_idx=2, time_unit='hours', out_name_raw='PCC_scatter.pdf', out_name_smooth='indicator_line.pdf')
    print("Verification graphics exported successfully.")