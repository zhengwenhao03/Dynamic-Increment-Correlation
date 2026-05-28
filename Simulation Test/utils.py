import numpy as np
from scipy.signal import find_peaks

def count_vehicles_in_event(event_ma_data, Fs):
    """
    Counts the number of vehicles within a single isolated event based on the 
    causal moving average signal without boundary truncation errors.
    
    Args:
        event_ma_data (np.ndarray): Causal moving average vector (N_samples, n_channels).
        Fs (float): Sampling frequency (Hz).
        
    Returns:
        int: Number of detected vehicle peaks within the event window.
    """
    # Identify the dominant controlling channel with the maximum response magnitude
    max_channel_idx = np.argmax(np.max(np.abs(event_ma_data), axis=0))
    local_ma = event_ma_data[:, max_channel_idx]
    
    max_val = np.max(local_ma)
    
    # Safety guard: Return 0 if the captured signal event is excessively weak (noise floor)
    if max_val < 1e-1: 
        return 0
        
    # Detect local peaks directly on the continuous moving average curve
    peaks, _ = find_peaks(
        local_ma,
        height=max_val * 0.3,       # Filter out micro-fluctuations and secondary waves
        distance=int(0.1 * Fs),     # Minimum physical time gap between successive axles (0.1s)
        prominence=max_val * 0.1    # Guarantee a distinct and completed individual wave envelope
    )
    
    return len(peaks)