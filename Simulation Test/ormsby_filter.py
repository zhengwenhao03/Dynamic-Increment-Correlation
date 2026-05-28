# Note: This module provides both the basic zero-phase Ormsby low-pass filter 
# for quasi-static baseline decoupling and an extended ormsby_bandpass solver 
# for targeted modal frequency band isolation.

import numpy as np
from scipy.ndimage import convolve1d

def ormsby(x, rf, tf, dt):
    """
    Apply a Time-Domain Zero-Phase Ormsby Filter to the input data.

    Args:
        x  (matrix or vector): Input signal data: (N_samples, n_channels) or (N_samples,).
        rf (float): Roll-off frequency (Hz).
        tf (float): Termination frequency (Hz).
        dt (float): Sample interval (seconds).

    Returns:
        np.ndarray: Filtered data signal.
    """
    x = np.array(x, dtype=float)
    
    # --- Filter Configuration ---
    nn = 100  # Filter half-length
    ll = 100  # Window half-length
    
    # Angular frequencies
    wp = 2.0 * np.pi * rf
    ws = 2.0 * np.pi * tf
    dw = ws - wp

    # --- 1. Generate Impulse Response (Sinc-like formulation) ---
    t = np.arange(1, nn + 1) * dt
    numerator = np.cos(wp * t) - np.cos(ws * t)
    denominator = dw * np.pi * (t**2)
    hd = numerator / denominator

    # --- 2. Generate Kaiser-like Window ---
    beta = 5.0
    k_indices = np.arange(ll + 1)
    ratios = k_indices / ll
    w_weights = np.i0(beta * np.sqrt(1 - ratios**2)) / np.i0(beta)

    # --- 3. Construct Final Filter Kernel ---
    # Create the base half-filter
    h_half = np.zeros(ll + 1)
    h_half[0] = rf + tf  # DC terms
    h_half[1:] = hd      # AC terms

    # Apply window weights
    h_windowed = h_half * w_weights

    # Mirror to create symmetric kernel: [Left Wing, Center, Right Wing]
    # h_windowed[0] is the center.
    kernel = np.concatenate([h_windowed[1:][::-1], [h_windowed[0]], h_windowed[1:]])
    
    # Normalize area to 1
    kernel /= np.sum(kernel)

    # --- 4. Pad and Convolve ---
    conv_axis = 0 if x.ndim > 1 else -1
    xx = convolve1d(x, weights=kernel, axis=conv_axis, mode='reflect')

    return xx

def ormsby_bandpass(x, f1, f2, f3, f4, dt):
    """
    Constructs a zero-phase bandpass filter based on the Ormsby low-pass formulation.
    
    Args:
        x (array-like): Input multi-channel or single-channel signal data.
        f1 (float): Lower stopband cutoff frequency (Hz). Energy below f1 is completely attenuated.
        f2 (float): Lower passband start frequency (Hz). Transition zone lies between f1 and f2.
        f3 (float): Upper passband cutoff frequency (Hz). Passband region lies between f2 and f3.
        f4 (float): Upper stopband start frequency (Hz). Transition zone lies between f3 and f4.
        dt (float): Sampling interval (seconds).
        
    Returns:
        np.ndarray: Bandpass filtered signal data.
    """
    # 1. Isolate all components below the upper high-frequency threshold
    # rf = f3, tf = f4
    lp_high = ormsby(x, rf=f3, tf=f4, dt=dt)
    
    # 2. Extract the low-frequency quasi-static components to be subtracted
    # rf = f1, tf = f2
    lp_low = ormsby(x, rf=f1, tf=f2, dt=dt)
    
    # 3. Perform subtraction to yield the clean bandpass signal containing f2 ~ f3 energy
    bp_signal = lp_high - lp_low
    
    return bp_signal

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Verifying Ormsby filter kernel parallel processing implementation...")

    # 1. Generate synthetic multi-channel test signals
    Fs = 50.0          # Sampling frequency (50Hz)
    dt = 1.0 / Fs
    N = 500            # Simulating 10 operational seconds
    time = np.arange(N) * dt

    # Simulate 3 parallel channels with transient Gaussian pulse, sine wave, and white noise
    ch1 = 10 * np.exp(-((time-5)**2) / 1.0) + 2.0 * np.sin(2 * np.pi * 3.0 * time) + np.random.normal(0, 0.2, N)
    ch2 =  8 * np.exp(-((time-5)**2) / 1.5) + 1.5 * np.sin(2 * np.pi * 3.5 * time) + np.random.normal(0, 0.2, N)
    ch3 =  5 * np.exp(-((time-5)**2) / 2.0) + 1.0 * np.sin(2 * np.pi * 4.0 * time) + np.random.normal(0, 0.2, N)

    # Reshape and stack into a 2D matrix of shape (N_samples, n_channels)
    data_matrix = np.column_stack((ch1, ch2, ch3))

    # 2. Execute concurrent multi-channel low-pass filtering
    # Configure cutoff frequencies: Retain components below 1.0 Hz, attenuate above 1.5 Hz
    rf = 1.0 
    tf = 1.5 

    # Apply filter kernel to strip high-frequency dynamic responses
    data_lowpass = ormsby(data_matrix, rf, tf, dt)
    data_highpass = data_matrix - data_lowpass

    # 3. Performance verification and high-fidelity graphics rendering
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    labels = ['Ch 1', 'Ch 2', 'Ch 3']

    # Subplot 1: Plot original composite multi-channel signals
    for i in range(3):
        axes[0].plot(time, data_matrix[:, i], color=colors[i], label=labels[i])
    axes[0].set_title('Original Multi-channel Signal', fontsize=12, fontweight='bold')

    # Subplot 2: Plot decoupled low-frequency quasi-static baselines
    for i in range(3):
        axes[1].plot(time, data_lowpass[:, i], color=colors[i], linestyle='--', linewidth=2.5, label=labels[i])
    axes[1].set_title(f'Filtered Low-Frequency Component (Ormsby Low-pass: $r_f$={rf}Hz, $t_f$={tf}Hz)', fontsize=12, fontweight='bold')

    # Subplot 3: Plot remaining high-frequency dynamic strain increments
    for i in range(3):
        axes[2].plot(time, data_highpass[:, i], color=colors[i], linewidth=1.5, label=labels[i])
    axes[2].set_title('High-Frequency Component', fontsize=12, fontweight='bold')

    # Graph decorations and academic aesthetics formatting
    for ax in axes:
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.set_ylabel('Amplitude', fontsize=11, fontweight='bold')

    axes[-1].set_xlabel('Time (Seconds)', fontsize=12, fontweight='bold')

    plt.tight_layout()
    print("Filter verification complete. Displaying interactive graphics.")
    plt.show()