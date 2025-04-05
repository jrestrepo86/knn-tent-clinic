import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def hilbert_phase(signal):

    signal = np.asarray(signal, dtype=np.float64)
    analytic_signal = hilbert(signal)
    analytic_signal = np.asarray(analytic_signal, dtype=np.complex128)
    instantaneous_phase = np.angle(analytic_signal)
    return instantaneous_phase


def filter_signal(signal, lowcut, highcut, fs, order=4):
    # Validate input arguments
    if lowcut >= highcut:
        raise ValueError("lowcut must be less than highcut")
    if highcut >= 0.5 * fs:
        raise ValueError("highcut must be less than Nyquist frequency (0.5 * fs)")
    if order <= 0:
        raise ValueError("order must be a positive integer")
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(N=order, Wn=[low, high], btype="band", output="ba")
    filtered = filtfilt(b, a, signal)  # Zero-phase filtering
    return filtered


def wavelet_phase(signal, lowcut, highcut, fs, omega0=5.0, num_scales=10):
    """
    Compute phase using complex Morlet wavelets.

    Parameters:
        signal (array): Input signal
        lowcut (float): Low frequency cutoff (Hz)
        highcut (float): High frequency cutoff (Hz)
        fs (float): Sampling frequency (Hz)
        omega0 (float): Wavelet central frequency parameter
        num_scales (int): Number of scales to use

    Returns:
        phase (array): Instantaneous phase (radians)
    """
    dt = 1.0 / fs
    scales = (omega0 / (2 * np.pi)) / (np.linspace(highcut, lowcut, num_scales) * dt)

    cwt_coeffs = []
    for scale in scales:
        t_win = np.arange(-3 * scale, 3 * scale + dt, dt)
        t_scaled = t_win / scale
        wavelet = (
            (np.pi**-0.25) * np.exp(1j * omega0 * t_scaled) * np.exp(-0.5 * t_scaled**2)
        )
        wavelet /= np.sqrt(scale)  # Energy normalization
        coeff = np.convolve(signal, wavelet, mode="same")
        cwt_coeffs.append(coeff)

    avg_coeffs = np.mean(cwt_coeffs, axis=0)
    return np.angle(avg_coeffs)
