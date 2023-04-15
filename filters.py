import numpy as np
from scipy.signal import firwin, filtfilt


def normalize(sig):
    """
    :param sig:
        Input signal to normalize to have zero-mean and unit variance
    :return:
        Normalized signal in [[R] [G] [B]] format
    """
    signal = np.array(sig)
    mean = np.mean(signal, axis=0)
    std_dev = np.std(signal, axis=0)
    normalized_signal = (signal - mean) / std_dev

    # Turn normalized signal to [[R] [G] [B]]
    normalized = np.array([normalized_signal[:, i] for i in range(0, 3)])
    return normalized


def fir_bp_filter(sig, fps, low=0.5, high=3.7):
    """
    :param sig:
        Takes in the signal to be filtered
    :param fps:
        This is the fps of the video file, which is also the sampling frequency
    :param low:
        This is the low frequency level
    :param high:
        This is the high frequency level
    :return:
        Returns the bandpass filtered signal in [[R] [G] [B]] format
    """
    signal = np.array(sig)

    # Coefficients of FIR bandpass filter
    filter_coefficients = firwin(numtaps=32, cutoff=[low, high], fs=fps, pass_zero=False, window='hamming')

    # Filtering using the FIR bandpass filter coefficients.
    # Since its FIR bandpass filter, the denominator coefficient is set as 1
    filtered_signal = filtfilt(filter_coefficients, 1, signal, axis=0)

    # Turn filtered signal to [[R] [G] [B]]
    filtered_signal = np.array([filtered_signal[:, i] for i in range(0, 3)])

    return filtered_signal
