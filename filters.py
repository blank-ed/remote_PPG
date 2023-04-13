import numpy as np
from scipy import signal


def normalize(sig):
    """
    :param sig:
        Input signal to normalize to have zero-mean and unit variance
    :return:
        Normalized signal
    """
    signal = np.array(sig)
    mean = np.mean(signal, axis=0)
    std_dev = np.std(signal, axis=0)
    normalized_signal = (signal - mean) / std_dev

    # Turn normalized signal to [[R] [G] [B]]
    normalized = np.array([[row[i] for row in normalized_signal] for i in range(0, 3)])

    return normalized


def bp_filter(sig, fps, low=0.5, high=3.7):
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
        Returns the bandpass filtered signal
    """

    filtered = signal.firwin(sig, fs=fps, cutoff=[low, high], window='hamming', pass_zero='bandpass')

    return filtered