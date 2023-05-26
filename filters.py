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

    return filtered_signal


def simple_skin_selection(frame, lower_rgb=75, higher_rgb=200):
    """
    :param frame:
        Input frames of video
    :param lower_rgb:
        Lower RGB threshold level
    :param higher_rgb:
        Higher RGB threshold level
    :return:
        Returns filtered pixels that lies between given RGB threshold
    """
    lower_rgb_threshold = np.any(frame < lower_rgb, axis=-1)
    higher_rgb_threshold = np.any(frame > higher_rgb, axis=-1)
    # Combine these indices
    indices = np.logical_or(lower_rgb_threshold, higher_rgb_threshold)

    # Create a copy of the image to not overwrite the original one
    img_copy = frame.copy()
    # Change these pixels to black
    img_copy[indices] = 0

    return img_copy

