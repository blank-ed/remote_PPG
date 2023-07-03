import numpy as np
from itertools import combinations


def normalize(signal, normalize_type=None):
    """
    :param signal:
        Input signal to normalize
    :param normalize_type:
        Normalize signal using one of three types:
            - Mean normalization: Dividing signal by its mean value
            - Zero mean: Subtracting signal by its mean value
            - Zero mean with unit variance: This is also known as standardization
    :return:
        Normalized signal in [[R] [G] [B]] format
    """
    signal = np.array(signal)
    mean = np.mean(signal, axis=0)
    std_dev = np.std(signal, axis=0)

    if normalize_type == 'mean_normalization':
        normalized_signal = signal / mean
    elif normalize_type == 'zero_mean':
        normalized_signal = signal - mean
    elif normalize_type == 'zero_mean_unit_variance':
        normalized_signal = (signal - mean) / std_dev
    else:
        assert False, "Invalid normalization type. Please choose one of the valid available types " \
                      "Types: 'mean_normalization', 'zero_mean', or 'zero_mean_unit_variance' "

    # Turn normalized signal to [[R] [G] [B]]
    normalized = np.array([normalized_signal[:, i] for i in range(0, 3)])
    return normalized


def moving_window(sig, fps, window_size, increment):
    """
    :param sig:
        RGB signal
    :param fps:
        Frame rate of the video file (number of frames per second)
    :param window_size:
        Select the window size in seconds (s)
    :param increment:
        Select amount to be incremented in seconds (s)
    :return:
        returns the windowed signal
    """

    windowed_sig = []
    for i in range(0, len(sig), int(increment * fps)):
        end = i + int(window_size * fps)
        if end > len(sig):
            # windowed_sig.append(sig[len(sig) - int(window_size * fps):len(sig)])
            break
        windowed_sig.append(sig[i:end])

    return np.array(windowed_sig)


def get_filtering_combinations(filtering_methods):
    """
    :param filtering_methods:
        Enter a list of filtering methods you want to apply to get unique combinations
    :return:
        Returns a list of unique combinations of different types of filters
    """
    unique_combinations = []

    # Generate combinations of different lengths (0 to 4)
    for r in range(0, len(filtering_methods) + 1):
        # Generate combinations
        for combo in combinations(filtering_methods, r):
            # Check the condition of not having Butterworth and FIRWIN in the same line
            if not ('butterworth_bp_filter' in combo and 'fir_bp_filter' in combo):
                # Add the combination to the list
                unique_combinations.append(combo)

    return unique_combinations


def calculate_mean_rgb(frame):
    """
    :param frame:
        Input frame of video
    :return:
        Returns the mean RGB values of non-black pixels
    """
    # Find the indices of the non-black pixels
    non_black_pixels = np.all(frame != [0, 0, 0], axis=-1)

    # Get the non-black pixels
    non_black_pixels_frame = frame[non_black_pixels]

    # Calculate and return the mean RGB values
    return np.mean(non_black_pixels_frame, axis=0)

