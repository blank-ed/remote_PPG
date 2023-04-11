import numpy as np


def normalize(sig):
    """
    :param sig:
        Input signal to normalize
    :return:
        Normalized signal to have zero-mean and unit variance
    """
    signal = np.array(sig)
    mean = np.mean(signal, axis=0)
    std_dev = np.std(signal, axis=0)

    # Normalize each color channel to have zero-mean and unit variance
    normalized_signal = (signal - mean) / std_dev

    return normalized_signal.tolist()