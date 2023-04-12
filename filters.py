import numpy as np


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
