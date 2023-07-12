import numpy as np
from remote_PPG.filters import *
from remote_PPG.utils import *
from scipy.signal import windows
from ICA_framework.jadeR import jadeR


from line_profiler import LineProfiler
def profile_print(func_to_call, *args, **kwargs):
    profiler = LineProfiler()
    profiler.add_function(func_to_call)
    profiler.runcall(func_to_call, *args, **kwargs)
    profiler.print_stats()


def CHROM(signal, fps, **params):

    w, l, c = signal.shape
    N = int(l + (w - 1) * (params['increment'] * fps))
    H = np.zeros(N)

    for enum, each_window in enumerate(signal):
        normalized = normalize(signal=each_window, normalize_type='mean_normalization')  # Normalize each windowed segment

        # Build two orthogonal chrominance signals
        Xs = 3 * normalized[0] - 2 * normalized[1]
        Ys = 1.5 * normalized[0] + normalized[1] - 1.5 * normalized[2]

        # Stack signals and apply the bandpass filter
        stacked_signals = np.stack([Xs, Ys], axis=-1)
        filtered_signals = fir_bp_filter(signal=stacked_signals, fps=30, low=0.67, high=4.0)
        Xf, Yf = filtered_signals[:, 0], filtered_signals[:, 1]

        if np.std(Yf) != 0:
            alpha = np.std(Xf) / np.std(Yf)
        else:
            alpha = 0
        S = Xf - alpha * Yf

        SWin = np.multiply(S, windows.hann(len(S)))

        start = enum * (l // 2)
        end = enum * (l // 2) + l

        H[start:end] = H[start:end] + SWin

    return H


def POS(signal, fps, **params):

    w, l, c = signal.shape
    N = int(l + (w - 1) * (params['increment'] * fps))
    H = np.zeros(N)

    for enum, each_window in enumerate(signal):
        normalized = normalize(signal=each_window, normalize_type='mean_normalization')  # Normalize each windowed segment

        # Projection
        S1 = normalized[1] - normalized[2]
        S2 = normalized[1] + normalized[2] - 2 * normalized[0]

        if np.std(S2) != 0:
            alpha = np.std(S1) / np.std(S2)
        else:
            alpha = 0

        h = S1 + alpha * S2

        start = enum
        end = enum + l

        H[start:end] += (h - np.mean(h))

    return H


def ICA(signal):

    comp = 1  # Change this to take in the comp from params

    bvp = []
    for each_window in signal:
        normalized = normalize(each_window, normalize_type='zero_mean_unit_variance')

        # Apply JADE ICA algorithm and select the second component
        W = jadeR(normalized, m=3)
        bvp.append(np.array(np.dot(W, normalized))[comp].flatten())

    return np.array(bvp)


def GREEN(signal):

    return np.array(signal)


def LiCVPR(signal, bg_signal):

    bvp = []
    for enum, each_window in enumerate(signal):
        continue

    return bvp

