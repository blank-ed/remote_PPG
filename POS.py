"""

This module contains the framework implemented by https://ieeexplore.ieee.org/document/7565547 which is
also known as POS rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""

from scipy.signal import find_peaks, stft
from remote_PPG.utils import *


def pos_framework(input_video):

    raw_sig = extract_raw_sig(input_video, framework='POS', width=1, height=1)
    fps = get_fps(input_video)

    N = len(raw_sig)
    H = np.zeros(N)
    l = int(fps * 1.6)

    windowed_sig = moving_window(raw_sig, fps=fps, window_size=1.6, increment=1/30)

    for enum, sig in enumerate(windowed_sig):
        sig = np.array(sig)
        sig = np.array([sig[:, i] for i in range(0, 3)])

        # Spatial Averaging
        mean_color = np.mean(sig, axis=1)

        # Temporal normalization
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv, sig)

        # Projection
        S1 = Cn[1] - Cn[2]
        S2 = Cn[1] + Cn[2] - 2 * Cn[0]
        alpha = np.std(S1) / np.std(S2)
        h = S1 + alpha * S2

        # Overlap-Adding
        H[enum:enum + l] = H[enum:enum + l] + (h - np.mean(h))

    # Compute STFT
    noverlap = fps * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = fps * 12  # Length of fourier window (12 seconds as per the paper)
    frequencies, times, Zxx = stft(H, fps, nperseg=nperseg, noverlap=noverlap)  # Perform STFT
    magnitude_Zxx = np.abs(Zxx)  # Calculate the magnitude of Zxx

    # Detect Peaks for each time slice
    hr = []
    for i in range(magnitude_Zxx.shape[1]):
        peaks, _ = find_peaks(magnitude_Zxx[:, i])
        if len(peaks) > 0:
            peak_freq = frequencies[peaks[np.argmax(magnitude_Zxx[peaks, i])]]
            hr.append(peak_freq * 60)
        else:
            hr.append(None)

    return hr

