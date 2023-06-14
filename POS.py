"""

This module contains the framework implemented by https://ieeexplore.ieee.org/document/7565547 which is
also known as POS rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""

from scipy.signal import find_peaks, stft
from remote_PPG.utils import *


def pos_framework(input_video, dataset=None):

    raw_sig = extract_raw_sig(input_video, framework='POS', width=1, height=1)
    if dataset is None:
        fps = get_fps(input_video)  # find the fps of the video
    elif dataset == 'UBFC1' or dataset == 'UBFC2':
        fps = 30
    else:
        assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                      "types: 'UBFC1', 'UBFC2'. If you are using your own dataset, enter 'None' "

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


### TEST SECTION

import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from scipy.signal import windows

def chrom_ubfc1(ground_truth_file, sampling_frequency=60):
    gtdata = pd.read_csv(ground_truth_file, header=None)
    gtTrace = gtdata.iloc[:, 3].tolist()
    gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
    gtHR = gtdata.iloc[:, 1]

    N = len(gtTrace)
    H = np.zeros(N)
    l = int(sampling_frequency * 1.6)

    window = moving_window(gtTrace, fps=sampling_frequency, window_size=1.6, increment=0.8)

    for enum, each_window in enumerate(window):
        normalized = np.array(each_window) / np.mean(each_window)

        # bandpass filter Xs and Ys here
        filtered = fir_bp_filter(signal=normalized, fps=sampling_frequency, low=0.67, high=4.0)

        SWin = np.multiply(filtered, windows.hann(len(filtered)))

        start = enum * (l // 2)
        end = enum * (l // 2) + l

        if end > len(gtTrace):
            H[len(gtTrace) - l:len(gtTrace)] = H[len(gtTrace) - l:len(gtTrace)] + SWin
        else:
            H[start:end] = H[start:end] + SWin

    # Compute STFT
    noverlap = sampling_frequency * (
                12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = sampling_frequency * 12  # Length of fourier window (12 seconds as per the paper)

    frequencies, times, Zxx = stft(H, sampling_frequency, nperseg=nperseg, noverlap=noverlap)  # Perform STFT

    magnitude_Zxx = np.abs(Zxx)  # Calculate the magnitude of Zxx

    # Detect Peaks for each time slice
    hrGT = []
    for i in range(magnitude_Zxx.shape[1]):
        peaks, _ = find_peaks(magnitude_Zxx[:, i])
        if len(peaks) > 0:
            peak_freq = frequencies[peaks[np.argmax(magnitude_Zxx[peaks, i])]]
            hrGT.append(peak_freq * 60)
        else:
            hrGT.append(None)

    return hrGT


def chrom_ubfc2(ground_truth_file, sampling_frequency=30):
    gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
    gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
    gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
    gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

    N = len(gtTrace)
    H = np.zeros(N)
    l = int(sampling_frequency * 1.6)

    window = moving_window(gtTrace, fps=sampling_frequency, window_size=1.6, increment=0.8)

    for enum, each_window in enumerate(window):
        normalized = np.array(each_window) / np.mean(each_window)

        # bandpass filter Xs and Ys here
        filtered = fir_bp_filter(signal=normalized, fps=sampling_frequency, low=0.67, high=4.0)

        SWin = np.multiply(filtered, windows.hann(len(filtered)))

        start = enum * (l // 2)
        end = enum * (l // 2) + l

        if end > len(gtTrace):
            H[len(gtTrace) - l:len(gtTrace)] = H[len(gtTrace) - l:len(gtTrace)] + SWin
        else:
            H[start:end] = H[start:end] + SWin

    # Compute STFT
    noverlap = sampling_frequency * (
                12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = sampling_frequency * 12  # Length of fourier window (12 seconds as per the paper)

    frequencies, times, Zxx = stft(H, sampling_frequency, nperseg=nperseg, noverlap=noverlap)  # Perform STFT

    magnitude_Zxx = np.abs(Zxx)  # Calculate the magnitude of Zxx

    # Detect Peaks for each time slice
    hrGT = []
    for i in range(magnitude_Zxx.shape[1]):
        peaks, _ = find_peaks(magnitude_Zxx[:, i])
        if len(peaks) > 0:
            peak_freq = frequencies[peaks[np.argmax(magnitude_Zxx[peaks, i])]]
            hrGT.append(peak_freq * 60)
        else:
            hrGT.append(None)

    return hrGT


# MAE = []
chrom_true = []
chrom_pred = []
base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
# base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
for sub_folders in os.listdir(base_dir):
    if sub_folders == 'UBFC1':
        for folders in os.listdir(os.path.join(base_dir, sub_folders)):
            subjects = os.path.join(base_dir, sub_folders, folders)
            for each_subject in os.listdir(subjects):
                if each_subject.endswith('.avi'):
                    vid = os.path.join(subjects, each_subject)
                elif each_subject.endswith('.xmp'):
                    gt = os.path.join(subjects, each_subject)

            print(vid, gt)
            hrES = pos_framework(input_video=vid, dataset='UBFC1')
            hrGT = chrom_ubfc1(ground_truth_file=gt)
            print(len(hrGT), len(hrES))
            print('')
            chrom_true.append(np.mean(hrGT))
            chrom_pred.append(np.mean(hrES))
            # if len(hrGT) > len(hrES):
            #     MAE.append(mean_absolute_error(hrGT[0:len(hrES)], hrES))
            # else:
            #     MAE.append(mean_absolute_error(hrGT, hrES[0:len(hrGT)]))

    elif sub_folders == 'UBFC2':
        for folders in os.listdir(os.path.join(base_dir, sub_folders)):
            subjects = os.path.join(base_dir, sub_folders, folders)
            for each_subject in os.listdir(subjects):
                if each_subject.endswith('.avi'):
                    vid = os.path.join(subjects, each_subject)
                elif each_subject.endswith('.txt'):
                    gt = os.path.join(subjects, each_subject)

            print(vid, gt)
            hrES = pos_framework(input_video=vid, dataset='UBFC2')
            hrGT = chrom_ubfc2(ground_truth_file=gt)
            print(len(hrGT), len(hrES))
            print('')
            chrom_true.append(np.mean(hrGT))
            chrom_pred.append(np.mean(hrES))
            # if len(hrGT) > len(hrES):
            #     MAE.append(mean_absolute_error(hrGT[0:len(hrES)], hrES))
            # else:
            #     MAE.append(mean_absolute_error(hrGT, hrES[0:len(hrGT)]))

print(mean_absolute_error(chrom_true, chrom_pred))

# print(MAE)
# print(np.mean(MAE))



### END TEST SECTION