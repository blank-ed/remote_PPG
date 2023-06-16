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

    for n in range(0, N):
        m = n - l + 1
        if n - l + 1 > 0:
            # Temporal normalization
            Cn = np.array(raw_sig[m:n + 1]) / np.mean(np.array(raw_sig[m:n + 1]))

            # Projection
            S1 = Cn[:, 1] - Cn[:, 2]
            S2 = Cn[:, 1] + Cn[:, 2] - 2 * Cn[:, 0]
            alpha = np.std(S1) / np.std(S2)
            h = S1 + alpha * S2

            # Overlap-Adding
            H[m:n + 1] += (h - np.mean(h))

    # Compute STFT
    noverlap = fps * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = fps * 12  # Length of fourier window (12 seconds as per the paper)
    frequencies, times, Zxx = stft(H, fps, nperseg=nperseg, noverlap=noverlap)  # Perform STFT
    magnitude_Zxx = np.abs(Zxx)  # Calculate the magnitude of Zxx

    # Detect Peaks for each time slice
    hr = []
    for i in range(magnitude_Zxx.shape[1]):
        mask = (frequencies >= 0.67) & (frequencies <= 4)  # create a mask for the desired frequency range
        masked_frequencies = frequencies[mask]
        masked_magnitude = magnitude_Zxx[mask, i]

        peaks, _ = find_peaks(masked_magnitude)
        if len(peaks) > 0:
            peak_freq = masked_frequencies[peaks[np.argmax(masked_magnitude[peaks])]]
            hr.append(peak_freq * 60)
        else:
            hr.append(None)

    return hr


### TEST SECTION

import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from scipy.signal import windows

def pos_ubfc1(ground_truth_file, sampling_frequency=60):
    gtdata = pd.read_csv(ground_truth_file, header=None)
    gtTrace = gtdata.iloc[:, 3].tolist()
    gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
    gtHR = gtdata.iloc[:, 1]

    # Compute STFT
    noverlap = sampling_frequency * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = sampling_frequency * 12  # Length of fourier window (12 seconds as per the paper)
    frequencies, times, Zxx = stft(np.array(gtTrace), sampling_frequency, nperseg=nperseg, noverlap=noverlap)  # Perform STFT
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


def pos_ubfc2(ground_truth_file, sampling_frequency=30):
    gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
    gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
    gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
    gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

    # Compute STFT
    noverlap = sampling_frequency * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = sampling_frequency * 12  # Length of fourier window (12 seconds as per the paper)
    frequencies, times, Zxx = stft(np.array(gtTrace), sampling_frequency, nperseg=nperseg,
                                   noverlap=noverlap)  # Perform STFT
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
pos_true = []
pos_pred = []
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
            hrGT = pos_ubfc1(ground_truth_file=gt)
            print(len(hrGT), len(hrES))
            print('')
            pos_true.append(np.mean(hrGT))
            pos_pred.append(np.mean(hrES))

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
            hrGT = pos_ubfc2(ground_truth_file=gt)
            print(len(hrGT), len(hrES))
            print('')
            pos_true.append(np.mean(hrGT))
            pos_pred.append(np.mean(hrES))

print(mean_absolute_error(pos_true, pos_pred))
print(mean_absolute_error(pos_true[8:], pos_pred[8:]))
print(pos_true)
print(pos_pred)

# print(MAE)
# print(np.mean(MAE))

# 28.782992049506166
# 31.589977378837375
# [75.33707865168539, 79.26136363636364, 91.55172413793103, 63.3974358974359, 69.8913043478261, 74.6470588235294, 90.0, 101.6260162601626, 108.01886792452831, 94.2741935483871, 105.1063829787234, 99.05660377358491, 112.02898550724638, 108.40579710144928, 110.3623188405797, 123.76811594202898, 68.75, 103.26086956521739, 69.92647058823529, 115.8695652173913, 93.07142857142857, 86.78571428571429, 120.72463768115942, 125.8695652173913, 102.68115942028986, 65.97014925373135, 106.54411764705883, 115.0, 98.33333333333333, 109.38775510204081, 98.04347826086956, 78.76811594202898, 105.43478260869566, 116.83823529411765, 116.44927536231884, 103.64285714285714, 119.28571428571429, 57.10144927536232, 109.14285714285714, 84.92753623188406, 86.15942028985508, 101.01449275362319, 95.14285714285714, 99.48529411764706, 82.89855072463769, 110.14492753623189, 97.20588235294117, 105.94202898550725, 90.8955223880597, 87.82608695652173]
# [45.36585365853659, 39.93827160493827, 90.0625, 64.58333333333333, 76.66666666666667, 56.34615384615385, 83.88888888888889, 92.41228070175438, 96.32075471698113, 88.70967741935483, 98.40425531914893, 30.28301886792453, 88.76811594202898, 36.81159420289855, 31.666666666666668, 76.95652173913044, 64.19117647058823, 63.69565217391305, 52.13235294117647, 37.2463768115942, 57.714285714285715, 59.92857142857143, 85.8695652173913, 79.42028985507247, 53.98550724637681, 60.149253731343286, 84.11764705882354, 78.0, 126.07843137254902, 43.46938775510204, 55.36231884057971, 67.46376811594203, 94.85507246376811, 153.4558823529412, 44.85507246376812, 103.92857142857143, 90.0, 52.875, 61.142857142857146, 71.95652173913044, 60.869565217391305, 32.2463768115942, 89.92857142857143, 78.01470588235294, 71.52173913043478, 60.289855072463766, 54.19117647058823, 86.81159420289855, 88.95522388059702, 79.42028985507247]


### END TEST SECTION