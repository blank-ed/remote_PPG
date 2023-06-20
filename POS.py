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
    elif dataset == 'LGI_PPGI':
        fps = 25
    else:
        assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                      "types: 'UBFC1', 'UBFC2', or 'LGI_PPGI'. If you are using your own dataset, enter 'None' "

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


def pos_lgi_ppgi(ground_truth_file, sampling_frequency=60):
    gtdata = pd.read_xml(ground_truth_file)
    gtTime = (gtdata.iloc[:, 0]).tolist()
    gtHR = gtdata.iloc[:, 1].tolist()
    gtTrace = gtdata.iloc[:, 2].tolist()

    # Compute STFT
    noverlap = sampling_frequency * (
                12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
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

pos_true = []
pos_pred = []
# # base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
# base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
# for sub_folders in os.listdir(base_dir):
#     if sub_folders == 'UBFC1':
#         for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#             subjects = os.path.join(base_dir, sub_folders, folders)
#             for each_subject in os.listdir(subjects):
#                 if each_subject.endswith('.avi'):
#                     vid = os.path.join(subjects, each_subject)
#                 elif each_subject.endswith('.xmp'):
#                     gt = os.path.join(subjects, each_subject)
#
#             print(vid, gt)
#             hrES = pos_framework(input_video=vid, dataset='UBFC1')
#             hrGT = pos_ubfc1(ground_truth_file=gt)
#             print(len(hrGT), len(hrES))
#             print('')
#             pos_true.append(np.mean(hrGT))
#             pos_pred.append(np.mean(hrES))
#
#     elif sub_folders == 'UBFC2':
#         for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#             subjects = os.path.join(base_dir, sub_folders, folders)
#             for each_subject in os.listdir(subjects):
#                 if each_subject.endswith('.avi'):
#                     vid = os.path.join(subjects, each_subject)
#                 elif each_subject.endswith('.txt'):
#                     gt = os.path.join(subjects, each_subject)
#
#             print(vid, gt)
#             hrES = pos_framework(input_video=vid, dataset='UBFC2')
#             hrGT = pos_ubfc2(ground_truth_file=gt)
#             print(len(hrGT), len(hrES))
#             print('')
#             pos_true.append(np.mean(hrGT))
#             pos_pred.append(np.mean(hrES))
#
# print(mean_absolute_error(pos_true, pos_pred))
# print(mean_absolute_error(pos_true[8:], pos_pred[8:]))
# print(pos_true)
# print(pos_pred)

# 7.673822303124896
# 7.68994833019809
# [75.33707865168539, 79.26136363636364, 91.55172413793103, 63.3974358974359, 69.8913043478261, 74.6470588235294, 90.0, 101.6260162601626, 108.01886792452831, 94.2741935483871, 105.1063829787234, 99.05660377358491, 112.02898550724638, 108.40579710144928, 110.3623188405797, 123.76811594202898, 68.75, 103.26086956521739, 69.92647058823529, 115.8695652173913, 93.07142857142857, 86.78571428571429, 120.72463768115942, 125.8695652173913, 102.68115942028986, 65.97014925373135, 106.54411764705883, 115.0, 98.33333333333333, 109.38775510204081, 98.04347826086956, 78.76811594202898, 105.43478260869566, 116.83823529411765, 116.44927536231884, 103.64285714285714, 119.28571428571429, 57.10144927536232, 109.14285714285714, 84.92753623188406, 86.15942028985508, 101.01449275362319, 95.14285714285714, 99.48529411764706, 82.89855072463769, 110.14492753623189, 97.20588235294117, 105.94202898550725, 90.8955223880597, 87.82608695652173]
# [81.28048780487805, 83.33333333333333, 94.4375, 69.30555555555556, 76.78571428571429, 88.97435897435898, 98.58024691358025, 113.7280701754386, 110.37735849056604, 94.2741935483871, 104.25531914893617, 100.28301886792453, 108.98550724637681, 100.0, 96.23188405797102, 99.20289855072464, 68.01470588235294, 85.0, 99.48529411764706, 90.07246376811594, 85.92857142857143, 82.57142857142857, 91.3768115942029, 115.8695652173913, 87.53623188405797, 64.17910447761194, 105.51470588235294, 119.9, 98.62745098039215, 91.0204081632653, 93.98550724637681, 78.6231884057971, 98.98550724637681, 117.72058823529412, 91.59420289855072, 103.85714285714286, 117.64285714285714, 62.25, 96.28571428571429, 84.85507246376811, 86.3768115942029, 87.7536231884058, 95.21428571428571, 99.11764705882354, 74.56521739130434, 104.92753623188406, 88.97058823529412, 97.02898550724638, 90.82089552238806, 88.6231884057971]


# base_dir = r'C:\Users\Admin\Desktop\LGI-PPG Dataset\LGI_PPGI'
# for sub_folders in os.listdir(base_dir):
#     for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#         subjects = os.path.join(base_dir, sub_folders, folders)
#         for each_subject in os.listdir(subjects):
#             if each_subject.endswith('.avi'):
#                 vid = os.path.join(subjects, each_subject)
#             elif each_subject.endswith('cms50_stream_handler.xml'):
#                 gt = os.path.join(subjects, each_subject)
#
#         print(vid, gt)
#         hrES = pos_framework(input_video=vid, dataset='LGI_PPGI')
#         hrGT = pos_lgi_ppgi(ground_truth_file=gt)
#         print(len(hrGT), len(hrES))
#         print('')
#         pos_true.append(np.mean(hrGT))
#         pos_pred.append(np.mean(hrES))
#
# print(pos_true)
# print(pos_pred)
# print(mean_absolute_error(pos_true, pos_pred))
# print(f"gym: {mean_absolute_error(pos_true[0:6], pos_pred[0:6])}")
# print(f"resting: {mean_absolute_error(pos_true[6:12], pos_pred[6:12])}")
# print(f"rotation: {mean_absolute_error(pos_true[12:18], pos_pred[12:18])}")
# print(f"talk: {mean_absolute_error(pos_true[18:24], pos_pred[18:24])}")

# [111.41439205955335, 103.68501529051987, 113.63829787234043, 115.85483870967742, 105.41176470588235, 100.38636363636364, 65.77922077922078, 60.714285714285715, 60.324675324675326, 78.6, 73.1159420289855, 52.0, 68.41772151898734, 58.26086956521739, 60.88235294117647, 82.53333333333333, 76.04166666666667, 51.40845070422535, 73.22916666666667, 63.0, 80.29411764705883, 87.7439024390244, 76.1875, 74.52380952380952]
# [75.01256281407035, 90.51083591331269, 81.39380530973452, 81.51960784313725, 84.74164133738601, 78.94675925925925, 66.46666666666667, 61.39705882352941, 60.74324324324324, 78.42465753424658, 73.48484848484848, 87.22222222222223, 68.28947368421052, 68.88059701492537, 60.97560975609756, 91.59722222222223, 78.84057971014492, 87.82608695652173, 74.10526315789474, 60.810810810810814, 65.0, 78.76543209876543, 75.44871794871794, 93.13253012048193]
# 12.56782351901217
# gym: 26.377576632906166
# resting: 6.259209676516022
# rotation: 9.853611714011569
# talk: 7.78089605261492

### END TEST SECTION