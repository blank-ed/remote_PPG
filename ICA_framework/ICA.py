"""

This module contains the framework implemented by https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-10-10762&id=199381
which is also known as ICA rPPG by other research papers. This is the closest implementation of the original framework
that has been proposed.

"""

from remote_PPG.utils import *
from remote_PPG.filters import *
from remote_PPG.ICA_framework.jadeR import jadeR as jadeR
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def ica_framework(input_video, comp=1, hr_change_threshold=12, dataset=None):
    """
    :param input_video:
        This takes in an input video file
    :param comp:
        Output ICA component to be selected. From literature, the second component is selected since
        it typically contains a strong plethysmographic signal
    :param hr_change_threshold:
        The threhold value change between the previous determined HR value and the next HR value.
        If the difference between them is greater than the threshold, then the next highest power
        and its corresponding frequency (HR value) is determined
    :return:
        Returns the estimated heart rate of the input video based on ICA framework
    """

    raw_sig = extract_raw_sig(input_video, framework='ICA', width=0.6, height=1)  # get the raw RGB signals
    if dataset is None:
        fps = get_fps(input_video)  # find the fps of the video
    elif dataset == 'UBFC1' or dataset == 'UBFC2':
        fps = 30
    else:
        assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                     "types: 'UBFC1', 'UBFC2'. If you are using your own dataset, enter 'None' "


    # signal windowing with 96.7% overlap
    windowed_sig = moving_window(sig=raw_sig, fps=fps, window_size=30, increment=1)
    hrES = []

    prev_hr = None  # Previous HR value
    for sig in windowed_sig:
        normalized = normalize(sig, framework='ICA')  # normalize the windowed signal

        # Apply JADE ICA algorithm and select the second component
        W = jadeR(normalized, m=3)
        bvp = np.array(np.dot(W, normalized))
        bvp = bvp[comp].flatten()
        bvp = fir_bp_filter(signal=bvp, fps=fps, low=0.75, high=4.0)

        # Compute the positive frequencies and the corresponding power spectrum
        freqs = rfftfreq(len(bvp), d=1 / fps)
        power_spectrum = np.abs(rfft(bvp)) ** 2

        # Find the maximum peak between 0.75 Hz and 4 Hz
        mask = (freqs >= 0.75) & (freqs <= 4)
        filtered_power_spectrum = power_spectrum[mask]
        filtered_freqs = freqs[mask]

        peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks
        peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
        peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers

        # For the first previous HR value
        if prev_hr is None:
            # Find the highest peak
            max_peak_index = np.argmax(peak_powers)
            max_peak_frequency = peak_freqs[max_peak_index]

            hr = int(max_peak_frequency * 60)
            prev_hr = hr
        else:
            max_peak_index = np.argmax(peak_powers)
            max_peak_frequency = peak_freqs[max_peak_index]

            hr = int(max_peak_frequency * 60)

            # If the difference between the current pulse rate estimation and the last computed value exceeded
            # the threshold, the algorithm rejected it and searched the operational frequency range for the
            # frequency corresponding to the next highest power that met this constraint
            while abs(prev_hr - hr) >= hr_change_threshold:
                # Remove the previously wrongly determined power and frequency values from the list
                max_peak_mask = (peak_freqs == max_peak_frequency)
                peak_freqs = peak_freqs[~max_peak_mask]
                peak_powers = peak_powers[~max_peak_mask]

                #  If no frequency peaks that met the criteria were located, then
                # the algorithm retained the current pulse frequency estimation
                if len(peak_freqs) == 0:
                    hr = prev_hr
                    break

                max_peak_index = np.argmax(peak_powers)
                max_peak_frequency = peak_freqs[max_peak_index]
                hr = int(max_peak_frequency * 60)

            prev_hr = hr
        hrES.append(hr)

    return hrES


import pandas as pd
import os
from sklearn.metrics import mean_absolute_error

def ica_ubfc1(ground_truth_file, sampling_frequency=60):
    gtdata = pd.read_csv(ground_truth_file, header=None)
    gtTrace = gtdata.iloc[:, 3].tolist()
    gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
    gtHR = gtdata.iloc[:, 1]

    # signal windowing with 96.7% overlap
    windowed_sig = moving_window(sig=gtTrace, fps=sampling_frequency, window_size=30, increment=1)
    hrGT = []

    prev_hr = None  # Previous HR value
    for sig in windowed_sig:
        normalized = (np.array(sig) - np.mean(sig)) / np.std(sig)

        # Compute the positive frequencies and the corresponding power spectrum
        freqs = rfftfreq(len(normalized), d=1 / sampling_frequency)
        power_spectrum = np.abs(rfft(normalized)) ** 2

        # Find the maximum peak between 0.75 Hz and 4 Hz
        mask = (freqs >= 0.75) & (freqs <= 4)
        filtered_power_spectrum = power_spectrum[mask]
        filtered_freqs = freqs[mask]

        peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks
        peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
        peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers

        # For the first previous HR value
        if prev_hr is None:
            # Find the highest peak
            max_peak_index = np.argmax(peak_powers)
            max_peak_frequency = peak_freqs[max_peak_index]

            hr = int(max_peak_frequency * 60)
            prev_hr = hr
        else:
            max_peak_index = np.argmax(peak_powers)
            max_peak_frequency = peak_freqs[max_peak_index]

            hr = int(max_peak_frequency * 60)

            # If the difference between the current pulse rate estimation and the last computed value exceeded
            # the threshold, the algorithm rejected it and searched the operational frequency range for the
            # frequency corresponding to the next highest power that met this constraint
            while abs(prev_hr - hr) >= 12:
                # Remove the previously wrongly determined power and frequency values from the list
                max_peak_mask = (peak_freqs == max_peak_frequency)
                peak_freqs = peak_freqs[~max_peak_mask]
                peak_powers = peak_powers[~max_peak_mask]

                #  If no frequency peaks that met the criteria were located, then
                # the algorithm retained the current pulse frequency estimation
                if len(peak_freqs) == 0:
                    hr = prev_hr
                    break

                max_peak_index = np.argmax(peak_powers)
                max_peak_frequency = peak_freqs[max_peak_index]
                hr = int(max_peak_frequency * 60)

            prev_hr = hr
        hrGT.append(hr)

    return hrGT


def ica_ubfc2(ground_truth_file, sampling_frequency=30):
    gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
    gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
    gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
    gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

    # signal windowing with 96.7% overlap
    windowed_sig = moving_window(sig=gtTrace, fps=sampling_frequency, window_size=30, increment=1)
    hrGT = []

    prev_hr = None  # Previous HR value
    for sig in windowed_sig:
        normalized = (np.array(sig) - np.mean(sig)) / np.std(sig)

        # Compute the positive frequencies and the corresponding power spectrum
        freqs = rfftfreq(len(normalized), d=1 / sampling_frequency)
        power_spectrum = np.abs(rfft(normalized)) ** 2

        # Find the maximum peak between 0.75 Hz and 4 Hz
        mask = (freqs >= 0.75) & (freqs <= 4)
        filtered_power_spectrum = power_spectrum[mask]
        filtered_freqs = freqs[mask]

        peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks
        peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
        peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers

        # For the first previous HR value
        if prev_hr is None:
            # Find the highest peak
            max_peak_index = np.argmax(peak_powers)
            max_peak_frequency = peak_freqs[max_peak_index]

            hr = int(max_peak_frequency * 60)
            prev_hr = hr
        else:
            max_peak_index = np.argmax(peak_powers)
            max_peak_frequency = peak_freqs[max_peak_index]

            hr = int(max_peak_frequency * 60)

            # If the difference between the current pulse rate estimation and the last computed value exceeded
            # the threshold, the algorithm rejected it and searched the operational frequency range for the
            # frequency corresponding to the next highest power that met this constraint
            while abs(prev_hr - hr) >= 12:
                # Remove the previously wrongly determined power and frequency values from the list
                max_peak_mask = (peak_freqs == max_peak_frequency)
                peak_freqs = peak_freqs[~max_peak_mask]
                peak_powers = peak_powers[~max_peak_mask]

                #  If no frequency peaks that met the criteria were located, then
                # the algorithm retained the current pulse frequency estimation
                if len(peak_freqs) == 0:
                    hr = prev_hr
                    break

                max_peak_index = np.argmax(peak_powers)
                max_peak_frequency = peak_freqs[max_peak_index]
                hr = int(max_peak_frequency * 60)

            prev_hr = hr
        hrGT.append(hr)

    return hrGT

# MAE = []
ica_true = []
ica_pred = []
# base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
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
            hrES = ica_framework(input_video=vid, dataset='UBFC1')
            hrGT = ica_ubfc1(ground_truth_file=gt)
            print(len(hrGT), len(hrES))
            print('')
            ica_true.append(np.mean(hrGT))
            ica_pred.append(np.mean(hrES))

    elif sub_folders == 'UBFC2':
        for folders in os.listdir(os.path.join(base_dir, sub_folders)):
            subjects = os.path.join(base_dir, sub_folders, folders)
            for each_subject in os.listdir(subjects):
                if each_subject.endswith('.avi'):
                    vid = os.path.join(subjects, each_subject)
                elif each_subject.endswith('.txt'):
                    gt = os.path.join(subjects, each_subject)

            print(vid, gt)
            hrES = ica_framework(input_video=vid, dataset='UBFC2')
            hrGT = ica_ubfc2(ground_truth_file=gt)
            print(len(hrGT), len(hrES))
            print('')
            ica_true.append(np.mean(hrGT))
            ica_pred.append(np.mean(hrES))

print(ica_true)
print(ica_pred)
print(mean_absolute_error(ica_true, ica_pred))
print(mean_absolute_error(ica_true[8:], ica_pred[8:]))

# [74.30508474576271, 76.06896551724138, 92.10526315789474, 60.625, 67.75, 74.83636363636364, 90.24137931034483, 104.06451612903226, 110.0, 89.0625, 110.58823529411765, 101.30434782608695, 113.43589743589743, 110.71794871794872, 118.92307692307692, 124.97435897435898, 68.0, 108.35897435897436, 73.6842105263158, 114.46153846153847, 93.75, 87.1, 123.15384615384616, 133.17948717948718, 105.38461538461539, 64.16216216216216, 127.3157894736842, 114.0, 96.95238095238095, 111.57894736842105, 98.05128205128206, 79.33333333333333, 107.02564102564102, 118.63157894736842, 115.8974358974359, 106.65, 121.3, 57.76923076923077, 109.85, 85.28205128205128, 87.07692307692308, 100.0, 98.35, 98.63157894736842, 84.56410256410257, 111.6923076923077, 94.94736842105263, 114.35897435897436, 91.62162162162163, 87.02564102564102]
# [78.8076923076923, 152.72549019607843, 98.36, 68.0952380952381, 79.83333333333333, 79.08333333333333, 84.92156862745098, 79.61904761904762, 109.3913043478261, 93.4375, 104.82352941176471, 71.91304347826087, 113.43589743589743, 80.61538461538461, 106.71794871794872, 125.23076923076923, 69.39473684210526, 74.58974358974359, 76.57894736842105, 111.74358974358974, 93.35, 80.325, 122.1025641025641, 133.89743589743588, 69.56410256410257, 65.02702702702703, 77.36842105263158, 115.0, 135.42857142857142, 120.47368421052632, 90.35897435897436, 78.56410256410257, 103.02564102564102, 111.89473684210526, 114.25641025641026, 74.725, 122.9, 68.2, 109.775, 88.2051282051282, 81.87179487179488, 71.6923076923077, 75.15, 92.6842105263158, 79.33333333333333, 59.94871794871795, 88.78947368421052, 109.7948717948718, 121.72972972972973, 83.7948717948718]
# 12.79793460245643
# 11.878977160036426