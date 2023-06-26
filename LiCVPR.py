"""

This module contains the framework implemented by
https://openaccess.thecvf.com/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf
also known as LiCVPR rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""

from remote_PPG.utils import *
from remote_PPG.filters import *
from scipy.signal import welch
import numpy as np
from scipy.stats import cumfreq


def licvpr_framework(input_video, raw_bg_green_signal, heart_rate_calculation_mode='average', hr_interval=None, dataset=None):
    """
    :param input_video:
        This takes in an input video file
    :param raw_bg_green_signal:
        Extract the raw background signal separately. There is an error with the latest mediapipe library.
        To extract the raw background signal separately, do:

        from remote_PPG.utils import *
        raw_bg_signal = extract_raw_bg_signal(input_video, color='g')

    :param heart_rate_calculation_mode:
        The mode of heart rate calculation to be used. It can be set to one of the following:
        - 'average': The function computes the average heart rate over the entire duration of the video.
        - 'continuous': The function computes the heart rate at regular specified intervals throughout the video.
        The default value is 'average'.
    :param hr_interval
        This parameter is used when 'heart_rate_calculation_mode' is set to 'continuous'. It specifies the time interval
        (in seconds) at which the heart rate is calculated throughout the video. If not set, a default interval of
        10 seconds is used.
    :return:
        Returns the estimated heart rate of the input video based on LiCVPR framework
    """

    if hr_interval is None:
        hr_interval = 10

    # raw_green_sig = extract_raw_sig(input_video, framework='LiCVPR', width=1, height=1)  # Get the raw green signal
    raw_green_sig = extract_raw_sig(input_video, framework='GREEN', ROI_type='ROI_I')  # MOD ---------------------------
    raw_green_sig = np.array(raw_green_sig)[:, 1]

    if dataset is None:
        fps = get_fps(input_video)  # find the fps of the video
    elif dataset == 'UBFC1' or dataset == 'UBFC2':
        fps = 30
    elif dataset == 'LGI_PPGI':
        fps = 25
    else:
        assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                      "types: 'UBFC1', 'UBFC2', or 'LGI_PPGI'. If you are using your own dataset, enter 'None' "

    if len(raw_green_sig) != len(raw_bg_green_signal):
        raw_bg_green_signal = raw_bg_green_signal[abs(len(raw_green_sig)-len(raw_bg_green_signal)):]

    # Apply the Illumination Rectification filter
    g_ir = rectify_illumination(face_color=np.array(raw_green_sig), bg_color=np.array(raw_bg_green_signal))
    g_ir = fir_bp_filter(g_ir, fps=fps, low=0.7, high=4)  # MOD --------------------------------------------------------

    # Apply the non-rigid motion elimination
    motion_eliminated = non_rigid_motion_elimination(signal=g_ir.tolist(), segment_length=1, fps=fps, threshold=0.05)
    motion_eliminated = fir_bp_filter(motion_eliminated, fps=fps, low=0.7, high=4)  # MOD ------------------------------

    # Filter the signal using detrending, moving average and bandpass filter
    detrended = detrending_filter(signal=np.array(motion_eliminated), Lambda=300)
    moving_average = moving_average_filter(signal=detrended, window_size=3)
    bp_filtered = fir_bp_filter(moving_average, fps=fps, low=0.7, high=4)

    if heart_rate_calculation_mode == 'continuous':
        windowed_pulse_sig = moving_window(sig=bp_filtered, fps=fps, window_size=hr_interval, increment=hr_interval)
        hr = []

        for each_signal_window in windowed_pulse_sig:
            frequencies, psd = welch(each_signal_window, fs=fps, nperseg=len(each_signal_window), nfft=4096)

            first = np.where(frequencies > 0.7)[0]
            last = np.where(frequencies < 4)[0]
            first_index = first[0]
            last_index = last[-1]
            range_of_interest = range(first_index, last_index + 1, 1)
            max_idx = np.argmax(psd[range_of_interest])
            f_max = frequencies[range_of_interest[max_idx]]
            hr.append(f_max * 60.0)

    elif heart_rate_calculation_mode == 'average':
        frequencies, psd = welch(bp_filtered, fs=fps, nperseg=512, nfft=4096)

        first = np.where(frequencies > 0.7)[0]
        last = np.where(frequencies < 4)[0]
        first_index = first[0]
        last_index = last[-1]
        range_of_interest = range(first_index, last_index + 1, 1)
        max_idx = np.argmax(psd[range_of_interest])
        f_max = frequencies[range_of_interest[max_idx]]
        hr = f_max * 60.0

    else:
        assert False, "Invalid heart rate calculation mode type. Please choose one of the valid available types " \
                       "types: 'continuous', or 'average' "

    return hr


def rectify_illumination(face_color, bg_color, step=0.003, length=3):
    """performs illumination rectification.

    The correction is made on the face green values using the background green values,
    to remove global illumination variations in the face green color signal.

    Parameters
    ----------
    face_color: numpy.ndarray
      The mean green value of the face across the video sequence.
    bg_color: numpy.ndarray
      The mean green value of the background across the video sequence.
    step: float
      Step size in the filter's weight adaptation.
    length: int
      Length of the filter.

    Returns
    -------
    rectified color: numpy.ndarray
      The mean green values of the face, corrected for illumination variations.

    """
    # first pass to find the filter coefficients
    # - y: filtered signal
    # - e: error (aka difference between face and background)
    # - w: filter coefficient(s)
    yg, eg, wg = nlms(bg_color, face_color, length, step)

    # second pass to actually filter the signal, using previous weights as initial conditions
    # the second pass just filters the signal and does NOT update the weights !
    yg2, eg2, wg2 = nlms(bg_color, face_color, length, step, initCoeffs=wg, adapt=False)
    return eg2


def nlms(signal, desired_signal, n_filter_taps, step, initCoeffs=None, adapt=True):
    """Normalized least mean square filter.

    Based on adaptfilt 0.2:  https://pypi.python.org/pypi/adaptfilt/0.2

    Parameters
    ----------
    signal: numpy.ndarray
      The signal to be filtered.
    desired_signal: numpy.ndarray
      The target signal.
    n_filter_taps: int
      The number of filter taps (related to the filter order).
    step: float
      Adaptation step for the filter weights.
    initCoeffs: numpy.ndarray
      Initial values for the weights. Defaults to zero.
    adapt: bool
      If True, adapt the filter weights. If False, only filters.

    Returns
    -------
    y: numpy.ndarray
      The filtered signal.

    e: numpy.ndarray
      The error signal (difference between filtered and desired)

    w: numpy.ndarray
      The found weights of the filter.

    """
    eps = 0.001
    number_of_iterations = len(signal) - n_filter_taps + 1
    if initCoeffs is None:
        initCoeffs = np.zeros(n_filter_taps)

    # Initialization
    y = np.zeros(number_of_iterations)  # Filter output
    e = np.zeros(number_of_iterations)  # Error signal
    w = initCoeffs  # Initial filter coeffs

    # Perform filtering
    errors = []
    for n in range(number_of_iterations):
        x = np.flipud(signal[n:(n + n_filter_taps)])  # Slice to get view of M latest datapoints
        y[n] = np.dot(x, w)
        e[n] = desired_signal[n + n_filter_taps - 1] - y[n]
        errors.append(e[n])

        if adapt:
            normFactor = 1. / (np.dot(x, x) + eps)
            w = w + step * normFactor * x * e[n]
            y[n] = np.dot(x, w)

    return y, e, w


def non_rigid_motion_elimination(signal, segment_length, fps, threshold=0.05):
    """
    :param signal:
        Input signal to segment
    :param segment_length:
        The length of each segment in seconds (s)
    :param fps:
        The frame rate of the video
    :param threshold:
        The cutoff threshold of the segments based on their standard deviation
    :return:
        Returns motion eliminated signal
    """

    # Divide the signal into m segments of the same length
    segments = []
    for i in range(0, len(signal), int(segment_length * fps)):
        end = i + int(segment_length * fps)
        if end > len(signal):
            end_segment_index = i
            break
        segments.append(signal[i:end])
    else:
        end_segment_index = len(segments) * fps

    sd = np.array([np.std(segment) for segment in segments])  # Find the standard deviation of each segment

    # calculate the cumulative frequency of the data, which is effectively the CDF
    # 'numbins' should be set to the number of unique standard deviations
    a = cumfreq(sd, numbins=len(np.unique(sd)))

    # get the value that is the cut-off for the top 5% which is done by finding the smallest standard deviation that
    # has a cumulative frequency greater than 95% of the data
    cut_off_index = np.argmax(a.cumcount >= len(sd) * (1 - threshold))
    cut_off_value = a.lowerlimit + np.linspace(0, a.binsize * a.cumcount.size, a.cumcount.size)[cut_off_index]

    # create a mask where True indicates the value is less than the cut-off
    mask = sd < cut_off_value

    # get the new list of segments excluding the top 5% of highest SD
    segments_95_percent = np.concatenate((np.array(segments)[mask]), axis=None)

    # Add residual signal (leftover original signal) due to segmentation if there is any
    if len(signal) != end_segment_index:
        residual_signal = np.array(signal[end_segment_index:len(signal)])
        motion_eliminated = np.concatenate((segments_95_percent, residual_signal), axis=None)
    else:
        motion_eliminated = segments_95_percent

    return motion_eliminated


# Test Section

import pandas as pd
import os
from sklearn.metrics import mean_absolute_error
from scipy.signal import welch

def licvpr_ubfc1(ground_truth_file, heart_rate_calculation_mode='average', sampling_frequency=60, hr_interval=None):
    gtdata = pd.read_csv(ground_truth_file, header=None)
    gtTrace = gtdata.iloc[:, 3].tolist()
    gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
    gtHR = gtdata.iloc[:, 1]

    if hr_interval is None:
        hr_interval = 10

    # Filter the signal using detrending, moving average and bandpass filter
    detrended = detrending_filter(signal=np.array(gtTrace), Lambda=300)
    moving_average = moving_average_filter(signal=detrended, window_size=3)
    bp_filtered = fir_bp_filter(moving_average, fps=sampling_frequency, low=0.7, high=4)

    if heart_rate_calculation_mode == 'continuous':
        windowed_pulse_sig = moving_window(sig=bp_filtered, fps=sampling_frequency, window_size=hr_interval,
                                           increment=hr_interval)
        hrGT = []

        for each_signal_window in windowed_pulse_sig:
            frequencies, psd = welch(each_signal_window, fs=sampling_frequency, nperseg=len(each_signal_window), nfft=4096)

            first = np.where(frequencies > 0.7)[0]
            last = np.where(frequencies < 4)[0]
            first_index = first[0]
            last_index = last[-1]
            range_of_interest = range(first_index, last_index + 1, 1)
            max_idx = np.argmax(psd[range_of_interest])
            f_max = frequencies[range_of_interest[max_idx]]
            hrGT.append(f_max * 60.0)

    elif heart_rate_calculation_mode == 'average':
        frequencies, psd = welch(bp_filtered, fs=sampling_frequency, nperseg=512, nfft=4096)

        first = np.where(frequencies > 0.7)[0]
        last = np.where(frequencies < 4)[0]
        first_index = first[0]
        last_index = last[-1]
        range_of_interest = range(first_index, last_index + 1, 1)
        max_idx = np.argmax(psd[range_of_interest])
        f_max = frequencies[range_of_interest[max_idx]]
        hrGT = f_max * 60.0

    else:
        assert False, "Invalid heart rate calculation mode type. Please choose one of the valid available types " \
                       "types: 'continuous', or 'average' "

    return hrGT


def licvpr_ubfc2(ground_truth_file, heart_rate_calculation_mode='average', sampling_frequency=30, hr_interval=None):
    gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
    gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
    gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
    gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

    if hr_interval is None:
        hr_interval = 10

    # Filter the signal using detrending, moving average and bandpass filter
    detrended = detrending_filter(signal=np.array(gtTrace), Lambda=300)
    moving_average = moving_average_filter(signal=detrended, window_size=3)
    bp_filtered = fir_bp_filter(moving_average, fps=sampling_frequency, low=0.7, high=4)

    if heart_rate_calculation_mode == 'continuous':
        windowed_pulse_sig = moving_window(sig=bp_filtered, fps=sampling_frequency, window_size=hr_interval,
                                           increment=hr_interval)
        hrGT = []

        for each_signal_window in windowed_pulse_sig:
            frequencies, psd = welch(each_signal_window, fs=sampling_frequency, nperseg=len(each_signal_window), nfft=4096)

            first = np.where(frequencies > 0.7)[0]
            last = np.where(frequencies < 4)[0]
            first_index = first[0]
            last_index = last[-1]
            range_of_interest = range(first_index, last_index + 1, 1)
            max_idx = np.argmax(psd[range_of_interest])
            f_max = frequencies[range_of_interest[max_idx]]
            hrGT.append(f_max * 60.0)

    elif heart_rate_calculation_mode == 'average':
        frequencies, psd = welch(bp_filtered, fs=sampling_frequency, nperseg=512, nfft=4096)

        first = np.where(frequencies > 0.7)[0]
        last = np.where(frequencies < 4)[0]
        first_index = first[0]
        last_index = last[-1]
        range_of_interest = range(first_index, last_index + 1, 1)
        max_idx = np.argmax(psd[range_of_interest])
        f_max = frequencies[range_of_interest[max_idx]]
        hrGT = f_max * 60.0

    else:
        assert False, "Invalid heart rate calculation mode type. Please choose one of the valid available types " \
                       "types: 'continuous', or 'average' "

    return hrGT


def licvpr_lgi_ppgi(ground_truth_file, heart_rate_calculation_mode='average', sampling_frequency=60, hr_interval=None):
    gtdata = pd.read_xml(ground_truth_file)
    gtTime = (gtdata.iloc[:, 0]).tolist()
    gtHR = gtdata.iloc[:, 1].tolist()
    gtTrace = gtdata.iloc[:, 2].tolist()

    if hr_interval is None:
        hr_interval = 10

    # Filter the signal using detrending, moving average and bandpass filter
    detrended = detrending_filter(signal=np.array(gtTrace), Lambda=300)
    moving_average = moving_average_filter(signal=detrended, window_size=3)
    bp_filtered = fir_bp_filter(moving_average, fps=sampling_frequency, low=0.7, high=4)

    if heart_rate_calculation_mode == 'continuous':
        windowed_pulse_sig = moving_window(sig=bp_filtered, fps=sampling_frequency, window_size=hr_interval,
                                           increment=hr_interval)
        hrGT = []

        for each_signal_window in windowed_pulse_sig:
            frequencies, psd = welch(each_signal_window, fs=sampling_frequency, nperseg=len(each_signal_window), nfft=4096)

            first = np.where(frequencies > 0.7)[0]
            last = np.where(frequencies < 4)[0]
            first_index = first[0]
            last_index = last[-1]
            range_of_interest = range(first_index, last_index + 1, 1)
            max_idx = np.argmax(psd[range_of_interest])
            f_max = frequencies[range_of_interest[max_idx]]
            hrGT.append(f_max * 60.0)

    elif heart_rate_calculation_mode == 'average':
        frequencies, psd = welch(bp_filtered, fs=sampling_frequency, nperseg=512, nfft=4096)

        first = np.where(frequencies > 0.7)[0]
        last = np.where(frequencies < 4)[0]
        first_index = first[0]
        last_index = last[-1]
        range_of_interest = range(first_index, last_index + 1, 1)
        max_idx = np.argmax(psd[range_of_interest])
        f_max = frequencies[range_of_interest[max_idx]]
        hrGT = f_max * 60.0

    else:
        assert False, "Invalid heart rate calculation mode type. Please choose one of the valid available types " \
                       "types: 'continuous', or 'average' "

    return hrGT


import ast

licvpr_true = []
licvpr_pred = []
# base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'

raw_bg_signals_ubfc1 = []
with open('UBFC1.txt', 'r') as f:
    lines = f.readlines()
    for x in lines:
        raw_bg_signals_ubfc1.append(ast.literal_eval(x))

raw_bg_signals_ubfc2 = []
with open('UBFC2.txt', 'r') as f:
    lines = f.readlines()
    for x in lines:
        raw_bg_signals_ubfc2.append(ast.literal_eval(x))

# # raw_bg_signal_ubfc1 = []
# raw_bg_signal_ubfc2 = []
#
# for sub_folders in os.listdir(base_dir):
#     # if sub_folders == 'UBFC1':
#     #     for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#     #         subjects = os.path.join(base_dir, sub_folders, folders)
#     #         for each_subject in os.listdir(subjects):
#     #             if each_subject.endswith('.avi'):
#     #                 vid = os.path.join(subjects, each_subject)
#     #                 print(vid)
#     #                 bg_sig = extract_raw_bg_signal(input_video=vid)
#     #                 print(bg_sig)
#     #                 raw_bg_signal_ubfc1.append(bg_sig)
#
#     if sub_folders == 'UBFC2':
#         for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#             subjects = os.path.join(base_dir, sub_folders, folders)
#             for each_subject in os.listdir(subjects):
#                 if each_subject.endswith('.avi'):
#                     vid = os.path.join(subjects, each_subject)
#                     print(vid)
#                     bg_sig = extract_raw_bg_signal(input_video=vid)
#                     # print(bg_sig)
#                     raw_bg_signal_ubfc2.append(bg_sig)
#
# print(raw_bg_signal_ubfc2)
#
# with open('UBFC2.txt', 'w') as f:
#     f.write(str(raw_bg_signal_ubfc2))
#     f.write('\n')


# for sub_folders in os.listdir(base_dir):
#     # if sub_folders == 'UBFC1':
#     #     for enum, folders in enumerate(os.listdir(os.path.join(base_dir, sub_folders))):
#     #         subjects = os.path.join(base_dir, sub_folders, folders)
#     #         for each_subject in os.listdir(subjects):
#     #             if each_subject.endswith('.avi'):
#     #                 vid = os.path.join(subjects, each_subject)
#     #             elif each_subject.endswith('.xmp'):
#     #                 gt = os.path.join(subjects, each_subject)
#     #
#     #         print(vid, gt)
#     #
#     #         # raw_bg_signal = extract_raw_bg_signal(vid, color='g')
#     #         hrES = licvpr_framework(input_video=vid, raw_bg_green_signal=raw_bg_signals_ubfc1[enum],
#     #                                 heart_rate_calculation_mode='continuous', hr_interval=None, dataset='UBFC1')
#     #         hrGT = licvpr_ubfc1(ground_truth_file=gt, heart_rate_calculation_mode='continuous', sampling_frequency=60,
#     #                             hr_interval=None)
#     #         # print(len(hrGT), len(hrES))
#     #         print('')
#     #         licvpr_true.append(np.mean(hrGT))
#     #         licvpr_pred.append(np.mean(hrES))
#
#     if sub_folders == 'UBFC2':
#         for enum, folders in enumerate(os.listdir(os.path.join(base_dir, sub_folders))):
#             subjects = os.path.join(base_dir, sub_folders, folders)
#             for each_subject in os.listdir(subjects):
#                 if each_subject.endswith('.avi'):
#                     vid = os.path.join(subjects, each_subject)
#                 elif each_subject.endswith('.txt'):
#                     gt = os.path.join(subjects, each_subject)
#
#             print(vid, gt)
#             # raw_bg_signal = extract_raw_bg_signal(vid, color='g')
#             hrES = licvpr_framework(input_video=vid, raw_bg_green_signal=raw_bg_signals_ubfc2[enum],
#                                     heart_rate_calculation_mode='continuous', hr_interval=None, dataset='UBFC2')
#             hrGT = licvpr_ubfc2(ground_truth_file=gt, heart_rate_calculation_mode='continuous', sampling_frequency=30,
#                                 hr_interval=None)
#             # print(len(hrGT), len(hrES))
#             print('')
#             licvpr_true.append(np.mean(hrGT))
#             licvpr_pred.append(np.mean(hrES))
#
# print(licvpr_true)
# print(licvpr_pred)
# print(mean_absolute_error(licvpr_true, licvpr_pred))
# print(mean_absolute_error(licvpr_true[8:], licvpr_pred[8:]))
#
# [75.68359375, 79.1015625, 91.015625, 61.962890625, 69.08203125, 73.046875, 90.4296875, 106.75330528846153, 107.373046875, 97.9248046875, 107.2265625, 98.2177734375, 112.75111607142857, 109.04715401785714, 110.67940848214286, 124.23967633928571, 68.80580357142857, 107.97991071428571, 75.33482142857143, 115.32505580357143, 93.98018973214286, 86.07003348214286, 120.47293526785714, 125.80915178571429, 102.392578125, 65.91796875, 106.28487723214286, 115.224609375, 99.4921875, 115.048828125, 98.56305803571429, 78.59933035714286, 106.03376116071429, 114.50892857142857, 114.697265625, 104.33872767857143, 117.83621651785714, 60.707310267857146, 110.7421875, 84.814453125, 85.88169642857143, 101.45089285714286, 94.04296875, 99.06529017857143, 82.74274553571429, 109.48660714285714, 97.74693080357143, 110.17717633928571, 90.52734375, 88.01618303571429]
# [70.587158203125, 84.759521484375, 83.38623046875, 82.30329241071429, 74.267578125, 88.5498046875, 96.624755859375, 91.96555397727273, 95.09765625, 97.705078125, 108.193359375, 86.484375, 67.17354910714286, 97.74693080357143, 69.81026785714286, 75.83705357142857, 78.28543526785714, 92.97572544642857, 75.20926339285714, 75.89983258928571, 83.55887276785714, 92.47349330357143, 77.46930803571429, 104.58984375, 110.86774553571429, 87.57672991071429, 78.41099330357143, 89.736328125, 91.93359375, 72.421875, 83.99832589285714, 77.21819196428571, 81.04771205357143, 79.541015625, 76.27650669642857, 85.50502232142857, 94.29408482142857, 113.1591796875, 78.662109375, 105.59430803571429, 103.39704241071429, 72.38420758928571, 83.935546875, 100.38364955357143, 93.310546875, 74.76981026785714, 67.36188616071429, 97.18191964285714, 92.59905133928571, 95.17299107142857]
# 18.96827525501842
# 20.667101801658163

# Improved (using forehead ROI, adding bp filter after g_ir and motion elimination stages)
# [106.435546875, 98.701171875, 107.77587890625, 98.61328125, 113.232421875, 109.6435546875, 111.4013671875, 125.0244140625, 68.7744140625, 106.4208984375, 75.6591796875, 115.9423828125, 94.1162109375, 85.7666015625, 119.7509765625, 127.6611328125, 100.341796875, 64.5263671875, 108.9111328125, 115.46630859375, 99.64599609375, 115.13671875, 97.265625, 78.515625, 106.93359375, 115.9423828125, 114.84375, 103.7841796875, 116.455078125, 61.0107421875, 110.3759765625, 84.4482421875, 87.01171875, 101.513671875, 93.5302734375, 97.9248046875, 82.6171875, 110.44921875, 96.97265625, 109.7900390625, 90.6005859375, 88.4033203125]
# [106.5, 99.6, 105.00000000000001, 130.5, 112.0, 108.0, 94.0, 72.0, 67.0, 103.0, 71.0, 99.0, 95.0, 86.0, 122.0, 113.0, 100.0, 66.0, 107.0, 115.5, 102.0, 114.0, 103.0, 79.0, 98.0, 116.0, 115.0, 106.0, 117.0, 66.0, 99.00000000000001, 86.0, 86.0, 94.0, 95.0, 91.0, 83.0, 111.0, 92.0, 112.0, 102.0, 89.0]
# 5.573309616815475

# raw_bg_signal_lgi_ppgi = []
# base_dir = r'C:\Users\Admin\Desktop\LGI-PPG Dataset\LGI_PPGI'
# for sub_folders in os.listdir(base_dir):
#     for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#         subjects = os.path.join(base_dir, sub_folders, folders)
#         for each_subject in os.listdir(subjects):
#             if each_subject.endswith('.avi'):
#                 vid = os.path.join(subjects, each_subject)
#                 print(vid)
#                 bg_sig = extract_raw_bg_signal(input_video=vid)
#                 print(bg_sig)
#                 raw_bg_signal_lgi_ppgi.append(bg_sig)
#
# print(raw_bg_signal_lgi_ppgi)
# with open('LGI_PPGI.txt', 'w') as f:
#     f.write(str(raw_bg_signal_lgi_ppgi))
#     f.write('\n')

# raw_bg_signals_lgi_ppgi = []
# with open('LGI_PPGI.txt', 'r') as f:
#     lines = f.readlines()
#     for x in lines:
#         raw_bg_signals_lgi_ppgi.append(ast.literal_eval(x))
#
# i = 0
# base_dir = r'C:\Users\Admin\Desktop\LGI-PPG Dataset\LGI_PPGI'
# for sub_folders in os.listdir(base_dir):
#     for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#         subjects = os.path.join(base_dir, sub_folders, folders)
#         for each_subject in os.listdir(subjects):
#             if each_subject.endswith('.avi'):
#                 vid = os.path.join(subjects, each_subject)
#             elif each_subject.endswith('cms50_stream_handler.xml'):
#                 gt = os.path.join(subjects, each_subject)
#         print(vid, gt)
#         hrES = licvpr_framework(input_video=vid, raw_bg_green_signal=raw_bg_signals_lgi_ppgi[i],
#                                 heart_rate_calculation_mode='continuous', hr_interval=None, dataset='LGI_PPGI')
#         hrGT = licvpr_lgi_ppgi(ground_truth_file=gt, heart_rate_calculation_mode='continuous', sampling_frequency=30,
#                             hr_interval=None)
#         print('')
#         licvpr_true.append(np.mean(hrGT))
#         licvpr_pred.append(np.mean(hrES))
#         i += 1
#
# print(licvpr_true)
# print(licvpr_pred)
# print(mean_absolute_error(licvpr_true, licvpr_pred))
# print(f"gym: {mean_absolute_error(licvpr_true[0:6], licvpr_pred[0:6])}")
# print(f"resting: {mean_absolute_error(licvpr_true[6:12], licvpr_pred[6:12])}")
# print(f"rotation: {mean_absolute_error(licvpr_true[12:18], licvpr_pred[12:18])}")
# print(f"talk: {mean_absolute_error(licvpr_true[18:24], licvpr_pred[18:24])}")

# [65.27235243055556, 63.01491477272727, 60.64453125, 66.08807963709677, 63.69485294117647, 67.37116033380681, 65.863037109375, 60.83286830357143, 60.0677490234375, 78.955078125, 72.91782924107143, 53.466796875, 68.9666748046875, 69.52776227678571, 61.083984375, 83.73046875, 75.3515625, 55.496651785714285, 73.20363898026316, 63.515625, 77.60225183823529, 87.94232536764706, 77.1514892578125, 78.94646139705883]
# [81.26027960526316, 94.70687373991936, 71.59423828125, 100.75851966594827, 77.20184326171875, 92.3655440167683, 80.9783935546875, 107.40443638392857, 84.64704241071429, 81.87430245535714, 130.68498883928572, 46.718052455357146, 86.24267578125, 58.384486607142854, 82.4432373046875, 98.61537388392857, 71.09723772321429, 60.738699776785715, 74.78841145833333, 91.34347098214286, 70.7244873046875, 82.3974609375, 82.672119140625, 90.6829833984375]
# 17.448123343851805
# gym: 21.966901200917494
# resting: 25.61689104352679
# rotation: 12.359967912946425
# talk: 9.848733218016507

# 9.772221935529775
# gym: 19.99033675697143
# resting: 1.3706752232142847
# rotation: 8.205664777930403
# talk: 9.522210984002976