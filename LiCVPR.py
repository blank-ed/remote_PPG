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

    raw_green_sig = extract_raw_sig(input_video, framework='LiCVPR', width=1, height=1)  # Get the raw green signal

    if dataset is None:
        fps = get_fps(input_video)  # find the fps of the video
    elif dataset == 'UBFC1' or dataset == 'UBFC2':
        fps = 30
    else:
        assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                      "types: 'UBFC1', 'UBFC2'. If you are using your own dataset, enter 'None' "

    if len(raw_green_sig) != len(raw_bg_green_signal):
        raw_bg_green_signal = raw_bg_green_signal[abs(len(raw_green_sig)-len(raw_bg_green_signal)):]

    # Apply the Illumination Rectification filter
    g_ir = rectify_illumination(face_color=np.array(raw_green_sig), bg_color=np.array(raw_bg_green_signal))

    # Apply the non-rigid motion elimination
    motion_eliminated = non_rigid_motion_elimination(signal=g_ir.tolist(), segment_length=1, fps=fps, threshold=0.05)

    # Filter the signal using detrending, moving average and bandpass filter
    detrended = detrending_filter(signal=np.array(motion_eliminated), Lambda=300)
    moving_average = moving_average_filter(signal=detrended, window_size=3)
    bp_filtered = fir_bp_filter(moving_average, fps=fps, low=0.7, high=4)

    if heart_rate_calculation_mode == 'continuous':
        windowed_pulse_sig = moving_window(sig=bp_filtered, fps=fps, window_size=hr_interval, increment=hr_interval)
        hr = []

        for each_signal_window in windowed_pulse_sig:
            frequencies, psd = welch(each_signal_window, fs=fps, nperseg=256, nfft=2048)

            first = np.where(frequencies > 0.7)[0]
            last = np.where(frequencies < 4)[0]
            first_index = first[0]
            last_index = last[-1]
            range_of_interest = range(first_index, last_index + 1, 1)
            max_idx = np.argmax(psd[range_of_interest])
            f_max = frequencies[range_of_interest[max_idx]]
            hr.append(f_max * 60.0)

    elif heart_rate_calculation_mode == 'average':
        frequencies, psd = welch(bp_filtered, fs=fps, nperseg=256, nfft=2048)

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
            frequencies, psd = welch(each_signal_window, fs=sampling_frequency, nperseg=256, nfft=2048)

            first = np.where(frequencies > 0.7)[0]
            last = np.where(frequencies < 4)[0]
            first_index = first[0]
            last_index = last[-1]
            range_of_interest = range(first_index, last_index + 1, 1)
            max_idx = np.argmax(psd[range_of_interest])
            f_max = frequencies[range_of_interest[max_idx]]
            hrGT.append(f_max * 60.0)

    elif heart_rate_calculation_mode == 'average':
        frequencies, psd = welch(bp_filtered, fs=sampling_frequency, nperseg=256, nfft=2048)

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
            frequencies, psd = welch(each_signal_window, fs=sampling_frequency, nperseg=256, nfft=2048)

            first = np.where(frequencies > 0.7)[0]
            last = np.where(frequencies < 4)[0]
            first_index = first[0]
            last_index = last[-1]
            range_of_interest = range(first_index, last_index + 1, 1)
            max_idx = np.argmax(psd[range_of_interest])
            f_max = frequencies[range_of_interest[max_idx]]
            hrGT.append(f_max * 60.0)

    elif heart_rate_calculation_mode == 'average':
        frequencies, psd = welch(bp_filtered, fs=sampling_frequency, nperseg=256, nfft=2048)

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


for sub_folders in os.listdir(base_dir):
    if sub_folders == 'UBFC1':
        for enum, folders in enumerate(os.listdir(os.path.join(base_dir, sub_folders))):
            subjects = os.path.join(base_dir, sub_folders, folders)
            for each_subject in os.listdir(subjects):
                if each_subject.endswith('.avi'):
                    vid = os.path.join(subjects, each_subject)
                elif each_subject.endswith('.xmp'):
                    gt = os.path.join(subjects, each_subject)

            print(vid, gt)

            # raw_bg_signal = extract_raw_bg_signal(vid, color='g')
            hrES = licvpr_framework(input_video=vid, raw_bg_green_signal=raw_bg_signals_ubfc1[enum],
                                    heart_rate_calculation_mode='average', hr_interval=None, dataset='UBFC1')
            hrGT = licvpr_ubfc1(ground_truth_file=gt, heart_rate_calculation_mode='average', sampling_frequency=60,
                                hr_interval=None)
            # print(len(hrGT), len(hrES))
            print('')
            licvpr_true.append(np.mean(hrGT))
            licvpr_pred.append(np.mean(hrES))

    elif sub_folders == 'UBFC2':
        for enum, folders in enumerate(os.listdir(os.path.join(base_dir, sub_folders))):
            subjects = os.path.join(base_dir, sub_folders, folders)
            for each_subject in os.listdir(subjects):
                if each_subject.endswith('.avi'):
                    vid = os.path.join(subjects, each_subject)
                elif each_subject.endswith('.txt'):
                    gt = os.path.join(subjects, each_subject)

            print(vid, gt)
            # raw_bg_signal = extract_raw_bg_signal(vid, color='g')
            hrES = licvpr_framework(input_video=vid, raw_bg_green_signal=raw_bg_signals_ubfc2[enum],
                                    heart_rate_calculation_mode='average', hr_interval=None, dataset='UBFC2')
            hrGT = licvpr_ubfc2(ground_truth_file=gt, heart_rate_calculation_mode='average', sampling_frequency=30,
                                hr_interval=None)
            # print(len(hrGT), len(hrES))
            print('')
            licvpr_true.append(np.mean(hrGT))
            licvpr_pred.append(np.mean(hrES))

print(licvpr_true)
print(licvpr_pred)
print(mean_absolute_error(licvpr_true, licvpr_pred))
print(mean_absolute_error(licvpr_true[8:], licvpr_pred[8:]))

true = [75.5859375, 79.1015625, 91.40625, 61.5234375, 68.5546875, 73.828125, 89.6484375, 105.46875, 109.86328125, 94.04296875, 113.37890625, 99.31640625, 113.37890625, 108.10546875, 117.7734375, 125.68359375, 68.5546875, 111.62109375, 72.0703125, 116.015625, 94.04296875, 87.01171875, 126.5625, 132.71484375, 105.46875, 67.67578125, 94.04296875, 114.2578125, 99.31640625, 113.37890625, 100.1953125, 78.22265625, 115.13671875, 116.89453125, 116.015625, 115.13671875, 122.16796875, 61.5234375, 109.86328125, 84.375, 87.890625, 100.1953125, 97.55859375, 100.1953125, 80.859375, 111.62109375, 93.1640625, 109.86328125, 90.52734375, 87.01171875]
pred = [59.765625, 70.3125, 67.67578125, 64.16015625, 74.70703125, 79.98046875, 80.859375, 89.6484375, 103.7109375, 72.94921875, 87.01171875, 101.07421875, 63.28125, 70.3125, 77.34375, 84.375, 86.1328125, 67.67578125, 68.5546875, 72.0703125, 69.43359375, 64.16015625, 60.64453125, 72.94921875, 102.83203125, 64.16015625, 54.4921875, 108.984375, 58.0078125, 88.76953125, 59.765625, 59.765625, 65.91796875, 57.12890625, 63.28125, 136.23046875, 67.67578125, 96.6796875, 65.0390625, 69.43359375, 75.5859375, 61.5234375, 69.43359375, 76.46484375, 72.94921875, 61.5234375, 66.796875, 99.31640625, 86.1328125, 60.64453125]

# 26.630859375
# 29.610770089285715
