"""

This module contains the framework implemented by https://ieeexplore.ieee.org/document/6523142 which is
also known as CHROM rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""

from scipy.signal import find_peaks, welch, windows, stft
from remote_PPG.sig_extraction_utils import *
from remote_PPG.utils import *
import numpy as np


def chrom_framework(input_video, subject_type='motion', dataset=None):
    """
    :param input_video:
        This takes in an input video file
    :param subject_type:
        - 'static':
        - 'motion':
    :param segment_length:
        To further minimize the impact of unintended motion, a segment, starting at i = is, of 500 consecutive pictures
        exhibiting the smallest amount of inter frame motion was selected from the longer video sequence
    :return:
        Returns the estimated heart rate of the input video based on CHROM framework
    """

    if subject_type == 'static':
        segment_length = 500

        frames = []
        for frame in extract_frames_yield(input_video):
            frames.append(frame)

        motion = calculate_motion(frames)  # Calculate motion between consecutive images
        i_s = find_least_motion_segment(motion, segment_length)  # Starting segment with the least inter frame motion

        raw_sig = extract_raw_sig(input_video, framework='CHROM', width=1, height=1)  # Get the raw RGB signals
        if dataset is None:
            fps = get_fps(input_video)  # find the fps of the video
        elif dataset == 'UBFC1' or dataset == 'UBFC2':
            fps = 30
        elif dataset == 'LGI_PPGI':
            fps = 25
        else:
            assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                          "types: 'UBFC1', 'UBFC2', or 'LGI_PPGI'. If you are using your own dataset, enter 'None' "

        selected_segment = raw_sig[i_s:i_s + segment_length]  # Select the segment with least inter frame motion
        normalized = normalize(selected_segment, framework='CHROM')  # Normalize the selected segment

        # Build two orthogonal chrominance signals
        Xs = 3 * normalized[0] - 2 * normalized[1]
        Ys = 1.5 * normalized[0] + normalized[1] - 1.5 * normalized[2]

        # bandpass filter Xs and Ys here
        Xf = fir_bp_filter(signal=Xs, fps=fps, low=0.67, high=4.0)
        Yf = fir_bp_filter(signal=Ys, fps=fps, low=0.67, high=4.0)

        alpha = np.std(Xf) / np.std(Yf)
        S = Xf - alpha * Yf

        frequencies, psd = welch(S, fs=30, nperseg=256, nfft=2048)

        first = np.where(frequencies > 0.7)[0]
        last = np.where(frequencies < 4)[0]
        first_index = first[0]
        last_index = last[-1]
        range_of_interest = range(first_index, last_index + 1, 1)
        max_idx = np.argmax(psd[range_of_interest])
        f_max = frequencies[range_of_interest[max_idx]]
        hr = f_max * 60.0

    elif subject_type == 'motion':

        # raw_sig = extract_raw_sig(input_video, framework='CHROM', width=1, height=1)
        raw_sig = extract_raw_sig(input_video, framework='GREEN', ROI_type='ROI_I')  # MOD ----------------------------

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

        window = moving_window(raw_sig, fps=fps, window_size=1.6, increment=0.8)

        for enum, each_window in enumerate(window):
            normalized = normalize(signal=each_window, normalize_type='mean_normalization')  # Normalize each windowed segment

            # Build two orthogonal chrominance signals
            Xs = 3 * normalized[0] - 2 * normalized[1]
            Ys = 1.5 * normalized[0] + normalized[1] - 1.5 * normalized[2]

            # bandpass filter Xs and Ys here
            Xf = fir_bp_filter(signal=Xs, fps=fps, low=0.67, high=4.0)
            Yf = fir_bp_filter(signal=Ys, fps=fps, low=0.67, high=4.0)

            alpha = np.std(Xf) / np.std(Yf)
            S = Xf - alpha * Yf

            SWin = np.multiply(S, windows.hann(len(S)))

            start = enum * (l // 2)
            end = enum * (l // 2) + l

            H[start:end] = H[start:end] + SWin

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

    else:
        assert False, "Invalid subject type. Please choose one of the valid available types " \
                      "types: 'static', or 'motion' "

    return hr


def calculate_intensity(image):
    # calculate the intensity of the image
    return np.sum(image, axis=2) / 3.0


def calculate_motion(images):
    # calculate the motion between consecutive images
    motion = []
    for i in range(len(images) - 1):
        motion.append(np.sum(np.abs(calculate_intensity(images[i]) - calculate_intensity(images[i+1]))))

    return motion


def find_least_motion_segment(motion, segment_length):
    # find the segment with the least motion
    min_motion = np.inf
    min_index = -1
    for i in range(len(motion) - segment_length):
        motion_in_segment = np.sum(motion[i:i+segment_length])
        if motion_in_segment < min_motion:
            min_motion = motion_in_segment
            min_index = i

    return min_index


### TEST SECTION

import pandas as pd
import os
from sklearn.metrics import mean_absolute_error

def chrom_ubfc1(ground_truth_file, sampling_frequency=60):
    gtdata = pd.read_csv(ground_truth_file, header=None)
    gtTrace = gtdata.iloc[:, 3].tolist()
    gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
    gtHR = gtdata.iloc[:, 1]

    normalized = np.array(gtTrace) / np.mean(gtTrace)
    filtered_signals = fir_bp_filter(signal=normalized, fps=30, low=0.67, high=4.0)

    # Compute STFT
    noverlap = sampling_frequency * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = sampling_frequency * 12  # Length of fourier window (12 seconds as per the paper)

    # Perform STFT
    frequencies, times, Zxx = stft(filtered_signals, sampling_frequency, nperseg=nperseg, noverlap=noverlap)

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

    normalized = np.array(gtTrace) / np.mean(gtTrace)
    filtered_signals = fir_bp_filter(signal=normalized, fps=30, low=0.67, high=4.0)

    # Compute STFT
    noverlap = sampling_frequency * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = sampling_frequency * 12  # Length of fourier window (12 seconds as per the paper)

    # Perform STFT
    frequencies, times, Zxx = stft(filtered_signals, sampling_frequency, nperseg=nperseg, noverlap=noverlap)

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


def chrom_lgi_ppgi(ground_truth_file, sampling_frequency=60):
    gtdata = pd.read_xml(ground_truth_file)
    gtTime = (gtdata.iloc[:, 0]).tolist()
    gtHR = gtdata.iloc[:, 1].tolist()
    gtTrace = gtdata.iloc[:, 2].tolist()

    normalized = np.array(gtTrace) / np.mean(gtTrace)
    filtered_signals = fir_bp_filter(signal=normalized, fps=30, low=0.67, high=4.0)

    # Compute STFT
    noverlap = sampling_frequency * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = sampling_frequency * 12  # Length of fourier window (12 seconds as per the paper)

    # Perform STFT
    frequencies, times, Zxx = stft(filtered_signals, sampling_frequency, nperseg=nperseg, noverlap=noverlap)

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


# chrom_true = []
# # chrom_pred = []
# # # base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
# base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
# for sub_folders in os.listdir(base_dir):
#     # if sub_folders == 'UBFC1':
#     #     for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#     #         subjects = os.path.join(base_dir, sub_folders, folders)
#     #         for each_subject in os.listdir(subjects):
#     #             if each_subject.endswith('.avi'):
#     #                 vid = os.path.join(subjects, each_subject)
#     #             elif each_subject.endswith('.xmp'):
#     #                 gt = os.path.join(subjects, each_subject)
#     #
#     #         print(vid, gt)
#     #         hrES = chrom_framework(input_video=vid, dataset='UBFC1')
#     #         hrGT = chrom_ubfc1(ground_truth_file=gt)
#     #         print(len(hrGT), len(hrES))
#     #         print('')
#     #         chrom_true.append(np.mean(hrGT))
#     #         chrom_pred.append(np.mean(hrES))
#
#     if sub_folders == 'UBFC2':
#         for folders in os.listdir(os.path.join(base_dir, sub_folders)):
#             subjects = os.path.join(base_dir, sub_folders, folders)
#             for each_subject in os.listdir(subjects):
#                 if each_subject.endswith('.avi'):
#                     vid = os.path.join(subjects, each_subject)
#                 elif each_subject.endswith('.txt'):
#                     gt = os.path.join(subjects, each_subject)
#
#             print(vid, gt)
#             # hrES = chrom_framework(input_video=vid, dataset='UBFC2')
#             hrGT = chrom_ubfc2(ground_truth_file=gt)
#             # print(len(hrGT), len(hrES))
#             # print('')
#             chrom_true.append(np.mean(hrGT))
#             # chrom_pred.append(np.mean(hrES))
#
# print(chrom_true)
# # print(chrom_pred)
# # print(mean_absolute_error(chrom_true, chrom_pred))
# # print(mean_absolute_error(chrom_true[8:], chrom_pred[8:]))

# Without Mod for UBFC2
# [108.01886792452831, 94.2741935483871, 109.04255319148936, 99.05660377358491, 112.02898550724638, 108.40579710144928, 110.28985507246377, 123.76811594202898, 68.82352941176471, 107.17391304347827, 76.3970588235294, 115.8695652173913, 93.07142857142857, 86.85714285714286, 120.5072463768116, 125.8695652173913, 102.68115942028986, 65.97014925373135, 106.54411764705883, 114.9, 98.23529411764706, 109.38775510204081, 98.04347826086956, 78.76811594202898, 105.43478260869566, 116.76470588235294, 116.44927536231884, 103.57142857142857, 119.28571428571429, 59.492753623188406, 109.14285714285714, 84.92753623188406, 86.15942028985508, 101.01449275362319, 95.07142857142857, 99.41176470588235, 82.82608695652173, 110.14492753623189, 97.13235294117646, 110.5072463768116, 90.8955223880597, 87.82608695652173]
# [106.32075471698113, 93.38709677419355, 100.74468085106383, 92.0754716981132, 110.0, 102.17391304347827, 79.78260869565217, 121.8840579710145, 84.70588235294117, 92.31884057971014, 95.44117647058823, 94.4927536231884, 100.21428571428571, 82.92857142857143, 120.94202898550725, 126.1029411764706, 101.81159420289855, 66.41791044776119, 106.61764705882354, 111.0, 95.68627450980392, 97.85714285714286, 89.6376811594203, 82.89855072463769, 96.23188405797102, 115.51470588235294, 108.6231884057971, 103.85714285714286, 118.42857142857143, 83.875, 94.42857142857143, 84.71014492753623, 96.23188405797102, 85.79710144927536, 94.71428571428571, 98.30882352941177, 76.23188405797102, 109.85507246376811, 91.91176470588235, 98.76811594202898, 90.67164179104478, 87.68115942028986]
# 6.738555704334287

# With the mod for UBFC2 (which is using forehead ROI)
# [108.01886792452831, 94.2741935483871, 109.04255319148936, 99.05660377358491, 112.02898550724638, 108.40579710144928, 110.28985507246377, 123.76811594202898, 68.82352941176471, 107.17391304347827, 76.3970588235294, 115.8695652173913, 93.07142857142857, 86.85714285714286, 120.5072463768116, 125.8695652173913, 102.68115942028986, 65.97014925373135, 106.54411764705883, 114.9, 98.23529411764706, 109.38775510204081, 98.04347826086956, 78.76811594202898, 105.43478260869566, 116.76470588235294, 116.44927536231884, 103.57142857142857, 119.28571428571429, 59.492753623188406, 109.14285714285714, 84.92753623188406, 86.15942028985508, 101.01449275362319, 95.07142857142857, 99.41176470588235, 82.82608695652173, 110.14492753623189, 97.13235294117646, 110.5072463768116, 90.8955223880597, 87.82608695652173]
# [108.01886792452831, 94.2741935483871, 105.53191489361703, 105.75471698113208, 111.95652173913044, 106.66666666666667, 110.57971014492753, 138.768115942029, 68.82352941176471, 105.79710144927536, 78.16176470588235, 108.18840579710145, 93.42857142857143, 86.92857142857143, 120.79710144927536, 126.69117647058823, 101.59420289855072, 65.5223880597015, 106.47058823529412, 114.7, 99.11764705882354, 107.55102040816327, 99.34782608695652, 78.47826086956522, 103.98550724637681, 116.98529411764706, 116.44927536231884, 104.21428571428571, 119.28571428571429, 62.0, 112.71428571428571, 84.92753623188406, 86.01449275362319, 100.07246376811594, 94.92857142857143, 97.20588235294117, 81.3768115942029, 110.0, 97.79411764705883, 110.21739130434783, 91.41791044776119, 87.7536231884058]
# 1.4467195072371448


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
#         hrES = chrom_framework(input_video=vid, dataset='LGI_PPGI')
#         hrGT = chrom_lgi_ppgi(ground_truth_file=gt)
#         print(len(hrGT), len(hrES))
#         print('')
#         chrom_true.append(np.mean(hrGT))
#         chrom_pred.append(np.mean(hrES))
#
# print(chrom_true)
# print(chrom_pred)
# print(mean_absolute_error(chrom_true, chrom_pred))
# print(f"gym: {mean_absolute_error(chrom_true[0:6], chrom_pred[0:6])}")
# print(f"resting: {mean_absolute_error(chrom_true[6:12], chrom_pred[6:12])}")
# print(f"rotation: {mean_absolute_error(chrom_true[12:18], chrom_pred[12:18])}")
# print(f"talk: {mean_absolute_error(chrom_true[18:24], chrom_pred[18:24])}")


# [55.63275434243176, 50.825688073394495, 48.361702127659576, 50.53225806451613, 47.1764705882353, 52.31818181818182, 63.63636363636363, 54.857142857142854, 59.09090909090909, 72.33333333333333, 74.20289855072464, 43.46666666666667, 68.60759493670886, 52.82608695652174, 58.705882352941174, 65.06666666666666, 72.01388888888889, 43.16901408450704, 72.60416666666667, 56.2, 68.6470588235294, 58.71951219512195, 70.625, 70.05952380952381]
# [56.381909547738694, 53.591331269349844, 47.56637168141593, 50.06535947712418, 51.109422492401215, 51.585648148148145, 50.0, 51.470588235294116, 52.770270270270274, 62.19178082191781, 55.833333333333336, 29.86111111111111, 52.36842105263158, 52.08955223880597, 47.4390243902439, 51.875, 65.79710144927536, 33.405797101449274, 54.21052631578947, 54.189189189189186, 56.103896103896105, 54.074074074074076, 55.57692307692308, 51.626506024096386]
# 8.474630322375537
# gym: 1.5737521681829112
# resting: 10.910038393868929
# rotation: 9.56903960897138
# talk: 11.845691118478923

# Improved
# [111.41439205955335, 103.68501529051987, 113.63829787234043, 115.85483870967742, 105.41176470588235, 100.38636363636364, 65.77922077922078, 60.714285714285715, 60.324675324675326, 78.6, 73.1159420289855, 52.0, 68.41772151898734, 58.26086956521739, 60.88235294117647, 82.53333333333333, 76.04166666666667, 51.40845070422535, 73.22916666666667, 63.0, 80.29411764705883, 87.7439024390244, 76.1875, 74.52380952380952]
# [114.98743718592965, 96.39318885448917, 114.82300884955752, 112.5, 107.87234042553192, 95.81018518518519, 66.13333333333334, 61.25, 60.810810810810814, 84.93150684931507, 73.63636363636364, 46.52777777777778, 68.48684210526316, 71.26865671641791, 63.84146341463415, 77.98611111111111, 76.8840579710145, 46.88405797101449, 84.73684210526316, 78.44594594594595, 89.6103896103896, 113.51851851851852, 79.16666666666667, 84.51807228915662]
# 5.712885489795973
# gym: 3.7401959033549232
# resting: 2.2833521674796287
# rotation: 4.325004078452488
# talk: 12.502989809896853





### END TEST SECTION


# Face segmentation
#
# import cv2
# import numpy as np
# import torch
# from torchvision import transforms
# from Necessary_Files.face_parsing.model import BiSeNet
# import os
#
# net = BiSeNet(n_classes=19)
# net.load_state_dict(torch.load('Necessary_Files\\79999_iter.pth', map_location='cpu'))
# net.eval()
# to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
#
#
#
# # cap = cv2.VideoCapture(r"C:\Users\ilyas\Desktop\VHR\Datasets\Distance vs Light Dataset\test_all_riccardo_distances_L00_NoEx\D01.mp4")
#
# def asdf(input_video):
#     raw_sig = []
#
#     for frame in extract_frames_yield(input_video):
#         # ret, frame = cap.read()
#
#         face_cascade = cv2.CascadeClassifier("Necessary_Files\\haarcascade_frontalface_default.xml")
#
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#         width = 1
#         height = 1
#         if len(faces) == 0 and face_coordinates_prev is not None:
#             x, y, w, h = face_coordinates_prev
#             x1 = int(x + (1 - width) / 2 * w)
#             y1 = int(y + (1 - height) / 2 * h)
#             x2 = int(x + (1 + width) / 2 * w)
#             y2 = int(y + (1 + height) / 2 * h)
#             roi = frame[y1:y2, x1:x2]
#
#         else:
#             for (x, y, w, h) in faces:
#                 face_coordinates_prev = (x, y, w, h)
#                 x1 = int(x + (1 - width) / 2 * w)
#                 y1 = int(y + (1 - height) / 2 * h)
#                 x2 = int(x + (1 + width) / 2 * w)
#                 y2 = int(y + (1 + height) / 2 * h)
#                 roi = frame[y1:y2, x1:x2]
#
#         image = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_LINEAR)
#         img = to_tensor(image)
#         img = torch.unsqueeze(img, 0)
#         out = net(img)[0]
#         out = out.squeeze().argmax(0).detach().cpu().numpy()
#
#         # Create the masks for classes 0 and 10.
#         mask_0 = np.where(out == 1, 255, 0).astype('uint8')
#         mask_10 = np.where(out == 10, 255, 0).astype('uint8')
#
#         # Combine the masks.
#         mask_combined = np.where((mask_0 == 255) | (mask_10 == 255), 255, 0).astype('uint8')
#
#         # Apply the combined mask to the original frame.
#         face = cv2.bitwise_and(image, image, mask=mask_combined)
#
#         # Get the original ROI size.
#         original_size = roi.shape[:2]
#
#         # Resize the face image back to the original ROI size.
#         face_resized = cv2.resize(face, (original_size[1], original_size[0]))
#
#         mask_non_black = cv2.inRange(face_resized, np.array([75, 75, 75]), np.array([200, 200, 200]))
#         b, g, r, a = cv2.mean(face_resized, mask=mask_non_black)
#
#         raw_sig.append([r, g, b])
#
#     return(raw_sig)
#
#
# #
# #
# #     # Display the result.
# #     cv2.imshow('Face', face_resized)
# #     cv2.imshow('frame', roi)
# #
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
#
# base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
# for sub_folders in os.listdir(base_dir):
#     if sub_folders == 'UBFC2':
#         for enum, folders in enumerate(os.listdir(os.path.join(base_dir, sub_folders))):
#             subjects = os.path.join(base_dir, sub_folders, folders)
#             for each_subject in os.listdir(subjects):
#                 if each_subject.endswith('.avi'):
#                     vid = os.path.join(subjects, each_subject)
#
#                     raw_sig = asdf(vid)
#                     print(enum)
#                     with open('raw_chrom_sig_face_segmentation.txt', 'a') as f:
#                         f.write(str(raw_sig))
#                         f.write('\n')
#
#
