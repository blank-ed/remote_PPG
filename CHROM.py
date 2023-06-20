"""

This module contains the framework implemented by https://ieeexplore.ieee.org/document/6523142 which is
also known as CHROM rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""

from remote_PPG.utils import *
from scipy.signal import find_peaks, welch, windows, stft
from remote_PPG.filters import normalize
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
        raw_sig = extract_raw_sig(input_video, framework='CHROM', width=1, height=1)
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
        l = int(fps*1.6)

        window = moving_window(raw_sig, fps=fps, window_size=1.6, increment=0.8)

        for enum, each_window in enumerate(window):
            normalized = normalize(signal=each_window, framework='CHROM')  # Normalize each windowed segment

            # Build two orthogonal chrominance signals
            Xs = 3 * normalized[0] - 2 * normalized[1]
            Ys = 1.5 * normalized[0] + normalized[1] - 1.5 * normalized[2]

            # bandpass filter Xs and Ys here
            Xf = fir_bp_filter(signal=Xs, fps=fps, low=0.67, high=4.0)
            Yf = fir_bp_filter(signal=Ys, fps=fps, low=0.67, high=4.0)

            alpha = np.std(Xf) / np.std(Yf)
            S = Xf - alpha * Yf

            SWin = np.multiply(S, windows.hann(len(S)))

            start = enum*(l//2)
            end = enum*(l//2) + l

            if end > len(raw_sig):
                H[len(raw_sig)-l:len(raw_sig)] = H[len(raw_sig)-l:len(raw_sig)] + SWin
            else:
                H[start:end] = H[start:end] + SWin

        H = butterworth_bp_filter(H, fps=fps, low=0.67, high=2)  # MOD ---------------------------------------------

        # Compute STFT
        noverlap = fps*(12-1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
        nperseg = fps*12  # Length of fourier window (12 seconds as per the paper)

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

    normalized = np.array(gtTrace) / np.mean(gtTrace)

    # bandpass filter Xs and Ys here
    filtered = fir_bp_filter(signal=normalized, fps=sampling_frequency, low=0.67, high=4.0)

    # Compute STFT
    noverlap = sampling_frequency * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = sampling_frequency * 12  # Length of fourier window (12 seconds as per the paper)

    frequencies, times, Zxx = stft(filtered, sampling_frequency, nperseg=nperseg, noverlap=noverlap)  # Perform STFT

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


# chrom_true = []
# chrom_pred = []
# base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
# # base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
# for sub_folders in os.listdir(base_dir):
    # if sub_folders == 'UBFC1':
    #     for folders in os.listdir(os.path.join(base_dir, sub_folders)):
    #         subjects = os.path.join(base_dir, sub_folders, folders)
    #         for each_subject in os.listdir(subjects):
    #             if each_subject.endswith('.avi'):
    #                 vid = os.path.join(subjects, each_subject)
    #             elif each_subject.endswith('.xmp'):
    #                 gt = os.path.join(subjects, each_subject)
    #
    #         print(vid, gt)
    #         hrES = chrom_framework(input_video=vid, dataset='UBFC1')
    #         hrGT = chrom_ubfc1(ground_truth_file=gt)
    #         print(len(hrGT), len(hrES))
    #         print('')
    #         chrom_true.append(np.mean(hrGT))
    #         chrom_pred.append(np.mean(hrES))

    # if sub_folders == 'UBFC2':
    #     for folders in os.listdir(os.path.join(base_dir, sub_folders)):
    #         subjects = os.path.join(base_dir, sub_folders, folders)
    #         for each_subject in os.listdir(subjects):
    #             if each_subject.endswith('.avi'):
    #                 vid = os.path.join(subjects, each_subject)
    #             elif each_subject.endswith('.txt'):
    #                 gt = os.path.join(subjects, each_subject)
    #
    #         print(vid, gt)
    #         hrES = chrom_framework(input_video=vid, dataset='UBFC2')
    #         hrGT = chrom_ubfc2(ground_truth_file=gt)
    #         print(len(hrGT), len(hrES))
    #         print('')
    #         chrom_true.append(np.mean(hrGT))
    #         chrom_pred.append(np.mean(hrES))

# print(chrom_true)
# print(chrom_pred)
# print(mean_absolute_error(chrom_true, chrom_pred))
# print(mean_absolute_error(chrom_true[8:], chrom_pred[8:]))

# [73.98876404494382, 67.32954545454545, 44.48275862068966, 58.97435897435897, 68.80434782608695, 72.41176470588235, 59.43181818181818, 48.53658536585366, 46.22641509433962, 45.725806451612904, 48.61702127659574, 45.37735849056604, 52.46376811594203, 45.65217391304348, 44.05797101449275, 50.36231884057971, 48.529411764705884, 72.97101449275362, 51.029411764705884, 46.30434782608695, 51.357142857142854, 49.0, 46.52173913043478, 49.34782608695652, 46.44927536231884, 47.76119402985075, 45.0, 47.5, 45.588235294117645, 43.775510204081634, 47.7536231884058, 46.44927536231884, 53.11594202898551, 50.36764705882353, 43.91304347826087, 48.0, 42.357142857142854, 55.65217391304348, 47.642857142857146, 69.92753623188406, 45.36231884057971, 50.289855072463766, 47.214285714285715, 48.8235294117647, 48.6231884057971, 46.30434782608695, 46.76470588235294, 45.36231884057971, 44.701492537313435, 53.26086956521739]
# [49.02439024390244, 52.407407407407405, 47.4375, 55.486111111111114, 62.73809523809524, 55.0, 49.629629629629626, 47.58771929824562, 47.35849056603774, 49.83870967741935, 57.12765957446808, 47.924528301886795, 43.98550724637681, 52.46376811594203, 48.84057971014493, 45.289855072463766, 55.588235294117645, 54.492753623188406, 47.5, 61.01449275362319, 50.07142857142857, 48.785714285714285, 48.55072463768116, 51.73913043478261, 44.20289855072464, 56.56716417910448, 50.0735294117647, 42.0, 48.8235294117647, 48.06122448979592, 48.69565217391305, 49.05797101449275, 48.55072463768116, 46.69117647058823, 45.869565217391305, 49.285714285714285, 49.42857142857143, 46.125, 56.5, 63.47826086956522, 52.2463768115942, 43.768115942028984, 54.714285714285715, 53.01470588235294, 52.82608695652174, 41.52173913043478, 54.55882352941177, 58.11594202898551, 52.53731343283582, 60.79710144927536]

# 6.35584093698124
# 5.648416043920325

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
