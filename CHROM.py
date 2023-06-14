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
        else:
            assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                         "types: 'UBFC1', 'UBFC2'. If you are using your own dataset, enter 'None' "

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
        else:
            assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                          "types: 'UBFC1', 'UBFC2'. If you are using your own dataset, enter 'None' "

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
            hrES = chrom_framework(input_video=vid, dataset='UBFC1')
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
            hrES = chrom_framework(input_video=vid, dataset='UBFC2')
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

# [24.878048780487806, 16.29629629629629, 15.1875, 11.041666666666664, 9.999999999999996, 18.397435897435898, 11.851851851851853, 10.175438596491228, 16.037735849056602, 11.53225806451613, 13.404255319148936, 9.905660377358489, 18.47826086956522, 13.623188405797102, 14.492753623188406, 14.202898550724637, 19.11764705882353, 33.69565217391305, 13.529411764705882, 26.44927536231884, 13.571428571428573, 9.5, 8.405797101449275, 9.492753623188406, 11.376811594202898, 15.671641791044776, 9.926470588235293, 10.3, 10.882352941176471, 10.000000000000002, 15.579710144927539, 11.956521739130432, 15.289855072463768, 13.235294117647058, 9.927536231884059, 7.285714285714286, 11.5, 16.0, 14.0, 29.63768115942029, 11.08695652173913, 13.043478260869563, 9.357142857142858, 9.63235294117647, 12.46376811594203, 10.0, 12.5, 18.840579710144926, 11.716417910447761, 11.014492753623191]


### END TEST SECTION


# Face segmentation

# import cv2
# import numpy as np
# import torch
# from torchvision import transforms
# from Necessary_Files.face_parsing.model import BiSeNet
#
# net = BiSeNet(n_classes=19)
# net.load_state_dict(torch.load('Necessary_Files\\79999_iter.pth', map_location='cpu'))
# net.eval()
# to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#
# transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#
# cap = cv2.VideoCapture(r"C:\Users\ilyas\Desktop\VHR\Datasets\Distance vs Light Dataset\test_all_riccardo_distances_L00_NoEx\D01.mp4")
# video_sequence = []
#
# while True:
#     ret, frame = cap.read()
#
#     face_cascade = cv2.CascadeClassifier("Necessary_Files\\haarcascade_frontalface_default.xml")
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
#     width = 1
#     height = 1
#     if len(faces) == 0 and face_coordinates_prev is not None:
#         x, y, w, h = face_coordinates_prev
#         x1 = int(x + (1 - width) / 2 * w)
#         y1 = int(y + (1 - height) / 2 * h)
#         x2 = int(x + (1 + width) / 2 * w)
#         y2 = int(y + (1 + height) / 2 * h)
#         roi = frame[y1:y2, x1:x2]
#
#     else:
#         for (x, y, w, h) in faces:
#             face_coordinates_prev = (x, y, w, h)
#             x1 = int(x + (1 - width) / 2 * w)
#             y1 = int(y + (1 - height) / 2 * h)
#             x2 = int(x + (1 + width) / 2 * w)
#             y2 = int(y + (1 + height) / 2 * h)
#             roi = frame[y1:y2, x1:x2]
#
#     image = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_LINEAR)
#     img = to_tensor(image)
#     img = torch.unsqueeze(img, 0)
#     out = net(img)[0]
#     out = out.squeeze().argmax(0).detach().cpu().numpy()
#
#     # Create the masks for classes 0 and 10.
#     mask_0 = np.where(out == 1, 255, 0).astype('uint8')
#     mask_10 = np.where(out == 10, 255, 0).astype('uint8')
#
#     # Combine the masks.
#     mask_combined = np.where((mask_0 == 255) | (mask_10 == 255), 255, 0).astype('uint8')
#
#     # Apply the combined mask to the original frame.
#     face = cv2.bitwise_and(image, image, mask=mask_combined)
#
#     # Get the original ROI size.
#     original_size = roi.shape[:2]
#
#     # Resize the face image back to the original ROI size.
#     face_resized = cv2.resize(face, (original_size[1], original_size[0]))
#
#     # Display the result.
#     cv2.imshow('Face', face_resized)
#     cv2.imshow('frame', roi)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


