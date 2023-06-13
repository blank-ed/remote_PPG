from ICA_framework.ICA import *
import os
import pandas as pd
from remote_PPG.utils import *
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error


def ica_ubfc1(ground_truth_file, sampling_frequency=60):
    gtdata = pd.read_csv(ground_truth_file, header=None)
    gtTrace = gtdata.iloc[:, 3]
    gtTime = gtdata.iloc[:, 0]
    gtHR = gtdata.iloc[:, 1]

    windowed_signals = moving_window(gtTrace.tolist(), sampling_frequency, 30, 1)
    hrGT = []
    for each_window in windowed_signals:
        # normalized = normalize(each_window, framework='ICA')
        signal = np.array(each_window)
        mean = np.mean(signal, axis=0)
        std_dev = np.std(signal, axis=0)
        normalized_signal = (signal - mean) / std_dev

        # Compute the positive frequencies and the corresponding power spectrum
        freqs = rfftfreq(len(normalized_signal), d=1 / sampling_frequency)
        power_spectrum = np.abs(rfft(normalized_signal)) ** 2

        # Find the maximum peak between 0.75 Hz and 4 Hz
        mask = (freqs >= 0.75) & (freqs <= 4)
        filtered_power_spectrum = power_spectrum[mask]
        filtered_freqs = freqs[mask]

        peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks
        peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
        peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers

        # Find the highest peak
        max_peak_index = np.argmax(peak_powers)
        max_peak_frequency = peak_freqs[max_peak_index]

        hr = int(max_peak_frequency * 60)

        hrGT.append(hr)

    return hrGT


def ica_ubfc2(ground_truth_file, sampling_frequency=30):
    gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
    gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
    gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
    gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

    windowed_signals = moving_window(gtTrace, sampling_frequency, 30, 1)
    hrGT = []
    for each_window in windowed_signals:
        signal = np.array(each_window)
        mean = np.mean(signal, axis=0)
        std_dev = np.std(signal, axis=0)
        normalized_signal = (signal - mean) / std_dev

        # Compute the positive frequencies and the corresponding power spectrum
        freqs = rfftfreq(len(normalized_signal), d=1 / sampling_frequency)
        power_spectrum = np.abs(rfft(normalized_signal)) ** 2

        # Find the maximum peak between 0.75 Hz and 4 Hz
        mask = (freqs >= 0.75) & (freqs <= 4)
        filtered_power_spectrum = power_spectrum[mask]
        filtered_freqs = freqs[mask]

        peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks
        peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
        peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers

        # Find the highest peak
        max_peak_index = np.argmax(peak_powers)
        max_peak_frequency = peak_freqs[max_peak_index]

        hr = int(max_peak_frequency * 60)

        hrGT.append(hr)

    return hrGT


# MAE = []
# base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
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
#             hrES = ica_framework(input_video=vid, dataset='UBFC1')
#             hrGT = ica_ubfc1(ground_truth_file=gt)
#             print(len(hrGT), len(hrES))
#             print('')
#             MAE.append(mean_absolute_error(hrGT[0:len(hrES)], hrES))
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
#             hrES = ica_framework(input_video=vid, dataset='UBFC2')
#             hrGT = ica_ubfc2(ground_truth_file=gt)
#             print(len(hrGT), len(hrES))
#             print('')
#             MAE.append(mean_absolute_error(hrGT, hrES))
#
# print(MAE)

ground_truth_file = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset\UBFC2\subject37\ground_truth.txt'
gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']
windowed_sig = moving_window(sig=gtTrace, fps=30, window_size=30, increment=1)
print(len(windowed_sig))
print(len(gtTrace))

cap = cv2.VideoCapture(r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset\UBFC2\subject37\vid.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
duration = frame_count/fps
frame_c = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_c += 1
    print(frame_c)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()



print(frame_count, fps, duration)
# raw_sig = extract_raw_sig(r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset\UBFC2\subject37\vid.avi', framework='ICA', width=0.6, height=1)
# print(len(raw_sig))
# windowed_sig = moving_window(sig=raw_sig, fps=30, window_size=30, increment=1)
# hrES = []
# print(len(windowed_sig))

# prev_hr = None  # Previous HR value
# for sig in windowed_sig:
#     normalized = normalize(sig, framework='ICA')  # normalize the windowed signal
#
#     # Apply JADE ICA algorithm and select the second component
#     W = jadeR(normalized)
#     bvp = np.array(np.dot(W, normalized))
#     bvp = bvp[1].flatten()
#
#     # Compute the positive frequencies and the corresponding power spectrum
#     freqs = rfftfreq(len(bvp), d=1 / fps)
#     power_spectrum = np.abs(rfft(bvp)) ** 2
#
#     # Find the maximum peak between 0.75 Hz and 4 Hz
#     mask = (freqs >= 0.75) & (freqs <= 4)
#     filtered_power_spectrum = power_spectrum[mask]
#     filtered_freqs = freqs[mask]
#
#     peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks
#     peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
#     peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers
#
#     # For the first previous HR value
#     # if prev_hr is None:
#         # Find the highest peak
#     max_peak_index = np.argmax(peak_powers)
#     max_peak_frequency = peak_freqs[max_peak_index]
#
#     hr = int(max_peak_frequency * 60)
#         # prev_hr = hr
#     # else:
#     #     max_peak_index = np.argmax(peak_powers)
#     #     max_peak_frequency = peak_freqs[max_peak_index]
#     #
#     #     hr = int(max_peak_frequency * 60)
#     #
#     #     # If the difference between the current pulse rate estimation and the last computed value exceeded
#     #     # the threshold, the algorithm rejected it and searched the operational frequency range for the
#     #     # frequency corresponding to the next highest power that met this constraint
#     #     while abs(prev_hr - hr) >= hr_change_threshold:
#     #         # Remove the previously wrongly determined power and frequency values from the list
#     #         max_peak_mask = (peak_freqs == max_peak_frequency)
#     #         peak_freqs = peak_freqs[~max_peak_mask]
#     #         peak_powers = peak_powers[~max_peak_mask]
#     #
#     #         #  If no frequency peaks that met the criteria were located, then
#     #         # the algorithm retained the current pulse frequency estimation
#     #         if len(peak_freqs) == 0:
#     #             hr = prev_hr
#     #             break
#     #
#     #         max_peak_index = np.argmax(peak_powers)
#     #         max_peak_frequency = peak_freqs[max_peak_index]
#     #         hr = int(max_peak_frequency * 60)
#     #
#     #     prev_hr = hr
#     hrES.append(hr)
#
# print(hrES)
# print(ica_framework(r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset\UBFC2\subject37\vid.avi', dataset='UBFC2'))