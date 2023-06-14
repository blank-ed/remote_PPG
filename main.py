from ICA_framework.ICA import *
import os
import pandas as pd
from remote_PPG.utils import *
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
from sklearn.metrics import mean_absolute_error


# def ica_ubfc1(ground_truth_file, sampling_frequency=60):
def ica_ubfc1(ground_truth_file):
    gtdata = pd.read_csv(ground_truth_file, header=None)
    gtTrace = gtdata.iloc[:, 3]
    gtTime = (gtdata.iloc[:, 0]/1000).tolist()
    gtHR = gtdata.iloc[:, 1]

    time = np.array(gtTime)
    sampling_frequency = np.round(1 / np.mean(np.diff(time)))
    print(sampling_frequency)

    windowed_signals = moving_window(gtTrace.tolist(), sampling_frequency, 30, 1)
    hrGT = []
    for each_window in windowed_signals:
        # normalized = normalize(each_window, framework='ICA')
        signal = np.array(each_window)
        mean = np.mean(signal)
        std_dev = np.std(signal)
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


# def ica_ubfc2(ground_truth_file, sampling_frequency=30):
def ica_ubfc2(ground_truth_file):
    gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
    gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
    gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
    gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

    time = np.array(gtTime)
    sampling_frequency = 1 / np.mean(np.diff(time))
    print(sampling_frequency)

    windowed_signals = moving_window(gtTrace, sampling_frequency, 30, 1)
    hrGT = []
    for each_window in windowed_signals:
        signal = np.array(each_window)
        mean = np.mean(signal)
        std_dev = np.std(signal)
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


MAE = []
ubfc1_es = []
ubfc1_gt = []
ubfc2_es = []
ubfc2_gt = []
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
            ubfc1_es.append(np.mean(hrES))
            ubfc1_gt.append(np.mean(hrGT))
            # abs_error = abs(np.mean(hrES) - np.mean(hrGT)) / np.mean(hrGT)
            # MAE.append(abs_error)
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
            hrES = ica_framework(input_video=vid, dataset='UBFC2')
            hrGT = ica_ubfc2(ground_truth_file=gt)
            print(len(hrGT), len(hrES))
            print('')
            ubfc2_es.append(np.mean(hrES))
            ubfc2_gt.append(np.mean(hrGT))
            # abs_error = abs(np.mean(hrES) - np.mean(hrGT)) / np.mean(hrGT)
            # MAE.append(abs_error)
            # if len(hrGT) > len(hrES):
            #     MAE.append(mean_absolute_error(hrGT[0:len(hrES)], hrES))
            # else:
            #     MAE.append(mean_absolute_error(hrGT, hrES[0:len(hrGT)]))

# print(MAE)
# print(np.mean(MAE))
print(ubfc1_es)
print(ubfc1_gt)
print(ubfc2_es)
print(ubfc2_gt)
print(mean_absolute_error(y_true=ubfc1_gt, y_pred=ubfc1_es))
print(mean_absolute_error(y_true=ubfc2_gt, y_pred=ubfc2_es))
# [6.384615384615385, 23.941176470588236, 37.84, 6.023809523809524, 7.666666666666667, 16.916666666666668, 32.13725490196079, 50.666666666666664, 0.6086956521739131, 4.1875, 57.88235294117647, 41.47826086956522, 0.9743589743589743, 30.76923076923077, 11.948717948717949, 2.4615384615384617, 6.184210526315789, 35.1025641025641, 13.236842105263158, 60.05128205128205, 41.075, 35.4, 67.58974358974359, 2.5128205128205128, 46.82051282051282, 2.4324324324324325, 40.23684210526316, 62.15, 44.095238095238095, 56.0, 5.333333333333333, 0.6666666666666666, 55.02564102564103, 68.89473684210526, 3.58974358974359, 58.2, 57.2, 6.0, 2.175, 4.461538461538462, 9.205128205128204, 47.58974358974359, 23.65, 5.684210526315789, 5.17948717948718, 54.53846153846154, 42.8421052631579, 2.4615384615384617, 31.45945945945946, 1.8974358974358974]
# edited=[9.035714285714286, 24.98181818181818, 18.62962962962963, 4.7555555555555555, 8.857142857142858, 7.134615384615385, 3.890909090909091, 39.1123595505618, 3.8333333333333335, 6.7272727272727275, 8.11111111111111, 30.166666666666668, 3.9, 35.325, 14.45, 5.560975609756097, 2.85, 7.7, 10.9, 58.41463414634146, 2.5365853658536586, 35.4390243902439, 48.26829268292683, 79.3170731707317, 3.951219512195122, 4.725, 62.75, 7.294117647058823, 13.61111111111111, 6.181818181818182, 28.170731707317074, 5.317073170731708, 56.78048780487805, 59.775, 5.073170731707317, 48.926829268292686, 59.51219512195122, 6.25, 4.780487804878049, 17.0, 35.41463414634146, 45.90243902439025, 18.146341463414632, 8.15, 7.15, 5.2926829268292686, 39.85, 4.95, 31.829268292682926, 5.853658536585366]

# ground_truth_file = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset\UBFC2\subject37\ground_truth.txt'
# gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
# gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
# gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
# gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']
# windowed_sig = moving_window(sig=gtTrace, fps=30, window_size=30, increment=1)
# print(len(windowed_sig))
# print(len(gtTrace))
#
# cap = cv2.VideoCapture(r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset\UBFC2\subject37\vid.avi')
# fps = cap.get(cv2.CAP_PROP_FPS)
# frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# duration = frame_count/fps
# frame_c = 0
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_c += 1
#     print(frame_c)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#
#
# cap.release()
# cv2.destroyAllWindows()
#
#
#
# print(frame_count, fps, duration)
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