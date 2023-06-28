"""

This module contains the framework implemented by https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-26-21434&id=175396
also known as GREEN rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""
from scipy.signal import welch
from scipy.signal.windows import windows
from remote_PPG.sig_extraction_utils import *
from remote_PPG.utils import *
from remote_PPG.filters import *
from scipy.fft import fft, fftfreq
import pandas as pd
from sklearn.metrics import mean_absolute_error


def green_test(input_video):
    fps = 30

    raw_sig = extract_raw_sig(input_video, framework='GREEN', ROI_type="ROI_I")
    raw_sig = np.array(raw_sig)[:, 1]  # Select the green channel

    pv_raw = raw_sig
    pv_ac = ((np.array(pv_raw) - np.mean(pv_raw)) / np.std(pv_raw)).tolist()
    pv_bp = fir_bp_filter(pv_ac, fps=fps, low=0.8, high=2.0)

    windowed_pulse_sig = moving_window(sig=pv_bp, fps=fps, window_size=11, increment=5)
    hrES = []
    prev_hr = None
    for each_signal_window in windowed_pulse_sig:
        windowed_signal = each_signal_window * windows.hann(len(each_signal_window))
        peak_freqs, peak_powers = welch(windowed_signal, fs=fps, nperseg=len(windowed_signal), nfft=4096)

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
        hrES.append(hr)

    window_size = 7
    rolling_mean_hrES = np.convolve(hrES, np.ones(window_size), mode='valid') / window_size
    hr = np.mean(rolling_mean_hrES)

    return hr

def green_framework(input_video, roi_type='ROI_I', signal='bp', lower_frequency=0.8, higher_frequency=2.0,
                    dataset=None):
    """
    :param input_video:
        This takes in an input video file
    :param roi_type:
        Select the type of ROI to extract the green channel signal from:
        - 'ROI_I': forehead bounding box
        - 'ROI_II': single pixel in the forehead ROI
        - 'ROI_III': beside head (This doesn't include any skin pixels, only the background)
        - 'ROI_IV': whole frame scaled down by a fraction
    :param signal:
        Select the type of signal to extract the heart rate from:
        - 'raw': PV_raw(t) no processing other than spatial averaging over ROI
        - 'ac': PV_AC(t) the mean over time of PV_raw(t) is subtracted (= PV_raw(t) minus DC)
        - 'bp': PV_BP(t) band-pass filtered PV_raw(t) signal. For the band-pass (BP) filter Butterworth coefficients
                (4th order) were used in a phase with the specified lower and higher frequencies.
    :param lower_frequency:
        This is the low frequency level
    :param higher_frequency:
        This is the high frequency level
    :return:
        Returns the estimated heart rate of the input video based on GREEN framework
    """

    raw_sig = extract_raw_sig(input_video, framework='GREEN', ROI_type=roi_type)
    raw_sig = np.array(raw_sig)[:, 1]  # Select the green channel

    if dataset is None:
        fps = get_fps(input_video)  # find the fps of the video
    elif dataset == 'UBFC1' or dataset == 'UBFC2':
        fps = 30
    elif dataset == 'LGI_PPGI':
        fps = 25
    else:
        assert False, "Invalid dataset name. Please choose one of the valid available datasets " \
                      "types: 'UBFC1', 'UBFC2'. If you are using your own dataset, enter 'None' "

    pv_raw = raw_sig
    pv_ac = normalize(pv_raw, normalize_type='zero_mean')
    pv_bp = butterworth_bp_filter(pv_raw, fps=fps, low=lower_frequency, high=higher_frequency)

    # Perform the FFT on selected signal
    if signal == 'raw':
        fft_pv = fft(pv_raw)
    elif signal == 'ac':
        fft_pv = fft(pv_ac)
    elif signal == 'bp':
        fft_pv = fft(pv_bp)
    else:
        assert False, "Invalid signal type for the 'GREEN' framework. Please choose one of the valid signals " \
                      "types: 'raw' (for raw G signal), 'ac' (removed DC component), or 'bp' (bandpass filtered " \
                      "raw signal with the specified lower and higher frequency) "

    # Calculate the power spectrum by taking the absolute square of the FFT
    power_spectrum = np.abs(fft_pv) ** 2

    # Calculate the corresponding frequencies
    frequencies = fftfreq(len(pv_bp), d=1 / fps)

    # Display only the positive frequencies and corresponding power spectrum
    positive_frequencies = frequencies[:len(frequencies) // 2]  # Take only the first half
    positive_spectrum = power_spectrum[:len(power_spectrum) // 2]  # Take only the first half

    freq_range = (lower_frequency, higher_frequency)  # Frequency range

    # Find the indices corresponding to the desired frequency range
    start_idx = np.argmax(positive_frequencies >= freq_range[0])
    end_idx = np.argmax(positive_frequencies >= freq_range[1])

    # Extract the frequencies and power spectrum within the desired range
    frequencies_range = positive_frequencies[start_idx:end_idx]
    power_spectrum_range = positive_spectrum[start_idx:end_idx]

    max_idx = np.argmax(power_spectrum_range)
    f_max = frequencies_range[max_idx]
    hr = f_max * 60.0

    return hr


def green_ubfc1(ground_truth_file, sampling_frequency=60, signal='bp', lower_frequency=0.8, higher_frequency=2.0):
    gtdata = pd.read_csv(ground_truth_file, header=None)
    gtTrace = gtdata.iloc[:, 3].tolist()
    gtTime = (gtdata.iloc[:, 0] / 1000).tolist()
    gtHR = gtdata.iloc[:, 1]

    pv_raw = gtTrace
    pv_ac = (np.array(pv_raw) - np.mean(pv_raw)).tolist()
    pv_bp = butterworth_bp_filter(pv_raw, fps=sampling_frequency, low=lower_frequency, high=higher_frequency)

    # Perform the FFT on selected signal
    if signal == 'raw':
        fft_pv = fft(pv_raw)
    elif signal == 'ac':
        fft_pv = fft(pv_ac)
    elif signal == 'bp':
        fft_pv = fft(pv_bp)
    else:
        assert False, "Invalid signal type for the 'GREEN' framework. Please choose one of the valid signals " \
                      "types: 'raw' (for raw G signal), 'ac' (removed DC component), or 'bp' (bandpass filtered " \
                      "raw signal with the specified lower and higher frequency) "

    # Calculate the power spectrum by taking the absolute square of the FFT
    power_spectrum = np.abs(fft_pv) ** 2

    # Calculate the corresponding frequencies
    frequencies = fftfreq(len(pv_bp), d=1 / sampling_frequency)

    # Display only the positive frequencies and corresponding power spectrum
    positive_frequencies = frequencies[:len(frequencies) // 2]  # Take only the first half
    positive_spectrum = power_spectrum[:len(power_spectrum) // 2]  # Take only the first half

    freq_range = (lower_frequency, higher_frequency)  # Frequency range

    # Find the indices corresponding to the desired frequency range
    start_idx = np.argmax(positive_frequencies >= freq_range[0])
    end_idx = np.argmax(positive_frequencies >= freq_range[1])

    # Extract the frequencies and power spectrum within the desired range
    frequencies_range = positive_frequencies[start_idx:end_idx]
    power_spectrum_range = positive_spectrum[start_idx:end_idx]

    max_idx = np.argmax(power_spectrum_range)
    f_max = frequencies_range[max_idx]
    hrGT = f_max * 60.0

    return hrGT


def green_ubfc2(ground_truth_file, sampling_frequency=30, signal='bp', lower_frequency=0.8, higher_frequency=2.0):
    gtdata = pd.read_csv(ground_truth_file, delimiter='\t', header=None)
    gtTrace = [float(item) for item in gtdata.iloc[0, 0].split(' ') if item != '']
    gtTime = [float(item) for item in gtdata.iloc[2, 0].split(' ') if item != '']
    gtHR = [float(item) for item in gtdata.iloc[1, 0].split(' ') if item != '']

    pv_raw = gtTrace
    pv_ac = (np.array(pv_raw) - np.mean(pv_raw)).tolist()
    pv_bp = butterworth_bp_filter(pv_raw, fps=sampling_frequency, low=lower_frequency, high=higher_frequency)

    # Perform the FFT on selected signal
    if signal == 'raw':
        fft_pv = fft(pv_raw)
    elif signal == 'ac':
        fft_pv = fft(pv_ac)
    elif signal == 'bp':
        fft_pv = fft(pv_bp)
    else:
        assert False, "Invalid signal type for the 'GREEN' framework. Please choose one of the valid signals " \
                      "types: 'raw' (for raw G signal), 'ac' (removed DC component), or 'bp' (bandpass filtered " \
                      "raw signal with the specified lower and higher frequency) "

    # Calculate the power spectrum by taking the absolute square of the FFT
    power_spectrum = np.abs(fft_pv) ** 2

    # Calculate the corresponding frequencies
    frequencies = fftfreq(len(pv_bp), d=1 / sampling_frequency)

    # Display only the positive frequencies and corresponding power spectrum
    positive_frequencies = frequencies[:len(frequencies) // 2]  # Take only the first half
    positive_spectrum = power_spectrum[:len(power_spectrum) // 2]  # Take only the first half

    freq_range = (lower_frequency, higher_frequency)  # Frequency range

    # Find the indices corresponding to the desired frequency range
    start_idx = np.argmax(positive_frequencies >= freq_range[0])
    end_idx = np.argmax(positive_frequencies >= freq_range[1])

    # Extract the frequencies and power spectrum within the desired range
    frequencies_range = positive_frequencies[start_idx:end_idx]
    power_spectrum_range = positive_spectrum[start_idx:end_idx]

    max_idx = np.argmax(power_spectrum_range)
    f_max = frequencies_range[max_idx]
    hrGT = f_max * 60.0

    return hrGT


def green_lgi_ppgi(ground_truth_file, sampling_frequency=60, signal='bp', lower_frequency=0.8, higher_frequency=2.0):
    gtdata = pd.read_xml(ground_truth_file)
    gtTime = (gtdata.iloc[:, 0]).tolist()
    gtHR = gtdata.iloc[:, 1].tolist()
    gtTrace = gtdata.iloc[:, 2].tolist()

    pv_raw = gtTrace
    pv_ac = (np.array(pv_raw) - np.mean(pv_raw)).tolist()
    pv_bp = butterworth_bp_filter(pv_raw, fps=sampling_frequency, low=lower_frequency, high=higher_frequency)

    # Perform the FFT on selected signal
    if signal == 'raw':
        fft_pv = fft(pv_raw)
    elif signal == 'ac':
        fft_pv = fft(pv_ac)
    elif signal == 'bp':
        fft_pv = fft(pv_bp)
    else:
        assert False, "Invalid signal type for the 'GREEN' framework. Please choose one of the valid signals " \
                      "types: 'raw' (for raw G signal), 'ac' (removed DC component), or 'bp' (bandpass filtered " \
                      "raw signal with the specified lower and higher frequency) "

    # Calculate the power spectrum by taking the absolute square of the FFT
    power_spectrum = np.abs(fft_pv) ** 2

    # Calculate the corresponding frequencies
    frequencies = fftfreq(len(pv_bp), d=1 / sampling_frequency)

    # Display only the positive frequencies and corresponding power spectrum
    positive_frequencies = frequencies[:len(frequencies) // 2]  # Take only the first half
    positive_spectrum = power_spectrum[:len(power_spectrum) // 2]  # Take only the first half

    freq_range = (lower_frequency, higher_frequency)  # Frequency range

    # Find the indices corresponding to the desired frequency range
    start_idx = np.argmax(positive_frequencies >= freq_range[0])
    end_idx = np.argmax(positive_frequencies >= freq_range[1])

    # Extract the frequencies and power spectrum within the desired range
    frequencies_range = positive_frequencies[start_idx:end_idx]
    power_spectrum_range = positive_spectrum[start_idx:end_idx]

    max_idx = np.argmax(power_spectrum_range)
    f_max = frequencies_range[max_idx]
    hrGT = f_max * 60.0

    return hrGT


green_true = []
green_pred = []
# # base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
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
#     #         hrES = green_framework(input_video=vid, dataset='UBFC1')
#     #         hrGT = green_ubfc1(ground_truth_file=gt)
#     #         # print(len(hrGT), len(hrES))
#     #         print('')
#     #         green_true.append(hrGT)
#     #         green_pred.append(hrES)
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
#             hrES = green_framework(input_video=vid, dataset='UBFC2')
#             hrGT = green_ubfc2(ground_truth_file=gt)
#             # print(len(hrGT), len(hrES))
#             print('')
#             green_true.append(hrGT)
#             green_pred.append(hrES)
#
# print(mean_absolute_error(green_true, green_pred))
# # print(mean_absolute_error(green_true[8:], green_pred[8:]))
# print(green_true)
# print(green_pred)

# 16.076485124847036
# 15.14179822781124
# [73.75071797817347, 81.81818181818181, 91.26365054602184, 56.593886462882104, 68.97069872276484, 73.57243037467441, 92.20103986135183, 97.0974808324206, 110.53652230122819, 88.9505830094392, 103.94736842105263, 101.03225806451613, 112.16617210682492, 107.14285714285714, 109.38735177865613, 114.32791728212702, 67.87330316742081, 109.71258671952428, 77.67185148018062, 112.27722772277228, 92.19512195121952, 88.85630498533725, 107.07964601769912, 112.5, 106.2992125984252, 60.67415730337079, 110.20408163265306, 113.12154696132596, 100.60606060606061, 111.94968553459118, 102.95857988165682, 77.25703009373458, 100.93457943925233, 113.4567283641821, 114.987714987715, 110.580204778157, 119.29824561403508, 55.80708661417323, 111.51219512195122, 84.83063328424153, 87.86982248520711, 99.85272459499262, 98.72673849167484, 103.03633648581385, 77.64005949429847, 111.61417322834646, 91.31089904570568, 104.99258526940187, 92.02453987730061, 85.84070796460178]
# [79.95018679950188, 54.38522870331514, 59.44115156646909, 68.53879105188005, 74.87765089722676, 79.82608695652173, 48.99497487437186, 61.99524940617578, 110.53652230122819, 88.9505830094392, 92.10526315789473, 68.51612903225808, 112.16617210682492, 105.35714285714286, 80.9288537549407, 53.17577548005908, 69.68325791855204, 109.71258671952428, 62.3181133968891, 53.62462760675274, 92.19512195121952, 88.85630498533725, 56.63716814159292, 57.14285714285714, 104.5275590551181, 61.59346271705822, 110.20408163265306, 113.12154696132596, 100.60606060606061, 50.31446540880503, 56.80473372781066, 76.36901825357671, 100.93457943925233, 113.4567283641821, 114.987714987715, 60.555826426133606, 119.29824561403508, 53.84615384615384, 54.4390243902439, 84.83063328424153, 87.86982248520711, 99.85272459499262, 98.72673849167484, 69.88551518168244, 77.64005949429847, 111.61417322834646, 94.02310396785535, 111.22095897182402, 60.736196319018404, 61.06194690265487]

# Improved
# [110.53652230122819, 88.9505830094392, 103.94736842105263, 101.03225806451613, 112.16617210682492, 107.14285714285714, 109.38735177865613, 114.32791728212702, 67.87330316742081, 109.71258671952428, 77.67185148018062, 112.27722772277228, 92.19512195121952, 88.85630498533725, 107.07964601769912, 112.5, 106.2992125984252, 60.67415730337079, 110.20408163265306, 113.12154696132596, 100.60606060606061, 111.94968553459118, 102.95857988165682, 77.25703009373458, 100.93457943925233, 113.4567283641821, 114.987714987715, 110.580204778157, 119.29824561403508, 55.80708661417323, 111.51219512195122, 84.83063328424153, 87.86982248520711, 99.85272459499262, 98.72673849167484, 103.03633648581385, 77.64005949429847, 111.61417322834646, 91.31089904570568, 104.99258526940187, 92.02453987730061, 85.84070796460178]
# [105.76190476190476, 94.32142857142857, 98.42857142857143, 93.09523809523809, 111.85714285714285, 107.85714285714285, 78.0952380952381, 76.88095238095238, 67.40476190476191, 94.47619047619047, 77.54761904761905, 86.97619047619048, 93.16666666666667, 86.30952380952381, 106.28571428571428, 78.09523809523809, 104.71428571428572, 63.114285714285714, 93.59523809523809, 112.28571428571428, 99.07142857142857, 99.07142857142857, 86.64285714285712, 78.38095238095238, 70.64285714285715, 115.04761904761904, 88.28571428571428, 106.52380952380952, 112.30952380952381, 54.285714285714285, 83.28571428571428, 83.90476190476191, 87.11904761904763, 100.59523809523809, 96.45238095238096, 90.09523809523809, 83.88095238095238, 111.42857142857143, 93.59523809523809, 105.85714285714288, 91.14285714285714, 72.28571428571429]
# 8.751367456543907

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
#         # hrES = green_framework(input_video=vid, dataset='LGI_PPGI')
#         hrES = green_test(input_video=vid)
#         hrGT = green_lgi_ppgi(ground_truth_file=gt)
#         print('')
#         green_true.append(np.mean(hrGT))
#         green_pred.append(np.mean(hrES))
#
# print(green_true)
# print(green_pred)
# print(mean_absolute_error(green_true, green_pred))
# print(f"gym: {mean_absolute_error(green_true[0:6], green_pred[0:6])}")
# print(f"resting: {mean_absolute_error(green_true[6:12], green_pred[6:12])}")
# print(f"rotation: {mean_absolute_error(green_true[12:18], green_pred[12:18])}")
# print(f"talk: {mean_absolute_error(green_true[18:24], green_pred[18:24])}")

# [106.426735218509, 117.06867414349364, 104.9689883795537, 85.73280051841452, 111.99054932073243, 88.63558645845208, 67.14944042132983, 61.20961865435997, 59.51068988318272, 78.68409193330328, 72.53071253071253, 50.935645823824736, 68.45289541918756, 58.34970530451867, 61.8705035971223, 83.24642938109272, 76.90481841917874, 49.76167778836987, 72.29170336445307, 62.460567823343844, 78.66880513231757, 86.17818484596171, 75.22388059701493, 79.64601769911505]
# [57.698132256436146, 57.146414342629484, 57.012847965738764, 60.94674556213018, 58.375015249481514, 56.227327690447396, 54.73856209150327, 61.78073894609328, 60.03289473684211, 79.03587443946188, 74.81527093596058, 95.56313993174061, 65.98712446351932, 55.147058823529406, 50.4950495049505, 64.37041219649915, 53.57142857142857, 66.884661117717, 50.727739726027394, 55.25013743815283, 51.03668261562999, 56.9331983805668, 60.66176470588235, 71.78095707942774]
# 21.359992953078034
# gym: 44.569475162048654
# resting: 10.128006415756962
# rotation: 12.729376981753362
# talk: 18.01311325275318

# Improved
# [106.426735218509, 117.06867414349364, 104.9689883795537, 85.73280051841452, 111.99054932073243, 88.63558645845208, 67.14944042132983, 61.20961865435997, 59.51068988318272, 78.68409193330328, 72.53071253071253, 50.935645823824736, 68.45289541918756, 58.34970530451867, 61.8705035971223, 83.24642938109272, 76.90481841917874, 49.76167778836987, 72.29170336445307, 62.460567823343844, 78.66880513231757, 86.17818484596171, 75.22388059701493, 79.64601769911505]
# [82.4375, 79.6923076923077, 66.61111111111111, 71.9795918367347, 77.0, 78.61428571428571, 80.0909090909091, 73.11111111111111, 72.3, 93.5, 87.22222222222223, 55.6, 79.0909090909091, 68.11111111111111, 63.333333333333336, 87.6, 74.55555555555556, 57.0, 66.92857142857143, 66.8, 68.45454545454545, 73.45454545454545, 68.63636363636364, 72.91666666666667]
# 13.00222153940654
# gym: 26.414756280786026
# resting: 11.967340529588228
# rotation: 5.967234151447599
# talk: 7.659555195804305