import numpy as np
from numpy.fft import fftfreq
from scipy.fft import fft
from GREEN import *
from CHROM import *
from POS import *
from ICA_framework.jadeR import jadeR
from ICA_framework.ICA import *
import os
import ast

def chrom_test(raw_sig):
    fps = 30

    N = len(raw_sig)
    H = np.zeros(N)
    l = int(fps*1.6)

    normalized_signal = (np.array(raw_sig) - np.mean(raw_sig))
    normalized = [normalized_signal[:, i] for i in range(0, 3)]

    b, a = butter(4, Wn=[0.67, 4.0], fs=fps, btype='bandpass')
    filtered = np.array([filtfilt(b, a, x) for x in normalized])
    transposed_list = list(map(list, zip(*filtered)))

    # filtered = butterworth_bp_filter(normalized_signal, fps=fps, low=0.67, high=4.0)

    window = moving_window(transposed_list, fps=fps, window_size=1.6, increment=0.8)

    for enum, each_window in enumerate(window):
        # normalized = np.array([each_window[:, i] for i in range(0, 3)])
        # normalized = normalize(signal=each_window, framework='CHROM')  # Normalize each windowed segment

        # Build two orthogonal chrominance signals
        Xs = 3 * np.array(each_window)[:,0] - 2 * np.array(each_window)[:,1]
        Ys = 1.5 * np.array(each_window)[:,0] + np.array(each_window)[:,1] - 1.5 * np.array(each_window)[:,2]

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

    return hr


def chrom_test_2(raw_sig):

    normalized_signal = (np.array(raw_sig) - np.mean(raw_sig)) / np.std(raw_sig)
    normalized = [normalized_signal[:, i] for i in range(0, 3)]

    b, a = butter(4, Wn=[0.67, 4.0], fs=30, btype='bandpass')
    filtered = np.array([filtfilt(b, a, x) for x in normalized])
    transposed_list = list(map(list, zip(*filtered)))

    window = moving_window(transposed_list, fps=30, window_size=10, increment=1)
    hr = []
    for enum, each_window in enumerate(window):

        # Build two orthogonal chrominance signals
        Xs = 3 * np.array(each_window)[:,0] - 2 * np.array(each_window)[:,1]
        Ys = 1.5 * np.array(each_window)[:,0] + np.array(each_window)[:,1] - 1.5 * np.array(each_window)[:,2]

        alpha = np.std(Xs) / np.std(Ys)
        S = Xs - alpha * Ys

        filtered_S = filtfilt(b, a, S)
        frequencies, psd = welch(filtered_S, fs=30, nperseg=len(S), nfft=4096)

        first = np.where(frequencies > 0.67)[0]
        last = np.where(frequencies < 4)[0]
        first_index = first[0]
        last_index = last[-1]
        range_of_interest = range(first_index, last_index + 1, 1)
        max_idx = np.argmax(psd[range_of_interest])
        f_max = frequencies[range_of_interest[max_idx]]
        hr.append(f_max * 60.0)

    return hr


def chrom_test_3(raw_sig):
    fps = 30

    N = len(raw_sig)
    H = np.zeros(N)
    l = int(fps * 1.6)

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

        start = enum * (l // 2)
        end = enum * (l // 2) + l

        if end > len(raw_sig):
            H[len(raw_sig) - l:len(raw_sig)] = H[len(raw_sig) - l:len(raw_sig)] + SWin
        else:
            H[start:end] = H[start:end] + SWin

    # Compute STFT
    noverlap = fps * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    nperseg = fps * 12  # Length of fourier window (12 seconds as per the paper)

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

    # fps = 30
    #
    # N = len(raw_sig)
    # H = np.zeros(N)
    # l = int(fps * 1.6)
    #
    # # Coefficients of FIR bandpass filter
    # filter_coefficients = firwin(numtaps=int(3 * (fps // 0.67)), cutoff=[0.67, 4.0], fs=fps, pass_zero=False, window='hamming')
    #
    # for n in range(0, N, 24):
    #     m = n - l
    #     if n - l + 1 > 0:
    #         # Temporal normalization
    #         Cn = np.array(raw_sig[m:n]) / np.mean(np.array(raw_sig[m:n]))
    #
    #         # Projection
    #         Xs = 3 * Cn[:, 0] - 2 * Cn[:, 1]  # 3Rn-2Gn
    #         Ys = 1.5 * Cn[:, 0] + Cn[:, 1] - 1.5 * Cn[:, 2]  # 1.5Rn+Gn-1.5Bn
    #
    #         Xf = filtfilt(filter_coefficients, 1, Xs, padlen=len(Xs)-1)
    #         Yf = filtfilt(filter_coefficients, 1, Ys, padlen=len(Ys)-1)
    #
    #         alpha = np.std(Xf) / np.std(Yf)
    #         S = Xf - alpha * Yf
    #
    #         SWin = np.multiply(S, windows.hann(len(S)))
    #
    #         # Overlap-Adding
    #         H[m:n] += SWin
    #
    # # Compute STFT
    # noverlap = fps * (12 - 1)  # Does not mention the overlap so incremented by 1 second (so ~91% overlap)
    # nperseg = fps * 12  # Length of fourier window (12 seconds as per the paper)
    # frequencies, times, Zxx = stft(H, fps, nperseg=nperseg, noverlap=noverlap)  # Perform STFT
    # magnitude_Zxx = np.abs(Zxx)  # Calculate the magnitude of Zxx
    # # Detect Peaks for each time slice
    # hr = []
    # for i in range(magnitude_Zxx.shape[1]):
    #     peaks, _ = find_peaks(magnitude_Zxx[:, i])
    #     if len(peaks) > 0:
    #         peak_freq = frequencies[peaks[np.argmax(magnitude_Zxx[peaks, i])]]
    #         hr.append(peak_freq * 60)
    #     else:
    #         hr.append(None)

    return hr


def pos_test(raw_sig):
    fps = 30

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

            S1_filtered = fir_bp_filter(signal=S1, fps=fps, low=0.67, high=4.0)  # MOD ---------------------------------
            S2_filtered = fir_bp_filter(signal=S2, fps=fps, low=0.67, high=4.0)  # MOD ---------------------------------

            alpha = np.std(S1_filtered) / np.std(S2_filtered)
            h = S1_filtered + alpha * S2_filtered

            # Overlap-Adding
            H[m:n + 1] += (h - np.mean(h))

    # H =

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


def ica_test(raw_sig):
    fps = 30

    # normalized_signal = (np.array(raw_sig) - np.mean(raw_sig))
    # normalized = [normalized_signal[:, i] for i in range(0, 3)]
    #
    # b, a = butter(4, Wn=[0.67, 4.0], fs=fps, btype='bandpass')
    # filtered = np.array([filtfilt(b, a, x) for x in normalized])
    # transposed_list = list(map(list, zip(*filtered)))

    # signal windowing with 96.7% overlap
    windowed_sig = moving_window(sig=raw_sig, fps=fps, window_size=30, increment=1)
    hrES = []

    prev_hr = None  # Previous HR value
    for sig in windowed_sig:
        normalized = normalize(sig, framework='ICA')  # normalize the windowed signal

        # signal = np.array([np.array(sig)[:, i] for i in range(0, 3)])

        # Apply JADE ICA algorithm and select the second component
        W = jadeR(normalized, m=3)
        bvp = np.array(np.dot(W, normalized))
        bvp = bvp[1].flatten()
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

    return hrES


def green_test(raw_sig):
    fps = 30

    filtered = butterworth_bp_filter(raw_sig, fps=fps, low=0.8, high=2.0)
    normalized = (np.array(filtered) - np.mean(filtered)) / np.std(filtered)
    # filtered = butterworth_bp_filter(normalized, fps=fps, low=0.8, high=2.0)

    # pv_raw = raw_sig
    # pv_ac = (np.array(pv_raw) - np.mean(pv_raw)).tolist()
    # pv_bp = butterworth_bp_filter(pv_raw, fps=fps, low=0.8, high=2.0)
    #
    # fft_pv_raw = fft(pv_raw)
    # fft_pv_ac = fft(pv_ac)
    # fft_pv_bp = fft(pv_bp)

    # Calculate the power spectrum by taking the absolute square of the FFT
    power_spectrum = np.abs(fft(normalized)) ** 2

    # Calculate the corresponding frequencies
    frequencies = fftfreq(len(normalized), d=1 / fps)

    # Display only the positive frequencies and corresponding power spectrum
    positive_frequencies = frequencies[:len(frequencies) // 2]  # Take only the first half
    positive_spectrum = power_spectrum[:len(power_spectrum) // 2]  # Take only the first half

    freq_range = (0.8, 2.0)  # Frequency range

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


raw_sig = []
with open('green_forehead_sig.txt', 'r') as f:
    read = f.readlines()
    for x in read:
        raw_sig.append(ast.literal_eval(x))

true = []
pred = []
base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
# base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
for sub_folders in os.listdir(base_dir):
    if sub_folders == 'UBFC2':
        for enum, folders in enumerate(os.listdir(os.path.join(base_dir, sub_folders))):
            subjects = os.path.join(base_dir, sub_folders, folders)
            for each_subject in os.listdir(subjects):
                # if each_subject.endswith('.avi'):
                #     print(enum)
                #     raw_sig = extract_raw_sig(os.path.join(subjects, each_subject), framework='GREEN', ROI_type='ROI_I')
                #     with open('green_forehead_sig.txt', 'a') as f:
                #         f.write(str(raw_sig))
                #         f.write('\n')
                if each_subject.endswith('.txt'):
                    gt = os.path.join(subjects, each_subject)
                    print(enum, gt)
                    hrGT = green_ubfc2(ground_truth_file=gt)

                    raw_signal = raw_sig[enum]
                    hrES = green_test(raw_signal)

                    true.append(np.mean(hrGT))
                    pred.append(np.mean(hrES))

print(true)
print(pred)
print(mean_absolute_error(true, pred))
