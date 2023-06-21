import numpy as np
from numpy.fft import fftfreq
from scipy.fft import fft
from GREEN import *
from CHROM import *
from LiCVPR import *
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


def licvpr_test(raw_green_sig, raw_bg_green_signal, heart_rate_calculation_mode='average', hr_interval=None, dataset=None):
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

    fps = 30

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


raw_bg_signals_ubfc2 = []
with open('UBFC2.txt', 'r') as f:
    lines = f.readlines()
    for x in lines:
        raw_bg_signals_ubfc2.append(ast.literal_eval(x))


raw_sig = []
with open('green_forehead_single_pixel_sig.txt', 'r') as f:
    read = f.readlines()
    for x in read:
        raw_sig.append(ast.literal_eval(x))

true = []
pred = []
# base_dir = r'C:\Users\ilyas\Desktop\VHR\Datasets\UBFC Dataset'
base_dir = r'C:\Users\Admin\Desktop\UBFC Dataset\UBFC_DATASET'
for sub_folders in os.listdir(base_dir):
    if sub_folders == 'UBFC2':
        for enum, folders in enumerate(os.listdir(os.path.join(base_dir, sub_folders))):
            subjects = os.path.join(base_dir, sub_folders, folders)
            for each_subject in os.listdir(subjects):
                # if each_subject.endswith('.avi'):
                #     print(enum)
                #     raw_sig = extract_raw_sig(os.path.join(subjects, each_subject), framework='GREEN', ROI_type='ROI_II')
                #     with open('green_forehead_single_pixel_sig.txt', 'a') as f:
                #         f.write(str(raw_sig))
                #         f.write('\n')
                if each_subject.endswith('.txt'):
                    gt = os.path.join(subjects, each_subject)
                    print(enum, gt)
                    # hrGT = licvpr_ubfc2(ground_truth_file=gt)

                    raw_signal = raw_sig[enum]
                    hrES = licvpr_test(np.array(raw_signal)[:, 1], raw_bg_green_signal=raw_bg_signals_ubfc2[enum],
                                    heart_rate_calculation_mode='continuous', hr_interval=None, dataset='UBFC2')
                    hrGT = licvpr_ubfc2(ground_truth_file=gt, heart_rate_calculation_mode='continuous', sampling_frequency=30,
                                        hr_interval=None)

                    true.append(np.mean(hrGT))
                    pred.append(np.mean(hrES))

print(true)
print(pred)
print(mean_absolute_error(true, pred))
