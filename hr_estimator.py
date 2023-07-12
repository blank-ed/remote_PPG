from importlib import import_module
import numpy as np
from scipy.signal import find_peaks, stft, welch
from scipy.fft import rfft, rfftfreq
from remote_PPG.utils import *


def get_bpm(signal, fps, type, remove_outlier, params):

    estimator_module = import_module('remote_PPG.hr_estimator')
    estimator_method = getattr(estimator_module, type)

    if type == 'stft_estimator' and 'bpm_type' in params:
        params_copy = params.copy()
        del params_copy['bpm_type']  # Delete bpm type since stft will give continuous hr values
        hr = estimator_method(signal, fps, remove_outlier, **params_copy)
    else:
        hr = estimator_method(signal, fps, remove_outlier, **params)

    return hr


def outlier_removal(frequencies, magnitude):

    prev_hr = None
    hr_estimated = []

    for i in range(min(magnitude.shape)):
        if magnitude.shape[1] > magnitude.shape[0]:
            mask = (frequencies[i] >= 0.67) & (frequencies[i] <= 4)
            masked_frequencies = frequencies[i][mask]
            masked_magnitude = magnitude[i][mask]
        else:
            mask = (frequencies >= 0.67) & (frequencies <= 4)  # create a mask for the desired frequency range
            masked_frequencies = frequencies[mask]
            masked_magnitude = magnitude[mask, i]

        peaks, _ = find_peaks(masked_magnitude)

        if len(peaks) == 0:
            if prev_hr is not None:
                hr_estimated.append(prev_hr)
            continue

        peak_freqs = masked_frequencies[peaks]  # corresponding peaks frequencies
        peak_powers = masked_magnitude[peaks]  # corresponding peaks powers

        # For the first previous HR value
        if prev_hr is None:
            # Find the highest peak
            max_peak_index = np.argmax(peak_powers)
            max_peak_frequency = peak_freqs[max_peak_index]

            hr = max_peak_frequency * 60
            prev_hr = hr
        else:
            max_peak_index = np.argmax(peak_powers)
            max_peak_frequency = peak_freqs[max_peak_index]

            hr = max_peak_frequency * 60

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
                hr = max_peak_frequency * 60

            prev_hr = hr
        hr_estimated.append(hr)

    return hr_estimated


def detect_peaks(frequencies, magnitude):
    # Detect Peaks for each time slice
    hr = []

    for i in range(min(magnitude.shape)):
        if magnitude.shape[1] > magnitude.shape[0]:
            mask = (frequencies[i] >= 0.67) & (frequencies[i] <= 4)
            masked_frequencies = frequencies[i][mask]
            masked_magnitude = magnitude[i][mask]
        else:
            mask = (frequencies >= 0.67) & (frequencies <= 4)  # create a mask for the desired frequency range
            masked_frequencies = frequencies[mask]
            masked_magnitude = magnitude[mask, i]

        peaks, _ = find_peaks(masked_magnitude)
        if len(peaks) > 0:
            peak_freq = masked_frequencies[peaks[np.argmax(masked_magnitude[peaks])]]
            hr.append(peak_freq * 60)
        else:
            if hr:
                hr.append(hr[-1])  # append the last recorded hr value
            else:
                continue  # skip the iteration if there are no peaks and no previous hr values

    return hr


def stft_estimator(signal, fps, remove_outlier, signal_length=None, increment=None):

    if signal.ndim == 1:
        noverlap = fps * (signal_length - increment)
        nperseg = fps * signal_length  # Length of fourier window

        frequencies, times, Zxx = stft(signal, fps, nperseg=nperseg, noverlap=noverlap)  # Perform STFT
        magnitude_Zxx = np.abs(Zxx)  # Calculate the magnitude of Zxx

    elif signal.ndim == 2:
        frequencies = []
        magnitude_Zxx = []

        for window in signal:

            freqs = rfftfreq(len(window), d=1 / fps)
            magnitudes = np.abs(rfft(window)) ** 2

            frequencies.append(freqs)
            magnitude_Zxx.append(magnitudes)

    else:
        raise ValueError("Signal must be either 1D or 2D array")


    if remove_outlier:
        hr = outlier_removal(np.array(frequencies), np.array(magnitude_Zxx))
    else:
        hr = detect_peaks(np.array(frequencies), np.array(magnitude_Zxx))

    return hr


def fft_estimator(signal, fps, remove_outlier, bpm_type, signal_length=None, increment=None):

    if bpm_type == 'average':
        # Compute the positive frequencies and the corresponding power spectrum
        freqs = rfftfreq(len(signal), d=1 / fps)
        power_spectrum = np.abs(rfft(signal)) ** 2

        # Find the maximum peak between 0.75 Hz and 4 Hz
        mask = (freqs >= 0.75) & (freqs <= 4)
        filtered_power_spectrum = power_spectrum[mask]
        filtered_freqs = freqs[mask]

        peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks
        peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
        peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers

        max_peak_index = np.argmax(peak_powers)
        max_peak_frequency = peak_freqs[max_peak_index]
        hr = max_peak_frequency * 60

    elif bpm_type == 'continuous':
        if signal.ndim == 2:
            windowed_sig = signal
        else:
            windowed_sig = moving_window(sig=signal, fps=fps, window_size=signal_length, increment=increment)

        frequencies = []
        magnitude = []
        hr = []
        prev_hr = None

        for each_sig in windowed_sig:
            # Compute the positive frequencies and the corresponding power spectrum
            freqs = rfftfreq(len(each_sig), d=1 / fps)
            power_spectrum = np.abs(rfft(each_sig)) ** 2

            frequencies.append(freqs)
            magnitude.append(power_spectrum)

        if remove_outlier:
            hr = outlier_removal(np.array(frequencies), np.array(magnitude))
        else:
            for freqs, power_spectrum in zip(frequencies, magnitude):
                # Find the maximum peak between 0.75 Hz and 4 Hz
                mask = (freqs >= 0.75) & (freqs <= 4)
                filtered_power_spectrum = power_spectrum[mask]
                filtered_freqs = freqs[mask]

                peaks, _ = find_peaks(filtered_power_spectrum)  # index of the peaks

                if len(peaks) == 0:
                    if prev_hr is not None:
                        hr.append(prev_hr)
                    continue

                peak_freqs = filtered_freqs[peaks]  # corresponding peaks frequencies
                peak_powers = filtered_power_spectrum[peaks]  # corresponding peaks powers

                max_peak_index = np.argmax(peak_powers)
                max_peak_frequency = peak_freqs[max_peak_index]
                hr.append(max_peak_frequency * 60)

    return hr


def welch_estimator(signal, fps, remove_outlier, bpm_type, signal_length=None, increment=None):

    if bpm_type == 'average':
        frequencies, psd = welch(signal, fs=fps, nperseg=len(signal), nfft=8192)

        first = np.where(frequencies > 0.7)[0]
        last = np.where(frequencies < 4)[0]
        first_index = first[0]
        last_index = last[-1]
        range_of_interest = range(first_index, last_index + 1, 1)
        max_idx = np.argmax(psd[range_of_interest])
        f_max = frequencies[range_of_interest[max_idx]]
        hr = f_max * 60.0

    elif bpm_type == 'continuous':
        if signal.ndim == 2:
            windowed_sig = signal
        else:
            windowed_sig = moving_window(sig=signal, fps=fps, window_size=signal_length, increment=increment)

        frequencies = []
        magnitude = []
        hr = []

        for each_sig in windowed_sig:
            freqs, psd = welch(each_sig, fs=fps, nperseg=len(each_sig), nfft=8192)

            frequencies.append(freqs)
            magnitude.append(psd)

        if remove_outlier:
            hr = outlier_removal(np.array(frequencies), np.array(magnitude))
        else:
            for freqs, power_spectrum in zip(frequencies, magnitude):

                first = np.where(freqs > 0.7)[0]
                last = np.where(freqs < 4)[0]
                first_index = first[0]
                last_index = last[-1]
                range_of_interest = range(first_index, last_index + 1, 1)
                max_idx = np.argmax(power_spectrum[range_of_interest])
                f_max = freqs[range_of_interest[max_idx]]
                hr.append(f_max * 60.0)

    return hr

