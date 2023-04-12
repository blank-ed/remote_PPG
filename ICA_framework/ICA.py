"""

This module contains the framework implemented by https://opg.optica.org/oe/fulltext.cfm?uri=oe-18-10-10762&id=199381
which is also known as ICA rPPG by other research papers. This is the closest implementation of the original framework
proposed by them.

"""

from remote_PPG.utils import *
from remote_PPG.filters import *
from jadeR import jadeR as jadeR
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks


def ica_framework(input_video, comp=1, hr_change_threshold=12):
    """
    :param input_video:
        This takes in an input video file
    :param comp:
        Output ICA component to be selected. From literature, the second component is selected since
        it typically contains a strong plethysmographic signal
    :param hr_change_threshold:
        The threhold value change between the previous determined HR value and the next HR value.
        If the difference between them is greater than the threshold, then the next highest power
        and its corresponding frequency (HR value) is determined
    :return:
        Returns the estimated heart rate of the input video based on ICA framework
    """
    raw_sig = VJ_face_detector(input_video)  # get the raw RGB signals
    fps = get_fps(input_video)  # find the fps of the video

    # signal windowing with 96.7% overlap
    windowed_sig = moving_window(sig=raw_sig, fps=fps, window_size=30, increment=1)
    hrES = []

    prev_hr = None  # Previous HR value
    for sig in windowed_sig:
        normalized = normalize(sig)  # normalize the windowed signal

        # Apply JADE ICA algorithm and select the second component
        W = jadeR(normalized)
        bvp = np.array(np.dot(W, normalized))
        bvp = bvp[comp].flatten()

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
            while abs(prev_hr - hr) >= hr_change_threshold:
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

