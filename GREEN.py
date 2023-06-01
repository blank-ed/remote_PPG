"""

This module contains the framework implemented by https://opg.optica.org/oe/fulltext.cfm?uri=oe-16-26-21434&id=175396
also known as GREEN rPPG by other research papers. This is the closest implementation of the original
framework that has been proposed.

"""

from remote_PPG.utils import *
from remote_PPG.filters import *
from scipy.fft import fft, fftfreq


def green_framework(input_video, roi_type='ROI_IV', signal='bp', lower_frequency=0.8, higher_frequency=2.0):
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
    fps = get_fps(input_video)  # find the fps of the video

    pv_raw = raw_sig
    pv_ac = (np.array(pv_raw) - np.mean(pv_raw)).tolist()
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

    # Define the frequency range to plot
    freq_range = (0, 6.5)  # Frequency range from 0 Hz to 6.5 Hz

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
