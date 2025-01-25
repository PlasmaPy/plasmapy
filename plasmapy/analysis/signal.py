"""_summary_
"""

import numpy as np
import scipy.signal


def high_pass_filter(
    data_array: np.ndarray, cutoff: float, sampling_frequency: float, order: int
) -> np.ndarray:
    """
    Applies a high pass filter to the data array to imporve
    proper integration.

    Parameters
    ----------
    data_array: `numpy.ndarray`
       An array containing data from to which filter
       is to be applied.

    cutoff: float
       The lowest frequency passed into the filter. Frequencies
       lower than the cutoff are removed from the dataset.

    sampling_frequency: float
       The maximum frequency passed into the filter based
       on the sampling rate of the data set.

    order: int
       The order of the filter created

    Returns
    -------
    filt_arr: `numpy.ndarray`
       The original data aray with the frequencies below
       the cuttoff removed

    """
    if cutoff > sampling_frequency:
        raise ValueError("sampling frequency must be greater than cutoff value\n")

    nyq = 0.5 * sampling_frequency
    normal_cutoff = cutoff / nyq
    b, a = scipy.signal.butter(order, normal_cutoff, btype="highpass", analog=False)
    filt_arr = scipy.signal.filtfilt(b, a, data_array)
    return filt_arr
