"""Functionality for determining the plasma potential of a Langmuir sweep."""
__all__ = ["find_plasma_potential_didv"]

import numpy as np

from scipy.signal import find_peaks, peak_prominences, savgol_filter


def find_plasma_potential_didv(i_probe: np.ndarray, v_probe: np.ndarray):
    win_size = np.int(0.05 * (i_probe.shape[0]))
    if (win_size % 2) == 0:
        win_size = win_size + 1
    i_p_s = savgol_filter(i_probe, win_size, 1)
    dI = np.diff(i_p_s)
    dI_s = savgol_filter(dI, win_size, 1)
    vp_index = np.argmax(dI_s)
    vp = v_probe[vp_index]
    # check if vp happens to be at a non peak locaiton
    if peak_prominences(dI_s, [vp_index])[0] == 0:
        print("CANNOT FIND PLASMA POTENTIAL. \n Vp is not a peak in dI/dV")
        vp = np.nan

        # Check that vp happens at a peak with max prominence
    peaks = find_peaks(dI_s, prominence=[.0001])
    if peak_prominences(dI_s, [vp_index])[0] < np.max(peaks[1]['prominences']):
        print("Best estimate of Vp is not the most prominent peak in dI/dV")
        vp = np.nan
    return vp
