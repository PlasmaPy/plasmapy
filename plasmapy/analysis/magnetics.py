"""_summary_
   
"""
__all__ = []

from warnings import warn
import scipy.integrate as sp
from typing import Tuple
import numpy as np


def compute_bfield(bdot_data: np.ndarray, times: np.ndarray, loop_area: float) -> np.ndarray:
    """
    Returns array of the magnetic field and the corresponding time array

    Parameters
    ----------
    bdot_data: `numpy.ndarray`
        A data array containing voltage fluctuations from a bdot probe. The
        array values are proportional to time changing magnetic field via Faradays Law

    times: `numpy.ndarray`
        The time series to the data collection.

    loop_area: float
        The area through which the changing flux is measured. 

    Returns
    -------
    field_arr: `numpy.ndarray`
        The array containing the magnetic field fluctuations. 

    new_time: `numpy.ndarray`
        The corresponding time series.   
    """
    if len(bdot_data) != len(times):
        raise Exception("length of time and voltage arrays in not equal\n")

    bdot_data /= loop_area
    field_arr = sp.cumtrapz(bdot_data, times, initial=0)  # Tesla

    return field_arr
