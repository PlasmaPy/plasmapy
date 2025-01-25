"""
Defines the magnetics analysis module as part of `plasmapy.diagnostics`.
"""

__all__ = [

]

from warnings import warn
import scipy.io as spio
import scipy.integrate as sp
from scipy import signal
import matplotlib.pylab as plt
import numpy as np
import astropy.units as u


class Magnetics:
    """
    Group together values for magnetic analysis
    """

    def __init__(self):
        print("Class created")

    def bdot_field(self, bdot, tloop_area, time_s, unit_flag="Gauss"):
        """_summary_ : Function to compute magnetic field for bdots

        Args:
            Bdot (array): - volts  
            tloop_area (float): - meters squared 
            times_s (array): - seconds   

        Returns:
            field_arr : array of magnetic field values for the bdots
        """
        if len(bdot) != len(time_s):
            raise Exception("Length of time and voltage arrays in not equal")

        bdot[:] = [x / tloop_area for x in bdot]
        field_arr = sp.cumtrapz(bdot, time_s)  # Tesla

        if unit_flag.lower() == "tesla":
            return np.array(field_arr) * u.tesla
        else:
            return np.array(field_arr*1e4) * u.gauss  # Gauss

    def mean_sub(self, data_array, start_idx, end_idx):
        """_summary_ : Function which preforms mean subtraction

        Args:     
            data_array (array): array containing values from which to mean sub
            start_idx (int): idx of where to start obtaining mean from
            end_idx (int): idx of where to stop obtaining mean from

        Returns:
            data_array (array): array after mean_sub is preformed 
        """
        data_array = np.array(data_array)
        meansub = np.mean(data_array[start_idx:end_idx])
        data_array[:] = [x - meansub for x in data_array]
        return data_array

    def band_pass_filter(self, data_array, cutoff, sampling_frequency, order):
        """_summary_ : Function which applies a band_pass_filter

        Args:
            data_array (array): _description_
            cuttoff (_type_): _description_
            sampling_frequency (_type_): _description_
            order (_type_): _description_

        Returns:
            filtered_arr : array with filter applied 
        """
        if cutoff > sampling_frequency:
            raise ValueError(
                "Sampling frequency must be greater than cutoff Value")
        nyq = 0.5 * sampling_frequency
        normal_cutoff = cutoff/nyq
        b, a = signal.butter(order, normal_cutoff,
                             btype='highpass', analog=False)
        filtered_arr = signal.filtfilt(b, a, data_array)
        return filtered_arr
