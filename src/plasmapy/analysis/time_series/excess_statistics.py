"""
Performing excess statistics analyses of time series.

.. attention::

   |expect-api-changes|
"""

__all__ = ["ExcessStatistics"]


import numbers
from collections.abc import Iterable

import astropy.units as u
import numpy as np


class ExcessStatistics:
    """
    Calculate total time, number of upwards crossings, average time and
    root-mean-square time above given thresholds of a sequence.

    Parameters
    ----------
    signal : 1D |array_like|
        Signal to be analyzed.

    thresholds : 1D |array_like|
        Threshold values.

    time_step : int
        Time step of ``signal``.

    Raises
    ------
    `ValueError`
        If ``time_step`` ≤ 0.

    Examples
    --------
    >>> from plasmapy.analysis.time_series.excess_statistics import ExcessStatistics
    >>> signal = [0, 0, 2, 2, 0, 4]
    >>> thresholds = [1, 3, 5]
    >>> time_step = 1
    >>> excess_statistics = ExcessStatistics(signal, thresholds, time_step)
    >>> excess_statistics.total_time_above_threshold
    [3, 1, 0]
    >>> excess_statistics.number_of_crossings
    [2, 1, 0]
    >>> excess_statistics.average_times
    [np.float64(1.5), np.float64(1.0), 0]
    >>> excess_statistics.rms_times
    [np.float64(0.5), np.float64(0.0), 0]
    """

    def __init__(self, signal, thresholds, time_step) -> None:
        if time_step <= 0:
            raise ValueError("time_step must be positive")

        # make sure thresholds is an iterable
        if not isinstance(thresholds, Iterable):
            thresholds = [thresholds]

        self._total_time_above_threshold = []
        self._number_of_crossings = []
        self._average_times = []
        self._rms_times = []
        self.events_per_threshold = {}

        self._calculate_excess_statistics(signal, thresholds, time_step)

    def _calculate_excess_statistics(self, signal, thresholds, time_step) -> None:
        for threshold in thresholds:
            indices_above_threshold = np.where(np.array(signal) > threshold)[0]

            if len(indices_above_threshold) == 0:
                self._times_above_threshold = []
                self._total_time_above_threshold.append(0)
                self._number_of_crossings.append(0)
                self._average_times.append(0)
                self._rms_times.append(0)

            else:
                self._total_time_above_threshold.append(
                    time_step * len(indices_above_threshold)
                )

                distances_to_next_index = (
                    indices_above_threshold[1:] - indices_above_threshold[:-1]
                )
                split_indices = np.where(distances_to_next_index != 1)[0]
                event_lengths = np.split(distances_to_next_index, split_indices)

                # set correct length for first event
                event_lengths[0] = np.append(event_lengths[0], 1)

                self._times_above_threshold = [
                    time_step * len(event_lengths[i]) for i in range(len(event_lengths))
                ]

                if isinstance(time_step, u.Quantity):
                    self._times_above_threshold *= time_step.unit

                self._number_of_crossings.append(len(event_lengths))
                if indices_above_threshold[0] == 0:
                    # Don't count the first event if there is no crossing.
                    self._number_of_crossings[-1] -= 1

                self._average_times.append(np.mean(self._times_above_threshold))
                self._rms_times.append(np.std(self._times_above_threshold))

            self.events_per_threshold.update({threshold: self._times_above_threshold})

    def hist(self, bins: int = 32):
        """
        Computes the probability density function of the time above each value
        in ``thresholds``.

        Parameters
        ----------
        bins : int, default: 32
            The number of bins in the estimation of the PDF above ``thresholds``.

        Returns
        -------
        hist: 2D `~numpy.ndarray`, shape (``thresholds.size``, ``bins`` )
            For each value in ``thresholds``, returns the estimated PDF of time
            above threshold.

        bin_centers: 2D `~numpy.ndarray`, shape (``thresholds.size``, ``bins`` )
            Bin centers for ``hist``.

        Raises
        ------
        `TypeError`
            If ``bins`` is not a positive integer.

        Examples
        --------
        >>> from plasmapy.analysis.time_series.excess_statistics import ExcessStatistics
        >>> signal = [0, 0, 2, 0, 4]
        >>> thresholds = [1, 3, 5]
        >>> time_step = 1
        >>> excess_statistics = ExcessStatistics(signal, thresholds, time_step)
        >>> excess_statistics.hist(2)
        (array([[0., 2.],
           [0., 2.],
           [0., 0.]]), array([[0.75, 1.25],
           [0.75, 1.25],
           [0.  , 0.  ]]))
        """

        if not isinstance(bins, numbers.Integral):
            raise TypeError("bins must be an integer")

        hist = np.zeros((len(self.events_per_threshold), bins))
        bin_centers = np.zeros((len(self.events_per_threshold), bins))

        for i, threshold in enumerate(self.events_per_threshold.keys()):
            if len(self.events_per_threshold[threshold]) >= 1:
                hist[i, :], bin_edges = np.histogram(
                    self.events_per_threshold[threshold], bins=bins, density=True
                )
                bin_centers[i, :] = (bin_edges[1:] + bin_edges[:-1]) / 2
        return hist, bin_centers

    @property
    def total_time_above_threshold(self):
        """
        Total time above threshold(s).

        Returns
        -------
        total_time_above_threshold: 1D |array_like|
            Total time above threshold for each value in ``thresholds``.
        """

        return self._total_time_above_threshold

    @property
    def number_of_crossings(self):
        """
        Total number of upwards crossings for threshold(s).

        Returns
        -------
        number_of_crossings: 1D |array_like|
            Total number of upwards crossings for each value in ``thresholds``.
        """

        return self._number_of_crossings

    @property
    def average_times(self):
        """
        Average time above threshold(s).

        Returns
        -------
        average_times: 1D |array_like|
            Average time above each value in ``thresholds``.
        """

        return self._average_times

    @property
    def rms_times(self):
        """
        Root-mean-square values of time above threshold(s).

        Returns
        -------
        rms_times: 1D |array_like|
            Root-mean-square values of time above each value in ``thresholds``.
        """

        return self._rms_times
