"""
Conditional averages and conditional variances of time series.

.. attention::

   |expect-api-changes|
"""

__all__ = ["ConditionalEvents"]


import astropy.units as u
import numpy as np
from scipy.signal import find_peaks


class ConditionalEvents:
    """
    Calculate conditional average, conditional variance, peaks, arrival
    times and waiting times of events of a time series.

    Parameters
    ----------
    signal : 1D |array_like|
        Signal to be analyzed.

    time : 1D |array_like|
        Corresponding time values for ``signal``.

    lower_threshold : `float` or `~astropy.units.Quantity`
        Lower threshold for event detection.

    upper_threshold : `float` or `~astropy.units.Quantity`, default: `None`
        Upper threshold for event detection.

    reference_signal : 1D |array_like|, default: `None`
        Reference signal. If `None`, ``signal`` is the reference signal.

    length_of_return : `float`, default: `None`
        Desired length of returned data. If `None`, estimated as
        ``len(signal) / len(number_of_events) * time_step``.

    distance : `float`, default: ``0``
        Minimum distance between peaks, in units of time.

    remove_non_max_peaks : `bool`, default: `False`
        Remove events where peak is not the largest value inside window.

    Raises
    ------
    `ValueError`:
        If length of ``signal`` and ``time`` are not equal. If length of
        ``reference_signal`` and ``time`` are not equal (when
        ``reference_signal`` is provided). If ``length_of_return`` is
        greater than the length of the time span. If
        ``length_of_return`` is negative. If ``upper_threshold`` is less
        than or equal to ``lower_threshold``.

    `~astropy.units.UnitsError`:
        If astropy units of ``signal``/``reference_signal`` and either
        ``lower_threshold`` or ``upper_threshold`` do not match. If the
        units of ``time``, ``length_of_return`` and ``distance`` do not
        match.

    Notes
    -----
    The method, in its simplest form, works by finding peaks in a signal
    that fulfill a certain size threshold. Equally sized excerpts of the
    signal around every peak are then cut out and averaged. This yields
    the average shape of the events that fulfill the condition.

    A detailed analysis of the conditional averaging method is presented
    in the Master thesis by :cite:t:`nilsen:2023`.

    Examples
    --------
    .. autolink-skip:: section

    >>> from plasmapy.analysis.time_series.conditional_averaging import (
    ...     ConditionalEvents,
    ... )
    >>> cond_events = ConditionalEvents(
    ...     signal=[1, 2, 1, 1, 2, 1],
    ...     time=[1, 2, 3, 4, 5, 6],
    ...     lower_threshold=1.5,
    ... )
    >>> cond_events.time
    array([-1.0, 0.0, 1.0])
    >>> cond_events.average
    array([1., 2., 1.])
    >>> cond_events.variance
    array([1., 1., 1.])
    >>> cond_events.peaks
    array([2, 2])
    >>> cond_events.waiting_times
    array([3])
    >>> cond_events.arrival_times
    array([2, 5])
    >>> cond_events.number_of_events
    2
    """

    def __init__(
        self,
        signal,
        time,
        lower_threshold,
        *,
        upper_threshold=None,
        reference_signal=None,
        length_of_return=None,
        distance: float = 0,
        remove_non_max_peaks: bool = False,
    ) -> None:
        self._check_for_value_errors(
            distance,
            signal,
            time,
            reference_signal,
            length_of_return,
            upper_threshold,
            lower_threshold,
        )

        if reference_signal is not None:
            self._check_units_consistency(
                [reference_signal, lower_threshold, upper_threshold]
            )
        else:
            self._check_units_consistency([signal, lower_threshold, upper_threshold])

        self._check_units_consistency([time, length_of_return, distance])

        self._astropy_signal_unit = None
        self._astropy_time_unit = None

        signal, self._astropy_signal_unit = self._separate_unit_from_variable(signal)
        time, self._astropy_time_unit = self._separate_unit_from_variable(time)
        lower_threshold, _ = self._separate_unit_from_variable(lower_threshold)
        upper_threshold, _ = self._separate_unit_from_variable(upper_threshold)
        reference_signal, _ = self._separate_unit_from_variable(reference_signal)
        length_of_return, _ = self._separate_unit_from_variable(length_of_return)
        distance, _ = self._separate_unit_from_variable(distance)

        self._reference_signal_provided = True

        if reference_signal is None:
            self._reference_signal_provided = False
            reference_signal = signal.copy()

        signal = self._ensure_numpy_array(signal)
        time = self._ensure_numpy_array(time)
        reference_signal = self._ensure_numpy_array(reference_signal)

        time_step = np.diff(time).sum() / (len(time) - 1)

        peak_locations, _ = find_peaks(
            reference_signal,
            height=[lower_threshold, upper_threshold],
            distance=int(distance / time_step) + 1,
        )

        conditional_events_indices = self._separate_events(
            reference_signal, lower_threshold, upper_threshold
        )

        peak_indices = self._choose_largest_peak_per_event(
            reference_signal,
            conditional_events_indices,
            peak_locations,
        )

        if length_of_return is None:
            length_of_return = len(signal) / len(conditional_events_indices) * time_step

        self._return_time = (
            np.arange(
                -int(length_of_return / (time_step * 2)),
                int(length_of_return / (time_step * 2)) + 1,
            )
            * time_step
        )

        conditional_events = self._calculate_all_events(signal, peak_indices)

        if remove_non_max_peaks:
            if self._reference_signal_provided:
                conditional_events_reference_signal = self._calculate_all_events(
                    reference_signal, peak_indices
                )

                conditional_events, peak_indices = self._check_if_largest_value_is_peak(
                    conditional_events,
                    peak_indices,
                    conditional_events_reference_signal,
                )
            else:
                conditional_events, peak_indices = self._check_if_largest_value_is_peak(
                    conditional_events, peak_indices, None
                )

        self._conditional_average = np.mean(conditional_events, axis=0)

        self._conditional_variance = self._calculate_conditional_variance(
            conditional_events
        )

        self._peaks = signal[peak_indices]
        self._number_of_events = len(self._peaks)

        self._arrival_times = time[peak_indices]
        self._waiting_times = np.diff(self._arrival_times)

        if self._astropy_signal_unit is not None:
            self._peaks *= self._astropy_signal_unit
            self._conditional_average *= self._astropy_signal_unit

        if self._astropy_time_unit is not None:
            self._return_time *= self._astropy_time_unit
            self._arrival_times *= self._astropy_time_unit
            self._waiting_times *= self._astropy_time_unit

    @property
    def time(self):
        """
        Time values corresponding to the analysis window.

        Returns
        -------
        time : 1D |array_like|
            Time values representing the analysis window.
        """
        return self._return_time

    @property
    def average(self):
        """
        Conditional average over events.

        Returns
        -------
        average : 1D |array_like|
            Array representing the conditional average over events.
        """
        return self._conditional_average

    @property
    def variance(self):
        """
        Conditional variance over events.

        Returns
        -------
        variance : 1D |array_like|
            Array representing the conditional variance over events.

        """
        return self._conditional_variance

    @property
    def peaks(self):
        """
        Peak values of conditional events.

        Returns
        -------
        peaks : 1D |array_like|
            Peak values of conditional events.
        """
        return self._peaks

    @property
    def waiting_times(self):
        """
        Waiting times between consecutive peaks.

        Returns
        -------
        waiting_times : 1D |array_like|
            Waiting times between consecutive peaks of conditional events.
        """
        return self._waiting_times

    @property
    def arrival_times(self):
        """
        Arrival times corresponding to the conditional events.

        Returns
        -------
        arrival_times : 1D |array_like|
            Arrival times the conditional events.
        """
        return self._arrival_times

    @property
    def number_of_events(self):
        """
        Total number of conditional events.

        Returns
        -------
        number_of_events : int
            Total number of conditional events.
        """
        return self._number_of_events

    def _check_for_value_errors(
        self,
        distance,
        signal,
        time,
        reference_signal,
        length_of_return,
        upper_threshold,
        lower_threshold,
    ):
        if distance < 0:
            raise ValueError("The distance parameter can't be negative")

        if len(signal) != len(time):
            raise ValueError("Length of signal and time must be equal")

        if reference_signal is not None and len(reference_signal) != len(time):
            raise ValueError("Length of reference_signal and time must be equal")

        if length_of_return is not None:
            if length_of_return > time[-1] - time[0]:
                raise ValueError(
                    "Choose length_of_return shorter or equal to time length"
                )
            if length_of_return < 0:
                raise ValueError("The length_of_return parameter must be bigger than 0")

        if upper_threshold and upper_threshold <= lower_threshold:
            raise ValueError(
                "The upper_threshold is higher than the lower_threshold, no events will be found"
            )

    def _separate_unit_from_variable(self, variable):
        if isinstance(variable, u.Quantity):
            return variable.value, variable.unit
        return variable, None

    def _check_units_consistency(self, variables):
        # check whether all variables have an astropy unit
        first_variable_has_unit = isinstance(variables[0], u.Quantity)
        for variable in variables:
            if variable is not None and first_variable_has_unit != isinstance(
                variable, u.Quantity
            ):
                raise u.UnitsError(f"Units do not match: {variable} and {variables[0]}")

        # check whether all variables have same astropy unit
        if first_variable_has_unit:
            first_unit = variables[0].unit
            for variable in variables:
                if variable is not None and first_unit != variable.unit:
                    raise u.UnitsError(
                        f"Units do not match: {variable} and {variables[0]}"
                    )

    def _ensure_numpy_array(self, variable):
        if not isinstance(variable, np.ndarray):
            variable = np.array(variable)
        return variable

    def _separate_events(self, reference_signal, lower_threshold, upper_threshold):
        places = np.where(reference_signal > lower_threshold)[0]
        if upper_threshold:
            higher = np.where(reference_signal < upper_threshold)[0]
            places = np.intersect1d(places, higher)

        distance_between_places = np.diff(places)
        _split = np.where(distance_between_places != 1)[0]
        return np.split(places, _split + 1)

    def _choose_largest_peak_per_event(
        self,
        reference_signal,
        conditional_events_indices,
        peak_indices,
    ):
        for event in conditional_events_indices:
            peaks_in_event = np.isin(peak_indices, event)

            if peaks_in_event.sum() > 1:
                peak_ind = peak_indices[peaks_in_event]
                highest_local_peak = reference_signal[peak_ind].argmax()
                not_highest_local_peaks = np.delete(peak_ind, highest_local_peak)
                peak_indices = np.delete(
                    peak_indices, np.isin(peak_indices, not_highest_local_peaks)
                )

        return peak_indices

    def _calculate_all_events(self, signal, peak_indices):
        t_half_len = int((len(self._return_time) - 1) / 2)
        conditional_events = np.zeros([len(peak_indices), len(self._return_time)])

        for i, global_peak_loc in enumerate(peak_indices):
            low_ind = int(max(0, global_peak_loc - t_half_len))
            high_ind = int(min(len(signal), global_peak_loc + t_half_len + 1))
            single_event = signal[low_ind:high_ind]
            if low_ind == 0:
                single_event = np.append(
                    np.zeros(-global_peak_loc + t_half_len), single_event
                )
            if high_ind == len(signal):
                single_event = np.append(
                    single_event,
                    np.zeros(global_peak_loc + t_half_len + 1 - len(signal)),
                )

            conditional_events[i, :] = single_event

        return conditional_events

    def _check_if_largest_value_is_peak(
        self, conditional_events, peak_indices, conditional_events_reference_signal
    ):
        def is_middle_value_highest(sequence):
            middle_index = len(sequence) // 2
            return np.max(sequence[middle_index]) == np.max(sequence)

        checked_conditional_events = []
        checked_peak_indices = []

        if self._reference_signal_provided:
            for event, peak, reference_event in zip(
                conditional_events,
                peak_indices,
                conditional_events_reference_signal,
                strict=False,
            ):
                if is_middle_value_highest(reference_event):
                    checked_conditional_events.append(event)
                    checked_peak_indices.append(peak)
        else:
            for event, peak in zip(conditional_events, peak_indices, strict=False):
                if is_middle_value_highest(event):
                    checked_conditional_events.append(event)
                    checked_peak_indices.append(peak)

        return np.array(checked_conditional_events), np.array(checked_peak_indices)

    def _calculate_conditional_variance(self, conditional_events):
        return self._conditional_average**2 / np.mean(conditional_events**2, axis=0)
