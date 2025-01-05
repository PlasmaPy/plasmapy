"""
Running moments of time series.

.. attention::

   |expect-api-changes|
"""

__all__ = ["running_mean", "running_moment"]


import numbers
from collections import namedtuple

import numpy as np

_run_moment_tuple = namedtuple("Running_Moment", ["run_moment", "time"])


def running_mean(signal, radius: int):
    """
    Calculate the running mean of a sequence.

    Parameters
    ----------
    signal : 1D |array_like|
        Signal to be averaged.

    radius : int
        The number of points on either side of each point for which
        the running mean is being calculated. The window size is
        ``2 * radius + 1``.

    Returns
    -------
    1D |array_like|
        Running mean of ``signal`` with length ``len(signal) - 2 * radius``.

    Raises
    ------
    `ValueError`
        If ``len(signal) <= 2 * radius``.

    `TypeError`
        If ``radius`` is not of type `int`.

    Examples
    --------
    >>> from plasmapy.analysis.time_series.running_moments import running_mean
    >>> running_mean([1, 2, 3, 4], 1)
    array([2., 3.])
    """
    if len(signal) <= 2 * radius:
        raise ValueError("len(signal) must be bigger than 2*radius")

    if not isinstance(radius, numbers.Integral):
        raise TypeError("radius must be of type integer")

    window = 2 * radius + 1
    run_mean = np.cumsum(signal, dtype=float)
    run_mean[window:] = run_mean[window:] - run_mean[:-window]
    return run_mean[window - 1 :] / window


def running_moment(signal, radius: int, moment: int = 1, time=None):
    """
    Calculate either the running mean, standard deviation, skewness or
    excess kurtosis of a sequence.

    Parameters
    ----------
    signal : 1D |array_like|
       Signal to be averaged.

    radius : int
        The number of points on either side of each point for which
        the running moment is being calculated. The window size is
        ``2 * radius + 1`` for running mean and ``4 * radius + 1``
        for higher moments.

    moment : int
        Choose between:

        - ``1``: running mean
        - ``2``: running standard deviation
        - ``3``: running skewness
        - ``4``: running excess kurtosis

    time : 1D |array_like|, optional
        Time base of ``signal``.

    Returns
    -------
    `~collections.namedtuple`
        Contains the following attributes:

        - ``run_moment``: 1D |array_like|
            Running moment of ``signal``. The length is ``(len(signal)
            - 2 * radius)`` for the running mean or ``(len(signal) -
            4 * signal)`` for higher moments.

        - ``time``: 1D |array_like|
            Time base corresponding to ``run_moment`` if ``time`` is
            not `None`.

    Raises
    ------
    `ValueError`
        If ``moment`` is not in (1, 2, 3, 4).

    `ValueError`
        If ``signal`` and ``time`` don't have same length.

    `ValueError`
        If ``len(signal) <= 4 * radius`` and ``moment > 1``.

    Notes
    -----
    The running rms divides by ``window``, not ``(window - 1)``.

    Examples
    --------
    >>> from plasmapy.analysis.time_series.running_moments import running_moment
    >>> running_moment([1, 2, 3, 2, 1], 1, 4, [1, 2, 3, 4, 5])
    Running_Moment(run_moment=array([3.]), time=[3])
    """
    if moment not in range(1, 5):
        raise ValueError("Only first four moments implemented")

    if time is not None and (len(signal) != len(time)):
        raise ValueError("signal and time must have same length")

    if moment == 1:
        if time is not None:
            time = time[radius:-radius]
        return running_mean(signal, radius), time

    if len(signal) <= 4 * radius:
        raise ValueError("len(signal) must be bigger than 4*radius for chosen moment")

    difference = signal[radius:-radius] - running_mean(signal, radius)

    if moment == 2:
        run_moment = np.sqrt(running_mean(difference**2, radius))

    elif moment == 3:
        run_moment = (
            running_mean(difference**3, radius)
            / running_mean(difference**2, radius) ** 1.5
        )

    elif moment == 4:
        run_moment = (
            running_mean(difference**4, radius)
            / running_mean(difference**2, radius) ** 2
        )

    if time is not None:
        time = time[2 * radius : -2 * radius]

    return _run_moment_tuple(run_moment=run_moment, time=time)
