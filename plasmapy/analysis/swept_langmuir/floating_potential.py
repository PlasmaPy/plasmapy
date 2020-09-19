"""Functionality for determining the floating potential of a Langmuir sweep."""
__all__ = ["find_floating_potential"]

import numpy as np

from collections import namedtuple
from warnings import warn
from typing import Union

from plasmapy.analysis.swept_langmuir.fit_functions import (
    ExponentialOffsetFitFunction,
    LinearFitFunction,
)

FloatingPotentialResults = namedtuple(
    "FloatingPotentialResults",
    ("vf", "vf_err", "rsq", "func", "islands", "indices"),
)


def find_floating_potential(
    voltage: np.ndarray,
    current: np.ndarray,
    threshold: int = 1,
    min_points: Union[int, float] = None,
    fit_type: str = "exponential",
):
    """
    Determines the floating potential (:math:`V_f`) for a given Current-Voltage
    (IV) curve obtained from a swept Langmuir probe.  The floating potential is
    the probe bias where the collected current equals to zero :math:`I = 0`.

    **How it works**

    #. The current array ``current`` is scanned for all points equal to zero and
       point pairs that straddle :math:`I = 0`.  This forms an array of
       "crossing-points."
    #. The crossing-points are then grouped into "crossing-islands" in based on
       the ``threshold`` keyword.

       - A new island is formed when a successive crossing-point is more (index)
         steps away from the previous crossing-point than allowed by
         ``threshold``.
       - If multiple crossing-islands are identified, then the total span of all
         crossing-islands is compared to ``min_points``.  If the span is greater
         than ``min_points`` then the function is incapable of identifying
         :math:`V_f` and will return `numpy.nan` values; otherwise, the span
         will form one larger crossing-island.

    #. To calculate the floating potential...

       - If the crossing-island contains less points than ``min_points``, then
         each side of the crossing-island is equally padded with the nearest
         neighbor points until ``min_points`` is satisfied.
       - A fit is then performed using `scipy.stats.linregress` for
         ``fit_type="linear"`` and `scipy.optimize.curve_fit` for
         ``fit_type="exponential"``.

    Parameters
    ----------

    voltage: ~numpy.ndarray
        1-D numpy array of monotonically ascending/descending probe biases
        (in volts)

    current: ~numpy.ndarray
        1-D numpy array of probe current (in amperes) corresponding to the
        :data:`voltage` array

    threshold: positive, non-zero `int`
        Max allowed index distance between crossing-points before a new
        crossing-island is formed.  That is, if `threshold=5` then consecutive
        crossing-points are considered to be in the same crossing-island if
        they are within 5 index steps of each other. (Default: 1)

    min_points: positive `int` or `float`
        Specifies the minimum number of points required for the fit to be
        applied to.

        - ``min_points = None`` (Default) The larger of 5 and
          ``factor * array_size`` is taken, where ``array_size`` is the size of
          ``voltage`` and ``factor = 0.1`` for ``fit_type = "linear"`` and
          ``0.2`` for ``"exponential"``.
        - ``min_points = 0`` The entire passed array is fitted.
        - ``min_points >= 1`` Exact minimum number of points.
        - ``0 < min_points < 0`` The minimum number of points is taken as
          ``min_points * array_size``.

    fit_type: str
        The type of curve to be fitted to the Langmuir trace.  There are two
        types of curves ``"linear"`` and ``"exponential"`` (Default).  These
        specified which `FitFunction` class should be applied to the trace.

        +-------------+--------------------------------------------------------------------------------+
        | linear      | `~plasmapy.analysis.swept_langmuir.fit_functions.LinearFitFunction`            |
        +-------------+--------------------------------------------------------------------------------+
        | exponential | `~plasmapy.analysis.swept_langmuir.fit_functions.ExponentialOffsetFitFunction` |
        +-------------+--------------------------------------------------------------------------------+

    Returns
    -------
    vf: `float` or `numpy.nan`
        The calculated floating potential (in volts).  Returns `numpy.nan` if the
        floating potential can not be determined.  How :math:`V_f` is calculated
        depends on the fit function.  This is described in the `root_solve()`
        method of the relevant fit function (e.g. the
        :meth:`~plasmapy.analysis.swept_langmuir.fit_functions.ExponentialOffsetFitFunction.root_solve`
        method of
        `~plasmapy.analysis.swept_langmuir.fit_functions.ExponentialOffsetFitFunction`).

    vf_err: `float` or `numpy.nan`
        The error associated with the floating potential calculation (in volts).
        Returns `numpy.nan` if the floating potential can not be determined.
        Like :math:`V_f`:, the calculation depends on the applied fit function.
        The `rood_solve()` method also describes how this is calculated.

    rsq: `float`
        The coefficient of determination (r-squared) value of the fit.  See the
        documentation of the `rsq` property on the associated fit function
        (e.g. the
        `~plasmapy.analysis.swept_langmuir.fit_functions.ExponentialOffsetFitFunction.rsq`
        property of
        `~plasmapy.analysis.swept_langmuir.fit_functions.ExponentialOffsetFitFunction`).

    func: sub-class of `~plasmapy.analysis.swept_langmuir.fit_functions.AbstractFitFunction`
        The callable function :math:`f(x)` repressing the fit and its results.

    islands: `List[slice]`
        List of `slice` objects representing the indices of the identified
        crossing-islands.

    indices: `slice`
        A `slice` object representing the indices of ``voltage`` and ``current``
        arrays used for the fit.

    """
    rtn = FloatingPotentialResults(
        vf=np.nan,
        vf_err=np.nan,
        rsq=None,
        func=None,
        islands=None,
        indices=None
    )._asdict()

    fit_funcs = {
        "linear": {
            "func": LinearFitFunction(),
            "min_point_factor": 0.1,
        },
        "exponential": {
            "func": ExponentialOffsetFitFunction(),
            "min_point_factor": 0.2,
        },
    }
    try:
        min_point_factor = fit_funcs[fit_type]["min_point_factor"]
        fit_func = fit_funcs[fit_type]["func"]
        rtn["func"] = fit_func
    except KeyError:
        raise KeyError(
            f"Requested fit function '{fit_type}' is  not a valid option.  "
            f"Examine kwarg 'fit_curve' for valid options."
        )

    if current.min() > 0.0 or current.max() < 0:
        warn("The Langmuir sweep has no floating potential.")

        return FloatingPotentialResults(**rtn)

    # check voltage is monotonically increasing/decreasing
    voltage_diff = np.diff(voltage)
    if not (np.all(voltage_diff >= 0) or np.all(voltage_diff <= 0)):
        warn("The voltage array is not monotonically increasing or decreasing.")

        return FloatingPotentialResults(**rtn)

    # condition kwarg threshold
    if isinstance(threshold, (int, float)):
        threshold = int(np.round(threshold))
        if threshold < 1:
            warn(f"threshold ({threshold}) is less than 1 and needs to be"
                 f" an int >= 1, using a value of 1")
            threshold = 1
    else:
        warn(f"threshold is NOT a integer >= 1, using a value of 1")

    # condition min_points
    if min_points is None:
        min_points = int(np.max([5, np.around(min_point_factor * voltage.size)]))
    elif min_points == 0:
        # this signals to use all points
        pass
    elif 0 < min_points < 1:
        min_points = int(np.round(min_points * voltage.size))
    elif min_points >= 1:
        min_points = int(np.round(min_points))
    else:
        raise ValueError(f"Got {min_points}, but 'min_points' must be an int or float "
                         f"greater than or equal to 0.")

    # find possible crossing points (cp)
    lower_vals = np.where(current <= 0, True, False)
    upper_vals = np.where(0 <= current, True, False)
    cp_exact = np.logical_and(lower_vals, upper_vals).nonzero()[0]
    cp_low2high = np.logical_and(np.roll(lower_vals, 1), upper_vals).nonzero()[0]
    cp_high2low = np.logical_and(np.roll(lower_vals, -1), upper_vals).nonzero()[0]

    # adjust for array wrapping cause by np.roll
    cp_low2high = cp_low2high[np.where(cp_low2high != 0, True, False)]
    cp_high2low = cp_high2low[np.where(cp_high2low != current.size - 1, True, False)]

    # collect all candidates
    cp_candidates = np.concatenate((
        cp_exact,
        cp_low2high,
        cp_low2high - 1,
        cp_high2low,
        cp_high2low + 1,
    ))
    cp_candidates = np.unique(cp_candidates)  # sorted and unique

    # How many crossing-islands?
    cp_intervals = np.diff(cp_candidates)
    threshold_indices = np.where(cp_intervals > threshold)[0]
    n_islands = threshold_indices.size + 1

    if min_points == 0:
        rtn["islands"] = [slice(cp_candidates[0], cp_candidates[-1] + 1)]
    elif n_islands == 1:
        rtn["islands"] = [slice(cp_candidates[0], cp_candidates[-1] + 1)]
    else:
        # There are multiple crossing points
        isl_start = np.concatenate((
            [cp_candidates[0]],
            cp_candidates[threshold_indices+1],
        ))
        isl_stop = np.concatenate((
            cp_candidates[threshold_indices]+1,
            [cp_candidates[-1]+1],
        ))
        rtn["islands"] = []
        for start, stop in zip(isl_start, isl_stop):
            rtn["islands"].append(slice(start, stop))

        # do islands fall within min_points window
        isl_window = np.abs(np.r_[rtn["islands"][-1]][-1]
                            - np.r_[rtn["islands"][0]][0]) + 1
        if isl_window > min_points:
            warn(f"Unable to determine floating potential, Langmuir sweep has "
                 f"{n_islands} crossing-islands.  Try adjusting keyword 'threshold' "
                 f"and/or smooth the current.")

            return FloatingPotentialResults(**rtn)

    # Construct crossing-island (pad if needed)
    if min_points == 0:
        # us all points
        istart = 0
        istop = voltage.size
    else:
        istart = cp_candidates[0]
        istop = cp_candidates[-1]
        iadd = (istop - istart + 1) - min_points
        if iadd < 0:
            # pad front
            ipad_2_start = ipad_2_stop = int(np.ceil(-iadd / 2.0))
            if istart - ipad_2_start < 0:
                ipad_2_stop += ipad_2_start - istart
                ipad_2_start = 0
                istart = 0
            else:
                istart -= ipad_2_start
                ipad_2_start = 0

            # pad rear
            if ((current.size - 1) - (istop + ipad_2_stop)) < 0:
                ipad_2_start += ipad_2_stop - (current.size - 1 - istop)
                istop = current.size - 1
            else:
                istop += ipad_2_stop

            # re-pad front if possible
            if ipad_2_start > 0:
                if istart - ipad_2_start < 0:
                    istart = 0
                else:
                    istart -= ipad_2_start

        if (istop - istart + 1) < min_points:
            warn(f"The number of elements in the current array "
                 f"({istop - istart + 1}) is less than 'min_points' "
                 f"({min_points}).")

    # Perform Linear Regression Fit
    volt_sub = voltage[istart:istop + 1]
    curr_sub = current[istart:istop + 1]
    fit_func.curve_fit(volt_sub, curr_sub)

    rtn["vf"], rtn["vf_err"] = fit_func.root_solve()
    rtn.update({
        "rsq": fit_func.rsq,
        "indices": slice(istart, istop + 1)
    })

    return FloatingPotentialResults(**rtn)
