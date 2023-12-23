"""
Functionality for determining the floating potential of a Langmuir
sweep.
"""
__all__ = ["find_floating_potential", "plot_floating_potential", "VFExtras"]
__aliases__ = ["find_vf_"]

import numbers
import numpy as np

from typing import NamedTuple, Optional, Union
from warnings import warn

from plasmapy.analysis import fit_functions as ffuncs
from plasmapy.analysis.swept_langmuir.helpers import check_sweep
from plasmapy.utils.exceptions import PlasmaPyWarning

if False:
    # noqa
    # for annotation, does not need real import
    from matplotlib.axes import Axes

__all__ += __aliases__


class VFExtras(NamedTuple):
    """
    Create a `tuple` containing the extra parameters calculated by
    `~plasmapy.analysis.swept_langmuir.floating_potential.find_floating_potential`.
    """

    vf_err: Optional[float]
    """
    Alias for field number 0, the error in the calculated floating
    potential from the floating potential curve fit.
    """

    rsq: Optional[float]
    """
    Alias for field number 1, the r-squared value of the ion-saturation
    curve fit.
    """

    fitted_func: Optional[ffuncs.AbstractFitFunction]
    """
    Alias for field number 2, the :term:`fit-function` fitted during
    the floating potential curve fit.
    """

    islands: Optional[list[slice]]
    """
    Alias for field number 3, a list of `slice` objects representing
    the indices of the identified crossing-islands discovered during
    the floating potential curve fit.
    """

    fitted_indices: Optional[slice]
    """
    Alias for field number 4, the indices used in the floating potential
    curve fit.
    """


def find_floating_potential(  # noqa: C901, PLR0912, PLR0915
    voltage: np.ndarray,
    current: np.ndarray,
    threshold: int = 1,
    min_points: Optional[Union[int, float]] = None,
    fit_type: str = "exponential",
) -> tuple[np.floating, VFExtras]:
    """
    Determine the floating potential (:math:`V_f`) for a given
    current-voltage (IV) curve obtained from a swept Langmuir probe.

    The floating potential is the probe bias where the collected
    current equals zero :math:`I = 0`.  (For additional details see
    the **Notes** section below.)

    **Aliases:**
    :func:`~plasmapy.analysis.swept_langmuir.floating_potential.find_vf_`

    Parameters
    ----------
    voltage: `numpy.ndarray`
        1-D numpy array of monotonically ascending probe biases
        (should be in volts)

    current: `numpy.ndarray`
        1-D numpy array of probe current (should be in amperes)
        corresponding to the ``voltage`` array

    threshold: positive, non-zero `int`
        Max allowed index distance between crossing-points before a new
        crossing-island is formed.  That is, if ``threshold=5`` then
        consecutive crossing-points are considered to be in the same
        crossing-island if they are within 5 index steps of each other.
        (Default: 1)

    min_points: positive `int` or `float`
        Minimum number of data points required for the fitting to be
        applied to.  See **Notes** section below for additional details.
        The following list specifies the optional values:

        - ``min_points = None`` (Default) The largest of 5 and
          ``factor * array_size`` is taken, where ``array_size`` is the
          size of ``voltage`` and ``factor = 0.1`` for
          ``fit_type = "linear"`` and ``0.2`` for ``"exponential"``.
        - ``min_points = numpy.inf`` The entire passed array is fitted.
        - ``min_points >= 1`` Exact minimum number of points.
        - ``0 < min_points < 0`` The minimum number of points is taken
          as ``min_points * array_size``.

    fit_type: str
        The type of curve to be fitted to the Langmuir trace,
        ``"linear"`` or ``"exponential"`` (Default).  This selects
        which ``FitFunction`` class should be applied to the trace.

        +-------------+----------------------------------------------------------+
        | linear      | `~plasmapy.analysis.fit_functions.Linear`                |
        +-------------+----------------------------------------------------------+
        | exponential | `~plasmapy.analysis.fit_functions.ExponentialPlusOffset` |
        +-------------+----------------------------------------------------------+

    Returns
    -------
    vf: `float` or `numpy.nan`
        The calculated floating potential (same units as the
        ``voltage`` array).  Returns `numpy.nan` if the floating
        potential can not be determined.

    extras: `VFExtras`
        Additional information from the fit:

        ``extras.vf_err`` (`float` or `numpy.nan`)
            The uncertainty associated with the floating potential
            calculation (units same as ``vf``).  Returns `numpy.nan`
            if the floating potential can not be determined.  Like
            :math:`V_f`:, the calculation depends on the applied fit
            function.  The ``root_solve()`` method also describes how
            this is calculated.

        ``extras.rsq`` (`float`)
            The coefficient of determination (r-squared) value of the
            fit.  See the documentation of the ``rsq`` property on the
            associated fit function (e.g. the
            `~plasmapy.analysis.fit_functions.ExponentialPlusOffset.rsq`
            property of
            `~plasmapy.analysis.fit_functions.ExponentialPlusOffset`).

        ``extras.fitted_func`` (:term:`fit-function`)
            The computed :term:`fit-function` specified by ``fit_type``.

        ``extras.islands`` (``List[slice]``)
            List of `slice` objects representing the indices of the
            identified crossing-islands.

        ``extras.fitted_indices`` (`slice`)
            A `slice` object representing the indices of the ``voltage``
            and ``current`` arrays used for the fit.

    Notes
    -----
    The internal functionality works like:

    1. The current array ``current`` is scanned for all points equal to
       zero and point pairs that straddle :math:`I = 0`.  This forms an
       array of "crossing-points."
    2. The crossing-points are then grouped into "crossing-islands" in
       based on the ``threshold`` keyword.

       - A new island is formed when a successive crossing-point is more
         (index) steps away from the previous crossing-point than
         allowed by ``threshold``.
       - If multiple crossing-islands are identified, then the span
         from the first point in the first island to the last point in
         the last island is compared to ``min_points``.  If the span is
         less than or equal to ``min_points``, then that span is taken
         as one larger crossing-island for the fit; otherwise, the
         function is incapable of identifying :math:`V_f` and will
         return `numpy.nan` values.

    3. To calculate the floating potential...

       - If the crossing-island contains fewer points than
         ``min_points``, then each side of the crossing-island is
         equally padded with the nearest neighbor points until
         ``min_points`` is satisfied.
       - A fit is then performed using `scipy.stats.linregress` for
         ``fit_type="linear"`` and `scipy.optimize.curve_fit` for
         ``fit_type="exponential"``.
    """
    rtn_extras = VFExtras(
        vf_err=np.nan, rsq=None, fitted_func=None, islands=None, fitted_indices=None
    )._asdict()

    _settings = {
        "linear": {"func": ffuncs.Linear, "min_point_factor": 0.1},
        "exponential": {"func": ffuncs.ExponentialPlusOffset, "min_point_factor": 0.2},
    }
    try:
        min_point_factor = _settings[fit_type]["min_point_factor"]
        fit_func = _settings[fit_type]["func"]()
        rtn_extras["fitted_func"] = fit_func
    except KeyError as ex:
        raise ValueError(
            f"Requested fit '{fit_type}' is not a valid option.  Valid options "
            f"are {list(_settings.keys())}."
        ) from ex

    # check voltage and current arrays
    voltage, current = check_sweep(voltage, current, strip_units=True)

    # condition kwarg threshold
    if not isinstance(threshold, numbers.Integral):
        raise TypeError(
            f"Keyword 'threshold' is of type {type(threshold)}, expected an int "
            f"int >= 1."
        )
    elif threshold < 1:
        raise ValueError(
            f"Keyword 'threshold' has value ({threshold}) less than 1, "
            f"value must be an int >= 1."
        )

    # condition min_points
    if min_points is None:
        min_points = int(np.max([5, np.around(min_point_factor * voltage.size)]))
    elif not isinstance(min_points, (float, np.floating, int, np.integer)):
        raise TypeError(
            f"Argument 'min_points' is wrong type '{type(min_points)}', expecting "
            f"an int or float."
        )
    elif np.isinf(min_points):
        # this signals to use all points
        pass
    elif 0 < min_points < 1:
        min_points = int(np.round(min_points * voltage.size))
    elif min_points >= 1:
        min_points = int(np.round(min_points))
    else:
        raise ValueError(f"Argument 'min_points' can not be negative ({min_points}).")

    # find possible crossing points (cp)
    lower_vals = current < 0
    upper_vals = current > 0
    cp_exact = (current == 0.0).nonzero()[0]
    cp_low2high = np.logical_and(np.roll(lower_vals, 1), upper_vals).nonzero()[0]
    cp_high2low = np.logical_and(np.roll(lower_vals, -1), upper_vals).nonzero()[0]

    # adjust for array wrapping cause by np.roll
    cp_low2high = cp_low2high[cp_low2high != 0]
    cp_high2low = cp_high2low[cp_high2low != current.size - 1]

    # collect all candidates
    cp_candidates = np.concatenate(
        (cp_exact, cp_low2high, cp_low2high - 1, cp_high2low, cp_high2low + 1)
    )
    cp_candidates = np.unique(cp_candidates)  # sorted and unique

    # How many crossing-islands?
    cp_intervals = np.diff(cp_candidates)
    threshold_indices = np.where(cp_intervals > threshold)[0]
    n_islands = threshold_indices.size + 1

    if np.isinf(min_points) or n_islands == 1:
        rtn_extras["islands"] = [slice(cp_candidates[0], cp_candidates[-1] + 1)]
    else:
        # There are multiple crossing points
        isl_start = np.concatenate(
            ([cp_candidates[0]], cp_candidates[threshold_indices + 1])
        )
        isl_stop = np.concatenate(
            (cp_candidates[threshold_indices] + 1, [cp_candidates[-1] + 1])
        )
        rtn_extras["islands"] = [
            slice(start, stop) for start, stop in zip(isl_start, isl_stop)
        ]

        # do islands fall within the min_points window?
        isl_window = (
            np.abs(
                np.r_[rtn_extras["islands"][-1]][-1]
                - np.r_[rtn_extras["islands"][0]][0]
            )
            + 1
        )
        if isl_window > min_points:
            warn(
                f"Unable to determine floating potential, Langmuir sweep has "
                f"{n_islands} crossing-islands.  Try adjusting keyword 'threshold' "
                f"and/or smooth the current.",
                PlasmaPyWarning,
            )

            return np.nan, VFExtras(**rtn_extras)

    # Construct crossing-island (pad if needed)
    if np.isinf(min_points):
        # us all points
        istart = 0
        istop = voltage.size - 1
    else:
        istart = cp_candidates[0]
        istop = cp_candidates[-1]
        iadd = (istop - istart + 1) - min_points
        if iadd < 0:
            # pad front
            ipad_2_start = ipad_2_stop = int(np.ceil(-iadd / 2.0))
            if istart - ipad_2_start < 0:
                ipad_2_stop += ipad_2_start - istart
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
            warn(
                f"The number of elements in the current array ({istop - istart + 1}) "
                f"is less than 'min_points' ({min_points}).",
                PlasmaPyWarning,
            )

    # Perform Linear Regression Fit
    volt_sub = voltage[istart : istop + 1]
    curr_sub = current[istart : istop + 1]
    fit_func.curve_fit(volt_sub, curr_sub)

    vf, rtn_extras["vf_err"] = fit_func.root_solve()
    rtn_extras.update({"rsq": fit_func.rsq, "fitted_indices": slice(istart, istop + 1)})

    return vf, VFExtras(**rtn_extras)


find_vf_ = find_floating_potential
"""
Alias to
:func:`~plasmapy.analysis.swept_langmuir.floating_potential.find_floating_potential`.
"""


def plot_floating_potential(
    voltage, current, *, vf, vf_extras: VFExtras, ax: "Axes" = None
):
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gcf()
    elif not isinstance(ax, plt.Axes):
        raise ValueError

    if not isinstance(vf_extras, VFExtras):
        raise ValueError

    # calc island points
    isl_pts = np.array([], dtype=np.int64)
    for isl in vf_extras.islands:
        isl_pts = np.concatenate((isl_pts, np.r_[isl]))

    # calc xrange for plot
    xlim = [
        voltage[vf_extras.fitted_indices].min(),
        voltage[vf_extras.fitted_indices].max(),
    ]
    xlim_pad = 0.25 * (xlim[1] - xlim[0])
    xlim = [xlim[0] - xlim_pad, xlim[1] + xlim_pad]

    # calc data points for fit curve
    mask1 = np.where(voltage >= xlim[0], True, False)
    mask2 = np.where(voltage <= xlim[1], True, False)
    mask = np.logical_and(mask1, mask2)
    vfit = np.linspace(xlim[0], xlim[1], 201, endpoint=True)
    ifit, ifit_err = vf_extras.fitted_func(vfit, reterr=True)

    ax.set_xlabel("Bias Voltage (V)", fontsize=12)
    ax.set_ylabel("Current (A)", fontsize=12)
    ax.set_xlim(*xlim)

    # plot original data
    ax.plot(
        voltage[mask],
        current[mask],
        marker="o",
        zorder=10,
        label="Sweep Data",
    )

    # highlight points used for fit
    ax.scatter(
        voltage[vf_extras.fitted_indices],
        current[vf_extras.fitted_indices],
        linewidth=2,
        s=6 ** 2,
        facecolors="deepskyblue",
        edgecolors="deepskyblue",
        zorder=11,
        label="Points for Fit",
    )

    # highlight crossing islands
    ax.scatter(
        voltage[isl_pts],
        current[isl_pts],
        linewidth=2,
        s=8 ** 2,
        facecolors="deepskyblue",
        edgecolors="black",
        zorder=12,
        label="Island Points",
    )

    # plot fitted curve
    if isinstance(vf_extras.fitted_func, ffuncs.Linear):
        label = "Linear Fit"
    elif isinstance(vf_extras.fitted_func, ffuncs.ExponentialPlusOffset):
        label = "Exponential Fit"
    else:
        label = "Fit"

    ax.autoscale(False)
    ax.plot(vfit, ifit, color="orange", zorder=13, label=label)
    ax.fill_between(
        vfit,
        ifit + ifit_err,
        ifit - ifit_err,
        color="orange",
        alpha=0.12,
        zorder=0,
        label="Fit Error",
    )

    # plot fiducia
    ax.axhline(0.0, color="r", linestyle="--")
    ax.fill_between(
        [vf - vf_extras.vf_err, vf + vf_extras.vf_err],
        ax.get_ylim()[0],
        ax.get_ylim()[1],
        color="grey",
        alpha=0.1,
    )
    ax.axvline(vf, color="grey")

    ax.legend(fontsize=12)

    # add text
    rsq = vf_extras.rsq
    txt = f"$V_f = {vf:.2f} \\pm {vf_extras.vf_err:.2f}$ V\n"
    txt += f"$r^2 = {rsq:.3f}$"
    txt_loc = [vf, ax.get_ylim()[1]]
    txt_loc = ax.transData.transform(txt_loc)
    txt_loc = ax.transAxes.inverted().transform(txt_loc)
    txt_loc[0] -= 0.02
    txt_loc[1] -= 0.26
    ax.text(
        txt_loc[0],
        txt_loc[1],
        txt,
        fontsize="large",
        transform=ax.transAxes,
        ha="right",
    )

    return ax
