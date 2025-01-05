"""Finding the ion saturation current of a Langmuir sweep."""

__all__ = ["find_ion_saturation_current", "ISatExtras"]
__aliases__ = ["find_isat_"]

import numbers
from typing import Any, NamedTuple

import numpy as np

from plasmapy.analysis import fit_functions as ffuncs
from plasmapy.analysis.swept_langmuir.helpers import check_sweep

__all__ += __aliases__


class ISatExtras(NamedTuple):
    """
    Create a `tuple` containing the extra parameters calculated by
    `~plasmapy.analysis.swept_langmuir.ion_saturation_current.find_ion_saturation_current`.
    """

    rsq: float | None
    """
    Alias for field number 0, the r-squared value of the ion-saturation
    curve fit.
    """

    fitted_func: ffuncs.AbstractFitFunction | None
    """
    Alias for field number 1, the :term:`fit-function` fitted during
    the ion-saturation curve fit.
    """

    fitted_indices: slice | None
    """
    Alias for field number 2, the indices used in the ion-saturation
    curve fit.
    """


def find_ion_saturation_current(
    voltage: np.ndarray,
    current: np.ndarray,
    *,
    fit_type: str = "exp_plus_linear",
    current_bound: float | None = None,
    voltage_bound: float | None = None,
) -> tuple[ffuncs.Linear, ISatExtras]:
    """
    Determines the ion-saturation current (:math:`I_{sat}`) for a given
    current-voltage (IV) curve obtained from a swept Langmuir probe.
    The current collected by a Langmuir probe reaches ion-saturation
    when the probe is sufficiently biased so the influx of electrons is
    completely repelled, which leads to only the collection of ions.
    (For additional details see the **Notes** section below.).

    **Aliases:** :func:`~plasmapy.analysis.swept_langmuir.ion_saturation_current.find_isat_`

    Parameters
    ----------
    voltage: `numpy.ndarray`
        1-D numpy array of monotonically increasing probe biases
        (should be in volts).

    current: `numpy.ndarray`
        1-D numpy array of probe current (should be in amperes)
        corresponding to the ``voltage`` array.

    fit_type: `str`
        The type of curve (:term:`fit-function`) to be fitted to the
        Langmuir trace, valid options are listed below.
        (DEFAULT ``"exp_plus_linear"``)

        +-----------------------+----------------------------------------------------------+
        | ``"linear"``          | `~plasmapy.analysis.fit_functions.Linear`                |
        +-----------------------+----------------------------------------------------------+
        | ``"exp_plus_offset"`` | `~plasmapy.analysis.fit_functions.ExponentialPlusOffset` |
        +-----------------------+----------------------------------------------------------+
        | ``"exp_plus_linear"`` | `~plasmapy.analysis.fit_functions.ExponentialPlusLinear` |
        +-----------------------+----------------------------------------------------------+

    current_bound: `float`
        A fraction representing a percentile window around the minimum
        current.  The points to be fitted are defined to be within this
        window.  For example, a value of ``0.1`` indicates to use all
        points within 10% of the minimum current.  (DEFAULT ``None``)

        |

        If neither ``current_bound`` or ``voltage_bound`` are specified,
        then the routine will collect indices based on an internal
        ``current_bound`` setting for the specified ``fit_type``.

        +-----------------------+--------------------------------------+
        | ``"linear"``          | 0.4                                  |
        +-----------------------+--------------------------------------+
        | ``"exponential"``     | 1.0                                  |
        +-----------------------+--------------------------------------+
        | ``"exp_plus_linear"`` | 1.0                                  |
        +-----------------------+--------------------------------------+

        |

        Cannot be used with keyword ``voltage_bound``.

    voltage_bound: `float`
        A bias voltage (in volts) that specifies an upper bound used to
        collect the points for the curve fit.  That is, points that
        satisfy ``voltage <= voltage_bound`` are used in the fit.
        (DEFAULT ``None``)

        |

        Cannot be used with keyword ``current_bound``.

    Returns
    -------
    isat: `~plasmapy.analysis.fit_functions.Linear`
        A :term:`fit-function` representing the linear portion of the
        fitted curve.  **Note:** All ``isat`` parameters will be in the
        same units as those of ``voltage`` and ``current``.  For
        example, if the ``voltage`` array is in milli-volts and the
        ``current`` array is in milli-amperes, then all parameters and
        computed values of ``isat`` will have the same units.

    extras: `ISatExtras`
        Additional information from the curve fit:

        ``extras.fitted_func`` (:term:`fit-functions`)
            The computed :term:`fit-function` specified by
            ``fit_type``.

        ``extras.rsq`` (`float`)
            The coefficient of determination (r-squared) value of the
            fit, that is of ``extras.fitted_func``.

        ``extras.fitted_indices`` (`slice`)
            A `slice` object representing the indices of the ``voltage``
            and ``current`` arrays used for the fit.

    Notes
    -----
    This routine works by:

    1. Selecting the points to be used in the fit as determined by
       ``voltage_bound`` or ``current_bound``.
    2. Fitting the selected points with the :term:`fit-function`
       specified by ``fit_type``.
    3. Extracting the linear component of the fit and returning that as
       the ion-saturation current.

    This routine opts to return a function representing a linear
    ion-saturation current, since, while ideal planar Langmuir probes
    reach a steady-state ion-saturation current, real world Langmuir
    probes "suffer" from expanding sheaths as the bias voltage
    becomes increasingly negative.  This sheath expansion results in the
    ion-saturation current also increasing.
    """
    rtn_extras = ISatExtras(rsq=None, fitted_func=None, fitted_indices=None)._asdict()

    _settings: dict[str, dict[str, Any]] = {
        "linear": {
            "func": ffuncs.Linear,
            "current_bound": 0.4,
        },
        "exp_plus_linear": {
            "func": ffuncs.ExponentialPlusLinear,
            "current_bound": 1.0,
        },
        "exp_plus_offset": {
            "func": ffuncs.ExponentialPlusOffset,
            "current_bound": 1.0,
        },
    }
    try:
        default_current_bound = _settings[fit_type]["current_bound"]
        fit_func = _settings[fit_type]["func"]()
        rtn_extras["fitted_func"] = fit_func
    except KeyError as ex:
        raise ValueError(
            f"Requested fit '{fit_type}' is not a valid option.  Valid options "
            f"are {list(_settings.keys())}."
        ) from ex

    # check voltage and current arrays
    voltage, current = check_sweep(voltage, current, strip_units=True)

    # condition kwargs voltage_bound and current_bound
    if voltage_bound is current_bound is None:
        current_bound = default_current_bound
    elif voltage_bound is not None and current_bound is not None:
        raise ValueError(
            "Both keywords 'current_bound' and 'voltage_bound' are specified, "
            "use only one."
        )

    if current_bound is not None:
        if not isinstance(current_bound, numbers.Real):
            raise TypeError(
                f"Keyword 'current_bound' is of type {type(current_bound)}, "
                f"expected an int or float."
            )

        current_min = current.min()
        current_bound = (1.0 - current_bound) * current_min
        mask = np.where(current <= current_bound)[0]
    elif isinstance(voltage_bound, numbers.Real):
        mask = np.where(voltage <= voltage_bound)[0]

    else:
        raise TypeError(
            f"Keyword 'voltage_bound' is of type {type(voltage_bound)}, "
            f"expected an int or float."
        )

    if mask.size == 0:
        raise ValueError(
            f"The specified bounding keywords, 'voltage_bound' "
            f"({voltage_bound}) and 'current_bound' ({current_bound}), "
            f"resulted in a fit window containing no points."
        )

    mask = slice(0, mask[-1] + 1)
    rtn_extras["fitted_indices"] = mask

    volt_sub = voltage[mask]
    curr_sub = current[mask]
    fit_func.curve_fit(volt_sub, curr_sub)

    rtn_extras["rsq"] = fit_func.rsq

    m = getattr(fit_func.params, "m", 0.0)
    b = getattr(fit_func.params, "b", 0.0)
    m_err = getattr(fit_func.param_errors, "m", 0.0)
    b_err = getattr(fit_func.param_errors, "b", 0.0)
    isat = ffuncs.Linear(params=(m, b), param_errors=(m_err, b_err))

    return isat, ISatExtras(**rtn_extras)


find_isat_ = find_ion_saturation_current
"""
Alias to
:func:`~plasmapy.analysis.swept_langmuir.ion_saturation_current.find_ion_saturation_current`.
"""
