"""
Module containing functionality focused on the plasma dispersion function
:math:`Z(ζ)`.
"""
__all__ = ["plasma_dispersion_func", "plasma_dispersion_func_deriv"]


import astropy.units as u
import numpy as np

from numbers import Complex
from scipy.special import wofz as faddeeva_function
from typing import Union


def plasma_dispersion_func(
    zeta: Union[Complex, np.ndarray, u.Quantity[u.dimensionless_unscaled]]
) -> Union[Complex, np.ndarray, u.Quantity[u.dimensionless_unscaled]]:
    r"""
    Calculate the plasma dispersion function.

    The plasma dispersion function is defined as:

    .. math::
        Z(ζ) = π^{-0.5} \int_{-∞}^{+∞}
        \frac{e^{-x^2}}{x-ζ} dx

    where the argument is a complex number :cite:p:`fried:1961`.

    Parameters
    ----------
    zeta : |array_like| or |Quantity|
        The real or complex value to be provided as an argument to the
        plasma dispersion function.

    Returns
    -------
    |array_like| or |Quantity|
        The real or complex value of plasma dispersion function
        evaluated at ``zeta``.

    Raises
    ------
    ~astropy.units.UnitsError
        If ``zeta`` is a |Quantity| but is not dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_function.plasma_dispersion_func_deriv`

    Notes
    -----
    In plasma wave theory, the plasma dispersion function appears
    frequently when the background medium has a Maxwellian
    distribution function.  The argument of this function then refers
    to the ratio of a wave's phase velocity to a thermal velocity.

    Examples
    --------
    >>> from plasmapy.dispersion import plasma_dispersion_func
    >>> plasma_dispersion_func(0)
    1.7724538509055159j
    >>> plasma_dispersion_func(1 + 1j)
    (-0.369...+0.540...j)
    >>> plasma_dispersion_func([0.3, 0.7 + 2.3j])
    array([-0.56526333+1.61990085j, -0.09995023+0.37685142j])
    """
    try:
        return 1j * np.sqrt(np.pi) * faddeeva_function(zeta)
    except u.UnitTypeError as wrong_units:
        raise u.UnitsError(
            "The argument to plasma_dispersion_func "
            "must be dimensionless if it is a Quantity."
        ) from wrong_units
    except TypeError as wrong_type:
        raise TypeError(
            "The argument to plasma_dispersion_func should be a real or "
            "complex number or array, or a dimensionless Quantity."
        ) from wrong_type


def plasma_dispersion_func_deriv(
    zeta: Union[Complex, np.ndarray, u.Quantity[u.dimensionless_unscaled]]
) -> Union[Complex, np.ndarray, u.Quantity[u.dimensionless_unscaled]]:
    r"""
    Calculate the derivative of the plasma dispersion function.

    The derivative of the plasma dispersion function is:

    .. math::
        Z'(ζ) = π^{-1/2} \int_{-∞}^{+∞} \frac{e^{-x^2}}{(x-ζ)^2} dx

    where the argument :math:`ζ` is a complex number
    :cite:p:`fried:1961`.

    Parameters
    ----------
    zeta : |array_like| or |Quantity|
        Argument of plasma dispersion function.

    Returns
    -------
    complex, `~numpy.ndarray`, or |Quantity|
        First derivative of plasma dispersion function.

    Raises
    ------
    ~astropy.units.UnitsError
        If the argument is a `~astropy.units.Quantity` but is not
        dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_func`

    Examples
    --------
    >>> plasma_dispersion_func_deriv(0)
    (-2+0j)
    >>> plasma_dispersion_func_deriv(1j)
    (-0.484255687717376...+0j)
    >>> plasma_dispersion_func_deriv(-1.52+0.47j)
    (0.165871331498228...+0.445879788059350...j)
    """
    try:
        return -2 * (1 + zeta * plasma_dispersion_func(zeta))
    except u.UnitTypeError as wrong_units:
        raise u.UnitsError(
            "The argument to plasma_dispersion_func_deriv "
            "must be dimensionless if it is a Quantity."
        ) from wrong_units
    except TypeError as wrong_type:
        raise TypeError(
            "The argument to plasma_dispersion_function_deriv "
            "must be one of the following types: complex, float, "
            "int, ndarray, or a dimensionless Quantity."
        ) from wrong_type
