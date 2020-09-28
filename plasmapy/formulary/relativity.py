r"""Functionality for calculating relativistic quantities (:math:`v \to c`)."""
__all__ = ["Lorentz_factor", "relativistic_energy"]

import astropy.units as u
import numpy as np

from astropy.constants import c

from plasmapy import utils
from plasmapy.utils.decorators import validate_quantities


@validate_quantities(V={"can_be_negative": True})
def Lorentz_factor(V: u.m / u.s):
    r"""
    Return the Lorentz factor.

    Parameters
    ----------

    V : ~astropy.units.Quantity
        The velocity in units convertible to meters per second.

    Returns
    -------
    gamma : float or ~numpy.ndarray
        The Lorentz factor associated with the inputted velocities.

    Raises
    ------
    TypeError
        The `V` is not a `~astropy.units.Quantity` and cannot be
        converted into a ~astropy.units.Quantity.

    ~astropy.units.UnitConversionError
        If the `V` is not in appropriate units.

    ValueError
        If the magnitude of `V` is faster than the speed of light.

    Warns
    -----
    ~astropy.units.UnitsWarning
        If units are not provided, SI units are assumed.

    Notes
    -----
    The Lorentz factor is a dimensionless number given by

    .. math::
        \gamma = \frac{1}{\sqrt{1-\frac{V^2}{c^2}}}

    The Lorentz factor is approximately one for sub-relativistic
    velocities, and goes to infinity as the velocity approaches the
    speed of light.

    Examples
    --------
    >>> from astropy import units as u
    >>> velocity = 1.4e8 * u.m / u.s
    >>> Lorentz_factor(velocity)
    1.130885603948959
    >>> Lorentz_factor(299792458*u.m/u.s)
    inf
    """

    if not np.all(np.abs(V) <= c):
        raise utils.RelativityError(
            "The Lorentz factor cannot be calculated for "
            "speeds faster than the speed of light. "
        )

    if V.size > 1:

        gamma = np.zeros_like(V.value)

        equals_c = np.abs(V) == c
        is_slow = ~equals_c

        gamma[is_slow] = ((1 - (V[is_slow] / c) ** 2) ** -0.5).value
        gamma[equals_c] = np.inf

    else:
        if np.abs(V) == c:
            gamma = np.inf
        else:
            gamma = ((1 - (V / c) ** 2) ** -0.5).value

    return gamma


@validate_quantities(
    m={"can_be_negative": False}, validations_on_return={"can_be_negative": False}
)
def relativistic_energy(m: u.kg, v: u.m / u.s) -> u.Joule:
    """
    Calculate the relativistic energy (in Joules) of an object of mass
    `m` and velocity `v`.

    .. math::

        E = \\gamma m c^{2}

    where :math:`\\gamma` is the `Lorentz_factor`.

    Parameters
    ----------
    m : `~astropy.units.Quantity`
        The mass in units convertible to kilograms.

    v : `~astropy.units.Quantity`
        The velocity in units convertible to meters per second.

    Returns
    -------
    `~astropy.Quantity`
        The relativistic energy (in Joules) of an object of mass `m` moving at velocity `v`.

    Raises
    ------
    TypeError
        If input arguments are not instances `~astropy.units.Quantity` or
        convertible to a `~astropy.units.Quantity`.

    ~astropy.units.UnitConversionError
        If the `v` is not in appropriate units.

    ValueError
        If the magnitude of `m` is negative or arguments are complex.

    :exc:`~plasmapy.utils.exceptions.RelativityError`
        If the velocity `v` is greater than the speed of light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Examples
    --------
    >>> from astropy import units as u
    >>> velocity = 1.4e8 * u.m / u.s
    >>> mass = 1 * u.kg
    >>> relativistic_energy(mass, velocity)
    <Quantity 1.01638929e+17 J>
    >>> relativistic_energy(mass, 299792458*u.m / u.s)
    <Quantity inf J>
    >>> relativistic_energy(1 * u.mg, 1.4e8 * u.m / u.s)
    <Quantity 1.01638929e+11 J>
    >>> relativistic_energy(-mass, velocity)
    Traceback (most recent call last):
        ...
    ValueError: The argument 'm' to function relativistic_energy() can not contain negative numbers.
    """

    gamma = Lorentz_factor(v)
    E = gamma * m * c ** 2
    return E


def quiver_velocity(E: u.V / u.m, w: 1 / u.s, q: u.C, m: u.kg):
    """
    Quiver velocity or normalized momentum is the term given to the amplitude of the oscillation of
    a charged particle due to electromagnetic radiation. It is usually expressed as a dimensionless
    quantity by dividing the amplitude by c.

    .. math::

        a = /frac{q E_0}{m \\omega c}}

    :math: `E_0` is Electric field intensity
    :math: `m` is mass of the particle
    :math: `\\omega` is the angular frequency of the radiation
    :math: `q` is the charge of the particle
    :math: `c` is the speed of light
    This is true when the electric field :math: `E` is modeled as :math: `E = E_0 sin(\\omega t)`

    Parameters
    ----------
    E : `~astropy.units.Quantity`
        Electric field intensity.

    w : `~astropy.units.Quantity`
        The angular frequency of the radiation.

    q : `~astropy.units.Quantity`
        The charge of the particle

    m : `~astropy.units.Quantity`
        The mass of the particle

    Returns
    -------
    a : `~astropy.Quantity`
        The dimensionless electron quiver velocity.

    NOTE
    ----
    This quantity can be greater than one, in which case it is more appropriate to refer to this quantity
    as "normalized momentum" since the real velocity of the particle can not be greater than the speed
    of light.

    """

    a = E * q / (m * w * c)

    return a
