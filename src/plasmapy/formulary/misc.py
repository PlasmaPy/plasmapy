"""Miscellaneous plasma parameters."""

__all__ = [
    "Bohm_diffusion",
    "magnetic_energy_density",
    "magnetic_pressure",
    "thermal_pressure",
]
__aliases__ = ["DB_", "pmag_", "pth_", "ub_"]

import astropy.units as u
from astropy.constants.si import e, k_B, mu0

from plasmapy import particles
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.utils.decorators import validate_quantities

__all__ += __aliases__


def _grab_charge(ion: ParticleLike, z_mean=None):
    """
    Merge two possible inputs for particle charge.

    Parameters
    ----------
    ion : `~plasmapy.particles.particle_class.Particle`
        a string representing a charged particle, or a Particle object.

    z_mean : `float`
        An optional float describing the average ionization of a particle
        species.

    Returns
    -------
    float
        if ``z_mean`` was passed, ``z_mean``, otherwise, the charge number
        of ``ion``.
    """
    return particles.charge_number(ion) if z_mean is None else z_mean


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    B={"can_be_negative": False},
)
def Bohm_diffusion(
    T_e: u.Quantity[u.K], B: u.Quantity[u.T]
) -> u.Quantity[u.m**2 / u.s]:
    r"""
    Return the Bohm diffusion coefficient.

    The Bohm diffusion coefficient was conjectured to follow Bohm model
    of the diffusion of plasma across a magnetic field and describe the
    diffusion of early fusion energy machines :cite:p:`bohm:1949`. The
    rate predicted by Bohm diffusion is much higher than classical
    diffusion, and if there were no exceptions, magnetically confined
    fusion would be impractical.

    .. math::

        D_B = \frac{1}{16} \frac{k_B T}{e B}

    where :math:`k_B` is the Boltzmann constant
    and :math:`e` is the fundamental charge.

    **Aliases:** `DB_`

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        The electron temperature.

    B : `~astropy.units.Quantity`
        The magnitude of the magnetic field in the plasma.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Raises
    ------
    `TypeError`
        ``T_e`` is not a `~astropy.units.Quantity` and cannot be
        converted into one.

    `~astropy.units.UnitConversionError`
        If ``T_e`` is not in appropriate units.

    Examples
    --------
    >>> import astropy.units as u
    >>> T_e = 5000 * u.K
    >>> B = 10 * u.T
    >>> Bohm_diffusion(T_e, B)
    <Quantity 0.00269292 m2 / s>
    >>> T_e = 50 * u.eV
    >>> B = 10 * u.T
    >>> Bohm_diffusion(T_e, B)
    <Quantity 0.3125 m2 / s>

    Returns
    -------
    D_B : `~astropy.units.Quantity`
        The Bohm diffusion coefficient in meters squared per second.

    """
    return k_B * T_e / (16 * e * B)


DB_ = Bohm_diffusion
"""Alias to `~plasmapy.formulary.misc.Bohm_diffusion`."""


@validate_quantities
def magnetic_energy_density(B: u.Quantity[u.T]) -> u.Quantity[u.J / u.m**3]:
    r"""
    Calculate the magnetic energy density.

    **Aliases:** `ub_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field in units convertible to tesla.

    Returns
    -------
    E_B : `~astropy.units.Quantity`
        The magnetic energy density in units of joules per cubic meter.

    Raises
    ------
    `TypeError`
        If the input is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the input is not in units convertible to tesla.

    `ValueError`
        If the magnetic field strength does not have an appropriate
        value.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed

    See Also
    --------
    magnetic_pressure : Returns an equivalent `~astropy.units.Quantity`,
        except in units of pascals.

    Notes
    -----
    The magnetic energy density is given by:

    .. math::
        E_B = \frac{B^2}{2 μ_0}

    The motivation behind having two separate functions for magnetic
    pressure and magnetic energy density is that it allows greater
    insight into the physics that are being considered by the user and
    thus more readable code.

    Examples
    --------
    >>> import astropy.units as u
    >>> magnetic_energy_density(0.1 * u.T)
    <Quantity 3978.87... J / m3>
    """
    return magnetic_pressure(B)


ub_ = magnetic_energy_density
"""Alias to `~plasmapy.formulary.misc.magnetic_energy_density`."""


@validate_quantities
def magnetic_pressure(B: u.Quantity[u.T]) -> u.Quantity[u.Pa]:
    r"""
    Calculate the magnetic pressure.

    **Aliases:** `pmag_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field in units convertible to tesla.

    Returns
    -------
    p_B : `~astropy.units.Quantity`
        The magnetic pressure in units in pascals (newtons per square meter).

    Raises
    ------
    `TypeError`
        If the input is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the input is not in units convertible to tesla.

    `ValueError`
        If the magnetic field strength is not a real number between
        :math:`±∞`\ .

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    See Also
    --------
    magnetic_energy_density : returns an equivalent `~astropy.units.Quantity`,
        except in units of joules per cubic meter.

    Notes
    -----
    The magnetic pressure is given by:

    .. math::
        p_B = \frac{B^2}{2 μ_0}

    The motivation behind having two separate functions for magnetic
    pressure and magnetic energy density is that it allows greater
    insight into the physics that are being considered by the user and
    thus more readable code.

    Examples
    --------
    >>> import astropy.units as u
    >>> magnetic_pressure(0.1 * u.T).to(u.Pa)
    <Quantity 3978.87... Pa>
    """
    return (B**2) / (2 * mu0)


pmag_ = magnetic_pressure
"""Alias to `~plasmapy.formulary.misc.magnetic_pressure`."""


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n={"can_be_negative": False},
)
def thermal_pressure(T: u.Quantity[u.K], n: u.Quantity[u.m**-3]) -> u.Quantity[u.Pa]:
    r"""
    Return the thermal pressure for a Maxwellian distribution.

    **Aliases:** `pth_`

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The particle temperature in either kelvin or energy per particle.

    n : `~astropy.units.Quantity`
        The particle number density in units convertible to m\ :sup:`-3`\ .

    Returns
    -------
    p_th : `~astropy.units.Quantity`
        Thermal pressure.

    Raises
    ------
    `TypeError`
        The temperature or number density is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the particle temperature is not in units of temperature or
        energy per particle.

    Notes
    -----
    The thermal pressure is given by:

    .. math::
        T_{th} = n k_B T

    Examples
    --------
    >>> import astropy.units as u
    >>> thermal_pressure(1 * u.eV, 1e20 / u.m**3)
    <Quantity 16.021... Pa>
    >>> thermal_pressure(10 * u.eV, 1e20 / u.m**3)
    <Quantity 160.21... Pa>
    """
    return n * k_B * T


pth_ = thermal_pressure
"""Alias to `~plasmapy.formulary.misc.thermal_pressure`."""
