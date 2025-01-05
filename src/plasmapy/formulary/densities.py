"""Formulas for plasma densities."""

__all__ = [
    "critical_density",
    "mass_density",
]
__aliases__ = ["rho_"]

import astropy.units as u
import numpy as np
from astropy.constants.si import e, eps0, m_e

from plasmapy.particles.particle_class import Particle, ParticleLike
from plasmapy.utils.decorators import validate_quantities

__all__ += __aliases__


@validate_quantities(
    omega={"can_be_negative": False},
    validations_on_return={
        "units": [u.m**-3],
    },
)
def critical_density(omega: u.Quantity[u.rad / u.s]) -> u.Quantity[u.m**-3]:
    r"""
    Calculate the plasma critical density for a radiation of a given frequency.

    Parameters
    ----------
    omega: `~astropy.units.Quantity`
        The radiation frequency in units of angular frequency.

    Returns
    -------
    n_c : `~astropy.units.Quantity`
        The plasma critical density.

    Notes
    -----
    The critical density for a given frequency of radiation is
    defined as the value at which the electron plasma frequency equals
    the frequency of the radiation.

    The critical density is given by the formula

    .. math::
        n_c=\frac{m_e ε_0 θ^{2}}{e^2}

    where :math:`m_{e}` is the mass of an electron,
    :math:`ε_0` is the permittivity of free space, :math:`θ`
    is the radiation frequency, and :math:`e` is the elementary charge.

    Examples
    --------
    >>> import astropy.units as u
    >>> critical_density(5e15 * u.rad / u.s)
    <Quantity 7.85519457e+27 1 / m3>

    """

    n_c = m_e * eps0 * omega**2 / (e**2)

    return n_c.to(u.m**-3, equivalencies=u.dimensionless_angles())


@validate_quantities(
    density={"can_be_negative": False}, validations_on_return={"can_be_negative": False}
)
def mass_density(
    density: (u.m**-3, u.kg / (u.m**3)),
    particle: ParticleLike,
    z_ratio: float | None = 1,
) -> u.Quantity[u.kg / u.m**3]:
    r"""
    Calculate the mass density from a number density.

    .. math::

        ρ = \left| \frac{Z_s}{Z_{particle}} \right| n_s m_{particle}
              = | Z_{ratio} | n_s m_{particle}

    where :math:`m_{particle}` is the particle mass, :math:`n_s` is a
    number density for plasma species :math:`s`, :math:`Z_s` is the
    charge number of species :math:`s`, and :math:`Z_{particle}` is the
    charge number of ``particle``. For example, if the electron density
    is given for :math:`n_s` and ``particle`` is a doubly ionized atom,
    then :math:`Z_{ratio} = -1 / 2`\ .

    Parameters
    ----------
    density : `~astropy.units.Quantity`
        Either a particle number density (in units of m\ :sup:`-3` or
        equivalent) or a mass density (in units of kg / m\ :sup:`3` or
        equivalent).  If ``density`` is a mass density, then it will be passed
        through and returned without modification.

    particle : `~plasmapy.particles.particle_class.Particle`
        The particle for which the mass density is being calculated.  Must
        be a `~plasmapy.particles.particle_class.Particle` or a value convertible to
        a `~plasmapy.particles.particle_class.Particle` (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly ionized helium-4).

    z_ratio : `int` | `float`, default: ``1``
        The ratio of the charge numbers corresponding to the plasma species
        represented by ``density`` and the ``particle``.  For example, if the
        given ``density`` is an electron density and ``particle`` is doubly
        ionized ``He``, then ``z_ratio = -0.5``.

    Raises
    ------
    `~astropy.units.UnitTypeError`
        If the ``density`` does not have units equivalent to a number density
        or mass density.

    `TypeError`
        If ``density`` is not of type `~astropy.units.Quantity`, or convertible.

    `TypeError`
        If ``particle`` is not of type or convertible to
        `~plasmapy.particles.particle_class.Particle`.

    `TypeError`
        If ``z_ratio`` is not of type `int` or `float`.

    `ValueError`
        If ``density`` is negative.

    Returns
    -------
    `~astropy.units.Quantity`
        The mass density for the plasma species represented by ``particle``.

    Examples
    --------
    >>> import astropy.units as u
    >>> mass_density(1 * u.m**-3, "p+")
    <Quantity 1.67262...e-27 kg / m3>
    >>> mass_density(4 * u.m**-3, "D+")
    <Quantity 1.33743...e-26 kg / m3>
    >>> mass_density(2.0e12 * u.cm**-3, "He")
    <Quantity 1.32929...e-08 kg / m3>
    >>> mass_density(2.0e12 * u.cm**-3, "He", z_ratio=0.5)
    <Quantity 6.64647...e-09 kg / m3>
    >>> mass_density(1.0 * u.g * u.m**-3, "")
    <Quantity 0.001 kg / m3>
    """
    if density.unit.is_equivalent(u.kg / u.m**3):
        return density

    if not isinstance(particle, Particle):
        try:
            particle = Particle(particle)
        except TypeError as e:
            raise TypeError(
                f"If passing a number density, you must pass a plasmapy Particle "
                f"(not type {type(particle)}) to calculate the mass density!"
            ) from e

    if not isinstance(z_ratio, float | np.floating | int | np.integer):
        raise TypeError(
            f"Expected type int or float for keyword z_ratio, got type {type(z_ratio)}."
        )

    return abs(z_ratio) * density * particle.mass


rho_ = mass_density
"""Alias to `~plasmapy.formulary.densities.mass_density`."""
