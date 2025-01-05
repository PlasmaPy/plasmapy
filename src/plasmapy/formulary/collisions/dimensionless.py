"""Dimensionless numbers related to particle collisions."""

__all__ = [
    "coupling_parameter",
    "Knudsen_number",
]

from typing import Literal

import astropy.units as u
import numpy as np
from astropy.constants.si import e, eps0, k_B

from plasmapy import particles
from plasmapy.formulary.collisions import lengths, misc
from plasmapy.formulary.mathematics import Fermi_integral
from plasmapy.formulary.quantum import (
    Wigner_Seitz_radius,
    chemical_potential,
    thermal_deBroglie_wavelength,
)
from plasmapy.utils.decorators import validate_quantities


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def coupling_parameter(
    T: u.Quantity[u.K],
    n_e: u.Quantity[u.m**-3],
    species,
    z_mean: float = np.nan,
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    method: Literal["classical", "quantum"] = "classical",
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Ratio of the Coulomb energy to the kinetic (usually thermal) energy.

    Classical plasmas are weakly coupled (:math:`Γ ≪ 1`, where :math:`Γ`
    is the coupling parameter). Dense plasmas tend to have significant
    to strong coupling (:math:`Γ ≥ 1`\ ).  For more details, see the
    notes section below.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Temperature in units of temperature or energy per particle,
        which is assumed to be equal for both the test particle and the
        target particle.

    n_e : `~astropy.units.Quantity`
        The electron number density in units convertible to per cubic
        meter.

    species : `tuple`
        A tuple containing string representations of the test particle
        (listed first) and the target particle (listed second).

    z_mean : `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) of a plasma for which a
        macroscopic description is valid. This parameter is used to
        compute the average ion density (given the average ionization
        and electron density) for calculating the ion sphere radius for
        non-classical impact parameters. ``z_mean`` is a required
        parameter if ``method`` is ``"ls_full_interp"``,
        ``"hls_max_interp"``, or ``"hls_full_interp"``.

    V : `~astropy.units.Quantity`, optional
        The relative velocity between particles. If not provided,
        thermal velocity is assumed: :math:`μ V^2 \sim 2 k_B T` where
        :math:`μ` is the reduced mass.

    method : {``"classical"``, ``"quantum"``}, default: "classical"
        The method by which to compute the coupling parameter: either
        ``"classical"`` or ``"quantum"``. The Notes section of this
        docstring has more information about these two methods.

    Returns
    -------
    coupling : `float` or `~numpy.ndarray`
        The coupling parameter for a plasma.

    Raises
    ------
    `ValueError`
        If the mass or charge of either particle cannot be found, or any
        of the inputs contain incorrect values.

    `~astropy.units.UnitConversionError`
        If the units on any of the inputs are incorrect.

    `TypeError`
        If any of ``n_e``, ``T``, or ``V`` is not a
        `~astropy.units.Quantity`.

    `~plasmapy.utils.exceptions.RelativityError`
        If the input velocity is same or greater than the speed of
        light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the input velocity is greater than 5% of the speed of light.

    Notes
    -----
    The coupling parameter is given by

    .. math::

        Γ = \frac{E_{Coulomb}}{E_{Kinetic}}

    The Coulomb energy is given by

    .. math::

        E_{Coulomb} = \frac{Z_1 Z_2 q_e^2}{4 π ε_0 r}

    where :math:`r` is the Wigner-Seitz radius, and 1 and 2 refer to
    particle species 1 and 2 between which we want to determine the
    coupling.

    In the classical case the kinetic energy is the thermal energy:

    .. math::

        E_{kinetic} = k_B T_e

    The quantum case is more complex. The kinetic energy is dominated by
    the Fermi energy, modulated by a correction factor based on the
    ideal chemical potential. This is obtained more precisely by taking
    the thermal kinetic energy and dividing by the degeneracy
    parameter, modulated by the Fermi integral :cite:p:`gericke:2002`\ :

    .. math::

        E_{kinetic} = 2 k_B T_e / χ f_{3/2} (μ_{ideal} / k_B T_e)

    where :math:`χ` is the degeneracy parameter, :math:`f_{3/2}` is the
    Fermi integral, and :math:`μ_{ideal}` is the ideal chemical
    potential.

    The degeneracy parameter is given by

    .. math::

        χ = n_e Λ_{de Broglie} ^ 3

    where :math:`n_e` is the electron density and :math:`Λ_{de Broglie}`
    is the thermal de Broglie wavelength.

    See equations 1.2, 1.3 and footnote 5 in :cite:t:`bonitz:1998` for
    details on the ideal chemical potential.

    Examples
    --------
    >>> import astropy.units as u
    >>> n = 1e19 * u.m**-3
    >>> T = 1e6 * u.K
    >>> species = ("e", "p")
    >>> coupling_parameter(T, n, species)
    <Quantity 5.8033...e-05>
    >>> coupling_parameter(T, n, species, V=1e6 * u.m / u.s)
    <Quantity 5.8033...e-05>
    """
    T, masses, charges, reduced_mass, V = misc._process_inputs(  # noqa: SLF001
        T=T, species=species, V=V
    )

    if np.isnan(z_mean):
        # using mean charge to get average ion density.
        # If you are running this, you should strongly consider giving
        # a value of z_mean as an argument instead.
        Z1 = np.abs(particles.atomic.charge_number(species[0]))
        Z2 = np.abs(particles.atomic.charge_number(species[1]))
        Z = (Z1 + Z2) / 2
        # getting ion density from electron density
        n_i = n_e / Z
    else:
        # getting ion density from electron density
        n_i = n_e / z_mean
    # getting Wigner-Seitz radius based on ion density
    radius = Wigner_Seitz_radius(n_i)
    # Coulomb potential energy between particles
    if np.isnan(z_mean):
        coulomb_energy = charges[0] * charges[1] / (4 * np.pi * eps0 * radius)
    else:
        coulomb_energy = (z_mean * e) ** 2 / (4 * np.pi * eps0 * radius)

    if method == "classical":
        # classical thermal kinetic energy
        kinetic_energy = k_B * T
    elif method == "quantum":
        # quantum kinetic energy for dense plasmas
        lambda_deBroglie = thermal_deBroglie_wavelength(T)
        chem_potential = chemical_potential(n_e, T)
        fermi_integral = Fermi_integral(chem_potential.si.value, 1.5)
        denominator = (n_e * lambda_deBroglie**3) * fermi_integral
        kinetic_energy = 2 * k_B * T / denominator
        if np.all(np.imag(kinetic_energy) < 1e-15 * u.J):
            kinetic_energy = np.real(kinetic_energy)
        else:
            raise ValueError(
                "Kinetic energy should not be imaginary."
                "Something went horribly wrong."
            )
    else:
        raise ValueError(
            f"Keyword 'method' must be either 'classical' or "
            f"'quantum', instead of '{method}'."
        )

    return coulomb_energy / kinetic_energy


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def Knudsen_number(
    characteristic_length,
    T: u.Quantity[u.K],
    n_e: u.Quantity[u.m**-3],
    species,
    z_mean: float = np.nan,
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    method: str = "classical",
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Knudsen number (dimensionless).

    Parameters
    ----------
    characteristic_length : `~astropy.units.Quantity`
        Rough order-of-magnitude estimate of the relevant size of the
        system.

    T : `~astropy.units.Quantity`
        Temperature in units of temperature or energy per particle,
        which is assumed to be equal for both the test particle and the
        target particle.

    n_e : `~astropy.units.Quantity`
        The electron number density in units convertible to m\ :sup:`-3`.

    species : `tuple`
        A tuple containing string representations of the test particle
        (listed first) and the target particle (listed second).

    z_mean : `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) of a plasma for which a
        macroscopic description is valid. This parameter is used to
        compute the average ion density (given the average ionization
        and electron density) for calculating the ion sphere radius for
        non-classical impact parameters. ``z_mean`` is a required
        parameter if ``method`` is ``"ls_full_interp"``,
        ``"hls_max_interp"``, or ``"hls_full_interp"``.

    V : `~astropy.units.Quantity`, optional
        The relative velocity between particles. If not provided,
        thermal velocity is assumed: :math:`μ V^2 \sim 2 k_B T` where
        :math:`μ` is the reduced mass.

    method : `str`, optional
        The method by which to compute the Coulomb logarithm.  The
        default method is the classical straight-line Landau-Spitzer
        method (``"classical"`` or ``"ls"``). The other 6 supported
        methods are ``"ls_min_interp"``, ``"ls_full_interp"``,
        ``"ls_clamp_mininterp"``, ``"hls_min_interp"``,
        ``"hls_max_interp"``, and ``"hls_full_interp"``.  Please refer
        to the docstring of
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for more
        information about these methods.

    Returns
    -------
    knudsen_param : `float` or `numpy.ndarray`
        The dimensionless Knudsen number.

    Raises
    ------
    `ValueError`
        If the mass or charge of either particle cannot be found, or any
        of the inputs contain incorrect values.

    `~astropy.units.UnitConversionError`
        If the units on any of the inputs are incorrect.

    `TypeError`
        If any of ``n_e``, ``T``, or ``V`` is not a
        `~astropy.units.Quantity`.

    `~plasmapy.utils.exceptions.RelativityError`
        If the input velocity is same or greater than the speed of
        light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the input velocity is greater than 5% of the speed of light.

    Notes
    -----
    The `Knudsen number <https://en.wikipedia.org/wiki/Knudsen_number>`_
    is given by

    .. math::

        Kn = \frac{λ_{mfp}}{L}

    where :math:`λ_{mfp}` is the collisional mean free path for
    particles in a plasma and :math:`L` is the characteristic scale
    length of interest.

    The characteristic scale length is typically the plasma size or the
    size of a diagnostic (such as the length or radius of a Langmuir
    probe tip). The Knudsen number tells us whether collisional effects
    are important on this scale length.

    Examples
    --------
    >>> import astropy.units as u
    >>> L = 1e-3 * u.m
    >>> n = 1e19 * u.m**-3
    >>> T = 1e6 * u.K
    >>> species = ("e", "p")
    >>> Knudsen_number(L, T, n, species)
    <Quantity 7839.5...>
    >>> Knudsen_number(L, T, n, species, V=1e6 * u.m / u.s)
    <Quantity 10.91773...>
    """
    path_length = lengths.mean_free_path(
        T=T, n_e=n_e, species=species, z_mean=z_mean, V=V, method=method
    )
    return path_length / characteristic_length
