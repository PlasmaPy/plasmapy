"""Functions to calculate fundamental plasma parameters."""

__all__ = [
    "Alfven_speed",
    "Bohm_diffusion",
    "betaH_",
    "cs_",
    "cwp_",
    "DB_",
    "Debye_length",
    "Debye_number",
    "gyrofrequency",
    "gyroradius",
    "Hall_parameter",
    "inertial_length",
    "ion_sound_speed",
    "kappa_thermal_speed",
    "lambdaD_",
    "lower_hybrid_frequency",
    "magnetic_energy_density",
    "magnetic_pressure",
    "mass_density",
    "nD_",
    "oc_",
    "plasma_frequency",
    "pmag_",
    "pth_",
    "rc_",
    "rho_",
    "rhoc_",
    "thermal_pressure",
    "thermal_speed",
    "thermal_speed_coefficients",
    "thermal_speed_lite",
    "ub_",
    "upper_hybrid_frequency",
    "va_",
    "vth_",
    "vth_kappa_",
    "wc_",
    "wp_",
    "wlh_",
    "wuh_",
]

import astropy.units as u
import numbers
import numpy as np
import warnings

from astropy.constants.si import c, e, eps0, k_B, mu0
from numba import njit
from typing import Optional, Union

from plasmapy import particles
from plasmapy.particles import Particle
from plasmapy.particles.exceptions import ChargeError
from plasmapy.utils import PhysicsError
from plasmapy.utils.decorators import (
    add_lite,
    angular_freq_to_hz,
    check_relativistic,
    preserve_signature,
    validate_quantities,
)
from plasmapy.utils.exceptions import PhysicsWarning

k_B_si_unitless = k_B.value


def _grab_charge(ion: Particle, z_mean=None):
    """
    Merge two possible inputs for particle charge.

    Parameters
    ----------
    ion : `~plasmapy.particles.Particle`
        a string representing a charged particle, or a Particle object.

    z_mean : `float`
        An optional float describing the average ionization of a particle
        species.

    Returns
    -------
    float
        if ``z_mean`` was passed, ``z_mean``, otherwise, the integer charge
        of ``ion``.

    """
    if z_mean is None:
        # warnings.warn("No z_mean given, defaulting to atomic charge",
        #               PhysicsWarning)
        Z = particles.integer_charge(ion)
    else:
        # using average ionization provided by user
        Z = z_mean
    return Z


@validate_quantities(
    density={"can_be_negative": False}, validations_on_return={"can_be_negative": False}
)
def mass_density(
    density: [u.m ** -3, u.kg / (u.m ** 3)],
    particle: Union[Particle, str],
    z_ratio: Optional[numbers.Real] = 1,
) -> u.kg / u.m ** 3:
    r"""
    Calculate the mass density from a number density.

    .. math::

        \rho = \left| \frac{Z_{s}}{Z_{particle}} \right| n_{s} m_{particle}
              = | Z_{ratio} | n_{s} m_{particle}

    where :math:`m_{particle}` is the particle mass, :math:`n_{s}` is a number
    density for plasma species :math:`s`, :math:`Z_{s}` is the integer charge of
    species :math:`s`, and :math:`Z_{particle}` is the integer charge of
    ``particle``.  For example, if the electron density is given for :math:`n_s`
    and ``particle`` is a doubly ionized atom, then :math:`Z_{ratio} = -1 / 2`\ .

    **Aliases:** `rho_`

    Parameters
    ----------
    density : `~astropy.units.Quantity`
        Either a particle number density (in units of m\ :sup:`-3` or
        equivalent) or a mass density (in units of kg / m\ :sup:`3` or
        equivalent).  If ``density`` is a mass density, then it will be passed
        through and returned without modification.

    particle : `~plasmapy.particles.Particle`
        The particle for which the mass density is being calculated for.  Must
        be a `~plasmapy.particles.Particle` or a value convertible to
        a `~plasmapy.particles.Particle` (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly ionized helium-4).

    z_ratio : `int`, `float`, optional
        The ratio of the integer charges corresponding to the plasma species
        represented by ``density`` and the ``particle``.  For example, if the
        given ``density`` is and electron density and ``particle`` is doubly
        ionized ``He``, then ``z_ratio = -0.5``.  Default is ``1``.

    Raises
    ------
    `~astropy.units.UnitTypeError`
        If the ``density`` does not have units equivalent to a number density
        or mass density.

    `TypeError`
        If ``density`` is not of type `~astropy.units.Quantity`, or convertible.

    `TypeError`
        If ``particle`` is not of type or convertible to
        `~plasmapy.particles.Particle`.

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
    >>> mass_density(1 * u.m ** -3, 'p')
    <Quantity 1.67262...e-27 kg / m3>
    >>> mass_density(4 * u.m ** -3, 'D+')
    <Quantity 1.33743...e-26 kg / m3>
    >>> mass_density(2.e12 * u.cm ** -3, 'He')
    <Quantity 1.32929...e-08 kg / m3>
    >>> mass_density(2.e12 * u.cm ** -3, 'He', z_ratio=0.5)
    <Quantity 6.64647...e-09 kg / m3>
    >>> mass_density(1.0 * u.g * u.m ** -3, "")
    <Quantity 0.001 kg / m3>
    """
    if density.unit.is_equivalent(u.kg / u.m ** 3):
        return density

    if not isinstance(particle, Particle):
        try:
            particle = Particle(particle)
        except TypeError:
            raise TypeError(
                f"If passing a number density, you must pass a plasmapy Particle "
                f"(not type {type(particle)}) to calculate the mass density!"
            )

    if not isinstance(z_ratio, (float, np.floating, int, np.integer)):
        raise TypeError(
            f"Expected type int or float for keyword z_ratio, got type {type(z_ratio)}."
        )

    return abs(z_ratio) * density * particle.mass


rho_ = mass_density
""" Alias to :func:`mass_density`. """


@check_relativistic
@validate_quantities(density={"can_be_negative": False})
def Alfven_speed(
    B: u.T,
    density: [u.m ** -3, u.kg / u.m ** 3],
    ion: Optional[Particle] = None,
    z_mean: Optional[numbers.Real] = None,
) -> u.m / u.s:
    r"""
    Calculate the Alfvén speed.

    The Alfvén speed :math:`V_A` is the typical propagation speed of magnetic
    disturbances in a plasma, and is given by:

    .. math::

        V_A = \frac{B}{\sqrt{μ_0 ρ}}

    where :math:`B` is the magnetic field and :math:`ρ = n_i m_i + n_e m_e`
    is the total mass density (:math:`n_i` is the ion number density,
    :math:`n_e` is the electron number density, :math:`m_i` is the ion mass,
    and :math:`m_e` is the electron mass).

    **Aliases:** `va_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    density : `~astropy.units.Quantity`
        Either the ion number density :math:`n_i` in units convertible to
        m\ :sup:`-3` or the total mass density :math:`ρ` in units
        convertible to kg / m\ :sup:`-3`\ .

    ion : `~plasmapy.particles.Particle`, optional
        Representation of the ion species (e.g., `'p'` for protons, `'D+'` for
        deuterium, `'He-4 +1'` for singly ionized helium-4, etc.). If no charge
        state information is provided, then the ions are assumed to be singly
        ionized. If the density is an ion number density, then this paramter
        is required in order to convert to mass density.

    z_mean : `~numbers.Real`, optional
        The average ionization state (arithmetic mean) of the ``ion`` composing
        the plasma.  This is used in calculating the mass density
        :math:`ρ = n_i (m_i + Z_{mean} m_e)`.  ``z_mean`` is ignored if
        ``density`` is passed as a mass density and overrides any charge state
        info provided by ``ion``.

    Returns
    -------
    V_A : `~astropy.units.Quantity`
        The Alfvén speed in units :math:`m/s`.

    Raises
    ------
    `~plasmapy.utils.exceptions.RelativityError`
        If the Alfvén velocity is greater than or equal to the speed of light.

    `TypeError`
        If ``B`` and/or ``density`` are not of type `~astropy.units.Quantity`,
        or convertible.

    `TypeError`
        If ``ion`` is not of type or convertible to `~plasmapy.particles.Particle`.

    `TypeError`
        If ``z_mean`` is not of type `int` or `float`.

    `~astropy.units.UnitTypeError`
        If the magnetic field ``B`` does not have units equivalent to
        tesla.

    `~astropy.units.UnitTypeError`
        If the ``density`` does not have units equivalent to a number density
        or mass density.

    `ValueError`
        If ``density`` is negative.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the Alfvén velocity exceeds 5% of the speed of light.

    : `~astropy.units.UnitsWarning`
        If units are not provided for the magnetic field ``B``, units of
        tesla are assumed.

    Notes
    -----
    This expression does not account for relativistic effects, and
    loses validity when the resulting speed is a significant fraction
    of the speed of light.

    Examples
    --------
    >>> from astropy import units as u
    >>> from astropy.constants.si import m_p, m_e
    >>> B = 0.014*u.T
    >>> n = 5e19*u.m**-3
    >>> rho = n*(m_p+m_e)
    >>> ion = 'p'
    >>> Alfven_speed(B, n, ion=ion)
    <Quantity 43173.870... m / s>
    >>> Alfven_speed(B, rho)
    <Quantity 43173.870... m / s>
    >>> Alfven_speed(B, rho).to(u.cm/u.us)
    <Quantity 4.317387 cm / us>
    >>> Alfven_speed(B, n, ion="He +2")
    <Quantity 21664.18... m / s>
    >>> Alfven_speed(B, n, ion="He++")
    <Quantity 21664.18... m / s>
    >>> Alfven_speed(B, n, ion="He", z_mean=1.8)
    <Quantity 21661.51... m / s>
    """
    if density.unit.is_equivalent(u.kg / u.m ** 3):
        rho = density
    else:
        if not isinstance(ion, Particle):
            try:
                ion = Particle(ion)
            except TypeError:
                raise TypeError(
                    f"If passing a number density, you must pass a plasmapy Particle "
                    f"(not type {type(ion)}) to calculate the mass density!"
                )
        if z_mean is None:
            try:
                z_mean = abs(ion.integer_charge)
            except ChargeError:
                z_mean = 1

        z_mean = abs(z_mean)
        rho = mass_density(density, ion) + mass_density(density, "e", z_ratio=z_mean)

    V_A = np.abs(B) / np.sqrt(mu0 * rho)
    return V_A


va_ = Alfven_speed
""" Alias to :func:`Alfven_speed`. """


@check_relativistic
@validate_quantities(
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False, "none_shall_pass": True},
    k={"can_be_negative": False, "none_shall_pass": True},
)
def ion_sound_speed(
    T_e: u.K,
    T_i: u.K,
    ion: Particle,
    n_e: u.m ** -3 = None,
    k: u.m ** -1 = None,
    gamma_e=1,
    gamma_i=3,
    z_mean=None,
) -> u.m / u.s:
    r"""
    Return the ion sound speed for an electron-ion plasma.

    **Aliases:** `cs_`

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        Electron temperature in units of temperature or energy per
        particle. If this is not given, then the electron temperature
        is assumed to be zero.

    T_i : `~astropy.units.Quantity`
        Ion temperature in units of temperature or energy per
        particle.  If this is not given, then the ion temperature is
        assumed to be zero.

    ion : `~plasmapy.particles.Particle`
        Representation of the ion species (e.g., `'p'` for protons,
        `'D+'` for deuterium, or 'He-4 +1' for singly ionized
        helium-4). If no charge state information is provided, then the
        ions are assumed to be singly charged.

    n_e : `~astropy.units.Quantity`
        Electron number density. If this is not given, then ion_sound_speed
        will be approximated in the non-dispersive limit
        (:math:`k^2 λ_{D}^2` will be assumed zero). If ``n_e`` is given,
        a value for ``k`` must also be given.

    k : `~astropy.units.Quantity`
        Wavenumber (in units of inverse length, e.g. m\ :sup:`-1`\ ). If this
        is not given, then ion_sound_speed will be approximated in the
        non-dispersive limit (:math:`k^2 λ_{D}^2` will be assumed zero).
        If ``k`` is given, a value for ``n_e`` must also be given.

    gamma_e : `float` or `int`
        The adiabatic index for electrons, which defaults to 1.  This
        value assumes that the electrons are able to equalize their
        temperature rapidly enough that the electrons are effectively
        isothermal.

    gamma_i : `float` or `int`
        The adiabatic index for ions, which defaults to 3.  This value
        assumes that ion motion has only one degree of freedom, namely
        along magnetic field lines.

    z_mean : `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) for a plasma where the
        a macroscopic description is valid. If this quantity is not
        given then the atomic charge state (integer) of the ion
        is used. This is effectively an average ion sound speed for the
        plasma where multiple charge states are present.

    Returns
    -------
    V_S : `~astropy.units.Quantity`
        The ion sound speed in units of meters per second.

    Raises
    ------
    `TypeError`
        If any of the arguments are not entered as keyword arguments
        or are of an incorrect type.

    `ValueError`
        If the ion mass, adiabatic index, or temperature are invalid.

    `~plasmapy.utils.exceptions.PhysicsError`
        If an adiabatic index is less than one.

    `~astropy.units.UnitConversionError`
        If the temperature, electron number density, or wavenumber
        is in incorrect units.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the ion sound speed exceeds 5% of the speed of light.

    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    : `~plasmapy.utils.exceptions.PhysicsWarning`
        If only one of ``k`` or ``n_e`` is given, the non-dispersive
        limit is assumed.

    Notes
    -----
    The ion sound speed :math:`V_S` is given by

    .. math::

        V_S = \sqrt{\frac{γ_e Z k_B T_e + γ_i k_B T_i}{m_i (1 + k^2 λ_{D}^2)}}

    where :math:`γ_e` and :math:`γ_i` are the electron and
    ion adiabatic indices, :math:`k_B` is the Boltzmann constant,
    :math:`T_e` and :math:`T_i` are the electron and ion temperatures,
    :math:`Z` is the charge state of the ion, :math:`m_i` is the
    ion mass, :math:`λ_D` is the Debye length, and :math:`k` is the
    wavenumber.

    In the non-dispersive limit (:math:`k^2 λ_D^2` is small) the
    equation for :math:`V_S` is approximated (the denominator reduces
    to :math:`m_i`).

    When the electron temperature is much greater than the ion
    temperature, the ion sound velocity reduces to
    :math:`\sqrt{γ_e k_B T_e / m_i}`. Ion acoustic waves can
    therefore occur even when the ion temperature is zero.

    Example
    -------
    >>> from astropy import units as u
    >>> n = 5e19*u.m**-3
    >>> k_1 = 3e1*u.m**-1
    >>> k_2 = 3e7*u.m**-1
    >>> ion_sound_speed(T_e=5e6*u.K, T_i=0*u.K, ion='p', gamma_e=1, gamma_i=3)
    <Quantity 203155... m / s>
    >>> ion_sound_speed(T_e=5e6*u.K, T_i=0*u.K, n_e=n, k=k_1, ion='p', gamma_e=1, gamma_i=3)
    <Quantity 203155... m / s>
    >>> ion_sound_speed(T_e=5e6*u.K, T_i=0*u.K, n_e=n, k=k_2, ion='p', gamma_e=1, gamma_i=3)
    <Quantity 310.31... m / s>
    >>> ion_sound_speed(T_e=5e6*u.K, T_i=0*u.K, n_e=n, k=k_1, ion='p')
    <Quantity 203155... m / s>
    >>> ion_sound_speed(T_e=500*u.eV, T_i=200*u.eV, n_e=n, k=k_1, ion='D+')
    <Quantity 229585... m / s>

    """

    m_i = particles.particle_mass(ion)
    Z = _grab_charge(ion, z_mean)

    for gamma, species in zip([gamma_e, gamma_i], ["electrons", "ions"]):
        if not isinstance(gamma, (numbers.Real, numbers.Integral)):
            raise TypeError(
                f"The adiabatic index gamma for {species} must be a float or int"
            )
        if gamma < 1:
            raise PhysicsError(
                f"The adiabatic index for {species} must be between "
                f"one and infinity"
            )

    # Assume non-dispersive limit if values for n_e (or k) are not specified
    klD2 = 0.0
    if (n_e is None) ^ (k is None):
        warnings.warn(
            "The non-dispersive limit has been assumed for "
            "this calculation. To prevent this, values must "
            "be specified for both n_e and k.",
            PhysicsWarning,
        )
    elif n_e is not None and k is not None:
        lambda_D = Debye_length(T_e, n_e)
        klD2 = (k * lambda_D) ** 2

    try:
        V_S_squared = (gamma_e * Z * k_B * T_e + gamma_i * k_B * T_i) / (
            m_i * (1 + klD2)
        )
        V_S = np.sqrt(V_S_squared).to(u.m / u.s)
    except Exception:
        raise ValueError("Unable to find ion sound speed.")

    return V_S


cs_ = ion_sound_speed
""" Alias to :func:`ion_sound_speed`. """

# -----                                                             thermal_speed  -----

# This dictionary defines coefficients for thermal speeds
# calculated for different methods and values of ndim.
# Created here to avoid re-instantiating on each call
#
thermal_speed_coefficients = {
    1: {"most_probable": 0, "rms": 1, "mean_magnitude": 2 / np.pi},
    2: {"most_probable": 1, "rms": 2, "mean_magnitude": np.pi / 2},
    3: {"most_probable": 2, "rms": 3, "mean_magnitude": 8 / np.pi},
}
"""
Dictionary of various coefficients used in calculating the
`~plasmapy.formulary.parameters.thermal_speed`.
"""


@preserve_signature
@njit
def thermal_speed_lite(T, mass, coeff):
    """
    A lite weight version of `thermal_speed` intended for computational used
    and, thus, does not do any argument validation/conditioning.
    """
    return np.sqrt(coeff * k_B_si_unitless * T / mass)


@add_lite(
    thermal_speed_lite,
    attrs=[("coefficients", "thermal_speed_coefficients")],
    scope=globals(),
)
@check_relativistic
@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    mass={"can_be_negative": False, "can_be_nan": True},
)
@particles.particle_input
def thermal_speed(
    T: u.K, particle: Particle, method="most_probable", mass: u.kg = None, ndim=3,
) -> u.m / u.s:
    r"""
    Return the most probable speed for a particle within a Maxwellian
    distribution.

    **Aliases:** `~plasmapy.formulary.parameters.vth_`

    **Lightwieght Version:** `~plasmapy.formulary.parameters.thermal_speed_lite`

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The particle temperature in either kelvin or energy per particle

    particle : `~plasmapy.particles.Particle`
        Representation of the particle species (e.g., ``'p'`` for protons, ``'D+'``
        for deuterium, or ``'He-4 +1'`` for singly ionized helium-4). If no
        charge state information is provided, then the particles are
        assumed to be singly charged.

    method : `str`, optional
        Method to be used for calculating the thermal speed. Options are
        ``'most_probable'`` (default), ``'rms'``, and ``'mean_magnitude'``.

    mass : `~astropy.units.Quantity`
        The particle's mass override. Defaults to `~np.nan` and if so, doesn't do
        anything, but if set, overrides mass acquired from `particle`. Useful
        with relative velocities of particles.

    ndim : `int`
        Dimensionality of space in which to calculate thermal velocity. Valid
        values are 1, 2, or 3.

    Returns
    -------
    V : `~astropy.units.Quantity`
        The particle's thermal speed.

    Raises
    ------
    `TypeError`
        The particle temperature is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the particle temperature is not in units of temperature or
        energy per particle.

    `ValueError`
        The particle temperature is invalid or particle cannot be used to
        identify an isotope or particle.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the ion sound speed exceeds 5% of the speed of light.

    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The particle thermal speed is given by:

    .. math::
        V_{th,i} = \sqrt{\frac{N k_B T_i}{m_i}}

    where the value of :math:`N` depends on the dimensionality and the
    definition of :math:`v_{th}`: most probable, root-mean-square (RMS),
    or mean magnitude. The value of :math:`N` in each case is

    .. list-table:: Values of constant :math:`N`
       :widths: 50, 25, 25, 25
       :header-rows: 1

       * - Dim.
         - Most-Probable
         - RMS
         - Mean-Magnitude
       * - 1D
         - 0
         - 1
         - :math:`2/π`
       * - 2D
         - 1
         - 2
         - :math:`π/2`
       * - 3D
         - 2
         - 3
         - :math:`8/π`

    The definition of thermal velocity varies by
    the square root of two depending on whether or not this velocity
    absorbs that factor in the expression for a Maxwellian
    distribution.  In particular, the expression given in the NRL
    Plasma Formulary [1] is a square root of two smaller than the
    result from this function.

    Examples
    --------
    >>> from astropy import units as u
    >>> thermal_speed(5*u.eV, 'p')
    <Quantity 30949.6... m / s>
    >>> thermal_speed(1e6*u.K, particle='p')
    <Quantity 128486... m / s>
    >>> thermal_speed(5*u.eV, particle='e-')
    <Quantity 132620... m / s>
    >>> thermal_speed(1e6*u.K, particle='e-')
    <Quantity 550569... m / s>
    >>> thermal_speed(1e6*u.K, "e-", method="rms")
    <Quantity 674307... m / s>
    >>> thermal_speed(1e6*u.K, "e-", method="mean_magnitude")
    <Quantity 621251... m / s>
    """
    if mass is None:
        mass = particles.particle_mass(particle)

    # different methods, as per https://en.wikipedia.org/wiki/Thermal_velocity
    try:
        coeff = thermal_speed_coefficients[ndim]
    except KeyError:
        raise ValueError("{ndim} is not a supported value for ndim in thermal_speed")
    try:
        coeff = coeff[method]
    except KeyError:
        raise ValueError("Method {method} not supported in thermal_speed")

    speed = thermal_speed_lite(T=T.value, mass=mass.value, coeff=coeff)
    return speed * u.m / u.s


vth_ = thermal_speed
""" Alias to :func:`~plasmapy.formulary.parameters.thermal_speed`. """

# -----                                                          thermal_pressure  -----


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n={"can_be_negative": False},
)
def thermal_pressure(T: u.K, n: u.m ** -3) -> u.Pa:
    r"""
    Return the thermal pressure for a Maxwellian distribution.

    **Aliases:** `pth_`

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The particle temperature in either kelvin or energy per particle.

    n : `~astropy.units.Quantity`
        The particle number density in units convertible to m\ :sup:`-3`\ .

    Examples
    --------
    >>> import astropy.units as u
    >>> thermal_pressure(1*u.eV, 1e20/u.m**3)
    <Quantity 16.021... Pa>
    >>> thermal_pressure(10*u.eV, 1e20/u.m**3)
    <Quantity 160.21... Pa>

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
    """
    return n * k_B * T


pth_ = thermal_pressure
""" Alias to :func:`thermal_pressure`. """


@check_relativistic
@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()}
)
def kappa_thermal_speed(
    T: u.K, kappa, particle: Particle, method="most_probable"
) -> u.m / u.s:
    r"""Return the most probable speed for a particle within a Kappa
    distribution.

    **Aliases:** `vth_kappa_`

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The particle temperature in either kelvin or energy per particle

    kappa: `float`
        The ``kappa`` parameter is a dimensionless number which sets the slope
        of the energy spectrum of suprathermal particles forming the tail
        of the Kappa velocity distribution function. ``kappa`` must be greater
        than 3/2.

    particle : `~plasmapy.particles.Particle`
        Representation of the particle species (e.g., 'p' for protons, 'D+'
        for deuterium, or 'He-4 +1' for singly ionized helium-4). If no
        charge state information is provided, then the particles are
        assumed to be singly charged.

    method : `str`, optional
        Method to be used for calculating the thermal speed. Options are
        ``'most_probable'`` (default), ``'rms'``, and ``'mean_magnitude'``.

    Returns
    -------
    V : `~astropy.units.Quantity`
        Particle thermal speed.

    Raises
    ------
    `TypeError`
        The particle temperature is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the particle temperature is not in units of temperature or
        energy per particle.

    `ValueError`
        The particle temperature is invalid or particle cannot be used to
        identify an isotope or particle.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the particle thermal speed exceeds 5% of the speed of light.

    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The particle thermal speed is given by:

    .. math::
        V_{th,i} = \sqrt{(2 κ - 3)\frac{2 k_B T_i}{κ m_i}}

    For more discussion on the ``'mean_magnitude'`` calculation method,
    see [1]_.


    Examples
    --------
    >>> from astropy import units as u
    >>> kappa_thermal_speed(5*u.eV, 4, 'p') # defaults to most probable
    <Quantity 24467.87... m / s>
    >>> kappa_thermal_speed(5*u.eV, 4, 'p', 'rms')
    <Quantity 37905.47... m / s>
    >>> kappa_thermal_speed(5*u.eV, 4, 'p', 'mean_magnitude')
    <Quantity 34922.98... m / s>

    References
    ----------
    .. [1] PlasmaPy Issue #186, https://github.com/PlasmaPy/PlasmaPy/issues/186

    See Also
    --------
    ~plasmapy.formulary.kappa_thermal_speed
    ~plasmapy.formulary.kappa_velocity_1D
    """
    # Checking thermal units
    if kappa <= 3 / 2:
        raise ValueError(
            f"Must have kappa > 3/2, instead of {kappa}, for "
            "kappa distribution function to be valid."
        )
    # different methods, as per https://en.wikipedia.org/wiki/Thermal_velocity
    vTh = thermal_speed(T=T, particle=particle, method=method)

    if method == "most_probable":
        # thermal velocity of Kappa distribution function is just Maxwellian
        # thermal speed modulated by the following factor.
        # This is only true for "most probable" case. RMS and mean
        # magnitude velocities are same as Maxwellian.
        coeff = np.sqrt((kappa - 3 / 2) / kappa)
    else:
        coeff = 1

    return vTh * coeff


vth_kappa_ = kappa_thermal_speed
""" Alias to :func:`kappa_thermal_speed`. """


@validate_quantities(
    n={"can_be_negative": False},
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def Hall_parameter(
    n: u.m ** -3,
    T: u.K,
    B: u.T,
    ion: Particle,
    particle: Particle,
    coulomb_log=None,
    V=None,
    coulomb_log_method="classical",
):
    r"""
    Calculate the ``particle`` Hall parameter for a plasma.

    The Hall parameter for plasma species :math:`s` (``particle``) is given by:

    .. math::

        β_{s} = \frac{Ω_{c s}}{ν_{s s^{\prime}}}

    where :math:`Ω_{c s}` is the gyrofrequncy for plasma species :math:`s`
    (``particle``) and :math:`ν_{s s^{\prime}}` is the collision frequency
    between plasma species :math:`s` (``particle``) and species
    :math:`s^{\prime}` (``ion``).

    **Aliases:** `betaH_`

    Parameters
    ----------
    n : `~astropy.units.quantity.Quantity`
        The number density associated with ``particle``.

    T : `~astropy.units.quantity.Quantity`
        The temperature of associated with ``particle``.

    B : `~astropy.units.quantity.Quantity`
        The magnetic field.

    ion : `~plasmapy.particles.Particle`
        The type of ion ``particle`` is colliding with.

    particle : `~plasmapy.particles.Particle`
        The particle species for which the Hall parameter is calculated for.
        Representation of the particle species (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly ionized helium-4).
        If no charge state information is provided, then the particles are
        assumed to be singly charged.

    coulomb_log : `float`, optional
        Preset value for the Coulomb logarithm. Used mostly for testing purposes.

    V : `~astropy.units.quantity.Quantity`
        The relative velocity between ``particle`` and ``ion``.  If not provided,
        then the ``particle`` thermal velocity is assumed
        (`~plasmapy.formulary.parameters.thermal_speed`).

    coulomb_log_method : `str`, optional
        The method by which to compute the Coulomb logarithm.
        The default method is the classical straight-line Landau-Spitzer
        method (``"classical"`` or ``"ls"``). The other 6 supported methods
        are ``"ls_min_interp"``, ``"ls_full_interp"``, ``"ls_clamp_mininterp"``,
        ``"hls_min_interp"``, ``"hls_max_interp"``, and ``"hls_full_interp"``.
        Please refer to the docstring of
        `~plasmapy.formulary.collisions.Coulomb_logarithm` for more
        information about these methods.

    See Also
    --------
    ~plasmapy.formulary.parameters.gyrofrequency
    ~plasmapy.formulary.collisions.fundamental_electron_collision_freq
    ~plasmapy.formulary.collisions.fundamental_ion_collision_freq
    ~plasmapy.formulary.collisions.Coulomb_logarithm

    Returns
    -------
    `~astropy.units.quantity.Quantity`
        Hall parameter for ``particle``.

    Notes
    -----
    * For calculating the collision frequency
      `~plasmapy.formulary.collisions.fundamental_electron_collision_freq` is used
      when ``particle`` is an electron and
      `~plasmapy.formulary.collisions.fundamental_ion_collision_freq` when
      ``particle`` is an ion.
    * The collision frequencies are calculated assuming a slowly moving
      Maxwellian distribution.

    Examples
    --------
    >>> from astropy import units as u
    >>> Hall_parameter(1e10 * u.m**-3, 2.8e3 * u.eV, 2.3 * u.T, 'He-4 +1', 'e-')
    <Quantity 7.26446...e+16>
    >>> Hall_parameter(1e10 * u.m**-3, 5.8e3 * u.eV, 2.3 * u.T, 'He-4 +1', 'e-')
    <Quantity 2.11158...e+17>
    """
    from plasmapy.formulary.collisions import (
        fundamental_electron_collision_freq,
        fundamental_ion_collision_freq,
    )

    gyro_frequency = gyrofrequency(B, particle)
    gyro_frequency = gyro_frequency / u.radian
    if particles.Particle(particle).symbol == "e-":
        coll_rate = fundamental_electron_collision_freq(
            T, n, ion, coulomb_log, V, coulomb_log_method=coulomb_log_method
        )
    else:
        coll_rate = fundamental_ion_collision_freq(T, n, ion, coulomb_log, V)
    return gyro_frequency / coll_rate


betaH_ = Hall_parameter
""" Alias to :func:`Hall_parameter`. """


@validate_quantities(
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    }
)
@angular_freq_to_hz
def gyrofrequency(B: u.T, particle: Particle, signed=False, Z=None) -> u.rad / u.s:
    r"""
    Calculate the particle gyrofrequency in units of radians per second.

    **Aliases:** `oc_`, `wc_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    particle : `~plasmapy.particles.Particle`
        Representation of the particle species (e.g., 'p' for protons, 'D+'
        for deuterium, or 'He-4 +1' for singly ionized helium-4). If no
        charge state information is provided, then the particles are assumed
        to be singly charged.

    signed : `bool`, optional
        The gyrofrequency can be defined as signed (negative for electron,
        positive for ion). Default is `False` (unsigned, i.e. always
        positive).

    Z : `float` or `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) for a plasma where the
        a macroscopic description is valid. If this quantity is not
        given then the atomic charge state (integer) of the ion
        is used. This is effectively an average gyrofrequency for the
        plasma where multiple charge states are present, and should
        not be interpreted as the gyrofrequency for any single particle.
        If not provided, it defaults to the integer charge of the ``particle``.

    Returns
    -------
    omega_c : `~astropy.units.Quantity`
        The particle gyrofrequency in units of radians per second.

    Raises
    ------
    `TypeError`
        If the magnetic field is not a `~astropy.units.Quantity` or
        ``particle`` is not of an appropriate type.

    `ValueError`
        If the magnetic field contains invalid values or particle cannot
        be used to identify an particle or isotope.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The particle gyrofrequency is the angular frequency of particle
    gyration around magnetic field lines and is given by:

    .. math::
        ω_{ci} = \frac{Z e B}{m_i}

    The particle gyrofrequency is also known as the particle cyclotron
    frequency or the particle Larmor frequency.

    The recommended way to convert from angular frequency to frequency
    is to use an equivalency between cycles per second and hertz, as
    Astropy's `~astropy.units.dimensionles_angles` equivalency does not
    account for the factor of :math:`2π` needed during this conversion.  The
    `~astropy.units.dimensionless_angles` equivalency is appropriate
    when dividing a velocity by an angular frequency to get a length scale.

    Examples
    --------
    >>> from astropy import units as u
    >>> gyrofrequency(0.1*u.T, 'e-')
    <Quantity 1.7588...e+10 rad / s>
    >>> gyrofrequency(0.1*u.T, 'e-', to_hz=True)
    <Quantity 2.79924...e+09 Hz>
    >>> gyrofrequency(0.1*u.T, 'e-', signed=True)
    <Quantity -1.75882...e+10 rad / s>
    >>> gyrofrequency(0.01*u.T, 'p')
    <Quantity 957883.32... rad / s>
    >>> gyrofrequency(0.01*u.T, 'p', signed=True)
    <Quantity 957883.32... rad / s>
    >>> gyrofrequency(0.01*u.T, particle='T+')
    <Quantity 319964.5... rad / s>
    >>> gyrofrequency(0.01*u.T, particle='T+', to_hz=True)
    <Quantity 50923.9... Hz>
    >>> omega_ce = gyrofrequency(0.1*u.T, 'e-')
    >>> print(omega_ce)
    1758820... rad / s
    >>> f_ce = omega_ce.to(u.Hz, equivalencies=[(u.cy/u.s, u.Hz)])
    >>> print(f_ce)
    279924... Hz

    """
    m_i = particles.particle_mass(particle)
    Z = _grab_charge(particle, Z)
    if not signed:
        Z = abs(Z)

    omega_ci = u.rad * (Z * e * np.abs(B) / m_i).to(1 / u.s)

    return omega_ci


oc_ = gyrofrequency
""" Alias to :func:`gyrofrequency`. """

wc_ = gyrofrequency
""" Alias to :func:`gyrofrequency`. """


@validate_quantities(
    Vperp={"can_be_nan": True},
    T_i={"can_be_nan": True, "equivalencies": u.temperature_energy()},
    validations_on_return={"equivalencies": u.dimensionless_angles()},
)
def gyroradius(
    B: u.T,
    particle: Particle,
    *,
    Vperp: u.m / u.s = np.nan * u.m / u.s,
    T_i: u.K = np.nan * u.K,
) -> u.m:
    r"""Return the particle gyroradius.

    **Aliases:** `rc_`, `rhoc_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    particle : `~plasmapy.particles.Particle`
        Representation of the particle species (e.g., `'p'` for protons, `'D+'`
        for deuterium, or `'He-4 +1'` for singly ionized helium-4).  If no
        charge state information is provided, then the particles are assumed
        to be singly charged.

    Vperp : `~astropy.units.Quantity`, optional, keyword-only
        The component of particle velocity that is perpendicular to the
        magnetic field in units convertible to meters per second.

    T_i : `~astropy.units.Quantity`, optional, keyword-only
        The particle temperature in units convertible to kelvin.

    Returns
    -------
    r_Li : `~astropy.units.Quantity`
        The particle gyroradius in units of meters.  This
        `~astropy.units.Quantity` will be based on either the
        perpendicular component of particle velocity as inputted, or
        the most probable speed for an particle within a Maxwellian
        distribution for the particle temperature.

    Raises
    ------
    `TypeError`
        The arguments are of an incorrect type.

    `~astropy.units.UnitConversionError`
        The arguments do not have appropriate units.

    `ValueError`
        If any argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    One but not both of ``Vperp`` and ``T_i`` must be inputted.

    If any of ``B``, ``Vperp``, or ``T_i`` is a number rather than a
    `~astropy.units.Quantity`, then SI units will be assumed and a
    warning will be raised.

    The particle gyroradius is also known as the particle Larmor
    radius and is given by

    .. math::
        r_{Li} = \frac{V_{\perp}}{ω_{ci}}

    where :math:`V_⟂` is the component of particle velocity that is
    perpendicular to the magnetic field and :math:`ω_{ci}` is the
    particle gyrofrequency.  If a temperature is provided, then
    :math:`V_⟂` will be the most probable thermal velocity of an
    particle at that temperature.

    Examples
    --------
    >>> from astropy import units as u
    >>> gyroradius(0.2*u.T, particle='p+', T_i=1e5*u.K)
    <Quantity 0.002120... m>
    >>> gyroradius(0.2*u.T, particle='p+', T_i=1e5*u.K)
    <Quantity 0.002120... m>
    >>> gyroradius(5*u.uG, particle='alpha', T_i=1*u.eV)
    <Quantity 288002.38... m>
    >>> gyroradius(400*u.G, particle='Fe+++', Vperp=1e7*u.m/u.s)
    <Quantity 48.23129... m>
    >>> gyroradius(B=0.01*u.T, particle='e-', T_i=1e6*u.K)
    <Quantity 0.003130... m>
    >>> gyroradius(0.01*u.T, 'e-', Vperp=1e6*u.m/u.s)
    <Quantity 0.000568... m>
    >>> gyroradius(0.2*u.T, 'e-', T_i=1e5*u.K)
    <Quantity 4.94949...e-05 m>
    >>> gyroradius(5*u.uG, 'e-', T_i=1*u.eV)
    <Quantity 6744.25... m>
    >>> gyroradius(400*u.G, 'e-', Vperp=1e7*u.m/u.s)
    <Quantity 0.001421... m>
    """

    isfinite_Ti = np.isfinite(T_i)
    isfinite_Vperp = np.isfinite(Vperp)

    # check 1: ensure either Vperp or T_i invalid, keeping in mind that
    # the underlying values of the astropy quantity may be numpy arrays
    if np.any(np.logical_not(np.logical_xor(isfinite_Vperp, isfinite_Ti))):
        raise ValueError(
            "Must give Vperp or T_i, but not both, as arguments to gyroradius"
        )

    # check 2: get Vperp as the thermal speed if is not already a valid input
    if np.isscalar(Vperp.value) and np.isscalar(
        T_i.value
    ):  # both T_i and Vperp are scalars
        # we know exactly one of them is nan from check 1
        if isfinite_Ti:
            # T_i is valid, so use it to determine Vperp
            Vperp = thermal_speed(T_i, particle=particle)
        # else: Vperp is already valid, do nothing
    elif np.isscalar(Vperp.value):  # only T_i is an array
        # this means either Vperp must be nan, or T_i must be array of all nan,
        # or else we couldn't have gotten through check 1
        if isfinite_Vperp:
            # Vperp is valid, T_i is a vector that is all nan
            # uh...
            Vperp = np.repeat(Vperp, len(T_i))
        else:
            # normal case where Vperp is scalar nan and T_i is valid array
            Vperp = thermal_speed(T_i, particle=particle)
    elif np.isscalar(T_i.value):  # only Vperp is an array
        # this means either T_i must be nan, or V_perp must be array of all nan,
        # or else we couldn't have gotten through check 1
        if isfinite_Ti:
            # T_i is valid, V_perp is an array of all nan
            # uh...
            Vperp = thermal_speed(np.repeat(T_i, len(Vperp)), particle=particle)
        # else: normal case where T_i is scalar nan and Vperp is already a valid array
        # so, do nothing
    else:  # both T_i and Vperp are arrays
        # we know all the elementwise combinations have one nan and one finite, due to check 1
        # use the valid Vperps, and replace the others with those calculated from T_i
        Vperp = Vperp.copy()  # avoid changing Vperp's value outside function
        Vperp[isfinite_Ti] = thermal_speed(T_i[isfinite_Ti], particle=particle)

    omega_ci = gyrofrequency(B, particle)

    r_Li = np.abs(Vperp) / omega_ci

    return r_Li


rc_ = gyroradius
""" Alias to :func:`gyroradius`. """

rhoc_ = gyroradius
""" Alias to :func:`gyroradius`. """


@validate_quantities(
    n={"can_be_negative": False},
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    },
)
@angular_freq_to_hz
def plasma_frequency(n: u.m ** -3, particle: Particle, z_mean=None) -> u.rad / u.s:
    r"""Calculate the particle plasma frequency.

    **Aliases:** `wp_`

    Parameters
    ----------
    n : `~astropy.units.Quantity`
        Particle number density in units convertible to per cubic meter.

    particle : `~plasmapy.particles.Particle`
        Representation of the particle species (e.g., 'p' for protons, 'D+'
        for deuterium, or 'He-4 +1' for singly ionized helium-4). If no
        charge state information is provided, then the particles are assumed
        to be singly charged.

    z_mean : `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) for a plasma where the
        a macroscopic description is valid. If this quantity is not
        given then the atomic charge state (`int`) of the ion
        is used. This is effectively an average plasma frequency for the
        plasma where multiple charge states are present.

    Returns
    -------
    omega_p : `~astropy.units.Quantity`
        The particle plasma frequency in radians per second.

    Raises
    ------
    `TypeError`
        If ``n_i`` is not a `~astropy.units.Quantity` or particle is not of
        an appropriate type.

    `~astropy.units.UnitConversionError`
        If ``n_i`` is not in correct units.

    `ValueError`
        If ``n_i`` contains invalid values or particle cannot be used to
        identify an particle or isotope.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The particle plasma frequency is

    .. math::
        ω_{pi} = Z e \sqrt{\frac{n_i}{\epsilon_0 m_i}}

    At present, `astropy.units` does not allow direct conversions from
    radians/second for angular frequency to 1/second or Hz for
    frequency.  The `~astropy.units.dimensionless_angles` equivalency
    allows for that conversion, but does not account for the factor of
    :math:`2π`\ . The alternatives are to convert to cycle/second or to
    do the conversion manually, as shown in the examples.

    Example
    -------
    >>> from astropy import units as u
    >>> plasma_frequency(1e19*u.m**-3, particle='p')
    <Quantity 4.16329...e+09 rad / s>
    >>> plasma_frequency(1e19*u.m**-3, particle='p', to_hz=True)
    <Quantity 6.62608...e+08 Hz>
    >>> plasma_frequency(1e19*u.m**-3, particle='D+')
    <Quantity 2.94462...e+09 rad / s>
    >>> plasma_frequency(1e19*u.m**-3, 'e-')
    <Quantity 1.78398...e+11 rad / s>
    >>> plasma_frequency(1e19*u.m**-3, 'e-', to_hz=True)
    <Quantity 2.83930...e+10 Hz>
    """

    try:
        m = particles.particle_mass(particle)
        if z_mean is None:
            # warnings.warn("No z_mean given, defaulting to atomic charge",
            #               PhysicsWarning)
            try:
                Z = particles.integer_charge(particle)
            except Exception:
                Z = 1
        else:
            # using user provided average ionization
            Z = z_mean
        Z = np.abs(Z)
        # TODO REPLACE WITH Z = np.abs(_grab_charge(particle, z_mean)), some bugs atm
    except Exception:
        raise ValueError(f"Invalid particle, {particle}, in plasma_frequency.")

    omega_p = u.rad * Z * e * np.sqrt(n / (eps0 * m))

    return omega_p.si


wp_ = plasma_frequency
""" Alias to :func:`plasma_frequency`. """


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def Debye_length(T_e: u.K, n_e: u.m ** -3) -> u.m:
    r"""Calculate the characteristic decay length for electric fields,
     due to charge screening.

    **Aliases:** `lambdaD_`

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        Electron temperature.

    n_e : `~astropy.units.Quantity`
        Electron number density.

    Returns
    -------
    lambda_D : `~astropy.units.Quantity`
        The Debye length in meters.

    Raises
    ------
    `TypeError`
        If either argument is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If either argument is in incorrect units.

    `ValueError`
        If either argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The Debye length is the exponential scale length for charge
    screening and is given by

    .. math::
        λ_D = \sqrt{\frac{ε_0 k_b T_e}{n_e e^2}}

    for an electron plasma with nearly stationary ions.

    The electrical potential will drop by a factor of :math:`1/e` every Debye
    length.

    Plasmas will generally be quasineutral on length scales significantly
    larger than the Debye length.

    See Also
    --------
    Debye_number

    Example
    -------
    >>> from astropy import units as u
    >>> Debye_length(5e6*u.K, 5e15*u.m**-3)
    <Quantity 0.002182... m>

    """
    lambda_D = np.sqrt(eps0 * k_B * T_e / (n_e * e ** 2))
    return lambda_D


lambdaD_ = Debye_length
""" Alias to :func:`Debye_length`. """


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def Debye_number(T_e: u.K, n_e: u.m ** -3) -> u.dimensionless_unscaled:
    r"""Return the number of electrons within a sphere with a radius
    of the Debye length.

    **Aliases:** `nD_`

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        Electron temperature.

    n_e : `~astropy.units.Quantity`
        Electron number density.

    Raises
    ------
    `TypeError`
        If either argument is not a `~astropy.units.Quantity`.

    `astropy.units.UnitConversionError`
        If either argument is in incorrect units.

    `ValueError`
        If either argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Returns
    -------
    N_D : `~astropy.units.Quantity`
        Number of electrons within a sphere with a radius of the Debye
        length.

    Notes
    -----
    The Debye number is the number of electrons contained within a
    sphere with a radius of a Debye length and is given by

    .. math::
        N_D = \frac{4π}{3} n_e λ_D^3

    The Debye number is also known as the plasma parameter.

    Collective behavior requires :math:`N_D ≫ 1`\ .

    See Also
    --------
    Debye_length

    Example
    -------
    >>> from astropy import units as u
    >>> Debye_number(5e6*u.K, 5e9*u.cm**-3)
    <Quantity 2.17658...e+08>

    """

    lambda_D = Debye_length(T_e, n_e)
    N_D = (4 / 3) * np.pi * n_e * lambda_D ** 3

    return N_D


nD_ = Debye_number
""" Alias to :func:`Debye_number`. """


@validate_quantities(
    n={"can_be_negative": False},
    validations_on_return={"equivalencies": u.dimensionless_angles()},
)
@particles.particle_input(require="charged")
def inertial_length(n: u.m ** -3, particle: Particle) -> u.m:
    r"""
    Calculate a charged particle's inertial length.

    **Aliases:** `cwp_`

    Parameters
    ----------
    n : `~astropy.units.Quantity`
        Particle number density in units convertible to m\ :sup:`-3`\ .

    particle : `~plasmapy.particles.Particle`
        Representation of the particle species (e.g., 'p+' for protons,
        'D+' for deuterium, or 'He-4 +1' for singly ionized helium-4).

    Returns
    -------
    d : `~astropy.units.Quantity`
        The particle's inertial length in meters.

    Raises
    ------
    `TypeError`
        If ``n`` is not a `~astropy.units.Quantity` or ``particle`` is
        not a string.

    `~astropy.units.UnitConversionError`
        If ``n`` is not in units of a number density.

    `ValueError`
        The particle density does not have an appropriate value.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided and SI units are assumed.

    Notes
    -----
    The inertial length of a particle of species :math:`s` is given by

    .. math::
        d = \frac{c}{ω_{ps}}

    The inertial length is the characteristic length scale for a
    particle to be accelerated in a plasma.  The Hall effect becomes
    important on length scales shorter than the ion inertial length.

    The inertial length is also known as the skin depth.

    Example
    -------
    >>> from astropy import units as u
    >>> inertial_length(5 * u.m ** -3, 'He+')
    <Quantity 2.02985...e+08 m>
    >>> inertial_length(5 * u.m ** -3, 'e-')
    <Quantity 2376534.75... m>

    """
    omega_p = plasma_frequency(n, particle=particle)

    return c / omega_p


cwp_ = inertial_length
"""
Alias to :func:`inertial_length`.

* Name is shorthand for :math:`c / \\omega_p`.
"""


@validate_quantities
def magnetic_pressure(B: u.T) -> u.Pa:
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

    Notes
    -----
    The magnetic pressure is given by:

    .. math::
        p_B = \frac{B^2}{2 μ_0}

    The motivation behind having two separate functions for magnetic
    pressure and magnetic energy density is that it allows greater
    insight into the physics that are being considered by the user and
    thus more readable code.

    See Also
    --------
    magnetic_energy_density : returns an equivalent `~astropy.units.Quantity`,
        except in units of joules per cubic meter.

    Example
    -------
    >>> from astropy import units as u
    >>> magnetic_pressure(0.1*u.T).to(u.Pa)
    <Quantity 3978.87... Pa>

    """
    return (B ** 2) / (2 * mu0)


pmag_ = magnetic_pressure
""" Alias to :func:`magnetic_pressure`. """


@validate_quantities
def magnetic_energy_density(B: u.T) -> u.J / u.m ** 3:
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
        If the magnetic field strength does not have an appropriate.
        value.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed

    Notes
    -----
    The magnetic energy density is given by:

    .. math::
        E_B = \frac{B^2}{2 μ_0}

    The motivation behind having two separate functions for magnetic
    pressure and magnetic energy density is that it allows greater
    insight into the physics that are being considered by the user and
    thus more readable code.

    See Also
    --------
    magnetic_pressure : Returns an equivalent `~astropy.units.Quantity`,
        except in units of pascals.

    Example
    -------
    >>> from astropy import units as u
    >>> magnetic_energy_density(0.1*u.T)
    <Quantity 3978.87... J / m3>

    """
    return magnetic_pressure(B)


ub_ = magnetic_energy_density
""" Alias to :func:`magnetic_energy_density`. """


@validate_quantities(
    n_e={"can_be_negative": False},
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    },
)
@angular_freq_to_hz
def upper_hybrid_frequency(B: u.T, n_e: u.m ** -3) -> u.rad / u.s:
    r"""
    Return the upper hybrid frequency.

    **Aliases:** `wuh_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    n_e : `~astropy.units.Quantity`
        The electron number density.

    Returns
    -------
    omega_uh : `~astropy.units.Quantity`
        The upper hybrid frequency in radians per second.

    Raises
    ------
    `TypeError`
        If either of ``B`` or ``n_e`` is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If either of ``B`` or ``n_e`` is in incorrect units.

    `ValueError`
        If either of ``B`` or ``n_e`` contains invalid values or are of
        incompatible dimensions.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The upper hybrid frequency is given through the relation

    .. math::
        ω_{uh}^2 = ω_{ce}^2 + ω_{pe}^2

    where :math:`ω_{ce}` is the electron gyrofrequency and
    :math:`ω_{pe}` is the electron plasma frequency.

    Example
    -------
    >>> from astropy import units as u
    >>> upper_hybrid_frequency(0.2*u.T, n_e=5e19*u.m**-3)
    <Quantity 4.00459...e+11 rad / s>
    >>> upper_hybrid_frequency(0.2*u.T, n_e=5e19*u.m**-3, to_hz = True)
    <Quantity 6.37350...e+10 Hz>

    """
    omega_pe = plasma_frequency(n=n_e, particle="e-")
    omega_ce = gyrofrequency(B, "e-")
    omega_uh = np.sqrt(omega_pe ** 2 + omega_ce ** 2)

    return omega_uh


wuh_ = upper_hybrid_frequency
""" Alias to :func:`upper_hybrid_frequency`. """


@validate_quantities(
    n_i={"can_be_negative": False},
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    },
)
@angular_freq_to_hz
def lower_hybrid_frequency(B: u.T, n_i: u.m ** -3, ion: Particle) -> u.rad / u.s:
    r"""
    Return the lower hybrid frequency.

    **Aliases:** `wlh_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    n_i : `~astropy.units.Quantity`
        Ion number density.

    ion : `~plasmapy.particles.Particle`
        Representation of the ion species (e.g., ``'p'`` for protons, ``'D+'``
        for deuterium, or ``'He-4 +1'`` for singly ionized helium-4). If no
        charge state information is provided, then the ions are assumed to
        be singly charged.

    Returns
    -------
    omega_lh : `~astropy.units.Quantity`
        The lower hybrid frequency in radians per second.

    Raises
    ------
    `TypeError`
        If either of ``B`` or ``n_i`` is not a `~astropy.units.Quantity`,
        or ion is of an inappropriate type.

    `~astropy.units.UnitConversionError`
        If either of ``B`` or ``n_i`` is in incorrect units.

    `ValueError`
        If either of ``B`` or ``n_i`` contains invalid values or are of
        incompatible dimensions, or ion cannot be used to identify an
        ion or isotope.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The lower hybrid frequency is given through the relation

    .. math::
        \frac{1}{ω_{lh}^2} = \frac{1}{ω_{ci}^2 + ω_{pi}^2} +
        \frac{1}{ω_{ci}ω_{ce}}

    where :math:`ω_{ci}` is the ion gyrofrequency,
    :math:`ω_{ce}` is the electron gyrofrequency, and
    :math:`ω_{pi}` is the ion plasma frequency.

    Example
    -------
    >>> from astropy import units as u
    >>> lower_hybrid_frequency(0.2*u.T, n_i=5e19*u.m**-3, ion='D+')
    <Quantity 5.78372...e+08 rad / s>
    >>> lower_hybrid_frequency(0.2*u.T, n_i=5e19*u.m**-3, ion='D+', to_hz = True)
    <Quantity 92050879.3... Hz>

    """

    # We do not need a charge state here, so the sole intent is to
    # catch invalid ions.
    try:
        particles.integer_charge(ion)
    except Exception:
        raise ValueError("Invalid ion in lower_hybrid_frequency.")

    omega_ci = gyrofrequency(B, particle=ion)
    omega_pi = plasma_frequency(n_i, particle=ion)
    omega_ce = gyrofrequency(B, particle="e-")
    omega_lh = ((omega_ci * omega_ce) ** -1 + omega_pi ** -2) ** -0.5
    # TODO possibly optimize the above line via np.sqrt
    omega_lh = omega_lh

    return omega_lh


wlh_ = lower_hybrid_frequency
""" Alias to :func:`lower_hybrid_frequency`. """


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    B={"can_be_negative": False},
)
def Bohm_diffusion(T_e: u.K, B: u.T) -> u.m ** 2 / u.s:
    r"""
    Return the Bohm diffusion coefficient.

    The Bohm diffusion coefficient was conjectured to follow Bohm model
    of the diffusion of plasma across a magnetic field and describe the
    diffusion of early fusion energy machines. The rate predicted by
    Bohm diffusion is much higher than classical diffusion, and if there
    were no exceptions, magnetically confined fusion would be impractical.

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
    D_B = k_B * T_e / (16 * e * B)
    return D_B


DB_ = Bohm_diffusion
""" Alias to :func:`Bohm_diffusion`. """
