"""Functions to calculate fundamental plasma frequency parameters."""
__all__ = [
    "gyrofrequency",
    "lower_hybrid_frequency",
    "plasma_frequency",
    "upper_hybrid_frequency",
]
__aliases__ = ["oc_", "wc_", "wlh_", "wp_", "wuh_"]
__lite_funcs__ = ["plasma_frequency_lite"]

import astropy.units as u
import numbers
import numpy as np

from astropy.constants.si import e, eps0
from numba import njit

from plasmapy import particles
from plasmapy.formulary import misc
from plasmapy.particles import ParticleLike
from plasmapy.particles.exceptions import ChargeError, InvalidParticleError
from plasmapy.utils.decorators import (
    angular_freq_to_hz,
    bind_lite_func,
    preserve_signature,
    validate_quantities,
)

__all__ += __aliases__ + __lite_funcs__

e_si_unitless = e.value
eps0_si_unitless = eps0.value


@validate_quantities(
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    }
)
@angular_freq_to_hz
def gyrofrequency(
    B: u.Quantity[u.T], particle: ParticleLike, signed=False, Z=None
) -> u.Quantity[u.rad / u.s]:
    r"""
    Calculate the particle gyrofrequency in units of radians per second.

    **Aliases:** `oc_`, `wc_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    particle : `~plasmapy.particles.particle_class.Particle`
        Representation of the particle species (e.g., 'p' for protons, 'D+'
        for deuterium, or 'He-4 +1' for singly ionized helium-4). If no
        charge state information is provided, then the particles are assumed
        to be singly charged.

    signed : `bool`, optional
        The gyrofrequency can be defined as signed (negative for electron,
        positive for ion). Default is `False` (unsigned, i.e. always
        positive).

    Z : `float` or `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) for a plasma where
        a macroscopic description is valid. If this quantity is not
        given then the charge number of the ion
        is used. This is effectively an average gyrofrequency for the
        plasma where multiple charge states are present, and should
        not be interpreted as the gyrofrequency for any single particle.
        If not provided, it defaults to the charge number of the ``particle``.

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
        be used to identify a particle or isotope.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The particle gyrofrequency is the angular frequency of particle
    gyration around magnetic field lines and is given by:

    .. math::
        ω_{c} = \frac{Z e B}{m}

    The particle gyrofrequency is also known as the particle cyclotron
    frequency or the particle Larmor frequency.

    The recommended way to convert from angular frequency to frequency
    is to use an equivalency between cycles per second and hertz, as
    Astropy's `~astropy.units.dimensionless_angles` equivalency does not
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
    m = particles.particle_mass(particle)
    Z = misc._grab_charge(particle, Z)
    if not signed:
        Z = abs(Z)

    return u.rad * (Z * e * np.abs(B) / m).to(1 / u.s)


oc_ = gyrofrequency
"""Alias to `~plasmapy.formulary.frequencies.gyrofrequency`."""

wc_ = gyrofrequency
"""Alias to `~plasmapy.formulary.frequencies.gyrofrequency`."""


@preserve_signature
@njit
def plasma_frequency_lite(
    n: numbers.Real, mass: numbers.Real, z_mean: numbers.Real, to_hz: bool = False
) -> numbers.Real:
    r"""
    The :term:`lite-function` for
    `~plasmapy.formulary.frequencies.plasma_frequency`.  Performs the
    same plasma frequency calculation as
    `~plasmapy.formulary.frequencies.plasma_frequency`, but is intended
    for computational use and, thus, has all data conditioning
    safeguards removed.

    Parameters
    ----------
    n : `~numbers.Real`
        Particle number density, in units of m\ :sup:`-3`.

    mass : `~numbers.Real`
        Mass of the particle, in units of kg.

    z_mean : `~numbers.Real`
        The average ionization (arithmetic mean) for the particle
        species in the plasma.  For example, a proton would have a value
        of ``z_mean=1``.

    to_hz : `bool`
        (Default `False`) Set `True` to apply the factor of
        :math:`1/2π` and return a value in units of Hz.

    Returns
    -------
    wp : `~numbers.Real`
        The particle plasma frequency in radians per second.  Setting
        keyword ``to_hz=True`` will apply the factor of :math:`1/2π`
        and yield a value in Hz.

    Notes
    -----

    The particle plasma frequency is

    .. math::
        ω_{p} = Z |e| \sqrt{\frac{n}{\epsilon_0 m}}

    where :math:`m` is the mass of the particle, :math:`e` is the
    fundamental unit of charge, :math:`Z` is the average charge state
    ``z_mean`` of the particle species, :math:`n` is the particle number
    density.  This form of the plasma frequency has units of
    radians / s, but when using the ``to_hz`` keyword a factor of
    :math:`1/2π` will be applied to give a value in Hz.

    Examples
    --------

    >>> from plasmapy.particles import Particle
    >>> mass = Particle("p").mass.value
    >>> plasma_frequency_lite(n=1e19, mass=mass, z_mean=1)
    416329...
    >>> plasma_frequency_lite(n=1e19, mass=mass, z_mean=1, to_hz=True)
    662608...
    """
    omega_p = z_mean * e_si_unitless * np.sqrt(n / (eps0_si_unitless * mass))

    return omega_p / (2.0 * np.pi) if to_hz else omega_p


@bind_lite_func(plasma_frequency_lite)
@validate_quantities(
    n={"can_be_negative": False},
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    },
)
@angular_freq_to_hz
def plasma_frequency(
    n: u.Quantity[u.m**-3], particle: ParticleLike, z_mean=None
) -> u.Quantity[u.rad / u.s]:
    r"""Calculate the particle plasma frequency.

    **Aliases:** `wp_`

    **Lite Version:** `~plasmapy.formulary.frequencies.plasma_frequency_lite`

    Parameters
    ----------
    n : `~astropy.units.Quantity`
        Particle number density in units convertible to m\ :sup:`-3`.

    particle : `~plasmapy.particles.particle_class.Particle`
        Representation of the particle species (e.g., ``"p"`` for
        protons, ``"D+"`` for deuterium, or ``"He-4 +1"`` for singly
        ionized helium-4). If no charge state information is provided,
        then the particles are assumed to be singly charged.

    z_mean : `~numbers.Real`, optional
        The average ionization (arithmetic mean) for the particle
        species in the plasma.  Typically, the charge state will be
        derived from the ``particle`` argument, but this keyword will
        override that behavior.

    Returns
    -------
    omega_p : `~astropy.units.Quantity`
        The particle plasma frequency in radians per second.  Setting
        keyword ``to_hz=True`` will apply the factor of :math:`1/2π`
        and yield a value in Hz.

    Raises
    ------
    `TypeError`
        If ``n`` is not a `~astropy.units.Quantity` or particle is not
        of an appropriate type.

    `~astropy.units.UnitConversionError`
        If ``n`` is not in correct units.

    `ValueError`
        If ``n`` contains invalid values or particle cannot be used to
        identify a particle or isotope.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The particle plasma frequency is

    .. math::
        ω_{p} = Z |e| \sqrt{\frac{n}{\epsilon_0 m}}

    where :math:`m` is the mass of the particle, :math:`e` is the
    fundamental unit of charge, :math:`Z` is the average charge state
    ``z_mean`` of the particle species, :math:`n` is the particle number
    density.  This form of the plasma frequency has units of
    radians / s, but using the ``to_hz`` will apply the factor of
    :math:`1/2π` to give a value in Hz.

    Examples
    --------
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

    For user convenience
    `~plasmapy.formulary.frequencies.plasma_frequency_lite` is bound to
    this function and can be used as follows.

    >>> from plasmapy.particles import Particle
    >>> mass = Particle("p").mass.value
    >>> plasma_frequency.lite(n=1e19, mass=mass, z_mean=1)
    416329...
    >>> plasma_frequency.lite(n=1e19, mass=mass, z_mean=1, to_hz=True)
    662608...
    """

    try:
        m = particles.particle_mass(particle).value

        if z_mean is None:
            # warnings.warn("No z_mean given, defaulting to atomic charge",
            #               PhysicsWarning)
            try:
                Z = particles.charge_number(particle)
            except ChargeError:
                Z = 1
        else:
            # using user provided average ionization
            Z = z_mean
        Z = np.abs(Z)
        # TODO REPLACE WITH Z = np.abs(_grab_charge(particle, z_mean)), some bugs atm
    except InvalidParticleError as e:
        raise ValueError(f"Invalid particle, {particle}, in plasma_frequency.") from e

    return plasma_frequency_lite(n=n, mass=m, z_mean=Z) * u.rad / u.s


wp_ = plasma_frequency
"""Alias to `~plasmapy.formulary.frequencies.plasma_frequency`."""


@validate_quantities(
    n_i={"can_be_negative": False},
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    },
)
@angular_freq_to_hz
def lower_hybrid_frequency(
    B: u.Quantity[u.T], n_i: u.m**-3, ion: ParticleLike
) -> u.Quantity[u.rad / u.s]:
    r"""
    Return the lower hybrid frequency.

    **Aliases:** `wlh_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    n_i : `~astropy.units.Quantity`
        Ion number density.

    ion : `~plasmapy.particles.particle_class.Particle`
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

    The lower hybrid frequency constitutes a resonance for electromagnetic
    waves in magnetized plasmas, namely for the X-mode. These are waves
    with their wave electric field being perpendicular to the background
    magnetic field. For the lower hybrid frequency, ion and electron
    dynamics both play a role. As the name suggests, it has a lower frequency
    compared to the upper hybrid frequency. It can play an important role
    for heating and current drive in fusion plasmas.

    Examples
    --------
    >>> from astropy import units as u
    >>> lower_hybrid_frequency(0.2*u.T, n_i=5e19*u.m**-3, ion='D+')
    <Quantity 5.78372...e+08 rad / s>
    >>> lower_hybrid_frequency(0.2*u.T, n_i=5e19*u.m**-3, ion='D+', to_hz = True)
    <Quantity 92050879.3... Hz>

    """
    # We do not need a charge state here, so the sole intent is to
    # catch invalid ions.
    try:
        particles.charge_number(ion)
    except InvalidParticleError as ex:
        raise ValueError("Invalid ion in lower_hybrid_frequency.") from ex

    omega_ci = gyrofrequency(B, particle=ion)
    omega_pi = plasma_frequency(n_i, particle=ion)
    omega_ce = gyrofrequency(B, particle="e-")
    return ((omega_ci * omega_ce) ** -1 + omega_pi**-2) ** -0.5


wlh_ = lower_hybrid_frequency
"""Alias to `~plasmapy.formulary.frequencies.lower_hybrid_frequency`."""


@validate_quantities(
    n_e={"can_be_negative": False},
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    },
)
@angular_freq_to_hz
def upper_hybrid_frequency(
    B: u.Quantity[u.T], n_e: u.Quantity[u.m**-3]
) -> u.Quantity[u.rad / u.s]:
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

    The upper hybrid frequency is a resonance for electromagnetic
    waves in magnetized plasmas, namely for the X-mode. These are
    waves with their wave electric field being perpendicular to
    the background magnetic field. In the cold plasma model, i.e.
    without any finite temperature effects, the resonance acts
    merely as a resonance such that power can be deposited there.
    If finite temperature effects are considered, mode conversion
    can occur at the upper hybrid resonance, coupling to the
    electrostatic electron Bernstein wave.

    Examples
    --------
    >>> from astropy import units as u
    >>> upper_hybrid_frequency(0.2*u.T, n_e=5e19*u.m**-3)
    <Quantity 4.00459...e+11 rad / s>
    >>> upper_hybrid_frequency(0.2*u.T, n_e=5e19*u.m**-3, to_hz = True)
    <Quantity 6.37350...e+10 Hz>

    """
    omega_pe = plasma_frequency(n=n_e, particle="e-")
    omega_ce = gyrofrequency(B, "e-")
    return np.sqrt(omega_pe**2 + omega_ce**2)


wuh_ = upper_hybrid_frequency
"""Alias to `~plasmapy.formulary.frequencies.upper_hybrid_frequency`."""
