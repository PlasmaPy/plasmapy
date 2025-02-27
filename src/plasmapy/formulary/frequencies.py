"""Functions to calculate fundamental plasma frequency parameters."""

__all__ = [
    "gyrofrequency",
    "lower_hybrid_frequency",
    "plasma_frequency",
    "upper_hybrid_frequency",
    "Buchsbaum_frequency",
]
__aliases__ = ["oc_", "wc_", "wlh_", "wp_", "wuh_"]
__lite_funcs__ = ["plasma_frequency_lite"]


import astropy.units as u
import numpy as np
from astropy.constants.si import e, eps0

from plasmapy import particles
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.exceptions import InvalidParticleError
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.utils.decorators import (
    angular_freq_to_hz,
    bind_lite_func,
    preserve_signature,
    validate_quantities,
)

__all__ += __aliases__ + __lite_funcs__

e_si_unitless = e.value
eps0_si_unitless = eps0.value


@particle_input(any_of={"charged", "uncharged"})
@validate_quantities(
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    }
)
@angular_freq_to_hz
def gyrofrequency(
    B: u.Quantity[u.T],
    particle: ParticleLike,
    signed: bool = False,
    Z: float | None = None,
    mass_numb: int | None = None,
) -> u.Quantity[u.rad / u.s]:
    r"""
    Calculate the particle gyrofrequency in units of radians per second.

    **Aliases:** `oc_`, `wc_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    particle : |particle-like|
        Representation of the particle species (e.g., ``'p+'`` for
        protons, ``'D+'`` for deuterium, or ``'He-4 1+'`` for singly
        ionized helium-4).

    signed : `bool`, default: `False`
        The gyrofrequency can be defined as signed, where its sign is
        the sign of the |charge number|. Defaults to unsigned (i.e.,
        always positive).

    Z : real number, optional
        The |charge number| of an ion or neutral atom, if not provided
        in ``particle``.

    mass_numb : integer, optional
        The mass number of an isotope, if not provided in ``particle``.

    Returns
    -------
    `~astropy.units.Quantity`
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
    `~astropy.units.UnitsWarning`
        If units are not provided, and SI units are assumed.

    Notes
    -----
    The particle gyrofrequency is the angular frequency of particle
    gyration around magnetic field lines and is given by:

    .. math::
        ω_c = \frac{|Z| e B}{m}

    If ``signed`` is `True`, then :math:`|Z|` is replaced with
    :math:`Z`. A particle's gyrofrequency is also known as its
    *cyclotron frequency* or *Larmor frequency*.

    The recommended way to convert from angular frequency to frequency
    is to use an equivalency between cycles per second and hertz, as
    Astropy's `~astropy.units.dimensionless_angles` equivalency does not
    account for the factor of :math:`2π` needed during this conversion.
    The `~astropy.units.dimensionless_angles` equivalency is appropriate
    when dividing a velocity by an angular frequency to get a length
    scale.

    Examples
    --------
    >>> import astropy.units as u
    >>> gyrofrequency(0.1 * u.T, "e-")
    <Quantity 1.7588...e+10 rad / s>
    >>> gyrofrequency(0.1 * u.T, "e-", to_hz=True)
    <Quantity 2.79924...e+09 Hz>
    >>> gyrofrequency(0.1 * u.T, "e-", signed=True)
    <Quantity -1.75882...e+10 rad / s>
    >>> gyrofrequency(0.01 * u.T, "p+")
    <Quantity 957883.32... rad / s>
    >>> gyrofrequency(0.01 * u.T, "p+", signed=True)
    <Quantity 957883.32... rad / s>
    >>> gyrofrequency(0.01 * u.T, particle="T+")
    <Quantity 319964.5... rad / s>
    >>> gyrofrequency(0.01 * u.T, particle="T+", to_hz=True)
    <Quantity 50923.9... Hz>
    >>> gyrofrequency(250 * u.G, particle="Fe", mass_numb=56, Z=13)
    <Quantity 560682.34875287 rad / s>
    >>> omega_ce = gyrofrequency(0.1 * u.T, "e-")
    >>> print(omega_ce)
    1758820... rad / s
    >>> f_ce = omega_ce.to(u.Hz, equivalencies=[(u.cy / u.s, u.Hz)])
    >>> print(f_ce)
    279924... Hz
    """
    q = particle.charge if signed else abs(particle.charge)
    return u.rad * (q * np.abs(B) / particle.mass).to(1 / u.s)


oc_ = gyrofrequency
"""Alias to `~plasmapy.formulary.frequencies.gyrofrequency`."""

wc_ = gyrofrequency
"""Alias to `~plasmapy.formulary.frequencies.gyrofrequency`."""


# NEED TO HANDLE DEPRECATION OF Z_MEAN IN ω_p LITE FUNCTION!


@preserve_signature
def plasma_frequency_lite(
    n: float,
    mass: float,
    Z: float,
    to_hz: bool = False,
) -> float:
    r"""
    The :term:`lite-function` for
    `~plasmapy.formulary.frequencies.plasma_frequency`. Performs the
    same plasma frequency calculation as
    `~plasmapy.formulary.frequencies.plasma_frequency`, but is intended
    for computational use and, thus, has all data conditioning
    safeguards removed.

    Parameters
    ----------
    n : `float`
        Particle number density, in units of m\ :sup:`-3`.

    mass : `float`
        Mass of the particle, in units of kg.

    Z : `float`
        The average ionization (arithmetic mean) for the particle
        species in the plasma. For example, a proton would have a value
        of ``Z=1``.

    to_hz : `bool`, default: `False`.
        Set `True` to apply the factor of :math:`1/2π` and return a
        value in units of Hz.

    Returns
    -------
    wp : `float`
        The particle plasma frequency in radians per second. Setting
        keyword ``to_hz=True`` will apply the factor of :math:`1/2π`
        and yield a value in Hz.

    Notes
    -----
    The particle plasma frequency is

    .. math::
        ω_p = \sqrt{\frac{n |q|}{ε_0 m}}

    where :math:`n` is the number density, :math:`q` is the particle
    charge, and :math:`m` is the particle mass.

    This form of the plasma frequency has units of rad/s, but when using
    the ``to_hz`` keyword a factor of :math:`1/2π` will be applied to
    give a value in Hz.

    Examples
    --------
    >>> from plasmapy.particles import Particle
    >>> mass = Particle("p+").mass.value
    >>> plasma_frequency_lite(n=1e19, mass=mass, Z=1)
    np.float64(4163294534.0...)
    >>> plasma_frequency_lite(n=1e19, mass=mass, Z=1, to_hz=True)
    np.float64(662608904.6...)
    """
    omega_p = Z * e_si_unitless * np.sqrt(n / (eps0_si_unitless * mass))

    return omega_p / (2.0 * np.pi) if to_hz else omega_p


@bind_lite_func(plasma_frequency_lite)
@particle_input(any_of={"charged", "uncharged"})
@validate_quantities(
    n={"can_be_negative": False},
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    },
)
@angular_freq_to_hz
def plasma_frequency(
    n: u.Quantity[u.m**-3],
    particle: ParticleLike,
    *,
    mass_numb: int | None = None,
    Z: float | None = None,
) -> u.Quantity[u.rad / u.s]:
    r"""Calculate the particle plasma frequency.

    **Aliases:** `wp_`

    **Lite Version:** `~plasmapy.formulary.frequencies.plasma_frequency_lite`

    Parameters
    ----------
    n : `~astropy.units.Quantity`
        Particle number density in units convertible to m\ :sup:`-3`.

    particle : |particle-like|
        Representation of the particle species (e.g., ``"p+"`` for
        protons, ``"D+"`` for deuterium, or ``"He-4 1+"`` for singly
        ionized helium-4). If no charge state information is provided,
        then the particles are assumed to be singly charged.

    Z : real number, optional
        The |charge number| of an ion or neutral atom, if not provided
        in ``particle``.

    mass_numb : integer, optional
        The mass number of an isotope, if not provided in ``particle``.

    Returns
    -------
    `~astropy.units.Quantity`
        The particle plasma frequency in radians per second. Setting
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
    `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The particle plasma frequency is

    .. math::
        ω_p = \sqrt{\frac{n |q|}{ε_0 m}}

    where :math:`n` is the number density, :math:`q` is the particle
    charge, and :math:`m` is the particle mass.

    This form of the plasma frequency has units of rad/s, but using the
    ``to_hz`` keyword argument will apply the factor of :math:`1/2π` to
    give the frequency in Hz.

    Examples
    --------
    >>> import astropy.units as u
    >>> plasma_frequency(1e19 * u.m**-3, particle="p+")
    <Quantity 4.16329...e+09 rad / s>
    >>> plasma_frequency(1e19 * u.m**-3, particle="p+", to_hz=True)
    <Quantity 6.62608...e+08 Hz>
    >>> plasma_frequency(1e19 * u.m**-3, particle="D+")
    <Quantity 2.94462...e+09 rad / s>
    >>> plasma_frequency(1e19 * u.m**-3, "e-")
    <Quantity 1.78398...e+11 rad / s>
    >>> plasma_frequency(1e19 * u.m**-3, "e-", to_hz=True)
    <Quantity 2.83930...e+10 Hz>

    For user convenience
    `~plasmapy.formulary.frequencies.plasma_frequency_lite` is bound to
    this function and can be used as follows.

    >>> from plasmapy.particles import Particle
    >>> mass = Particle("p+").mass.value
    >>> plasma_frequency.lite(n=1e19, mass=mass, Z=1)
    np.float64(4163294534.0...)
    >>> plasma_frequency.lite(n=1e19, mass=mass, Z=1, to_hz=True)
    np.float64(662608904.6...)
    """
    return (
        plasma_frequency_lite(
            n=n.value,
            mass=particle.mass.value,
            Z=np.abs(particle.charge_number),
        )
        * u.rad
        / u.s
    )


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
    B: u.Quantity[u.T], n_i: u.Quantity[u.m**-3], ion: ParticleLike
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
        Representation of the ion species (e.g., ``'p+'`` for protons, ``'D+'``
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
    `~astropy.units.UnitsWarning`
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
    >>> import astropy.units as u
    >>> lower_hybrid_frequency(0.2 * u.T, n_i=5e19 * u.m**-3, ion="D+")
    <Quantity 5.78372...e+08 rad / s>
    >>> lower_hybrid_frequency(0.2 * u.T, n_i=5e19 * u.m**-3, ion="D+", to_hz=True)
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
    `~astropy.units.UnitsWarning`
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
    >>> import astropy.units as u
    >>> upper_hybrid_frequency(0.2 * u.T, n_e=5e19 * u.m**-3)
    <Quantity 4.00459...e+11 rad / s>
    >>> upper_hybrid_frequency(0.2 * u.T, n_e=5e19 * u.m**-3, to_hz=True)
    <Quantity 6.37350...e+10 Hz>
    """
    omega_pe = plasma_frequency(n=n_e, particle="e-")
    omega_ce = gyrofrequency(B, "e-")
    return np.sqrt(omega_pe**2 + omega_ce**2)


wuh_ = upper_hybrid_frequency
"""Alias to `~plasmapy.formulary.frequencies.upper_hybrid_frequency`."""


@validate_quantities(
    validations_on_return={
        "units": [u.rad / u.s, u.Hz],
        "equivalencies": [(u.cy / u.s, u.Hz)],
    }
)
@angular_freq_to_hz
def Buchsbaum_frequency(
    B: u.Quantity[u.T],
    n1: u.Quantity[u.m**-3],
    n2: u.Quantity[u.m**-3],
    ion1: ParticleLike,
    ion2: ParticleLike,
    Z1: float | None = None,
    Z2: float | None = None,
) -> u.Quantity[u.rad / u.s]:
    r"""
    Return the Buchsbaum frequency for a two-ion-species plasma.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    n1 : `~astropy.units.Quantity`
        Particle number density of ion species #1 in units convertible to m\ :sup:`-3`.

    n2 : `~astropy.units.Quantity`
        Particle number density of ion species #2 in units convertible to m\ :sup:`-3`.

    ion1 : `~plasmapy.particles.particle_class.Particle`
        Representation of ion species #1 (e.g., 'p+' for protons, 'D+'
        for deuterium, or 'He-4 +1' for singly ionized helium-4). If no
        charge state information is provided, then species #1 is assumed
        to be singly charged.

    ion2 : `~plasmapy.particles.particle_class.Particle`
        Representation of ion species #2 (same behavior as for ion1).

    Z1 : `float` or `~astropy.units.Quantity`, optional
        The charge state for ion species #1. If not provided, it
        defaults to the charge number of ``ion1``.

    Z2 : `float` or `~astropy.units.Quantity`, optional
        The charge state for ion species #2. If not provided, it
        defaults to the charge number of ``ion2``.

    Returns
    -------
    omega_BB : `~astropy.units.Quantity`
        The Buchsbaum frequency of the plasma in units of radians per second.
        Setting keyword ``to_hz=True`` will apply the factor of :math:`1/2π`
        and yield a value in Hz.

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
    `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    In a magnetized plasma, the presence of two ion species allows the
    perpendicular component of the cold-plasma dielectric coefficient
    :math:`ε_⟂` to vanish at an angular frequency referred
    to as the Buchsbaum frequency :cite:p:`buchsbaum:1960`, also called
    the bi-ion hybrid resonance frequency :cite:p:`thompson:1995`, or
    ion-ion hybrid frequency :cite:p:`vincena:2013`. This frequency
    can be defined as

    .. math::

        ω_{BB} ≡ \sqrt{\frac{ω_{p1}^2 ω_{c2}^2
        + ω_{p2}^2 ω_{c1}^2}{ω_{p2}^2 + ω_{p2}^2}}

    Examples
    --------
    >>> import astropy.units as u
    >>> fbb = Buchsbaum_frequency(
    ...     0.1 * u.T, 1e18 * u.m**-3, 1e18 * u.m**-3, "proton", "He+", to_hz=True
    ... )
    >>> fbb
    <Quantity 764831.28372462 Hz>
    >>> fc_helium = gyrofrequency(0.1 * u.T, "He+", to_hz=True)
    >>> fc_proton = gyrofrequency(0.1 * u.T, "proton", to_hz=True)
    >>> fbb / fc_helium
    <Quantity 1.99327444>
    >>> fbb / fc_proton
    <Quantity 0.50168706>
    """
    omega_c1_squared = gyrofrequency(B, ion1, signed=False, Z=Z1) ** 2
    omega_c2_squared = gyrofrequency(B, ion2, signed=False, Z=Z2) ** 2
    omega_p1_squared = plasma_frequency(n1, ion1, Z=Z1) ** 2
    omega_p2_squared = plasma_frequency(n2, ion2, Z=Z2) ** 2

    return np.sqrt(
        (omega_p1_squared * omega_c2_squared + omega_p2_squared * omega_c1_squared)
        / (omega_p1_squared + omega_p2_squared)
    )
