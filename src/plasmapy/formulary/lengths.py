"""Characteristic length scales in plasmas."""

__all__ = ["Debye_length", "gyroradius", "inertial_length"]
__aliases__ = ["cwp_", "lambdaD_", "rc_", "rhoc_"]

import warnings

import astropy.units as u
import numpy as np
from astropy.constants.si import c, e, eps0, k_B

from plasmapy.formulary import frequencies, speeds
from plasmapy.formulary.relativity import RelativisticBody
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.utils.decorators import validate_quantities

__all__ += __aliases__


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def Debye_length(T_e: u.Quantity[u.K], n_e: u.Quantity[u.m**-3]) -> u.Quantity[u.m]:
    r"""
    Calculate the exponential scale length for charge screening in an
    electron plasma with stationary ions.

    The Debye length is given by

    .. math::
        λ_D = \sqrt{\frac{ε_0 k_B T_e}{n_e q_e^2}},

    where :math:`n_e` is the electron number density, :math:`T_e` is the
    electron temperature, :math:`k_B` is the Boltzmann constant,
    :math:`q_e` is the elementary charge, and :math:`ε_0` is the vacuum
    permittivity.

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

    See Also
    --------
    ~plasmapy.formulary.dimensionless.Debye_number

    Notes
    -----
    Plasmas will generally be quasineutral on length scales
    significantly longer than the Debye length.

    The electrical potential will drop by a factor of
    :math:`∼\frac{1}{e}` every Debye length away from the vicinity of a
    charged particle.

    Examples
    --------
    >>> import astropy.units as u
    >>> Debye_length(5e6 * u.K, 5e15 * u.m**-3)
    <Quantity 0.002182... m>
    """
    return np.sqrt(eps0 * k_B * T_e / (n_e * e**2))


lambdaD_ = Debye_length
"""Alias to `~plasmapy.formulary.lengths.Debye_length`."""


@validate_quantities(
    Vperp={"can_be_nan": True},  # none_shall_pass
    T={
        "can_be_nan": True,
        "equivalencies": u.temperature_energy(),
        "none_shall_pass": True,
    },
    validations_on_return={"equivalencies": u.dimensionless_angles()},
)
@particle_input(any_of={"charged", "uncharged"})
def gyroradius(
    B: u.Quantity[u.T],
    particle: ParticleLike,
    *,
    Vperp: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    T: u.Quantity[u.K] = None,
    lorentzfactor=np.nan,
    relativistic: bool = True,
    mass_numb: int | None = None,
    Z: float | None = None,
) -> u.Quantity[u.m]:
    r"""
    Calculate the radius of circular motion for a charged particle in a
    uniform magnetic field (including relativistic effects by default).

    **Aliases:** `rc_`, `rhoc_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    particle : |particle-like|
        Representation of the particle species (e.g., ``"p+"`` for
        protons, ``"D+"`` for a deuteron, or ``"He-4 1+"`` for singly
        ionized helium-4).

    Vperp : `~astropy.units.Quantity`, |keyword-only|, optional
        The component of particle velocity that is perpendicular to
        the magnetic field in units convertible to meters per second.

    T : `~astropy.units.Quantity`, |keyword-only|, optional
        The particle temperature in units convertible to kelvin or
        electron-volts. If provided, the perpendicular velocity gets set
        to the most probable *non-relativistic* thermal velocity for
        that particle at this temperature. Cannot be provided if
        ``Vperp`` is provided.

    lorentzfactor : `float` or `~numpy.ndarray`, |keyword-only|, optional
        The :wikipedia:`Lorentz factor` of the particle corresponding to
        the direction perpendicular to the magnetic field. Cannot be
        provided if ``Vperp`` or ``T`` is provided.

    relativistic : `bool`, |keyword-only|, default: `True`
        If `True`, the relativistic formula for gyroradius will be used.
        If `False`, the non-relativistic formula will be used.

    Returns
    -------
    r_L : `~astropy.units.Quantity`
        The particle gyroradius in units of meters. This
        `~astropy.units.Quantity` will be based on either the
        perpendicular component of particle velocity as inputted, or
        the most probable speed for a particle within a Maxwellian
        distribution for the particle temperature.

    Other Parameters
    ----------------
    mass_numb : integer, |keyword-only|, optional
        The mass number, if not provided in ``particle``.

    Z : real number, |keyword-only|, optional
        The |charge number|, if not provided in ``particle``.

    Raises
    ------
    `~astropy.units.UnitConversionError`
        If a |Quantity| argument has units of an incorrect physical
        type.

    `ValueError`
        If any argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        Issued if any of ``B``, ``Vperp``, or ``T`` do not have units,
        in which case SI units will be assumed.

    Warnings
    --------
    The Lorentz factor can be inferred from ``Vperp`` or ``T`` but near
    the speed of light, this can lead to rounding errors. For very high
    values of the Lorentz factor, its use should be preferred.

    Notes
    -----
    The relativistic :wikipedia:`gyroradius` for a particle of species
    :math:`s` is given by

    .. math::

        r_{L,s} = \frac{γ m_s V_{⟂,s}}{ |q_s| B}

    where :math:`V_⟂` is the component of particle velocity that is
    perpendicular to the magnetic field, :math:`m_s` is the particle
    mass, :math:`q_s` is the particle charge, :math:`B` is the magnetic
    field magnitude, and :math:`γ` is the :wikipedia:`Lorentz factor`.

    In the non-relativistic limit, the gyroradius reduces to

    .. math::

        r_{Ls} = \frac{V_{⟂,s}}{ω_{c,s}}

    where :math:`ω_{c,s}` is the particle gyrofrequency. To turn off
    relativistic effects, set the ``relativistic`` keyword to `False`.

    The gyroradius is sometimes called the Larmor radius, cyclotron
    radius, or radius of gyration.

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.formulary import gyroradius
    >>> from astropy.constants import c

    Let's estimate the proton gyroradius in the solar corona.

    >>> gyroradius(B=0.2 * u.T, particle="p+", T=1e6 * u.K)
    <Quantity 0.0067... m>

    Let's estimate the gyroradius of a deuteron and a triton in ITER by
    providing the characteristic thermal energy per particle,
    :math:`k_B T`, to ``T``.

    >>> gyroradius(B=5 * u.T, particle=["D+", "T+"], T=13 * u.keV)
    <Quantity [0.00465..., 0.00570...] m>

    Relativistic effects are included by default, but can be turned
    off using the ``relativistic`` parameter. Let's use this in the
    calculation of the gyroradius of a cosmic ray in the interstellar
    medium (ISM). We will provide the magnetic field in units of
    microgauss (μG).

    >>> gyroradius(B=10 * u.uG, particle="p+", Vperp=0.99 * c)
    <Quantity 2.19642688e+10 m>
    >>> gyroradius(B=10 * u.uG, particle="p+", Vperp=0.99 * c, relativistic=False)
    <Quantity 3.09844141e+09 m>

    Let's calculate the gyroradius of a much higher energy cosmic ray
    in the ISM using ``lorentzfactor``.

    >>> gyroradius(B=10 * u.uG, particle="p+", lorentzfactor=3e6).to("pc")
    <Quantity 0.30428378 pc>
    """

    if T is None:
        T = np.nan * u.K

    # Determine output shape and broadcast inputs accordingly
    target_shape = np.broadcast(T, Vperp, lorentzfactor, particle).shape  # type: ignore[arg-type]
    lorentzfactor_in = lorentzfactor
    lorentzfactor = np.array(np.broadcast_to(lorentzfactor, target_shape))
    Vperp = np.array(np.broadcast_to(Vperp, target_shape, subok=True), subok=True)
    T = np.array(np.broadcast_to(T, target_shape, subok=True), subok=True)

    isfinite_T = np.isfinite(T)
    isfinite_Vperp = np.isfinite(Vperp)
    isfinite_lorentzfactor = np.isfinite(lorentzfactor)

    # Check if lorentzfactor is given despite relativistic being false
    if not relativistic and not np.isnan(lorentzfactor):
        raise ValueError("Lorentz factor is provided but relativistic is set to false")

    # Check if V and T are both given at the same time at any position
    if np.any(isfinite_Vperp & isfinite_T):
        raise ValueError(
            "Must give Vperp or T, but not both, as arguments to gyroradius"
        )

    # Check if V or T and lorentzfactor are both given at the same time at any position
    if np.any(isfinite_lorentzfactor & (isfinite_Vperp | isfinite_T)):
        warnings.warn(
            "lorentzfactor is given along with Vperp or T, will lead "
            "to inaccurate predictions unless they correspond"
        )

    # Check if V and T are both missing at any position but lorentzfactor is not scalar
    if np.any(~isfinite_Vperp & ~isfinite_T) and not np.isscalar(lorentzfactor_in):
        raise ValueError(
            "Inferring velocity(s) from more than one Lorentz "
            "factor is not currently supported"
        )

    # In the positions where Vperp is missing but T is given, calculate the thermal speed
    if np.any(isfinite_T):
        Vperp[isfinite_T] = speeds.thermal_speed(T[isfinite_T], particle=particle)
    isfinite_Vperp = np.isfinite(Vperp)

    # In the positions where Vperp is still missing, calculate Vperp from lorentzfactor
    if np.any(~isfinite_Vperp):
        Vperp[~isfinite_Vperp] = RelativisticBody(
            particle, lorentz_factor=lorentzfactor_in
        ).velocity

    # Calculate the final lorentzfactor
    if relativistic:
        # Fill in missing entries of lorentzfactor by calculating it using Vperp
        lorentzfactor[~isfinite_lorentzfactor] = RelativisticBody(
            particle, V=Vperp
        ).lorentz_factor[~isfinite_lorentzfactor]
    else:
        lorentzfactor = 1.0

    # Calculate gyrofrequency and finally the gyroradius
    omega_ci = frequencies.gyrofrequency(B, particle)
    return lorentzfactor * np.abs(Vperp) / omega_ci


rc_ = gyroradius
"""Alias to `~plasmapy.formulary.lengths.gyroradius`."""

rhoc_ = gyroradius
"""Alias to `~plasmapy.formulary.lengths.gyroradius`."""


@validate_quantities(
    n={"can_be_negative": False},
    validations_on_return={"equivalencies": u.dimensionless_angles()},
)
@particle_input(require="charged")
def inertial_length(
    n: u.Quantity[u.m**-3],
    particle: ParticleLike,
    *,
    mass_numb: int | None = None,
    Z: float | None = None,
) -> u.Quantity[u.m]:
    r"""
    Calculate a charged particle's inertial length.

    The inertial length of a particle of species :math:`s` is given by

    .. math::

        d = \frac{c}{ω_{ps}}

    The inertial length is the characteristic length scale for a
    particle to be accelerated in a plasma. The Hall effect becomes
    important on length scales shorter than the ion inertial length.

    **Aliases:** `cwp_`

    Parameters
    ----------
    n : `~astropy.units.Quantity`
        Particle number density in units convertible to m\ :sup:`-3`\ .

    particle : |particle-like|
        Representation of the particle species (e.g., 'p+' for protons,
        'D+' for deuterium, or 'He-4 +1' for singly ionized helium-4).

    Returns
    -------
    d : `~astropy.units.Quantity`
        The particle's inertial length in meters.

    Other Parameters
    ----------------
    mass_numb : integer, |keyword-only|, optional
        The mass number, if not provided in ``particle``.

    Z : real number, |keyword-only|, optional
        The |charge number|, if not provided in ``particle``.

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
    The inertial length is also known as the skin depth.

    Examples
    --------
    >>> import astropy.units as u
    >>> inertial_length(5 * u.m**-3, "He+")
    <Quantity 2.02985...e+08 m>
    >>> inertial_length(5 * u.m**-3, "e-")
    <Quantity 2376534.75... m>
    """
    omega_p = frequencies.plasma_frequency(n, particle=particle)
    return c / omega_p


cwp_ = inertial_length
"""
Alias to `~plasmapy.formulary.lengths.inertial_length`.

* Name is shorthand for :math:`c / ω_p`.
"""
