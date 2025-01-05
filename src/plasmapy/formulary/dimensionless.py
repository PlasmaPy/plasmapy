"""
Dimensionless plasma parameters.

Dimensionless plasma parameters help us determine what regime a plasma
is in. For example, plasma :math:`β` is the ratio of thermal pressure to
magnetic pressure in a plasma. When :math:`β ≫ 1`, thermal pressure is
more important to dynamical behavior than magnetic pressure.  When
:math:`β ≪ 1`, magnetic pressure becomes more important than thermal
pressure.
"""

__all__ = [
    "beta",
    "Debye_number",
    "Hall_parameter",
    "Mag_Reynolds",
    "quantum_theta",
    "Reynolds_number",
    "Lundquist_number",
]
__aliases__ = ["betaH_", "nD_", "Re_", "Rm_"]


import astropy.units as u
import numpy as np
from astropy.constants.si import mu0

from plasmapy.formulary import frequencies, lengths, misc, speeds
from plasmapy.formulary.quantum import quantum_theta
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.utils.decorators import validate_quantities

__all__ += __aliases__


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def Debye_number(
    T_e: u.Quantity[u.K], n_e: u.Quantity[u.m**-3]
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""Return the number of electrons within a sphere with a radius
    of the Debye length.

    **Aliases:** `nD_`

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        Electron temperature.

    n_e : `~astropy.units.Quantity`
        Electron number density.

    Returns
    -------
    N_D : `~astropy.units.Quantity`
        Number of electrons within a sphere with a radius of the Debye
        length.

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

    See Also
    --------
    ~plasmapy.formulary.lengths.Debye_length

    Notes
    -----
    The Debye number is the number of electrons contained within a
    sphere with a radius of a Debye length and is given by

    .. math::
        N_D = \frac{4π}{3} n_e λ_D^3

    The Debye number is also known as the plasma parameter.

    Collective behavior requires :math:`N_D ≫ 1`\ .

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.constants.si import m_p, m_e
    >>> Debye_number(5e6 * u.K, 5e9 * u.cm**-3)
    <Quantity 2.17658...e+08>
    """

    lambda_D = lengths.Debye_length(T_e, n_e)
    return (4 / 3) * np.pi * n_e * lambda_D**3


nD_ = Debye_number
"""Alias to `~plasmapy.formulary.dimensionless.Debye_number`."""


@validate_quantities(
    n={"can_be_negative": False},
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
@particle_input
def Hall_parameter(
    n: u.Quantity[u.m**-3],
    T: u.Quantity[u.K],
    B: u.Quantity[u.T],
    ion: ParticleLike,
    particle: ParticleLike,
    coulomb_log: float | None = None,
    V: u.Quantity[u.m / u.s] | None = None,
    coulomb_log_method: str = "classical",
):
    r"""
    Calculate the ``particle`` Hall parameter for a plasma.

    The Hall parameter for plasma species :math:`s` (``particle``) is
    given by:

    .. math::

        β_{s} = \frac{Ω_{c s}}{ν_{s s^{\prime}}}

    where :math:`Ω_{c s}` is the gyrofrequncy for plasma species
    :math:`s` (``particle``) and :math:`ν_{s s^{\prime}}` is the
    collision frequency between plasma species :math:`s` (``particle``)
    and species :math:`s^{\prime}` (``ion``).

    **Aliases:** `betaH_`

    Parameters
    ----------
    n : `~astropy.units.quantity.Quantity`
        The number density associated with ``particle``.

    T : `~astropy.units.quantity.Quantity`
        The temperature of associated with ``particle``.

    B : `~astropy.units.quantity.Quantity`
        The magnetic field.

    ion : `~plasmapy.particles.particle_class.Particle`
        The type of ion ``particle`` is colliding with.

    particle : `~plasmapy.particles.particle_class.Particle`
        The particle species for which the Hall parameter is calculated
        for. Representation of the particle species (e.g., ``'p+'`` for
        protons, ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly
        ionized helium-4). If no charge state information is provided,
        then the particles are assumed to be singly charged.

    coulomb_log : `float`, optional
        Preset value for the Coulomb logarithm. Used mostly for testing
        purposes.

    V : `~astropy.units.quantity.Quantity`
        The relative velocity between ``particle`` and ``ion``. If not
        provided, then the ``particle`` thermal velocity is assumed
        (`~plasmapy.formulary.speeds.thermal_speed`).

    coulomb_log_method : `str`, optional
        The method by which to compute the Coulomb logarithm.
        The default method is the classical straight-line
        Landau-Spitzer method (``"classical"`` or ``"ls"``). The other
        6 supported methods are ``"ls_min_interp"``,
        ``"ls_full_interp"``, ``"ls_clamp_mininterp"``,
        ``"hls_min_interp"``, ``"hls_max_interp"``, and
        ``"hls_full_interp"``. Please refer to the docstring of
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for
        more information about these methods.

    Returns
    -------
    `~astropy.units.quantity.Quantity`
        Hall parameter for ``particle``.

    See Also
    --------
    ~plasmapy.formulary.frequencies.gyrofrequency
    ~plasmapy.formulary.collisions.frequencies.fundamental_electron_collision_freq
    ~plasmapy.formulary.collisions.frequencies.fundamental_ion_collision_freq
    ~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm

    Notes
    -----
    * For calculating the collision frequency
      `~plasmapy.formulary.collisions.frequencies.fundamental_electron_collision_freq`
      is used when ``particle`` is an electron and
      `~plasmapy.formulary.collisions.frequencies.fundamental_ion_collision_freq`
      when ``particle`` is an ion.
    * The collision frequencies are calculated assuming a slowly moving
      Maxwellian distribution.

    Examples
    --------
    >>> import astropy.units as u
    >>> Hall_parameter(1e10 * u.m**-3, 2.8e2 * u.eV, 2.3 * u.T, "He-4 +1", "e-")
    <Quantity 2.500...e+15>
    """
    from plasmapy.formulary.collisions import (
        fundamental_electron_collision_freq,
        fundamental_ion_collision_freq,
    )

    gyro_frequency = frequencies.gyrofrequency(B, particle)
    gyro_frequency = gyro_frequency / u.radian
    if particle == "e-":
        coll_rate = fundamental_electron_collision_freq(
            T, n, ion, coulomb_log, V, coulomb_log_method=coulomb_log_method
        )
    else:
        coll_rate = fundamental_ion_collision_freq(T, n, ion, coulomb_log, V)
    return gyro_frequency / coll_rate


betaH_ = Hall_parameter
"""Alias to `~plasmapy.formulary.dimensionless.Hall_parameter`."""


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n={"can_be_negative": False},
)
def beta(
    T: u.Quantity[u.K], n: u.Quantity[u.m**-3], B: u.Quantity[u.T]
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Compute the ratio of thermal pressure to magnetic pressure.

    The beta (:math:`β`) of a plasma is defined by

    .. math::
        β = \frac{p_{th}}{p_{mag}} = \frac{n k_B T}{B^2/2μ_0}

    where :math:`p_{th}` is the `~plasmapy.formulary.misc.thermal_pressure`
    of the plasma and :math:`p_{mag}` is the
    `~plasmapy.formulary.misc.magnetic_pressure` of the plasma.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The temperature of the plasma.

    n : `~astropy.units.Quantity`
        The particle density of the plasma.

    B : `~astropy.units.Quantity`
        The magnetic field in the plasma.

    Returns
    -------
    beta: `~astropy.units.Quantity`
        Dimensionless quantity.

    See Also
    --------
    ~plasmapy.formulary.misc.thermal_pressure
    ~plasmapy.formulary.misc.magnetic_pressure

    Examples
    --------
    >>> import astropy.units as u
    >>> beta(1 * u.eV, 1e20 * u.m**-3, 1 * u.T)
    <Quantity 4.0267...e-05>
    >>> beta(8.8e3 * u.eV, 1e20 * u.m**-3, 5.3 * u.T)
    <Quantity 0.01261...>
    """
    thermal_pressure = misc.thermal_pressure(T, n)
    magnetic_pressure = misc.magnetic_pressure(B)
    return thermal_pressure / magnetic_pressure


@validate_quantities(U={"can_be_negative": True})
def Reynolds_number(
    rho: u.Quantity[u.kg / u.m**3],
    U: u.Quantity[u.m / u.s],
    L: u.Quantity[u.m],
    mu: u.Quantity[u.kg / (u.m * u.s)],
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Compute the Reynolds number.

    The Reynolds number is a dimensionless quantity that is used to
    predict flow patterns in fluids. The Reynolds number is defined as
    the ratio of inertial forces to viscous forces. A low Reynolds
    number describes smooth, laminar flow while a high Reynolds number
    describes rough, turbulent flow.

    .. math::

        Re = \frac{ρ U L}{μ}

    **Aliases:** `Re_`

    Parameters
    ----------
    rho : `~astropy.units.Quantity`
        The density of the plasma.

    U : `~astropy.units.Quantity`
        The flow velocity of the plasma.

    L : `~astropy.units.Quantity`
        The characteristic length scale.

    mu : `~astropy.units.Quantity`
        The dynamic viscosity of the plasma.

    Returns
    -------
    Re: `~astropy.units.Quantity`
        Dimensionless quantity.

    Raises
    ------
    `TypeError`
        If ``U`` is not a `~astropy.units.Quantity` and cannot be
        converted into a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If ``U`` is not in appropriate units.

    :exc:`~plasmapy.utils.exceptions.RelativityError`
        If ``U`` is greater than the speed of light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Examples
    --------
    >>> import astropy.units as u
    >>> rho = 1000 * u.kg / u.m**3
    >>> U = 10 * u.m / u.s
    >>> L = 1 * u.m
    >>> mu = 8.9e-4 * u.kg / (u.m * u.s)
    >>> Reynolds_number(rho, U, L, mu)
    <Quantity 11235955.05617978>
    >>> rho = 1490 * u.kg / u.m**3
    >>> U = 0.1 * u.m / u.s
    >>> L = 0.05 * u.m
    >>> mu = 10 * u.kg / (u.m * u.s)
    >>> Reynolds_number(rho, U, L, mu)
    <Quantity 0.745>
    """
    return abs(rho * U * L / mu)


Re_ = Reynolds_number
"""Alias to `~plasmapy.formulary.dimensionless.Reynolds_number`."""


@validate_quantities(U={"can_be_negative": True})
def Mag_Reynolds(
    U: u.Quantity[u.m / u.s], L: u.Quantity[u.m], sigma: u.Quantity[u.S / u.m]
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Compute the magnetic Reynolds number.

    The magnetic Reynolds number is a dimensionless quantity that
    estimates the relative contributions of advection and induction
    to magnetic diffusion in a conducting medium.

    .. math::

        Rm = \frac{U L}{η}

    where :math:`η = \frac{1}{μ_0 σ}`
    and :math:`μ_0` is the permeability of free space.

    **Aliases:** `Rm_`

    Parameters
    ----------
    U : `~astropy.units.Quantity`
        The velocity scale of the plasma.

    L : `~astropy.units.Quantity`
        The length scale of the plasma.

    sigma : `~astropy.units.Quantity`
        The conductivity of the plasma.

    Returns
    -------
    Rm : `~astropy.units.Quantity`
        The magnetic Reynolds number.

    Raises
    ------
    `TypeError`
        If ``U`` is not a `~astropy.units.Quantity` and cannot be
        converted into a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If ``U`` is not in appropriate units.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Examples
    --------
    >>> import astropy.units as u
    >>> sigma = 5.96e7 * u.S / u.m
    >>> U = 10 * u.m / u.s
    >>> L = 1 * u.cm
    >>> Mag_Reynolds(U, L, sigma)
    <Quantity 7.48955689>
    >>> rho = 1e-8 * u.S / u.m
    >>> U = 0.1 * u.m / u.s
    >>> L = 0.05 * u.m
    >>> Mag_Reynolds(U, L, sigma)
    <Quantity 0.37447784>
    """
    eta = 1 / (mu0 * sigma)
    return abs(U * L / eta)


Rm_ = Mag_Reynolds
"""Alias to `~plasmapy.formulary.dimensionless.Mag_Reynolds`."""


def Lundquist_number(
    L: u.Quantity[u.m],
    B: u.Quantity[u.T],
    density: u.Quantity[u.m**-3, u.kg / u.m**3],
    sigma: u.Quantity[u.S / u.m],
    ion: ParticleLike | None = None,
    z_mean: float | None = None,
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Compute the ratio of the Alfén wave crossing timescale to the
    magnetic diffusion timescale.

    The Lundquist number :math:`S` is a dimensionless quantity that
    compares the Alfvén wave crossing timescale to the magnetic
    diffusion timescale in a conducting medium. It is given by

    .. math::

        S = \frac{L V_A}{η}

    where L is the length scale, :math:`V_A = B / \sqrt{μ_0 ρ}` is the
    Alfvén speed, :math:`B` is the magnetic field, :math:`ρ` is the mass
    density, :math:`μ_0` is the permeability of free space,
    :math:`η = 1 / (μ_0 σ)` is the magnetic diffusivity, and :math:`σ`
    is the electrical conductivity.

    Parameters
    ----------
    L : `~astropy.units.Quantity`
        The length scale of the plasma.

    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    density : `~astropy.units.Quantity`
        Either the ion number density :math:`n_i` in units convertible
        to m\ :sup:`-3` or the total mass density :math:`ρ` in units
        convertible to kg m\ :sup:`-3`\ .

    sigma : `~astropy.units.Quantity`
        The conductivity of the plasma.

    ion : `~plasmapy.particles.particle_class.Particle`, optional
        Representation of the ion species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized. If the density
        is an ion number density, then this parameter is required in
        order to convert to mass density.

    z_mean : `float`, optional
        The average ionization state (arithmetic mean) of the ``ion``
        composing the plasma. This is used in calculating the mass
        density :math:`ρ = n_i (m_i + Z_{mean} m_e)`. ``z_mean`` is
        ignored if ``density`` is passed as a mass density and overrides
        any charge state info provided by ``ion``.

    Returns
    -------
    S : `~astropy.units.Quantity`
        The Lundquist number.

    Raises
    ------
    `~plasmapy.utils.exceptions.RelativityError`
        If the Alfvén velocity is greater than or equal to the speed of
        light.

    `TypeError`
        If ``B`` and/or ``density`` are not of type `~astropy.units.Quantity`,
        or convertible to one.

    `TypeError`
        If ``ion`` is not of type or convertible to
        `~plasmapy.particles.particle_class.Particle`.

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
        If units are not provided, SI units are assumed.

    Notes
    -----
    For calculating the Alfvén speed `~plasmapy.formulary.speeds.Alfven_speed`
    is used and for calculating the Lundquist number
    `~plasmapy.formulary.dimensionless.Mag_Reynolds` is used.

    The Lundquist number is an important quantity in the study of
    magnetic reconnection. For example, reconnection rates in both the
    Sweet-Parker and Petschek models of magnetic reconnection can be expressed
    in terms of the Lundquist number. In the Sweet-Parker model, a current
    sheet with half-width :math:`L`, conductivity :math:`σ`, magnetic
    diffusivity :math:`η = 1 / (μ_0 σ)`, and Alfvén speed :math:`V_A`
    at the inflow has a Lundquist number of
    :math:`S = L V_A / η`. The dimensionless reconnection rate :math:`R`,
    i.e., the ratio of the inflow to outflow speeds, can then be expressed as
    :math:`R \sim 1 / \sqrt{S}`. Similarly, the maximum reconnection rate
    in the Petschek model can be expressed as approximately
    :math:`π / (8 \ln S)` :cite:p:`priest:2000`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.constants.si import m_p, m_e
    >>> L = 10**8 * u.m
    >>> B = 10**2 * u.G
    >>> n = 10**19 * u.m**-3
    >>> rho = n * (m_p + m_e)
    >>> sigma = 10**-7 * u.S / u.m
    >>> Lundquist_number(L, B, rho, sigma)
    <Quantity 0.866538...>
    >>> Lundquist_number(L, B, n, sigma, ion="p+")
    <Quantity 0.866538...>
    >>> Lundquist_number(L, B, n, sigma, ion="He +2")
    <Quantity 0.434819...>
    >>> Lundquist_number(L, B, n, sigma, ion="He", z_mean=1.8)
    <Quantity 0.434819...>
    >>> sigma = 10**-2 * u.S / u.m
    >>> Lundquist_number(L, B, n, sigma, ion="He", z_mean=1.8)
    <Quantity 43481.96672...>
    """
    alfven = speeds.Alfven_speed(B, density, ion=ion, Z=z_mean)
    return Mag_Reynolds(alfven, L, sigma)
