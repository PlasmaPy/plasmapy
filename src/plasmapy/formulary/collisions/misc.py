"""
Module of miscellaneous parameters related to collisions.
"""

__all__ = [
    "mobility",
    "Bethe_stopping",
    "Mott_Bethe_mean_squared_scattering",
    "Spitzer_resistivity",
]
__lite_funcs__ = ["Bethe_stopping_lite"]

from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np
import numpy.typing as npt
from scipy.special import erf

from plasmapy.formulary.collisions import frequencies
from plasmapy.formulary.lengths import Debye_length
from plasmapy.formulary.speeds import thermal_speed
from plasmapy.particles.atomic import reduced_mass
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.particle_class import Particle, ParticleLike
from plasmapy.utils.decorators import bind_lite_func, validate_quantities
from plasmapy.utils.decorators.checks import _check_relativistic
from plasmapy.utils.exceptions import PhysicsError

__all__ += __lite_funcs__

_c = const.c
_e = const.e.si
_eps0 = const.eps0
_m_e = const.m_e


@validate_quantities(T={"equivalencies": u.temperature_energy()})
@particle_input
def _process_inputs(T: u.Quantity[u.K], species: (Particle, Particle), V):
    """
    Helper function for processing inputs to functionality contained
    in `plasmapy.formulary.collisions`.

    Also obtains the reduced mass in a 2 particle collision system
    along with thermal velocity.
    """
    masses = [p.mass for p in species]
    charges = [np.abs(p.charge) for p in species]

    # obtaining reduced mass of 2 particle collision system
    reduced_mass_ = reduced_mass(*species)

    # getting thermal velocity of system if no velocity is given
    V = _replace_nan_velocity_with_thermal_velocity(V, T, reduced_mass_)

    _check_relativistic(V, "V")

    return T, masses, charges, reduced_mass_, V


# TODO: Remove redundant mass parameter
def _replace_nan_velocity_with_thermal_velocity(
    V,
    T,
    m,
    species=Particle("e-"),  # noqa: B008
):
    """
    Get thermal velocity of system if no velocity is given, for a given
    mass.  Handles vector checks for ``V``, you must already know that
    ``T`` and ``m`` are okay.
    """
    if np.any(V == 0):
        raise PhysicsError("Collisions are not possible with a zero velocity.")

    if V is None:
        return thermal_speed(T, species, mass=m)

    if not np.any(np.isnan(V)):
        return V

    if not (np.isscalar(V.value) or np.isscalar(T.value)):
        V = V.copy()
        V[np.isnan(V)] = thermal_speed(T[np.isnan(V)], species, mass=m)
        return V

    if np.isscalar(V.value):
        return thermal_speed(T, species, mass=m)

    if np.isscalar(T.value):
        V[np.isnan(V)] = thermal_speed(T, species, mass=m)

    return V


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def mobility(
    T: u.Quantity[u.K],
    n_e: u.Quantity[u.m**-3],
    species,
    z_mean: float = np.nan,
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    method: str = "classical",
) -> u.Quantity[u.m**2 / (u.V * u.s)]:
    r"""
    Return the electrical mobility.

    Parameters
    ----------
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
        non-classical impact parameters. It is also used to obtain the
        average mobility of a plasma with multiple charge state
        species. When ``z_mean`` is not given, the average charge
        between the two particles is used instead. ``z_mean`` is a
        required parameter if ``method`` is ``"ls_full_interp"``,
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
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for
        more information about these methods.

    Returns
    -------
    mobility_value : `float` or `numpy.ndarray`
        The electrical mobility of particles in a collisional plasma.

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
    The `mobility
    <https://en.wikipedia.org/wiki/Electrical_mobility#Mobility_in_gas_phase>`_
    is given by

    .. math::

        μ = \frac{q}{m ν}

    where :math:`q` is the particle charge, :math:`m` is the particle
    mass and :math:`ν` is the collisional frequency of the particle in
    the plasma.

    The mobility describes the forced diffusion of a particle in a
    collisional plasma which is under the influence of an electric
    field. The mobility is essentially the ratio of drift velocity due
    to collisions and the electric field driving the forced diffusion.

    Examples
    --------
    .. autolink-skip:: section

    >>> import astropy.units as u
    >>> n = 1e19 * u.m**-3
    >>> T = 1e6 * u.K
    >>> species = ("e", "p")
    >>> mobility(T, n, species)  # doctest: +SKIP
    <Quantity 250505... m2 / (V s)>
    >>> mobility(T, n, species, V=1e6 * u.m / u.s)  # doctest: +SKIP
    <Quantity 1921.2784... m2 / (V s)>
    """
    freq = frequencies.collision_frequency(
        T=T, n=n_e, species=species, z_mean=z_mean, V=V, method=method
    )
    # we do this after collision_frequency since collision_frequency
    # already has a _process_inputs check and we are doing this just
    # to recover the charges, mass, etc.
    T, masses, charges, reduced_mass_, V = _process_inputs(T=T, species=species, V=V)
    z_val = (charges[0] + charges[1]) / 2 if np.isnan(z_mean) else z_mean * _e
    return z_val / (reduced_mass_ * freq)


def Bethe_stopping_lite(
    I: npt.NDArray[np.integer[Any] | np.floating[Any]],  # noqa: E741
    n: npt.NDArray[np.integer[Any] | np.floating[Any]],
    v: npt.NDArray[np.integer[Any] | np.floating[Any]],
    z: int,
) -> npt.NDArray[np.integer[Any] | np.floating[Any]]:
    r"""
    The :term:`lite-function` version of `~plasmapy.formulary.collisions.misc.Bethe_stopping`. Performs the same
    calculations as `~plasmapy.formulary.collisions.misc.Bethe_stopping`, but is intended for computational use
    and thus has data conditioning safeguards removed.

    The theoretical electronic stopping power for swift charged particles
    calculated from the Bethe formula.

    The Bethe formula should only be used for high energy particles, as
    higher order corrections become non-negligible for smaller energies.

    By convention, this function returns a positive value for the stopping
    energy.

    Parameters
    ----------
    I: `~numpy.ndarray`
        The mean excitation energy for the material in which the particle is
        being stopped. Expressed in units of energy.

    n: `~numpy.ndarray`
        The electron number density of the material. Expressed in units of number density.

    v: `~numpy.ndarray`
        The velocity of the particle being stopped. Expressed in units of speed.

    z: `int`
        The charge of the charged particle in multiples of the electron charge.
        Expressed only as an integer.

    Returns
    -------
    dEdx : `~numpy.ndarray`
        The stopping power of the material given the particle's energy.

    """

    beta = v / _c.si.value

    return -np.asarray(
        4
        * np.pi
        * n
        * z**2
        / (_m_e.si.value * _c.si.value**2 * beta**2)
        * (_e.si.value**2 / (4 * np.pi * _eps0.si.value)) ** 2
        * (
            np.log(2 * _m_e.si.value * _c.si.value**2 * beta**2 / (I * (1 - beta**2)))
            - beta**2
        )
    )


@bind_lite_func(Bethe_stopping_lite)
@validate_quantities()
def Bethe_stopping(
    I: u.Quantity[u.J],  # noqa: E741
    n: u.Quantity[1 / u.m**3],
    v: u.Quantity[u.m / u.s],
    z: int,
) -> u.Quantity[u.J / u.m]:
    r"""
    The theoretical electronic stopping power for swift charged particles
    calculated from the Bethe formula.

    The Bethe formula should only be used for high energy particles, as
    higher order corrections become non-negligible for smaller energies.

    By convention, this function returns a positive value for the stopping
    energy.

    Parameters
    ----------
    I: `~astropy.units.Quantity`
        The mean excitation energy for the material in which the particle is
        being stopped. Expressed in units of energy.

    n: `~astropy.units.Quantity`
        The electron number density of the material. Expressed in units of number density.

    v: `~astropy.units.Quantity`
        The velocity of the particle being stopped. Expressed in units of speed.

    z: `int`
        The charge of the charged particle in multiples of the electron charge.
        Expressed only as an integer.

    Returns
    -------
    dEdx : `~astropy.units.Quantity`
        The stopping power of the material given the particle's energy.

    """

    return Bethe_stopping_lite(I.si.value, n.si.value, v.si.value, z) * u.J / u.m


# TODO: figure out if there's a better name for this. And where it came from.
#  See https://pubs.aip.org/aip/pop/article/30/9/093902/2910853/Monte-Carlo-simulations-of-charged-particle
#  which states that the original source is not in English
def Mott_Bethe_mean_squared_scattering(
    speed: u.Quantity[u.m / u.s],
    beam_particle: ParticleLike,
    target_particle: ParticleLike,
    T_s: u.Quantity[u.K],
    n_s: u.Quantity[1 / u.m**3],
) -> u.Quantity[u.Hz]:
    r"""
    Calculate the mean-squared rate of scattering for protons scattering
    through a screened Coulomb potential.

    Parameters
    ----------
    speed : `~astropy.units.Quantity`
        The speed of the particle.

    beam_particle : `~plasmapy.particles.particle_class.ParticleLike`
        A particle class representation of the incident particle beam.

    target_particle : `~plasmapy.particles.particle_class.ParticleLike`
        A particle class representation of the target.

    T_s : `~astropy.units.Quantity`
        The temperature of the target species.

    n_s : `~astropy.units.Quantity`
        The number density of scatterers in the material.

    Notes
    -----
    The mean-square rate of scattering of incident protons is given in Mott & Massey (1950) to be

    .. math::
        \frac{d\langle \Delta \theta^2 \rangle}{dt} = \frac{n_s q_{\alpha}^2 q_{\beta}^2}
        {2 \pi \epsilon_0 m_{\alpha}^2 v_{\alpha}^3} \ln{(\bar{\Lambda})}
        \Phi{\left( \frac{v_{\alpha}}{v_{\beta T}} \right)}

    where :math:`n_s` is the number density of scatterers, :math:`\ln{(\bar{\Lambda})}`,
    :math:`q_{\alpha, \beta}`, :math:`m_{\alpha, \beta}` and :math:`v_{\alpha, \beta}` are the particle charge,
    mass, and velocity in the rest frame, respectively. The subscript :math:`\alpha`
    refers to the beam species whereas :math:`\beta` refers to the target species.
    :math:`\ln{\left(\bar{\Lambda}\right)}` is the averaged Coulomb logarithm, given by :math:`\ln{(\lambda_D / \langle b_0 \rangle)}`
    where :math:`\lambda_D` is the Debye length and :math:`\langle b_0 \rangle` is the
    average value for the distance of closest approach associated with the
    classical Coulomb potential in the center of mass frame, given as :math:`b_0 = q_{\alpha}
    q_{\beta} / 4 \pi \eps_0 \mu \bar{u}^2`, where :math:`\bar{u}` is the average relative velocity.
    """
    mu = reduced_mass(target_particle, beam_particle)

    b_0 = (
        target_particle.atomic_number
        * beam_particle.atomic_number
        * const.e.si**2
        / (4 * np.pi * const.eps0 * mu * speed**2)
    )

    lambda_D = Debye_length(T_s * u.K, n_s / u.m**3)
    L = np.log((lambda_D / b_0).si.value)

    result = (
        n_s
        * (beam_particle.atomic_number * target_particle.atomic_number * const.e.si**2)
        ** 2
        / (2 * np.pi * const.eps0**2 * beam_particle.mass**2 * speed**3)
        * L
        * erf((speed / thermal_speed(T_s * u.K, target_particle)).si.value)
    )

    return result.si.value


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n={"can_be_negative": False},
)
def Spitzer_resistivity(
    T: u.Quantity[u.K],
    n: u.Quantity[u.m**-3],
    species,
    z_mean: float = np.nan,
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    method: str = "classical",
) -> u.Quantity[u.Ohm * u.m]:
    r"""
    Spitzer resistivity of a plasma.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Temperature in units of Kelvin or eV.  This should be the
        electron temperature for electron-electron and electron-ion
        collisions, and the ion temperature for ion-ion collisions. An
        example of temperature given in eV can be found below.

    n : `~astropy.units.Quantity`
        The density in units convertible to per cubic meter.  This
        should be the electron density for electron-electron collisions,
        and the ion density for electron-ion and ion-ion collisions.

    z_mean : `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) of a plasma for which a
        macroscopic description is valid. This parameter is used to
        compute the average ion density (given the average ionization
        and electron density) for calculating the ion sphere radius for
        non-classical impact parameters. ``z_mean`` is a required
        parameter if ``method`` is ``"ls_full_interp"``,
        ``"hls_max_interp"``, or ``"hls_full_interp"``.

    species : `tuple`
        A tuple containing string representations of the test particle
        (listed first) and the target particle (listed second).

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
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for
        more information about these methods.

    Returns
    -------
    spitzer : `float` or `numpy.ndarray`
        The resistivity of the plasma in ohm meters.

    Raises
    ------
    `ValueError`
        If the mass or charge of either particle cannot be found, or any
        of the inputs contain incorrect values.

    `~astropy.units.UnitConversionError`
        If the units on any of the inputs are incorrect.

    `TypeError`
        If any of ``n_e``, ``T``, or ``V`` are not of type
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
    The Spitzer resistivity (see Ch. 5 of :cite:t:`chen:2016`) is given
    by:

    .. math::

        η = \frac{m}{n Z_1 Z_2 q_e^2} ν_{1,2}

    where :math:`m` is the ion mass or the reduced mass, :math:`n` is
    the ion density, :math:`Z` is the particle charge state, :math:`q_e`
    is the charge of an electron, :math:`ν_{1,2}` is the collisional
    frequency between particle species 1 and 2.

    Typically, particle species 1 and 2 are selected to be an electron
    and an ion, since electron-ion collisions are inelastic and
    therefore produce resistivity in the plasma.

    Examples
    --------
    .. autolink-skip:: section

    >>> import astropy.units as u
    >>> n = 1e19 * u.m**-3
    >>> T = 1e6 * u.K
    >>> species = ("e", "p")
    >>> Spitzer_resistivity(T, n, species)  # doctest: +SKIP
    <Quantity 2.4915...e-06 Ohm m>
    >>> Spitzer_resistivity(T, n, species, V=1e6 * u.m / u.s)  # doctest: +SKIP
    <Quantity 0.000324... Ohm m>
    >>> T_eV = 86.173 * u.eV
    >>> T_K = (T_eV).to("K", equivalencies=u.temperature_energy())
    >>> Spitzer_resistivity(T_K, n, species)
    """
    # collisional frequency
    freq = frequencies.collision_frequency(
        T=T, n=n, species=species, z_mean=z_mean, V=V, method=method
    )
    # fetching additional parameters
    T, masses, charges, reduced_mass_, V = _process_inputs(T=T, species=species, V=V)
    return (
        freq * reduced_mass_ / (n * charges[0] * charges[1])
        if np.isnan(z_mean)
        else freq * reduced_mass_ / (n * (z_mean * _e) ** 2)
    )
