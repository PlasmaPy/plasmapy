"""Calculation of quantities related to relativity."""

__all__ = ["Lorentz_factor", "relativistic_energy", "RelativisticBody"]

from numbers import Real

import astropy.units as u
import numpy as np
from astropy.constants import c
from numpy.typing import DTypeLike

from plasmapy.particles.decorators import particle_input
from plasmapy.particles.particle_class import CustomParticle, Particle, ParticleLike
from plasmapy.particles.particle_collections import ParticleList
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import RelativityError


@validate_quantities(V={"can_be_negative": True})
def Lorentz_factor(V: u.Quantity[u.m / u.s]):
    r"""
    Return the Lorentz factor.

    Parameters
    ----------
    V : `~astropy.units.Quantity`
        The velocity in units convertible to meters per second.

    Returns
    -------
    gamma : `float` or `~numpy.ndarray`
        The Lorentz factor associated with the inputted velocities.

    Raises
    ------
    `TypeError`
        If ``V`` is not a `~astropy.units.Quantity` and cannot be
        converted into a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the ``V`` is not in appropriate units.

    `ValueError`
        If the magnitude of ``V`` is faster than the speed of light.

    Warns
    -----
    `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The Lorentz factor is a dimensionless number given by

    .. math::
        γ = \frac{1}{\sqrt{1-\frac{V^2}{c^2}}}

    The Lorentz factor is approximately one for sub-relativistic
    velocities, and :math:`γ → ∞` as the velocity approaches the
    speed of light.

    Examples
    --------
    >>> import astropy.units as u
    >>> velocity = 1.4e8 * u.m / u.s
    >>> Lorentz_factor(velocity)
     np.float64(1.1308856039...)
    >>> Lorentz_factor(299792458 * u.m / u.s)
    inf
    """

    if not np.all((np.abs(V) <= c) | (np.isnan(V))):
        raise RelativityError(
            "The Lorentz factor cannot be calculated for "
            "speeds faster than the speed of light."
        )

    if V.size > 1:
        γ = np.zeros_like(V.value)

        equals_c = np.abs(V) == c
        is_slow = ~equals_c

        γ[is_slow] = ((1 - (V[is_slow] / c) ** 2) ** -0.5).value
        γ[equals_c] = np.inf

    else:
        γ = np.inf if np.abs(V) == c else ((1 - (V / c) ** 2) ** -0.5).value
    return γ


@validate_quantities(validations_on_return={"can_be_negative": False})
@particle_input
def relativistic_energy(
    particle: ParticleLike,
    V: u.Quantity[u.m / u.s],
    *,
    mass_numb: int | None = None,
    Z: int | None = None,
    m=None,
    v=None,
) -> u.Quantity[u.J]:
    """
    Calculate the sum of the mass energy and kinetic energy of a
    relativistic body.

    The total energy of a relativistic body is:

    .. math::

        E = γ m c^2,

    where :math:`m` is the rest mass of the body and :math:`γ` is its
    `~plasmapy.formulary.relativity.Lorentz_factor`.

    Parameters
    ----------
    particle : |particle-like|
        A representation of a particle from which to get the mass
        of the relativistic body. If it is a |Quantity|, then it must
        have units of mass and describe the body's rest mass.

    V : `~astropy.units.Quantity`
        The velocity in units convertible to meters per second.

    mass_numb : integer, |keyword-only|, optional
        The mass number of an isotope, if not provided to ``particle``.

    Z : integer, |keyword-only|, optional
        The |charge number| of an ion or neutral atom, if not provided
        to ``particle``.

    Returns
    -------
    `~astropy.units.Quantity`
        The total energy of the relativistic body.

    Other Parameters
    ----------------
    m : `object`
        Formerly the mass of the body. Will raise a `TypeError` if
        provided. Use ``particle`` instead.

    v : `object`
        Formerly the velocity of the body. Will raise a `TypeError` if
        provided. Use ``V`` instead.

    Raises
    ------
    |InvalidParticleError|
        If ``particle`` does not represent a valid |Particle|,
        |CustomParticle|, or |ParticleList|.

    `~astropy.units.UnitConversionError`
        If ``V`` is not in appropriate units.

    `~plasmapy.utils.exceptions.RelativityError`
        If ``V`` exceeds the speed of light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Examples
    --------
    >>> import astropy.units as u
    >>> velocity = 1.4e8 * u.m / u.s
    >>> mass = 1 * u.kg
    >>> relativistic_energy(mass, velocity)
    <Quantity 1.01638929e+17 J>
    >>> relativistic_energy(mass, 299792458 * u.m / u.s)
    <Quantity inf J>
    >>> relativistic_energy(1 * u.mg, 1.4e8 * u.m / u.s)
    <Quantity 1.01638929e+11 J>
    """
    # TODO: Remove references to the parameters ``m`` and ``v`` in the
    # docstring and below no sooner than 2024.

    if m is not None or v is not None:
        raise TypeError(
            "The parameters 'm' and 'v' to relativistic_energy have "
            " been removed. Use 'particle' instead of 'm' and 'V' "
            "instead of 'v'."
        )

    γ = Lorentz_factor(V)
    return γ * particle.mass * c**2


class RelativisticBody:
    r"""
    A physical body that is moving at a velocity relative to the speed
    of light.

    Parameters
    ----------
    particle : |particle-like|
        A representation of a particle from which to get the mass
        of the relativistic body. If it is a |Quantity|, then it must
        have units of mass and describe the body's rest mass.

    V : |Quantity|, optional
        The velocity of the relativistic body in units convertible to
        m/s. The absolute magnitude of ``V`` cannot be greater than
        :math:`c`.

    momentum : |Quantity|, optional
        The momentum of the relativistic body in units convertible to
        kg·m/s.

    total_energy : |Quantity|, |keyword-only|, optional
       The sum of the mass energy and the kinetic energy in units
       convertible to joules. Must be non-negative.

    kinetic_energy : |Quantity|, |keyword-only|, optional
       The kinetic energy of the relativistic body in units convertible
       to joules. Must be non-negative.

    v_over_c : real number or |Quantity|, |keyword-only|, optional
       The ratio of the velocity to the speed of light. Must have an
       absolute magnitude :math:`≤ 1`.

    lorentz_factor : real number or |Quantity|, |keyword-only|, optional
       The Lorentz factor, :math:`γ` of the relativistic body. Must have
       :math:`γ ≥ 1`.

    Z : integer, |keyword-only|, optional
        The charge number associated with ``particle``.

    mass_numb : integer, |keyword-only|, optional
        The mass number associated with ``particle``.

    dtype : |DTypeLike|, |keyword-only|, default: `numpy.longdouble`
        The `numpy` data type to use to store the inputs.

    Notes
    -----
    At most one of ``V``, ``momentum``, ``total_energy``,
    ``kinetic_energy``, ``v_over_c``, and ``lorentz_factor`` must be
    provided.

    .. caution::

       For ultra-high-energy cosmic rays (UHECRs), the velocity may be
       within roundoff error of :math:`c` and :math:`\frac{V}{c}` may be
       within roundoff error of 1.

    Examples
    --------
    >>> import astropy.units as u
    >>> relativistic_proton = RelativisticBody("p+", total_energy=1 * u.GeV)
    >>> relativistic_proton.particle
    Particle("p+")
    >>> relativistic_proton.velocity
    <Quantity 1.03697...e+08 m / s>
    >>> print(relativistic_proton.v_over_c)
    0.34589808987468...
    >>> print(relativistic_proton.lorentz_factor)
    1.06578892478889...
    >>> relativistic_proton.mass_energy.to("GeV")
    <Quantity 0.93827... GeV>
    >>> relativistic_proton.total_energy.to("GeV")
    <Quantity 1. GeV>
    >>> relativistic_proton.mass
    <Quantity 1.67262...e-27 kg>

    |RelativisticBody| also works with multiple particles and/or
    velocities.

    >>> particles = ["p+", "e-"]
    >>> velocities = [2e5, 2e8] * u.m / u.s
    >>> relativistic_particles = RelativisticBody(particles, velocities)
    >>> relativistic_particles.momentum
    <Quantity [3.345244...e-22, 2.445659...e-22] kg m / s>
    """

    @staticmethod
    def _get_speed_like_input(
        velocity_like_arguments: dict[str, u.Quantity | float],
    ) -> dict[str, u.Quantity | float]:
        not_none_arguments = {
            key: value
            for key, value in velocity_like_arguments.items()
            if value is not None
        }

        if len(not_none_arguments) != 1:
            raise ValueError(
                "RelativisticBody can accept no more than one of the following "
                "arguments: V, v_over_c, momentum, total_energy, kinetic_energy, "
                "and lorentz_factor."
            )

        return not_none_arguments or {"velocity": np.nan * u.m / u.s}

    def _store_velocity_like_argument(
        self, speed_like_input: dict[str, u.Quantity | float]
    ) -> None:
        """
        Take the velocity-like argument and store it via the setter for
        the corresponding attribute.
        """
        name = next(iter(speed_like_input.keys()))
        value = speed_like_input[name]
        if self._dtype:
            value = u.Quantity(value, dtype=self._dtype)
        setattr(self, name, value)

    @particle_input
    @validate_quantities(
        V={"can_be_inf": False, "none_shall_pass": True},
        momentum={"can_be_inf": False, "none_shall_pass": True},
        total_energy={"can_be_negative": False, "none_shall_pass": True},
        kinetic_energy={"can_be_negative": False, "none_shall_pass": True},
    )
    def __init__(
        self,
        particle: ParticleLike,
        V: u.Quantity[u.m / u.s] = None,
        momentum: u.Quantity[u.kg * u.m / u.s] = None,
        *,
        total_energy: u.Quantity[u.J] = None,
        kinetic_energy: u.Quantity[u.J] = None,
        v_over_c: float | None = None,
        lorentz_factor: float | None = None,
        Z: int | None = None,
        mass_numb: int | None = None,
        dtype: DTypeLike | None = np.longdouble,
    ) -> None:
        self._particle = particle

        self._dtype = dtype

        velocity_like_inputs = {
            "velocity": V,
            "momentum": momentum,
            "total_energy": total_energy,
            "kinetic_energy": kinetic_energy,
            "v_over_c": v_over_c,
            "lorentz_factor": lorentz_factor,
        }

        speed_like_input = self._get_speed_like_input(velocity_like_inputs)
        self._store_velocity_like_argument(speed_like_input)

    def __repr__(self) -> str:
        return f"RelativisticBody({self.particle}, {self.velocity})"

    @property
    def particle(self) -> CustomParticle | Particle | ParticleList:
        """
        Representation of the particle(s).

        Returns
        -------
        |Particle|, |CustomParticle|, or |ParticleList|
        """
        return self._particle

    @property
    @validate_quantities
    def mass(self) -> u.Quantity[u.kg]:
        r"""
        The rest mass of the body, :math:`m_0`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        return u.Quantity(self.particle.mass, dtype=self._dtype)

    @property
    @validate_quantities
    def mass_energy(self) -> u.Quantity[u.J]:
        r"""
        The rest mass energy of the body, :math:`m_0 c^2`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        return self.mass * c**2

    @property
    @validate_quantities
    def total_energy(self) -> u.Quantity[u.J]:
        r"""
        The sum of the rest mass energy and the kinetic energy of the
        body.

        .. math::

           E_\mathrm{tot} ≡ γ m_0 c^2.

        Returns
        -------
        ~astropy.units.Quantity
        """
        return np.sqrt(self.momentum**2 * c**2 + self.mass_energy**2)

    @property
    @validate_quantities
    def kinetic_energy(self) -> u.Quantity[u.J]:
        """
        The kinetic energy of the body.

        .. math::

           E_K ≡ m_0 c^2 (γ-1).

        Returns
        -------
        ~astropy.units.Quantity
        """
        return self.total_energy - self.mass_energy

    @property
    @validate_quantities
    def v_over_c(self) -> float:
        r"""
        The velocity of the body divided by the velocity of light:
        :math:`\frac{V}{c}`\ .

        Returns
        -------
        float
        """
        return (self.velocity / c).to(u.dimensionless_unscaled).value

    @property
    @validate_quantities
    def velocity(self) -> u.Quantity[u.m / u.s]:
        r"""
        The velocity of the body, :math:`V`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        velocity = self.momentum / np.sqrt(self.mass**2 + self.momentum**2 / c**2)
        return velocity.to(u.m / u.s)

    @property
    @validate_quantities
    def lorentz_factor(self) -> float:
        r"""
        The Lorentz factor of the body.

        .. math::

           γ ≡ \frac{1}{\sqrt{1 - \frac{V^2}{c^2}}}.

        Returns
        -------
        float
        """
        return Lorentz_factor(self.velocity)

    @property
    @validate_quantities
    def momentum(self) -> u.Quantity[u.kg * u.m / u.s]:
        r"""
        The magnitude of the momentum of the body.

        .. math::

           p ≡ γ m_0 V.

        Returns
        -------
        ~astropy.units.Quantity
        """
        return self._momentum

    @kinetic_energy.setter
    @validate_quantities(E_K={"can_be_negative": False})
    def kinetic_energy(self, E_K: u.Quantity[u.J]) -> None:
        self.total_energy = E_K + self.mass_energy

    @total_energy.setter
    @validate_quantities(E_tot={"can_be_negative": False})
    def total_energy(self, E_tot: u.Quantity[u.J]) -> None:
        self._momentum = np.sqrt(E_tot**2 - self.mass_energy**2) / c

    @v_over_c.setter
    def v_over_c(self, v_over_c_: float) -> None:
        self.velocity = v_over_c_ * c

    @velocity.setter
    @validate_quantities
    def velocity(self, V: u.Quantity[u.m / u.s]) -> None:
        self._momentum = (Lorentz_factor(V) * self.mass * V).to(u.kg * u.m / u.s)

    @lorentz_factor.setter
    def lorentz_factor(self, γ: float | u.Quantity):
        if not isinstance(γ, Real | u.Quantity):
            raise TypeError("Invalid type for Lorentz factor")

        if isinstance(γ, u.Quantity):
            try:
                γ = γ.to(u.dimensionless_unscaled).value
            except u.UnitConversionError as exc:
                raise u.UnitConversionError(
                    "The Lorentz factor must be dimensionless."
                ) from exc

        if γ < 1:
            raise ValueError("The Lorentz factor must be ≥ 1")

        self.velocity = c * np.sqrt(1 - γ**-2)

    @momentum.setter
    @validate_quantities
    def momentum(self, p: u.Quantity[u.kg * u.m / u.s]) -> None:
        self._momentum = p.to(u.kg * u.m / u.s)

    def __eq__(self, other) -> bool:
        _attributes_to_compare = (
            "particle",
            "kinetic_energy",
            "mass_energy",
            "total_energy",
            "v_over_c",
            "momentum",
        )

        for attr in _attributes_to_compare:
            if not hasattr(other, attr):
                return False
            self_value = getattr(self, attr)
            other_value = getattr(other, attr)
            if self_value != other_value:
                return False
        return True
