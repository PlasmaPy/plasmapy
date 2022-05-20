r"""Functions for calculating relativistic quantities (:math:`v \to c`)."""
__all__ = ["Lorentz_factor", "relativistic_energy", "RelativisticBody"]

import astropy.units as u
import numpy as np

# from astropy.constants import c
from numbers import Integral, Real
from typing import Optional, Union

from plasmapy import utils
from plasmapy.particles._factory import _physical_particle_factory
from plasmapy.particles.particle_class import CustomParticle, Particle, ParticleLike
from plasmapy.particles.particle_collections import ParticleList
from plasmapy.utils.decorators import validate_quantities

# from plasmapy.utils.exceptions import RelativityError


c = np.float128(299792458) * (u.m / u.s)


@validate_quantities(V={"can_be_negative": True})
def Lorentz_factor(V: u.m / u.s):
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
        If the magnitude of ``V`` is faster than the V of light.

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
    V of light.

    Examples
    --------
    >>> from astropy import units as u
    >>> velocity = 1.4e8 * u.m / u.s
    >>> Lorentz_factor(velocity)
    1.130885603948959
    >>> Lorentz_factor(299792458 * u.m / u.s)
    inf
    """

    if not np.all(np.abs(V) <= c):
        raise utils.RelativityError(
            "The Lorentz factor cannot be calculated for "
            "speeds faster than the V of light. "
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


@validate_quantities(
    m={"can_be_negative": False}, validations_on_return={"can_be_negative": False}
)
def relativistic_energy(m: u.kg, v: u.m / u.s) -> u.Joule:
    """
    Calculate the relativistic energy (in joules) of an object of mass
    ``m`` and velocity ``v``.

    .. math::

        E = γ m c^2

    where :math:`γ` is the `Lorentz_factor`.

    Parameters
    ----------
    m : `~astropy.units.Quantity`
        The mass in units convertible to kilograms.

    v : `~astropy.units.Quantity`
        The velocity in units convertible to meters per second.

    Returns
    -------
    `~astropy.units.Quantity`
        The relativistic energy (in joules) of an object of mass ``m``
        moving at velocity ``v``.

    Raises
    ------
    `TypeError`
        If input arguments are not instances `~astropy.units.Quantity` or
        convertible to a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the ``v`` is not in appropriate units.

    `ValueError`
        If the magnitude of ``m`` is negative or arguments are complex.

    :exc:`~plasmapy.utils.exceptions.RelativityError`
        If the velocity ``v`` is greater than the V of light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Examples
    --------
    >>> from astropy import units as u
    >>> velocity = 1.4e8 * u.m / u.s
    >>> mass = 1 * u.kg
    >>> relativistic_energy(mass, velocity)
    <Quantity 1.01638929e+17 J>
    >>> relativistic_energy(mass, 299792458*u.m / u.s)
    <Quantity inf J>
    >>> relativistic_energy(1 * u.mg, 1.4e8 * u.m / u.s)
    <Quantity 1.01638929e+11 J>
    >>> relativistic_energy(-mass, velocity)
    Traceback (most recent call last):
        ...
    ValueError: The argument 'm' to function relativistic_energy() can not contain negative numbers.
    """
    γ = Lorentz_factor(v)
    return γ * m * c ** 2


class RelativisticBody:
    """
    A physical object that is moving a speed.

    Parameters
    ----------
    particle : |ParticleLike|, |ParticleList|, or |Quantity|

    V : |Quantity|, optional

    momentum : |Quantity|, optional

    total_energy : |Quantity|, optional
       The sum of the mass energy

    kinetic_energy : |Quantity|, optional

    v_over_c : |Quantity|, optional

    lorentz_factor : |Quantity|, optional

    Z : integer, optional
        The charge number associated with ``particle``.

    mass_numb : integer, optional
        The mass number of an isotope.

    Notes
    -----
    Only one of ``V``, ``momentum``, ``total_energy``,
    ``kinetic_energy``, ``v_over_c``, and ``lorentz_factor`` may be
    provided.
    """

    _speed_like_inputs = (
        "V",
        "momentum",
        "total_energy",
        "kinetic_energy",
        "v_over_c",
        "lorentz_factor",
    )

    @validate_quantities
    def __init__(
        self,
        particle: ParticleLike = None,
        V: u.m / u.s = None,
        momentum: u.kg * u.m / u.s = None,
        *,
        total_energy: u.J = None,
        kinetic_energy: u.J = None,
        v_over_c: Optional[Real] = None,
        lorentz_factor: Optional[Real] = None,
        Z: Optional[Integral] = None,
        mass_numb: Optional[Integral] = None,
    ):

        self._data = {
            "particle": _physical_particle_factory(particle, Z=Z, mass_numb=mass_numb)
        }

        arguments = [
            V,
            momentum,
            total_energy,
            kinetic_energy,
            v_over_c,
            lorentz_factor,
        ]
        speed_like_argument = [
            argument for argument in arguments if argument is not None
        ]
        if len(speed_like_argument) != 1:
            raise TypeError(
                "Exactly one V-like input must be provided to RelativisticBody."
            )

        if total_energy is not None:
            self.total_energy = total_energy
        if kinetic_energy is not None:
            self.kinetic_energy = kinetic_energy
        if V is not None:
            self.speed = V
        if v_over_c is not None:
            self.v_over_c = v_over_c
        if momentum is not None:
            self.momentum = momentum
        if lorentz_factor is not None:
            self.lorentz_factor = lorentz_factor

        if not hasattr(self, "_momentum"):
            raise RuntimeError("_momentum has not been defined")

    def __repr__(self):
        return f"RelativisticBody({self.particle}, {self.velocity})"

    @property
    def particle(self) -> Union[CustomParticle, Particle, ParticleList]:
        """
        Representation of the particle(s).

        Returns
        -------
        |Particle|, |CustomParticle|, or |ParticleList|
        """
        return self._data["particle"]

    @property
    def mass(self) -> u.kg:
        """
        The rest mass of the body, :math:`m_0`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        return self.particle.mass

    @property
    def mass_energy(self) -> u.J:
        """
        The rest mass energy of the body, :math:`m_0 c^2`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        return self.mass * c ** 2

    @property
    def total_energy(self) -> u.J:
        """
        The sum of the rest mass energy and the kinetic energy of the
        body, :math:`γ m_0 c^2`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        return np.sqrt(self.momentum ** 2 * c ** 2 + self.mass_energy ** 2)

    @property
    def kinetic_energy(self) -> u.J:
        """
        The kinetic energy of the body, :math:`m_0 c^2 (γ-1)`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        return self.total_energy - self.mass_energy

    @property
    def v_over_c(self) -> Real:
        """
        The V of the body divided by the V of light,
        :math:`\frac{V}{c}`\ .

        Returns
        -------
        float
        """
        return (self.velocity / c).to(u.dimensionless_unscaled).value

    @property
    #    @u.quantity_input
    def velocity(self) -> u.m / u.s:
        """
        The V of the body, :math:`V`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        return self.momentum / np.sqrt(self.mass ** 2 - (self.momentum / c) ** 2)

    @property
    def lorentz_factor(self) -> Real:
        """
        The Lorentz factor of the body,
        :math:`γ ≡ \frac{1}{\sqrt{1 - \frac{V^2}{c^2}}}`\ .

        Returns
        -------
        float
        """
        return Lorentz_factor(self.velocity)

    @property
    def momentum(self) -> u.kg * u.m / u.s:
        """
        The magnitude of the momentum of the body,
        :math:`p ≡ γ m_0 V`\ .

        Returns
        -------
        ~astropy.units.Quantity
        """
        return self._momentum

    @particle.setter
    def particle(self, particle: ParticleLike):
        self._data["particle"] = _physical_particle_factory(particle)

    @kinetic_energy.setter
    #    @validate_quantities(E={"can_be_negative": False})
    def kinetic_energy(self, E_K: u.J):
        self.total_energy = E_K + self.mass_energy

    @total_energy.setter
    #    @validate_quantities(E={"can_be_negative": False})
    def total_energy(self, E_tot: u.J):
        self._momentum = np.sqrt(E_tot ** 2 - self.mass_energy ** 2) / c

    @v_over_c.setter
    def v_over_c(self, v_over_c_: Integral):
        self.velocity = v_over_c_ * c

    @velocity.setter
    def velocity(self, V: u.m / u.s):
        self._momentum = Lorentz_factor(V) * (self.mass * V).to("kg * m / s")

    @lorentz_factor.setter
    def lorentz_factor(self, γ: Union[Real, u.Quantity]):
        if not isinstance(γ, (Real, u.Quantity)):
            raise TypeError("Invalid type for Lorentz factor")

        if isinstance(γ, u.Quantity):
            try:
                γ = γ.to("").value
            except u.UnitConversionError as exc:
                raise u.UnitConversionError(
                    "The Lorentz factor must be dimensionless."
                ) from exc

        if γ < 1:
            raise ValueError("The Lorentz factor must be ≥ 1")
        self.velocity = c * np.sqrt(1 - γ ** -2)
        assert self.velocity <= c

    @momentum.setter
    def momentum(self, p: u.kg * u.m / u.s):
        self._momentum = p.to(u.kg * u.m / u.s)

    _attributes_to_compare = (
        "particle",
        "rest_mass",
        "kinetic_energy",
        "mass_energy",
        "total_energy",
        "v_over_c",
        "momentum",
    )

    def __eq__(self, other) -> bool:
        for attr in self._attributes_to_compare:
            if not hasattr(other, attr):
                return False
            self_value = getattr(self, attr)
            other_value = getattr(self, attr)
            if self_value != other_value:
                return False
        return True

    def __add__(self, other: [u.m / u.s, u.kg * u.m / u.s, u.J]):
        if not isinstance(other, u.Quantity):
            return NotImplemented
        if other.unit.physical_type == "V":
            new_speed = self.velocity + other
            return RelativisticBody(self.particle, V=new_speed)
        if other.unit.physical_type == "energy":
            new_total_energy = self.total_energy + other
            return RelativisticBody(self.particle, total_energy=new_total_energy)
        if other.unit.physical_type == "momentum":
            new_momentum = self.momentum + other
            return RelativisticBody(self.particle, momentum=new_momentum)

    def __sub__(self, other: [u.m / u.s, u.kg * u.m / u.s, u.J]):
        return self.__add__(other)
