"""Classes to represent normalizations of plasma equations."""

__all__ = ["NormalizationError", "MHDNormalizations"]

from collections.abc import Iterable

import astropy.units as u
from astropy.constants import k_B, m_e
from astropy.constants import mu0 as μ0

from plasmapy.formulary import Alfven_speed
from plasmapy.particles import CustomParticle, Particle, ParticleLike, particle_input
from plasmapy.particles.exceptions import ChargeError
from plasmapy.simulation.abstractions import AbstractNormalizations
from plasmapy.utils._units_helpers import _get_physical_type_dict
from plasmapy.utils.exceptions import PlasmaPyError

_length = u.get_physical_type(u.m)
_magnetic_field = u.get_physical_type(u.T)
_mass_density = u.get_physical_type(u.kg * u.m**-3)
_time = u.get_physical_type(u.s)
_number_density = u.get_physical_type(u.m**-3)
_velocity = u.get_physical_type(u.m / u.s)

_allowed_physical_types = {
    _length,
    _magnetic_field,
    _mass_density,  # allow this?
    _number_density,
    _time,
    _velocity,
}


@particle_input
def _resolve_ion(
    ion: ParticleLike,
    Z: float | None = None,
    mass_numb: int | None = None,
) -> Particle | CustomParticle:
    """
    Take particle-like arguments and return the corresponding ion object.

    This workaround is needed because |particle_input| does not work with
    setters (see :issue:`2507`).
    """
    return ion


class NormalizationError(PlasmaPyError):
    """An exception for errors involving unable to find normalizations."""


class MHDNormalizations(AbstractNormalizations):
    r"""
    A class containing the |normalization constants| for the equations
    of magnetohydrodynamics.

    This class assumes a two-component electron-ion plasma, where the
    |charge number| of the ion is :math:`Z`.

    Parameters
    ----------
    *quantities : |Quantity|
        Three normalization constants.

    ion : |atom-like|
        The ion of the electron-ion plasma.

    Z : `int`, optional
        The charge number of the ion. Must be positive.

    mass_numb : `int`, optional
        The mass number of the isotope of the ion.

    Notes
    -----
    A |normalization constant| is a multiplicative factor which is used
    to transform a physical quantity into a dimensionless number and
    vice versa. For example, if we use the sound speed :math:`V_s` as
    the normalization for velocity, then the result will be the
    dimensionless :wikipedia:`Mach number`. Normalizations are often
    employed to provide a convenient reference value for the
    interpretation of data, to avoid calculations between very large and
    very small numbers in numerical simulations, and/or to simplify
    systems of equations for analytical theory or numerical solutions.

    This class represents the normalization coefficients of the
    equations of magnetohydrodynamics in SI units. We define :math:`n`
    as the number density, :math:`ρ` as mass density, :math:`\mathbf{B}`
    as the magnetic field, :math:`\mathbf{E}` as the electric field,
    :math:`\mathbf{V}` as the bulk plasma velocity, and
    :math:`\mathbf{J}` as the current density.

    The :wikipedia:`continuity equation` is:

    .. math::

        \frac{∂n}{∂t} + ∇ · \left( n \mathbf{V} \right) = 0.

    :wikipedia:`Ampere's law` without :wikipedia:`displacement current`
    is:

    .. math::

        μ_0 \mathbf{J} = ∇ × \mathbf{B}.

    :wikipedia:`Faraday's law` is: (trying out the mathematical bold
    unicode B on the left

    .. math::

        \frac{∂\mathbf{B} \mathbf{B}}{∂t} = - ∇ × \mathbf{B}

    The generalized Ohm's law is:

    .. math::

        \mathbf{E} + \mathbf{V} × \mathbf{B}
        = η \mathbf{J} + \frac{\mathbf{J} × \mathbf{B}{n_e e},

    where :math:`e` is the :wikipedia:`elementary charge`.

    The momentum equation is:

    .. math::

        ρ \left( \frac{∂}{∂t} + \mathbf{V} · ∇ \right) \mathbf{V}
        = \mathbf{J} × \mathbf{B} - ∇ p.

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.simulation.normalizations import MHDNormalizations
    >>> n0 = 1e19 * u.m**-3
    >>> L0 = 1 * u.km
    >>> B0 = 10 * u.G
    >>> normalizations = MHDNormalizations(n0, L0, B0, ion="p+")
    >>> normalizations.ion
    Particle("p+")
    >>> normalizations.number_density
    <Quantity 1.e+19 1 / m3>
    >>> normalizations.magnetic_field
    <Quantity 0.001 T>
    >>> normalizations.length
    <Quantity 1000. m>
    """

    def __init__(
        self,
        *quantities: Iterable[u.Quantity],
        ion: ParticleLike | None = None,
        Z: float | None = None,
        mass_numb: int | None = None,
    ):
        self._ptypes_to_quantities = _get_physical_type_dict(
            quantities,
            only_quantities=True,
            strict=True,
            allowed_physical_types=_allowed_physical_types,
        )

        self.ion = ion

        include_spatiotemporal = [
            _length in self._ptypes_to_quantities,
            _time in self._ptypes_to_quantities,
            _velocity in self._ptypes_to_quantities,
        ]

        if not include_spatiotemporal or all(include_spatiotemporal):
            raise NormalizationError(
                "Must provide at least one but no more than two Quantity "
                "objects with a physical type of length, time, or velocity."
            )

    @property
    def acceleration(self) -> u.Quantity[u.m * u.s**-2]:
        r"""
        The |normalization constant| for acceleration,
        :math:`a_⭑ ≡ \frac{L_⭑}{t_⭑^2}`.

        Returns
        -------
        |Quantity|
        """
        return self.length / self.time**2

    @property
    def area(self) -> u.Quantity[u.m**2]:
        r"""
        The |normalization constant| for area.

        .. math::

           A_⋆ ≡ L_⋆^2
        """
        return self.length**2

    @property
    def current_density(self) -> u.Quantity[u.A * u.m**-2]:
        r"""
        The |normalization constant| for :wikipedia:`current density`,
        :math:`J_⭑ ≡ \frac{B_⭑}{μ_0 L_⭑}`.

        Returns
        -------
        |Quantity|
        """
        return self.magnetic_field / (μ0 * self.length)

    @property
    def diffusivity(self) -> u.Quantity[u.m**2 / u.s]:
        r"""
        The |normalization constant| for :wikipedia:`diffusivity`,
        :math:`D_⭑ ≡ \frac{L_⭑^2}{t_⭑}`.

        Returns
        -------
        |Quantity|
        """
        return self.length**2 / self.time

    @property
    def dynamic_viscosity(self) -> u.Quantity[u.Pa * u.s]:
        r"""
        The |normalization constant| for :wikipedia:`dynamic viscosity`,
        :math:`μ_⭑ ≡ p_⭑ t_⭑`.

        .. danger::

           Verify this!

        Returns
        -------
        |Quantity|

        Notes
        -----
        For `kinematic viscosity`_, use
        ~`plasmapy.simulation.abstractions.MHDNormalizations.diffusivity`.

        .. _kinematic viscosity: https://en.wikipedia.org/wiki/Viscosity#Kinematic_viscosity
        """
        return self.pressure * self.time

    @property
    def electric_field(self) -> u.Quantity[u.V / u.m]:
        """
        The |normalization constant| for electric field strength,
        :math:`E_⭑ ≡ V_⭑ B_⭑`.

        Returns
        -------
        |Quantity|
        """
        return self.velocity * self.magnetic_field

    @property
    def energy(self) -> u.Quantity[u.J]:
        r"""
        The |normalization constant| for energy,
        :math:`e_⋆ ≡ \frac{m_⋆ L_⋆^2}{t_⋆^2}`.

        .. danger::

           Verify this!

        Returns
        -------
        |Quantity|
        """
        return ...

    @property
    def frequency(self) -> u.Quantity[u.s**-1]:
        """
        The |normalization constant| for frequency,
        :math:`f_⭑ ≡ t_⭑^{-1}`.

        .. danger::

           Clear up whether this related to regular frequency vs angular
           frequency.

        Returns
        -------
        |Quantity|
        """
        return u.time**-1

    @property
    def heat_flux(self) -> u.Quantity[u.J * u.m**-2 * u.s**-1]:
        r"""
        The |normalization constant| for heat flux,
        :math:`q_⭑ ≡ \frac{n_⋆ e_⋆ L_⋆}{t_⋆}`.

        .. danger::

           Verify this!

        Returns
        -------
        |Quantity|
        """
        return ...

    @property
    def ion(self) -> Particle | CustomParticle:
        """
        The ion of the plasma.

        Returns
        -------
        |Particle|
        """
        return self._ion

    @ion.setter
    def ion(self, ion_: ParticleLike):
        ion = _resolve_ion(ion_)

        if ion is not None and ion.charge_number <= 0:
            raise ChargeError("The charge of the ion must be positive.")

        if ion is None and _mass_density not in self._ptypes_to_quantities:
            raise NormalizationError(
                "If no ion is provided, then a mass density must be provided."
            )

        self._ion = ion

    @property
    def length(self) -> u.Quantity[u.m]:
        r"""
        The length |normalization constant|, :math:`L_⭑`\ .

        Returns
        -------
        |Quantity|
        """
        return self._length

    @property
    def magnetic_field(self) -> u.Quantity[u.T]:
        r"""
        The magnetic field |normalization constant|, :math:`B_⭑`\ .

        Returns
        -------
        |Quantity|
        """
        return self._magnetic_field

    @property
    def magnetic_flux(self) -> u.Quantity[u.Wb]:
        r"""
        The magnetic flux |normalization constant|, :math:`Φ_⋆ ≡ B_⋆ A_⋆`.

        Returns
        -------
        |Quantity|
        """
        return self.magnetic_field * self.area

    @property
    def mass(self) -> u.Quantity[u.kg]:
        r"""
        The |normalization constant| for mass, :math:`m_⭑ ≡ m_i + Z m_e`.

        Here, :math:`m_i` is the ion mass, :math:`m_e` is the electron
        mass, and :math:`Z` is the |charge number| of the ion.

        .. danger::

           Verify this!

        Returns
        -------
        |Quantity|
        """
        return self.ion.mass + self.ion.charge_number * m_e

    @property
    def mass_density(self) -> u.Quantity[u.kg * u.m**-3]:
        r"""
        The |normalization constant| for mass density, :math:`ρ_⋆ ≡ m_⋆ n_⋆`.

        .. danger::

           How do we get :math:`m_⋆`\ ?

        Returns
        -------
        |Quantity|
        """
        raise NotImplementedError

    @property
    def number_density(self) -> u.Quantity[u.m**-3]:
        r"""
        The |normalization constant| for number density, :math:`n_⭑`.

        Returns
        -------
        |Quantity|
        """
        return self._quantities[_number_density]

    @property
    def pressure(self) -> u.Quantity[u.Pa]:
        r"""
        The |normalization constant| for pressure,
        :math:`p_⭑ ≡ \frac{B_⭑^2}{μ_0}`.

        Returns
        -------
        |Quantity|
        """
        return self.magnetic_field**2 / μ0

    @property
    def resistivity(self) -> u.Quantity[u.ohm * u.m]:
        r"""
        The |normalization constant| for resistivity.

        .. math::

           η_⭑ ≡ μ_0 L_⭑ V_⭑.

        Returns
        -------
        |Quantity|
        """
        return μ0 * self.length

    @property
    def temperature(self) -> u.Quantity[u.K]:
        r"""
        The temperature |normalization constant|,
        :math:`T_⭑ ≡ \frac{B_⭑^2}{k_B μ_0 n_⭑}`.

        Returns
        -------
        |Quantity|
        """
        return self.magnetic_field**2 / (k_B * μ0 * self.number_density)

    @property
    def thermal_conductivity(self) -> u.Quantity[u.W * u.K**-1 * u.m**-1]:
        r"""
        The thermal conduction |normalization constant|.

        Returns
        -------
        |Quantity|
        """
        raise NotImplementedError(
            "The thermal conductivity normalization has not yet been " "implemented."
        )

    @property
    def time(self) -> u.Quantity[u.s]:
        r"""
        The time |normalization constant|, :math:`t_⭑ ≡ \frac{L_⭑}{V_⭑}`.

        Returns
        -------
        |Quantity|
        """
        return self.length / self.velocity

    @property
    def velocity(self) -> u.Quantity[u.m / u.s]:
        r"""
        The velocity |normalization constant|, :math:`V_⭑ ≡ \frac{B_⭑}{\sqrt{μ_0 ρ_⭑}}`.

        Returns
        -------
        |Quantity|
        """
        return Alfven_speed(B=self.magnetic_field, density=self.mass_density)

    @property
    def volume(self) -> u.Quantity[u.m**3]:
        r"""
        The volume |normalization constant|, :math:`V_⭑ ≡ L_⭑^3`.

        Returns
        -------
        |Quantity|
        """

    @property
    def wavenumber(self) -> u.Quantity[u.m**-1]:
        r"""
        The wavenumber |normalization constant|, :math:`k_⭑ ≡ \frac{1}{L_⭑}`.

        Returns
        -------
        |Quantity|
        """
        return 1 / self.length
