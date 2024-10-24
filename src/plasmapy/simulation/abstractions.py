"""
Abstract classes for numerical simulations.

.. attention::

   |expect-api-changes|
"""

__all__ = [
    "AbstractNormalizations",
    "AbstractSimulation",
    "AbstractTimeDependentSimulation",
]

from abc import ABC, abstractmethod

import astropy.units as u


class AbstractSimulation(ABC):
    """
    A prototype abstract interface for numerical simulations.

    .. attention::

       |expect-api-changes|
    """

    @abstractmethod
    def summarize(self):
        """
        Print a summary of the simulation parameters and status.
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """Prepare the simulation to be run."""
        ...

    @abstractmethod
    def simulate(self):
        """Perform the simulation."""
        ...

    @abstractmethod
    def finalize(self) -> None:
        """Perform the steps to close the simulation."""
        ...


class AbstractTimeDependentSimulation(AbstractSimulation):
    """
    A prototype abstract interface for time-dependent numerical simulations.

    .. warning::
        This interface is unstable and is subject to change.
    """


class AbstractNormalizations(ABC):
    """
    An abstract base class to represent the |normalization constants|
    for systems of equations describing plasmas.

    For a list of physical types, see `astropy.units.physical`.

    .. warning::

        This interface is unstable and is subject to change.
    """

    @property
    @abstractmethod
    def acceleration(self) -> u.Quantity[u.m / u.s**2]:
        r"""The |normalization constant| for acceleration, :math:`a_⋆`."""

    @property
    @abstractmethod
    def area(self) -> u.Quantity[u.m**2]:
        r"""The |normalization constant| for area, :math:`A_⋆`."""

    @property
    @abstractmethod
    def current_density(self) -> u.Quantity[u.A / u.m**2]:
        """
        The |normalization constant| for :wikipedia:`current
        density`, :math:`J_⋆`.
        """

    @property
    @abstractmethod
    def diffusivity(self) -> u.Quantity[u.m**2 / u.s]:
        """
        The |normalization constant| for :wikipedia`diffusivity`,
        :math:`D_⋆`.

        .. note::

           Use this normalization for :wikipedia:`kinematic viscosity`,
           :wikipedia:`magnetic diffusivity`, :wikipedia:`mass
           diffusivity`, and :wikipedia:`thermal diffusivity`.
        """

    @property
    @abstractmethod
    def dynamic_viscosity(self) -> u.Quantity[u.Pa * u.s]:
        """
        The |normalization constant| for :wikipedia:`dynamic
        viscosity`, :math:`μ_⋆`.

        .. note::

           For `kinematic viscosity`_, use
           ~`plasmapy.simulation.abstractions.AbstractNormalizations.diffusivity`.

        .. _kinematic viscosity: https://en.wikipedia.org/wiki/Viscosity#Kinematic_viscosity
        """

    @property
    @abstractmethod
    def electric_field(self) -> u.Quantity[u.V / u.m]:
        """
        The |normalization constant| for :wikipedia`electric
        field`, :math:`E_⋆`.
        """

    @property
    @abstractmethod
    def energy(self) -> u.Quantity[u.J]:
        """The |normalization constant| for energy, :math:`e_⋆`."""

    @property
    @abstractmethod
    def frequency(self) -> u.Quantity[u.s**-1]:
        """The |normalization constant| for frequency, :math:`f_⋆`."""

    @property
    @abstractmethod
    def heat_flux(self) -> u.Quantity[u.J * u.m**-2 * u.s**-1]:
        """
        The |normalization constant| for :wikipedia:`heat flux`,
        :math:`q_⋆`.

        .. todo::

           Check that this is the correct symbol

        .. note::

           For :wikipedia:`thermal diffusivity`, use the normalization
           constant for
           ~`plasmapy.simulation.abstractions.AbstractNormalizations.diffusivity`.
        """

    @property
    @abstractmethod
    def length(self) -> u.Quantity[u.m]:
        """The |normalization constant| for length, :math:`L_⋆`."""

    @property
    @abstractmethod
    def magnetic_field(self) -> u.Quantity[u.T]:
        """
        The |normalization constant| for :wikipedia:`magnetic field`,
        :math:`B_⋆`.
        """

    @property
    @abstractmethod
    def magnetic_flux(self) -> u.Quantity[u.T * u.m]:
        """
        The |normalization constant| for :wikipedia:`magnetic
        flux` or :wikipedia:`magnetic vector potential`, :math:`Φ_⋆`.

        .. todo::

            Check this this is an appropriate symbol.  Can't use A since
            it's overloaded already with acceleration and area.
        """

    @property
    @abstractmethod
    def mass(self) -> u.Quantity[u.kg]:
        """The |normalization constant| for mass, :math:`m_⋆`."""

    @property
    @abstractmethod
    def mass_density(self) -> u.Quantity[u.kg / u.m**3]:
        """The |normalization constant| for mass density, :math:`ρ_⋆`."""

    @property
    @abstractmethod
    def number_density(self) -> u.Quantity[u.m**-3]:
        """
        The |normalization constant| for :wikipedia:`number density`,
        :math:`n_⋆`.
        """

    @property
    @abstractmethod
    def pressure(self) -> u.Quantity[u.Pa]:
        """
        The |normalization constant| for :wikipedia:`pressure`,
        :math:`p_⋆`.
        """

    @property
    @abstractmethod
    def resistivity(self) -> u.Quantity[u.Ohm * u.m]:
        """
        The |normalization constant| for electrical resistivity,
        :math:`η_⋆`.
        """

    @property
    @abstractmethod
    def temperature(self) -> u.Quantity[u.K]:
        """
        The |normalization constant| for :term:`temperature` in
        units of kelvin, :math:`T_⋆`.
        """

    @property
    @abstractmethod
    def thermal_conductivity(self) -> u.Quantity[u.W / (u.K * u.m)]:
        """
        The |normalization constant| for :wikipedia:`thermal
        conductivity`, :math:`κ_⋆`.
        """

    @property
    @abstractmethod
    def time(self) -> u.Quantity[u.s]:
        """The |normalization constant| for time, :math:`t_⋆`."""

    @property
    @abstractmethod
    def velocity(self) -> u.Quantity[u.m / u.s]:
        """The |normalization constant| for velocity, :math:`v_⋆`."""

    @property
    @abstractmethod
    def volume(self) -> u.Quantity[u.m**3]:
        """The |normalization constant| for volume, :math:`V_⋆`."""

    @property
    @abstractmethod
    def wavenumber(self) -> u.Quantity[u.m**-1]:
        """
        The |normalization constant| for :wikipedia:`wavenumber`,
        :math:`k_⋆`.
        """
