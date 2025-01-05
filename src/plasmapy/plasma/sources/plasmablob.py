"""
Defines the core Plasma class used by PlasmaPy to represent plasma properties.

.. attention::

   |expect-api-changes|
"""

__all__ = ["PlasmaBlob"]

import warnings

import astropy.units as u

from plasmapy.formulary import coupling_parameter, quantum_theta
from plasmapy.formulary.misc import _grab_charge
from plasmapy.particles import particle_mass
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.plasma.plasma_base import GenericPlasma
from plasmapy.utils import code_repr
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import CouplingWarning


class PlasmaBlob(GenericPlasma):
    """
    Class for describing and calculating plasma parameters without
    spatial/temporal description.
    """

    @validate_quantities(T_e=u.K, n_e=u.m**-3)
    def __init__(self, T_e, n_e, Z=None, particle: ParticleLike = "p+") -> None:
        """
        Initialize plasma parameters.
        The most basic description is composition (ion), temperature,
        density, and ionization.
        """
        self.T_e = T_e
        self.n_e = n_e
        self.particle = particle
        self.Z = _grab_charge(particle, Z)
        # extract mass from particle
        self.ionMass = particle_mass(self.particle)

    def __str__(self) -> str:
        """
        Fetch regimes for easy printing.

        Examples
        --------
        >>> print(PlasmaBlob(1e4 * u.K, 1e20 / u.m**3, particle="p+"))
        PlasmaBlob(T_e=10000.0*u.K, n_e=1e+20*u.m**-3, particle='p+', Z=1)
        Intermediate coupling regime: Gamma = 0.01250283...
        Thermal kinetic energy dominant: Theta = 109690.5...

        """
        return self.__repr__() + "\n" + "\n".join(self.regimes())

    def __repr__(self) -> str:
        """
        Return a string representation of this instance.

        Returns
        -------
        str

        Examples
        --------
        >>> import astropy.units as u
        >>> PlasmaBlob(1e4 * u.K, 1e20 / u.m**3, particle="p+")
        PlasmaBlob(T_e=10000.0*u.K, n_e=1e+20*u.m**-3, particle='p+', Z=1)
        """
        argument_dict = {
            "T_e": self.T_e,
            "n_e": self.n_e,
            "particle": self.particle,
            "Z": self.Z,
        }

        return code_repr.call_string(PlasmaBlob, (), argument_dict)

    @property
    def electron_temperature(self):
        return self.T_e

    @property
    def electron_density(self):
        return self.n_e

    @property
    def ionization(self):
        return self.Z

    @property
    def composition(self):
        return self.particle

    def regimes(self):
        """
        Generate a comprehensive description of the plasma regimes
        based on plasma properties and consequent plasma parameters.
        """
        coupling = self.coupling()
        quantum_theta = self.quantum_theta()

        if coupling <= 0.01:
            coupling_str = f"Weakly coupled regime: Gamma = {coupling}."
        elif coupling >= 100:
            coupling_str = f"Strongly coupled regime: Gamma = {coupling}."
        else:
            coupling_str = f"Intermediate coupling regime: Gamma = {coupling}."

        if quantum_theta <= 0.01:
            quantum_theta_str = (
                f"Fermi quantum energy dominant: Theta = {quantum_theta}"
            )
        elif quantum_theta >= 100:
            quantum_theta_str = (
                f"Thermal kinetic energy dominant: Theta = {quantum_theta}"
            )
        else:
            quantum_theta_str = (
                f"Both Fermi and thermal energy important: Theta = {quantum_theta}"
            )

        return [coupling_str, quantum_theta_str]

    def coupling(self):
        """
        Ion-ion coupling parameter to determine if quantum/coupling effects
        are important. This compares Coulomb potential energy to thermal
        kinetic energy.
        """
        couple = coupling_parameter(
            self.T_e, self.n_e, (self.particle, self.particle), self.Z
        )
        if couple < 0.01:
            warnings.warn(
                f"Coupling parameter is {couple}, you might have strong coupling effects",
                CouplingWarning,
            )

        return couple

    def quantum_theta(self):
        """
        Quantum theta parameter, which compares Fermi kinetic energy to
        thermal kinetic energy to check if quantum effects are important.
        """
        return quantum_theta(self.T_e, self.n_e)

    @classmethod
    def is_datasource_for(cls, **kwargs) -> bool:
        return "T_e" in kwargs and "n_e" in kwargs
