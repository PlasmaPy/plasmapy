"""Classes to represent particle interactions and reactions."""

__all__ = ["AbstractParticleInteraction"]

from abc import ABC, abstractmethod
from collections.abc import Callable

import astropy.units as u
import numpy as np

from plasmapy.particles import Particle
from plasmapy.particles.particle_collections import ParticleList, ParticleListLike
from plasmapy.utils.decorators import validate_quantities


def _baryon_number(particles: ParticleList) -> int:
    """
    Find the total number of baryons minus the number of
    antibaryons in a list of particles.

    Parameters
    ----------
    particles : object
    """
    return sum(particle.baryon_number for particle in particles)


def _nucleon_number(particles: ParticleList) -> int:
    total_nucleon_number = 0
    for particle in particles:
        particle: Particle
        if particle.isotope:
            total_nucleon_number = particle.atomic_number + particle.mass_number


def _charge_number_excluding_bound_electrons(particles: ParticleList) -> int:
    """
    Find the total charge number in a list of nucleons
    (excluding bound electrons) and other particles.
    """
    total_charge = 0
    for particle in particles:
        if particle.isotope:
            total_charge += particle.atomic_number
        elif not particle.element:
            total_charge += particle.charge_number
    return total_charge


class AbstractParticleInteraction(ABC):
    """..."""

    def __init__(self, reactants: ParticleList, products: ParticleList):
        self.reactants = reactants
        self.products = products
        self.validate_interaction()

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    @property
    def reactants(self) -> ParticleList:
        """
        The reactants to the particle interaction.

        Returns
        -------
        `~plasmapy.particles.particle_collection.ParticleList`
        """
        return self._reactants

    @reactants.setter
    def reactants(self, reactants_: ParticleListLike) -> None:
        self._reactants = ParticleList(reactants_)

    @property
    def products(self) -> ParticleList:
        """
        The products from the particle interaction.

        Returns
        -------
        `~plasmapy.particles.particle_collection.ParticleList`
        """
        return self._products

    @products.setter
    def products(self, products_: ParticleListLike) -> None:
        self._products = ParticleList(products_)

    @abstractmethod
    def validate_interaction(self) -> None:
        """
        If the interaction violates any required conservation laws,
        raise a `~plasmapy.particles.exceptions.InvalidReactionError`.
        """

    def validate_conservation_law(
        self,
        conserved_quantity: Callable[[ParticleList], int],
    ):
        amount_for_reactants = conserved_quantity(self.reactants)
        amount_for_products = conserved_quantity(self.products)
        if amount_for_reactants != amount_for_products:
            name = conserved_quantity.__name__.replace("_", " ").strip()
            raise


class NuclearReaction(AbstractParticleInteraction):
    """..."""

    def validate_interaction(self) -> None:
        """..."""

    @validate_quantities
    def energy(self) -> u.Quantity[u.J]:
        """Return the mass energy of the nuclear reaction."""
        return np.sum(self.reactants.mass_energy) - np.sum(self.products.mass_energy)
