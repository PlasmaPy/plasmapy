"""Representations of particle interactions."""

from typing import List, Tuple, Union

from plasmapy.particles.exceptions import InvalidParticleError
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.particles.particle_collections import ParticleList

ParticleListLike = Union[
    ParticleLike,
    ParticleList,
    List,
    Tuple,
]


def _make_particle_list(x: ParticleListLike):

    if isinstance(x, ParticleList):
        return x

    try:
        return ParticleList(x)
    except (TypeError, InvalidParticleError):
        pass
    else:
        return ParticleList([x])


class ParticleInteraction:
    """A class to represent particle interactions."""

    pass


class Reaction(ParticleInteraction):
    """A class to represent identity-changing particle interactions."""

    def __init__(self, *, reactants: ParticleListLike, products: ParticleListLike):
        self.reactants = reactants
        self.products = products

    @property
    def reactants(self):
        return self._reactants

    @reactants.setter
    def reactants(self, reactants_):
        self._reactants = _make_particle_list(reactants_)

    @property
    def products(self):
        return self._products

    @products.setter
    def products(self, products_):
        self._products = _make_particle_list(products_)


class NuclearReaction(Reaction):
    """A class to represent nuclear reactions."""

    pass


class FissionReaction(NuclearReaction):
    """A class to represent nuclear fission reactions."""

    pass


class FusionReaction(NuclearReaction):
    """A class to represent nuclear fusion reactions."""

    pass


class NonNuclearReaction(Reaction):
    """A class to represent non-nuclear reactions."""

    pass


class IonizationReaction(Reaction):
    """
    A class to represent ionization reactions of the form

    .. math::

       ...
    """

    pass


class ElectronIonizationReaction(Reaction):
    """
    A class to represent electron ionization reactions of the form

    .. math::

       ...

    Notes
    -----
    Electron ionization is also known as electron impact ionization and
    electron bombardment ionization.
    """

    pass


class PhotoionizationReaction(IonizationReaction):
    pass


class RecombinationReaction(Reaction):
    """A class to represent recombination reactions."""

    pass


class DielectronicRecombinationReaction(RecombinationReaction):
    """A class to represent dielectronic recombination reactions."""

    pass


# Types of reactions
# * Nuclear
#   - Fission
#   - Fusion
# * Recombination
#   -
# * Ionization
#    - ElectronIonization (aka ElectronImpactIonization, ElectronBombar


class Collision(ParticleInteraction):
    pass


class CoulombCollision(ParticleInteraction):
    pass
