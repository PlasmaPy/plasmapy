"""Exceptions and warnings."""

__all__ = [
    "ParticleError",
    "ParticleWarning",
    "ChargeError",
    "InvalidElementError",
    "InvalidIonError",
    "InvalidIsotopeError",
    "InvalidParticleError",
    "MissingParticleDataError",
    "MissingParticleDataWarning",
    "UnexpectedParticleError",
]

from plasmapy.utils.exceptions import PlasmaPyError, PlasmaPyWarning


class ParticleError(PlasmaPyError):
    """Base exception for errors in the `~plasmapy.particles` subpackage."""


class MissingParticleDataError(ParticleError):
    """
    An exception for missing atomic or particle data in the
    `~plasmapy.particles` subpackage.
    """


class ChargeError(ParticleError):
    """An exception for incorrect or missing charge information."""


class UnexpectedParticleError(ParticleError):
    """An exception for when a particle is not of the expected category."""


class InvalidIonError(UnexpectedParticleError):
    """
    An exception for when an argument is a valid particle but not a
    valid ion.
    """


class InvalidIsotopeError(UnexpectedParticleError):
    """
    An exception for when an argument is a valid particle but not a
    valid isotope.
    """


class InvalidElementError(UnexpectedParticleError):
    """
    An exception for when an argument is a valid particle is not a
    valid element.
    """


class InvalidParticleError(ParticleError):
    """An exception for when a particle is invalid."""


class ParticleWarning(PlasmaPyWarning):
    """The base warning for the `~plasmapy.particles` subpackage."""


class MissingParticleDataWarning(ParticleWarning):
    """Warning for use when atomic or particle data is missing."""
