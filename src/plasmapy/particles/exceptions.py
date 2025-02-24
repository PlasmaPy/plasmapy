"""Collection of exceptions and warnings for `plasmapy.particles`."""

__all__ = [
    "ParticleError",
    "ParticleWarning",
    "ChargeError",
    "ConservationError",
    "InvalidElementError",
    "InvalidIonError",
    "InvalidIsotopeError",
    "InvalidParticleError",
    "InvalidReactionError",
    "MissingParticleDataError",
    "MissingParticleDataWarning",
    "UnexpectedParticleError",
]

from plasmapy.utils.exceptions import PlasmaPyError, PlasmaPyWarning


class ParticleError(PlasmaPyError):
    """Base exception for errors in `plasmapy.particles`."""


class MissingParticleDataError(ParticleError):
    """Exception for missing atomic or particle data in `plasmapy.particles`."""


class ChargeError(ParticleError):
    """Exception for incorrect or missing charge information."""


class UnexpectedParticleError(ParticleError):
    """Exception for when a particle is not of the expected category."""


class InvalidIonError(UnexpectedParticleError):
    """
    Exception for when an argument is a valid particle but not a valid
    ion.
    """


class InvalidIsotopeError(UnexpectedParticleError):
    """
    Exception for when an argument is a valid particle but not a
    valid isotope.
    """


class InvalidElementError(UnexpectedParticleError):
    """
    Exception for when an argument is a valid particle is not a
    valid element.
    """


class InvalidParticleError(ParticleError):
    """Exception for when a particle is invalid."""


class InvalidReactionError(ParticleError):
    """Exception for when a reaction in invalid."""


class ConservationError(InvalidReactionError):
    """
    Exception for when a conserved property is not conserved, such as a
    violation of a conservation law.
    """


class ParticleWarning(PlasmaPyWarning):
    """Base warning for `plasmapy.particles`."""


class MissingParticleDataWarning(ParticleWarning):
    """Warning for use when atomic or particle data is missing."""
