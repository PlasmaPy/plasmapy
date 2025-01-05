"""Exceptions and warnings."""

__all__ = [
    "PlasmaPyError",
    "PhysicsError",
    "InvalidRomanNumeralError",
    "OutOfRangeError",
    "RelativityError",
    "RomanError",
    "PlasmaPyWarning",
    "CouplingWarning",
    "PhysicsWarning",
    "PlasmaPyDeprecationWarning",
    "PlasmaPyFutureWarning",
    "RelativityWarning",
]

# ------------------------------------------------------------------------------
#   Exceptions
# ------------------------------------------------------------------------------


class PlasmaPyError(Exception):
    """
    Base class of PlasmaPy custom errors.

    All custom exceptions raised by PlasmaPy should inherit from this
    class and be defined in this module.
    """


class PhysicsError(PlasmaPyError, ValueError):
    """
    The base exception for physics-related errors.
    """


class RomanError(PlasmaPyError):
    """A base exception for errors from `plasmapy.utils.roman`."""


# ^^^^^^^^^^^^ Base Exceptions should be defined above this comment ^^^^^^^^^^^^


class RelativityError(PhysicsError):
    """
    An exception for speeds greater than the speed of light.
    """


class OutOfRangeError(RomanError):
    """
    An exception to be raised for integers that outside of the range
    that can be converted to Roman numerals.
    """


class InvalidRomanNumeralError(RomanError):
    """
    An exception to be raised when the input is not a valid Roman
    numeral.
    """


# ------------------------------------------------------------------------------
#   Warnings
# ------------------------------------------------------------------------------


class PlasmaPyWarning(Warning):
    """
    Base class of PlasmaPy custom warnings.

    All PlasmaPy custom warnings should inherit from this class and be
    defined in this module.

    Warnings should be issued using `warnings.warn`, which will not break
    execution if unhandled.
    """


class PhysicsWarning(PlasmaPyWarning):
    """The base warning for warnings related to non-physical situations."""


# ^^^^^^^^^^^^^ Base Warnings should be defined above this comment ^^^^^^^^^^^^^


class RelativityWarning(PhysicsWarning):
    """
    A warning for when relativistic velocities are being used in or are
    returned by non-relativistic functionality.
    """


class CouplingWarning(PhysicsWarning):
    """
    A warning for functions that rely on a particular coupling regime to
    be valid.
    """


class PlasmaPyDeprecationWarning(PlasmaPyWarning, DeprecationWarning):
    """
    A warning for deprecated features when the warning is intended for
    other Python developers.
    """


class PlasmaPyFutureWarning(PlasmaPyWarning, FutureWarning):
    """
    A warning for deprecated features when the warning is intended for
    end users of PlasmaPy.
    """
