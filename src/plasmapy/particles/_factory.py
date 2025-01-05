"""Contains a factory function for creating particles."""

__all__: list[str] = []

import contextlib
from collections.abc import Callable, Sequence
from numbers import Integral, Real

import astropy.units as u
from astropy.constants import m_e

from plasmapy.particles.exceptions import ChargeError, InvalidParticleError
from plasmapy.particles.particle_class import (
    CustomParticle,
    Particle,
    ParticleLike,
)
from plasmapy.particles.particle_collections import ParticleList


def _generate_particle_factory_error_message(
    args: ParticleLike | u.Quantity | CustomParticle | Sequence[ParticleLike],
    kwargs: dict[str, object],
) -> str:
    """Compose an error message for invalid particles."""

    errmsg = "Unable to create a particle from: "

    if args:
        errmsg += repr(args)
        if kwargs:
            errmsg += " and "

    if kwargs:
        errmsg += repr(kwargs)

    errmsg += (
        ". For information on creating particles, see: "
        "https://docs.plasmapy.org/en/stable/api/plasmapy.particles.ParticleLike.html"
    )

    return errmsg


def _make_custom_particle_with_real_charge_number(
    arg: ParticleLike,
    *,
    mass_numb: int | None = None,
    symbol: str | None = None,
    Z: float | None = None,
) -> CustomParticle:
    """
    Create a |CustomParticle| for mean or composite ions.

    This function is intended to produce |CustomParticle| instances
    provided a string representing an element or isotope without charge
    information (e.g., ``"He"`` or ``"He-4"`` but not ``"He-4 2+"``)
    along with a charge number that is a real number but not an integer.

    Parameters
    ----------
    arg : |particle-like|
        An appropriate first argument to |Particle|.

    mass_numb : real number, optional
        The mass number of an isotope.

    symbol : str, optional
        The symbol of the particle.

    Z : real number, optional
        The charge number.

    Raises
    ------
    |InvalidParticleError|
        If the |CustomParticle| cannot be created.

    Examples
    --------
    >>> _make_custom_particle_with_real_charge_number("He-4", Z=1.5)
    CustomParticle(mass=6.64511...e-27 kg, charge=2.40326...e-19 C)
    >>> _make_custom_particle_with_real_charge_number("He", Z=1.5, mass_numb=4)
    CustomParticle(mass=6.64511...e-27 kg, charge=2.40326...e-19 C)
    """

    if not isinstance(Z, Real | u.Quantity) and Z is not None:
        raise ChargeError("The charge number must be a real number.")

    base_particle = Particle(arg, mass_numb=mass_numb, Z=0)

    if not base_particle.is_category(require="element", exclude="ion"):
        # Add tests if this function becomes part of public API
        raise InvalidParticleError("Cannot create CustomParticle.")

    if Z is not None and Z > base_particle.atomic_number:
        raise ChargeError("The charge number cannot exceed the atomic number.")

    mass = base_particle.mass - m_e * Z
    return CustomParticle(mass=mass, Z=Z, symbol=symbol)


_particle_constructors: tuple[
    type
    | Callable[
        [ParticleLike | u.Quantity | CustomParticle | Sequence[ParticleLike]],
        object,
    ],
    ...,
] = (
    Particle,
    CustomParticle,
    CustomParticle._from_quantities,  # noqa: SLF001
    ParticleList,
    _make_custom_particle_with_real_charge_number,
)

_particle_types: tuple[type, ...] = (Particle, CustomParticle, ParticleList)


def _physical_particle_factory(
    *args: ParticleLike | u.Quantity | CustomParticle | Sequence[ParticleLike],
    **kwargs: ParticleLike | u.Quantity | None,
) -> Particle | CustomParticle | ParticleList:
    """
    Return a representation of one or more physical particles.

    This function will select the appropriate type among |Particle|,
    |CustomParticle|, and |ParticleList|.

    .. caution::

       If |Quantity| instances are provided to this function as
       positional arguments, then they must presently be in the order
       expected by |CustomParticle|.

    Parameters
    ----------
    *args
        Positional arguments to be supplied to |Particle|,
        |CustomParticle|, or |ParticleList|.

    **kwargs
        Keyword arguments to be supplied to |Particle|,
        |CustomParticle|, or |ParticleList|.

    Returns
    -------
    |Particle|, |CustomParticle|, or |ParticleList|

    Raises
    ------
    `InvalidParticleError`
        If an appropriate particle could not be constructed.

    `TypeError`
        If no positional arguments and no keyword arguments were
        provided.

    See Also
    --------
    ~plasmapy.particles.particle_class.Particle
    ~plasmapy.particles.particle_class.CustomParticle
    ~plasmapy.particles.particle_class.ParticleList

    Notes
    -----
    If ``Z`` or ``mass_numb`` is provided as keyword arguments but are
    equal to `None`, then they will not be provided to any of the calls.
    This is to allow |CustomParticle| instances to be created.

    Examples
    --------
    >>> from plasmapy.particles._factory import _physical_particle_factory
    >>> import astropy.units as u
    >>> _physical_particle_factory("p+")
    Particle("p+")
    >>> _physical_particle_factory(mass=9e-26 * u.kg, charge=8e20 * u.C)
    CustomParticle(mass=9e-26 kg, charge=8e+20 C)
    >>> _physical_particle_factory(["p+", "e-"])
    ParticleList(['p+', 'e-'])
    """

    # We need to remove Z and mass_numb from kwargs when they are `None`
    # because they are not allowed as arguments to `CustomParticle`, and
    # are not needed in kwargs if they are their default values. Note
    # that this affects `not kwargs` below.

    for parameter in ("Z", "mass_numb"):
        if parameter in kwargs and kwargs[parameter] is None:
            kwargs.pop(parameter)

    if len(args) == 1 and not kwargs and isinstance(args[0], _particle_types):
        return args[0]  # type: ignore[return-value]

    if not args and not kwargs:
        raise TypeError("Particle information has not been provided.")

    for constructor in _particle_constructors:
        with contextlib.suppress(ChargeError, InvalidParticleError, TypeError):
            return constructor(*args, **kwargs)  # type: ignore[return-value]

    if args and not isinstance(args[0], str | Integral | u.Quantity):
        raise TypeError("Invalid type for particle.")

    raise InvalidParticleError(_generate_particle_factory_error_message(args, kwargs))
