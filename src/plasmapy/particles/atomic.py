"""Functions that retrieve or are related to elemental or isotopic data."""

__all__ = [
    "atomic_number",
    "charge_number",
    "common_isotopes",
    "electric_charge",
    "half_life",
    "ionic_levels",
    "isotopic_abundance",
    "is_stable",
    "known_isotopes",
    "mass_number",
    "particle_mass",
    "periodic_table_block",
    "periodic_table_category",
    "periodic_table_group",
    "periodic_table_period",
    "reduced_mass",
    "stable_isotopes",
    "standard_atomic_weight",
    "stopping_power",
]

from collections.abc import Callable, Iterable
from numbers import Integral
from typing import Any, Literal

import astropy.units as u
import h5py
import numpy as np
from scipy.interpolate import CubicSpline

from plasmapy.particles import _elements, _isotopes
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.exceptions import (
    ChargeError,
    InvalidElementError,
    InvalidIsotopeError,
    InvalidParticleError,
)
from plasmapy.particles.particle_class import Particle, ParticleLike
from plasmapy.particles.particle_collections import ParticleList
from plasmapy.particles.symbols import atomic_symbol
from plasmapy.utils.data.downloader import Downloader
from plasmapy.utils.decorators import validate_quantities

__all__.sort()


@particle_input
def atomic_number(element: ParticleLike) -> int:
    """
    Return the number of protons in an atom, isotope, or ion.

    Parameters
    ----------
    element : |atom-like|
        A string representing an element, isotope, or ion; or an
        instance of the `~plasmapy.particles.particle_class.Particle` class.

    Returns
    -------
    `int`
        The atomic number of an element.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidElementError`
        If the argument is a valid particle but not a valid element.

    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle.

    See Also
    --------
    ~plasmapy.particles.atomic.mass_number

    Examples
    --------
    >>> atomic_number("H")
    1
    >>> atomic_number("tritium")
    1
    >>> atomic_number("alpha")
    2
    >>> atomic_number("oganesson")
    118
    """
    return element.atomic_number  # type: ignore[union-attr]


@particle_input
def mass_number(isotope: ParticleLike) -> int:
    """Get the mass number (the number of protons and neutrons) of an
    isotope.

    Parameters
    ----------
    isotope : |atom-like|
        A string representing an isotope or a neutron; or an instance of
        the `plasmapy.particles.particle_class.Particle` class.

    Returns
    -------
    `int`
       The total number of protons plus neutrons in a nuclide.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle.

    `~plasmapy.particles.exceptions.InvalidIsotopeError`
        If the argument does not correspond to a valid isotope.

    See Also
    --------
    ~plasmapy.particles.atomic.atomic_number

    Examples
    --------
    >>> mass_number("H-1")
    1
    >>> mass_number("Pb-208")
    208
    >>> mass_number("tritium")
    3
    >>> mass_number("alpha")
    4
    """
    return isotope.mass_number  # type: ignore[union-attr]


@particle_input(exclude={"isotope", "ion"})
def standard_atomic_weight(element: ParticleLike) -> u.Quantity[u.kg]:
    """Return the standard (conventional) atomic weight of an element
    based on the relative abundances of isotopes in terrestrial
    environments.

    Parameters
    ----------
    element : |atom-like|
        A string representing an element or an integer representing an
        atomic number, or an instance of the Particle class.

    Returns
    -------
    `~astropy.units.Quantity`
        The standard atomic weight of an element based on values from
        NIST.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidElementError`
        If the argument is a valid particle but not a valid element.

    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle.

    See Also
    --------
    ~plasmapy.particles.atomic.particle_mass

    Notes
    -----
    Standard atomic weight data are most readily available for the
    terrestrial environment, so this function may not be wholly
    appropriate for space and astrophysical environments.

    The relative abundances of different isotopes of an element
    sometimes vary naturally in different locations within the
    terrestrial environment.  The CIAAW provides ranges for these
    element, which include H, Li, B, C, N, O, Mg, Si, S, Cl, Br, Tl.
    This function provides a single value from the CIAWW 2015 standard
    values when a single value is given, and the lower accuracy
    conventional value given by Meija et al. (2013,
    doi:10.1515/pac-2015-0305) for the elements where a range is
    given.

    Examples
    --------
    >>> standard_atomic_weight("H")
    <Quantity 1.6738233e-27 kg>
    >>> standard_atomic_weight("lead")
    <Quantity 3.440636e-25 kg>
    """
    # TODO: Put in ReST links into above docstring
    return element.standard_atomic_weight  # type: ignore[union-attr]


@particle_input(exclude={"neutrino", "antineutrino"})
def particle_mass(
    particle: ParticleLike,
    *,
    mass_numb: int | None = None,
    Z: float | None = None,
) -> u.Quantity[u.kg]:
    """
    Return the mass of a particle.

    Parameters
    ----------
    particle : |particle-like|
        A string representing an element, isotope, ion, or special
        particle; an integer representing an atomic number; or a
        |Particle|.

    Returns
    -------
    `~astropy.units.Quantity`
        The mass of the particle.

    Other Parameters
    ----------------
    mass_numb : integer, |keyword-only|, optional
        The mass number of an isotope.

    Z : integer, |keyword-only|, optional
        The |charge number| of an ion or neutral atom.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle.

    `~plasmapy.particles.exceptions.MissingParticleDataError`
        If the standard atomic weight, the isotope mass, or the particle
        mass is not available.

    See Also
    --------
    ~plasmapy.particles.atomic.standard_atomic_weight

    Notes
    -----
    This function will return the ion mass for ions, the isotope mass
    for isotopes (when available), the standard atomic weight for
    elements (when available), or the mass of special particles, as
    appropriate.

    The masses of neutrinos are not available because primarily upper
    limits are presently known.
    """
    return particle.mass  # type: ignore[union-attr]


@particle_input
def isotopic_abundance(isotope: ParticleLike, mass_numb: int | None = None) -> float:
    """
    Return the isotopic abundances if known, and otherwise zero.

    Parameters
    ----------
    isotope : |atom-like|
        A string representing an element or isotope, or an integer
        representing the atomic number of an element.

    Returns
    -------
    `float`
        The relative isotopic abundance in the terrestrial environment.

    Other Parameters
    ----------------
    mass_numb : integer, |keyword-only|, optional
        The mass number of an isotope.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidIsotopeError`
        If the argument is a valid particle but not a valid isotope.

    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle
        or contradictory information is provided.

    Notes
    -----
    Isotopic composition data are most readily available for the
    terrestrial environment, so this function may not be wholly
    appropriate for space and astrophysical applications.

    The data retrieved from this routine are those recommended by NIST
    as of 2017.

    Examples
    --------
    >>> isotopic_abundance("Pb-208")
    0.524
    >>> isotopic_abundance("hydrogen", 1)
    0.999885
    """
    return isotope.isotopic_abundance  # type: ignore[union-attr]


@particle_input(any_of={"charged", "uncharged"})
def charge_number(particle: ParticleLike) -> int:
    """Return the charge number of a particle.

    Parameters
    ----------
    particle : |particle-like|
        String representing a particle.

    Returns
    -------
    `int`
        The charge as a multiple of the elementary charge.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle
        or contradictory information is provided.

    `~plasmapy.particles.exceptions.ChargeError`
        If charge information for the particle is not available.

    `~plasmapy.particles.exceptions.ParticleWarning`
        If the input represents an ion with a charge number that is
        less than or equal to ``-3``, which is unlikely to occur in
        nature.

    Notes
    -----
    This function supports two formats for charge number information.

    The first format is a string that has information for the element
    or isotope at the beginning, a space in between, and the charge
    number information in the form of an integer followed by a plus or
    minus sign, or a plus or minus sign followed by an integer.

    The second format is a string containing element information at
    the beginning, following by one or more plus or minus signs.

    Examples
    --------
    >>> charge_number("Fe-56 2+")
    2
    >>> charge_number("He -2")
    -2
    >>> charge_number("H+")
    1
    >>> charge_number("N-14++")
    2
    """
    return particle.charge_number  # type: ignore[union-attr, return-value]


@particle_input(any_of={"charged", "uncharged"})
def electric_charge(particle: ParticleLike) -> u.Quantity[u.C]:
    """
    Return the electric charge (in coulombs) of a particle.

    Parameters
    ----------
    particle : |particle-like|
        String representing an element or isotope followed by integer
        charge information.

    Returns
    -------
    `~astropy.units.Quantity`
        The electric charge in coulombs.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle
        or contradictory information is provided.

    `~plasmapy.particles.exceptions.ChargeError`
        If charge information for the particle is not available.

    `~plasmapy.particles.exceptions.ParticleWarning`
        If the input represents an ion with a charge number that is
        below ``-3``.

    Notes
    -----
    This function supports two formats for charge number information.

    The first format is a string that has information for the element
    or isotope at the beginning, a space in between, and the charge
    number information in the form of an integer followed by a plus or
    minus sign, or a plus or minus sign followed by an integer.

    The second format is a string containing element information at
    the beginning, following by one or more plus or minus signs.

    This function returns -1.6021766208e-19 C for electrons and
    1.6021766208e-19 C for positrons.

    Examples
    --------
    >>> electric_charge("p+")
    <class 'astropy.constants.codata...'> name='Electron charge' ...>
    >>> electric_charge("H-")
    <Quantity -1.60217662e-19 C>
    """
    return particle.charge  # type: ignore[union-attr]


@particle_input
def is_stable(particle: ParticleLike, mass_numb: int | None = None) -> bool:
    """
    Return `True` for stable isotopes and particles and `False` for
    unstable isotopes.

    Parameters
    ----------
    particle : |particle-like|
        A string representing an isotope or particle, or an integer
        representing an atomic number.

    Returns
    -------
    `bool`
        `True` if the isotope is stable, `False` if it is unstable.

    Other Parameters
    ----------------
    mass_numb : integer, |keyword-only|, optional
        The mass number of an isotope.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidIsotopeError`
        If the arguments correspond to a valid element but not a
        valid isotope.

    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the arguments do not correspond to a valid particle.

    `~plasmapy.particles.exceptions.MissingParticleDataError`
        If stability information is not available.

    Examples
    --------
    >>> is_stable("H-1")
    True
    >>> is_stable("tritium")
    False
    >>> is_stable("e-")
    True
    >>> is_stable("tau+")
    False
    """
    if particle.element and not particle.isotope:  # type: ignore[union-attr]
        raise InvalidIsotopeError(
            "The input to is_stable must be either an isotope or a special particle."
        )
    return particle.is_category("stable")  # type: ignore[union-attr]


@particle_input(any_of={"stable", "unstable", "isotope"})
def half_life(particle: ParticleLike, mass_numb: int | None = None) -> u.Quantity[u.s]:
    """
    Return the half-life in seconds for unstable isotopes and particles,
    and |inf| seconds for stable isotopes and particles.

    Parameters
    ----------
    particle : |particle-like|
        A string representing an isotope or particle, an integer
        representing an atomic number, or an instance of the |Particle|
        class.

    Returns
    -------
    `~astropy.units.Quantity`
        The half-life of the isotope or particle in units of seconds.

    Other Parameters
    ----------------
    mass_numb : integer, |keyword-only|, optional
        The mass number of an isotope.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle
        or contradictory information is provided.

    `~plasmapy.particles.exceptions.MissingParticleDataError`
        If no half-life data is available for the isotope.

    Notes
    -----
    Accurate half-life data is not known for all isotopes. Some isotopes
    may have upper or lower limits on the half-life, in which case this
    function will return a string with that information and issue a
    `~plasmapy.particles.exceptions.MissingParticleDataError`.  When no
    isotope information is available, then this function raises a
    `~plasmapy.particles.exceptions.MissingParticleDataError`.

    Examples
    --------
    >>> half_life("T")
    <Quantity 3.888e+08 s>
    >>> half_life("n")
    <Quantity 881.5 s>
    >>> half_life("H-1")
    <Quantity inf s>
    """
    return particle.half_life  # type: ignore[union-attr]


def known_isotopes(argument: ParticleLike | None = None) -> ParticleList:
    """
    Return a list of all known isotopes of an element, or a list of all
    known isotopes of every element if no input is provided.

    Parameters
    ----------
    argument : |atom-like|
        A string representing an element, isotope, or ion or an
        integer representing an atomic number

    Returns
    -------
    |ParticleList|
        List of all the isotopes of an element that have been
        discovered, sorted from low to high mass number. If no argument
        is provided, then a list of all known isotopes of every element
        will be returned that is sorted first by atomic number and
        second by mass number.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidElementError`
        If the argument is a valid particle but not a valid element.

    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle.

    Notes
    -----
    This list returns both natural and artificially produced isotopes.

    See Also
    --------
    ~plasmapy.particles.atomic.common_isotopes :
        Returns isotopes with non-zero isotopic abundances.

    ~plasmapy.particles.atomic.stable_isotopes :
        Returns isotopes that are stable against radioactive decay.

    Examples
    --------
    >>> known_isotopes("H")
    ParticleList(['H-1', 'D', 'T', 'H-4', 'H-5', 'H-6', 'H-7'])
    >>> known_isotopes("helium 1+")
    ParticleList(['He-3', 'He-4', 'He-5', 'He-6', 'He-7', 'He-8', 'He-9', 'He-10'])
    >>> known_isotopes()[0:10]
    ParticleList(['H-1', 'D', 'T', 'H-4', 'H-5', 'H-6', 'H-7', 'He-3', 'He-4', 'He-5'])
    >>> len(known_isotopes())  # the number of known isotopes
    3352
    """

    # TODO: Allow Particle objects representing elements to be inputs

    def known_isotopes_for_element(argument_: ParticleLike) -> list[Particle]:
        element = atomic_symbol(argument_)
        isotopes = [
            isotope
            for isotope in _isotopes.data_about_isotopes
            if f"{element}-" in isotope and isotope[: len(element)] == element
        ]

        if element == "H":
            isotopes.insert(1, "D")
            isotopes.insert(2, "T")
        mass_numbers = [mass_number(isotope) for isotope in isotopes]
        return [
            mass_number
            for (isotope, mass_number) in sorted(
                zip(mass_numbers, isotopes, strict=False)
            )
        ]

    if argument is not None:
        try:
            element = atomic_symbol(argument)
            isotopes_list = known_isotopes_for_element(element)
        except InvalidElementError as ex:
            raise InvalidElementError(
                f"known_isotopes is unable to get isotopes from an input of: {argument}"
            ) from ex
        except InvalidParticleError as ex:
            raise InvalidParticleError("Invalid particle in known_isotopes.") from ex
    elif argument is None:
        isotopes_list = []
        for atomic_numb in range(1, len(_elements.data_about_elements) + 1):
            isotopes_list += known_isotopes_for_element(atomic_numb)

    return ParticleList(isotopes_list)


def common_isotopes(
    argument: ParticleLike | None = None, most_common_only: bool = False
) -> ParticleList:
    """
    Return a list of isotopes of an element with an isotopic abundances
    greater than zero, or if no input is provided, a list of all such
    isotopes for every element.

    Parameters
    ----------
    argument : |atom-like|, optional
        A string or integer representing an atomic number or element,
        or a string representing an isotope.

    most_common_only : `bool`
        If set to `True`, return only the most common isotope.

    Returns
    -------
    |ParticleList|
        List of all isotopes of an element with isotopic abundances
        greater than zero, sorted from most abundant to least
        abundant.  If no isotopes have isotopic abundances greater
        than zero, this function will return an empty list.  If no
        arguments are provided, then a list of all common isotopes of
        all elements will be provided that is sorted first by low to
        high atomic number and second by most abundant to least abundant
        isotope.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidElementError`
        If the argument is a valid particle but not a valid element.

    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle.

    Notes
    -----
    The isotopic abundances are based on the terrestrial environment
    and may not be appropriate for space and astrophysical applications.

    See Also
    --------
    ~plasmapy.particles.atomic.known_isotopes :
        Returns a list of isotopes that have been discovered.

    ~plasmapy.particles.atomic.stable_isotopes :
        Returns isotopes that are stable against radioactive decay.

    ~plasmapy.particles.atomic.isotopic_abundance :
        Returns the relative isotopic abundance.

    Examples
    --------
    >>> common_isotopes("H")
    ParticleList(['H-1', 'D'])
    >>> common_isotopes(44)
    ParticleList(['Ru-102', 'Ru-104', 'Ru-101', 'Ru-99', 'Ru-100', 'Ru-96', 'Ru-98'])
    >>> common_isotopes("beryllium 2+")
    ParticleList(['Be-9'])
    >>> common_isotopes("Fe")
    ParticleList(['Fe-56', 'Fe-54', 'Fe-57', 'Fe-58'])
    >>> common_isotopes("Fe", most_common_only=True)
    ParticleList(['Fe-56'])
    >>> common_isotopes()[0:7]
    ParticleList(['H-1', 'D', 'He-4', 'He-3', 'Li-7', 'Li-6', 'Be-9'])
    """

    # TODO: Allow Particle objects representing elements to be inputs

    def common_isotopes_for_element(
        argument: ParticleLike, most_common_only: bool | None
    ) -> list[Particle]:
        isotopes = known_isotopes(argument)

        common_isotopes_ = [
            isotope
            for isotope in isotopes
            if "abundance" in _isotopes.data_about_isotopes[isotope.isotope]
        ]

        isotopic_abundances = [
            _isotopes.data_about_isotopes[isotope.isotope]["abundance"]
            for isotope in common_isotopes_
        ]

        sorted_isotopes = [
            iso_comp
            for (isotope, iso_comp) in sorted(
                zip(isotopic_abundances, common_isotopes_, strict=False)
            )
        ]

        sorted_isotopes.reverse()

        if most_common_only and len(sorted_isotopes) > 1:
            sorted_isotopes = sorted_isotopes[:1]

        return sorted_isotopes

    if argument is not None:
        try:
            element = atomic_symbol(argument)
            isotopes_list = common_isotopes_for_element(element, most_common_only)
        except InvalidParticleError as ex:
            raise InvalidParticleError("Invalid particle") from ex
        except InvalidElementError as ex:
            raise InvalidElementError(
                "common_isotopes is unable to get isotopes "
                f"from an input of: {argument}"
            ) from ex

    elif argument is None:
        isotopes_list = []
        for atomic_numb in range(1, 119):
            isotopes_list += common_isotopes_for_element(atomic_numb, most_common_only)

    return ParticleList(isotopes_list)


def stable_isotopes(
    argument: ParticleLike | None = None, unstable: bool = False
) -> ParticleList:
    """
    Return a list of all stable isotopes of an element, or if no input is
    provided, a list of all such isotopes for every element.

    Parameters
    ----------
    argument : |atom-like|
        A string or integer representing an atomic number or element,
        or a string representing an isotope.

    unstable : `bool`
        If set to `True`, this function will return a list of the
        unstable isotopes instead of the stable isotopes.

    Returns
    -------
    |ParticleList|
        List of all stable isotopes of an element, sorted from low to
        high mass number.  If an element has no stable isotopes, this
        function returns an empty list.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidElementError`
        If the argument is a valid particle but not a valid element.

    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the argument does not correspond to a valid particle.

    Notes
    -----
    There are 254 isotopes for which no radioactive decay has been
    observed.  It is possible that some isotopes will be discovered to
    be unstable but with extremely long half-lives.  For example,
    bismuth-209 was recently discovered to have a half-life of about
    :math:`1.9 × 10^{19}` years.  However, such isotopes can be regarded
    as virtually stable for most applications.

    See Also
    --------
    ~plasmapy.particles.atomic.known_isotopes :
        Returns a list of isotopes that have been discovered.

    ~plasmapy.particles.atomic.common_isotopes :
        Returns isotopes with non-zero isotopic abundances.

    Examples
    --------
    >>> stable_isotopes("H")
    ParticleList(['H-1', 'D'])
    >>> stable_isotopes(44)
    ParticleList(['Ru-96', 'Ru-98', 'Ru-99', 'Ru-100', 'Ru-101', 'Ru-102', 'Ru-104'])
    >>> stable_isotopes("beryllium")
    ParticleList(['Be-9'])
    >>> stable_isotopes("Pb-209")
    ParticleList(['Pb-204', 'Pb-206', 'Pb-207', 'Pb-208'])
    >>> stable_isotopes(118)
    ParticleList([])

    Find unstable isotopes using the ``unstable`` keyword.

    >>> stable_isotopes("He", unstable=True)
    ParticleList(['He-5', 'He-6', 'He-7', 'He-8', 'He-9', 'He-10'])
    """

    # TODO: Allow Particle objects representing elements to be inputs

    def stable_isotopes_for_element(
        argument: str | int, stable_only: bool | None
    ) -> list[Particle]:
        KnownIsotopes = known_isotopes(argument)
        return [
            isotope
            for isotope in KnownIsotopes
            if _isotopes.data_about_isotopes[isotope.isotope]["stable"] == stable_only
        ]

    if argument is not None:
        try:
            element = atomic_symbol(argument)
            isotopes_list = stable_isotopes_for_element(element, not unstable)
        except InvalidParticleError as ex:
            raise InvalidParticleError("Invalid particle in stable_isotopes") from ex
        except InvalidElementError as ex:
            raise InvalidElementError(
                "stable_isotopes is unable to get isotopes "
                f"from an input of: {argument}"
            ) from ex
    elif argument is None:
        isotopes_list = []
        for atomic_numb in range(1, 119):
            isotopes_list += stable_isotopes_for_element(atomic_numb, not unstable)

    return ParticleList(isotopes_list)


@particle_input
@validate_quantities  # type: ignore[misc]
def reduced_mass(
    test_particle: ParticleLike,
    target_particle: ParticleLike,
) -> u.Quantity[u.kg]:
    r"""
    Find the :wikipedia:`reduced mass` between two particles.

    The reduced mass is given by:

    .. math::

        μ ≡ \frac{m_1 m_2}{m_1 + m_2}

    for two objects of mass :math:`m_1` and :math:`m_2`. The reduced
    mass is used to describe the effective mass in the
    :wikipedia:`two-body problem`.

    Parameters
    ----------
    test_particle, target_particle : |particle-like|
        The test particle as represented by a string, an integer
        representing atomic number, a `~plasmapy.particles.particle_class.Particle`
        object, or a `~astropy.units.Quantity` or
        `~astropy.constants.Constant` with units of mass.

    Returns
    -------
    `~astropy.units.Quantity`
        The reduced mass between the test particle and target particle.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidParticleError`
        If either particle is invalid.

    `~astropy.units.UnitConversionError`
        If an argument is a `~astropy.units.Quantity` or
        `~astropy.constants.Constant` but does not have units of mass.

    `~plasmapy.particles.exceptions.MissingParticleDataError`
        If the mass of either particle is not known.

    Notes
    -----
    The reduced mass is always less than or equal to the mass of each
    body (i.e., :math:`μ ≤ m_1` and :math:`μ ≤ m_2`).

    When :math:`m_1 = m_2 ≡ m`, the reduced mass becomes :math:`μ =
    \frac{m}{2}`.

    When :math:`m_1 ≫ m_2`, the reduced mass becomes :math:`μ ≈ m_2`.

    Examples
    --------
    >>> import astropy.units as u
    >>> reduced_mass("p+", "e-")
    <Quantity 9.10442...e-31 kg>
    >>> reduced_mass(5.4e-27 * u.kg, 8.6e-27 * u.kg)
    <Quantity 3.31714...e-27 kg>
    """
    return (test_particle.mass * target_particle.mass) / (  # type: ignore[union-attr]
        test_particle.mass + target_particle.mass  # type: ignore[union-attr]
    )


def periodic_table_period(argument: ParticleLike) -> int:
    """
    Return the periodic table period.

    Parameters
    ----------
    argument : |atom-like|
        Atomic number (either integer or string), atomic symbol (e.g.,
        ``"H"``, string), or element name (e.g. ``"Francium"``, string).

    Returns
    -------
    `int`
        The periodic table period of the element.

    See Also
    --------
    periodic_table_group : returns periodic table group of element.

    periodic_table_block : returns periodic table block of element.

    Examples
    --------
    >>> periodic_table_period(5)
    2
    >>> periodic_table_period("5")
    2
    >>> periodic_table_period("Au")
    6
    >>> periodic_table_period("nitrogen")
    2
    """
    # TODO: Implement @particle_input
    if not isinstance(argument, str | Integral):
        raise TypeError(
            "The argument to periodic_table_period must be either a "
            "string representing the element or its symbol, or an "
            "integer representing its atomic number."
        )
    symbol = atomic_symbol(argument)
    return _elements.data_about_elements[symbol]["period"]  # type: ignore[index, return-value]


def periodic_table_group(argument: ParticleLike) -> int:
    """
    Return the periodic table group.

    Parameters
    ----------
    argument : |atom-like|
        Atomic number (either integer or string), atomic symbol (e.g.,
        ``"H"``, string), or element name (e.g., ``"francium"``,
        string).

    Returns
    -------
    `int`
        The periodic table group of the element.

    See Also
    --------
    periodic_table_period : returns periodic table period of element.

    periodic_table_block : returns periodic table block of element.

    periodic_table_category : returns periodic table category of element.

    Examples
    --------
    >>> periodic_table_group(18)
    18
    >>> periodic_table_group(24)
    6
    >>> periodic_table_group("Al")
    13
    >>> periodic_table_group("neon")
    18
    >>> periodic_table_group("barium")
    2
    """
    # TODO: Implement @particle_input
    if not isinstance(argument, str | Integral):
        raise TypeError(
            "The argument to periodic_table_group must be "
            "either a string representing the element or its "
            "symbol, or an integer representing its atomic number."
        )
    symbol = atomic_symbol(argument)
    return _elements.data_about_elements[symbol]["group"]  # type: ignore[index, return-value]


def periodic_table_block(argument: ParticleLike) -> str:
    """
    Return the periodic table block.

    Parameters
    ----------
    argument : |atom-like|
        Atomic number (either integer or string), atomic symbol (e.g.,
        ``"H"``, string), or element name (e.g., ``"francium"``,
        string).

    Returns
    -------
    `str`
        The periodic table block of the element.

    See Also
    --------
    ~plasmapy.particles.atomic.periodic_table_period :
        Returns periodic table period of element.

    ~plasmapy.particles.atomic.periodic_table_group :
        Returns periodic table group of element.

    ~plasmapy.particles.atomic.periodic_table_category :
        Returns periodic table category of element.

    Examples
    --------
    >>> periodic_table_block(66)
    'f'
    >>> periodic_table_block(72)
    'd'
    >>> periodic_table_block("Tl")
    'p'
    >>> periodic_table_block("thallium")
    'p'
    >>> periodic_table_block("francium")
    's'
    """
    # TODO: Implement @particle_input
    if not isinstance(argument, str | Integral):
        raise TypeError(
            "The argument to periodic_table_block must be "
            "either a string representing the element or its "
            "symbol, or an integer representing its atomic number."
        )
    symbol = atomic_symbol(argument)
    return _elements.data_about_elements[symbol]["block"]  # type: ignore[index]


def periodic_table_category(argument: str | int) -> str:
    """
    Return the periodic table category.

    Parameters
    ----------
    argument : |atom-like|
        Atomic number (either integer or string), atomic symbol (e.g.,
        ``"H"``, string), or element name (e.g., ``"francium"``,
        string).

    Returns
    -------
    `str`
        The periodic table category of the element.

    See Also
    --------
    periodic_table_period : returns periodic table period of element.

    periodic_table_group : returns periodic table group of element.

    periodic_table_block : returns periodic table block of element.

    Examples
    --------
    >>> periodic_table_category(82)
    'post-transition metal'
    >>> periodic_table_category("85")
    'halogen'
    >>> periodic_table_category("Ra")
    'alkaline earth metal'
    >>> periodic_table_category("rhodium")
    'transition metal'
    """
    # TODO: Implement @particle_input
    if not isinstance(argument, str | Integral):
        raise TypeError(
            "The argument to periodic_table_category must be "
            "either a string representing the element or its "
            "symbol, or an integer representing its atomic number."
        )
    symbol = atomic_symbol(argument)
    return _elements.data_about_elements[symbol]["category"]  # type: ignore[index]


@particle_input(any_of={"element", "isotope", "ion"})
def ionic_levels(
    particle: ParticleLike,
    min_charge: int | None = None,
    max_charge: int | None = None,
    step: int = 1,
    *,
    Z: Iterable[int] | None = None,
) -> ParticleList:
    """
    Return a |ParticleList| that includes different ionic levels of a
    base atom.

    Parameters
    ----------
    particle : |atom-like|
        Representation of an element, ion, or isotope.

    min_charge : `int`, optional
        The starting |charge number|. Defaults to zero if ``Z`` is not
        provided.

    max_charge : `int`, optional
        The ending charge number, which will be included in the
        |ParticleList|.  Defaults to the atomic number of ``particle``
        if ``Z`` is not provided.

    step : `int`, default: ``1``
        The increment between different charge numbers.

    Z : |array_like| of `int`, |keyword-only|, optional
        The charge numbers corresponding to ``particle``. May not be
        provided if ``min_charge`` or ``max_charge`` is provided.

    Returns
    -------
    `~plasmapy.particles.particle_collections.ParticleList`
        The ionic levels of the atom as provided from ``min_charge``
        ``max_charge``, and ``step`` or from ``Z``.

    Examples
    --------
    >>> from plasmapy.particles import ionic_levels
    >>> ionic_levels("He")
    ParticleList(['He 0+', 'He 1+', 'He 2+'])
    >>> ionic_levels("Fe-56", min_charge=13, max_charge=15)
    ParticleList(['Fe-56 13+', 'Fe-56 14+', 'Fe-56 15+'])
    >>> ionic_levels("O", min_charge=2, max_charge=8, step=2)
    ParticleList(['O 2+', 'O 4+', 'O 6+', 'O 8+'])
    >>> ionic_levels("Pb", Z=[3, 4, 6])
    ParticleList(['Pb 3+', 'Pb 4+', 'Pb 6+'])
    """
    base_particle = Particle(particle.isotope or particle.element)  # type: ignore[union-attr]

    if Z is not None:
        if min_charge or max_charge is not None:
            raise TypeError(
                "Z may not be provided to ionic_levels if min_charge "
                "or max_charge is provided."
            )
        return ParticleList([Particle(base_particle, Z=Z_) for Z_ in Z])

    if max_charge is None:
        max_charge = particle.atomic_number  # type: ignore[union-attr]

    if min_charge is None:
        min_charge = 0

    if not min_charge <= max_charge <= particle.atomic_number:  # type: ignore[union-attr]
        raise ChargeError(
            f"Need min_charge ({min_charge}) "
            f"≤ max_charge ({max_charge}) "
            f"≤ atomic number ({base_particle.atomic_number})."
        )

    return ParticleList(
        [Particle(base_particle, Z=Z) for Z in range(min_charge, max_charge + 1, step)]
    )


def _is_electron(arg: Any) -> bool:
    """
    Return `True` if the argument corresponds to an electron, and
    `False` otherwise.
    """
    # TODO: Remove _is_electron from all parts of code.

    return (
        arg in {"e", "e-"} or arg.lower() == "electron"
        if isinstance(arg, str)
        else False
    )


@particle_input
@validate_quantities(energies=u.MeV)
def stopping_power(
    incident_particle: ParticleLike,
    material: str,
    energies: u.Quantity[u.MeV] | None = None,
    return_interpolator: bool = False,
    component: Literal["total", "electronic", "nuclear"] = "total",
) -> (
    tuple[u.Quantity, u.Quantity]
    | Callable[[u.Quantity[u.J]], u.Quantity[u.MeV * u.cm**2 / u.g]]
):
    """
    Calculate stopping powers for a provided particle in a provided
    material.

    Parameters
    ----------
    incident_particle : |particle-like|
        The incident particle. Only protons and alpha particles are
        currently supported.

    material : `str`
        The material the particle is being stopped in. See notes for
        details on supported materials.

    energies : `~astropy.units.Quantity`, default: See notes.
        The particle kinetic energies for which the stopping power is
        calculated.

    return_interpolator : `bool`, default: `False`
        The function will by default return a tuple of energies and their
        associated stopping power. By setting this argument to `True`, the
        function will instead return a `Callable`, which takes in energies and
        will return the associated stopping energies. Under the hood, this is
        just a wrapped instance of `~scipy.interpolate.CubicSpline` with units.

    component : {"total", "electronic", "nuclear"}, default: ``total``
        The component of the stopping power to be calculated. Supported
        values are ``electronic``, ``nuclear``, and ``total`` for the
        electronic, nuclear, and total energies, respectively.

    Returns
    -------
    ``Tuple[u.Quantity, u.Quantity[u.MeV * u.cm**2 / u.g]]``
        A two-tuple where the first element represents the energy values. The
        second element is an array of the associated stopping powers.

    Notes
    -----
    The data for stopping power is taken from the National Institute of
    Standards and Technology's Stopping-Power and Range Tables :cite:p:`niststar:2005`.
    Valid materials can be found on the NIST STAR website. The default energies
    are taken from the data points in the STAR database.
    """
    # TODO: figure out a better way of handling the Downloader() here
    nist_data_path = Downloader().get_file("NIST_STAR.hdf5")

    # Validate particle input. Currently, the only supported particles are protons and electrons.
    with h5py.File(nist_data_path, "r") as nist_data:
        if incident_particle == Particle("He-4"):
            group_name = "helium_ions"
        elif incident_particle == Particle("e-"):
            raise NotImplementedError(
                "Stopping calculations for electrons have not been implemented yet!"
            )
        elif incident_particle in {Particle("H+"), Particle("p+")}:
            group_name = "protons"
        else:
            raise ValueError(
                "Please pass a valid particle type for stopping power calculations."
            )

        group_data = nist_data[group_name]

        if material not in group_data:
            raise ValueError(
                f"Please pass a valid material string! Material {material} not found in {group_name}."
            )

        # Energies are not included in the material data. They must be loaded from a separate data set.
        # To differentiate from "energies" which refers to the user provided energies, we use "baseline_energies"
        baseline_energies_data = group_data["energy"]
        material_data = group_data[material]

        if component == "total":
            relevant_stopping_data = (
                material_data["electronic_stopping_power"]
                + material_data["nuclear_stopping_power"]
            )
        elif component == "electronic":
            relevant_stopping_data = material_data["electronic_stopping_power"]
        elif component == "nuclear":
            relevant_stopping_data = material_data["nuclear_stopping_power"]
        else:
            raise ValueError(
                f"Please specify one of: total, electronic, or nuclear for component! (Got {component}.)"
            )

        if energies is None and not return_interpolator:
            return (
                baseline_energies_data * u.MeV,
                relevant_stopping_data * u.MeV * u.cm**2 / u.g,
            )

        # Interpolate NIST data to the user-provided energy values. Uses log-log scale fed into a cubic spline.
        cs = CubicSpline(
            x=np.log(baseline_energies_data), y=np.log(relevant_stopping_data)
        )

        # If it has been indicated that the user wants the interpolator, construct
        # an anonymous function to handle units and sanitize IO
        if return_interpolator:
            return (
                lambda x: np.exp(cs(np.log(x.to(u.MeV).value))) * u.MeV * u.cm**2 / u.g
            )

        return (
            energies,
            np.exp(cs(np.log(energies.to("MeV").value))) * u.MeV * u.cm**2 / u.g,  # type: ignore[union-attr]
        )
