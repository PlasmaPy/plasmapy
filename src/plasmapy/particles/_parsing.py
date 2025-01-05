"""Functionality for parsing representations of particles into standard form."""

__all__ = [
    "create_alias_dicts",
    "dealias_particle_aliases",
    "invalid_particle_errmsg",
    "extract_charge",
    "parse_and_check_atomic_input",
    "parse_and_check_molecule_input",
]

import re
import warnings
from numbers import Integral

import numpy as np

from plasmapy.particles import _elements, _isotopes, _special_particles
from plasmapy.particles.exceptions import (
    InvalidElementError,
    InvalidParticleError,
    ParticleWarning,
)
from plasmapy.utils import roman


def create_alias_dicts(particles: dict) -> (dict[str, str], dict[str, str]):
    """
    Create dictionaries for case sensitive aliases and case
    insensitive aliases of special particles and antiparticles.

    The keys of these dictionaries are the aliases, and the values
    are the corresponding standardized symbol for the particle or
    antiparticle.
    """

    case_sensitive_aliases = {}
    case_insensitive_aliases = {}

    for symbol in particles:
        name = particles[symbol]["name"]
        case_insensitive_aliases[name.lower()] = symbol

    case_sensitive_aliases_for_a_symbol = [
        (["beta-", "β-", "β⁻", "e", "e⁻"], "e-"),
        (["beta+", "β+", "β⁺", "e⁺"], "e+"),
        (["p", "p⁺", "H-1+", "H-1 1+", "H-1 +1", "H-1 II"], "p+"),
        (["n-1", "n⁰"], "n"),
        (["H-2"], "D"),
        (["H-2+", "H-2 1+", "H-2 +1", "D+", "D II"], "D 1+"),
        (["H-3+", "H-3 1+", "H-3 +1", "T+", "T II"], "T 1+"),
        (["α"], "He-4 2+"),
        (["τ", "τ-", "τ⁻"], "tau-"),
        (["τ+", "τ⁺"], "tau+"),
        (["μ-", "μ⁻"], "mu-"),
        (["μ+", "μ⁺"], "mu+"),
        (["ν_e"], "nu_e"),
        (["ν_μ"], "nu_mu"),
        (["ν_τ"], "nu_tau"),
    ]

    case_insensitive_aliases_for_a_symbol = [
        (["antielectron", "anti_electron"], "e+"),
        (["antipositron", "anti_positron"], "e-"),
        (["muon-"], "mu-"),
        (["muon+"], "mu+"),
        (["tau particle"], "tau-"),
        (["protium"], "H-1"),
        (
            [
                "protium+",
                "protium 1+",
                "protium +1",
                "hydrogen-1+",
                "hydrogen-1 1+",
                "hydrogen-1 +1",
            ],
            "p+",
        ),
        (["deuterium", "hydrogen-2"], "D"),
        (["deuteron", "deuterium+", "deuterium 1+", "deuterium +1"], "D 1+"),
        (["tritium", "hydrogen-3"], "T"),
        (["triton", "tritium+", "tritium 1+", "tritium +1"], "T 1+"),
        (["diproton"], "He-2 2+"),
        (["helion"], "He-3 2+"),
        (["alpha"], "He-4 2+"),
        (["Freddie"], "Hg"),
    ]

    for aliases, symbol in case_sensitive_aliases_for_a_symbol:
        for alias in aliases:
            case_sensitive_aliases[alias] = symbol

    for aliases, symbol in case_insensitive_aliases_for_a_symbol:
        for alias in aliases:
            case_insensitive_aliases[alias.lower()] = symbol

    alias_keys = list(case_insensitive_aliases.keys())

    for alias in alias_keys:
        if "anti" in alias and "anti-" not in alias:
            symbol = case_insensitive_aliases[alias].lower()
            new_alias = alias.replace("anti", "anti-")
            case_insensitive_aliases[new_alias] = symbol

    return case_sensitive_aliases, case_insensitive_aliases


case_sensitive_aliases, case_insensitive_aliases = create_alias_dicts(
    _special_particles.data_about_special_particles
)


def dealias_particle_aliases(alias: str | int) -> str:
    """
    Return the standard symbol for a particle or antiparticle
    when the argument is a valid alias.  If the argument is not a
    valid alias, then this function returns the original argument
    (which will usually be a `str` but may be an `int` representing
    atomic number).
    """
    if alias in case_sensitive_aliases:
        return case_sensitive_aliases[alias]
    elif isinstance(alias, str) and alias.lower() in case_insensitive_aliases:
        return case_insensitive_aliases[alias.lower()]

    return alias


def invalid_particle_errmsg(
    argument,
    mass_numb: int | None = None,
    Z: int | None = None,
):
    """
    Return an appropriate error message for an
    `~plasmapy.particles.exceptions.InvalidParticleError`.
    """
    errmsg = f"The argument {argument!r} "
    if mass_numb is not None or Z is not None:
        errmsg += "with "
    if mass_numb is not None:
        errmsg += f"{mass_numb = } "
    if mass_numb is not None and Z is not None:
        errmsg += "and "
    if Z is not None:
        errmsg += f"charge number {Z = } "
    errmsg += "does not correspond to a valid particle."
    return errmsg


def extract_charge(arg: str):  # noqa: C901
    """
    Receive a `str` representing an element, isotope, or ion.
    Return a `tuple` containing a `str` that should represent an
    element or isotope, and either an `int` representing the
    charge or `None` if no charge information is provided.  Raise
    an `~plasmapy.particles.exceptions.InvalidParticleError` if charge information
    is inputted incorrectly.
    """

    invalid_charge_errmsg = (
        f"Invalid charge information in the particle string '{arg}'."
    )

    match = re.fullmatch(
        r"\s*(?P<isotope>\w+(-[0-9]+)*([+-]*)*)"
        r"(\s*(?P<charge>[+-]?[IVXLCDM0-9]+[+-]?)*)*\s*",
        arg,
    )

    if match is None:
        raise InvalidParticleError(invalid_charge_errmsg) from None

    isotope_info = match.groupdict()["isotope"]
    charge_info = match.groupdict()["charge"]
    Z_from_arg = None

    if charge_info is not None and isotope_info.endswith(("-", "+")):
        # charge info is defined on both the charge_info and isotope_into strings
        raise InvalidParticleError(invalid_charge_errmsg) from None
    elif charge_info is not None:  # Cases like 'H 1-' and 'Fe-56 1+'
        sign_indicator_only_on_one_end = charge_info.endswith(
            ("-", "+")
        ) ^ charge_info.startswith(("-", "+"))

        just_one_sign_indicator = (
            charge_info.count("-") == 1 and charge_info.count("+") == 0
        ) or (charge_info.count("-") == 0 and charge_info.count("+") == 1)

        if not sign_indicator_only_on_one_end and just_one_sign_indicator:
            raise InvalidParticleError(invalid_charge_errmsg) from None

        charge_str = charge_info.strip("+-")

        try:
            if roman.is_roman_numeral(charge_info):
                Z_from_arg = roman.from_roman(charge_info) - 1
            elif "-" in charge_info:
                Z_from_arg = -int(charge_str)
            elif "+" in charge_info:
                Z_from_arg = int(charge_str)
            else:
                raise InvalidParticleError(invalid_charge_errmsg) from None
        except ValueError:
            raise InvalidParticleError(invalid_charge_errmsg) from None

    elif isotope_info.endswith(("-", "+")):  # Cases like 'H-' and 'Pb-209+++'
        match = re.fullmatch(
            r"\s*(?P<isotope>\w+(-[0-9]+)?)(?P<charge>[-+]+)\s*",
            isotope_info,
        )
        isotope_info = match.groupdict()["isotope"]
        charge_info = match.groupdict()["charge"]

        if len(set(charge_info)) != 1:
            raise InvalidParticleError(invalid_charge_errmsg) from None

        Z_from_arg = len(charge_info)
        if charge_info[0] == "-":
            Z_from_arg = -Z_from_arg

    return isotope_info, Z_from_arg


def parse_and_check_atomic_input(  # noqa: C901, PLR0912, PLR0915
    argument: str | int,
    mass_numb: int | None = None,
    Z: int | None = None,
):
    """
    Parse information about a particle into a dictionary of standard
    symbols, and check the validity of the particle.

    Parameters
    ----------
    argument : `str` or `int`
        String containing information for an element, isotope, or ion
        in any of the allowed formats; or an integer representing an
        atomic number.

    mass_numb : `int`, optional
        The mass number of an isotope.

    Z : `int`, optional
        The charge number of an ion.

    Returns
    -------
    `dict`
        A dictionary containing information about the element, isotope,
        or ion.  The key ``'symbol'`` corresponds to the particle symbol
        containing the most information, ``'element'`` corresponds to
        the atomic symbol, ``'isotope'`` corresponds to the isotope
        symbol, ``'ion'`` corresponds to the ion symbol, ``'mass_numb'``
        corresponds to the mass number, and ``'Z'`` corresponds to the
        charge number.  The corresponding items will be given by `None`
        if the necessary information is not provided.

    Raises
    ------
    `~plasmapy.particles.exceptions.InvalidParticleError`
        If the arguments do not correspond to a valid particle or
        antiparticle.

    `~plasmapy.particles.exceptions.InvalidElementError`
        If the particle is valid but does not correspond to an element,
        ion, or isotope.

    `TypeError`
        If the argument or any of the keyword arguments is not of the
        correct type.
    """

    def atomic_number_to_symbol(atomic_numb: int):
        """
        Return the atomic symbol associated with an integer representing
        an atomic number, or raises an
        `~plasmapy.particles.exceptions.InvalidParticleError` if the
        atomic number does not represent a known element.
        """
        if atomic_numb in _elements.atomic_numbers_to_symbols:
            return _elements.atomic_numbers_to_symbols[atomic_numb]
        else:
            raise InvalidParticleError(f"{atomic_numb} is not a valid atomic number.")

    def extract_mass_number(isotope_info: str):
        """
        Receives a string representing an element or isotope. Return a
        tuple containing a string that should represent an element, and
        either an integer representing the mass number or None if no
        mass number is available.  Raises an
        `~plasmapy.particles.exceptions.InvalidParticleError` if the
        mass number information is inputted incorrectly.
        """

        if isotope_info == "D":
            element_info, mass_numb = "H", 2
        elif isotope_info == "T":
            element_info = "H"
            mass_numb = 3
        elif isotope_info == "p":
            element_info = "H"
            mass_numb = 1
        elif "-" not in isotope_info:
            element_info = isotope_info
            mass_numb = None
        elif isotope_info.count("-") == 1:
            element_info, mass_numb_str = isotope_info.split("-")
            try:
                mass_numb = int(mass_numb_str)
            except ValueError:
                raise InvalidParticleError(
                    f"Invalid mass number in isotope string '{isotope_info}'."
                ) from None

        return element_info, mass_numb

    def get_element(element_info: str) -> str:
        """
        Receive a `str` representing an element's symbol or
        name, and returns a `str` representing the atomic symbol.
        """
        if element_info.lower() in _elements.element_names_to_symbols:
            element = _elements.element_names_to_symbols[element_info.lower()]
        elif element_info in _elements.atomic_numbers_to_symbols.values():
            element = element_info
        else:
            raise InvalidParticleError(
                f"The string '{element_info}' does not correspond to "
                f"a valid element."
            )
        return element

    def reconstruct_isotope_symbol(element: str, mass_numb: int) -> str:
        """
        Receive a `str` representing an atomic symbol and an
        `int` representing a mass number.  Return the isotope symbol
        or `None` if no mass number information is available.  Raises an
        `~plasmapy.particles.exceptions.InvalidParticleError` for isotopes that have
        not yet been discovered.
        """

        if mass_numb is not None:
            isotope = f"{element}-{mass_numb}"

            if isotope == "H-2":
                isotope = "D"
            elif isotope == "H-3":
                isotope = "T"

            if isotope not in _isotopes.data_about_isotopes:
                raise InvalidParticleError(
                    f"The string '{isotope}' does not correspond to "
                    f"a valid isotope."
                )

        else:
            isotope = None

        return isotope

    def reconstruct_ion_symbol(
        element: str, isotope: int | None = None, Z: int | None = None
    ):
        """
        Receive a `str` representing an atomic symbol and/or a
        string representing an isotope, and an `int` representing the
        charge number.  Return a `str` representing the ion symbol,
        or `None` if no charge information is available.
        """

        if Z is not None:
            sign = "-" if Z < 0 else "+"
            base = element if isotope is None else isotope
            ion = f"{base} {np.abs(Z)}{sign}"
        else:
            ion = None

        if ion == "H-1 1+":
            ion = "p+"

        return ion

    if not isinstance(argument, str | int):
        raise TypeError(f"The argument {argument} is not an integer or string.")

    arg = dealias_particle_aliases(argument)

    if arg in _special_particles.particle_zoo.everything - {"p+"}:
        if (mass_numb is not None) or (Z is not None):
            raise InvalidParticleError(
                f"The keywords mass_numb and Z should not be specified "
                f"for particle '{argument}', which is a special particle."
            )
        else:
            raise InvalidElementError(f"{argument} is not a valid element.")

    if isinstance(arg, str) and arg.isdigit():
        arg = int(arg)

    if isinstance(arg, Integral):
        element = atomic_number_to_symbol(arg)
        Z_from_arg = None
        mass_numb_from_arg = None
    elif isinstance(arg, str):
        isotope_info, Z_from_arg = extract_charge(arg)
        element_info, mass_numb_from_arg = extract_mass_number(isotope_info)
        element = get_element(element_info)

    if mass_numb is not None and mass_numb_from_arg is not None:
        if mass_numb != mass_numb_from_arg:
            raise InvalidParticleError(
                "The mass number extracted from the particle string "
                f"'{argument}' is inconsistent with the keyword mass_numb = "
                f"{mass_numb}."
            )
        else:
            warnings.warn(
                "Redundant mass number information for particle "
                f"'{argument}' with mass_numb = {mass_numb}.",
                ParticleWarning,
            )

    if mass_numb_from_arg is not None:
        mass_numb = mass_numb_from_arg

    if Z is not None and Z_from_arg is not None:
        if Z != Z_from_arg:
            raise InvalidParticleError(
                "The charge number extracted from the particle string "
                f"'{argument}' is inconsistent with the keyword Z = {Z}."
            )
        else:
            warnings.warn(
                "Redundant charge information for particle "
                f"'{argument}' with Z = {Z}.",
                ParticleWarning,
            )

    if Z_from_arg is not None:
        Z = Z_from_arg

    if isinstance(Z, Integral):
        if Z > _elements.data_about_elements[element]["atomic number"]:
            raise InvalidParticleError(
                f"The charge number Z = {Z} cannot exceed the atomic number "
                f"of {element}, which is "
                f"{_elements.data_about_elements[element]['atomic number']}."
            )
        elif Z <= -3:
            warnings.warn(
                f"Particle '{argument}' has a charge number "
                f"of Z = {Z}, which is unlikely to occur in "
                f"nature.",
                ParticleWarning,
            )

    isotope = reconstruct_isotope_symbol(element, mass_numb)
    ion = reconstruct_ion_symbol(element, isotope, Z)

    if ion:
        symbol = ion
    elif isotope:
        symbol = isotope
    else:
        symbol = element

    return {
        "symbol": symbol,
        "element": element,
        "isotope": isotope,
        "ion": ion,
        "mass number": mass_numb,
        "charge number": Z,
    }


def parse_and_check_molecule_input(argument: str, Z: int | None = None):
    """
    Separate the constitutive elements and charge of a molecule symbol.

    Parameters
    ----------
    argument : `str`
        The molecule symbol to be parsed.

    Z : integer, optional
        The provided charge number.

    Returns
    -------
    `dict`
        A dictionary with identified element symbols as keys and the
        number of each element that make up the molecule as values. For
        example, ``argument="CO2"`` would lead to ``elements_dict``
        being ``{"C": 1, "O": 2}``.

    molecule_info : `str`
        The molecule symbol stripped of its charge.

    Z : `int`
        The charge number of the molecule.

    Raises
    ------
    `InvalidParticleError`
        If ``argument`` could not be parsed as a molecule.

    Warns
    -----
    : `ParticleWarning`
        If the charge is given both as an argument and in the symbol.
    """
    molecule_info, z_from_arg = extract_charge(argument)
    if not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", molecule_info):
        raise InvalidParticleError(
            f"{molecule_info} is not recognized as a molecule symbol."
        )

    elements_dict = {}
    for match in re.finditer(r"([A-Z][a-z]?)(\d+)?", molecule_info):
        element, amount = match.groups(default="1")
        if element in elements_dict:
            elements_dict[element] += int(amount)
        else:
            elements_dict[element] = int(amount)

    if Z is not None and z_from_arg is not None:
        if Z != z_from_arg:
            raise InvalidParticleError(
                "The charge number extracted from the particle string "
                f"{argument!r} is inconsistent with the keyword Z = {Z}."
            )
        else:
            warnings.warn(
                "Redundant charge information for particle "
                f"'{argument}' with Z = {Z}.",
                ParticleWarning,
            )

    if z_from_arg is not None:
        Z = z_from_arg
    return elements_dict, molecule_info, Z
