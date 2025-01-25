import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

from plasmapy.particles._isotopes import data_about_isotopes
from plasmapy.particles.atomic import (
    _is_electron,
    atomic_number,
    charge_number,
    common_isotopes,
    electric_charge,
    half_life,
    ionic_levels,
    is_stable,
    isotopic_abundance,
    known_isotopes,
    mass_number,
    particle_mass,
    periodic_table_block,
    periodic_table_category,
    periodic_table_group,
    periodic_table_period,
    reduced_mass,
    stable_isotopes,
    standard_atomic_weight,
    stopping_power,
)
from plasmapy.particles.exceptions import (
    ChargeError,
    InvalidElementError,
    InvalidIsotopeError,
    InvalidParticleError,
    MissingParticleDataError,
    ParticleWarning,
)
from plasmapy.particles.particle_class import Particle
from plasmapy.particles.symbols import atomic_symbol, element_name, isotope_symbol
from plasmapy.utils._pytest_helpers import run_test

# function to be tested, argument(s), expected result/outcome

# The following lists (with the name of a function

table_functions_args_kwargs_output = [
    [
        atomic_symbol,
        [
            1,
        ],
        {},
        "H",
    ],
    [atomic_symbol, [1], {}, "H"],
    [atomic_symbol, ["H"], {}, "H"],
    [atomic_symbol, ["p"], {}, "H"],
    [atomic_symbol, ["T"], {}, "H"],
    [atomic_symbol, ["deuterium"], {}, "H"],
    [atomic_symbol, ["deuteron"], {}, "H"],
    [atomic_symbol, ["Tritium"], {}, "H"],
    [atomic_symbol, ["triton"], {}, "H"],
    [atomic_symbol, ["H-2"], {}, "H"],
    [atomic_symbol, ["D"], {}, "H"],
    [atomic_symbol, ["T"], {}, "H"],
    [atomic_symbol, ["H-3"], {}, "H"],
    [atomic_symbol, ["Hydrogen-3"], {}, "H"],
    [atomic_symbol, ["helium"], {}, "He"],
    [atomic_symbol, [2], {}, "He"],
    [atomic_symbol, ["alpha"], {}, "He"],
    [atomic_symbol, ["gold"], {}, "Au"],
    [atomic_symbol, ["Gold"], {}, "Au"],
    [atomic_symbol, [79], {}, "Au"],
    [atomic_symbol, ["79"], {}, "Au"],
    [atomic_symbol, ["P"], {}, "P"],
    [atomic_symbol, [118], {}, "Og"],
    [atomic_symbol, ["N-14"], {}, "N"],
    [atomic_symbol, ["N"], {}, "N"],
    [atomic_symbol, ["H +1"], {}, "H"],
    [atomic_symbol, ["H 1+"], {}, "H"],
    [atomic_symbol, ["hydrogen 1+"], {}, "H"],
    [atomic_symbol, ["deuterium 1+"], {}, "H"],
    [atomic_symbol, ["Fe 24+"], {}, "Fe"],
    [atomic_symbol, ["Fe +24"], {}, "Fe"],
    [atomic_symbol, ["Fe 2-"], {}, "Fe"],
    [atomic_symbol, ["Fe -2"], {}, "Fe"],
    [atomic_symbol, ["Fe+"], {}, "Fe"],
    [atomic_symbol, ["Fe++"], {}, "Fe"],
    [atomic_symbol, ["Fe-"], {}, "Fe"],
    [atomic_symbol, ["Fe++++++++++++++"], {}, "Fe"],
    [isotope_symbol, ("He", 4), {}, "He-4"],
    [isotope_symbol, ("helium-4",), {}, "He-4"],
    [isotope_symbol, ("H-2",), {}, "D"],
    [isotope_symbol, ("Deuterium",), {}, "D"],
    [isotope_symbol, ("deuterium",), {}, "D"],
    [isotope_symbol, ("deuteron",), {}, "D"],
    [isotope_symbol, ("tritium",), {}, "T"],
    [isotope_symbol, ("triton",), {}, "T"],
    [isotope_symbol, ("Hydrogen-3",), {}, "T"],
    [isotope_symbol, ("hydrogen-3",), {}, "T"],
    [isotope_symbol, ("H-3",), {}, "T"],
    [isotope_symbol, (1, 2), {}, "D"],
    [isotope_symbol, ("Hydrogen", 3), {}, "T"],
    [isotope_symbol, ("tritium",), {}, "T"],
    [isotope_symbol, ("H", 2), {}, "D"],
    [isotope_symbol, ("Alpha",), {}, "He-4"],
    [isotope_symbol, ("alpha",), {}, "He-4"],
    [isotope_symbol, (79, 197), {}, "Au-197"],
    [isotope_symbol, ("p",), {}, "H-1"],
    [isotope_symbol, ("beryllium-8",), {}, "Be-8"],
    [isotope_symbol, ("N-13",), {}, "N-13"],
    [isotope_symbol, ("p",), {}, "H-1"],
    [isotope_symbol, ("proton",), {}, "H-1"],
    [isotope_symbol, ("protium",), {}, "H-1"],
    [isotope_symbol, ("N-13 2+",), {}, "N-13"],
    [isotope_symbol, ("Hydrogen-3 +1",), {}, "T"],
    [atomic_number, ["H"], {}, 1],
    [atomic_number, ["D"], {}, 1],
    [atomic_number, ["deuterium"], {}, 1],
    [atomic_number, ["Deuterium"], {}, 1],
    [atomic_number, ["tritium"], {}, 1],
    [atomic_number, ["p"], {}, 1],
    [atomic_number, ["P"], {}, 15],
    [atomic_number, ["Alpha"], {}, 2],
    [atomic_number, ["C-12"], {}, 6],
    [atomic_number, ["Argon"], {}, 18],
    [atomic_number, ["protium"], {}, 1],
    [atomic_number, ["H-3"], {}, 1],
    [atomic_number, ["p+"], {}, 1],
    [atomic_number, ["Be-8"], {}, 4],
    [atomic_number, ["N"], {}, 7],
    [atomic_number, ["N 2+"], {}, 7],
    [atomic_number, ["N +1"], {}, 7],
    [atomic_number, ["N+++"], {}, 7],
    [mass_number, ["helium-3"], {}, 3],
    [mass_number, ["Au-197"], {}, 197],
    [mass_number, ["deuterium"], {}, 2],
    [mass_number, ["D"], {}, 2],
    [mass_number, ["H-2"], {}, 2],
    [mass_number, ["tritium"], {}, 3],
    [mass_number, ["T"], {}, 3],
    [mass_number, ["alpha"], {}, 4],
    [mass_number, ["p"], {}, 1],
    [mass_number, ["Be-8"], {}, 8],
    [mass_number, ["N-13"], {}, 13],
    [mass_number, ["N-13 2+"], {}, 13],
    [mass_number, ["N-13 +2"], {}, 13],
    [mass_number, ["N-13+++"], {}, 13],
    [element_name, ["D"], {}, "hydrogen"],
    [element_name, ["deuterium"], {}, "hydrogen"],
    [element_name, ["Au"], {}, "gold"],
    [element_name, ["alpha"], {}, "helium"],
    [element_name, ["helium-4"], {}, "helium"],
    [element_name, ["H-2"], {}, "hydrogen"],
    [element_name, ["Deuterium"], {}, "hydrogen"],
    [element_name, ["Hydrogen-3"], {}, "hydrogen"],
    [element_name, ["hydrogen-3"], {}, "hydrogen"],
    [element_name, ["H-3"], {}, "hydrogen"],
    [element_name, ["tritium"], {}, "hydrogen"],
    [element_name, ["Alpha"], {}, "helium"],
    [element_name, ["alpha"], {}, "helium"],
    [element_name, [1], {}, "hydrogen"],
    [element_name, [26], {}, "iron"],
    [element_name, [79], {}, "gold"],
    [element_name, ["p"], {}, "hydrogen"],
    [element_name, ["P"], {}, "phosphorus"],
    [element_name, ["Be-8"], {}, "beryllium"],
    [element_name, ["Li-7"], {}, "lithium"],
    [element_name, ["N"], {}, "nitrogen"],
    [element_name, ["N+++"], {}, "nitrogen"],
    [element_name, ["D-"], {}, "hydrogen"],
    [standard_atomic_weight, ["H"], {}, (1.008 * u.u).to(u.kg)],
    [standard_atomic_weight, [1], {}, (1.008 * u.u).to(u.kg)],
    [standard_atomic_weight, ["Hydrogen"], {}, (1.008 * u.u).to(u.kg)],
    [standard_atomic_weight, ["Au"], {}, u.kg],
    [particle_mass, ["proton"], {}, const.m_p],
    [particle_mass, ["H-1+"], {}, const.m_p],
    [particle_mass, ["H-1 +1"], {}, const.m_p],
    [particle_mass, ["H-1 1+"], {}, const.m_p],
    [particle_mass, ["H-1"], {"Z": 1}, const.m_p],
    [particle_mass, ["hydrogen-1"], {"Z": 1}, const.m_p],
    [particle_mass, ["p+"], {}, const.m_p],
    [particle_mass, ["F-19"], {"Z": 3}, u.kg],
    [particle_mass, ["H"], {}, standard_atomic_weight("H")],
    [is_stable, ["H-1"], {}, True],
    [is_stable, [1, 1], {}, True],
    [is_stable, ["N-14"], {}, True],
    [is_stable, ["N", 14], {}, True],
    [is_stable, ["P-31"], {}, True],
    [is_stable, ["P", 31], {}, True],
    [is_stable, ["p"], {}, True],
    [is_stable, ["alpha"], {}, True],
    [is_stable, ["Xe-124"], {}, True],
    [is_stable, ("Fe",), {"mass_numb": 56}, True],
    [is_stable, ["Fe-56"], {}, True],
    [is_stable, ["iron-56"], {}, True],
    [is_stable, ["Iron-56"], {}, True],
    [is_stable, [26, 56], {}, True],
    [is_stable, ["Be-8"], {}, False],
    [is_stable, ["U-235"], {}, False],
    [is_stable, ["uranium-235"], {}, False],
    [is_stable, ["T"], {}, False],
    [is_stable, [4, 8], {}, False],
    [is_stable, ["tritium"], {}, False],
    [is_stable, ["Pb-209"], {}, False],
    [is_stable, ["lead-209"], {}, False],
    [is_stable, ["Lead-209"], {}, False],
    [is_stable, ("Pb",), {"mass_numb": 209}, False],
    [is_stable, [82, 209], {}, False],
    [charge_number, ["H+"], {}, 1],
    [charge_number, ["D +1"], {}, 1],
    [charge_number, ["tritium 1+"], {}, 1],
    [charge_number, ["H-"], {}, -1],
    [charge_number, ["Fe -2"], {}, -2],
    [charge_number, ["Fe 2-"], {}, -2],
    [charge_number, ["N--"], {}, -2],
    [charge_number, ["N++"], {}, 2],
    [charge_number, ["alpha"], {}, 2],
    [charge_number, ["proton"], {}, 1],
    [charge_number, ["deuteron"], {}, 1],
    [charge_number, ["triton"], {}, 1],
    [charge_number, ["electron"], {}, -1],
    [charge_number, ["e-"], {}, -1],
    [charge_number, ["e+"], {}, 1],
    [charge_number, ["positron"], {}, 1],
    [charge_number, ["n"], {}, 0],
    [charge_number, ["neutron"], {}, 0],
    [charge_number, ["p-"], {}, -1],
    [charge_number, ["antiproton"], {}, -1],
    [electric_charge, ["p"], {}, u.C],
    [electric_charge, ["p"], {}, 1.6021766208e-19 * u.C],
    [electric_charge, ["e"], {}, -1.6021766208e-19 * u.C],
    [electric_charge, ["alpha"], {}, 3.2043532416e-19 * u.C],
    [electric_charge, ["n"], {}, 0 * u.C],
    [half_life, ["H-1"], {}, u.s],
    [half_life, ["tritium"], {}, u.s],
    [half_life, ["H-1"], {}, np.inf * u.s],
]


@pytest.mark.parametrize(
    ("tested_function", "args", "kwargs", "expected_output"),
    table_functions_args_kwargs_output,
)
def test_functions_and_values(tested_function, args, kwargs, expected_output) -> None:
    run_test(tested_function, args, kwargs, expected_output)


class TestInvalidPeriodicElement:
    def test_periodic_table_period(self) -> None:
        with pytest.raises(TypeError):
            periodic_table_period(("Ne", "Na"))

    def test_periodic_table_block(self) -> None:
        with pytest.raises(TypeError):
            periodic_table_block(("N", "C", "F"))

    def test_periodic_table_category(self) -> None:
        with pytest.raises(TypeError):
            periodic_table_category(["Rb", "He", "Li"])

    def test_periodic_table_group(self) -> None:
        with pytest.raises(TypeError):
            periodic_table_group(("B", "Ti", "Ge"))


# Next we have tests that do not fall nicely into equality comparisons.


def test_standard_atomic_weight_value_between() -> None:
    """Test that `standard_atomic_weight` returns approximately the
    correct value for phosphorus."""
    assert 30.973 < standard_atomic_weight("P").to(u.u).value < 30.974, (
        "Incorrect standard atomic weight for phosphorus."
    )


def test_particle_mass_berkelium_249() -> None:
    """Test that `particle_mass` returns the correct value for Bk-249."""
    assert np.isclose(particle_mass("berkelium-249").to(u.u).value, 249.0749877), (
        "Incorrect isotope mass for berkelium."
    )


def test_particle_mass_for_hydrogen_with_no_mass_number() -> None:
    """Test that `particle_mass` does not return the proton mass when no
    mass number is specified for hydrogen.  In this case, the
    standard atomic weight should be used to account for the small
    fraction of deuterium."""
    assert particle_mass("H", Z=1) > const.m_p
    assert particle_mass("hydrogen", Z=1) > const.m_p


def test_particle_mass_helium() -> None:
    """Test miscellaneous cases for `particle_mass`."""
    assert particle_mass("alpha") > particle_mass("He-3 2+")


# (arg1, kwargs1, arg2, kwargs2, expected)
equivalent_particle_mass_args = [
    ["e+", {}, "positron", {}, const.m_e],
    ["alpha", {}, "He-4++", {}, None],
    ["alpha", {}, "helium-4 2+", {}, None],
    ["deuteron", {}, "H", {"Z": 1, "mass_numb": 2}, None],
    ["D+", {}, "H-2+", {}, None],
    ["D+", {}, "D 1+", {}, None],
    ["Deuterium+", {}, "D", {"Z": 1}, None],
    ["triton", {}, "H", {"Z": 1, "mass_numb": 3}, None],
    ["T+", {}, "H-3+", {}, None],
    ["T+", {}, "T 1+", {}, None],
    ["Tritium+", {}, "T", {"Z": 1}, None],
    [
        "Fe-56 1+",
        {},
        "Fe",
        {"mass_numb": 56, "Z": 1},
        particle_mass("Fe-56 1-") - 2 * const.m_e,
    ],
    ["Fe-56 +1", {}, 26, {"mass_numb": 56, "Z": 1}, None],
]


@pytest.mark.parametrize(
    ("arg1", "kwargs1", "arg2", "kwargs2", "expected"), equivalent_particle_mass_args
)
def test_particle_mass_equivalent_args(arg1, kwargs1, arg2, kwargs2, expected) -> None:
    """Test that `particle_mass` returns equivalent results for
    equivalent positional and keyword arguments."""

    result1 = particle_mass(arg1, **kwargs1)
    result2 = particle_mass(arg2, **kwargs2)

    assert u.isclose(result1, result2), (
        f"particle_mass({arg1!r}, **{kwargs1}) = {result1!r}, whereas "
        f"particle_mass({arg2!r}, **{kwargs2}) = {result2!r}.  "
        f"These results are not equivalent as expected."
    )

    if expected is not None:
        assert u.isclose(result1, result2) and u.isclose(  # noqa: PT018
            result2, expected
        ), (
            f"particle_mass({arg1!r}, **{kwargs1}) = {result1!r} and "
            f"particle_mass({arg2!r}, **{kwargs2}) = {result2!r}, but "
            f"these results are not equal to {expected!r} as expected."
        )


@pytest.mark.slow
def test_known_common_stable_isotopes() -> None:
    """Test that `known_isotopes`, `common_isotopes`, and
    `stable_isotopes` return the correct values for hydrogen."""

    known_should_be = ["H-1", "D", "T", "H-4", "H-5", "H-6", "H-7"]
    common_should_be = ["H-1", "D"]
    stable_should_be = ["He-3", "He-4"]

    assert known_isotopes("H") == known_should_be, (
        f"known_isotopes('H') should return {known_should_be}, but is "
        f"instead returning {known_isotopes('H')}"
    )

    assert common_isotopes("H") == common_should_be, (
        f"common_isotopes('H') should return {common_should_be}, but is "
        f"instead returning {common_isotopes('H')}"
    )

    assert stable_isotopes("He") == stable_should_be, (
        f"stable_isotopes('He') should return {stable_should_be}, but is "
        f"instead returning {stable_isotopes('He')}"
    )


def test_half_life() -> None:
    """Test that `half_life` returns the correct values for various
    isotopes."""
    assert np.isclose(
        half_life("tritium").to(u.s).value, (12.32 * u.yr).to(u.s).value, rtol=2e-4
    ), "Incorrect half-life for tritium."


def test_half_life_unstable_isotopes() -> None:
    """Test that `half_life` returns `None` and raises an exception for
    all isotopes that do not yet have half-life data."""
    for isotope in data_about_isotopes:
        if (
            "half_life" not in data_about_isotopes[isotope]
            and not data_about_isotopes[isotope]
        ):
            with pytest.raises(MissingParticleDataError):
                half_life(isotope)


def test_half_life_u_220() -> None:
    """Test that `half_life` returns `None` and issues a warning for an
    isotope without half-life data."""

    isotope_without_half_life_data = "No-248"

    with pytest.raises(MissingParticleDataError):
        half_life(isotope_without_half_life_data)
        pytest.fail(
            f"This test assumes that {isotope_without_half_life_data} does "
            f"not have half-life data.  If half-life data is added for this "
            f"isotope, then a different isotope that does not have half-life "
            f"data should be chosen for this test."
        )


def test_known_common_stable_isotopes_cases() -> None:
    """Test that known_isotopes, common_isotopes, and stable_isotopes
    return certain isotopes that fall into these categories."""
    assert "H-1" in known_isotopes("H")
    assert "D" in known_isotopes("H")
    assert "T" in known_isotopes("H")
    assert "Be-8" in known_isotopes("Be")
    assert "Og-294" in known_isotopes(118)
    assert "H-1" in common_isotopes("H")
    assert "H-4" not in common_isotopes(1)
    assert "H-1" in stable_isotopes("H")
    assert "D" in stable_isotopes("H")
    assert "T" not in stable_isotopes("H")
    assert "Fe-56" in common_isotopes("Fe", most_common_only=True)
    assert "He-4" in common_isotopes("He", most_common_only=True)


@pytest.mark.slow
def test_known_common_stable_isotopes_len() -> None:
    """Test that `known_isotopes`, `common_isotopes`, and
    `stable_isotopes` each return a `list` of the expected length.

    The number of common isotopes may change if isotopic composition
    data has any significant changes.

    The number of stable isotopes may decrease slightly if some isotopes
    are discovered to be unstable but with extremely long half-lives.

    The number of known isotopes will increase as new isotopes are
    discovered, so a buffer is included in the test.

    """

    assert len(common_isotopes()) == 288, (
        "The length of the list returned by common_isotopes() is "
        f"{len(common_isotopes())}, which is not the expected value."
    )

    assert len(stable_isotopes()) == 254, (
        "The length of the list returned by stable_isotopes() is "
        f"{len(stable_isotopes())}, which is not the expected value."
    )

    assert 3352 <= len(known_isotopes()) <= 3400, (
        "The length of the list returned by known_isotopes() is "
        f"{len(known_isotopes())}, which is not within the expected range."
    )


@pytest.mark.parametrize("func", [common_isotopes, stable_isotopes, known_isotopes])
def test_known_common_stable_isotopes_error(func) -> None:
    """Test that `known_isotopes`, `common_isotopes`, and
    `stable_isotopes` raise an `~plasmapy.utils.InvalidElementError` for
    neutrons."""
    with pytest.raises(InvalidElementError):
        func("n")
        pytest.fail(f"{func} is not raising a ElementError for neutrons.")


def test_isotopic_abundance() -> None:
    """Test that `isotopic_abundance` returns the appropriate values or
    raises appropriate errors for various isotopes."""
    assert isotopic_abundance("H", 1) == isotopic_abundance("protium")
    assert np.isclose(isotopic_abundance("D"), 0.000115)
    assert isotopic_abundance("Be-8") == 0.0, "Be-8"
    assert isotopic_abundance("Li-8") == 0.0, "Li-8"

    with pytest.warns(ParticleWarning):
        isotopic_abundance("Og", 294)

    with pytest.raises(InvalidIsotopeError):
        isotopic_abundance("neutron")
        pytest.fail("No exception raised for neutrons.")

    with pytest.raises(InvalidParticleError):
        isotopic_abundance("Og-2")


atomic_numbers = range(1, 119)

isotopic_abundance_isotopes = (common_isotopes(element) for element in atomic_numbers)

isotopic_abundance_sum_table = (
    (element, isotopes)
    for element, isotopes in zip(
        atomic_numbers, isotopic_abundance_isotopes, strict=False
    )
    if isotopes
)


@pytest.mark.parametrize(("element", "isotopes"), isotopic_abundance_sum_table)
def test_isotopic_abundances_sum(element, isotopes) -> None:
    """Test that the sum of isotopic abundances for each element with
    isotopic abundances is one."""
    sum_of_iso_abund = sum(isotopic_abundance(isotope) for isotope in isotopes)
    assert np.isclose(sum_of_iso_abund, 1, atol=1e-6), (
        f"The sum of the isotopic abundances for {element} does not equal 1."
    )


class TestReducedMassInput:
    def test_incorrect_units(self) -> None:
        with pytest.raises(InvalidParticleError):
            reduced_mass("N", 6e-26 * u.l)

    def test_missing_atomic_data(self) -> None:
        assert u.isclose(reduced_mass("Og", "H"), np.nan * u.kg, equal_nan=True)


def test_ion_list_example() -> None:
    ions = ionic_levels("He-4")
    np.testing.assert_equal(ions.charge_number, [0, 1, 2])
    assert ions.symbols == ["He-4 0+", "He-4 1+", "He-4 2+"]


@pytest.mark.parametrize(
    ("particle", "min_charge", "max_charge", "expected_charge_numbers"),
    [
        ("H-1", 0, 1, [0, 1]),
        ("p+", 1, 1, [1]),
        (Particle("p+"), 0, 0, [0]),
        ("C", 3, 5, [3, 4, 5]),
    ],
)
def test_ionic_levels(
    particle, min_charge, max_charge, expected_charge_numbers
) -> None:
    """Test that inputs to ionic_levels are interpreted correctly."""
    particle = Particle(particle)
    ions = ionic_levels(particle, min_charge, max_charge)
    np.testing.assert_equal(ions.charge_number, expected_charge_numbers)
    assert ions[0].element == particle.element
    if particle.is_category("isotope"):
        assert ions[0].isotope == particle.isotope


@pytest.mark.parametrize(
    ("element", "min_charge", "max_charge"), [("Li", 0, 4), ("Li", 3, 2)]
)
def test_invalid_inputs_to_ionic_levels(element, min_charge, max_charge) -> None:
    with pytest.raises(ChargeError):
        ionic_levels(element, min_charge, max_charge)


str_electron_table = [
    ("e-", True),
    ("e+", False),
    ("e", True),
    ("electron", True),
    ("ELECTRON", True),
    ("H", False),
    ("positron", False),
    ("Carbon", False),
    (("e", "e-"), False),
    (["e+", "proton"], False),
    ("merry-go-round", False),
]


@pytest.mark.parametrize(("particle", "electron"), str_electron_table)
def test_is_electron(particle, electron) -> None:
    assert _is_electron(particle) == electron


def test_ionic_levels_example() -> None:
    """
    Test that `ionic_levels` can be used to create a |ParticleList|
    containing all the ions for a particular element.
    """
    ions = ionic_levels("He-4")
    np.testing.assert_equal(ions.charge_number, [0, 1, 2])
    assert ions.symbols == ["He-4 0+", "He-4 1+", "He-4 2+"]


@pytest.mark.parametrize(
    ("particle", "min_charge", "max_charge", "Z", "expected_charge_numbers"),
    [
        ("H-1", 0, 1, None, [0, 1]),
        ("p+", 1, 1, None, [1]),
        (Particle("p+"), 0, 0, None, [0]),
        ("C", 3, 5, None, [3, 4, 5]),
        ("Fe", None, None, [1, 2], [1, 2]),
    ],
)
def test_ionic_levels_inputs(
    particle, min_charge, max_charge, Z, expected_charge_numbers
) -> None:
    """Test that inputs to ionic_levels are interpreted correctly."""
    particle = Particle(particle)
    ions = ionic_levels(particle, min_charge, max_charge, Z=Z)
    np.testing.assert_equal(ions.charge_number, expected_charge_numbers)
    assert ions[0].element == particle.element
    if particle.is_category("isotope"):
        assert ions[0].isotope == particle.isotope


@pytest.mark.parametrize(
    ("element", "min_charge", "max_charge"), [("Li", 0, 4), ("Li", 3, 2)]
)
def test_invalid_inputs_to_ionic_levels2(element, min_charge, max_charge) -> None:
    with pytest.raises(ChargeError):
        ionic_levels(element, min_charge, max_charge)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("incident_particle", "material", "kwargs", "expected_error"),
    [
        # ESTAR not implemented
        (Particle("e-"), "OXYGEN", {}, NotImplementedError),
        # Invalid incident particle
        (Particle("O"), "OXYGEN", {}, ValueError),
        # Invalid material
        (Particle("H+"), "BENZENE", {}, ValueError),
        # Invalid component
        (Particle("H+"), "OXYGEN", {"component": "Lorem Ipsum"}, ValueError),
    ],
)
def test_stopping_power_errors(
    incident_particle, material, kwargs, expected_error
) -> None:
    with pytest.raises(expected_error):
        stopping_power(incident_particle, material, **kwargs)


@pytest.mark.parametrize(
    (
        "incident_particle",
        "material",
        "energies",
        "component",
        "expected_stopping_power",
    ),
    [
        # Test ASTAR interpolation
        (
            Particle("He-4"),
            "OXYGEN",
            [1.25e-3, 1.25e-2, 1.25e-1, 1.25, 1.25e1, 1.25e2] * u.MeV,
            "total",
            [2.383e2, 4.055e2, 1.048e3, 1.669e3, 3.837e2, 6.273e1]
            * u.MeV
            * u.cm**2
            / u.g,
        ),
        # Test PSTAR interpolation
        (
            Particle("H+"),
            "HYDROGEN",
            [1.25e-3, 1.25e-2, 1.25e-1, 1.25, 1.25e1, 1.25e2] * u.MeV,
            "total",
            [1.030e3, 2.621e3, 3.188e3, 5.673e2, 8.44e1, 1.295e1]
            * u.MeV
            * u.cm**2
            / u.g,
        ),
        # Test electronic component
        (
            Particle("H+"),
            "ACETYLENE",
            [1.25e-3, 1.25e-2, 1.25e-1, 1.25, 1.25e1, 1.25e2] * u.MeV,
            "electronic",
            [2.073e2, 6.002e2, 8.871e2, 2.307e2, 3.859e1, 6.185]
            * u.MeV
            * u.cm**2
            / u.g,
        ),
        # Test nuclear component
        (
            Particle("He-4"),
            "POLYSTYRENE",
            [1.25e-3, 1.25e-2, 1.25e-1, 1.25, 1.25e1, 1.25e2] * u.MeV,
            "nuclear",
            [2.143e2, 7.921e1, 1.526e1, 2.244, 2.936e-1, 3.450e-2]
            * u.MeV
            * u.cm**2
            / u.g,
        ),
        # # Test no interpolation
        # (
        #     Particle("H+"),
        #     "COPPER",
        #     None,
        #     "total",
        #     [0, 0, 0, 0, 0, 0] * u.MeV * u.cm**2 / u.g,
        # ),
    ],
)
def test_stopping_power_interpolation(
    incident_particle, material, energies, component, expected_stopping_power
) -> None:
    """Test the interpolation functionality of the stopping power function against NIST values"""
    _, actual_stopping_power = stopping_power(
        incident_particle, material, energies, component=component
    )

    # NIST data is given to four significant figures: use a tolerance of 1 part in 1000
    assert np.isclose(actual_stopping_power, expected_stopping_power, rtol=0.001).all()


def test_stopping_power_no_interpolation() -> None:
    result = stopping_power(Particle("H+"), "COPPER")

    assert type(result) is tuple
