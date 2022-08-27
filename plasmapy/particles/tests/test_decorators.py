import astropy.units as u
import inspect
import pytest

from typing import List, Optional, Tuple, Union

from plasmapy.particles.decorators import particle_input
from plasmapy.particles.exceptions import (
    ChargeError,
    InvalidElementError,
    InvalidIonError,
    InvalidIsotopeError,
    InvalidParticleError,
    ParticleError,
)
from plasmapy.particles.particle_class import CustomParticle, Particle, ParticleLike
from plasmapy.utils.code_repr import call_string
from plasmapy.utils.decorators.validators import validate_quantities


@particle_input
def func_simple_noparens(
    a, particle: ParticleLike, b=None, Z: int = None, mass_numb: int = None
) -> Particle:
    """
    A simple function that, when decorated with `@particle_input`,
    returns the instance of the Particle class corresponding to the
    inputs.
    """
    if not isinstance(particle, Particle):
        raise TypeError(
            "The argument `particle` in func_simple_noparens should be "
            f"a Particle. Instead, {particle=}."
        )
    return particle


@particle_input()
def func_simple_parens(
    a, particle: ParticleLike, b=None, Z: int = None, mass_numb: int = None
) -> Particle:
    """
    A simple function that, when decorated with `@particle_input()`,
    returns the instance of the Particle class corresponding to the
    inputs.
    """
    if not isinstance(particle, Particle):
        raise TypeError(
            "The argument `particle` in func_simple_parens should be "
            f"a Particle. Instead, {particle=}."
        )
    return particle


particle_input_simple_table = [
    (func_simple_noparens, (1, "p+"), {"b": 2}, "p+"),
    (func_simple_parens, (1, "p+"), {"b": 2}, "p+"),
    (func_simple_noparens, (1, "Fe"), {"mass_numb": 56, "Z": 3}, "Fe-56 3+"),
    (func_simple_parens, (1, "Fe"), {"mass_numb": 56, "Z": 3}, "Fe-56 3+"),
    (func_simple_parens, (1,), {"particle": "e-"}, "e-"),
    (func_simple_noparens, (1,), {"particle": "e-"}, "e-"),
    (func_simple_noparens, (1,), {"particle": Particle("e-")}, "e-"),
    (func_simple_parens, (1,), {"particle": Particle("e-")}, "e-"),
]


@pytest.mark.parametrize("func, args, kwargs, symbol", particle_input_simple_table)
def test_particle_input_simple(func, args, kwargs, symbol):
    """
    Test that simple functions decorated by particle_input correctly
    return the correct Particle object.
    """
    try:
        expected = Particle(symbol)
    except Exception as e:
        raise ParticleError(f"Cannot create Particle class from symbol {symbol}") from e

    try:
        result = func(*args, **kwargs)
    except Exception as e:
        raise ParticleError(
            f"An exception was raised while trying to execute:\n\n "
            f"{call_string(func, args, kwargs)}\n"
        ) from e

    assert result == expected, (
        f"The result {repr(result)} does not equal the expected value of "
        f"{repr(expected)}.\n\n"
        f"func = {func}\n"
        f"args = {args}\n"
        f"kwargs = {kwargs}\nsymbol = {symbol}\n"
        f"{result._attributes}\n"
        f"{expected._attributes}\n"
    )


# function, kwargs, expected_error
particle_input_error_table = [
    (func_simple_noparens, {"a": 1, "particle": "asdf"}, InvalidParticleError)
]


@pytest.mark.parametrize("func, kwargs, expected_error", particle_input_error_table)
def test_particle_input_errors(func, kwargs, expected_error):
    """
    Test that functions decorated with particle_input raise the
    expected errors.
    """
    with pytest.raises(expected_error):
        func(**kwargs)
        pytest.fail(f"{func} did not raise {expected_error} with kwargs = {kwargs}")


class HasDecoratedMethods:
    """
    A sample class with methods that will be used to check that
    @particle_input works both with and without parentheses.
    """

    @particle_input
    def method_noparens(self, particle: ParticleLike) -> Particle:
        return particle

    @particle_input()
    def method_parens(self, particle: ParticleLike) -> Particle:
        return particle


@pytest.mark.parametrize("symbol", ["muon", "He 3+"])
def test_particle_input_classes(symbol):
    instance = HasDecoratedMethods()

    symbol = "muon"
    expected = Particle(symbol)

    result_noparens = instance.method_noparens(symbol)
    result_parens = instance.method_parens(symbol)

    assert result_parens == result_noparens == expected


@particle_input
def function_with_no_annotations():
    """A trivial function that is incorrectly decorated with
    particle_input because no arguments are annotated with Particle."""
    pass


def test_no_annotations_exception():
    """Test that a function decorated with particle_input that has no
    annotated arguments will raise an ParticleError."""
    with pytest.raises(ParticleError):
        function_with_no_annotations()


@particle_input
def ambiguous_keywords(p1: ParticleLike, p2: ParticleLike, Z=None, mass_numb=None):
    """A trivial function with two annotated arguments plus the keyword
    arguments `Z` and `mass_numb`."""
    pass


def test_function_with_ambiguity():
    """Test that a function decorated with particle_input that has two
    annotated arguments"""
    with pytest.raises(ParticleError):
        ambiguous_keywords("H", "He", Z=1, mass_numb=4)
    with pytest.raises(ParticleError):
        ambiguous_keywords("H", "He", Z=1)
    with pytest.raises(ParticleError):
        ambiguous_keywords("H", "He", mass_numb=4)
    # TODO: should particle_input raise an exception when Z and mass_numb
    # are given as keyword arguments but are not explicitly set?


def function_to_test_annotations(particles: Union[Tuple, List], resulting_particles):
    """
    Test that a function with an argument annotated with (Particle,
    Particle, ...) or [Particle] returns a tuple of expected Particle
    instances.

    Arguments
    =========
    particles: tuple or list
        A collection containing many items, each of which may be a valid
        representation of a particle or a `~plasmapy.particles.Particle`
        instance
    """

    expected = [
        particle if isinstance(particle, Particle) else Particle(particle)
        for particle in particles
    ]

    # Check that the returned values are Particle instances because
    # running:
    #     Particle('p+') == 'p+'
    # will return True because of how Particle.__eq__ is set up.

    returned_particle_instances = all(
        isinstance(p, Particle) for p in resulting_particles
    )

    returned_correct_instances = all(
        expected[i] == resulting_particles[i] for i in range(len(particles))
    )

    if not returned_particle_instances:
        raise ParticleError(
            f"A function decorated by particle_input did not return "
            f"a collection of Particle instances for input of "
            f"{repr(particles)}, and instead returned"
            f"{repr(resulting_particles)}."
        )

    if not returned_correct_instances:
        raise ParticleError(
            f"A function decorated by particle_input did not return "
            f"{repr(expected)} as expected, and instead returned "
            f"{repr(resulting_particles)}."
        )


class TestOptionalArgs:
    def particle_iter(self, particles):
        return [Particle(particle) for particle in particles]

    def test_optional_particle(self):
        particle = "He"

        @particle_input
        def has_default_particle(particle: ParticleLike = particle):
            return particle

        assert has_default_particle() == Particle(particle)
        assert has_default_particle("Ne") == Particle("Ne")


def test_no_annotations_found():
    @particle_input
    def invalid_list_type(particles):
        pass

    with pytest.raises(ParticleError):
        invalid_list_type((Particle("He"), "Ne"))


def test_nonhashable_annotation():
    """
    Verify that the annotation for a different parameter can be a
    non-hashable object such as a `list`.

    Notes
    -----
    This problem arose when the collections containing the annotations
    to be processed were sets, and there was an operation like
    ``[] in set()``.
    """

    @particle_input
    def has_nonhashable_annotation(x: [], particle: ParticleLike):
        pass

    has_nonhashable_annotation(1, "e+")


# decorator_kwargs, particle, expected_exception
decorator_categories_table = [
    ({"exclude": {"element"}}, "Fe", ParticleError),
    ({"any_of": {"lepton", "antilepton"}}, "tau-", None),
    ({"require": {"isotope", "ion"}}, "Fe-56+", None),
    ({"require": {"isotope", "ion"}}, "Fe+", ParticleError),
    ({"any_of": {"isotope", "ion"}}, "Fe+", None),
    ({"any_of": {"charged", "uncharged"}}, "Fe", ChargeError),
    ({"any_of": ["charged", "uncharged"]}, "Fe", ChargeError),
    ({"any_of": ("charged", "uncharged")}, "Fe", ChargeError),
    ({"require": "charged"}, "Fe 0+", ChargeError),
    (
        {
            "require": ["fermion", "charged"],
            "any_of": ["lepton", "baryon"],
            "exclude": ["antimatter"],
        },
        "p+",
        None,
    ),
    (
        {
            "require": ["fermion", "charged"],
            "any_of": ["lepton", "baryon"],
            "exclude": ["antimatter"],
        },
        "p+",
        None,
    ),
    (
        {
            "require": ["fermion", "charged"],
            "any_of": ["lepton", "baryon"],
            "exclude": ["matter"],
        },
        "p+",
        ParticleError,
    ),
]


@pytest.mark.parametrize(
    "decorator_kwargs, particle, expected_exception", decorator_categories_table
)
def test_decorator_categories(decorator_kwargs, particle, expected_exception):
    """Tests the require, any_of, and exclude categories lead to an
    ParticleError being raised when an inputted particle does not meet
    the required criteria, and do not lead to an ParticleError when the
    inputted particle matches the criteria."""

    @particle_input(**decorator_kwargs)
    def decorated_function(argument: ParticleLike) -> Particle:
        return argument

    if expected_exception:
        with pytest.raises(expected_exception):
            decorated_function(particle)
            pytest.fail(
                f"{call_string(decorated_function, [], decorator_kwargs)} "
                f"did not raise {expected_exception}"
            )
    else:
        decorated_function(particle)


def test_optional_particle_annotation_argname():
    """Tests the `Optional[Particle]` annotation argument in a function
    decorated by `@particle_input` such that the annotated argument allows
    `None` to be passed through to the decorated function."""

    @particle_input
    def func_optional_particle(particle: Optional[ParticleLike]) -> Optional[Particle]:
        return particle

    assert func_optional_particle(None) is None, (
        "The particle keyword in the particle_input decorator is set "
        "to accept Optional[Particle], but is not passing through None."
    )


# TODO: The following tests might be able to be cleaned up and/or
# further parametrized since there's a fair bit of repetition.

sample_elements = ["H", "Fe-56", "p+", "alpha", "Fe", "D+", "T 1-"]
not_element = ["e-", "e+", "n", "mu-", "tau+"]

sample_isotopes = ["D", "T", "alpha", "proton", "Fe-56", "Be-8"]
not_isotope = ["H", "e-", "n", "p-", "e+", "Fe", "Au", "Og"]

sample_ions = ["p+", "D+", "T+", "alpha", "Be-8+", "Fe 26+"]
not_ion = ["D", "T", "H-1", "He-4", "e-", "e+", "n"]


@particle_input
def function_with_element_argument(element: ParticleLike) -> Particle:
    """A function decorated with `~plasmapy.particles.particle_input`
    where the argument annotated with `~plasmapy.particles.Particle`
    is named `element`.  This function should raise an
    `~plasmapy.utils.InvalidElementError` when the argument is not
    an element, isotope, or ion."""
    return element


@particle_input
def function_with_isotope_argument(isotope: ParticleLike) -> Particle:
    """A function decorated with `~plasmapy.particles.particle_input`
    where the argument annotated with `~plasmapy.particles.Particle`
    is named `isotope`.  This function should raise an
    `~plasmapy.utils.InvalidIsotopeError` when the argument is not an
    isotope or an ion of an isotope."""
    return isotope


@particle_input
def function_with_ion_argument(ion: ParticleLike) -> Particle:
    """
    A function decorated with `~plasmapy.particles.particle_input`
    where the argument annotated with `~plasmapy.particles.Particle`
    is named `ion`.  This function should raise an
    `~plasmapy.utils.InvalidIonError` when the argument is not an
    ion.
    """
    return ion


@pytest.mark.parametrize("element", sample_elements)
def test_is_element(element):
    """
    Test that particle_input will not raise an
    `~plasmapy.utils.InvalidElementError` if the annotated argument is
    named 'element' and is assigned values that are elements, isotopes,
    or ions.
    """
    particle = function_with_element_argument(element)
    assert particle.is_category("element")


@pytest.mark.parametrize("particle", not_element)
def test_not_element(particle):
    """
    Test that particle_input will raise an
    `~plasmapy.utils.InvalidElementError` if an argument decorated with
    `~plasmapy.particles.Particle` is named 'element', but the annotated
    argument ends up not being an element, isotope, or ion.
    """
    with pytest.raises(InvalidElementError):
        function_with_element_argument(particle)
        pytest.fail(
            "@particle_input is not raising an InvalidElementError for "
            f"{repr(particle)} even though the annotated argument is "
            f"named 'element'."
        )


@pytest.mark.parametrize("isotope", sample_isotopes)
def test_is_isotope(isotope):
    """
    Test that particle_input will not raise an
    `~plasmapy.utils.InvalidIsotopeError` if the annotated argument is
    named 'isotope' and is assigned values that are isotopes or
    ions of isotopes."""
    particle = function_with_isotope_argument(isotope)
    assert particle.is_category("isotope")


@pytest.mark.parametrize("particle", not_isotope)
def test_not_isotope(particle):
    """
    Test that particle_input will raise an
    `~plasmapy.utils.InvalidIsotopeError` if an argument decorated with
    `~plasmapy.particles.Particle` is named 'isotope', but the annotated
    argument ends up not being an isotope or an ion of an isotope.
    """
    with pytest.raises(InvalidIsotopeError):
        function_with_isotope_argument(particle)
        pytest.fail(
            "@particle_input is not raising an InvalidIsotopeError for "
            f"{repr(particle)} even though the annotated argument is named "
            "'isotope'."
        )


@pytest.mark.parametrize("ion", sample_ions)
def test_is_ion(ion):
    """
    Test that particle_input will not raise an
    `~plasmapy.utils.InvalidIonError` if the annotated argument is
    named 'ion' and is assigned values that are ions.
    """
    particle = function_with_ion_argument(ion)
    assert particle.is_category("ion")


@pytest.mark.parametrize("particle", not_ion)
def test_not_ion(particle):
    """
    Test that particle_input will raise an
    `~plasmapy.utils.InvalidIonError` if an argument decorated with
    `~plasmapy.particles.Particle` is named 'ion', but the annotated
    argument ends up not being an ion.
    """
    with pytest.raises(InvalidIonError):
        function_with_ion_argument(particle)
        pytest.fail(
            "@particle_input is not raising an InvalidIonError for "
            f"{repr(particle)} even though the annotated argument is named "
            "'ion'."
        )


def undecorated_function(particle: ParticleLike, distance: u.m):
    return particle, distance


# Both particle_input and validate_quantities can be used with or
# without arguments, so the following list is to test the cases where

decorator_pairs = [
    (particle_input, validate_quantities),
    (particle_input(), validate_quantities),
    (particle_input, validate_quantities()),
    (particle_input(), validate_quantities()),
]


@pytest.mark.parametrize("decorator1, decorator2", decorator_pairs)
def test_stacking_decorators(decorator1, decorator2):
    """
    Test that particle_input and validate_quantities behave as expected.
    """
    decorated_function_1_2 = decorator1(decorator2(undecorated_function))
    decorated_function_2_1 = decorator2(decorator1(undecorated_function))

    particle_1_2, distance_1_2 = decorated_function_1_2("p+", distance=3 * u.cm)
    particle_2_1, distance_2_1 = decorated_function_2_1("p+", distance=3 * u.cm)

    # Test that particle_input is working as expected
    assert isinstance(particle_1_2, Particle)
    assert isinstance(particle_2_1, Particle)
    assert particle_1_2 == particle_2_1 == "p+"

    # Test that validate_quantities is working as expected
    assert distance_1_2.unit == distance_2_1.unit == u.m
    assert distance_1_2 == distance_2_1 == 3 * u.cm


@pytest.mark.parametrize("decorator1, decorator2", decorator_pairs)
def test_preserving_signature_with_stacked_decorators(decorator1, decorator2):
    """
    Test that particle_input & validate_quantities preserve the function
    signature after being stacked.
    """
    decorated_function_1_2 = decorator1(decorator2(undecorated_function))
    decorated_function_2_1 = decorator2(decorator1(undecorated_function))

    undecorated_signature = inspect.signature(undecorated_function)
    decorated_signature_1_2 = inspect.signature(decorated_function_1_2)
    decorated_signature_2_1 = inspect.signature(decorated_function_2_1)

    assert undecorated_signature == decorated_signature_1_2 == decorated_signature_2_1


def test_annotated_classmethod():
    """
    Test that `particle_input` behaves as expected for a method that is
    decorated with `classmethod`.
    """

    class HasAnnotatedClassMethod:
        @particle_input
        @classmethod
        def f(cls, particle: ParticleLike):
            return particle

    has_annotated_classmethod = HasAnnotatedClassMethod()
    assert has_annotated_classmethod.f("p+") == Particle("p+")


@pytest.mark.parametrize(
    "inner_decorator, outer_decorator",
    [
        (particle_input, particle_input),
        (particle_input, particle_input()),
        (particle_input(), particle_input),
        (particle_input(), particle_input()),
    ],
)
def test_self_stacked_decorator(outer_decorator, inner_decorator):
    """Test that particle_input can be stacked with itself."""

    @outer_decorator
    @inner_decorator
    def f(x, particle: ParticleLike):
        return particle

    result = f(1, "p+")
    assert result == "p+"
    assert isinstance(result, Particle)


validate_quantities_ = validate_quantities(
    T_e={"equivalencies": u.temperature_energy()}
)


def test_annotated_init():
    """Test that `particle_input` can decorate an __init__ method."""

    class HasAnnotatedInit:
        @particle_input(require="element")
        def __init__(self, particle: ParticleLike, ionic_fractions=None):
            self.particle = particle

    x = HasAnnotatedInit("H-1", ionic_fractions=32)
    assert x.particle == "H-1"


@pytest.mark.parametrize(
    "outer_decorator, inner_decorator",
    [
        (particle_input, validate_quantities_),
        (particle_input(), validate_quantities_),
        pytest.param(validate_quantities_, particle_input, marks=pytest.mark.xfail),
        pytest.param(validate_quantities_, particle_input(), marks=pytest.mark.xfail),
    ],
)
def test_particle_input_with_validate_quantities(outer_decorator, inner_decorator):
    """Test that particle_input can be stacked with validate_quantities."""

    class C:
        @outer_decorator
        @inner_decorator
        def __init__(
            self,
            particle: ParticleLike,
            T_e: u.K = None,
        ):
            self.particle = particle
            self.T_e = T_e

    instance = C("p+", T_e=3.8 * u.eV)

    assert instance.particle == "p+"
    assert isinstance(instance.particle, Particle)

    assert instance.T_e.unit == u.K


def test_allow_custom_particles_is_true():
    """Test the allow_custom_particles keyword argument to particle_input."""

    @particle_input(allow_custom_particles=False)
    def f(particle: ParticleLike):
        return particle

    custom_particle = CustomParticle()

    with pytest.raises(InvalidParticleError):
        f(custom_particle)


def test_allow_custom_particles_is_true():
    """Test the allow_custom_particles keyword argument to particle_input."""

    @particle_input(allow_custom_particles=False)
    def f(particle: ParticleLike):
        return particle

    custom_particle = CustomParticle()

    with pytest.raises(InvalidParticleError):
        f(custom_particle)
