import astropy.units as u
import numpy as np
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
from plasmapy.particles.particle_collections import ParticleList


@particle_input
def func_decorated():
    return 42


@particle_input
def func_decorated_args(arg1, arg2):
    return arg1 + arg2


@particle_input()
def func_decorated_kwargs(kwarg1=None, kwarg2=None):
    return kwarg1 + kwarg2


@particle_input
def func_decorated_args_kwargs(
    argu1: int, argu2, *, kwargu1: float = None, kwargu2=None, default_kwarg=5
):
    return (argu1, argu2, kwargu1, kwargu2, default_kwarg)


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
            f"The argument particle in func_simple_noparens is not a Particle"
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
            f"The argument particle in func_simple_parens is not a Particle"
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
            f"An exception was raised while trying to execute "
            f"{func.__name__} with args = {args} and kwargs = {kwargs}."
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


class Test_particle_input:
    """
    A sample class with methods that will be used to check that
    @particle_input works both with and without parentheses.
    """

    @particle_input
    def method_noparens(self, particle: ParticleLike):
        return particle

    @particle_input()
    def method_parens(self, particle: ParticleLike):
        return particle


def test_particle_input_classes():
    instance = Test_particle_input()

    symbol = "muon"
    expected = Particle(symbol)

    try:
        result_noparens = instance.method_noparens(symbol)
    except Exception as e:
        raise ParticleError("Problem with method_noparens") from e

    try:
        result_parens = instance.method_parens(symbol)
    except Exception as e:
        raise ParticleError("Problem with method_parens") from e

    assert result_parens == result_noparens == expected


@particle_input
def function_with_no_annotations():
    """A trivial function that is incorrectly decorated with
    particle_input because no arguments are annotated with Particle."""
    pass


@pytest.mark.xfail
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


@pytest.mark.parametrize("mass_numb, Z", [(4, None), (None, 1), (4, 1)])
def test_function_with_ambiguity(mass_numb, Z):
    """Test that a function decorated with particle_input that has two
    annotated arguments"""
    with pytest.raises(ParticleError):
        ambiguous_keywords("H", "He", Z=Z, mass_numb=mass_numb)


def function_to_test_annotations(particles: Union[Tuple, List], resulting_particles):
    """
    Test that a function with an argument annotated with (Particle,
    Particle, ...) or [Particle] returns a tuple of expected Particle
    instances.

    Parameters
    ----------
    particles: tuple or list
        A collection containing many items, each of which may be a valid
        representation of a particle or a `~plasmapy.particles.Particle`
        instance

    resulting_particles
        TODO: Add line here
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
        [isinstance(p, Particle) for p in resulting_particles]
    )
    returned_correct_instances = all(
        [expected[i] == resulting_particles[i] for i in range(len(particles))]
    )

    if not returned_particle_instances:
        raise ParticleError(
            f"A function decorated by particle_input did not return "
            f"a collection of Particle instances for input of "
            f"{repr(particles)}, and instead returned "
            f"{repr(resulting_particles)} of types "
            f"{[type(p) for p in resulting_particles]}"
        )

    if not returned_correct_instances:
        raise ParticleError(
            f"A function decorated by particle_input did not return "
            f"{repr(expected)} as expected, and instead returned "
            f"{repr(resulting_particles)}."
        )


@particle_input
def function_with_tuple_annotation(particles: (ParticleLike, ParticleLike), q, x=5):
    return particles


tuple_annotation_test_table = [
    ("e-", "e+"),
    (Particle("alpha"), Particle("Fe 6+")),
    ("e+", Particle("p+")),
    ["nu_mu", "anti_nu_mu"],
]


@pytest.mark.parametrize("particles", tuple_annotation_test_table)
def test_tuple_annotation(particles: Union[Tuple, List]):
    try:
        resulting_particles = function_with_tuple_annotation(
            particles, "ignore", x="ignore"
        )
    except Exception as exc2:
        raise ParticleError(
            f"Unable to evaluate a function decorated by particle_input"
            f" with an annotation of (Particle, Particle) for inputs of"
            f" {repr(particles)}."
        ) from exc2

    function_to_test_annotations(particles, resulting_particles)


@particle_input
def function_with_list_annotation(particles: ParticleList, q, x=5):
    print(particles)
    print(type(particles[0]))
    return particles


list_annotation_test_table = [
    ("e-", "e+"),
    ("alpha", Particle("Fe 2-"), 3),
    ("nu_mu",),
    ["e+", "p+", "anti_nu_mu"],
]


@pytest.mark.parametrize("particles", list_annotation_test_table)
def test_list_annotation(particles):
    try:
        resulting_particles = function_with_list_annotation(
            particles, "ignore", x="ignore"
        )
    except Exception as exc2:
        raise ParticleError(
            f"Unable to evaluate a function decorated by particle_input"
            f" with an annotation of ParticleList for inputs of"
            f" {repr(particles)}."
        ) from exc2

    print(resulting_particles)
    print(type(resulting_particles))
    print(type(resulting_particles[0]))

    function_to_test_annotations(particles, resulting_particles)


class TestOptionalArgs:
    def particle_iter(self, particles):
        return [Particle(particle) for particle in particles]

    def test_optional_particle(self):
        particle = "He"

        @particle_input
        def optional_particle(particle: ParticleLike = particle):
            return particle

        assert optional_particle() == Particle(particle)
        assert optional_particle("Ne") == Particle("Ne")

    def test_optional_tuple(self):
        tuple_of_particles = ("Mg", "Al")

        @particle_input
        def optional_tuple(particles: (Particle, Particle) = tuple_of_particles):
            return particles

        function_to_test_annotations(
            optional_tuple(), self.particle_iter(tuple_of_particles)
        )
        elements = ("C", "N")
        function_to_test_annotations(
            optional_tuple(elements), self.particle_iter(elements)
        )

    def test_optional_list(self):
        list_of_particles = ("Ca", "Ne")

        @particle_input
        def optional_list(particles: ParticleList = list_of_particles):
            return particles

        function_to_test_annotations(
            optional_list(), self.particle_iter(list_of_particles)
        )
        elements = ("Na", "H", "C")
        function_to_test_annotations(
            optional_list(elements), self.particle_iter(elements)
        )


def test_invalid_number_of_tuple_elements():
    with pytest.raises(ValueError):
        # Passed 3 elements when function only takes 2 in tuple
        function_with_tuple_annotation(("e+", "e-", "alpha"), q="test")


@pytest.mark.parametrize(
    "particle_list_like_input",
    [ParticleList([]), ParticleList(["He"]), ("Li", "Be"), ["e+", "alpha"],],
)
def test_unexpected_tuple_and_list_argument_types(particle_list_like_input):
    @particle_input(allow_particle_lists=False)
    def take_particle(particle: ParticleLike):
        return particle

    with pytest.raises(TypeError):
        take_particle(particle_list_like_input)


# decorator_kwargs, particle, expected_exception
decorator_categories_table = [
    ({"exclude": {"element"}}, "Fe", ParticleError),
    ({"any_of": {"lepton", "antilepton"}}, "tau-", None),
    ({"require": {"isotope", "ion"}}, "Fe-56+", None),
    ({"require": {"isotope", "ion"}}, "Fe+", ParticleError),
    ({"any_of": {"isotope", "ion"}}, "Fe+", None),
    ({"any_of": {"charged", "uncharged"}}, "Fe", ParticleError),
    ({"any_of": ["charged", "uncharged"]}, "Fe", ParticleError),
    ({"any_of": ("charged", "uncharged")}, "Fe", ParticleError),
    ({"require": "charged"}, "Fe 0+", ParticleError),
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
    else:
        decorated_function(particle)


def test_optional_particles():
    """Tests the `none_shall_pass` keyword argument in is_particle.
    If `none_shall_pass=True`, then an annotated argument should allow
    `None` to be passed through to the decorated function."""

    @particle_input
    def func_none_shall_pass(particle: Optional[ParticleLike]):
        return particle

    @particle_input
    def func_none_shall_pass_with_tuple(
        particles: (Optional[ParticleLike], Optional[ParticleLike])
    ):
        return particles

    @particle_input
    def func_none_shall_pass_with_list(particles: Optional[ParticleList]):
        return particles

    assert func_none_shall_pass(None) is None, (
        "The none_shall_pass keyword in the particle_input decorator is set "
        "to True, but is not passing through None."
    )

    assert func_none_shall_pass_with_tuple(None) == None, (
        "The none_shall_pass keyword in the particle_input decorator is set "
        "to True, but is not passing through None."
    )

    assert func_none_shall_pass_with_list(None) == None, (
        "The none_shall_pass keyword in the particle_input decorator is set "
        "to True, but is not passing through None."
    )


@pytest.mark.xfail
def test_none_shall_not_pass():
    """Tests the `none_shall_pass` keyword argument in is_particle.
    If `none_shall_pass=False`, then particle_input should raise a
    `TypeError` if an annotated argument is assigned the value of
    `None`."""

    @particle_input
    def func_none_shall_not_pass(particle: ParticleLike):
        return particle

    @particle_input
    def func_required_particle_pair(particles: (ParticleLike, ParticleLike)):
        return particles

    @particle_input
    def func_required_particle_list(particles: Optional[ParticleList]):
        return particles

    with pytest.raises(TypeError):
        func_none_shall_not_pass(None)
        pytest.fail(
            "The none_shall_pass keyword in the particle_input "
            "decorator is set to False, but is not raising a TypeError."
        )

    with pytest.raises(TypeError):
        func_required_particle_pair(("He", None))
        pytest.fail(
            "The none_shall_pass keyword in the particle_input "
            "decorator is set to False, but is not raising a TypeError."
        )

    with pytest.raises(TypeError):
        func_required_particle_list(("He", None))
        pytest.fail(
            "The none_shall_pass keyword in the particle_input "
            "decorator is set to False, but is not raising a TypeError."
        )


def test_optional_particle_annotation_argname():
    """Tests the `Optional[Particle]` annotation argument in a function
    decorated by `@particle_input` such that the annotated argument allows
    `None` to be passed through to the decorated function."""

    @particle_input
    def func_optional_particle(particle: Optional[ParticleLike]):
        return particle

    assert func_optional_particle(None) is None, (
        "The particle keyword in the particle_input decorator is set "
        "to accept Optional[Particle], but is not passing through None."
    )


@pytest.mark.xfail
def test_not_optional_particle_annotation_argname():
    """Tests the `Optional[Particle]` annotation argument in a function
    decorated by `@particle_input` such that the annotated argument does
    not allows `None` to be passed through to the decorated function."""

    @particle_input
    def func_not_optional_particle_with_tuple(
        particles: (ParticleLike, Optional[ParticleLike])
    ) -> (ParticleLike, Optional[ParticleLike]):
        return particles

    @particle_input
    def func_not_optional_particle_with_list(
        particles: [ParticleLike],
    ) -> [ParticleLike]:
        return particles

    with pytest.raises(TypeError):
        func_not_optional_particle_with_tuple((None, "He"))
        pytest.fail(
            "The particle keyword in the particle_input decorator "
            "received None instead of a Particle, but is not raising a "
            "TypeError."
        )

    with pytest.raises(TypeError):

        func_not_optional_particle_with_list(("He", None))
        pytest.fail(
            "The particle keyword in the particle_input decorator "
            "received None instead of a Particle, but is not raising a "
            "TypeError."
        )


# TODO: The following tests might be able to be cleaned up and/or
# further parametrized since there's a fair bit of repetition.

is_element = ["H", "Fe-56", "p+", "alpha", "Fe", "D+", "T 1-"]
not_element = ["e-", "e+", "n", "mu-", "tau+"]

is_isotope = ["D", "T", "alpha", "proton", "Fe-56", "Be-8"]
not_isotope = ["H", "e-", "n", "p-", "e+", "Fe", "Au", "Og"]

is_ion = ["p+", "D+", "T+", "alpha", "Be-8+", "Fe 26+"]
not_ion = ["D", "T", "H-1", "He-4", "e-", "e+", "n"]


@particle_input
def function_with_element_argument(element: ParticleLike):
    """A function decorated with `~plasmapy.particles.particle_input`
    where the argument annotated with `~plasmapy.particles.Particle`
    is named `element`.  This function should raise an
    `~plasmapy.utils.InvalidElementError` when the argument is not
    an element, isotope, or ion."""
    return element


@particle_input
def function_with_isotope_argument(isotope: ParticleLike):
    """A function decorated with `~plasmapy.particles.particle_input`
    where the argument annotated with `~plasmapy.particles.Particle`
    is named `isotope`.  This function should raise an
    `~plasmapy.utils.InvalidIsotopeError` when the argument is not an
    isotope or an ion of an isotope."""
    return isotope


@particle_input
def function_with_ion_argument(ion: ParticleLike):
    """
    A function decorated with `~plasmapy.particles.particle_input`
    where the argument annotated with `~plasmapy.particles.Particle`
    is named `ion`.  This function should raise an
    `~plasmapy.utils.InvalidIonError` when the argument is not an
    ion.
    """
    return ion


@pytest.mark.parametrize("element", is_element)
def test_is_element(element):
    """
    Test that particle_input will not raise an
    `~plasmapy.utils.InvalidElementError` if the annotated argument is
    named 'element' and is assigned values that are elements, isotopes,
    or ions.
    """
    function_with_element_argument(element)


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


@pytest.mark.parametrize("isotope", is_isotope)
def test_is_isotope(isotope):
    """
    Test that particle_input will not raise an
    `~plasmapy.utils.InvalidIsotopeError` if the annotated argument is
    named 'isotope' and is assigned values that are isotopes or
    ions of isotopes."""
    function_with_isotope_argument(isotope)


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


@pytest.mark.parametrize("ion", is_ion)
def test_is_ion(ion):
    """
    Test that particle_input will not raise an
    `~plasmapy.utils.InvalidIonError` if the annotated argument is
    named 'ion' and is assigned values that are ions.
    """
    function_with_ion_argument(ion)


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


@particle_input
def return_particle(particle: ParticleLike):
    return particle


@pytest.mark.parametrize(
    "attribute, quantity_like", [("mass", 5 * u.kg), ("charge", 2 * u.C),],
)
def test_quantity_as_particle_like(attribute, quantity_like):
    particle = return_particle(quantity_like)
    assert isinstance(particle, (CustomParticle, ParticleList))
    assert getattr(particle, attribute) == quantity_like
