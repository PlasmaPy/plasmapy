"""Tests for `plasmapy.particles.decorators`."""

import inspect
import sys
from collections.abc import Callable, Iterable
from typing import Any

import astropy.constants as const
import astropy.units as u
import pytest

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
from plasmapy.particles.particle_collections import ParticleList, ParticleListLike
from plasmapy.utils.code_repr import call_string
from plasmapy.utils.decorators.validators import validate_quantities
from plasmapy.utils.exceptions import PlasmaPyDeprecationWarning


@particle_input
def function_decorated_with_particle_input(
    a: Any,
    particle: ParticleLike,
    b: Any = None,
    Z: float | None = None,
    mass_numb: int | None = None,
) -> Particle | CustomParticle | ParticleList:
    """
    A simple function that is decorated with `particle_input` and
    returns the particle instance corresponding to the inputs.

    .. important::

       For this function, `particle_input` **is not** called (i.e.,
       there are no parentheses after the decorator).
    """
    if not isinstance(particle, Particle):
        raise TypeError(
            f"The argument `particle` should be a Particle. Instead, {particle = }."
        )
    return particle


@particle_input()
def function_decorated_with_call_of_particle_input(
    a: Any,
    particle: ParticleLike,
    b: Any = None,
    Z: float | None = None,
    mass_numb: int | None = None,
) -> Particle | CustomParticle | ParticleList:
    """
    A simple function that is decorated with `@particle_input()` and
    returns the Particle instance corresponding to the inputs.

    .. important::

       For this function, `particle_input` **is** called (i.e., there
       are parentheses after the decorator).
    """
    if not isinstance(particle, Particle):
        raise TypeError(
            f"The argument `particle` should be a Particle. Instead, {particle=}."
        )
    return particle


particle_input_simple_table: list[tuple[tuple[int | str, ...], dict[str, Any], str]] = [
    ((1, "p+"), {"b": 2}, "p+"),
    ((1, "Fe"), {"mass_numb": 56, "Z": 3}, "Fe-56 3+"),
    ((1,), {"particle": "e-"}, "e-"),
    ((1,), {"particle": Particle("e-")}, "e-"),
]


@pytest.mark.parametrize(
    "func",
    [
        function_decorated_with_call_of_particle_input,
        function_decorated_with_particle_input,
    ],
)
@pytest.mark.parametrize(("args", "kwargs", "symbol"), particle_input_simple_table)
def test_particle_input_simple(
    func: Callable[..., Any],
    args: tuple[int | str, ...],
    kwargs: dict[str, Any],
    symbol: str,
) -> None:
    """
    Test that simple functions decorated by particle_input correctly
    return the correct Particle object.
    """
    expected = Particle(symbol)
    result = func(*args, **kwargs)
    assert result == expected, (
        f"The result {result!r} does not equal the expected value of "
        f"{expected!r} with {func =}, {args =}, and {kwargs =}."
    )


# func, kwargs, expected_error
particle_input_error_table = [
    (
        function_decorated_with_particle_input,
        {"a": 1, "particle": "invalid particle"},
        InvalidParticleError,
    ),
    (
        function_decorated_with_particle_input,
        {"a": 1, "particle": 5 * u.m},
        InvalidParticleError,
    ),
    (
        function_decorated_with_particle_input,
        {"a": 1, "particle": "He-4", "Z": 2.00001},
        InvalidParticleError,
    ),
    (
        function_decorated_with_particle_input,
        {"a": 1, "particle": "He-4", "Z": 1 + 1j},
        TypeError,
    ),
]


@pytest.mark.parametrize(
    ("func", "kwargs", "expected_error"), particle_input_error_table
)
def test_particle_input_errors(
    func: Callable[..., Any], kwargs: dict[str, Any], expected_error: type
) -> None:
    """
    Test that functions decorated with `@particle_input` raise the
    expected errors.
    """
    with pytest.raises(expected_error):
        func(**kwargs)
        pytest.fail(f"{func} did not raise {expected_error} with kwargs = {kwargs}")


class ClassWithDecoratedMethods:
    """
    A sample class with methods that will be used to check that
    @particle_input works both with and without parentheses.
    """

    @particle_input
    def method_decorated_with_no_parentheses(
        self, particle: ParticleLike
    ) -> Particle | CustomParticle | ParticleList:
        # Because run-time logic is needed, mypy is unable to infer
        # that @particle_input is doing a type conversion here. We
        # would need a dynamic type checker to address this case.
        return particle  # type: ignore[return-value]

    @particle_input()
    def method_decorated_with_parentheses(
        self, particle: ParticleLike
    ) -> Particle | CustomParticle | ParticleList:
        return particle  # type: ignore[return-value]


@pytest.mark.parametrize("symbol", ["muon", "He 2+"])
def test_particle_input_classes(symbol: str) -> None:
    """Test that @particle_input works for instance methods."""
    expected = Particle(symbol)
    instance = ClassWithDecoratedMethods()

    result_noparens = instance.method_decorated_with_no_parentheses(symbol)
    result_parens = instance.method_decorated_with_parentheses(symbol)

    assert isinstance(result_noparens, Particle)
    assert isinstance(result_parens, Particle)

    assert result_parens == result_noparens == expected


def test_no_annotations_exception() -> None:
    """
    Test that a function decorated with @particle_input that has no
    annotated arguments will raise a ParticleError.
    """

    @particle_input
    def function_with_no_annotations() -> None:
        pass

    with pytest.raises(ParticleError):
        function_with_no_annotations()


@particle_input
def ambiguous_keywords(
    p1: ParticleLike,
    p2: ParticleLike,
    Z: float | None = None,
    mass_numb: int | None = None,
) -> None:
    """
    A trivial function with two annotated arguments plus the keyword
    arguments ``Z`` and ``mass_numb``.
    """


ambiguous_arguments = [
    [("H", "He"), {"Z": 1, "mass_numb": 4}],
    [("H", "He"), {"Z": 1}],
    [("H", "He"), {"mass_numb": 4}],
]


@pytest.mark.parametrize(("args", "kwargs"), ambiguous_arguments)
def test_function_with_ambiguity(args: tuple[str, ...], kwargs: dict[str, int]) -> None:
    """
    Test that a function decorated with particle_input that has two
    annotated arguments along with `Z` and `mass_numb` raises an
    exception because it is not clear which argument that `Z` and
    `mass_numb` should belong to.
    """
    with pytest.raises(ParticleError):
        ambiguous_keywords(*args, **kwargs)


def test_optional_particle() -> None:
    particle = "He"

    @particle_input
    def has_default_particle(
        particle: ParticleLike = particle,
    ) -> Particle | CustomParticle | ParticleList:
        return particle  # type: ignore[return-value]

    assert has_default_particle() == Particle(particle)
    assert has_default_particle("Ne") == Particle("Ne")


categorization_particle_exception = [
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
    ("categorization", "particle", "exception"), categorization_particle_exception
)
def test_decorator_categories(
    categorization: dict[str, str | Iterable[str]],
    particle: ParticleLike,
    exception: type,
) -> None:
    """
    Test that the ``require``, ``any_of``, and ``exclude`` categories
    lead to a |ParticleError| being raised when an inputted particle
    does not meet the required categorization criteria, and do not lead
    to a |ParticleError| when the inputted particle matches the criteria.
    """

    @particle_input(**categorization)  # type: ignore[arg-type]
    def decorated_function(
        argument: ParticleLike,
    ) -> Particle | CustomParticle | ParticleList:
        return argument  # type: ignore[return-value]

    if exception:
        with pytest.raises(exception):
            decorated_function(particle)
            pytest.fail(
                f"{call_string(decorated_function, [], categorization)} "
                f"did not raise {exception}"
            )
    else:
        decorated_function(particle)


def test_optional_particle_annotation_parameter() -> None:
    """
    Tests the `Optional[Particle]` annotation argument in a function
    decorated by `@particle_input` such that the annotated argument allows
    `None` to be passed through to the decorated function.
    """

    @particle_input
    def func_optional_particle(particle: ParticleLike | None) -> Particle | None:
        return particle  # type: ignore[return-value]

    assert func_optional_particle(None) is None, (
        "The particle keyword in the particle_input decorator is set "
        "to accept Optional[ParticleLike], but is not passing through "
        "None."
    )


def undecorated_function(
    particle: ParticleLike, distance: u.Quantity[u.m]
) -> tuple[ParticleLike, u.Quantity[u.m]]:
    return particle, distance


decorator_pairs = [
    (particle_input, validate_quantities),
    (particle_input(), validate_quantities),
    (particle_input, validate_quantities()),  # type: ignore[no-untyped-call]
    (particle_input(), validate_quantities()),  # type: ignore[no-untyped-call]
]


@pytest.mark.parametrize(("decorator1", "decorator2"), decorator_pairs)
def test_stacking_decorators(
    decorator1: Callable[..., Any], decorator2: Callable[..., Any]
) -> None:
    """
    Test that particle_input and validate_quantities can be stacked in
    either order with or without parentheses.
    """
    decorated_function_1_2 = decorator1(decorator2(undecorated_function))
    decorated_function_2_1 = decorator2(decorator1(undecorated_function))

    particle_1_2, distance_1_2 = decorated_function_1_2("p+", distance=3 * u.cm)
    particle_2_1, distance_2_1 = decorated_function_2_1("p+", distance=3 * u.cm)

    assert isinstance(particle_1_2, Particle)
    assert isinstance(particle_2_1, Particle)
    assert particle_1_2 == particle_2_1 == "p+"

    assert distance_1_2.unit == distance_2_1.unit == u.m
    assert distance_1_2 == distance_2_1 == 3 * u.cm


@pytest.mark.parametrize(("decorator1", "decorator2"), decorator_pairs)
def test_preserving_signature_with_stacked_decorators(
    decorator1: Callable[..., Any], decorator2: Callable[..., Any]
) -> None:
    """
    Test that |particle_input| & |validate_quantities| preserve the
    function signature after being stacked.
    """
    decorated_function_1_2 = decorator1(decorator2(undecorated_function))
    decorated_function_2_1 = decorator2(decorator1(undecorated_function))

    undecorated_signature = inspect.signature(undecorated_function)
    decorated_signature_1_2 = inspect.signature(decorated_function_1_2)
    decorated_signature_2_1 = inspect.signature(decorated_function_2_1)

    assert undecorated_signature == decorated_signature_1_2 == decorated_signature_2_1


@pytest.mark.skipif(
    not sys.version_info < (3, 13),
    reason="Class methods can no longer wrap other descriptors. See #2873.",
)
def test_annotated_classmethod() -> None:
    """
    Test that `particle_input` behaves as expected for a method that is
    decorated with `classmethod`.
    """

    class HasAnnotatedClassMethod:
        @classmethod
        @particle_input
        def f(cls, particle: ParticleLike) -> Particle | CustomParticle | ParticleList:
            return particle  # type: ignore[return-value]

    has_annotated_classmethod = HasAnnotatedClassMethod()
    assert has_annotated_classmethod.f("p+") == Particle("p+")


@pytest.mark.parametrize("outer_decorator", [particle_input, particle_input()])
@pytest.mark.parametrize("inner_decorator", [particle_input, particle_input()])
def test_self_stacked_decorator(
    outer_decorator: Callable[..., Any], inner_decorator: Callable[..., Any]
) -> None:
    """Test that particle_input can be stacked with itself."""

    @outer_decorator
    @inner_decorator
    def f(x: Any, particle: ParticleLike) -> Particle | CustomParticle | ParticleList:
        return particle  # type: ignore[return-value]

    result = f(1, "p+")
    assert result == "p+"
    assert isinstance(result, Particle)


@pytest.mark.parametrize("outer_decorator", [particle_input, particle_input()])
@pytest.mark.parametrize("inner_decorator", [particle_input, particle_input()])
def test_class_stacked_decorator(
    outer_decorator: Callable[..., Any], inner_decorator: Callable[..., Any]
) -> None:
    """
    Test that particle_input can be stacked with itself for an
    instance method.
    """

    class Sample:
        @outer_decorator
        @inner_decorator
        def __init__(self, particle: ParticleLike) -> None:
            self.particle = particle

    result = Sample("p+")
    assert result.particle == "p+"
    assert isinstance(result.particle, Particle)


validate_quantities_ = validate_quantities(  # type: ignore[no-untyped-call]
    T_e={"equivalencies": u.temperature_energy()}
)


def test_annotated_init() -> None:
    """Test that `particle_input` can decorate an __init__ method."""

    class HasAnnotatedInit:
        @particle_input(require="element")
        def __init__(self, particle: ParticleLike, ionic_fractions: Any = None) -> None:
            self.particle = particle

    x = HasAnnotatedInit("H-1", ionic_fractions=32)
    assert x.particle == "H-1"


@pytest.mark.parametrize(
    ("outer_decorator", "inner_decorator"),
    [
        (particle_input, validate_quantities_),
        (particle_input(), validate_quantities_),
        pytest.param(
            validate_quantities_,
            particle_input,
            marks=pytest.mark.xfail(
                reason="For instance methods, particle_input must currently "
                "be the outer decorator. See #2035."
            ),
        ),
        pytest.param(
            validate_quantities_,
            particle_input(),
            marks=pytest.mark.xfail(
                reason="For instance methods, particle_input must currently "
                "be the outer decorator. See #2035."
            ),
        ),
    ],
)
def test_particle_input_with_validate_quantities(
    outer_decorator: Callable[..., Any],
    inner_decorator: Callable[..., Any],
) -> None:
    """Test that particle_input can be stacked with validate_quantities."""

    class C:
        @outer_decorator
        @inner_decorator
        def __init__(
            self,
            particle: ParticleLike,
            T_e: u.Quantity[u.K] = None,
        ) -> None:
            self.particle = particle
            self.T_e = T_e

    instance = C("p+", T_e=3.8 * u.eV)

    assert instance.particle == "p+"
    assert isinstance(instance.particle, Particle)

    assert instance.T_e.unit == u.K


kwargs_to_decorator_and_args = [
    ({"allow_custom_particles": False}, CustomParticle()),
    ({"allow_particle_lists": False}, ParticleList(["p+", "Fe"])),
    (
        {"allow_custom_particles": False, "allow_particle_lists": True},
        ParticleList([CustomParticle()]),
    ),
]


@pytest.mark.parametrize(
    ("kwargs_to_particle_input", "arg"), kwargs_to_decorator_and_args
)
def test_particle_input_verification(
    kwargs_to_particle_input: dict[str, Any], arg: Any
) -> None:
    """Test the allow_custom_particles keyword argument to particle_input."""

    @particle_input(**kwargs_to_particle_input)
    def f(particle: ParticleLike) -> Particle | CustomParticle | ParticleList:
        return particle  # type: ignore[return-value]

    with pytest.raises(ParticleError):
        f(arg)


class ParameterNamesCase:
    """
    A class to store test case information for when particle_input is
    used and the parameters are given special names.
    """

    def __init__(
        self,
        category: str,
        function: Callable[..., Any],
        particles_in_category: list[ParticleLike],
        particles_not_in_category: list[ParticleLike],
        exception: type,
    ) -> None:
        self.category = category
        self.function = function
        self.particles_in_category = particles_in_category
        self.particles_not_in_category = particles_not_in_category
        self.particles_some_in_category = (
            particles_in_category + particles_not_in_category
        )
        self.exception = exception


@particle_input
def get_element(element: ParticleLike) -> Particle | CustomParticle | ParticleList:
    return element  # type: ignore[return-value]


@particle_input
def get_isotope(isotope: ParticleLike) -> Particle | CustomParticle | ParticleList:
    return isotope  # type: ignore[return-value]


@particle_input
def get_ion(
    ion: ParticleLike, Z: float | None = None
) -> Particle | CustomParticle | ParticleList:
    return ion  # type: ignore[return-value]


cases = [
    ParameterNamesCase(
        category="element",
        function=get_element,
        particles_in_category=["H", "Fe-56", "p+", "alpha", "Fe", "D+", "T 1-"],
        particles_not_in_category=["e-", "e+", "n", "mu-", "tau+"],
        exception=InvalidElementError,
    ),
    ParameterNamesCase(
        category="isotope",
        function=get_isotope,
        particles_in_category=["D", "T", "alpha", "proton", "Fe-56", "Be-8"],
        particles_not_in_category=["H", "e-", "n", "p-", "e+", "Fe", "Au", "Og"],
        exception=InvalidIsotopeError,
    ),
    ParameterNamesCase(
        category="ion",
        function=get_ion,
        particles_in_category=["p+", "D+", "T+", "alpha", "Be-8+", "Fe 26+"],
        particles_not_in_category=["D", "T", "H-1", "He-4", "e-", "e+", "n"],
        exception=InvalidIonError,
    ),
]


@pytest.mark.parametrize("case", cases)
class TestParticleInputParameterNames:
    """
    Test the behavior associated with annotated special parameter names
    such as ``element``, ``isotope``, and ``ion``. In particular, make
    sure that the resulting particle(s) belong to the expected
    categories.
    """

    def test_individual_particles_not_in_category(
        self, case: ParameterNamesCase
    ) -> None:
        """
        Test that the appropriate exception is raised when the function
        is provided with individual particles that are not in the
        category.
        """
        for particle in case.particles_not_in_category:
            with pytest.raises(case.exception):
                case.function(particle)

    def test_particle_list_not_in_category(self, case: ParameterNamesCase) -> None:
        """
        Test that the appropriate exception is raised when the function
        is provided with multiple particles at once that are all not in
        the category.
        """
        with pytest.raises(case.exception):
            case.function(case.particles_not_in_category)

    def test_particle_list_some_in_category(self, case: ParameterNamesCase) -> None:
        """
        Test that the appropriate exception is raised when the function
        is provided with multiple particles at once, of which some are
        in the category and some are not.
        """
        with pytest.raises(case.exception):
            case.function(case.particles_some_in_category)

    # If "ionic_level" gets added as a particle category, then add
    # assertions to the following test using is_category.

    def test_individual_particles_all_in_category(
        self, case: ParameterNamesCase
    ) -> None:
        """
        Test that no exception is raised when the function is provided
        with individual particles that are all in the category.
        """
        for particle in case.particles_in_category:
            case.function(particle)

    def test_particle_list_all_in_category(self, case: ParameterNamesCase) -> None:
        """
        Test that no exception is raised when the function is provided
        with multiple particles at once which are all in the category.
        """
        case.function(case.particles_in_category)


def test_custom_particle_for_parameter_named_ion() -> None:
    """
    Test that a positively charged CustomParticle is treated as a valid
    ion when the parameter is named ``ion``.
    """
    custom_ion = CustomParticle(mass=2e-27 * u.kg, charge=3e-19 * u.C)
    result = get_ion(custom_ion)
    assert result == custom_ion


def test_creating_mean_particle_for_parameter_named_ion() -> None:
    Z = 1.3
    ion = get_ion(ion="He", Z=Z)
    assert u.isclose(ion.charge, Z * const.e.si)


@pytest.mark.parametrize("particle", ["p+", ("p+", "D+"), ["He-4", "Al", "Si"]])
@particle_input
def test_particle_list_input(particle: ParticleListLike) -> None:
    assert isinstance(particle, ParticleList)


@particle_input
def return_particle(
    particle: ParticleLike, Z: float | None = None, mass_numb: int | None = None
) -> Particle | CustomParticle | ParticleList:
    """A simple function that is decorated by particle_input."""
    return particle  # type: ignore[return-value]


def test_particle_input_warning_for_integer_z_mean() -> None:
    """
    Test that if a function decorated by `particle_input` is passed
    an integer called `z_mean`, then `z_mean` becomes `Z` and a warning
    is issued.
    """
    with pytest.warns(PlasmaPyDeprecationWarning):
        result = return_particle("H", z_mean=1, mass_numb=1)
    assert result == "p+"


def test_particle_input_warning_for_float_z_mean() -> None:
    """
    Test that if a function decorated by `particle_input` is passed
    a float called `z_mean`, then `z_mean` becomes `Z` and a warning
    is issued.
    """
    z_mean = 0.432

    with pytest.warns(PlasmaPyDeprecationWarning):
        result = return_particle("H", z_mean=z_mean, mass_numb=1)

    Z = result.charge / const.e.si

    assert u.isclose(Z, z_mean)


def test_particle_input_with_var_positional_arguments() -> None:
    """
    Test that |particle_input| works with functions that accept
    variadic positional arguments and keyword arguments.
    """

    @particle_input
    def function_with_var_positional_arguments(
        *args: Any, particle: ParticleLike = None
    ) -> tuple[tuple[Any], Particle]:
        return args, particle  # type: ignore[return-value]

    args = (5, 6, 7)
    particle = "p+"
    expected = (args, Particle(particle))

    actual = function_with_var_positional_arguments(*args, particle=particle)

    assert actual == expected


@pytest.mark.xfail(reason="See issue #2150.")
def test_particle_input_with_pos_and_var_positional_arguments() -> None:
    """
    Test that |particle_input| works with functions that accept
    positional followed by variadic positional arguments.
    """

    @particle_input
    def function_with_pos_and_var_positional_arguments(
        a: Any, *args: Any, particle: ParticleLike = None
    ) -> tuple[tuple[Any, ...], Particle]:
        return args, particle  # type: ignore[return-value]

    a = 1
    args = (5, 6, 7)
    particle = "p+"
    expected = (a, args, Particle(particle))
    actual = function_with_pos_and_var_positional_arguments(a, *args, particle=particle)
    assert actual == expected


@pytest.mark.parametrize(
    ("criteria", "kwargs", "exception"),
    [
        ({"require": "ion"}, {"particle": ["p+", "He"]}, ParticleError),
        ({"require": "isotope"}, {"particle": ["p+", "He"]}, ParticleError),
        (
            {"any_of": {"lepton", "neutrino"}},
            {"particle": ["p+", "alpha"]},
            ParticleError,
        ),
        ({"exclude": "lepton"}, {"particle": ["p+", "e-"]}, ParticleError),
    ],
)
def test_particle_categorization_of_particle_lists(
    criteria: dict[str, str | Iterable[str]],
    kwargs: dict[str, ParticleListLike],
    exception: Exception,
) -> None:
    @particle_input(**criteria)  # type: ignore[arg-type]
    def get_particle(
        particle: ParticleLike,
    ) -> Particle | ParticleList | CustomParticle:
        return particle  # type: ignore[return-value]

    with pytest.raises(exception):  # type: ignore[call-overload]
        get_particle(**kwargs)
