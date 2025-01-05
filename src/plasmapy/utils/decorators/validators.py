"""Decorators to validate arguments and return values."""

__all__ = ["validate_class_attributes", "validate_quantities", "ValidateQuantities"]

import functools
import inspect
import warnings
from collections.abc import Iterable
from typing import Any

import astropy.units as u

from plasmapy.utils.decorators.checks import CheckUnits, CheckValues
from plasmapy.utils.decorators.helpers import preserve_signature


class ValidateQuantities(CheckUnits, CheckValues):
    """
    A decorator class to 'validate' -- control and convert -- the units and values
    of input and return arguments to a function or method.  Arguments are expected to
    be astropy :class:`~astropy.units.quantity.Quantity` objects.

    Parameters
    ----------
    validations_on_return: dictionary of validation specifications
        Specifications for unit and value validations on the return of the
        function being wrapped. (see `quantity validations`_ for valid
        specifications.

    **validations: dictionary of validation specifications
        Specifications for unit and value validations on the input arguments of the
        function being wrapped.  Each keyword argument in ``validations`` is the
        name of a function argument to be validated and the keyword value contains
        the unit and value validation specifications.

        .. _`quantity validations`:

        Unit and value validations can be defined by passing one of the astropy
        :mod:`~astropy.units`, a list of astropy units, or a dictionary containing
        the keys defined below.  Units can also be defined with function annotations,
        but must be consistent with decorator ``**validations`` arguments if used
        concurrently.  If a key is omitted, then the default value will be assumed.

        ====================== ======= ================================================
        Key                    Type    Description
        ====================== ======= ================================================
        units                          list of desired astropy :mod:`~astropy.units`
        equivalencies                  | [DEFAULT `None`] A list of equivalent pairs to
                                         try if
                                       | the units are not directly convertible.
                                       | (see |Astropy Equivalencies|)
        pass_equivalent_units  `bool`  | [DEFAULT `False`] allow equivalent units
                                       | to pass
        can_be_negative        `bool`  [DEFAULT `True`] values can be negative
        can_be_complex         `bool`  [DEFAULT `False`] values can be complex numbers
        can_be_inf             `bool`  [DEFAULT `True`] values can be :data:`~numpy.inf`
        can_be_nan             `bool`  [DEFAULT `True`] values can be :data:`~numpy.nan`
        none_shall_pass        `bool`  [DEFAULT `False`] values can be a python `None`
        can_be_zero            `bool`  [DEFAULT `True`] values can be zero
        ====================== ======= ================================================

    Notes
    -----
    * Validation of function arguments ``*args`` and ``**kwargs`` is not supported.
    * `None` values will pass when `None` is included in the list of specified units,
      is set as a default value for the function argument, or ``none_shall_pass`` is
      set to `True`.  If ``none_shall_pass`` is doubly/triply defined through the
      mentioned options, then they all must be consistent with each other.
    * If units are not specified in ``validations``, then the decorator will attempt
      to identify desired units by examining the function annotations.

    Examples
    --------
    Define unit and value validations with decorator parameters::

        import astropy.units as u
        from plasmapy.utils.decorators import ValidateQuantities


        @ValidateQuantities(
            mass={"units": u.g, "can_be_negative": False},
            vel=u.cm / u.s,
            validations_on_return=[u.g * u.cm / u.s, u.kg * u.m / u.s],
        )
        def foo(mass, vel):
            return mass * vel


        # on a method
        class Foo:
            @ValidateQuantities(
                mass={"units": u.g, "can_be_negative": False},
                vel=u.cm / u.s,
                validations_on_return=[u.g * u.cm / u.s, u.kg * u.m / u.s],
            )
            def bar(self, mass, vel):
                return mass * vel

    Define units with function annotations::

        @ValidateQuantities(mass={"can_be_negative": False})
        def foo(mass: u.g, vel: u.cm / u.s) -> u.g * u.cm / u.s:
            return mass * vel


        # on a method
        class Foo:
            @ValidateQuantities(mass={"can_be_negative": False})
            def bar(self, mass: u.g, vel: u.cm / u.s) -> u.g * u.cm / u.s:
                return mass * vel

    Allow `None` values to pass::

        @ValidateQuantities(validations_on_return=[u.cm, None])
        def foo(arg1: u.cm = None):
            return arg1

    Allow return values to have equivalent units::

        @ValidateQuantities(
            arg1={"units": u.cm},
            validations_on_return={"units": u.km, "pass_equivalent_units": True},
        )
        def foo(arg1):
            return arg1

    Allow equivalent units to pass with specified equivalencies::

        @ValidateQuantities(
            arg1={
                "units": u.K,
                "equivalencies": u.temperature(),
                "pass_equivalent_units": True,
            }
        )
        def foo(arg1):
            return arg1

    .. _astropy equivalencies:
        https://docs.astropy.org/en/stable/units/equivalencies.html
    """

    def __init__(
        self, validations_on_return=None, **validations: dict[str, Any]
    ) -> None:
        if "checks_on_return" in validations:
            raise TypeError(
                "keyword argument 'checks_on_return' is not allowed, "
                "use 'validations_on_return' to set validations "
                "on the return variable"
            )

        self._validations = validations

        checks = validations.copy()
        if validations_on_return is not None:
            self._validations["validations_on_return"] = validations_on_return
            checks["checks_on_return"] = validations_on_return

        super().__init__(**checks)

    def __call__(self, f):
        """
        Decorate a function.

        Parameters
        ----------
        f
            Function to be wrapped

        Returns
        -------
        function
            wrapped function of ``f``
        """
        self.f = f
        wrapped_sign = inspect.signature(f, eval_str=True)

        @preserve_signature
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # combine args and kwargs into dictionary
            bound_args = wrapped_sign.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # get conditioned validations
            validations = self._get_validations(bound_args)

            # validate (input) argument units and values
            for arg_name in validations:
                # skip check of output/return
                if arg_name == "validations_on_return":
                    continue

                # validate argument & update for conversion
                arg = self._validate_quantity(
                    bound_args.arguments[arg_name], arg_name, validations[arg_name]
                )
                bound_args.arguments[arg_name] = arg

            # call function
            _return = f(**bound_args.arguments)

            # validate output
            if "validations_on_return" in validations:
                _return = self._validate_quantity(
                    _return,
                    "validations_on_return",
                    validations["validations_on_return"],
                )

            return _return

        return wrapper

    def _get_validations(
        self, bound_args: inspect.BoundArguments
    ) -> dict[str, dict[str, Any]]:
        """
        Review :attr:`validations` and function bound arguments to build a complete
        'validations' dictionary.  If a validation key is omitted from the argument
        validations, then a default value is assumed (see `quantity validations`_).

        Parameters
        ----------
        bound_args: :class:`inspect.BoundArguments`
            arguments passed into the function being wrapped

            .. code-block:: python

                bound_args = inspect.signature(f).bind(*args, **kwargs)

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A complete 'validations' dictionary for validating function input arguments
            and return.
        """
        unit_checks = self._get_unit_checks(bound_args)
        value_checks = self._get_value_checks(bound_args)

        # combine all validations
        # * `unit_checks` will encompass all argument "checks" defined either by
        #   function annotations or **validations.
        # * `value_checks` may miss some arguments if **validations only defines
        #   unit validations or some validations come from function annotations
        validations = unit_checks.copy()
        for arg_name in validations:
            # augment 'none_shall_pass' (if needed)
            try:
                # if 'none_shall_pass' was in the original passed-in validations,
                # then override the value determined by CheckUnits
                _none_shall_pass = self.validations[arg_name]["none_shall_pass"]
                # if validations[arg_name]['none_shall_pass'] != _none_shall_pass:
                if (
                    _none_shall_pass is False
                    and validations[arg_name]["none_shall_pass"] is True
                ):
                    raise ValueError(
                        f"Validation 'none_shall_pass' for argument '{arg_name}' is "
                        f"inconsistent between function annotations "
                        f"({validations[arg_name]['none_shall_pass']}) and decorator "
                        f"argument ({_none_shall_pass})."
                    )
                validations[arg_name]["none_shall_pass"] = _none_shall_pass
            except (KeyError, TypeError):
                # 'none_shall_pass' was not in the original passed-in validations, so
                # rely on the value determined by CheckUnits
                pass
            finally:
                try:
                    del value_checks[arg_name]["none_shall_pass"]
                except KeyError:
                    dvc = self._CheckValues__check_defaults.copy()
                    del dvc["none_shall_pass"]
                    value_checks[arg_name] = dvc

            # update the validations dictionary
            validations[arg_name].update(value_checks[arg_name])

        if "checks_on_return" in validations:
            validations["validations_on_return"] = validations.pop("checks_on_return")

        return validations

    def _validate_quantity(  # noqa: C901
        self,
        arg,
        arg_name: str,
        arg_validations: dict[str, Any],
    ):
        """
        Perform validations `arg_validations` on function argument `arg`
        named `arg_name`.

        Parameters
        ----------
        arg
            The argument to be validated.

        arg_name: str
            The name of the argument to be validated

        arg_validations: Dict[str, Any]
            The requested validations for the argument

        Raises
        ------
        TypeError
            if argument is not an Astropy :class:`~astropy.units.Quantity`
            or not convertible to a :class:`~astropy.units.Quantity`

        ValueError
            if validations fail
        """
        # rename to work with "check" methods
        if arg_name == "validations_on_return":
            arg_name = "checks_on_return"

        # initialize str for error message
        if arg_name == "checks_on_return":
            err_msg = "The return value  "
        else:
            err_msg = f"The argument '{arg_name}' "
        err_msg += f"to function {self.f.__name__}()"

        # initialize TypeError message
        typeerror_msg = (
            f"{err_msg} should be an astropy Quantity with units"
            f" equivalent to one of ["
        )
        for ii, unit in enumerate(arg_validations["units"]):
            typeerror_msg += f"{unit}"

            if ii != len(arg_validations["units"]) - 1:
                typeerror_msg += ", "
        typeerror_msg += "]"

        # add units to arg if possible
        # * a None value will be taken care of by `_check_unit_core`
        #
        if arg is None or hasattr(arg, "unit"):
            pass
        elif len(arg_validations["units"]) != 1:
            raise TypeError(typeerror_msg)
        else:
            try:
                arg = arg * arg_validations["units"][0]
            except (TypeError, ValueError) as ex:
                raise TypeError(typeerror_msg) from ex
            else:
                warnings.warn(
                    u.UnitsWarning(
                        f"{err_msg} has no specified units. Assuming units of "
                        f"{arg_validations['units'][0]}. To silence this warning, "
                        f"explicitly pass in an astropy Quantity "
                        f"(e.g. 5. * astropy.units.cm) "
                        f"(see http://docs.astropy.org/en/stable/units/)"
                    )
                )

        # check units
        arg, unit, equiv, err = self._check_unit_core(arg, arg_name, arg_validations)

        # convert quantity
        if (
            arg is not None
            and unit is not None
            and not arg_validations["pass_equivalent_units"]
        ):
            arg = arg.to(unit, equivalencies=equiv)
        elif err is not None:
            raise err

        self._check_value(arg, arg_name, arg_validations)

        return arg

    @property
    def validations(self):
        """
        Requested validations on the decorated function's input arguments and
        return variable.
        """
        return self._validations


def validate_quantities(func=None, validations_on_return=None, **validations):
    """
    A decorator to 'validate' — control and convert — the units and values
    of input and return arguments to a function or method.  Arguments are expected to
    be astropy :class:`~astropy.units.quantity.Quantity` objects.

    Parameters
    ----------
    func:
        The function to be decorated

    validations_on_return: dictionary of validation specifications
        Specifications for unit and value validations on the return of the
        function being wrapped. (see `quantity validations`_ for valid
        specifications.

    **validations: dictionary of validation specifications
        Specifications for unit and value validations on the input arguments of the
        function being wrapped.  Each keyword argument in ``validations`` is the
        name of a function argument to be validated and the keyword value contains
        the unit and value validation specifications.

        .. _`quantity validations`:

        Unit and value validations can be defined by passing one of the astropy
        :mod:`~astropy.units`, a list of astropy units, or a dictionary containing
        the keys defined below.  Units can also be defined with function annotations,
        but must be consistent with decorator ``**validations`` arguments if used
        concurrently.  If a key is omitted, then the default value will be assumed.

        ====================== ======= ================================================
        Key                    Type    Description
        ====================== ======= ================================================
        units                          list of desired astropy :mod:`~astropy.units`
        equivalencies                  | [DEFAULT `None`] A list of equivalent pairs to
                                         try if
                                       | the units are not directly convertible.
                                       | (see |Astropy Equivalencies|)
        pass_equivalent_units  `bool`  | [DEFAULT `False`] allow equivalent units
                                       | to pass
        can_be_negative        `bool`  [DEFAULT `True`] values can be negative
        can_be_complex         `bool`  [DEFAULT `False`] values can be complex numbers
        can_be_inf             `bool`  [DEFAULT `True`] values can be :data:`~numpy.inf`
        can_be_nan             `bool`  [DEFAULT `True`] values can be :data:`~numpy.nan`
        none_shall_pass        `bool`  [DEFAULT `False`] values can be a python `None`
        can_be_zero            `bool`  [DEFAULT `True`] values can be zero
        ====================== ======= ================================================

    Notes
    -----
    * Validation of function arguments ``*args`` and ``**kwargs`` is not supported.
    * `None` values will pass when `None` is included in the list of specified units,
      is set as a default value for the function argument, or ``none_shall_pass`` is
      set to `True`.  If ``none_shall_pass`` is doubly/triply defined through the
      mentioned options, then they all must be consistent with each other.
    * If units are not specified in ``validations``, then the decorator will attempt
      to identify desired units by examining the function annotations.
    * Full functionality is defined by the class :class:`ValidateQuantities`.

    Examples
    --------
    Define unit and value validations with decorator parameters::

        import astropy.units as u
        from plasmapy.utils.decorators import validate_quantities


        @validate_quantities(
            mass={"units": u.g, "can_be_negative": False},
            vel=u.cm / u.s,
            validations_on_return=[u.g * u.cm / u.s, u.kg * u.m / u.s],
        )
        def foo(mass, vel):
            return mass * vel


        # on a method
        class Foo:
            @validate_quantities(
                mass={"units": u.g, "can_be_negative": False},
                vel=u.cm / u.s,
                validations_on_return=[u.g * u.cm / u.s, u.kg * u.m / u.s],
            )
            def bar(self, mass, vel):
                return mass * vel

    Define units with function annotations::

        @validate_quantities(mass={"can_be_negative": False})
        def foo(mass: u.g, vel: u.cm / u.s) -> u.g * u.cm / u.s:
            return mass * vel


        # rely only on annotations
        @validate_quantities
        def foo(x: u.cm, time: u.s) -> u.cm / u.s:
            return x / time


        # on a method
        class Foo:
            @validate_quantities(mass={"can_be_negative": False})
            def bar(self, mass: u.g, vel: u.cm / u.s) -> u.g * u.cm / u.s:
                return mass * vel

    Define units using type hint annotations::

        @validate_quantities
        def foo(x: u.Quantity[u.m], time: u.Quantity[u.s]) -> u.Quantity[u.m / u.s]:
            return x / time

    Allow `None` values to pass::

        @validate_quantities(
            arg2={"none_shall_pass": True}, validations_on_return=[u.cm, None]
        )
        def foo(arg1: u.cm, arg2: u.cm = None):
            return None

    Allow return values to have equivalent units::

        @validate_quantities(
            arg1={"units": u.cm},
            validations_on_return={"units": u.km, "pass_equivalent_units": True},
        )
        def foo(arg1):
            return arg1

    Allow equivalent units to pass with specified equivalencies::

        @validate_quantities(
            arg1={
                "units": u.K,
                "equivalencies": u.temperature(),
                "pass_equivalent_units": True,
            }
        )
        def foo(arg1):
            return arg1

    .. _astropy equivalencies:
        https://docs.astropy.org/en/stable/units/equivalencies.html
    """

    if validations_on_return is not None:
        validations["validations_on_return"] = validations_on_return

    if func is not None:
        # `validate_quantities` called as a function
        return ValidateQuantities(**validations)(func)

    # `validate_quantities` called as a decorator "sugar-syntax"
    return ValidateQuantities(**validations)


def get_attributes_not_provided(
    self,
    expected_attributes: list[str] | None = None,
    both_or_either_attributes: list[Iterable[str]] | None = None,
    mutually_exclusive_attributes: list[Iterable[str]] | None = None,
):
    """
    Collect attributes that weren't provided during instantiation needed
    to access a method.
    """

    attributes_not_provided = []

    if expected_attributes is not None:
        attributes_not_provided.extend(
            attribute
            for attribute in expected_attributes
            if getattr(self, attribute) is None
        )
    if both_or_either_attributes is not None:
        for attribute_tuple in both_or_either_attributes:
            number_of_attributes_provided = sum(
                getattr(self, attribute) is not None for attribute in attribute_tuple
            )
            if number_of_attributes_provided == 0:
                attributes_not_provided.append(
                    f"at least one of {' or '.join(attribute_tuple)}"
                )

    if mutually_exclusive_attributes is not None:
        for attribute_tuple in mutually_exclusive_attributes:
            number_of_attributes_provided = sum(
                getattr(self, attribute) is not None for attribute in attribute_tuple
            )
            if number_of_attributes_provided != 1:
                attributes_not_provided.append(
                    f"exactly one of {' or '.join(attribute_tuple)}"
                )

    return attributes_not_provided


def validate_class_attributes(
    expected_attributes: list[str] | None = None,
    both_or_either_attributes: list[Iterable[str]] | None = None,
    mutually_exclusive_attributes: list[Iterable[str]] | None = None,
):
    """
    A decorator responsible for raising errors if the expected arguments weren't
    provided during class instantiation.
    """

    def decorator(attribute):
        def wrapper(self, *args, **kwargs):
            arguments_not_provided = get_attributes_not_provided(
                self,
                expected_attributes,
                both_or_either_attributes,
                mutually_exclusive_attributes,
            )

            if len(arguments_not_provided) > 0:
                raise ValueError(
                    f"{attribute.__name__} expected the following "
                    f"additional arguments: {', '.join(arguments_not_provided)}"
                )

            return attribute(self, *args, **kwargs)

        return wrapper

    return decorator
