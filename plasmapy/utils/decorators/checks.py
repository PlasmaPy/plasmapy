"""
Decorator for checking input/output arguments of functions.
"""
__all__ = [
    "check_values",
    "check_units",
    "check_relativistic",
    "CheckBase",
    "CheckUnits",
    "CheckValues",
]

import collections
import functools
import inspect
import numpy as np
import warnings

from astropy import units as u
from astropy.constants import c
from astropy.units.equivalencies import Equivalency
from functools import reduce
from operator import add
from typing import Any, Optional, Union

from plasmapy.utils.decorators.helpers import preserve_signature
from plasmapy.utils.exceptions import (
    PlasmaPyWarning,
    RelativityError,
    RelativityWarning,
)


class CheckBase:
    """
    Base class for 'Check' decorator classes.

    Parameters
    ----------
    checks_on_return
        specified checks on the return of the wrapped function

    **checks
        specified checks on the input arguments of the wrapped function
    """

    def __init__(self, checks_on_return=None, **checks):
        self._checks = checks
        if checks_on_return is not None:
            self._checks["checks_on_return"] = checks_on_return

    @property
    def checks(self):
        """
        Requested checks on the decorated function's input arguments
        and/or return.
        """
        return self._checks


class CheckValues(CheckBase):
    """
    A decorator class to 'check' — limit/control — the values of input and return
    arguments to a function or method.

    Parameters
    ----------
    checks_on_return: Dict[str, bool]
        Specifications for value checks on the return of the function being wrapped.
        (see `check values`_ for valid specifications)

    **checks: Dict[str, Dict[str, bool]]
        Specifications for value checks on the input arguments of the function
        being wrapped.  Each keyword argument in ``checks`` is the name of a function
        argument to be checked and the keyword value contains the value check
        specifications.

        .. _`check values`:

        The value check specifications are defined within a dictionary containing
        the keys defined below.  If the dictionary is empty or omitting keys,
        then the default value will be assumed for the missing keys.

        ================ ======= ================================================
        Key              Type    Description
        ================ ======= ================================================
        can_be_negative  `bool`  [DEFAULT `True`] values can be negative
        can_be_complex   `bool`  [DEFAULT `False`] values can be complex numbers
        can_be_inf       `bool`  [DEFAULT `True`] values can be :data:`~numpy.inf`
        can_be_nan       `bool`  [DEFAULT `True`] values can be :data:`~numpy.nan`
        none_shall_pass  `bool`  [DEFAULT `False`] values can be a python `None`
        can_be_zero      `bool`  [DEFAULT `True`] values can be zero
        ================ ======= ================================================

    Notes
    -----
    * Checking of function arguments ``*args`` and ``**kwargs`` is not supported.

    Examples
    --------
    .. code-block:: python

        from plasmapy.utils.decorators.checks import CheckValues


        @CheckValues(
            arg1={"can_be_negative": False, "can_be_nan": False},
            arg2={"can_be_inf": False},
            checks_on_return={"none_shall_pass": True},
        )
        def foo(arg1, arg2):
            return None


        # on a method
        class Foo:
            @CheckValues(
                arg1={"can_be_negative": False, "can_be_nan": False},
                arg2={"can_be_inf": False},
                checks_on_return={"none_shall_pass": True},
            )
            def bar(self, arg1, arg2):
                return None
    """

    #: Default values for the possible 'check' keys.
    # To add a new check to the class, the following needs to be done:
    #   1. Add a key & default value to the `__check_defaults` dictionary
    #   2. Add a corresponding if-statement to method `_check_value`
    #
    __check_defaults = {
        "can_be_negative": True,
        "can_be_complex": False,
        "can_be_inf": True,
        "can_be_nan": True,
        "none_shall_pass": False,
        "can_be_zero": True,
    }

    def __init__(
        self, checks_on_return: dict[str, bool] = None, **checks: dict[str, bool]
    ):

        super().__init__(checks_on_return=checks_on_return, **checks)

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
        wrapped_sign = inspect.signature(f)

        @preserve_signature
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # map args and kwargs to function parameters
            bound_args = wrapped_sign.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # get checks
            checks = self._get_value_checks(bound_args)

            # check input arguments
            for arg_name in checks:
                # skip check of output/return
                if arg_name == "checks_on_return":
                    continue

                # check argument
                self._check_value(
                    bound_args.arguments[arg_name], arg_name, checks[arg_name]
                )

            # call function
            _return = f(**bound_args.arguments)

            # check function return
            if "checks_on_return" in checks:
                self._check_value(
                    _return, "checks_on_return", checks["checks_on_return"]
                )

            return _return

        return wrapper

    def _get_value_checks(
        self, bound_args: inspect.BoundArguments
    ) -> dict[str, dict[str, bool]]:
        """
        Review :attr:`checks` and function bound arguments to build a complete 'checks'
        dictionary.  If a check key is omitted from the argument checks, then a default
        value is assumed (see `check values`_).

        Parameters
        ----------
        bound_args: :class:`inspect.BoundArguments`
            arguments passed into the function being wrapped

            .. code-block:: python

                bound_args = inspect.signature(f).bind(*args, **kwargs)

        Returns
        -------
        Dict[str, Dict[str, bool]]
            A complete 'checks' dictionary for checking function input arguments
            and return.
        """
        # initialize validation dictionary
        out_checks = {}

        # Iterate through function bound arguments + return and build `out_checks:
        #
        # artificially add "return" to parameters
        things_to_check = bound_args.signature.parameters.copy()
        things_to_check["checks_on_return"] = inspect.Parameter(
            "checks_on_return",
            inspect.Parameter.POSITIONAL_ONLY,
            annotation=bound_args.signature.return_annotation,
        )
        for param in things_to_check.values():
            # variable arguments are NOT checked
            # e.g. in foo(x, y, *args, d=None, **kwargs) variable arguments
            #      *args and **kwargs will NOT be checked
            #
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            # grab the checks dictionary for the desired parameter
            try:
                param_in_checks = self.checks[param.name]
            except KeyError:
                # checks for parameter not specified
                continue

            # build `out_checks`
            # read checks and/or apply defaults values
            out_checks[param.name] = {}
            for v_name, v_default in self.__check_defaults.items():
                try:
                    out_checks[param.name][v_name] = param_in_checks.get(
                        v_name, v_default
                    )
                except AttributeError:
                    # for the case that checks are defined for an argument,
                    # but is NOT a dictionary
                    # (e.g. CheckValues(x=u.cm) ... this scenario could happen
                    # during subclassing)
                    out_checks[param.name][v_name] = v_default

        # Does `self.checks` indicate arguments not used by f?
        if missing_params := list(set(self.checks) - set(out_checks)):
            params_str = ", ".join(missing_params)
            warnings.warn(
                PlasmaPyWarning(
                    f"Expected to value check parameters {params_str} but they "
                    f"are missing from the call to {self.f.__name__}"
                )
            )

        return out_checks

    def _check_value(self, arg, arg_name: str, arg_checks: dict[str, bool]):
        """
        Perform checks ``arg_checks`` on function argument ``arg``.

        Parameters
        ----------
        arg
            The argument to be checked

        arg_name: str
            The name of the argument to be checked

        arg_checks: Dict[str, bool]
            The requested checks for the argument

        Raises
        ------
        ValueError
            raised if a check fails

        """
        if arg_name == "checks_on_return":
            valueerror_msg = "The return value "
        else:
            valueerror_msg = f"The argument '{arg_name}' "
        valueerror_msg += f"to function {self.f.__name__}() can not contain"

        # check values
        # * 'none_shall_pass' always needs to be checked first
        ckeys = list(self.__check_defaults.keys())
        ckeys.remove("none_shall_pass")
        ckeys = ("none_shall_pass",) + tuple(ckeys)
        for ckey in ckeys:
            if ckey == "can_be_complex":
                if not arg_checks[ckey] and np.any(np.iscomplexobj(arg)):
                    raise ValueError(f"{valueerror_msg} complex numbers.")

            elif ckey == "can_be_inf":
                if not arg_checks[ckey] and np.any(np.isinf(arg)):
                    raise ValueError(f"{valueerror_msg} infs.")

            elif ckey == "can_be_nan":
                if not arg_checks["can_be_nan"] and np.any(np.isnan(arg)):
                    raise ValueError(f"{valueerror_msg} NaNs.")

            elif ckey == "can_be_negative":
                if not arg_checks[ckey] and np.any(arg < 0):
                    raise ValueError(f"{valueerror_msg} negative numbers.")

            elif ckey == "can_be_zero":
                if not arg_checks[ckey] and np.any(arg == 0):
                    raise ValueError(f"{valueerror_msg} zeros.")

            elif ckey == "none_shall_pass":
                if arg is None and arg_checks[ckey]:
                    break
                elif arg is None:
                    raise ValueError(f"{valueerror_msg} Nones.")


class CheckUnits(CheckBase):
    """
    A decorator class to 'check' — limit/control — the units of input and return
    arguments to a function or method.

    Parameters
    ----------
    checks_on_return: list of :mod:`~astropy.units` or dict of unit specifications
        Specifications for unit checks on the return of the function being wrapped.
        (see `check units`_ for valid specifications)

    **checks: list of astropy :mod:`~astropy.units` or dict of unit specifications
        Specifications for unit checks on the input arguments of the function
        being wrapped.  Each keyword argument in ``checks`` is the name of a function
        argument to be checked and the keyword value contains the unit check
        specifications.

        .. _`check units`:

        Unit checks can be defined by passing one of the astropy
        :mod:`~astropy.units`, a list of astropy units, or a dictionary containing
        the keys defined below.  Units can also be defined with function
        annotations, but must be consistent with decorator ``**checks`` arguments if
        used concurrently. If a key is omitted, then the default value will be assumed.

        ====================== ======= ================================================
        Key                    Type    Description
        ====================== ======= ================================================
        units                          list of desired astropy :mod:`~astropy.units`
        equivalencies                  | [DEFAULT `None`] A list of equivalent pairs to
                                         try if
                                       | the units are not directly convertible.
                                       | (see :mod:`~astropy.units.equivalencies`,
                                         and/or `astropy equivalencies`_)
        pass_equivalent_units  `bool`  | [DEFAULT `False`] allow equivalent units
                                       | to pass
        ====================== ======= ================================================

    Notes
    -----
    * Checking of function arguments ``*args`` and ``**kwargs`` is not supported.
    * Decorator does NOT perform any unit conversions.
    * If it is desired that `None` values do not raise errors or warnings, then
      include `None` in the list of units or as a default value for the function
      argument.
    * If units are not specified in ``checks``, then the decorator will attempt
      to identify desired units by examining the function annotations.

    Examples
    --------
    Define units with decorator parameters::

        import astropy.units as u
        from plasmapy.utils.decorators import CheckUnits

        @CheckUnits(arg1={'units': u.cm},
                    arg2=u.cm,
                    checks_on_return=[u.cm, u.km])
        def foo(arg1, arg2):
            return arg1 + arg2

        # or on a method
        class Foo:
            @CheckUnits(arg1={'units': u.cm},
                        arg2=u.cm,
                        checks_on_return=[u.cm, u.km])
            def bar(self, arg1, arg2):
                return arg1 + arg2

    Define units with function annotations::

        import astropy.units as u
        from plasmapy.utils.decorators import CheckUnits

        @CheckUnits()
        def foo(arg1: u.cm, arg2: u.cm) -> u.cm:
            return arg1 + arg2

        # or on a method
        class Foo:
            @CheckUnits()
            def bar(self, arg1: u.cm, arg2: u.cm) -> u.cm:
                return arg1 + arg2

    Allow `None` values to pass, on input and output::

        import astropy.units as u
        from plasmapy.utils.decorators import CheckUnits

        @CheckUnits(checks_on_return=[u.cm, None])
        def foo(arg1: u.cm = None):
            return arg1

    Allow return values to have equivalent units::

        import astropy.units as u
        from plasmapy.utils.decorators import CheckUnits

        @CheckUnits(arg1={'units': u.cm},
                    checks_on_return={'units': u.km,
                                      'pass_equivalent_units': True})
        def foo(arg1):
            return arg1

    Allow equivalent units to pass with specified equivalencies::

        import astropy.units as u
        from plasmapy.utils.decorators import CheckUnits

        @CheckUnits(arg1={'units': u.K,
                          'equivalencies': u.temperature_energy(),
                          'pass_equivalent_units': True})
        def foo(arg1):
            return arg1

    .. _astropy equivalencies:
        https://docs.astropy.org/en/stable/units/equivalencies.html
    """

    #: Default values for the possible 'check' keys.
    # To add a new check the the class, the following needs to be done:
    #   1. Add a key & default value to the `__check_defaults` dictionary
    #   2. Add a corresponding conditioning statement to `_get_unit_checks`
    #   3. Add a corresponding behavior to `_check_unit`
    #
    __check_defaults = {
        "units": None,
        "equivalencies": None,
        "pass_equivalent_units": False,
        "none_shall_pass": False,
    }

    def __init__(
        self,
        checks_on_return: Union[u.Unit, list[u.Unit], dict[str, Any]] = None,
        **checks: Union[u.Unit, list[u.Unit], dict[str, Any]],
    ):

        super().__init__(checks_on_return=checks_on_return, **checks)

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
        wrapped_sign = inspect.signature(f)

        @preserve_signature
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # combine args and kwargs into dictionary
            bound_args = wrapped_sign.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # get checks
            checks = self._get_unit_checks(bound_args)

            # check (input) argument units
            for arg_name in checks:
                # skip check of output/return
                if arg_name == "checks_on_return":
                    continue

                # check argument
                self._check_unit(
                    bound_args.arguments[arg_name], arg_name, checks[arg_name]
                )

            # call function
            _return = f(**bound_args.arguments)

            # check output
            if "checks_on_return" in checks:
                self._check_unit(
                    _return, "checks_on_return", checks["checks_on_return"]
                )

            return _return

        return wrapper

    def _get_unit_checks(
        self, bound_args: inspect.BoundArguments
    ) -> dict[str, dict[str, Any]]:
        """
        Review :attr:`checks` and function bound arguments to build a complete 'checks'
        dictionary.  If a check key is omitted from the argument checks, then a default
        value is assumed (see `check units`_)

        Parameters
        ----------
        bound_args: :class:`inspect.BoundArguments`
            arguments passed into the function being wrapped

            .. code-block:: python

                bound_args = inspect.signature(f).bind(*args, **kwargs)

        Returns
        -------
        Dict[str, Dict[str, Any]]
            A complete 'checks' dictionary for checking function input arguments
            and return.
        """
        # initialize validation dictionary
        out_checks = {}

        # Iterate through function bound arguments + return and build `out_checks`:
        #
        # artificially add "return" to parameters
        things_to_check = bound_args.signature.parameters.copy()
        things_to_check["checks_on_return"] = inspect.Parameter(
            "checks_on_return",
            inspect.Parameter.POSITIONAL_ONLY,
            annotation=bound_args.signature.return_annotation,
        )
        for param in things_to_check.values():
            # variable arguments are NOT checked
            # e.g. in foo(x, y, *args, d=None, **kwargs) variable arguments
            #      *args and **kwargs will NOT be checked
            #
            if param.kind in (
                inspect.Parameter.VAR_KEYWORD,
                inspect.Parameter.VAR_POSITIONAL,
            ):
                continue

            # grab the checks dictionary for the desired parameter
            try:
                param_checks = self.checks[param.name]
            except KeyError:
                param_checks = None

            # -- Determine target units `_units` --
            # target units can be defined in one of three ways (in
            # preferential order):
            #   1. direct keyword pass-through
            #      i.e. CheckUnits(x=u.cm)
            #           CheckUnits(x=[u.cm, u.s])
            #   2. keyword pass-through via dictionary definition
            #      i.e. CheckUnits(x={'units': u.cm})
            #           CheckUnits(x={'units': [u.cm, u.s]})
            #   3. function annotations
            #
            # * if option (3) is used simultaneously with option (1) or (2), then
            #   checks defined by (3) must be consistent with checks from (1) or (2)
            #   to avoid raising an error.
            # * if None is included in the units list, then None values are allowed
            #
            _none_shall_pass = False
            _units = None
            _units_are_from_anno = False
            if param_checks is not None:
                # checks for argument were defined with decorator
                try:
                    _units = param_checks["units"]
                except TypeError:
                    # if checks is NOT None and is NOT a dictionary, then assume
                    # only units were specified
                    #   e.g. CheckUnits(x=u.cm)
                    #
                    _units = param_checks
                except KeyError:
                    # if checks does NOT have 'units' but is still a dictionary,
                    # then other check conditions may have been specified and the
                    # user is relying on function annotations to define desired
                    # units
                    _units = None

            # If no units have been specified by decorator checks, then look for
            # function annotations.
            #
            # Reconcile units specified by decorator checks and function annotations
            _units_anno = None
            if param.annotation is not inspect.Parameter.empty:
                # unit annotations defined
                _units_anno = param.annotation

            if _units is _units_anno is param_checks is None:
                # no checks specified and no unit annotations defined
                continue
            elif _units is _units_anno is None:
                # checks specified, but NO unit checks
                msg = "No astropy.units specified for "
                if param.name == "checks_on_return":
                    msg += "return value "
                else:
                    msg += f"argument {param.name} "
                msg += f"of function {self.f.__name__}()."
                raise ValueError(msg)
            elif _units is None and hasattr(_units_anno, "__metadata__"):
                _units = _units_anno.__metadata__[
                    0
                ]  # TODO will this blow up on us somewhere?
                _units_are_from_anno = True
                _units_anno = None
            elif _units is None:
                _units = _units_anno
                _units_are_from_anno = True
                _units_anno = None

            # Ensure `_units` is an iterable
            if not isinstance(_units, collections.abc.Iterable):
                _units = [_units]
            if not isinstance(_units_anno, collections.abc.Iterable):
                _units_anno = [_units_anno]

            # Is None allowed?
            if None in _units or param.default is None:
                _none_shall_pass = True

            # Remove Nones
            if None in _units:
                _units = [t for t in _units if t is not None]
            if None in _units_anno:
                _units_anno = [t for t in _units_anno if t is not None]

            # ensure all _units are astropy.units.Unit or physical types &
            # define 'units' for unit checks &
            # define 'none_shall_pass' check
            _units = self._condition_target_units(
                _units, from_annotations=_units_are_from_anno
            )
            _units_anno = self._condition_target_units(
                _units_anno, from_annotations=True
            )
            if not all(_u in _units for _u in _units_anno):
                raise ValueError(
                    f"For argument '{param.name}', "
                    f"annotation units ({_units_anno}) are not included in the units "
                    f"specified by decorator arguments ({_units}).  Use either "
                    f"decorator arguments or function annotations to defined unit "
                    f"types, or make sure annotation specifications match decorator "
                    f"argument specifications."
                )
            if not _units and not _units_anno and param_checks is None:
                # annotations did not specify units
                continue
            elif not _units and not _units_anno:
                # checks specified, but NO unit checks
                msg = "No astropy.units specified for "
                if param.name == "checks_on_return":
                    msg += "return value "
                else:
                    msg += f"argument {param.name} "
                msg += f"of function {self.f.__name__}()."
                raise ValueError(msg)

            out_checks[param.name] = {
                "units": _units,
                "none_shall_pass": _none_shall_pass,
            }

            # -- Determine target equivalencies --
            # Unit equivalences can be defined by:
            # 1. keyword pass-through via dictionary definition
            #    e.g. CheckUnits(x={'units': u.C,
            #                       'equivalencies': u.temperature})
            #
            # initialize equivalencies
            try:
                _equivs = param_checks["equivalencies"]
            except (KeyError, TypeError):
                _equivs = self.__check_defaults["equivalencies"]

            # ensure equivalences are properly formatted
            if _equivs is None or _equivs == [None]:
                _equivs = None
            elif isinstance(_equivs, Equivalency):
                pass
            elif isinstance(_equivs, (list, tuple)):

                # flatten list to non-list elements
                if isinstance(_equivs, tuple):
                    _equivs = [_equivs]
                else:
                    _equivs = self._flatten_equivalencies_list(_equivs)

                # ensure passed equivalencies list is structured properly
                #   [(), ...]
                #   or [Equivalency(), ...]
                #
                # * All equivalencies must be a list of 2, 3, or 4 element tuples
                #   structured like...
                #     (from_unit, to_unit, forward_func, backward_func)
                #
                if all(isinstance(el, Equivalency) for el in _equivs):
                    _equivs = reduce(add, _equivs)
                else:
                    _equivs = self._normalize_equivalencies(_equivs)

            out_checks[param.name]["equivalencies"] = _equivs

            # -- Determine if equivalent units pass --
            try:
                peu = param_checks.get(
                    "pass_equivalent_units",
                    self.__check_defaults["pass_equivalent_units"],
                )
            except (AttributeError, TypeError):
                peu = self.__check_defaults["pass_equivalent_units"]

            out_checks[param.name]["pass_equivalent_units"] = peu

        # Does `self.checks` indicate arguments not used by f?
        missing_params = [
            param for param in set(self.checks.keys()) - set(out_checks.keys())
        ]
        if missing_params:
            params_str = ", ".join(missing_params)
            warnings.warn(
                PlasmaPyWarning(
                    f"Expected to unit check parameters {params_str} but they "
                    f"are missing from the call to {self.f.__name__}"
                )
            )

        return out_checks

    def _check_unit(self, arg, arg_name: str, arg_checks: dict[str, Any]):
        """
        Perform unit checks ``arg_checks`` on function argument ``arg``.

        Parameters
        ----------
        arg
            The argument to be checked

        arg_name: str
            The name of the argument to be checked

        arg_checks: Dict[str, Any]
            The requested checks for the argument

        Raises
        ------
        ValueError
            If ``arg`` is `None` when `arg_checks['none_shall_pass']=False`

        TypeError
            If ``arg`` does not have units

        :class:`astropy.units.UnitTypeError`
            If the units of ``arg`` do not satisfy conditions of ``arg_checks``
        """
        arg, unit, equiv, err = self._check_unit_core(arg, arg_name, arg_checks)
        if err is not None:
            raise err

    def _check_unit_core(
        self, arg, arg_name: str, arg_checks: dict[str, Any]
    ) -> tuple[
        Optional[u.Quantity],
        Optional[u.Unit],
        Optional[list[Any]],
        Optional[Exception],
    ]:
        """
        Determines if `arg` passes unit checks `arg_checks` and if the units of
        `arg` is equivalent to any units specified in `arg_checks`.

        Parameters
        ----------
        arg
            The argument to be checked

        arg_name: str
            The name of the argument to be checked

        arg_checks: Dict[str, Any]
            The requested checks for the argument

        Returns
        -------
        (`arg`, `unit`, `equivalencies`, `error`)
            * `arg` is the original input argument `arg` or `None` if unit
              checks fail
            * `unit` is the identified astropy :mod:`~astropy.units` that `arg`
              can be converted to or `None` if none exist
            * `equivalencies` is the astropy :mod:`~astropy.units.equivalencies`
              used for the unit conversion or `None`
            * `error` is the `Exception` associated with the failed unit checks
              or `None` for successful unit checks
        """
        # initialize str for error messages
        if arg_name == "checks_on_return":
            err_msg = "The return value "
        else:
            err_msg = f"The argument '{arg_name}' "
        err_msg += f"to function {self.f.__name__}()"

        # initialize ValueError message
        valueerror_msg = f"{err_msg} can not contain"

        # initialize TypeError message
        typeerror_msg = f"{err_msg} should be an astropy Quantity with "
        if len(arg_checks["units"]) == 1:
            typeerror_msg += f"the following unit: {arg_checks['units'][0]}"
        else:
            typeerror_msg += "one of the following units: "
            for unit in arg_checks["units"]:
                typeerror_msg += str(unit)
                if unit != arg_checks["units"][-1]:
                    typeerror_msg += ", "
        if arg_checks["none_shall_pass"]:
            typeerror_msg += "or None "

        # pass Nones if allowed
        if arg is None:
            if arg_checks["none_shall_pass"]:
                return arg, None, None, None
            else:
                return None, None, None, ValueError(f"{valueerror_msg} Nones")

        # check units
        in_acceptable_units = []
        equiv = arg_checks["equivalencies"]
        for unit in arg_checks["units"]:
            try:
                in_acceptable_units.append(
                    arg.unit.is_equivalent(unit, equivalencies=equiv)
                )
            except AttributeError:
                if hasattr(arg, "unit"):
                    err_specifier = (
                        "a 'unit' attribute without an 'is_equivalent' method"
                    )
                else:
                    err_specifier = "no 'unit' attribute"

                msg = (
                    f"{err_msg} has {err_specifier}. "
                    f"Use an astropy Quantity instead."
                )
                return None, None, None, TypeError(msg)

        # How many acceptable units?
        nacceptable = np.count_nonzero(in_acceptable_units)
        unit = None
        equiv = None
        err = None
        if nacceptable == 0:
            # NO equivalent units
            arg = None
            err = u.UnitTypeError(typeerror_msg)
        else:
            # is there an exact match?
            units_arr = np.array(arg_checks["units"])
            units_equal_mask = np.equal(units_arr, arg.unit)
            units_mask = np.logical_and(units_equal_mask, in_acceptable_units)
            if np.count_nonzero(units_mask) == 1:
                # matched exactly to a desired unit
                unit = units_arr[units_mask][0]
                equiv = arg_checks["equivalencies"]
            elif nacceptable == 1:
                # there is a match to 1 equivalent unit
                unit = units_arr[in_acceptable_units][0]
                equiv = arg_checks["equivalencies"]
                if not arg_checks["pass_equivalent_units"]:
                    err = u.UnitTypeError(typeerror_msg)
            elif arg_checks["pass_equivalent_units"]:
                # there is a match to more than one equivalent units
                pass
            else:
                # there is a match to more than 1 equivalent units
                arg = None
                err = u.UnitTypeError(typeerror_msg)
        return arg, unit, equiv, err

    @staticmethod
    def _condition_target_units(targets: list, from_annotations: bool = False):
        """
        From a list of target units (either as a string or astropy
        :class:`~astropy.units.Unit` objects), return a list of conditioned
        :class:`~astropy.units.Unit` objects.

        Parameters
        ----------
        targets: list of target units
            list of units (either as a string or :class:`~astropy.units.Unit`)
            to be conditioned into astropy :class:`~astropy.units.Unit` objects

        from_annotations: bool
            (Default `False`) Indicates if `targets` originated from function/method
            annotations versus decorator input arguments.

        Returns
        -------
        list:
            list of `targets` converted into astropy
            :class:`~astropy.units.Unit` objects

        Raises
        ------
        TypeError
            If `target` is not a valid type for :class:`~astropy.units.Unit` when
            `from_annotations == True`,

        ValueError
            If a `target` is a valid unit type but not a valid value for
            :class:`~astropy.units.Unit`.
        """
        # Note: this method does not allow for astropy physical types. This is
        #       done because we expect all use cases of CheckUnits to define the
        #       exact units desired.
        #
        allowed_units = []
        for target in targets:
            try:
                target_unit = u.Unit(target)
                allowed_units.append(target_unit)
            except TypeError:
                # not a unit type
                if not from_annotations:
                    raise

                continue

        return allowed_units

    @staticmethod
    def _normalize_equivalencies(equivalencies):
        """
        Normalizes equivalencies to ensure each is in a 4-tuple form::

            (from_unit, to_unit, forward_func, backward_func)

        `forward_func` maps `from_unit` into `to_unit` and `backward_func` does
        the reverse.

        Parameters
        ----------
        equivalencies: list of equivalent pairs
            list of astropy :mod:`~astropy.units.equivalencies` to be normalized

        Raises
        ------
        ValueError
            if an equivalency can not be interpreted

        Notes
        -----
        * the code here was copied and modified from
          :func:`astropy.units.core._normalize_equivalencies` from AstroPy
          version 3.2.3
        * this will work on both the old style list equivalencies (pre AstroPy v3.2.1)
          and the modern equivalencies defined with the
          :class:`~astropy.units.equivalencies.Equivalency` class
        """
        if equivalencies is None:
            return []

        normalized = []

        def return_argument(x):
            return x

        for i, equiv in enumerate(equivalencies):
            if len(equiv) == 2:
                from_unit, to_unit = equiv
                a = b = return_argument
            elif len(equiv) == 3:
                from_unit, to_unit, a = equiv
                b = a
            elif len(equiv) == 4:
                from_unit, to_unit, a, b = equiv
            else:
                raise ValueError(f"Invalid equivalence entry {i}: {equiv!r}")

            if not (
                from_unit is u.Unit(from_unit)
                and (to_unit is None or to_unit is u.Unit(to_unit))
                and callable(a)
                and callable(b)
            ):
                raise ValueError(f"Invalid equivalence entry {i}: {equiv!r}")
            normalized.append((from_unit, to_unit, a, b))

        return normalized

    def _flatten_equivalencies_list(self, elist):
        """
        Given a list of equivalencies, flatten out any sub-element lists


        Parameters
        ----------
        elist: list
            list of astropy :mod:`~astropy.units.equivalencies` to be flattened

        Returns
        -------
        list
            a flattened list of astropy :mod:`~astropy.units.equivalencies`

        """
        new_list = []
        for el in elist:
            if not isinstance(el, list):
                new_list.append(el)
            else:
                new_list.extend(self._flatten_equivalencies_list(el))

        return new_list


# TODO rewrite as per https://github.com/astropy/astropy/pull/12335/files


def check_units(
    func=None, checks_on_return: dict[str, Any] = None, **checks: dict[str, Any]
):
    """
    A decorator to 'check' — limit/control — the units of input and return
    arguments to a function or method.

    Parameters
    ----------
    func:
        The function to be decorated

    checks_on_return: list of :mod:`~astropy.units` or dict of unit specifications
        Specifications for unit checks on the return of the function
        being wrapped. (see "check units"_ for valid specifications)

    **checks: list of :mod:`~astropy.units` or dict of unit specifications
        Specifications for unit checks on the input arguments of the
        function being wrapped.  Each keyword argument in ``checks`` is
        the name of a function argument to be checked and the keyword
        value contains the unit check specifications.

        .. _`check units`:

        Unit checks can be defined by passing one of the astropy
        :mod:`~astropy.units`, a list of astropy units, or a dictionary
        containing the keys defined below.  Units can also be defined
        with function annotations, but must be consistent with decorator
        ``**checks`` arguments if used concurrently. If a key is
        omitted, then the default value will be assumed.

        ====================== ======= ================================================
        Key                    Type    Description
        ====================== ======= ================================================
        units                          list of desired astropy :mod:`~astropy.units`
        equivalencies                  | [DEFAULT `None`] A list of equivalent pairs to
                                         try if
                                       | the units are not directly convertible.
                                       | (see :mod:`~astropy.units.equivalencies`,
                                         and/or `astropy equivalencies`_)
        pass_equivalent_units  `bool`  | [DEFAULT `False`] allow equivalent units
                                       | to pass
        ====================== ======= ================================================

    Notes
    -----
    * Checking of function arguments ``*args`` and ``**kwargs`` is not
      supported.
    * Decorator does NOT perform any unit conversions, look to
      :func:`~plasmapy.utils.decorators.validators.validate_quantities`
      if that functionality is desired.
    * If it is desired that `None` values do not raise errors or
      warnings, then include `None` in the list of units or as a default
      value for the function argument.
    * If units are not specified in ``checks``, then the decorator will
      attempt to identify desired units by examining the function
      annotations.
    * Full functionality is defined by the class :class:`CheckUnits`.

    Examples
    --------
    Define units with decorator parameters::

        import astropy.units as u
        from plasmapy.utils.decorators import check_units

        @check_units(arg1={'units': u.cm},
                     arg2=u.cm,
                     checks_on_return=[u.cm, u.km])
        def foo(arg1, arg2):
            return arg1 + arg2

        # or on a method
        class Foo:
            @check_units(arg1={'units': u.cm},
                         arg2=u.cm,
                         checks_on_return=[u.cm, u.km])
            def bar(self, arg1, arg2):
                return arg1 + arg2

    Define units with function annotations::

        import astropy.units as u
        from plasmapy.utils.decorators import check_units

        @check_units
        def foo(arg1: u.cm, arg2: u.cm) -> u.cm:
            return arg1 + arg2

        # or on a method
        class Foo:
            @check_units
            def bar(self, arg1: u.cm, arg2: u.cm) -> u.cm:
                return arg1 + arg2

    Allow `None` values to pass::

        import astropy.units as u
        from plasmapy.utils.decorators import check_units

        @check_units(checks_on_return=[u.cm, None])
        def foo(arg1: u.cm = None):
            return arg1

    Allow return values to have equivalent units::

        import astropy.units as u
        from plasmapy.utils.decorators import check_units

        @check_units(arg1={'units': u.cm},
                     checks_on_return={'units': u.km,
                                       'pass_equivalent_units': True})
        def foo(arg1):
            return arg1

    Allow equivalent units to pass with specified equivalencies::

        import astropy.units as u
        from plasmapy.utils.decorators import check_units

        @check_units(arg1={'units': u.K,
                           'equivalencies': u.temperature(),
                           'pass_equivalent_units': True})
        def foo(arg1):
            return arg1

    .. _astropy equivalencies:
        https://docs.astropy.org/en/stable/units/equivalencies.html
    """
    if checks_on_return is not None:
        checks["checks_on_return"] = checks_on_return

    if func is not None:
        # `check_units` called as a function
        return CheckUnits(**checks)(func)

    # `check_units` called as a decorator "sugar-syntax"
    return CheckUnits(**checks)


def check_values(
    func=None, checks_on_return: dict[str, bool] = None, **checks: dict[str, bool]
):
    """
    A decorator to 'check' — limit/control — the values of input and
    return arguments to a function or method.

    Parameters
    ----------

    func:
        The function to be decorated

    checks_on_return: Dict[str, bool]
        Specifications for value checks on the return of the function
        being wrapped. (see `check values`_ for valid specifications)

    **checks: Dict[str, Dict[str, bool]]
        Specifications for value checks on the input arguments of the
        function being wrapped.  Each keyword argument in ``checks`` is
        the name of a function argument to be checked and the keyword
        value contains the value check specifications.

        .. _`check values`:

        The value check specifications are defined within a dictionary
        containing the keys defined below.  If the dictionary is empty
        or omitting keys, then the default value will be assumed for the
        missing keys.

        ================ ======= ================================================
        Key              Type    Description
        ================ ======= ================================================
        can_be_negative  `bool`  [DEFAULT `True`] values can be negative
        can_be_complex   `bool`  [DEFAULT `False`] values can be complex numbers
        can_be_inf       `bool`  [DEFAULT `True`] values can be :data:`~numpy.inf`
        can_be_nan       `bool`  [DEFAULT `True`] values can be :data:`~numpy.nan`
        none_shall_pass  `bool`  [DEFAULT `False`] values can be a python `None`
        can_be_zero      `bool`  [DEFAULT `True`] values can be zero
        ================ ======= ================================================

    Notes
    -----
    * Checking of function arguments ``*args`` and ``**kwargs`` is not
      supported.
    * Full functionality is defined by the class :class:`CheckValues`.

    Examples
    --------
    .. code-block:: python

        from plasmapy.utils.decorators import check_values


        @check_values(
            arg1={"can_be_negative": False, "can_be_nan": False},
            arg2={"can_be_inf": False},
            checks_on_return={"none_shall_pass": True},
        )
        def foo(arg1, arg2):
            return None


        # on a method
        class Foo:
            @check_values(
                arg1={"can_be_negative": False, "can_be_nan": False},
                arg2={"can_be_inf": False},
                checks_on_return={"none_shall_pass": True},
            )
            def bar(self, arg1, arg2):
                return None
    """
    if checks_on_return is not None:
        checks["checks_on_return"] = checks_on_return

    if func is not None:
        # `check_values` called as a function
        return CheckValues(**checks)(func)

    # `check_values` called as a decorator "sugar-syntax"
    return CheckValues(**checks)


def check_relativistic(func=None, betafrac=0.05):
    """
    Warns or raises an exception when the output of the decorated
    function is greater than ``betafrac`` times the speed of light.

    Parameters
    ----------
    func : function, optional
        The function to decorate.

    betafrac : float, optional
        The minimum fraction of the speed of light that will raise a
        `~plasmapy.utils.exceptions.RelativityWarning`. Defaults to 5%.

    Returns
    -------
    function
        Decorated function.

    Raises
    ------
    TypeError
        If ``V`` is not a `~astropy.units.Quantity`.

    ~astropy.units.UnitConversionError
        If ``V`` is not in units of velocity.

    ValueError
        If ``V`` contains any `~numpy.nan` values.

    ~plasmapy.utils.exceptions.RelativityError
        If ``V`` is greater than or equal to the speed of light.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If ``V`` is greater than or equal to ``betafrac`` times the
        speed of light, but less than the speed of light.

    Examples
    --------
    >>> from astropy import units as u
    >>> @check_relativistic
    ... def speed():
    ...     return 1 * u.m / u.s

    Passing in a custom ``betafrac``:

    >>> @check_relativistic(betafrac=0.01)
    ... def speed():
    ...     return 1 * u.m / u.s

    """

    def decorator(f):
        @preserve_signature
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return_ = f(*args, **kwargs)
            _check_relativistic(return_, f.__name__, betafrac=betafrac)
            return return_

        return wrapper

    if func:
        return decorator(func)
    return decorator


def _check_relativistic(V, funcname, betafrac=0.05):
    r"""
    Warn or raise error for relativistic or superrelativistic
    velocities.

    Parameters
    ----------
    V : ~astropy.units.Quantity
        A velocity.

    funcname : str
        The name of the original function to be printed in the error
        messages.

    betafrac : float, optional
        The minimum fraction of the speed of light that will generate
        a warning. Defaults to 5%.

    Raises
    ------
    TypeError
        If ``V`` is not a `~astropy.units.Quantity`.

    ~astropy.units.UnitConversionError
        If ``V`` is not in units of velocity.

    ValueError
        If ``V`` contains any `~numpy.nan` values.

    RelativityError
        If ``V`` is greater than or equal to the speed of light.

    Warns
    -----
    ~plasmapy.utils.RelativityWarning
        If ``V`` is greater than or equal to the specified fraction of
        the speed of light.

    Examples
    --------
    >>> from astropy import units as u
    >>> _check_relativistic(1*u.m/u.s, 'function_calling_this')

    """

    # TODO: Replace `funcname` with func.__name__?

    errmsg = "V must be a Quantity with units of velocity in _check_relativistic"

    if not isinstance(V, u.Quantity):
        raise TypeError(errmsg)

    try:
        V_over_c = (V / c).to_value(u.dimensionless_unscaled)
    except u.UnitConversionError as ex:
        raise u.UnitConversionError(errmsg) from ex

    beta = np.max(np.abs(V_over_c))

    if beta == np.inf:
        raise RelativityError(f"{funcname} is yielding an infinite velocity.")
    elif beta >= 1:
        raise RelativityError(
            f"{funcname} is yielding a velocity that is {round(beta, 3)} "
            f"times the speed of light."
        )
    elif beta >= betafrac:
        warnings.warn(
            f"{funcname} is yielding a velocity that is "
            f"{round(beta * 100, 3)}% of the speed of "
            f"light. Relativistic effects may be important.",
            RelativityWarning,
        )
