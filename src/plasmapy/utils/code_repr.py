"""Creation of string representations of function and method calls."""

__all__ = ["call_string", "attribute_call_string", "method_call_string"]

import inspect
from collections.abc import Callable
from typing import Any

import astropy.units as u
import numpy as np


def _code_repr_of_ndarray(array: np.ndarray, max_items=np.inf) -> str:
    """
    Transform a `~numpy.ndarray` into a format as would appear in a
    function call, after having done ``import numpy as np``.

    If ``max_items`` is less than ``array.size``, then include only the
    first ``max_items`` elements of the array in the resulting string
    and replace the remaining items with ``"..."``.
    """

    def remove_excess_spaces(s: str) -> str:
        s = " ".join(s.split())
        s = s.replace(" ,", ",")
        return s.replace("[ ", "[")

    def put_np_before_infs_and_nans(s: str) -> str:
        s = s.replace("inf", "np.inf")
        return s.replace("nan", "np.nan")

    def replace_excess_items_with_ellipsis(s: str, max_items: int):
        substrings_between_commas = s.split(",")
        to_comma_before_ellipsis = ",".join(substrings_between_commas[:max_items])
        closing_brackets = "]" * substrings_between_commas[-1].count("]")
        closing = f", ...{closing_brackets})"
        return to_comma_before_ellipsis + closing

    s = np.array_repr(array, max_line_width=np.inf, suppress_small=False)
    s = remove_excess_spaces(s)
    s = put_np_before_infs_and_nans(s)

    if array.size > max_items:
        s = replace_excess_items_with_ellipsis(s, max_items)

    return f"np.{s}"


def _code_repr_of_quantity(arg: u.Quantity, max_items=np.inf) -> str:
    """
    Transform a `~astropy.units.Quantity` into a format as would appear
    in a function call.
    """
    if isinstance(arg.value, np.ndarray):
        formatted = _code_repr_of_ndarray(arg.value, max_items=max_items)
    else:
        formatted = str(arg.value)

    if arg.unit == u.dimensionless_unscaled:
        formatted += "*u.dimensionless_unscaled"
    else:
        for base, power in zip(arg.unit.bases, arg.unit.powers, strict=False):
            if power == -1:
                formatted += f"/u.{base}"
            elif power == 1:
                formatted += f"*u.{base}"
            else:
                formatted += f"*u.{base}**{power}"

    return formatted


def _code_repr_of_arg(arg, max_items=np.inf) -> str:
    """Transform an argument into a format as would appear in a function call."""
    if hasattr(arg, "__name__"):
        return arg.__name__
    function_for_type = {
        u.Quantity: _code_repr_of_quantity,
        np.ndarray: _code_repr_of_ndarray,
    }
    return next(
        (
            code_repr_func(arg, max_items=max_items)
            for arg_type, code_repr_func in function_for_type.items()
            if isinstance(arg, arg_type)
        ),
        repr(arg),
    )


def _code_repr_of_args_and_kwargs(
    args: Any = None, kwargs: dict | None = None, max_items=np.inf
) -> str:
    """
    Take positional and keyword arguments, and format them into a
    string as they would appear in a function call (excluding parentheses).
    """
    args = () if args is None else args
    kwargs = {} if kwargs is None else kwargs

    args_collection = args if isinstance(args, tuple | list) else (args,)

    args_and_kwargs = "".join(
        f"{_code_repr_of_arg(arg, max_items)}, " for arg in args_collection
    )

    for kw_name, kw_val in kwargs.items():
        args_and_kwargs += f"{kw_name}={_code_repr_of_arg(kw_val, max_items)}, "

    if args_and_kwargs[-2:] == ", ":
        args_and_kwargs = args_and_kwargs[:-2]

    return args_and_kwargs


def _name_with_article(ex: Exception) -> str:
    """
    Return a string with an indefinite article and the name of
    the exception ``ex`` (e.g. ``Exception`` -> ``'an Exception'``).

    Notes
    -----
    If this function is to be expanded for cases beyond exceptions,
    we would need to either expand the treatment of cases that do not
    follow the general rule, or use a library like `inflect` on PyPI.
    """
    starts_with_vowel_but_uses_a = ["use", "uni"]
    name = ex.__name__
    use_an = all(
        [
            name[0] in "aeiouAEIOU",
            name[:3].lower() not in starts_with_vowel_but_uses_a,
        ]
    )

    indefinite_article = "an" if use_an else "a"
    return f"{indefinite_article} {name}"


def _object_name(obj: Any, showmodule: bool = False) -> str:
    """
    Return the name of an `object`.

    If the `object` has a `__name__` attribute and ``showmodule`` is
    `True`, then prepend the module name if not in `builtins`.  Replace
    ``"numpy"`` with ``"np"`` and substitute ``"u"`` for several modules
    in `astropy.units`.
    """

    def substitute_module_shortcuts(module_name: str):
        """Substitute common import shortcuts within module names."""
        replacements = {
            "numpy": "np",
            "builtins": "",
            "astropy.units.core": "u",
            "astropy.units.quantity": "u",
            "astropy.units.decorators": "u",
            "astropy.units": "u",
        }
        for old, new in replacements.items():
            if module_name.startswith(old):
                module_name = module_name.replace(old, new, 1)
        return module_name

    obj_name = obj.__name__ if hasattr(obj, "__name__") else repr(obj)
    module_name = inspect.getmodule(obj).__name__ if showmodule else ""
    module_name = substitute_module_shortcuts(module_name)
    if module_name:
        obj_name = f"{module_name}.{obj_name}"
    return obj_name


def _string_together_warnings_for_printing(
    warning_types: list[Warning], warning_messages: list[str]
):
    """
    Take a list of warning types with a list of corresponding warning
    messages, and create a string that prints out each warning type
    followed by the corresponding message, separated by a full line.
    """
    warnings_with_messages = [
        f"{_object_name(warning, showmodule=False)}: {message}"
        for warning, message in zip(warning_types, warning_messages, strict=False)
    ]

    return "\n\n".join(warnings_with_messages)


def call_string(
    f: Callable,
    args: Any = None,
    kwargs: dict[str, Any] | None = None,
    max_items: int = 12,
) -> str:
    """
    Approximate a call of a function or class with positional and
    keyword arguments.

    Parameters
    ----------
    f : callable
        A function, class, or other callable object.

    args : `tuple`, `list`, or any `object`; optional
        A `tuple` or `list` containing positional arguments, or any
        other `object` if there is only one positional argument.

    kwargs : `dict`, optional
        A `dict` containing keyword arguments.

    max_items : `int`, default: 12
        The maximum number of items to include in a `~numpy.ndarray` or
        `~astropy.units.Quantity`; additional items will be truncated
        with an ellipsis.

    Returns
    -------
    `str`
        Approximation to a call of ``f`` with ``args`` as positional
        arguments and ``kwargs`` as keyword arguments.

    See Also
    --------
    attribute_call_string
    method_call_string

    Notes
    -----
    This function will generally provide an exact call string for most
    common types of simple positional and keyword arguments.  When
    dealing with types that are not accounted for, this function will
    fall back on `repr`.

    This function assumes aliases of ``u`` for `astropy.units` and ``np``
    for `numpy`.

    Examples
    --------
    >>> call_string(int, 3.14159)
    'int(3.14159)'
    >>> call_string(int, args=(9.2,), kwargs={"base": 2})
    'int(9.2, base=2)'
    """
    args = () if args is None else args
    kwargs = {} if kwargs is None else kwargs
    args_and_kwargs = _code_repr_of_args_and_kwargs(args, kwargs, max_items)
    return f"{f.__name__}({args_and_kwargs})"


def attribute_call_string(
    cls,
    attr: str,
    args_to_cls: tuple | Any | None = None,
    kwargs_to_cls: dict[str, Any] | None = None,
    max_items: int = 12,
) -> str:
    """
    Approximate the command to instantiate a class, and access an
    attribute of the resulting class instance.

    Parameters
    ----------
    cls : `class`
        The class to be used in the string representation.

    attr: `str`
        The name of the attribute of class ``cls``.

    args_to_cls : `tuple`, `list`, or any `object`; optional
        A `tuple` or `list` containing positional arguments, or any
        other type of `object` if there is only one positional argument,
        to be used during instantiation of ``cls``

    kwargs_to_cls: `dict`, optional
        A `dict` containing the keyword arguments to be used during
        instantiation of ``cls``.

    max_items : `int`, default: 12
        The maximum number of items to include in a `~numpy.ndarray` or
        `~astropy.units.Quantity`; additional items will be truncated
        with an ellipsis.

    Returns
    -------
    `str`
        Approximation of a command to instantiate ``cls`` with
        ``args_to_cls`` as positional arguments and ``kwargs_to_cls`` as
        keyword arguments, and then access the attribute ``attr``.

    See Also
    --------
    attribute_call_string
    method_call_string

    Notes
    -----
    This function will generally provide an exact call string for most
    common types of simple positional and keyword arguments.  When
    dealing with types that are not accounted for, this function will
    fall back on `repr`.

    This function assumes aliases of ``u`` for `astropy.units` and ``np``
    for `numpy`.

    Examples
    --------
    >>> class SampleClass:
    ...     def __init__(self, arg1, kwarg1=None):
    ...         pass
    ...
    ...     @property
    ...     def attribute(self):
    ...         return 42
    >>> args_to_cls = (1, 2)
    >>> kwargs_to_cls = {"kwarg1": 2}
    >>> attribute_call_string(SampleClass, "attribute", args_to_cls, kwargs_to_cls)
    'SampleClass(1, 2, kwarg1=2).attribute'
    """
    args_to_cls = () if args_to_cls is None else args_to_cls
    kwargs_to_cls = {} if kwargs_to_cls is None else kwargs_to_cls
    return f"{call_string(cls, args_to_cls, kwargs_to_cls, max_items)}.{attr}"


def method_call_string(
    cls,
    method: str,
    *,
    args_to_cls: Any | None = None,
    kwargs_to_cls: dict[str, Any] | None = None,
    args_to_method: Any | None = None,
    kwargs_to_method: dict[str, Any] | None = None,
    max_items: int = 12,
) -> str:
    """
    Approximate the command to instantiate a class, and then call a
    method in the resulting class instance.

    Parameters
    ----------
    cls : `class`
        The class to be used in the string representation

    method: `str`
        The name of the method in class ``cls``

    args_to_cls : `tuple`, `list`, or any `object`; keyword-only, optional
        A `tuple` or `list` containing positional arguments, or any
        other type of `object` if there is only one positional argument,
        to be used during instantiation of ``cls``.

    kwargs_to_cls: `dict`, |keyword-only| optional
        A `dict` containing the keyword arguments to be used during
        instantiation of ``cls``.

    args_to_method : `tuple`, `list`, or any `object`; keyword-only, optional
        A `tuple` or `list` containing the positional arguments to be
        used in the method call, or any other `object` if there is only
        one positional argument.

    kwargs_to_method: `dict`, |keyword-only|, optional
        A `dict` containing the keyword arguments to be used during
        the method call.

    max_items : int, |keyword-only|, default: 12
        The maximum number of items to include in a `~numpy.ndarray` or
        `~astropy.units.Quantity`; additional items will be truncated
        with an ellipsis.

    Returns
    -------
    `str`
        Approximation of a command to instantiate ``cls`` with
        ``args_to_cls`` as positional arguments and ``kwargs_to_cls`` as
        keyword arguments, and then call a method of the resulting
        instance with ``args_to_method`` as positional arguments and
        ``kwargs_to_method`` as keyword arguments.

    See Also
    --------
    call_string
    attribute_call_string

    Notes
    -----
    This function will generally provide an exact call string for most
    common types of simple positional and keyword arguments.  When
    dealing with types that are not accounted for, this function will
    fall back on `repr`.

    This function assumes aliases of ``u`` for `astropy.units` and ``np``
    for `numpy`.

    Examples
    --------
    >>> class SampleClass:
    ...     def __init__(self, cls_arg, cls_kwarg=None):
    ...         pass
    ...
    ...     def method(self, method_arg, method_kwarg=None):
    ...         return 42
    >>> c_args = (1,)
    >>> c_kwargs = {"cls_kwarg": 2}
    >>> m_args = 3
    >>> m_kwargs = {"method_kwarg": 4}
    >>> method_call_string(
    ...     SampleClass,
    ...     "method",
    ...     args_to_cls=c_args,
    ...     kwargs_to_cls=c_kwargs,
    ...     args_to_method=m_args,
    ...     kwargs_to_method=m_kwargs,
    ... )
    'SampleClass(1, cls_kwarg=2).method(3, method_kwarg=4)'
    """
    class_call_string = f"{call_string(cls, args_to_cls, kwargs_to_cls, max_items)}"
    args_to_method_and_kwargs = _code_repr_of_args_and_kwargs(
        args_to_method, kwargs_to_method
    )
    return f"{class_call_string}.{method}({args_to_method_and_kwargs})"
