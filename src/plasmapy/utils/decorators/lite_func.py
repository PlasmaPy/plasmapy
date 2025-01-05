"""Defines a decorator for binding |lite-functions| to regular functions."""

__all__ = ["bind_lite_func"]

import functools
import inspect
from collections.abc import Callable


class _LiteFuncDict(dict):
    """
    Dictionary of Lite-Function functionality bound to the parent
    function.  The dictionary key is a string indicating the name
    the functionality was bound with and the dictionary value is a
    string containing the fully qualified path of the original
    functionality.
    """

    # This is only to give __bound_lite_func__ a docstring.


def bind_lite_func(lite_func, attrs: dict[str, Callable] | None = None):
    """
    Decorator to bind a lightweight "lite" version of a formulary
    function to the full formulary function, as well as any supporting
    attributes.

    Parameters
    ----------
    lite_func: Callable
        The lightweight function to be bound as the ``lite`` attribute
        to the function being decorated.

    attrs: Dict[str, Callable]
        A dictionary where the key is a string defining the bound name
        and the associated value is the functionality to be bound.

    Examples
    --------
    .. autolink-skip:: section

    .. code-block:: python

        def foo_lite(x):
            return x


        def bar():
            print("Supporting function.")


        @bind_lite_func(
            foo_lite,
            attrs=[
                ("bar", bar),
            ],
        )
        def foo(x):
            if not isinstance(x, float):
                raise TypeError("Argument x can only be a float.")
            return x

    .. code-block:: pycon

        >>> foo(5)  # doctest: +SKIP
        5
        >>> foo.lite(5)  # doctest: +SKIP
        5
        >>> foo.bar()  # doctest: +SKIP
        Supporting function.

    Notes
    -----
    In addition to binding the functionality defined by the inputs, a
    ``__bound_lite_func__`` dunder is bound.  This dunder is a
    dictionary where a key is a string representing the bound name of
    the bound functionality and the associated value is a string
    representing the fully qualified path of the original bound
    functionality.
    """
    if attrs is None:
        attrs = {}
    elif not isinstance(attrs, dict):
        raise TypeError(
            f"Argument 'attrs' is a type '{type(attrs)}', expected a dictionary."
        )
    elif "lite" in attrs:
        raise ValueError(
            "Argument 'attr' can NOT define key 'lite', this is reserved for"
            " the 'lite_func' argument."
        )

    if inspect.isbuiltin(lite_func) or not inspect.isfunction(lite_func):
        raise ValueError("The given lite-function is not a user-defined function.")

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        __bound_lite_func__ = _LiteFuncDict()

        attrs["lite"] = lite_func
        for bound_name, attr in attrs.items():
            # only allow functions or jitted functions
            if not inspect.isfunction(attr):
                raise ValueError(
                    f"Can not bind obj '{attr}' to function '{wrapper.__name__}'."
                    f"  Only functions are allowed to be bound. Skipping."
                )

            # build origin name
            if hasattr(attr, "__module__"):
                modname = attr.__module__
            else:
                # assume attr is defined in the module the function being
                # decorated in
                modname = wrapper.__module__
            origin = f"{modname}.{attr.__name__}"
            __bound_lite_func__[bound_name] = origin

            # bind
            setattr(wrapper, bound_name, attr)

        wrapper.__bound_lite_func__ = __bound_lite_func__

        return wrapper

    return decorator
