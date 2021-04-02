"""
Miscellaneous decorators for various package uses.
"""
__all__ = ["add_lite", "modify_docstring", "preserve_signature"]

import functools
import inspect


def modify_docstring(func=None, prepend: str = None, append: str = None):
    """
    A decorator which programmatically prepends and/or appends the docstring
    of the decorated method/function.  The unmodified/original docstring is
    saved as the ``__original_doc__`` attribute.

    Parameters
    ----------
    func: callable
        The method/function to be decorated.

    prepend: `str`
        The string to be prepended to the ``func``'s docstring.

    append: `str`
        The string to be appended to the ``func``'s docstring.

    Returns
    -------
    callable
        Wrapped version of the function.

    Examples
    --------

        >>> @modify_docstring(prepend='''Hello''', append='''World''')
        ... def foo():
        ...     '''Beautiful'''
        ...     pass
        >>> foo.__original_doc__
        'Beautiful'
        >>> foo.__doc__
        'Hello\\n\\nBeautiful\\n\\nWorld'

    """

    def decorator(f):
        sig = inspect.signature(f)

        @preserve_signature
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # combine args and kwargs into dictionary
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            return f(*bound_args.args, **bound_args.kwargs)

        if prepend is None and append is None:
            raise TypeError(
                "Decorator @modify_docstring() missing argument 'prepend' and/or"
                " 'append', at least one argument is required."
            )

        # save the original docstring
        setattr(wrapper, "__original_doc__", wrapper.__doc__)
        doclines = inspect.cleandoc(wrapper.__doc__).splitlines()

        # prepend docstring lines
        if isinstance(prepend, str):
            prependlines = inspect.cleandoc(prepend).splitlines()
            prependlines.append("")
        elif prepend is None:
            prependlines = []
        else:
            raise TypeError(
                f"Expected type str for argument 'prepend', got {type(prepend)}."
            )

        # append docstring lines
        if isinstance(append, str):
            appendlines = inspect.cleandoc(append).splitlines()
            appendlines = [""] + appendlines
        elif append is None:
            appendlines = []
        else:
            raise TypeError(
                f"Expected type str for argument 'append', got {type(append)}."
            )

        # define new docstring
        wrapper.__doc__ = "\n".join(prependlines + doclines + appendlines)

        return wrapper

    if func is not None:
        # `modify_docstring` called as a function
        return decorator(func)
    else:
        # `modify_docstring` called as a decorator "sugar-syntax"
        return decorator


def preserve_signature(f):
    """
    A decorator for decorators, which preserves the signature of the function
    being wrapped. This preservation allows IDE function parameter hints to work
    on the wrapped function. To do this, the `__signature__` dunder is defined, or
    inherited, from the function being wrapped to the resulting wrapped function.

    Parameters
    ----------
    f: callable
        The function being wrapped.

    Returns
    -------
    callable
        Wrapped version of the function.

    Examples
    --------
    >>> def a_decorator(f):
    ...     @preserve_signature
    ...     @functools.wraps(f)
    ...     def wrapper(*args, **kwargs):
    ...         return wrapper(*args, **kwargs)
    ...
    ...     return wrapper
    """
    # add '__signature__' to methods that are copied from
    # f onto wrapper
    assigned = list(functools.WRAPPER_ASSIGNMENTS)
    assigned.append("__signature__")

    @functools.wraps(f, assigned=assigned)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    # add '__signature__' if it does not exist
    # - this will preserve parameter hints in IDE's
    if not hasattr(wrapper, "__signature__"):
        wrapper.__signature__ = inspect.signature(f)

    return wrapper


def add_lite(flite, attrs=None, scope=None):
    """
    Decorator to bind lightweight "lite" versions of formulary functions to the full
    formulary function.
    """
    if attrs is None:
        attrs = []
    elif scope is None:
        raise ValueError(
            f"If 'attrs' are being bound to the decorate function, then 'scope'"
            f" needs to be defined and contain the objects to be bound.  "
            f"Setting 'scope=globals()' will typically suffice."
        )

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            return f(*args, **kwargs)

        wrapper_doc = inspect.cleandoc(getattr(wrapper, "__doc__", """"""))

        setattr(wrapper, "lite", flite)

        wrapper_doc += "\n\n" + inspect.cleandoc(
            f"""
            .. note:: To increase usability of `{f.__name__}` several
               attributes/functions are manually bound to this function.

               - `{f.__name__}.lite` <--> `~{flite.__module__}.{flite.__name__}`
            """
        )

        for bound_name, attr_name in attrs:
            attr = scope[attr_name]

            setattr(wrapper, bound_name, attr)

            wrapper_doc += (
                f"\n   - `{f.__name__}.{bound_name}` <--> `~{f.__module__}.{attr_name}`"
            )

        wrapper.__doc__ = wrapper_doc

        return wrapper

    return decorator
