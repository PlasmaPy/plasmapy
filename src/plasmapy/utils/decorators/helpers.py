"""Miscellaneous decorators."""

__all__ = ["modify_docstring", "preserve_signature"]

import functools
import inspect


def modify_docstring(func=None, prepend: str | None = None, append: str | None = None):
    r"""
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
        'Hello\n\nBeautiful\n\nWorld'

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

        if prepend is append is None:
            raise TypeError(
                "Decorator @modify_docstring() missing argument 'prepend' and/or"
                " 'append', at least one argument is required."
            )

        # save the original docstring
        wrapper.__original_doc__ = wrapper.__doc__
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
            appendlines = ["", *appendlines]
        elif append is None:
            appendlines = []
        else:
            raise TypeError(
                f"Expected type str for argument 'append', got {type(append)}."
            )

        # define new docstring
        wrapper.__doc__ = "\n".join(prependlines + doclines + appendlines)

        return wrapper

    # func is None: usage like @modify_docstring
    # func is not None: usage like @modify_docstring()

    return decorator(func) if func is not None else decorator


def preserve_signature(f):
    """
    A decorator for decorators, which preserves the signature of the function
    being wrapped. This preservation allows IDE function parameter hints to work
    on the wrapped function. To do this, the ``__signature__`` dunder is defined, or
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
    # add '__signature__' if it does not exist
    # - this will preserve parameter hints in IDE's
    if not hasattr(f, "__signature__"):
        f.__signature__ = inspect.signature(f)

    return f
