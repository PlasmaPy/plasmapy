"""
Conversion to and from Roman numerals.

This file is adapted from the ``roman`` package that is available on
PyPI, which has the following copyright/license information:

Copyright (c) 2001 Mark Pilgrim

This program is free software; you can redistribute it and/or modify
it under the terms of the Python 2.1.1 license, available at
https://www.python.org/download/releases/2.1.1/license/
"""

__all__ = [
    "to_roman",
    "from_roman",
    "is_roman_numeral",
]

import re
from numbers import Integral

import numpy as np

from plasmapy.utils.exceptions import InvalidRomanNumeralError, OutOfRangeError

# Define digit mapping
_romanNumeralMap: tuple[tuple[str, int], ...] = (
    ("M", 1000),
    ("CM", 900),
    ("D", 500),
    ("CD", 400),
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
)

# Define pattern to detect valid Roman numerals
_romanNumeralPattern = re.compile(
    """
    ^                   # beginning of string
    M{0,4}              # thousands - 0 to 4 M's
    (CM|CD|D?C{0,3})    # hundreds - 900 (CM), 400 (CD), 0-300 (0 to 3 C's),
                        #            or 500-800 (D, followed by 0 to 3 C's)
    (XC|XL|L?X{0,3})    # tens - 90 (XC), 40 (XL), 0-30 (0 to 3 X's),
                        #        or 50-80 (L, followed by 0 to 3 X's)
    (IX|IV|V?I{0,3})    # ones - 9 (IX), 4 (IV), 0-3 (0 to 3 I's),
                        #        or 5-8 (V, followed by 0 to 3 I's)
    $                   # end of string
    """,
    re.VERBOSE,
)


def to_roman(n: int) -> str:
    """
    Convert an integer to a Roman numeral.

    Parameters
    ----------
    n : `int` or `~numpy.integer`
        The integer to be converted to a Roman numeral that must be
        between 1 and 4999, inclusive.

    Returns
    -------
    result : `str`
        The number in Roman numeral notation.

    Raises
    ------
    `TypeError`
        If the input is not an integer.

    `~plasmapy.utils.exceptions.OutOfRangeError`
        If the number is not between 1 and 4999, inclusive.

    See Also
    --------
    from_roman

    Examples
    --------
    >>> to_roman(5)
    'V'
    >>> to_roman(2525)
    'MMDXXV'

    """
    if not isinstance(n, Integral | np.integer):
        raise TypeError(f"{n} cannot be converted to a Roman numeral.")
    if not (0 < n < 5000):
        raise OutOfRangeError("Number is out of range (need 0 < n < 5000)")

    result = ""
    for numeral, integer in _romanNumeralMap:
        while n >= integer:
            result += numeral
            n -= integer
    return result


def from_roman(s: str) -> int:
    """
    Convert a Roman numeral to an integer.

    Parameters
    ----------
    s : `str`
        A Roman numeral.

    Returns
    -------
    result : `int`
        The positive integer corresponding to the Roman numeral.

    Raises
    ------
    `TypeError`
        The argument is not a `str`.

    `~plasmapy.utils.exceptions.InvalidRomanNumeralError`
        The argument is not a valid Roman numeral.

    See Also
    --------
    to_roman

    Examples
    --------
    >>> from_roman("V")
    5
    >>> from_roman("MMMMCCCLXVII")
    4367

    """
    if not isinstance(s, str):
        raise TypeError("The argument to from_roman must be a string.")
    if not _romanNumeralPattern.search(s):
        raise InvalidRomanNumeralError(f"Invalid Roman numeral: {s}")

    result = 0
    index = 0
    for numeral, integer in _romanNumeralMap:
        while s[index : index + len(numeral)] == numeral:
            result += integer
            index += len(numeral)
    return result


def is_roman_numeral(s: str) -> bool:
    """
    Check whether or not a string is a valid Roman numeral.

    Parameters
    ----------
    s : `str`
        The possible Roman numeral.

    Returns
    -------
    result : `bool`
        `True` if the `str` input is a valid Roman numeral, and `False`
        if it is not.

    Raises
    ------
    `TypeError`
        If the argument is not a `str`.

    See Also
    --------
    to_roman
    from_roman

    Examples
    --------
    >>> is_roman_numeral("CXVII")
    True
    >>> is_roman_numeral("42")
    False

    """
    if not isinstance(s, str):
        raise TypeError("Only strings may be tested ")
    return bool(_romanNumeralPattern.match(s))
