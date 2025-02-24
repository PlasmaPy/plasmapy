"""
PlasmaPy is an open source Python package for plasma research and
education.

For more information about the software, please check out `PlasmaPy's
online documentation <https://docs.plasmapy.org>`__ or use
`plasmapy.online_help`.

For more information about the PlasmaPy community, please check out
`PlasmaPy's website <https://www.plasmapy.org>`__.
"""

__all__ = [
    "online_help",
    "analysis",
    "diagnostics",
    "dispersion",
    "formulary",
    "particles",
    "plasma",
    "simulation",
    "transport",
    "utils",
    "__version__",
    "__citation__",
]

import sys

from plasmapy import (
    analysis,
    diagnostics,
    dispersion,
    formulary,
    particles,
    plasma,
    simulation,
    utils,
)

try:
    try:
        from plasmapy._dev.scm_version import version as __version__
    except ImportError:
        from plasmapy._version import (  # type: ignore[import-not-found,no-redef,unused-ignore]
            version as __version__,
        )
except Exception:  # coverage: ignore  # noqa: BLE001
    __version__ = "0.0.0"  # package is not installed

    import warnings

    warnings.warn(
        message=(
            "plasmapy.__version__ was not automatically generated, so "
            f"it was set to {__version__} instead. The installation may "
            "be broken."
        ),
        category=ImportWarning,
    )

    del warnings

__citation__ = (
    "Instructions on how to cite and acknowledge PlasmaPy are provided "
    "in the online documentation at: "
    "https://docs.plasmapy.org/en/stable/about/citation.html"
)


def online_help(query: str) -> None:  # coverage: ignore
    """
    Open a search page in |PlasmaPy's documentation|, or another page
    that contains relevant online help.

    This function requires an active internet connection, and will open
    the page in the default web browser.

    Parameters
    ----------
    query : str
        The search query.
    """
    import webbrowser
    from urllib.parse import urlencode

    url = (
        "http://docs.plasmapy.org/en/stable/search.html?"
        "{}&check_keywords=yes&area=default"
    ).format(urlencode({"q": query}))

    if query.lower() in {"unit", "units", "quantity", "quantities"}:
        url = "http://docs.astropy.org/en/stable/units"

    webbrowser.open(url)


del sys
