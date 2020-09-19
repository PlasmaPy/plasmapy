"""
The diagnostics subpackage contains functionality for defining diagnostic parameters
and processing data collected by diagnostics, synthetic or experimental.
"""
__all__ = [
    "AbstractProbe",
    "XAbstractDiagnostic",
    "xdiagnostics",
    "core",
    "langmuir",
    "thomson",
]

from plasmapy.diagnostics import core, langmuir, thomson

from .core import AbstractProbe, XAbstractDiagnostic, XDiagnosticEnabler


xdiagnostics = XDiagnosticEnabler()
"""
An instance of `~plasmapy.diagnostics.core.XDiagnosticEnabler` that allows users 
to only enable the xarray diagnostic accessors they want to use.

Examples
--------
>>> import xarray as xr
>>> from plasmapy.diagnostics import xdiagnostics

>>> xdiagnostics
<plasmapy.diagnostics.core.XDiagnosticEnabler>
Enabled   Available Diagnostic   Bound As
    [ ]   swept_langmuir

>>> xdiagnostics.enable('swept_langmuir')
>>> xdiagnostics
<plasmapy.diagnostics.core.XDiagnosticEnabler>
Enabled   Available Diagnostic   Bound As
    [x]   swept_langmuir         swept_langmuir

>>> hasattr(xr.Dataset, 'swept_langmuir')
True
"""
