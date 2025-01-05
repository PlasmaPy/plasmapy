"""
Commonly used formulas for plasma science.

Many of these formulas come from the NRL Plasma Formulary
:cite:p:`nrlformulary`.
"""

__all__: list[str] = []
__aliases__: list[str] = []
__lite_funcs__: list[str] = []

from plasmapy.formulary.braginskii import *
from plasmapy.formulary.collisions import *
from plasmapy.formulary.densities import *
from plasmapy.formulary.dielectric import *
from plasmapy.formulary.dimensionless import *
from plasmapy.formulary.distribution import *
from plasmapy.formulary.drifts import *
from plasmapy.formulary.frequencies import *
from plasmapy.formulary.ionization import *
from plasmapy.formulary.lengths import *
from plasmapy.formulary.magnetostatics import *
from plasmapy.formulary.mathematics import *
from plasmapy.formulary.misc import *
from plasmapy.formulary.quantum import *
from plasmapy.formulary.radiation import *
from plasmapy.formulary.relativity import *
from plasmapy.formulary.speeds import *

# auto populate __all__
import inspect  # isort: skip

for obj_name in list(globals()):
    if not (
        obj_name.startswith("__") or obj_name.endswith("__")
    ) and not inspect.ismodule(globals()[obj_name]):
        __all__.append(obj_name)  # noqa: PYI056
__all__.sort()

# Put non-formulary imports here so that they don't get included in __all__

import contextlib  # isort: skip

# auto populate __aliases__ & __lite_funcs__
for modname in (
    "braginskii",
    "collisions",
    "densities",
    "dielectric",
    "dimensionless",
    "distribution",
    "drifts",
    "frequencies",
    "ionization",
    "lengths",
    "magnetostatics",
    "mathematics",
    "misc",
    "quantum",
    "radiation",
    "relativity",
    "speeds",
):
    try:
        obj = globals()[modname]
    except KeyError:
        continue

    with contextlib.suppress(AttributeError):
        __aliases__.extend(obj.__aliases__)

    with contextlib.suppress(AttributeError):
        __lite_funcs__.extend(obj.__lite_funcs__)

__aliases__ = sorted(set(__aliases__))
__lite_funcs__ = sorted(set(__lite_funcs__))

# cleanup namespace
del contextlib, inspect, modname, obj, obj_name
