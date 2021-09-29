"""
The `~plasmapy.formulary` subpackage contains commonly used formulae
from plasma science.
"""
# __all__ & __aliases__ will be auto populated below
__all__ = []
__aliases__ = []

from .braginskii import *
from .collisions import *
from .dielectric import *
from .dimensionless import *
from .distribution import *
from .drifts import *
from .ionization import *
from .magnetostatics import *
from .mathematics import *
from .parameters import *
from .quantum import *
from .relativity import *
from .shocks import *

# auto populate __all__
for obj_name in list(globals()):
    if not (obj_name.startswith("__") or obj_name.endswith("__")):
        __all__.append(obj_name)
__all__.sort()

# auto populate __aliases__
for modname in (
    "braginskii",
    "collisions",
    "dielectric",
    "dimensionless",
    "distribution",
    "drifts",
    "ionization",
    "magnetostatics",
    "mathematics",
    "parameters",
    "quantum",
    "relativity",
):
    try:
        obj = globals()[modname]
        __aliases__.extend(obj.__aliases__)
    except (KeyError, AttributeError):
        pass
__aliases__.sort()

# cleanup namespace
del modname, obj, obj_name
