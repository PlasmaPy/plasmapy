"""
The `~plasmapy.formulary` subpackage contains commonly used formulae
from plasma science.
"""
# __all__ will be auto populated below
__all__ = []

# auto populate __all__
for obj_name in list(globals()):
    if not (obj_name.startswith("__") or obj_name.endswith("__")):
        __all__.append(obj_name)
__all__.sort()
