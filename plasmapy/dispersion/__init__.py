"""
The `~plasmapy.dispersion` subpackage contains functionality associated with
plasma dispersion relations, solvers and analytical solutions.
"""
# __all__ will be auto populated below
__all__ = []

from plasmapy.dispersion.dispersionfunction import (
    plasma_dispersion_func,
    plasma_dispersion_func_deriv,
)

# auto populate __all__
for obj_name in list(globals()):
    if not (obj_name.startswith("__") or obj_name.endswith("__")):
        __all__.append(obj_name)
__all__.sort()
