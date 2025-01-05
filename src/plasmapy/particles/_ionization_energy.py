"""
Functionality for loading ionization energy data from
:file:`src/plasmapy/particles/data/ionization_energy.json`.
"""

__all__ = [
    "ionization_energy_obj_hook",
    "data_about_ionization_energy",
]

import json
import pkgutil

import astropy.units as u


def ionization_energy_obj_hook(obj):
    """Provide an ``object_hook`` designed for `json.load` and `json.loads`."""
    return (
        (obj["ionization energy"] * u.eV).to(u.J) if "ionization energy" in obj else obj
    )


#: Dictionary of ionization energy data.
data_about_ionization_energy = json.loads(
    pkgutil.get_data("plasmapy", "particles/data/ionization_energy.json"),
    object_hook=ionization_energy_obj_hook,
)
