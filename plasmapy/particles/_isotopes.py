"""
Module for loading isotope data from
:file:`plasmapy/particles/data/isotopes.json`.
"""
__all__ = []

import astropy.units as u
import json
import pkgutil

# this code was used to create the JSON file as per vn-ki on Matrix:
# https://matrix.to/#/!hkWCiyhQyxiYJlUtKF:matrix.org/
#    $1554667515670438wIKlP:matrix.org?via=matrix.org&via=cadair.com
#
# def _isotope_default(obj):
#     if isinstance(obj, u.Quantity):
#         return {
#             "unit": obj.unit.name,
#             "value": obj.value,
#         }
# with open("isotopes.json", "w") as f:
#     json.dump(_Isotopes, f, default=plasma_default, indent=2)


def isotope_obj_hook(obj):
    """Provide an ``object_hook`` designed for `json.load` and `json.loads`."""
    if "unit" in obj:
        return obj["value"] * u.Unit(obj["unit"])
    return obj


#: Dictionary of isotope data.
data_about_isotopes = json.loads(
    pkgutil.get_data("plasmapy", "particles/data/isotopes.json"),
    object_hook=isotope_obj_hook,
)
