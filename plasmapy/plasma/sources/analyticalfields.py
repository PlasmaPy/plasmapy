"""
Defines the AnalyticalFields class used by PlasmaPy to represent electric and magnetic fields
varying as known functions of position.
"""

__all__ = ["AnalyticalFields"]

import astropy.units as u
import numpy as np

from typing import Callable

from plasmapy.plasma.plasma_base import GenericPlasma

_volt_over_meter = u.V / u.m  # for performance reasons


def _make_function(field_function: Callable):
    """Helper function that checks whether the passed function's return value is
    of shape (N, 3) or (3,), and standardizes the output to (N,3)."""
    test_array = np.zeros((4, 3))
    if field_function(test_array).shape != test_array.shape:
        # TODO speed this up! Numba usage is nonobvious here...
        output_function = lambda r: np.array([field_function(ri) for ri in r])
    else:
        output_function = field_function
    return output_function


class AnalyticalFields(GenericPlasma):
    """
    Allows passing analytical functions as fields.

    This is primarily helpful for `plasmapy.simulation.ParticleTracker`.

    Parameters
    ----------
    magnetic_field : Callable
        Function of position returning the magnetic field.
    electric_field : Callable
        Function of position returning the electric field.

    Notes
    -----
    Units are assumed SI - they will be overwritten with Tesla and V/m, respectively.
    """

    def __init__(self, magnetic_field: Callable, electric_field: Callable):
        self._interpolate_B = _make_function(magnetic_field)
        self._interpolate_E = _make_function(electric_field)

    def interpolate_B(self, r: u.m) -> u.T:
        """
        Returns the magnetic field, with units.

        Parameters
        ----------
        r : u.m
            Position in Cartesian (x, y, z) coordinates; can be a (3,) array or a (N, 3) array.

        Returns
        -------
        Magnetic field array of the same shape as the input.
        """
        return u.Quantity(self._interpolate_B(r), u.T, copy=False)

    def interpolate_E(self, r: u.m) -> u.V / u.m:
        """
        Returns the electric field, with units.

        Parameters
        ----------
        r : u.m
            Position in Cartesian (x, y, z) coordinates; can be a (3,) array or a (N, 3) array.

        Returns
        -------
        electric field array of the same shape as the input.
        """
        return u.Quantity(self._interpolate_E(r), _volt_over_meter, copy=False)
