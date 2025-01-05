"""
Defines the core class used to represent plasma properties.

.. attention::

   |expect-api-changes|
"""

__all__ = ["Plasma3D"]

import itertools

import astropy.units as u
import numpy as np
from astropy.constants import mu0

from plasmapy.formulary.magnetostatics import MagnetoStatics
from plasmapy.plasma.plasma_base import GenericPlasma
from plasmapy.utils.decorators import validate_quantities


class Plasma3D(GenericPlasma):
    """
    Core class for describing and calculating plasma parameters with
    spatial dimensions.

    Parameters
    ----------
    domain_x : `~astropy.units.Quantity`
        1D array of x-coordinates for the plasma domain. Must have
        units convertible to length.

    domain_y : `~astropy.units.Quantity`
        1D array of y-coordinates for the plasma domain. Must have
        units convertible to length.

    domain_z : `~astropy.units.Quantity`
        1D array of z-coordinates for the plasma domain. Must have
        units convertible to length.

    **kwargs:
        Any keyword accepted by `~plasmapy.plasma.plasma_base.GenericPlasma`
    """

    @validate_quantities(domain_x=u.m, domain_y=u.m, domain_z=u.m)
    def __init__(self, domain_x, domain_y, domain_z, **kwargs) -> None:
        super().__init__(**kwargs)

        # Define domain sizes
        self._x = domain_x
        self._y = domain_y
        self._z = domain_z

        self._grid = np.array(np.meshgrid(self._x, self._y, self._z, indexing="ij"))
        self._domain_shape = (len(self._x), len(self._y), len(self._z))

        # Initiate core plasma variables
        self._density = np.zeros(self.domain_shape) * u.kg / u.m**3
        self._momentum = np.zeros((3, *self.domain_shape)) * u.kg / (u.m**2 * u.s)
        self._pressure = np.zeros(self.domain_shape) * u.Pa
        self._magnetic_field = np.zeros((3, *self.domain_shape)) * u.T
        self._electric_field = np.zeros((3, *self.domain_shape)) * u.V / u.m

    @property
    def x(self):
        """
        (`~astropy.units.Quantity`) x-coordinates within the plasma domain. Equal to
        the ``domain_x`` input parameter.
        """
        return self._x

    @property
    def y(self):
        """
        (`~astropy.units.Quantity`) y-coordinates within the plasma domain. Equal
        to the ``domain_y`` input parameter.
        """
        return self._y

    @property
    def z(self):
        """
        (`~astropy.units.Quantity`) z-coordinates within the plasma domain. Equal
        to the ``domain_z`` input parameter.
        """
        return self._z

    @property
    def grid(self):
        """
        (`~astropy.units.Quantity`) (3, x, y, z) array containing the values of each
        coordinate at every point in the domain.
        """
        return self._grid

    @property
    def domain_shape(self) -> tuple:
        """(`tuple`) Shape of the plasma domain."""
        return self._domain_shape

    @property
    def density(self):
        """
        (`~astropy.units.Quantity`) (x, y, z) array of mass density at every
        point in the domain.
        """
        return self._density

    @property
    def momentum(self):
        """
        (`~astropy.units.Quantity`) (3, x, y, z) array of the momentum vector at
        every point in the domain.
        """
        return self._momentum

    @property
    def pressure(self):
        """
        (`~astropy.units.Quantity`) (x, y, z) array of pressure at every point
        in the domain.
        """
        return self._pressure

    @property
    def magnetic_field(self):
        """
        (`~astropy.units.Quantity`) (3, x, y, z) array of the magnetic field vector
        at every point in the domain.
        """
        return self._magnetic_field

    @property
    def electric_field(self):
        """
        (`~astropy.units.Quantity`) (3, x, y, z) array of the magnetic
        field vector at every point in the domain.
        """
        return self._electric_field

    @property
    def velocity(self):
        """
        (`~astropy.units.Quantity`) (3, x, y, z) array of the fluid
        velocity vector at every point in the domain.
        """
        return self.momentum / self.density

    @property
    def magnetic_field_strength(self):
        r"""
        Total field strength.

        .. math::
            \sqrt{ \sum{ B^2 }}

        """
        B = self.magnetic_field
        return np.sqrt(np.sum(B * B, axis=0))

    @property
    def electric_field_strength(self):
        r"""
        Total field strength.

        .. math::
            \sqrt{ \sum{ E^2 }}
        """
        E = self.electric_field
        return np.sqrt(np.sum(E * E, axis=0))

    @property
    def alfven_speed(self):
        """
        (`~astropy.units.Quantity`) (x, y, z) array of the Alfvén speed at
        every point in the domain.
        """
        B = self.magnetic_field
        rho = self.density
        return np.sqrt(np.sum(B * B, axis=0) / (mu0 * rho))

    @classmethod
    def is_datasource_for(cls, **kwargs):
        return (
            all(f"domain_{direction}" in kwargs for direction in "xyz")
            if len(kwargs) == 3
            else False
        )

    def add_magnetostatic(self, *mstats: MagnetoStatics) -> None:
        # for each MagnetoStatic argument
        prod = itertools.product(*[list(range(n)) for n in self.domain_shape])
        for mstat in mstats:
            # loop over 3D-index (ix,iy,iz)
            for point_index in prod:
                # get coordinate
                p = self.grid[(slice(None), *point_index)]  # function as [:, *index]
                # calculate magnetic field at this point and add back
                self.magnetic_field[(slice(None), *point_index)] += (
                    mstat.magnetic_field(p)
                )
