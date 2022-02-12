import astropy.units as u

from dataclasses import dataclass
from typing import Optional

from plasmapy.formulary import gyrofrequency, mass_density, thermal_speed

from . import particle_class


@dataclass
class Species:
    """
    A group of particles in a plasma with associated parameters:
    (number) density, temperature and/or pressure.

    Not all parameters need be specified at creation time.

    Parameters may be arrays, representing, for example,
    profiles of electron temperature and density from a
    stellarator diagnostic.
    """

    particle: particle_class.Particle
    number_density: Optional[u.Quantity] = None
    temperature: Optional[u.Quantity] = None
    pressure: Optional[u.Quantity] = None

    def mass_density(self, *args, **kwargs):
        """Thin wrapper for `~plasmapy.formulary.mass_density`"""
        return mass_density(self.number_density, self.particle, *args, **kwargs)

    def thermal_speed(self, *args, **kwargs):
        """Thin wrapper for `~plasmapy.formulary.thermal_speed`"""
        return thermal_speed(self.temperature, self.particle, *args, **kwargs)
