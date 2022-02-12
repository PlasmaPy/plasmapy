import astropy.units as u
import numpy as np

from astropy.tests.helper import assert_quantity_allclose

import plasmapy

from ..species import Species


def test_single_valued_Species_instance():
    electron = plasmapy.particles.electron
    electrons = Species(
        plasmapy.particles.electron, 1e20 * u.m ** -3, temperature=10 * u.eV
    )
    assert (
        electrons.mass_density() / electrons.particle.mass == electrons.number_density
    )
    assert_quantity_allclose(electrons.thermal_speed(), 1875537.262105 * u.m / u.s)


def test_multivalued_Species_instance():

    carbon = plasmapy.particles.Particle("C+")
    rho = np.linspace(0, 1, 100)
    density = 1e18 * u.m ** -3 * (1 - rho ** 2)
    temperature = 1 * u.eV * (1 - rho ** 2)

    impurity = Species(carbon, density, temperature)
    thermal_speed = impurity.thermal_speed()

    assert_quantity_allclose(thermal_speed[0], 4008.35317849 * u.m / u.s)
    assert_quantity_allclose(thermal_speed[-1], 0 * u.m / u.s)
