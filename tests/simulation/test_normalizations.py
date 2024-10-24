import astropy.units as u

# from plasmapy.simulation.normalizations import MHDNormalizations

example_normalizations = {
    "length": 1 * u.m,
    "magnetic field": 1 * u.T,
    "number density": 1 * u.m**-3,
}


# def test_instantiation():
#    MHDNormalizations(1 * u.m, 1 * u.T, 1 * u.m**-3, ion="p+")
