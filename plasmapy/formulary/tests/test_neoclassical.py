import astropy.units as u
import pytest

from plasmapy.formulary.neoclassical import M_matrix, N_matrix
from plasmapy.particles import IonizationState, Particle, proton

hydrogen = IonizationState("p+", n_elem=1e20 * u.m ** -3, T_e=10 * u.eV)
carbon = IonizationState("C+", n_elem=1e19 * u.m ** -3, T_e=10 * u.eV)


@pytest.mark.parametrize(
    ["function", "shape"], [(N_matrix, (3, 3)), (M_matrix, (3, 3)),]
)
def test_matrix(function, shape, num_regression):
    data = function(hydrogen, carbon)
    assert data.shape == shape
    num_regression.check({function.__name__: data.flatten()})
