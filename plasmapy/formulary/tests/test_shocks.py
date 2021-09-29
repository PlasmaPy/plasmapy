import astropy.units as u
from numpy import isclose, isnan
import pytest

from plasmapy.formulary import entropy_jump_polytropic
from plasmapy.utils import PhysicsWarning


def test_entropy_jump_polytropic():
    c_v = 15 * u.J / u.K
    gamma = 5 / 3  # ideal gas approx
    p_1 = 101325 * u.Pa
    p_2 = 2 * p_1
    rho_1 = 2 * u.kg / u.m ** 3
    rho_2 = 3 * u.kg / u.m ** 3
    val = entropy_jump_polytropic(c_v, p_1, p_2, rho_1, rho_2, gamma)

    assert val.unit == u.J / u.K
    assert isclose(val.value, 0.26058001)

    with pytest.warns(PhysicsWarning):
        # entropy jump can not be negative
        val = entropy_jump_polytropic(c_v, p_1, 0.5 * p_1, rho_1, rho_2, gamma)
        assert isnan(val)

    with pytest.warns(PhysicsWarning):
        # entropy jump can not be zero (shocks are irreversible)
        val = entropy_jump_polytropic(c_v, p_1, p_1, rho_1, rho_1, gamma)
        assert isnan(val)
