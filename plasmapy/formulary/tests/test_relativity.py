"""Tests for functions in relativity.py."""

import numpy as np
import pytest

from astropy import units as u
from astropy.constants import c

from plasmapy.formulary.relativity import Lorentz_factor, relativistic_energy
from plasmapy.utils.exceptions import RelativityError


def test_Lorentz_factor():
    r"""Test Lorentz_factor in relativity.py"""

    V = 123456789 * u.m / u.s
    assert np.isclose(Lorentz_factor(V), (1 / np.sqrt(1 - V ** 2 / c ** 2)).value)
    assert Lorentz_factor(-V) == Lorentz_factor(V)

    assert np.isclose(Lorentz_factor(0 * u.m / u.s), 1.0)
    assert Lorentz_factor(c) == np.inf

    V_arr = np.array([987532.0, 299792458]) * u.m / u.s
    gamma_arr = Lorentz_factor(V_arr)
    assert np.isclose(gamma_arr[0], (1 / np.sqrt(1 - V_arr[0] ** 2 / c ** 2)).value)
    assert gamma_arr[1] == np.inf

    assert (
        Lorentz_factor(3 * u.m / u.s) * u.dimensionless_unscaled
    ).unit == u.dimensionless_unscaled

    with pytest.raises(RelativityError):
        Lorentz_factor(1.0000000001 * c)

    with pytest.raises(ValueError), pytest.warns(u.UnitsWarning):
        Lorentz_factor(299792459)

    with pytest.warns(u.UnitsWarning):
        Lorentz_factor(2.2)

    with pytest.raises(u.UnitTypeError):
        Lorentz_factor(4 * u.kg)


def test_relativistic_energy():
    r"""Test relativistic_energy in relativity.py"""

    v = 123456789 * u.m / u.s
    m = 1 * u.kg
    assert np.isclose(
        relativistic_energy(m, v).value,
        ((1 / np.sqrt(1 - v ** 2 / c ** 2)) * m * c ** 2).value,
    )
    assert relativistic_energy(m, -v) == relativistic_energy(m, v)

    assert np.isclose(relativistic_energy(m, 0 * u.m / u.s).value, (m * c ** 2).value)
    assert relativistic_energy(m, c) == np.inf

    V_arr = np.array([987532.0, 299792458]) * u.m / u.s
    Energy_arr = relativistic_energy(m, V_arr)
    assert np.isclose(
        Energy_arr[0].value,
        ((1 / np.sqrt(1 - V_arr[0] ** 2 / c ** 2)) * m * c ** 2).value,
    )
    assert Energy_arr[1] == np.inf

    assert relativistic_energy(2 * u.kg, 3 * u.m / u.s).unit == u.J

    with pytest.raises(RelativityError):
        relativistic_energy(m, 1.0000000001 * c)

    with pytest.raises(RelativityError), pytest.warns(u.UnitsWarning):
        relativistic_energy(1, 299792459)

    with pytest.warns(u.UnitsWarning):
        relativistic_energy(m, 2.2)

    with pytest.raises(u.UnitTypeError):
        relativistic_energy(m, 4 * u.kg)

    with pytest.raises(ValueError):
        relativistic_energy(-m, v)

def test_quiver_velocity():
