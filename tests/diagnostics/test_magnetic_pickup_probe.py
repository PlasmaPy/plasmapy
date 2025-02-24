"""Tests for Magnetic Pickup probe functions."""

import astropy.constants.si as const
import astropy.units as u
import numpy as np
import pytest

from plasmapy.diagnostics import magnetic_pickup_probe

def test_magnetic_pickup_probe_values():
    np.

@pytest.mark.parametrize(
        ("args", "kwargs", "expected"),
        [
            ((1.0 * u.g * u.m**-3, ""), {}, 1.0e-3 * u.kg * u.m**-3),
            ((5.0e12 * u.cm**-3, "He"), {}, 3.32323849e-8 * u.kg * u.m**-3),
            (
                (5.0e12 * u.cm**-3, Particle("He")),
                {},
                3.32323849e-8 * u.kg * u.m**-3,
            ),
            (
                (5.0e12 * u.cm**-3, "He"),
                {"z_ratio": 0.5},
                1.66161925e-08 * u.kg * u.m**-3,
            ),
            (
                (5.0e12 * u.cm**-3, "He"),
                {"z_ratio": -0.5},
                1.66161925e-08 * u.kg * u.m**-3,
            ),
        ],
    )
    def test_values(self, args, kwargs, expected) -> None:
        assert np.isclose(mass_density(*args, **kwargs), expected)
