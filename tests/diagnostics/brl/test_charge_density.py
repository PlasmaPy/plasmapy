"""Tests for calculating charge density."""

import numpy as np
import pytest

from plasmapy.diagnostics.brl import charge_density


@pytest.mark.parametrize(
    ("beta_M", "correct_index"),
    [(0, 8), (6, 0), (4.5, 3), (3.5, 3), (2.5, 4), (0.5, 6)],
)
def test_find_beta_M_index(beta_M, correct_index):
    """Test that the beta_M index is correct."""
    beta_G = np.array([3, 4, 5, 3, 2, 1, 0.5, 0.1])

    assert charge_density.find_beta_M_index(beta_M, beta_G) == correct_index


@pytest.mark.parametrize(
    ("beta_M", "beta_M_index", "beta_G_func", "omega_M", "omega_G_func"),
    [
        (1, 1, lambda x: 2 - x, 1, lambda x: x),
        (1.8, 0, lambda x: 2 - x, 0.2, lambda x: x),
        (1.8, 1, lambda x: 2 - x, 0.2, lambda x: x),
        (0.5, 8, lambda x: (x - 9) ** 2, 9 - 0.5**0.5, lambda x: x),
        (0.5, 9, lambda x: (x - 9) ** 2, 9 - 0.5**0.5, lambda x: x),
        (-5, 2, lambda x: -(x**2), 5**0.5, lambda x: x),
        (1, 1, lambda x: 2 - x, 2, lambda x: 2 * x),
        (1.8, 1, lambda x: 2 - x, 1.2, lambda x: 1 + x),
        (0.5, 9, lambda x: (x - 9) ** 2, (9 - 0.5**0.5) ** 2, lambda x: x**2),
        (-5, 2, lambda x: -(x**2), 1 - 5**0.5, lambda x: 1 - x),
    ],
)
def test_estimate_omega_M(beta_M, beta_M_index, beta_G_func, omega_M, omega_G_func):
    """Test that the returned omega_M is close."""
    sample_points = np.arange(0, 10)
    # beta_M_index = charge_density.find_beta_M_index(beta_M, beta_G_func(sample_points))
    assert np.isclose(
        charge_density.estimate_omega_M(
            beta_M,
            beta_M_index,
            beta_G_func(sample_points),
            omega_G_func(sample_points),
        ),
        omega_M,
    )


@pytest.mark.parametrize(
    ("chi", "x", "spherical", "expected_charge_density"),
    [
        (10 * np.ones(10), np.linspace(1, 0.1, num=10), False, np.zeros(10)),
        (10 * np.ones(10), np.linspace(1, 0.1, num=10), True, np.zeros(10)),
    ],
)
def test_delta_function_charge_density_high_potential(
    chi, x, spherical, expected_charge_density
):
    """Test that the delta function charge density is correct."""
    omega_G = -1 / (2 * x) * np.gradient(chi, x)
    beta_G = chi - x / 2 * np.gradient(chi, x)

    eta = charge_density.delta_function_charge_density(
        chi, x, omega_G, beta_G, spherical
    )
    assert eta.size == chi.size
    assert np.all(np.isclose(eta, expected_charge_density))


@pytest.mark.parametrize(
    ("chi", "x", "spherical"),
    [
        (np.zeros(10), np.linspace(1, 0.1, num=10), False),
        (np.zeros(10), np.linspace(1, 0.1, num=10), True),
    ],
)
def test_delta_function_charge_density_low_potential(chi, x, spherical):
    """Test that the delta function charge density is greater than 0 and increasing."""
    omega_G = -1 / (2 * x) * np.gradient(chi, x)
    beta_G = chi - x / 2 * np.gradient(chi, x)

    eta = charge_density.delta_function_charge_density(
        chi, x, omega_G, beta_G, spherical
    )
    assert eta.size == chi.size
    assert np.all(eta >= 0)
    assert np.all(np.diff(eta) >= 0)
