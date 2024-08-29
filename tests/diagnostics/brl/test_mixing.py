"""Tests for calculating the mixing parameter."""

import numpy as np
import pytest

from plasmapy.diagnostics.brl import mixing


@pytest.mark.parametrize(
    ("coefficient_1", "coefficient_2"),
    [
        (-1, 0),
        (2, 0),
        (0, -1),
        (0, 2),
        (0.6, 0.6),
    ],
)
def test_mixing_function_error(coefficient_1, coefficient_2):
    """Test the mixing function raises an error for invalid coefficients."""
    with pytest.raises(ValueError):
        mixing.mixing_function(coefficient_1, coefficient_2, None, None, None)


@pytest.mark.parametrize(
    (
        "coefficient_1",
        "coefficient_2",
        "x_points",
        "effective_attracted_to_repelled_temperature_ratio",
        "normalized_probe_radius",
        "spherical",
        "expected_mixing_factors",
    ),
    (
        (
            0.5,
            0.3,
            np.array(
                [
                    1.0,
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                    0.3999999999999999,
                    0.29999999999999993,
                    0.19999999999999996,
                    0.1,
                ]
            ),
            1,
            0.1,
            True,
            np.array(
                [
                    0.8,
                    0.688027675182265,
                    0.5821239648113331,
                    0.48231688551423907,
                    0.3886520955094853,
                    0.3012093545089899,
                    0.22014159528501148,
                    0.1457834349505172,
                    0.07903200460356391,
                    0.023328482987029955,
                ]
            ),
        ),
        (
            0.1,
            0.8,
            np.array(
                [
                    1.0,
                    0.9,
                    0.8,
                    0.7,
                    0.6,
                    0.5,
                    0.3999999999999999,
                    0.29999999999999993,
                    0.19999999999999996,
                    0.1,
                ]
            ),
            0.3,
            10,
            False,
            np.array(
                [
                    0.9,
                    0.7716200059062848,
                    0.6627437871676334,
                    0.5680000702645568,
                    0.48201027927941975,
                    0.4002956352111799,
                    0.32001709766212716,
                    0.2400001542386893,
                    0.16000000001366466,
                    0.08000000000000002,
                ]
            ),
        ),
    ),
)
def test_mixing_function(
    coefficient_1,
    coefficient_2,
    x_points,
    effective_attracted_to_repelled_temperature_ratio,
    normalized_probe_radius,
    spherical,
    expected_mixing_factors,
):
    """Test the mixing function for various inputs."""
    mixing_factors = mixing.mixing_function(
        coefficient_1,
        coefficient_2,
        x_points,
        effective_attracted_to_repelled_temperature_ratio,
        normalized_probe_radius,
        spherical=spherical,
    )

    assert np.allclose(mixing_factors, expected_mixing_factors)


def test_decrease_mixing_coefficients():
    """Test that mixing coefficient decreases the correct amount."""
    coefficient_1 = 0.5
    coefficient_2 = 0.3

    new_coefficient_1, new_coefficient_2 = mixing.decrease_mixing_coefficients(
        coefficient_1, coefficient_2
    )

    assert new_coefficient_1 == coefficient_1 * 0.9
    assert new_coefficient_2 == coefficient_2 * 0.9


@pytest.mark.parametrize(
    (
        "coefficient_1",
        "coefficient_2",
        "expected_new_coefficient_1",
        "expected_new_coefficient_2",
    ),
    [
        (0.1, 0.1, 0.1111, 0.1111),
        (0.9, 0.05, 0.94445, 0.05555),
        (0.1, 0.8, 0.0875, 0.7),
        (1.0, 0.0, 1.0, 0.0),
    ],
)
def test_increase_mixing_coefficients(
    coefficient_1, coefficient_2, expected_new_coefficient_1, expected_new_coefficient_2
):
    new_coefficient_1, new_coefficient_2 = mixing.increase_mixing_coefficients(
        coefficient_1, coefficient_2
    )

    assert np.isclose(new_coefficient_1, expected_new_coefficient_1)
    assert np.isclose(new_coefficient_2, expected_new_coefficient_2)


@pytest.mark.parametrize(
    (
        "currents",
        "diverges",
    ),
    [
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), False),
        (np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10]), True),
        (np.array([1, -2, 3, -4, 5, -6, 7, -8, 9, -10]) + 50, True),
        ((-1) ** np.arange(10), True),
        ((-0.85) ** np.arange(10), False),
    ],
)
def test_check_oscillation_divergence(currents, diverges):
    """Test that the function correctly identifies oscillation divergence."""
    assert mixing.check_oscillation_divergence(currents) == diverges


def test_estimate_num_iterations_decaying_diverging_points():
    """Test that the `estimate_num_iterations` function correctly returns that the points are not converging."""
    currents = np.zeros(21)
    currents[-1] = 1
    currents[-11] = -1
    currents[-21] = 1

    is_converging, _, _ = mixing.estimate_num_iterations_decaying(currents, 0)
    assert is_converging is False


@pytest.mark.parametrize(
    (
        "exponential_coefficients",
        "accuracy",
        "expected_num_iterations",
    ),
    [
        [[1, np.log(2), 10], 0.1 * 2**-30, 9],
        [[1, np.log(2), 1], 2**-25, 4],
        [[1, np.log(1.1), 1], 1.1**-100, 79],
        [[1, 2, 0], 4**-10, np.inf],
    ],
)
def test_estimate_num_iterations_decaying_converging_points(
    exponential_coefficients, accuracy, expected_num_iterations
):
    """
    Test that the `estimate_num_iterations` function correctly returns that the
    points are converging and estimates the correct number of iterations.

    Parameters
    ----------
    exponential_coefficients : np.ndarray
        The terms needed to define an exponential function.
    """

    def current_function(x):
        return (
            exponential_coefficients[0] * np.exp(-exponential_coefficients[1] * x)
            + exponential_coefficients[2]
        )

    currents = current_function(np.arange(21))
    (
        is_converging,
        num_iterations,
        final_current,
    ) = mixing.estimate_num_iterations_decaying(currents, accuracy)
    assert is_converging is True
    assert np.isclose(num_iterations, expected_num_iterations)
    assert np.isclose(final_current, exponential_coefficients[2])


@pytest.mark.parametrize(
    (
        "currents",
        "accuracy",
        "final_current",
        "oscillating",
        "fast_convergence",
    ),
    [
        (np.arange(40), 0.1, np.inf, False, False),
        ((-1) ** np.arange(40), 0.1, 0, True, False),
        (
            (-1) ** np.arange(40) * np.exp(-np.log(2) * np.arange(-40, 0)) + 1,
            2**-45,
            1,
            True,
            False,
        ),
        (
            (-1) ** np.arange(40) * np.exp(-np.log(2) * np.arange(-40, 0)) + 1,
            2**-35,
            1,
            True,
            True,
        ),
    ],
)
def test_estimate_num_iterations_oscillating(
    currents, accuracy, final_current, oscillating, fast_convergence
):
    """
    Test that the `estimate_num_iterations_oscillations` function correctly
    identifies oscillations and fast convergence.
    """
    is_oscillating, is_fast_convergence = mixing.estimate_num_iterations_oscillating(
        currents, accuracy, final_current
    )
    assert is_oscillating == oscillating
    assert is_fast_convergence == fast_convergence


@pytest.mark.parametrize(
    (
        "currents",
        "relative_accuracy",
        "has_converged",
    ),
    [
        (np.arange(10), 0.1, False),
        (np.ones(10), 0.1, True),
    ],
)
def test_determine_convergence_zero_temperature_repelled_particles(
    currents, relative_accuracy, has_converged
):
    """Test that the function correctly identifies convergence for zero temperature repelled particles."""
    converged = mixing.determine_convergence_zero_temperature_repelled_particles(
        currents, relative_accuracy
    )

    assert converged == has_converged


@pytest.mark.parametrize(
    (
        "currents",
        "relative_accuracy",
        "final_current",
        "new_net_charge_density",
        "old_net_charge_density",
        "x",
        "has_converged",
    ),
    [
        (np.array([1, 2]), 0.1, 2, None, None, None, False),
        (
            np.array([1, 1]),
            0.1,
            1,
            np.ones(5),
            np.zeros(5),
            np.linspace(1, 0.1, 5),
            False,
        ),
        (
            np.array([1, 1]),
            0.1,
            1,
            np.ones(5),
            np.ones(5),
            np.linspace(1, 0.1, 5),
            True,
        ),
        (
            np.array([1, 1]),
            0.1,
            1,
            np.ones(5),
            1.05 * np.ones(5),
            np.linspace(1, 0.1, 5),
            True,
        ),
    ],
)
def test_determine_convergence(
    currents,
    relative_accuracy,
    final_current,
    new_net_charge_density,
    old_net_charge_density,
    x,
    has_converged,
):
    """Test that the function correctly identifies convergence."""
    converged = mixing.determine_convergence(
        currents,
        relative_accuracy,
        final_current,
        new_net_charge_density,
        old_net_charge_density,
        x,
    )

    assert converged == has_converged
