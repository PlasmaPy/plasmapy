"""Tests for calculating the net spacing."""

import numpy as np
import pytest

from plasmapy.diagnostics.brl import net_spacing


def test_get_s_points():
    """Test the returned `s_points`."""
    with pytest.raises(ValueError):
        net_spacing.get_s_points(1, -1)

    num_points = 20
    s_end_point = 0.4

    expect_num_points = num_points
    expect_s_start_point = 0
    expect_s_end_point = s_end_point
    expect_s_spacing = s_end_point / (num_points - 1)

    s_points = net_spacing.get_s_points(num_points, s_end_point)

    assert s_points.size == expect_num_points
    assert s_points[0] == expect_s_start_point
    assert np.allclose(s_points[-1], expect_s_end_point)
    assert np.allclose(
        s_points[1:] - s_points[:-1], expect_s_spacing * np.ones(num_points - 1)
    )


class Test__private_x_and_dx_ds:
    r"""Test the different available functions for getting `x` and `dx_ds`."""

    s_points = net_spacing.get_s_points(1000, 0.8)
    ds = s_points[1] - s_points[0]

    def approximate_derivative(self, x_points):
        r"""Approximate the value of dx/ds."""
        return (x_points[2:] - x_points[:-2]) / (2 * self.ds)

    def test_large_probe_x_and_dx_ds(self):
        r"""Test large probe x and dx/ds."""

        x_points, dx_ds_points = net_spacing._large_probe_x_and_dx_ds(self.s_points, 80)
        approximate_derivative = self.approximate_derivative(x_points)

        assert x_points[0] == 1
        assert np.allclose(
            dx_ds_points[1:-1], approximate_derivative
        ), f"The derivative, dx/ds, with value {dx_ds_points} should be close to the approximate derivative with value {approximate_derivative}."

    def test_medium_probe_x_and_dx_ds(self):
        r"""Test medium probe x and dx/ds."""

        x_points, dx_ds_points = net_spacing._medium_probe_x_and_dx_ds(self.s_points)
        approximate_derivative = self.approximate_derivative(x_points)

        assert x_points[0] == 1
        assert np.allclose(
            dx_ds_points[1:-1], approximate_derivative
        ), f"The derivative, dx/ds, with value {dx_ds_points} should be close to the approximate derivative with value {approximate_derivative}."

    def test_small_probe_x_and_dx_ds(self):
        r"""Test small probe x and dx/ds."""

        x_points, dx_ds_points = net_spacing._small_probe_x_and_dx_ds(self.s_points)
        approximate_derivative = self.approximate_derivative(x_points)

        assert x_points[0] == 1
        assert np.allclose(
            dx_ds_points[1:-1], approximate_derivative
        ), f"The derivative, dx/ds, with value {dx_ds_points} should be close to the approximate derivative with value {approximate_derivative}."

    def test_zero_T_repelled_x_and_dx_ds(self):
        r"""Test zero temperature repelled particles x and dx/ds calculation."""

        x_points, dx_ds_points = net_spacing._zero_T_repelled_x_and_dx_ds(
            self.s_points, 10, 25
        )
        approximate_derivative = self.approximate_derivative(x_points)

        assert x_points[0] == 1
        assert np.allclose(
            dx_ds_points[1:-1], approximate_derivative
        ), f"The derivative, dx/ds, with value {dx_ds_points} should be close to the approximate derivative with value {approximate_derivative}."


class Test__get_x_and_dx_ds:
    r"""Test the get_x_and_dx_ds function in net_spacing.py"""

    # These are all values given by Laframboise in Table 4.
    normalized_probe_potential = 25
    normalized_probe_radius_values = np.array(
        [0.5, 1, 2, 5, 10, 20, 50, 100, 1, 2, 5, 10, 20, 50, 100]
    )
    ds_values = np.array(
        [
            0.0667,
            0.05,
            0.0333,
            0.0133,
            0.01,
            0.005,
            0.005,
            0.005,
            0.025,
            0.025,
            0.01,
            0.01,
            0.0067,
            0.01,
            0.01,
        ]
    )
    points_per_debye_length_at_probe_values = np.array(
        [30, 20, 15, 15, 10, 10, 10, 10, 40, 20, 20, 10, 7.5, 5, 5], dtype=float
    )
    ds_dx_at_probe_values = np.array(
        [-1, -1, -1, -1, -1, -1, -2.5, -5, -1, -1, -1, -1, -1, -2.5, -5]
    )
    s_end_point_values = np.array(
        [
            2.8,
            2.4,
            2.0,
            0.8,
            0.72,
            0.56,
            0.56,
            0.5,
            2.9,
            2.3,
            0.8,
            0.72,
            0.6,
            0.56,
            0.56,
        ]
    )
    r_end_over_r_probe_values = np.array(
        [
            16.44,
            11.02,
            7.39,
            5.0,
            3.57,
            2.27,
            1.64,
            1.4,
            18.17,
            9.97,
            5.0,
            3.57,
            2.5,
            1.64,
            1.53,
        ]
    )
    r_end_minus_r_probe_over_debye_length_values = np.array(
        [
            7.7,
            10.0,
            12.8,
            20.0,
            25.7,
            25.5,
            31.8,
            40.3,
            17.2,
            17.9,
            20.0,
            25.7,
            30.0,
            31.8,
            53.4,
        ]
    )

    @staticmethod
    def get_num_s_poits(ds, s_end_point):
        return round(s_end_point / ds) + 1

    @pytest.mark.parametrize(
        ("normalized_probe_radius", "zero_T_repelled_particles"),
        [(3, False), (30, False), (1, True)],
    )
    def test_too_large_s_points(
        self, normalized_probe_radius, zero_T_repelled_particles
    ):
        r"""Test that an error is raised if the maximum `s` is greater than 1 for a non-small probe."""
        with pytest.raises(ValueError):
            net_spacing.get_x_and_dx_ds(
                np.linspace(0, 10),
                normalized_probe_radius,
                1,
                zero_T_repelled_particles=zero_T_repelled_particles,
            )

    @pytest.mark.parametrize(
        (
            "normalized_probe_radius",
            "ds",
            "points_per_debye_length_at_probe",
            "ds_dx_at_probe",
            "s_end_point",
            "r_end_over_r_probe",
            "r_end_minus_r_probe_over_debye_length",
        ),
        zip(
            normalized_probe_radius_values,
            ds_values,
            points_per_debye_length_at_probe_values,
            ds_dx_at_probe_values,
            s_end_point_values,
            r_end_over_r_probe_values,
            r_end_minus_r_probe_over_debye_length_values,
            strict=False,
        ),
    )
    def test_laframboise_table(
        self,
        normalized_probe_radius,
        ds,
        points_per_debye_length_at_probe,
        ds_dx_at_probe,
        s_end_point,
        r_end_over_r_probe,
        r_end_minus_r_probe_over_debye_length,
    ):
        r"""Test that the values from Table 4 of Laframboise match our calculations."""
        s_points = net_spacing.get_s_points(
            self.get_num_s_poits(ds, s_end_point), s_end_point
        )
        x, dx_ds = net_spacing.get_x_and_dx_ds(
            s_points,
            normalized_probe_radius,
            1,
            normalized_probe_potential=self.normalized_probe_potential,
        )
        normalized_r = 1 / x
        dr_dx_at_probe = -1 / 1**2
        calculated_points_per_debye_length_at_probe = (
            1 / ds * (normalized_probe_radius * dr_dx_at_probe * dx_ds[0]) ** -1
        )

        assert np.isclose(
            calculated_points_per_debye_length_at_probe,
            points_per_debye_length_at_probe,
            atol=0.5,
        ), f"The calculated number of points per debye length at the probe was {calculated_points_per_debye_length_at_probe} but Laframboise has it as {points_per_debye_length_at_probe}."
        assert np.isclose(
            1 / dx_ds[0], ds_dx_at_probe
        ), f"The value of ds/dx at the probe was {1 / dx_ds[0]} but Laframboise has it as {ds_dx_at_probe}."
        assert np.isclose(
            normalized_r[-1], r_end_over_r_probe, atol=0.005
        ), f"The final x point converted to a normalized radius with value {normalized_r[-1]} is not equal to the value from Laframboise, {r_end_over_r_probe}."
        calculated_r_end_minus_r_probe_over_debye_length = (
            normalized_r[-1] - 1
        ) * normalized_probe_radius
        assert np.isclose(
            calculated_r_end_minus_r_probe_over_debye_length,
            r_end_minus_r_probe_over_debye_length,
            atol=0.05,
        ), f"The value of (r_end - r_probe) / debye length was {calculated_r_end_minus_r_probe_over_debye_length} but Laframboise has it as {r_end_minus_r_probe_over_debye_length}."
