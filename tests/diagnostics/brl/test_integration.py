"""Tests for doing integration."""

import warnings
from itertools import product

import numpy as np
import pytest

from plasmapy.diagnostics.brl import integration


class Test__integrate:
    r"""Test the integrate function in integration.py"""

    num_points = 10
    sample_points = np.arange(num_points)
    sample_spacing = 1
    constant_integrand = np.ones(num_points)[np.newaxis, :]

    @staticmethod
    def constant_integral_func(x):
        return x

    real_integrand = (sample_points**2 + 3)[np.newaxis, :]
    real_integral = (1 / 3 * sample_points**3 + 3 * sample_points)[np.newaxis, :]

    @staticmethod
    def real_integral_func(x):
        return 1 / 3 * x**3 + 3 * x

    def test_invalid_start_after_end(self):
        r"""Test error integral if start before end."""
        start = np.array([3.3])
        end = start - 1

        with pytest.raises(ValueError):
            integration.integrate(
                self.constant_integrand,
                start,
                end,
                np.zeros_like(start),
                np.zeros_like(start),
                self.sample_spacing,
            )

    def test_multiple_integrals(self):
        r"""Test multiple integrals are computed if multiple start points given."""
        start = np.array([0, 2, 5])
        end = np.array([0, 3, 8])
        integrand = np.tile(self.constant_integrand, [start.size, 1])

        assert np.allclose(
            integration.integrate(
                integrand,
                start,
                end,
                np.zeros_like(start),
                np.zeros_like(start),
                self.sample_spacing,
            ),
            end - start,
        )

    @pytest.mark.parametrize(
        ("start_index", "integration_distance"),
        product(
            [0, 1, 2, 7, 8, 9, 0.3, 1.3, 2.3, 6.3, 7.3, 8.3],
            [0, 1, 2, 3, 4, 5, 6, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
        ),
    )
    def test_non_complex_start_and_end_combinations(
        self, start_index, integration_distance
    ):
        r"""Test integration for start and end points that aren't complex."""
        end_index = min(self.num_points - 1, start_index + integration_distance)

        # The large error for the starting and ending points being one apart is due to using an integration method of lower order than the integrand.
        if (
            np.isclose(end_index - start_index, 1)
            or self.num_points - 1 - end_index < 1
        ) and np.isclose(start_index, int(start_index)):
            absolute_error = 2 * 10**-1
            relative_error = 10**-2
        else:
            absolute_error = 10**-6
            relative_error = 10**-5

        start = np.array([start_index])
        end = np.array([end_index])

        assert np.isclose(
            integration.integrate(
                self.constant_integrand,
                start,
                end,
                np.zeros_like(start),
                np.zeros_like(start),
                self.sample_spacing,
            ),
            (self.constant_integral_func(end) - self.constant_integral_func(start)),
        )
        assert np.isclose(
            integration.integrate(
                self.real_integrand,
                start,
                end,
                np.zeros_like(start),
                np.zeros_like(start),
                self.sample_spacing,
            ),
            (self.real_integral_func(end) - self.real_integral_func(start)),
            atol=absolute_error,
            rtol=relative_error,
        )

    @pytest.mark.parametrize(
        ("start_index", "integration_distance", "is_complex_tuple"),
        product(
            [0, 1, 2, 7, 8, 9, 0.3, 1.3, 2.3, 6.3, 7.3, 8.3],
            [0, 1, 2, 3, 4, 5, 6, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
            [(True, True), (True, False), (False, True)],
        ),
    )
    def test_complex_start_and_end_combinations(
        self, start_index, integration_distance, is_complex_tuple
    ):
        r"""Test integration for start and end points that may be complex."""
        end_index = min(self.num_points - 1, start_index + integration_distance)
        complex_before_start, complex_after_end = is_complex_tuple

        # When there are very few points between the start and end there is larger error due to the method of integration.
        if (
            end_index - start_index - complex_before_start - complex_after_end <= 1
            or self.num_points - 1 - end_index < 1
        ):
            absolute_error = 5 * 10**-1
            relative_error = 10**-1
        else:
            absolute_error = 10**-1
            relative_error = 10**-1

        start = np.array([start_index])
        end = np.array([end_index])

        # Construct the integrand.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            if complex_before_start and complex_after_end:
                integrand = (
                    (self.sample_points - start_index) ** 0.5
                    * (end_index - self.sample_points) ** 0.5
                )[np.newaxis, :]

                def integral_func(x):
                    if start_index == end_index:
                        return 0
                    else:
                        return (
                            (x - start_index) ** 0.5
                            * (end_index - x) ** 0.5
                            * (start_index + end_index - 2 * x)
                            - (start_index - end_index) ** 2
                            * np.arcsin(
                                (end_index - x) ** 0.5
                                / (end_index - start_index) ** 0.5
                            )
                        ) / 4

            elif complex_before_start:
                integrand = ((self.sample_points - start_index) ** 0.5)[np.newaxis, :]

                def integral_func(x):
                    return 2 / 3 * (x - start_index) ** 1.5

            elif complex_after_end:
                integrand = ((end_index - self.sample_points) ** 0.5)[np.newaxis, :]

                def integral_func(x):
                    return 2 / 3 * -((end_index - x) ** 1.5)

        correct_result = integral_func(end) - integral_func(start)
        assert np.isclose(
            integration.integrate(
                integrand,
                start,
                end,
                np.array([complex_before_start]),
                np.array([complex_after_end]),
                self.sample_spacing,
            ),
            correct_result,
            atol=absolute_error,
            rtol=relative_error,
        )
