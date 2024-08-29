"""Functions related to accurate integration of a sampled integrand."""

import numpy as np


def construct_integration_matrix(num_points, point_spacing):
    r"""Construct the matrix that is used for integrating a function, :math:`y(s)`, equally sampled in space.

    Parameters
    ----------
    num_points : `int`
        Number of sample points.
    point_spacing : `float`
        Distance between each sample point.

    Returns
    -------
    integration_matrix : `numpy.ndarray`
        2D-array of shape `(num_points, num_points)` of constants for each point, :math:`y(s_i)`.

    Notes
    -----
    This follows along the equations in appendix D from (D.20) onwards. We've
    changed the indexing to go from `0` to `n - 1` instead of `1` to `n`.

    Let :math:`Y(s)` be the integrated function where
    :math:`Y(s) = int_{s_0}^{s_{n - 1}} y(s) ds, n = num_points`. Then we can
    get :math:`Y(s_i)` for all :math:`i \in {0, ..., n - 1}` by doing
    `Y_values = integration_matrix @ y_values + integration_constant`,
    where :math:`integration_constant = Y(s_0)`.

    Examples
    --------
    >>> import numpy as np
    >>> num_points = 20
    >>> x_points, dx = np.linspace(0, 10, num_points, retstep=True)
    >>> integration_matrix = construct_integration_matrix(num_points, dx)
    >>> y_points = x_points**2
    >>> integrated_y_points = integration_matrix @ y_points
    """
    # TODO: Change this so that it takes in an array of points and then integrates instead of passing around this integration object.
    # TODO: This should be a slightly different matrix as the last term is -1 but should actually follow the last equation of (D.22) for all lines.
    if num_points < 3:
        raise ValueError("Can't create integration matrix for fewer than 3 points.")

    integration_matrix = np.zeros((num_points, num_points), dtype=float)
    for Y_index in range(1, num_points):
        if Y_index == 1:  # Y_1 is a parabolic arc through y_0, y_1, and y_2.
            integration_matrix[Y_index, : Y_index + 2] = np.array([5, 8, -1]) / 12
        elif (
            Y_index < num_points - 1
        ):  # Y_i is a cubic arc through y_{i - 2}, y_{i - 1}, y_{i}, and y_{i + 1}.
            integration_matrix[Y_index, : Y_index + 1] = integration_matrix[
                Y_index - 1, : Y_index + 1
            ]
            integration_matrix[Y_index, Y_index - 2 : Y_index + 2] += (
                np.array([-1, 13, 13, -1]) / 24
            )
        elif (
            Y_index == num_points - 1
        ):  # Y_{n - 1} is a parabolic arc through y_{n - 3}, y_{n - 2}, and y_{n - 1}.
            integration_matrix[Y_index, : Y_index + 1] = integration_matrix[
                Y_index - 1, : Y_index + 1
            ]
            integration_matrix[Y_index, Y_index - 2 : Y_index + 1] += (
                np.array([-1, 8, 5]) / 12
            )

    integration_matrix *= point_spacing
    return integration_matrix


def integrate(  # noqa: C901, PLR0912, PLR0915
    integrand,
    start_indeces,
    end_indeces,
    complex_before_start,
    complex_after_end,
    point_spacing,
):
    """Integrate an array where the start and end indices can be floats.

    This will only calculate the integrand where it is real.
    The integration is defined in the `CAL` function on
    page 37 - 39 of the computer program listing, appendix I.

    Parameters
    ----------
    integrand : `numpy.ndarray[float]`
        2D-array to integrate. Of shape `(num_x_points, num_y_points)`.
    start_indeces, end_indeces : `numpy.ndarray[float]`
        1D-array of indices of the `integrand` array to integrate between. Of
        shape `(num_x_points,)`. Additionally, `0 <= starts <= ends <= num_y_points - 1`.
    complex_before_start, complex_after_end : `numpy.ndarray[bool]`
        1D-array of booleans containing whether the integrand is complex before
        and after the start and end points respectively.
    point_spacing : `float`
        Distance between each sample point.

    Returns
    -------
    integral : `numpy.ndarray[float]`
        1D-array of shape `(num_x_points,)`.

    Notes
    -----
    This operates by using the integration described in appendix D for
    integration between the indices closest to the `starts` and `ends`. If the
    start or end point is not an integer then a Taylor expansion is done around
    either the closest integer index or one away from that depending on if the
    integrand is a complex number.

    The integrand must pass through zero to go from the reals to complex.
    """
    if np.any(start_indeces > end_indeces):
        raise ValueError("Start indices must be less than end indices.")

    # TODO: Convert this to a vectorized method.
    num_x_points, num_y_points = integrand.shape
    integral = np.zeros(num_x_points)

    for i, current_integrand in enumerate(integrand):
        start_index = start_indeces[i]
        end_index = end_indeces[i]

        if start_index == end_index:
            integral[i] = 0
            continue

        # Get `IA` and `IAX`. Lines 385, 201, 202, and 203.
        if np.isclose(start_index, round(start_index)):
            start_integer_index = round(start_index)
            start_bound_index = start_integer_index
        elif complex_before_start[i]:
            start_integer_index = int(start_index) + 1
            start_bound_index = start_integer_index
        else:
            start_integer_index = int(start_index)
            start_bound_index = max(0, start_integer_index - 1)
        # Get `IB` and `IBX`. Lines 205, 206, 207, and 208.
        if np.isclose(end_index, round(end_index)):
            end_integer_index = round(end_index)
            end_bound_index = end_integer_index
        elif complex_after_end[i]:
            end_integer_index = int(end_index)
            end_bound_index = end_integer_index
        else:
            end_integer_index = int(end_index) + 1
            end_bound_index = min(num_y_points - 1, end_integer_index + 1)

        # Calculate the contributions to the integral if the starting index is not an integer.
        internal_start_area = None
        if np.isclose(start_index, round(start_index)):
            # Line 231.
            interpolated_start_value = current_integrand[round(start_index)]
            start_area = 0
        elif complex_before_start[i]:
            # Line 233.
            # For the integrand to become complex it has to pass through 0. We'll do the trapezoidal rule with this in mind.
            interpolated_start_value = 0
            start_area = (
                (start_integer_index - start_index)
                * point_spacing
                * current_integrand[start_integer_index]
                / 2
            )
        else:
            if end_bound_index > start_integer_index + 1:
                # Line 326.
                # Do a second order Taylor expansion around `start_integer_index + 1`.
                distance = (start_index - (start_integer_index + 1)) * point_spacing
                zeroth_derivative = current_integrand[start_integer_index + 1]
                first_derivative = (
                    current_integrand[start_integer_index + 2]
                    - current_integrand[start_integer_index]
                ) / (2 * point_spacing)
                second_derivative = (
                    current_integrand[start_integer_index + 2]
                    - 2 * current_integrand[start_integer_index + 1]
                    + current_integrand[start_integer_index]
                ) / point_spacing**2
                interpolated_start_value = (
                    zeroth_derivative
                    + first_derivative * distance
                    + second_derivative / 2 * distance**2
                )
                # Integrate the expansion from start_index to start_integer_index + 1.
                start_area = (
                    -distance * zeroth_derivative
                    - 1 / 2 * distance**2 * first_derivative
                    - 1 / 3 * distance**3 * second_derivative / 2
                )
                # Also calculate the area between start_integer_index + 1 and start_integer_index + 2.
                internal_start_area = (
                    point_spacing * zeroth_derivative
                    + point_spacing**2 / 2 * first_derivative
                    + point_spacing**3 / 3 * second_derivative / 2
                )
            elif end_bound_index <= start_integer_index and complex_after_end[i]:
                # This is different than Laframboise in that we will do the
                # trapezoidal rule to estimate the start value when the start and
                # end are between the same grid points and the end is complex.
                interpolated_start_value = (
                    current_integrand[start_integer_index]
                    / (end_index - start_integer_index)
                    * (end_index - start_index)
                )
                start_area = interpolated_start_value / 2 * (end_index - start_index)
            elif start_integer_index <= start_bound_index:
                # Line 325.
                # Do a Taylor expansion up to the first derivative around `start_integer_index`.
                interpolated_start_value = current_integrand[start_integer_index] + (
                    current_integrand[start_integer_index + 1]
                    - current_integrand[start_integer_index]
                ) * (start_index - start_integer_index)
                # Now do the trapezoidal rule.
                start_area = (
                    (start_integer_index + 1 - start_index)
                    * point_spacing
                    * (
                        current_integrand[start_integer_index + 1]
                        + interpolated_start_value
                    )
                    * 0.5
                )
            else:
                # Line 363.
                # Do a second order Taylor expansion around the point just to the left of start.
                distance = (start_index - start_integer_index) * point_spacing
                zeroth_derivative = current_integrand[start_integer_index]
                first_derivative = (
                    current_integrand[start_integer_index + 1]
                    - current_integrand[start_integer_index - 1]
                ) / (2 * point_spacing)
                second_derivative = (
                    current_integrand[start_integer_index + 1]
                    - 2 * current_integrand[start_integer_index]
                    + current_integrand[start_integer_index - 1]
                ) / point_spacing**2
                interpolated_start_value = (
                    zeroth_derivative
                    + first_derivative * distance
                    + second_derivative / 2 * distance**2
                )
                # Now integrate from start to the point to the right of start.
                start_area = (
                    zeroth_derivative * (point_spacing - distance)
                    + 1 / 2 * first_derivative * (point_spacing**2 - distance**2)
                    + 1 / 6 * second_derivative * (point_spacing**3 - distance**3)
                )

            # Line 327.
            start_integer_index += 1

        # Calculate the contributions to the integral if the ending index is not an integer.
        internal_end_area = None
        if np.isclose(end_index, round(end_index)):
            # Line 236.
            interpolated_end_value = current_integrand[round(end_index)]
            end_area = 0
        elif complex_after_end[i]:
            # Line 238.
            interpolated_end_value = 0
            end_area = (
                (end_index - end_integer_index)
                * point_spacing
                * current_integrand[end_integer_index]
                / 2
            )
        else:
            if end_index > start_bound_index + 1:
                # Line 342.
                # Second order Taylor expansion at `end_integer_index - 1`.
                distance = (end_index - (end_integer_index - 1)) * point_spacing
                zeroth_derivative = current_integrand[end_integer_index - 1]
                first_derivative = (
                    current_integrand[end_integer_index]
                    - current_integrand[end_integer_index - 2]
                ) / (2 * point_spacing)
                second_derivative = (
                    current_integrand[end_integer_index]
                    - 2 * current_integrand[end_integer_index - 1]
                    + current_integrand[end_integer_index - 2]
                ) / point_spacing**2
                interpolated_end_value = (
                    zeroth_derivative
                    + first_derivative * distance
                    + second_derivative / 2 * distance**2
                )
                # Integrate from `end_integer_index - 1` to `end_index`.
                end_area = (
                    zeroth_derivative * distance
                    + first_derivative * 1 / 2 * distance**2
                    + 1 / 2 * second_derivative * 1 / 3 * distance**3
                )
                # Integrate from `end_integer_index - 2` to `end_integer_index - 1`.
                internal_end_area = (
                    point_spacing * zeroth_derivative
                    - point_spacing**2 / 2 * first_derivative
                    + point_spacing**3 / 3 * second_derivative / 2
                )
            elif end_integer_index <= start_bound_index and complex_before_start[i]:
                # This is different than Laframboise in that we will do the
                # trapezoidal rule to estimate the end value when the start and
                # end are between the same grid points and the start is complex.
                interpolated_end_value = (
                    current_integrand[end_integer_index]
                    / (end_integer_index - start_index)
                    * (end_index - start_index)
                )
                end_area = interpolated_end_value / 2 * (end_index - start_index)
            elif end_bound_index <= end_integer_index:
                # Line 341.
                # First order Taylor expansion at `end_integer_index - 1`.
                interpolated_end_value = current_integrand[end_integer_index - 1] + (
                    current_integrand[end_integer_index]
                    - current_integrand[end_integer_index - 1]
                ) * (end_index - (end_integer_index - 1))
                end_area = (
                    (end_index - (end_integer_index - 1))
                    * point_spacing
                    * (
                        current_integrand[end_integer_index - 1]
                        + interpolated_end_value
                    )
                    / 2
                )
            else:
                # Line 366.
                # Second order Taylor expansion at `end_integer_index`.
                distance = (end_index - end_integer_index) * point_spacing
                zeroth_derivative = current_integrand[end_integer_index]
                first_derivative = (
                    current_integrand[end_integer_index + 1]
                    - current_integrand[end_integer_index - 1]
                ) / (2 * point_spacing)
                second_derivative = (
                    current_integrand[end_integer_index + 1]
                    - 2 * current_integrand[end_integer_index]
                    + current_integrand[end_integer_index - 1]
                ) / point_spacing**2
                interpolated_end_value = (
                    zeroth_derivative
                    + first_derivative * distance
                    + second_derivative / 2 * distance**2
                )
                # Integrate from `end_integer_index - 1` to `end_index`.
                end_area = (
                    zeroth_derivative * (distance + point_spacing)
                    + 1 / 2 * first_derivative * (distance**2 - point_spacing**2)
                    + 1 / 6 * second_derivative * (distance**3 + point_spacing**3)
                )

            # Line 343.
            end_integer_index -= 1

        num_grid_points = end_integer_index - start_integer_index + 1
        # Add the partial contributions from the start and end areas.
        integral[i] = start_area + end_area
        # Line 239.
        if num_grid_points == 0:
            # If there are no points between `start` and `end` then either do the trapezoidal rule or a Taylor expansion.
            # We've addid in the additional qualifier of not complex so that we can do sub-grid integration.
            # Line 248.
            if start_bound_index < end_integer_index and not complex_after_end[i]:
                expansion_index = end_integer_index
            # Line 371.
            elif end_bound_index > start_integer_index and not complex_before_start[i]:
                expansion_index = start_integer_index
            else:
                # Line 372.
                # If there are no grid points and the valid bound points don't work to Taylor expand around then just do the trapezoidal rule.
                integral[i] = (
                    (interpolated_start_value + interpolated_end_value)
                    * (end_index - start_index)
                    * point_spacing
                    / 2
                )
                continue

            # Line 375.
            # Do a second order Taylor expansion around `expansion_index`.
            start_to_expansion = (start_index - expansion_index) * point_spacing
            end_to_expansion = (end_index - expansion_index) * point_spacing
            zeroth_derivative = current_integrand[expansion_index]
            first_derivative = (
                current_integrand[expansion_index + 1]
                - current_integrand[expansion_index - 1]
            ) / (2 * point_spacing)
            second_derivative = (
                current_integrand[expansion_index + 1]
                - 2 * current_integrand[expansion_index]
                + current_integrand[expansion_index - 1]
            ) / point_spacing**2
            # Integrate from `start` to `end`.
            integral[i] = (
                zeroth_derivative * (end_to_expansion - start_to_expansion)
                + 1
                / 2
                * first_derivative
                * (end_to_expansion**2 - start_to_expansion**2)
                + 1
                / 6
                * second_derivative
                * (end_to_expansion**3 - start_to_expansion**3)
            )
        elif num_grid_points == 1:
            # If there is only one point between `start` and `end` then just add the start and end area contributions.
            # Line 242.
            integral[i] = start_area + end_area
        elif num_grid_points == 2:
            # If there are two points between `start` and `end` then either do the trapezoidal rule or use the internal areas.
            if internal_end_area is not None and internal_start_area is not None:
                integral[i] += (internal_end_area + internal_start_area) / 2
            elif internal_end_area is not None:
                integral[i] += internal_end_area
            elif internal_start_area is not None:
                integral[i] += internal_start_area
            else:
                # TODO: This integration can be done better by checking if there are enough points to the left and right that you could do a Taylor expansion.
                integral[i] += (
                    (
                        current_integrand[start_integer_index]
                        + current_integrand[end_integer_index]
                    )
                    * point_spacing
                    / 2
                )

        # If there are at least 3 grid points then do the integration from equation (D.22).
        elif num_grid_points == 3:
            integral[i] += (
                (
                    current_integrand[start_integer_index]
                    + 4 * current_integrand[start_integer_index + 1]
                    + current_integrand[start_integer_index + 2]
                )
                * point_spacing
                / 3
            )
        elif num_grid_points == 4:
            integral[i] += (
                (
                    current_integrand[start_integer_index]
                    + 3 * current_integrand[start_integer_index + 1]
                    + 3 * current_integrand[start_integer_index + 2]
                    + current_integrand[start_integer_index + 3]
                )
                * point_spacing
                * 3
                / 8
            )
        elif num_grid_points == 5:
            integral[i] += (
                (
                    9 * current_integrand[start_integer_index]
                    + 28 * current_integrand[start_integer_index + 1]
                    + 22 * current_integrand[start_integer_index + 2]
                    + 28 * current_integrand[start_integer_index + 3]
                    + 9 * current_integrand[start_integer_index + 4]
                )
                * point_spacing
                / 24
            )
        else:
            integral[i] += (
                (
                    9 * current_integrand[start_integer_index]
                    + 28 * current_integrand[start_integer_index + 1]
                    + 23 * current_integrand[start_integer_index + 2]
                    + 23 * current_integrand[end_integer_index - 2]
                    + 28 * current_integrand[end_integer_index - 1]
                    + 9 * current_integrand[end_integer_index]
                    + 24
                    * np.sum(
                        current_integrand[
                            start_integer_index + 3 : end_integer_index - 2
                        ]
                    )
                )
                * point_spacing
                / 24
            )

    return integral
