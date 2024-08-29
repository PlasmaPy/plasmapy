"""Calculate the mixing function and adjust the coefficients as evaluation continues."""

import logging

import numpy as np


def mixing_function(
    coefficient_1,
    coefficient_2,
    x,
    effective_attracted_to_repelled_temperature_ratio,
    normalized_probe_radius,
    spherical=True,
):
    r"""Calculate the weighting for mixing solutions.

    The prior solution is mixed with the current solution using the
    `mixing_factors` according to
    `mixed_eta = (1 - mixing_factors) * prior_eta + mixing_factors * current_eta`.
    These mixing factors should be between 0 and 1.

    Parameters
    ----------
    coefficient_1, coefficient_2 : `float`
        Coefficients used for calculating the mixing factors.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    effective_attracted_to_repelled_temperature_ratio : `float`
        :math:`-\frac{T_+ Z_-}{T_- Z_+}` from `~plasmapy.diagnostics.brl.normalizations.get_effective_attracted_to_repelled_temperature_ratio`.
    normalized_probe_radius : `float`
        The radius of the probe normalized to the attracted particle Debye length as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_probe_radius`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    mixing_factors : `numpy.ndarray`

    Notes
    -----
    This is the `COOKIE` function from page 19 of the thesis. `coefficient_1`
    and `coefficient_2` are `QT1` and `QT2` respectively. This returns an array
    that is referred to as `COOKIE` in the code.
    """
    if not (0 <= coefficient_1 <= 1):
        raise ValueError("The coefficient_1 must be between 0 and 1.")
    if not (0 <= coefficient_2 <= 1):
        raise ValueError("The coefficient_2 must be between 0 and 1.")
    if coefficient_1 + coefficient_2 > 1:
        raise ValueError("The sum of the coefficients must be less than or equal to 1.")

    power = (
        normalized_probe_radius**2
        * min(effective_attracted_to_repelled_temperature_ratio, 1)
    ) ** 0.5

    if spherical:
        # Line 100.
        return coefficient_1 * x * np.exp(power * (1 - 1 / x)) + coefficient_2 * x**2
    else:
        # Line 200.
        return coefficient_1 * x**0.5 * np.exp(power * (1 - 1 / x)) + coefficient_2 * x


def decrease_mixing_coefficients(coefficient_1, coefficient_2):
    r"""
    Decrease the mixing coefficients when a solution is being approached.

    Parameters
    ----------
    coefficient_1, coefficient_2 : `float`
        Coefficients used for calculating the mixing factors.

    Returns
    -------
    new_coefficient_1, new_coefficient_2 : `float`
        New coefficients to use for calculating the mixing factors.
    """
    return coefficient_1 * 0.9, coefficient_2 * 0.9


def increase_mixing_coefficients(coefficient_1, coefficient_2):
    r"""
    Increase the mixing coefficients if convergence is too slow.

    Parameters
    ----------
    coefficient_1, coefficient_2 : `float`
        Coefficients used for calculating the mixing factors.

    Returns
    -------
    new_coefficient_1, new_coefficient_2 : `float`
        New coefficients to use for calculating the mixing factors.

    Notes
    -----
    This is line 126 on page 16 of the `ADJUST` program.
    """
    # FACT in code.
    factor_2 = 1.111
    # QTY in code.
    increased_coefficient_2 = coefficient_2 * factor_2
    # QT2 in code.
    new_coefficient_2 = min(0.7, increased_coefficient_2)
    if increased_coefficient_2 == 0:
        new_coefficient_1 = 1 - new_coefficient_2
    else:
        factor_1 = factor_2 * new_coefficient_2 / increased_coefficient_2
        new_coefficient_1 = min(1 - new_coefficient_2, coefficient_1 * factor_1)

    return new_coefficient_1, new_coefficient_2


def check_oscillation_divergence(all_found_currents):
    r"""
    Test if divergent oscillations exist in the past 10 iterations.

    Parameters
    ----------
    all_found_currents : `numpy.ndarray`
        1D-array of size `iteration_number` of all currents to the probe found so far.

    Returns
    -------
    `bool`
        Whether divergent oscillations exist.

    Notes
    -----
    This code reproduces the method used in `ADJUST` on page 15.

    This function checks if the current is the derivative is oscillating and
    checks that the average ratio of these derivatives is greater than 0.9.
    Thus if the derivative magnitude stays constant it will still count as
    diverging.
    """
    # Variable `DCHEK` from line 235.
    derivatives = all_found_currents[-9:] - all_found_currents[-10:-1]
    # Check that the derivative changes sign between each iteration.
    if 1 in derivatives[:-1] * derivatives[1:]:
        # The derivative didn't change signs.
        return False
    # Variable `ZCHEK` from line 240.
    derivative_magnitudes = (np.abs(derivatives[:-1]) + np.abs(derivatives[1:])) / 2
    # Variable `SMCHEK` from line 262.
    derivative_ratio = np.average(
        derivative_magnitudes[1:] / derivative_magnitudes[:-1]
    )

    return derivative_ratio > 0.9


def estimate_num_iterations_decaying(all_found_currents, relative_accuracy):
    r"""
    Estimate how many iterations are needed to get within the relative accuracy
    when the currents are decaying.

    Parameters
    ----------
    all_found_currents : `numpy.ndarray`
        1D-array of all currents to the probe found so far from all iterations.
    relative_accuracy : `float`
        Relative accuracy to determine how many more iterations are needed.

    Returns
    -------
    is_converging : `bool`
        Whether a decaying exponential can be fit to the points.
    num_iterations : `float`
        Estimated number of iterations till convergence. `num_iterations` is
        `None` if `is_converging` is `False`. If the final value is estimated to
        be `0` then `np.inf` is returned.
    estimated_final_current : `float`
        Estimate for what the current is converging to. If `is_converging` is
        `False` then `estimated_final_current` is `None`.

    Notes
    -----
    This code is similar to line 138 of `ADJUST` on page 16. This works by
    taking an exponential fit to three points (assuming they converge) and
    calculating the number of iterations until the result is within the
    relative accuracy. There is an issue with the code used by Laframboise in
    the calculation of `FINS`. On the `FINS` calculation, `C10` should be
    replaced by `A10`.
    """
    # `y_0`, `y_1`, and `y_2` are `A10`, `B10`, and `C10` respectively.
    y_0 = all_found_currents[-21]
    y_1 = all_found_currents[-11]
    y_2 = all_found_currents[-1]

    old_derivative = y_1 - y_0
    new_derivative = y_2 - y_1

    if old_derivative == 0 or not (0 < new_derivative / old_derivative < 1):
        return False, None, None

    # Fit the data to a function of the form `y = a e^(-b x) + c`.
    a = (y_0 - y_1) ** 2 / (y_0 - 2 * y_1 + y_2)
    b = 0.1 * np.log((y_0 - y_1) / (y_1 - y_2))
    c = (-(y_1**2) + y_0 * y_2) / (y_0 - 2 * y_1 + y_2)

    if c == 0:
        return True, np.inf, c

    num_iterations = np.log(np.abs(a / (relative_accuracy * c))) / b - 21

    return True, num_iterations, c


def estimate_num_iterations_oscillating(
    all_found_currents, relative_accuracy, estimated_final_current
):
    r"""
    Estimate how many iterations are needed to get within the relative accuracy.

    Determine if convergence is oscillating and estimate how many more
    iterations must be done.

    Parameters
    ----------
    all_found_currents : `numpy.ndarray`
        1D-array of all currents to the probe found so far from all iterations.
    relative_accuracy : `float`
        Relative accuracy to determine how many more iterations are needed.
    estimated_final_current : `float`
        Estimate for what the current is converging to.

    Returns
    -------
    is_oscillating : `bool`
        Whether the found currents are oscillating.
    converges_fast : `bool`
        Returns `True` if oscillations are small enough to be in error tolerance
        by 40 iterations.

    Notes
    -----
    This is the same code from `ADJUST` page 17 where oscillation decay rate is calculated.

    A function of the form

    .. math::
        I(j) = (-1)^j A e^{-B j} + C, \quad B > 0

    is assumed to match the oscillations. :math:`j` is the iteration index and
    :math:`N` is the most recent index. :math:`C` is the converged solution that
    we are approaching. We can then calculate two new terms,

    .. math::
        \alpha &= \frac{\lvert I(N - 20) + I(N - 21) \rvert + \lvert I(N - 21) + I(N - 22) \rvert}{2} \\
        &= \frac{\lvert A e^{-B (N - 20)} - A e^{-B (N - 21)} \rvert + \lvert A e^{-B (N - 21)} - A e^{-B (N - 22)} \rvert}{2} \\
        &= \lvert A \rvert e^{-BN} \frac{e^{20B} - e^{21B} + e^{21} - e^{22}}{2} \\
        &= \lvert A \rvert e^{-BN} \frac{e^{20B} - e^{22B}}{2} \\
        &= \lvert A \rvert e^{-B(N - 21)} \frac{e^{B} - e^{-B}}{2} \\
        \beta &= \frac{\lvert I(N - 1) + I(N - 2) \rvert + \lvert I(N) + I(N - 1) \rvert}{2} \\
        &= \lvert A \rvert e^{-B(N - 1)} \frac{e^{B} - e^{-B}}{2} \\

    Our solution should have oscillations less than :math:`a C` where :math:`a`
    is the `relative_accuracy`. We only want to know if oscillations will be
    smaller than that within 40 iterations and we can check that by calculating

    .. math::
        \gamma &= \beta \left( \frac{\beta}{\alpha} \right)^2 \\
        &= \lvert A \rvert e^{-B(N - 1)} \frac{e^{B} - e^{-B}}{2} \left( e^{-20B} \right)^2 \\
        &= \lvert A \rvert e^{-B(N + 38)} \frac{1 - e^{-2B}}{2} \\
        &< \lvert I(N + 38) \rvert

    If :math:`\gamma < a C` then oscillations should be small enough to stop
    convergence in 40 iterations and the returned value, `converges_fast`, is
    the result.
    """
    # Check that oscillations exist for the past 30 iterations. Only check every 10 iterations.
    derivatives = all_found_currents[-30::] - all_found_currents[-31:-1]
    if np.any(np.sign(derivatives[8::10]) * np.sign(derivatives[9::10]) != -1):
        # Oscillations don't persist.
        return False, False

    alpha = (np.abs(derivatives[8]) + np.abs(derivatives[9])) / 2
    beta = (np.abs(derivatives[-2]) + np.abs(derivatives[-1])) / 2

    gamma = beta * (beta / alpha) ** 2

    return True, (gamma <= relative_accuracy * estimated_final_current)


def determine_convergence_zero_temperature_repelled_particles(
    all_found_currents, relative_accuracy
):
    r"""
    Determine whether the solution has converged within the required accuracy and the error between iterations is small.

    Parameters
    ----------
    all_found_currents : `numpy.ndarray`
        1D-array of all currents to the probe found so far from all iterations.
    relative_accuracy : `float`
        Relative accuracy to determine how many more iterations are needed.

    Returns
    -------
    converged : `bool`
        Whether the solution has converged.

    Notes
    -----
    This follows the code in `ADJUST` on page 17.

    This function checks if the current for the last 10 iterations is within the relative accuracy.

    .. math::
        I_{\text{max}} - I_{\text{min}} < \text{relative_accuracy} \times \frac{I_{\text{max}} + I_{\text{min}}}{2}
    """
    # Lines 281, 282, and 283.
    min_current = np.min(all_found_currents[-10:])
    max_current = np.max(all_found_currents[-10:])
    return (max_current - min_current) < relative_accuracy * abs(
        max_current + min_current
    ) / 2


def determine_convergence(
    all_found_currents,
    relative_accuracy,
    estimated_final_current,
    new_net_charge_density,
    old_net_charge_density,
    x,
):
    r"""
    Determine whether the solution has converged within the required accuracy and the error between iterations is small.

    Parameters
    ----------
    all_found_currents : `numpy.ndarray`
        1D-array of all currents to the probe found so far from all iterations.
    relative_accuracy : `float`
        Relative accuracy to determine how many more iterations are needed.
    estimated_final_current : `float`
        Estimate for what the current is converging to.
    new_net_charge_density, old_net_charge_density : `numpy.ndarray`
        Arrays of charge density for the prior and current iteration.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    zero_temperature_repelled_particles : `bool`, default `False`
        Whether repelled particles have zero temperature.

    Returns
    -------
    converged : `bool`
        Whether the solution has converged.

    Notes
    -----
    This follows the code in `ADJUST` on page 17.

    The code first checks if the relative error between the two most recent
    currents and the `estimated_final_current` is in the relative accuracy.

    If the relative error is within the relative accuracy then the code checks
    if the average squared relative error between the two most recent charge
    densities is less than 0.01. It calculates the average squared relative
    error using the trapezoidal rule and the following equations:

    .. math::
        Y = \left(2 \frac{\rho_{\text{new}} - \rho_{\text{old}}}{\rho_{\text{new}} + \rho_{\text{old}}} \right)^2
        \frac{1}{R - 1} \int_1^R Y \, \text{d}r \approx \frac{1}{R - 1} \sum_{i=1}^{N - 1} (Y_{i + 1} + Y_i) \frac{r_{i + 1} - r_i}{2}
    """
    # This is `IJJ` on line 275.
    converged = True

    # Line 140, 144.
    for index in (-1, -2):
        if (
            abs(all_found_currents[index] - estimated_final_current)
            - abs(relative_accuracy * estimated_final_current)
            > 0
        ):
            return False

    # This is `Y` on line 266.
    squared_relative_charge_error = (
        2
        * (new_net_charge_density - old_net_charge_density)
        / (new_net_charge_density + old_net_charge_density)
    ) ** 2

    r = 1 / x
    # Integrate using the trapezoidal rule to find `TOT` in the code on line 267.
    integrated_error = np.sum(
        (squared_relative_charge_error[:-1] + squared_relative_charge_error[1:])
        * (r[1:] - r[:-1])
        / 2
    )
    # `AVGE` in code.
    average_error = integrated_error / (r[-1] - 1)

    if average_error > 0.01:
        converged = False

    return converged


def check_accuracy_for_zero_temperature_repelled_particles(
    coefficient_1,
    coefficient_2,
    all_found_currents,
    num_checks_since_coefficient_change,
    relative_accuracy,
    coefficients_previously_decreased,
):
    """Check if the current to the probe is sufficiently accurate for zero temperature repelled particles.

    Notes
    -----
    This is lines 281, 282, and 283 of the `ADJUST` code on page 16.
    """
    converged = determine_convergence_zero_temperature_repelled_particles(
        all_found_currents,
        relative_accuracy,
    )
    if not converged:
        logging.info("Current to the probe is not sufficiently accurate.")
        return (
            coefficient_1,
            coefficient_2,
            False,
            num_checks_since_coefficient_change + 1,
            coefficients_previously_decreased,
        )
    else:
        estimated_final_current = (all_found_currents[-1] + all_found_currents[-2]) / 2
        logging.info(
            "Current to the probe is sufficiently accurate. Ending iterations. Estimated final current: %f",
            estimated_final_current,
        )
        return (
            coefficient_1,
            coefficient_2,
            True,
            num_checks_since_coefficient_change + 1,
            coefficients_previously_decreased,
        )


def determine_coefficients(  # noqa: C901 PLR0911 PLR0912
    coefficient_1,
    coefficient_2,
    all_found_currents,
    control_level,
    num_checks_since_coefficient_change,
    relative_accuracy,
    new_net_charge_density,
    old_net_charge_density,
    x,
    zero_temperature_repelled_particles=False,
    coefficients_previously_decreased=False,
):
    r"""The top level program that controls how much mixing is done between iterations.

    Parameters
    ----------
    coefficient_1, coefficient_2 : `float`
        Coefficients used for calculating the mixing factors.
    control_level : int
        How much control this program has over adjusting the mixing level.
            - If `control_level = 1` then `determine_coefficients` takes no action.
            - If `control_level = 2` then `determine_coefficients` damps
            divergent oscillations.
            - If `control_level = 3` then `determine_coefficients` also ends
            execution when accuracy of results is sufficient.
            - If `control_level = 4` then `determine_coefficients` also corrects
            for slow oscillation and convergence.
    all_found_currents : `numpy.ndarray`
        1D-array of all currents to the probe found so far from all iterations.
    num_checks_since_coefficient_change : `int`
        The number of iterations // 10 since the last coefficient change.
    relative_accuracy : `float`
        Relative accuracy to determine how many more iterations are needed.
    new_net_charge_density, old_net_charge_density : `numpy.ndarray`
        Arrays of normalized charge density for the prior and current iteration.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    zero_temperature_repelled_particles : `bool`, default `False`
        Whether repelled particles have zero temperature.
    coefficients_previously_decreased : `bool`, default `False`
        Whether the coefficients were previously decreased.

    Returns
    -------
    new_coefficient_1, new_coefficient_2 : `float`
        New coefficients to use for calculating the mixing factors.
    end_execution : `bool`
        Whether to stop iterating for finding solutions.
    updated_num_checks_since_coefficient_change : `int`
    coefficients_previously_decreased : `bool`

    Notes
    -----
    This runs very similar to the `ADJUST` code of 15 of the thesis.
    The size of `all_found_currents` is `KEND`, `coefficient_1` is `QT1`, `coefficient_2` is
    `QT2`, `control_level` is `KWIT`, `num_no_oscillation_checks` is `MEW`, and `coefficients_previously_decreased` is `MAD`.
    """
    if control_level == 1:
        return (
            coefficient_1,
            coefficient_2,
            False,
            num_checks_since_coefficient_change,
            coefficients_previously_decreased,
        )

    iteration_number = all_found_currents.size
    if iteration_number % 10 != 0:
        return (
            coefficient_1,
            coefficient_2,
            False,
            num_checks_since_coefficient_change,
            coefficients_previously_decreased,
        )

    divergent_oscillations_exist = check_oscillation_divergence(all_found_currents)

    # Decrease the mixing coefficients if oscillations are found.
    if divergent_oscillations_exist:
        logging.info("Divergent oscillations exist. Decreasing mixing coefficients.")
        new_coefficient_1, new_coefficient_2 = decrease_mixing_coefficients(
            coefficient_1, coefficient_2
        )
        return new_coefficient_1, new_coefficient_2, False, 0, True

    # Line 123.
    if not divergent_oscillations_exist and control_level <= 2:
        logging.info(
            "Not checking for convergence rate since `control_level` is less than 3 and oscillations are not diverging."
        )
        return (
            coefficient_1,
            coefficient_2,
            False,
            num_checks_since_coefficient_change + 1,
            coefficients_previously_decreased,
        )

    # Line 132.
    if zero_temperature_repelled_particles:
        return check_accuracy_for_zero_temperature_repelled_particles(
            coefficient_1,
            coefficient_2,
            all_found_currents,
            num_checks_since_coefficient_change,
            relative_accuracy,
            coefficients_previously_decreased,
        )
    # Line 280.
    elif num_checks_since_coefficient_change < 2:
        logging.info(
            "Not checking for slow oscillation damping since `num_checks_since_coefficient_change` is less than 2."
        )
        return (
            coefficient_1,
            coefficient_2,
            False,
            num_checks_since_coefficient_change + 1,
            coefficients_previously_decreased,
        )

    (
        converging,
        num_iterations,
        estimated_final_current,
    ) = estimate_num_iterations_decaying(all_found_currents, relative_accuracy)
    if not converging:
        logging.info("Current to probe is not converging.")
        return (
            coefficient_1,
            coefficient_2,
            False,
            num_checks_since_coefficient_change + 1,
            coefficients_previously_decreased,
        )
    elif num_iterations > 40:
        # Lines 141 and 28.
        if control_level <= 3 or coefficients_previously_decreased:
            logging.info(
                "Convergence is too slow (iterations remaining = %.0f). Not increasing coefficients because either the `control_level` is not high enough or the coefficients were previously decreased.",
                num_iterations,
            )
            return (
                coefficient_1,
                coefficient_2,
                False,
                num_checks_since_coefficient_change + 1,
                coefficients_previously_decreased,
            )
        else:
            logging.info(
                "Convergence is too slow (if coefficient left unchanged, iterations remaining = %.0f). Increasing mixing coefficients.",
                num_iterations,
            )
            new_coefficient_1, new_coefficient_2 = increase_mixing_coefficients(
                coefficient_1, coefficient_2
            )
            return (
                new_coefficient_1,
                new_coefficient_2,
                False,
                0,
                coefficients_previously_decreased,
            )

    has_converged = determine_convergence(
        all_found_currents,
        relative_accuracy,
        estimated_final_current,
        new_net_charge_density,
        old_net_charge_density,
        x,
    )

    if has_converged:
        logging.info("Current to the probe is sufficiently accurate.")
        return (
            coefficient_1,
            coefficient_2,
            True,
            num_checks_since_coefficient_change + 1,
            coefficients_previously_decreased,
        )
    elif control_level <= 3:
        logging.info(
            "Not checking for slow oscillation damping since `control_level` is less than 4 (control_level = %d). Expected number of iterations remaining = %.0f.",
            control_level,
            num_iterations,
        )
        return (
            coefficient_1,
            coefficient_2,
            False,
            num_checks_since_coefficient_change + 1,
            coefficients_previously_decreased,
        )

    is_oscillating, converges_fast = estimate_num_iterations_oscillating(
        all_found_currents, relative_accuracy, estimated_final_current
    )

    if is_oscillating and not converges_fast:
        logging.info(
            "Oscillations are damping too slowly (iterations remaining = %.0f). Decreasing mixing coefficients.",
            num_iterations,
        )
        new_coefficient_1, new_coefficient_2 = decrease_mixing_coefficients(
            coefficient_1, coefficient_2
        )
        return new_coefficient_1, new_coefficient_2, False, 0, True
    else:
        logging.info(
            "Oscillations are damping quickly and convergence is in approximately %.0f iterations.",
            num_iterations,
        )
        return (
            coefficient_1,
            coefficient_2,
            False,
            num_checks_since_coefficient_change + 1,
            coefficients_previously_decreased,
        )
