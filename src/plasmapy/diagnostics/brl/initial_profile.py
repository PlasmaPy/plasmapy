"""Calculate the initial potential and net charge density profiles to start iterations from."""

import numpy as np

from plasmapy.diagnostics.brl.normalizations import (
    renormalize_probe_radius_to_larger_debye_length,
)


def evaluate_polynomial(
    coefficients,
    normalized_probe_potential,
    normalized_probe_radius,
    x_points,
    dx_ds_points,
    spherical=True,
):
    r"""Create an initial profile using a polynomial in `x`.

    Parameters
    ----------
    coefficients : `numpy.ndarray`
        An array of coefficients for a polynomial in increasing order.
    normalized_probe_potential : `float`
        The normalized probe potential as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    normalized_probe_radius : `float`
        The radius of the probe normalized to the attracted particle Debye length as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_probe_radius`.
    x_points : `numpy.ndarray`
        The normalized inverse distance from the origin.
    dx_ds_points : `numpy.ndarray`
        The derivative of `x_points` with respect to `s`.
    spherical : `bool`, optional
        Whether the probe is spherical or cylindrical. Default is `True`.

    Returns
    -------
    z : `numpy.ndarray`
        This is used to calculate the potential.
    dchi_ds, eta : `numpy.ndarray`
        The derivative of the potential and the net charge density at each point.

    Notes
    -----
    `coefficients` is the same as `A` in the thesis. This follows the code on
    page 25 of the code in the thesis.

    This operates by computing a polynomial for :math:`z(s)` of order from 2 to
    8 (inclusive) if `spherical` or from 1 to 7 (inclusive) in
    :math:`x = 1 / r`. Let the minimum and maximum order be given by :math:`p_i`
    and :math:`p_f`. The `coefficients`, :math:`c_i,\, i \in \{1, 2, \dots, 8\}`,
    are normalized by their sum divided by the normalized probe potential,

    .. math::

        N &= \sum_{i=p_i}^{p_f} \frac{c_i}{\chi_p} \\
        \overline{c}_i &= \frac{c_i}{N}.

    Then we can calculate :math:`z(s)`, :math:`\frac{d\chi}{ds}`, and :math:`\eta` as

    .. math::

        z(s) &= \sum_{i=p_i}^{p_f} \overline{c}_i x(s)^i, \\
        \frac{d\chi}{ds} &= \sum_{i=p_i}^{p_f} i \overline{c}_i x(s)^{i-1} \frac{dx}{ds}, \\
        \eta &= \frac{-1}{R_p^2} \sum_{i=p_i}^{p_f} a_i i \overline{c}_i x(s)^{i + 2}.

    :math:`a_i` is :math:`i - 1` if `spherical` and :math:`i` if not.
    """
    min_order = 2 if spherical else 1  # `JAA` in the thesis.
    max_order = 8 if spherical else 7  # `JBB` in the thesis.
    # Line 24. `normalization_inverse` is `ERROR` in the thesis.
    normalization_inverse = normalized_probe_potential / np.sum(
        coefficients[min_order - 1 : max_order], dtype=float
    )
    # Line 27.
    new_coefficients = np.array(coefficients, dtype=float, copy=True)
    new_coefficients[min_order - 1 : max_order] *= normalization_inverse

    z = np.zeros_like(x_points)
    dchi_ds = np.zeros_like(x_points)
    eta = np.zeros_like(x_points)
    for power in range(min_order, max_order + 1):
        z += new_coefficients[power - 1] * x_points**power
        dchi_ds += (
            power * new_coefficients[power - 1] * x_points ** (power - 1) * dx_ds_points
        )
        eta -= (
            power
            * (power - spherical)
            * new_coefficients[power - 1]
            * x_points ** (power + 2)
            / normalized_probe_radius**2
        )

    return z, dchi_ds, eta


def evaluate_debye_and_power_law(
    power_law_coefficient,
    normalized_probe_potential,
    normalized_probe_radius,
    effective_attracted_to_repelled_temperature,
    x_points,
    dx_ds_points,
    spherical=True,
):
    r"""Create an initial profile using a Debye term and a power law term in `x`.

    Parameters
    ----------
    power_law_coefficient : `float`
        The relative importance of the power law term compared to the Debye
        term. Must be between 0 and 1.
    normalized_probe_potential : `float`
        The normalized probe potential as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    normalized_probe_radius : `float`
        The radius of the probe normalized to the attracted particle Debye length as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_probe_radius`.
    effective_attracted_to_repelled_temperature : `float`
        The effective temperature ratio of the attracted to repelled particle
        species as defined in `~plasmapy.diagnostics.brl.normalizations.get_effective_temperature_ratio`.
    x_points : `numpy.ndarray`
        The normalized inverse distance from the origin.
    dx_ds_points : `numpy.ndarray`
        The derivative of `x_points` with respect to `s`.
    spherical : `bool`, optional
        Whether the probe is spherical or cylindrical. Default is `True`.

    Returns
    -------
    z : `numpy.ndarray`
        This is used to calculate the potential.
    dchi_ds, eta : `numpy.ndarray`
        The derivative of the potential and the net charge density at each point.

    """
    if power_law_coefficient < 0 or power_law_coefficient > 1:
        raise ValueError("The power law coefficient must be between 0 and 1.")

    renormalized_probe_radius = renormalize_probe_radius_to_larger_debye_length(
        normalized_probe_radius, effective_attracted_to_repelled_temperature
    )
    debye_coefficient = 1 - power_law_coefficient

    normalized_power_law_coefficient = (
        power_law_coefficient * normalized_probe_potential
    )
    normalized_debye_coefficient = debye_coefficient * normalized_probe_potential

    debye_term = np.exp(renormalized_probe_radius * (1 - 1 / x_points))
    if spherical:
        z = (
            normalized_debye_coefficient * x_points * debye_term
            + normalized_power_law_coefficient * x_points**2
        )
        dchi_ds = (
            normalized_debye_coefficient
            * debye_term
            * (1 + renormalized_probe_radius / x_points)
            + 2 * normalized_power_law_coefficient * x_points
        ) * dx_ds_points
        eta = (
            -normalized_debye_coefficient
            * debye_term
            * x_points
            * min(effective_attracted_to_repelled_temperature, 1)
            - 2
            * normalized_power_law_coefficient
            * x_points**4
            / normalized_probe_radius**2
        )
    else:
        z = (
            normalized_debye_coefficient * debye_term * x_points**0.5
            + normalized_power_law_coefficient * x_points
        )
        dchi_ds = (
            normalized_debye_coefficient
            * debye_term
            * (
                0.5 / x_points**0.5
                + x_points**0.5 * renormalized_probe_radius / x_points**2
            )
            + normalized_power_law_coefficient
        ) * dx_ds_points
        eta = (
            -(
                normalized_debye_coefficient
                * debye_term
                * x_points**0.5
                * (x_points**2 / 4 + renormalized_probe_radius**2)
                + normalized_power_law_coefficient * x_points**3
            )
            / normalized_probe_radius**2
        )

    return z, dchi_ds, eta


def enforce_boundary_condition(
    z, dchi_ds, x_points, dx_ds_points, spherical=True, zero_potential_boundary=False
):
    r"""Calculate the normalized potential by enforcing the required boundary condition.

    At the outer edge of the computational net we want to enforce some boundary
    condition on :math:`\chi`. According to page 34 of the thesis, the
    potential for a current collecting probe should asymptote to some power of
    :math:`r`. Another option is to enforce the potential to be zero at the
    outer edge.

    Parameters
    ----------
    z : `numpy.ndarray`
        The base array to manipulate to get the final profile.
    dchi_ds : `numpy.ndarray`
        The derivative of the potential.
    x_points : `numpy.ndarray`
        The normalized inverse distance from the origin.
    dx_ds_points : `numpy.ndarray`
        The derivative of `x_points` with respect to `s`.
    spherical : `bool`, optional
        Whether the probe is spherical or cylindrical. Default is `True`.
    zero_potential_boundary : `bool`, optional
        Whether the potential should be zero at the outer edge. Default is `False`.

    Returns
    -------
    chi, dchi_dx : `numpy.ndarray`
        The final potential profile and it's derivative.

    Notes
    -----
    This follows the code on page 6.

    The `zero_potential_boundary` option is `True` when repelled particles are
    at zero temperature.
    """
    if spherical:
        # Line 37.
        if zero_potential_boundary:
            correction = z[-1] / (1 - x_points[-1])
        else:
            correction = (2 * z[-1] - x_points[-1] * dchi_ds[-1] / dx_ds_points[-1]) / (
                2 - x_points[-1]
            )

        # Line 327.
        dchi_ds += correction * dx_ds_points
        chi = z + correction * (x_points - 1)

    else:
        # Line 283.
        if zero_potential_boundary:
            correction = -z[-1] / np.log(x_points[-1])
        else:
            correction = (z[-1] - x_points[-1] * dchi_ds[-1] / dx_ds_points[-1]) / (
                1 - np.log(x_points[-1])
            )

        # Line 332.
        dchi_ds += correction * dx_ds_points / x_points
        chi = z + correction * np.log(x_points)

    return chi, dchi_ds
