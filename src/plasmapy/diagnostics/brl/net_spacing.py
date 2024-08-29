"""The computational net spacing and it's inverse + derivative."""

import numpy as np

from plasmapy.diagnostics.brl.normalizations import (
    renormalize_probe_radius_to_larger_debye_length,
)


def get_s_points(num_points, s_end_point):
    r"""Get the equally spaced points for the variable `s`."""
    if s_end_point <= 0:
        raise ValueError(
            f"The `s_end_point` must be greater than or equal to 0 but was {s_end_point}."
        )
    return np.linspace(0, s_end_point, num_points)


def _large_probe_x_and_dx_ds(s_points, renormalized_probe_radius):
    r"""The values of `x` and `dx/ds` that correspond to the `s_points` for a probe of large radius.

    Parameters
    ----------
    s_points : `numpy.ndarray`
    renormalized_probe_radius : `float`
        The radio of probe radius to the larger Debye length.

    Returns
    -------
    x_points, dx_ds_points : `numpy.ndarray`
        Value at each :math:`s` of :math:`x(s)` and :math:`\frac{dx}{ds}(s)`.

    Notes
    -----
    Laframboise gives no explanation as to why this function is chosen. It can
    be found in appendix I: computer program listing, line 254 (page 3). This
    returns a function of the form
    .. math::

       x(s) = A (1 - s) - B (1 - s)^C

    where :math:`A`, :math:`B`, and :math:`C` are constants that depend on
    the `normalized_probe_radius`.
    """
    shared_constant = 0.5  # SNT in Laframboise
    linear_coefficient = 1 / (1 - shared_constant)  # ZNP in Laframboise
    power_coefficient = linear_coefficient - 1  # AYA in Laframboise
    power = (
        linear_coefficient - 10 / (shared_constant * renormalized_probe_radius)
    ) / power_coefficient

    x_points = (
        linear_coefficient * (1 - s_points)
        - power_coefficient * (1 - s_points) ** power
    )
    dx_ds_points = linear_coefficient * -1 + power_coefficient * power * (
        1 - s_points
    ) ** (power - 1)
    return x_points, dx_ds_points


def _medium_probe_x_and_dx_ds(s_points):
    r"""The values of `x` and `dx/ds` that correspond to the `s_points` for a probe of medium radius.

    Parameters
    ----------
    s_points : `numpy.ndarray`

    Returns
    -------
    x_points, dx_ds_points : `numpy.ndarray`
        Value at each :math:`s` of :math:`x(s)` and :math:`\frac{dx}{ds}(s)`.

    Notes
    -----
    Laframboise gives no explanation as to why this function is chosen. It can
    be found in appendix I: computer program listing, line 361 (page 3). This
    returns a function of the form
    .. math::

       x(s) = 1 - s.
    """
    x_points = 1 - s_points
    dx_ds_points = -np.ones_like(s_points)
    return x_points, dx_ds_points


def _small_probe_x_and_dx_ds(s_points):
    r"""The values of `x` and `dx/ds` that correspond to the `s_points` for a probe of small radius.

    Parameters
    ----------
    s_points : `numpy.ndarray`

    Returns
    -------
    x_points, dx_ds_points : `numpy.ndarray`
        Value at each :math:`s` of :math:`x(s)` and :math:`\frac{dx}{ds}(s)`.

    Notes
    -----
    Laframboise gives no explanation as to why this function is chosen. It can
    be found in appendix I: computer program listing, line 360 (page 3). This
    returns a function of the form
    .. math::

       x(s) = e^{-s}
    """
    x_points = np.exp(-s_points)
    dx_ds_points = -np.exp(-s_points)
    return x_points, dx_ds_points


def _zero_T_repelled_x_and_dx_ds(
    s_points, normalized_probe_radius, normalized_probe_potential
):
    r"""The values of `x` and `dx/ds` that correspond to the `s_points` for zero temperature repelled particles.

    Parameters
    ----------
    s_points : `numpy.ndarray`
    normalized_probe_radius : `float`
        The ratio of probe radius to attracted particle debye length,
        :math:`R_p / \lambda_D_+`, as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_probe_radius`.
    normalized_probe_potential : `float`
        The probe potential normalized to the attracted particles temperature.

    Returns
    -------
    x_points, dx_ds_points : `numpy.ndarray`
        Value at each :math:`s` of :math:`x(s)` and :math:`\frac{dx}{ds}(s)`.

    Notes
    -----
    Laframboise gives no explanation as to why this function is chosen. It can
    be found in appendix I: computer program listing, line 400 (page 10) and
    line 253 (page 11). This returns a function of the form
    .. math::

       x(s) = (A + B (1 - s) + C (1 - s)^2)^{-1}

    where :math:`A`, :math:`B`, and :math:`C` are constants that depend on
    the `normalized_probe_radius` and `normalized_probe_potential`.
    """
    # TODO: Figure out the geometric or physical meaning of this constant.
    const = (
        1 + 1.121 * normalized_probe_potential**0.75 / normalized_probe_radius
    )  # SHEDGE in Laframboise
    x_points = (
        const
        - (const - 1) * (0.125 * (1 - s_points) + (1 - 0.125) * (1 - s_points) ** 2)
    ) ** -1
    dx_ds_points = (
        -(x_points**2) * (const - 1) * (0.125 + 2 * (1 - s_points) * (1 - 0.125))
    )
    return x_points, dx_ds_points


def get_x_and_dx_ds(
    s_points,
    normalized_probe_radius,
    effective_attracted_to_repelled_temperature_ratio,
    normalized_probe_potential=None,
    zero_T_repelled_particles=False,
):
    r"""The values of `x` and `dx/ds` that correspond to the `s_points` for a probe of any radius.

    Parameters
    ----------
    s_points : `numpy.ndarray`
        Array of :math:`s` from `~plasmapy.diagnostics.brl.net_spacing.get_s_points`.
    normalized_probe_radius : `float`
        The ratio of probe radius to attracted particle debye length,
        :math:`R_p / \lambda_D_+`, as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_probe_radius`.
    effective_attracted_to_repelled_temperature_ratio : `float`
        :math:`-\frac{T_+ Z_-}{T_- Z_+}` as defined in `~plasmapy.diagnostics.brl.normalizations.get_effective_temperature_ratio`.
    normalized_probe_potential : `float`
        The probe potential normalized to the attracted particles temperature as
        defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
        This is only needed if the repelled particles have zero temperature.
        Default is `None`.
    zero_T_repelled_particles : `bool`
        Whether the repelled particles have zero temperature. Default is `False`.

    Returns
    -------
    x_points, dx_ds_points : `numpy.ndarray`
        Value at each :math:`s` of :math:`x(s)` and :math:`\frac{dx}{ds}(s)`.

    Notes
    -----
    This follows the code on page 3 of the thesis. Laframboise gives no
    explanation as to why these functions are chosen nor the boundary values on
    small, medium, and large probes. Also, :math:`x = 1 / r`.
    """
    # Normalized the probe radius to the larger Debye length.
    renormalized_probe_radius = renormalize_probe_radius_to_larger_debye_length(
        normalized_probe_radius, effective_attracted_to_repelled_temperature_ratio
    )

    if (renormalized_probe_radius > 2.6 or zero_T_repelled_particles) and np.max(
        s_points
    ) > 1:
        raise ValueError(
            "Maximum value of `s` must be less than or equal to 1 for all probes that aren't small."
        )

    if renormalized_probe_radius <= 2.6:
        return _small_probe_x_and_dx_ds(s_points)

    if zero_T_repelled_particles:
        if normalized_probe_potential is None:
            raise ValueError(
                "The `normalized_probe_potential` must be a `float` to get the net spacing for zero temperature repelled particles. The passed potential was `None`."
            )
        return _zero_T_repelled_x_and_dx_ds(
            s_points, normalized_probe_radius, normalized_probe_potential
        )
    elif renormalized_probe_radius <= 25.1:
        return _medium_probe_x_and_dx_ds(s_points)
    else:
        return _large_probe_x_and_dx_ds(s_points, renormalized_probe_radius)
