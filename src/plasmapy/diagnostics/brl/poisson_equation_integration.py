"""Evaluation and integration of Poisson's equation."""

import numpy as np


def evaluate_K0_term(eta_net, normalized_probe_radius, x, spherical=True):
    r"""Second derivative of normalized potential in terms of `x`.

    Parameters
    ----------
    eta_net : `numpy.ndarray`
        Normalized net charge density from equation (9.1).
    normalized_probe_radius : `float`
        The radius of the probe normalized to the attracted particle Debye length as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_probe_radius`.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    `numpy.ndarray`

    Notes
    -----
    This is the term :math:`K_0(s)` as defined in equations (D.1) and (D.12) for the spherical and cylindrical probes respectively. The charge density, `eta_net`, and inverse radius, `x`, should have the same shape and correspond to the same points in space.
    """
    if spherical:
        return -(normalized_probe_radius**2) * eta_net / x**4
    else:
        return -(normalized_probe_radius**2) * eta_net / x**3


def evaluate_K1_term(K0_term, x, dx_ds, integration_matrix, spherical=True):
    r"""The :math:`s` dependent term of :math:`d\chi / ds`.

    Parameters
    ----------
    K0_term : `numpy.ndarray`
        The :math:`K_0(s)` term calculated from `~plasmapy.diagnostics.brl.poisson_equation_integration.evaluate_K0_term`.
    x, dx_ds : `numpy.ndarray`
        The the normalized inverse radius and it's `s` derivative calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    integration_matrix : `numpy.ndarray`
        The matrix used for integrating functions from `~plasmapy.diagnostics.brl.integration.construct_integration_matrix`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    `numpy.ndarray`

    Notes
    -----
    This is the term :math:`K_1(s)` as defined in equations (D.4) and (D.14) for the spherical and cylindrical probes respectively.
    """
    if spherical:
        return dx_ds * (integration_matrix @ (K0_term * dx_ds))
    else:
        return (dx_ds / x) * (integration_matrix @ (K0_term * dx_ds))


def evaluate_K2_term(K1_term, normalized_probe_potential, integration_matrix):
    r"""The integral of :math:`K_1(s)` with the addition of the probe potential.

    Parameters
    ----------
    K1_term : `numpy.ndarray`
        The :math:`K_1(s)` term calculated from `~plasmapy.diagnostics.brl.poisson_equation_integration.evaluate_K1_term`.
    normalized_probe_potential : `float`
        The normalized potential of the probe as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    integration_matrix : `numpy.ndarray`
        The matrix used for integrating functions from `~plasmapy.diagnostics.brl.integration.construct_integration_matrix`.

    Returns
    -------
    `numpy.ndarray`

    Notes
    -----
    This is the term :math:`K_2(s)` as defined in equation (D.6). This is the same for both the spherical and cylindrical probes.
    """
    return normalized_probe_potential + integration_matrix @ K1_term


def dchi_dx_at_probe(
    K1_term, K2_term, x, dx_ds, spherical=True, zero_T_repelled_particles=False
):
    r"""The derivative of the normalized potential in terms of :math:`x` at the probe.

    Parameters
    ----------
    K1_term, K2_term : `numpy.ndarray`
        The :math:`K_1(s)` and :math:`K_2(s)` terms calculated from `~plasmapy.diagnostics.brl.poisson_equation_integration.evaluate_K1_term` and `~plasmapy.diagnostics.brl.poisson_equation_integration.evaluate_K2_term` respectively.
    x, dx_ds : `numpy.ndarray`
        The the normalized inverse radius and it's `s` derivative calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.
    zero_T_repelled_particles : `bool`, optional
        If `True` then the repelled particles have zero temperature which requires a slightly modified calculation. Default is `False`.

    Returns
    -------
    `float`

    Notes
    -----
    This is the term :math:`(d\chi / dx)_{s=0}` as defined in equations (D.10), (D.11), (D.18), and (D.19).
    """
    if zero_T_repelled_particles:
        if spherical:
            # Equation (D.11).
            return K2_term[-1] / (1 - x[-1])
        else:
            # Equation (D.19).
            return -K2_term[-1] / np.log(x[-1])
    else:  # noqa: PLR5501
        if spherical:
            # Equation (D.10).
            return (2 * K2_term[-1] - x[-1] * K1_term[-1] * dx_ds[-1]) / (2 - x[-1])
        else:
            # Equation (D.18).
            return (K2_term[-1] - x[-1] * K1_term[-1] / dx_ds[-1]) / (1 - np.log(x[-1]))


def chi_and_dchi_ds(
    eta_net,
    normalized_probe_radius,
    normalized_probe_potential,
    x,
    dx_ds,
    integration_matrix,
    spherical=True,
    zero_T_repelled_particles=False,
):
    r"""The normalized potential, :math:`\chi`, and it's :math:`s` derivative.

    Integrate the charge density using Poisson's equation to get the normalized potential and it's derivative.

    Parameters
    ----------
    eta_net : `numpy.ndarray`
        Normalized net charge density from equation (9.1).
    normalized_probe_radius : `float`
        The radius of the probe normalized to the attracted particle Debye length as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_probe_radius`.
    normalized_probe_potential : `float`
        The normalized potential of the probe as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    x, dx_ds : `numpy.ndarray`
        The the normalized inverse radius and it's `s` derivative calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    integration_matrix : `numpy.ndarray`
        The matrix used for integrating functions from `~plasmapy.diagnostics.brl.integration.construct_integration_matrix`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.
    zero_T_repelled_particles : `bool`, optional
        If `True` then the repelled particles have zero temperature which requires a slightly modified calculation. Default is `False`.

    Returns
    -------
    `chi`, `dchi_ds` : `numpy.ndarray`

    Notes
    -----
    These are the terms :math:`\chi(s)` and :math:`(d\chi / ds)(s)` from equations (D.7) and (D.8) (spherical probe) or (D.15) and (D.16) (cylindrical probe).
    """
    K0_term = evaluate_K0_term(eta_net, normalized_probe_radius, x, spherical)
    K1_term = evaluate_K1_term(K0_term, x, dx_ds, integration_matrix, spherical)
    K2_term = evaluate_K2_term(K1_term, normalized_probe_potential, integration_matrix)
    dchi_dx_0 = dchi_dx_at_probe(
        K1_term, K2_term, x, dx_ds, spherical, zero_T_repelled_particles
    )

    if spherical:
        # Equation (D.7)
        chi = dchi_dx_0 * (x - 1) + K2_term
        # Equation (D.8)
        dchi_dx = dchi_dx_0 + K1_term / dx_ds
    else:
        # Equation (D.15)
        chi = dchi_dx_0 * np.log(x) + K2_term
        # Equation (D.16)
        dchi_dx = dchi_dx_0 / x + K1_term / dx_ds

    dchi_ds = dchi_dx * dx_ds

    return chi, dchi_ds
