"""Using the potential, calculate the charge density."""

import numpy as np
from scipy.integrate import quad_vec
from scipy.special import erfc, erfi

from plasmapy.diagnostics.brl.integration import integrate


def _g(x):
    r"""Calculate :math:`g(x)`.

    This is the function defined in (E.21) as
    .. math::

        g(x) = \frac{\sqrt{\pi}}{2} e^{x^2} \erfc(x)

    Parameters
    ----------
    x : `numpy.ndarray`

    Returns
    -------
    `numpy.ndarray`
    """
    return np.pi**0.5 / 2 * np.exp(x**2) * erfc(x)


def eta_1(A, chi, spherical=True):
    r"""
    Calculate the contribution to the charge density along the :math:`J^2 = 0` line.

    This runs along the vertical :math:`E` (or :math:`\beta`) axis. `A` denotes
    where along that vertical axis the integration starts.

    Parameters
    ----------
    A : `numpy.ndarray`
        Lower bound of the integral. Also, `A >= chi`.
    chi : `numpy.ndarray`
        The normalized potential as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    `numpy.ndarray`

    Notes
    -----
    This is the term :math:`\eta_1(A)` that appears in various formula in appendix E.
    """
    if spherical:
        # Equation (E.23).
        return -np.exp(-A) / np.pi**0.5 * ((A - chi) ** 0.5 + _g((A - chi) ** 0.5))
    else:
        # Equation (E.10).
        return 0


def eta_2(A, chi, chi_p, x, spherical=True):
    r"""
    Calculate the contribution to the charge density between the region where particles are absorbed by the probe and non-striking particles exist.

    This is the straight line given by the equation :math:`\beta = \chi_p + \Omega`.

    Parameters
    ----------
    A : `numpy.ndarray`
        Lower bound of the integral. Also, `A >= kappa` as defined in (E.3).
    chi : `numpy.ndarray`
        The normalized potential as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    chi_p : `float`
        The normalized potential of the probe as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    `numpy.ndarray`

    Notes
    -----
    This is the term :math:`\eta_2(A)` that appears in various formula in appendix E.
    """
    kappa = (chi - x**2 * chi_p) / (1 - x**2)
    if spherical:
        # Equation (E.25)
        return (
            -((1 - x**2) ** 0.5)
            * np.exp(-A)
            / np.pi**0.5
            * ((A - kappa) ** 0.5 + _g(A - kappa**0.5))
        )
    else:
        # TODO: Change this to use the approximations given by equations (E.45) - (E.65).

        # Variables defined in (E.45).
        tau = (kappa - chi_p) / 2
        mu = np.max(kappa, chi_p) - tau
        theta = chi - tau
        B = A - tau

        # Integrand from (E.59). (E.59) is a formula that is ready to integrate numerically.
        def integrand(omega):
            return 1 / ((-np.log(omega) - theta) * (np.log(omega) ** 2 - mu**2) ** 0.5)

        # Evaluate H1 from (E.59).
        H1 = (
            np.exp(-tau)
            * quad_vec(
                lambda omega: (
                    (-np.log(omega) - theta) * ((np.log(omega)) ** 2 - mu**2) ** 0.5
                )
                ** -1,
                0,
                np.exp(-B),
            )[0]
        )

        # Equation (E.44)
        return (
            np.exp(-A)
            / np.pi
            * np.arctan((x**2 / (1 - x**2) * (A - chi_p) / (A - kappa)) ** 0.5)
            + x / (1 - x**2) ** 0.5 * (chi_p - chi) / (2 * np.pi) * H1
        )


def eta_3(A, chi, x, x_B, spherical=True):
    r"""Calculate the contribution to the charge density when there is a finite radius at which the potential is 0.

    This only appears when the temperature of the attracted particles is zero.
    This is the straight line given by :math:`\beta = \Omega x_B^2`.

    Parameters
    ----------
    A : `numpy.ndarray`
        Lower bound of the integral. Also, `A >= kappa` as defined in (E.3).
    chi : `numpy.ndarray`
        The normalized potential as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    x_B : `float`
        Normalized inverse radius at which the potential is 0.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    `numpy.ndarray`

    Notes
    -----
    This is the term :math:`\eta_3(A)` that appears in various formula in appendix E.

    Equation (E.20) along with figure (10b) show a good example of where to use this term.
    """
    # Equation (E.5).
    beta_B = -chi / (x**2 / x_B**2 - 1)
    if spherical:
        # Equation (E.27) where we've simplified the last integral.
        return -(((x**2 / x_B**2 - 1) / np.pi) ** 0.5) * (
            (beta_B - A) ** 0.5 * np.exp(-A)
            - np.exp(-beta_B) * np.pi**0.5 / 2 * erfi((beta_B - A) ** 0.5)
        )
    else:
        # TODO: See if this should be converted to using the approximations given in equations (E.66) - (E.85).
        # Equation (E.11).
        def integrand(beta):
            np.exp(-beta) * np.arcsin((x**2 / x_B**2 * beta / (beta - chi)) ** 0.5)

        return 1 / np.pi * quad_vec(integrand, A, beta_B)[0]


def eta_4(
    s1_indeces,
    s2_indeces,
    complex_integrand_at_s1,
    complex_integrand_at_s2,
    s_indeces,
    ds,
    chi,
    dchi_ds,
    x,
    dx_ds,
    eta_net,
    gamma,
    omega_G,
    beta_G,
    spherical=True,
):
    r"""Calculate the contribution to the charge density across the locus of extrema.

    Parameters
    ----------
    s1_indeces, s2_indeces : `numpy.ndarray[float]`
        Lower and upper indices of `s` to integrate between where `s1_indeces <= s2_indeces`.
    complex_integrand_at_s1, complex_integrand_at_s2 : `numpy.ndarray[bool]`
        Whether the integrand is complex at s1 or s2.
    s_indeces : `numpy.ndarray[int]`
        The `s` index at which `eta_4` is calculated.
    ds : `float`
        Spacing between any two points in `s`.
    chi : `numpy.ndarray`
        The normalized potential as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    eta_net : `numpy.ndarray`
        Normalized net charge density from equation (9.1).
    gamma : `float`
        :math:`\gamma = R_p^2 / \lambda_D^2` from equation (9.1).
    omega_G, beta_G : `numpy.ndarray`
        Normalized angular momentum and energy of the locus of extrema as functions of `s`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    `numpy.ndarray`

    Notes
    -----
    All the terms that enter the integrand (`chi`, `x`, `eta_net`, `omega_G`,
    and `beta_G`) should be 1D-arrays that contain the value for all points in
    `s`.

    This is the term :math:`\eta_4(\beta_1, \beta_2)` that appears in various
    formula in appendix E. The integration is defined in the `CAL` function on
    page 36 of the computer program listing, appendix I.
    """
    if spherical:
        # Equation (E.30).
        alpha_G = dchi_ds / 2 + gamma * eta_net / (2 * x**3) * dx_ds
        # Equation (E.31).
        epsilon_G = alpha_G * np.exp(-beta_G)
        # Equation (E.32).
        psi_G = (
            beta_G[np.newaxis, :]
            - chi[:, np.newaxis]
            - omega_G[np.newaxis, :] * x[:, np.newaxis] ** 2
        ) ** 0.5

        all_integrands = (
            -1 / np.pi**0.5 * epsilon_G[np.newaxis, :] * psi_G
        )  # Indexed as `[s, s']` in (E.33).
    else:
        # Equation (E.86).
        alpha_G = dchi_ds + gamma * eta_net / (2 * x**3) * dx_ds
        # Equation (E.31).
        epsilon_G = alpha_G * np.exp(-beta_G)
        # Equation (E.87).
        psi_G = np.arctan(
            (
                omega_G[np.newaxis, :]
                * x[:, np.newaxis] ** 2
                / (
                    beta_G[np.newaxis, :]
                    - chi[:, np.newaxis]
                    - omega_G[np.newaxis, :] * x[:, np.newaxis] ** 2
                )
            )
            ** 0.5
        )

        all_integrands = 1 / np.pi * epsilon_G[np.newaxis, :] * psi_G
    all_integrands = all_integrands[s_indeces]

    return integrate(
        all_integrands,
        s1_indeces,
        s2_indeces,
        complex_integrand_at_s1,
        complex_integrand_at_s2,
        ds,
    )


def eta_5(A, spherical=True):
    r"""Calculate the contribution to the charge density across the potential boundary at a specific radius.

    Parameters
    ----------
    A : `numpy.ndarray`
        Lower bound of the integral.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.
    """
    if spherical:
        return np.zeros_like(A)
    else:
        return np.exp(-A) / 2


def find_beta_M_index(beta_M, beta_G):
    r"""Find the point at which beta_G crosses beta_M.

    We assume that `x` is ordered from `1 -> 0`. The returned index is the index
    at which `beta_G` is less then `beta_M` for all `x < x_M`, (all `i > beta_M_index`).

    Parameters
    ----------
    beta_M : `float`
        Distribution function energy.
    beta_G : `numpy.ndarray`
        Normalized energy of the locus of extrema as functions of `s`.

    Returns
    -------
    beta_M_index : `int`
        Index of the last value of the `beta_G` array at which `beta_M >= beta_G`.
    """
    if beta_M > np.max(beta_G):
        return 0
    else:
        return beta_G.size - np.argmax(beta_M < beta_G[::-1])


def estimate_omega_M(beta_M, beta_M_index, beta_G, omega_G):
    r"""Do some Taylor expansions to find the omega that corresponds to `beta_M`.

    Parameters
    ----------
    beta_M : `float`
        Distribution function energy.
    beta_M_index : `int`
        Index or the `beta_G` array where `beta_M` crosses `beta_G` just before `beta_M_index` or just after.
    omega_G, beta_G : `numpy.ndarray`
        Normalized angular momentum and energy of the locus of extrema as functions of `s`.

    Returns
    -------
    omega_M : `float`
        Estimated value of `omega` where the crossing occurred.

    Notes
    -----
    We do a second order Taylor expansion of `beta_G` and then solve for the
    index where `beta_G(index) \approx beta_M`. Then we do a second order
    Taylor expansion of `omega_G` to find the correct `omega`. This only works
    if `np.min(beta_G) <= beta_M <= np.max(beta_G)`.
    """
    if beta_M == beta_G[beta_M_index]:
        return omega_G[beta_M_index]

    if beta_M_index == 0:
        expansion_index = 1
    else:
        expansion_index = min(
            beta_G.size - 2,
            beta_M_index
            + round(
                (beta_M - beta_G[beta_M_index - 1])
                / (beta_G[beta_M_index] - beta_G[beta_M_index - 1])
            ),
        )

    # Second order Taylor expansion of `beta_G`.
    zeroth_order = beta_G[expansion_index]
    first_order = (beta_G[expansion_index + 1] - beta_G[expansion_index - 1]) / 2
    second_order = (
        beta_G[expansion_index + 1]
        - 2 * beta_G[expansion_index]
        + beta_G[expansion_index - 1]
    )
    # Now solve the quadratic equation.
    if np.isclose(second_order, 0):
        # If the second derivative is very small then just use the first derivative.
        float_index = expansion_index - (zeroth_order - beta_M) / first_order
    else:
        # Solve the quadratic formula.
        quadratic_first_term = -first_order / second_order
        quadratic_second_term = (
            first_order**2 - 4 * second_order / 2 * (zeroth_order - beta_M)
        ) ** 0.5 / second_order
        terms_added = quadratic_first_term + quadratic_second_term
        terms_subtracted = quadratic_first_term - quadratic_second_term

        # Decide whether to add or subtract the second term.
        if beta_G[expansion_index] > beta_M:
            # The float index must be between `[expansion_index, expansion_index + 1]`.
            float_index = terms_added if 0 <= terms_added <= 1 else terms_subtracted
        else:
            # The float index must be between `[expansion_index - 1, expansion_index]`.
            float_index = terms_added if -1 <= terms_added <= 0 else terms_subtracted
        float_index += expansion_index

    # Second order Taylor expansion of `omega_G`.
    zeroth_order = omega_G[expansion_index]
    first_order = (omega_G[expansion_index + 1] - omega_G[expansion_index - 1]) / 2
    second_order = (
        omega_G[expansion_index + 1]
        - 2 * omega_G[expansion_index]
        + omega_G[expansion_index - 1]
    )
    return (
        zeroth_order
        + (float_index - expansion_index) * first_order
        + (float_index - expansion_index) ** 2 * second_order / 2
    )


def delta_function_charge_density(chi, x, omega_G, beta_G, spherical=True):
    r"""Calculate the charge density when the distribution function is a delta function.

    Calculate the charge density for a single species of particles.

    Parameters
    ----------
    chi : `numpy.ndarray`
        The normalized potential as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    omega_G, beta_G : `numpy.ndarray`
        Normalized angular momentum and energy of the locus of extrema as functions of `s`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    eta : `numpy.ndarray`
    """
    # Equation (13.1).
    beta_M = 4 / np.pi if spherical else np.pi / 4

    # Create an array that stores the omega corresponding to the boundary of no allowed particles.
    # Populate the array with the omega using only the local boundary.
    no_particle_omega_boundary = (beta_M - chi) / x**2
    # Find the indices where `beta_G` crosses `beta_M`. These are the indices
    # just before crossing. We only need crossings in which
    # `beta_G[index] < beta_M < beta_G[index + 1]`. This is because these
    # crossings are the ones that will globally limit omega.
    crossing_indeces = np.nonzero(
        np.logical_and(beta_G[:-1] < beta_M, beta_G[1:] > beta_M)
    )[0]
    # If there are any crossings then find the omega that corresponds to each of these crossings.
    if crossing_indeces.size > 0:
        for crossing_index in crossing_indeces:
            crossed_omega_G = estimate_omega_M(beta_M, crossing_index, beta_G, omega_G)
            # Apply the global boundary of the extrema.
            no_particle_omega_boundary[: crossing_index + 1] = np.min(
                crossed_omega_G, no_particle_omega_boundary[: crossing_index + 1]
            )

    no_particle_omega_boundary[no_particle_omega_boundary < 0] = 0
    # Determine the boundary where the probe consumes incoming particles.
    probe_consumption_omega_boundary = no_particle_omega_boundary[0]

    factors = beta_M > chi
    # Equation (13.1).
    if spherical:
        eta = (
            -1
            / (2 * beta_M**0.5)
            * (
                -1 * (factors * (beta_M - chi)) ** 0.5
                + -1
                * (factors * (beta_M - chi - probe_consumption_omega_boundary * x**2))
                ** 0.5
                + 2
                * (factors * (beta_M - chi - no_particle_omega_boundary * x**2)) ** 0.5
            )
        )
    else:
        eta = (
            1
            / np.pi
            * (
                # The zero boundary term evaluates to 0.
                -1
                * np.arcsin(
                    (
                        factors
                        * (probe_consumption_omega_boundary * x**2 / (beta_M - chi))
                    )
                    ** 0.5
                )
                + 2
                * np.arcsin(
                    (factors * (no_particle_omega_boundary * x**2 / (beta_M - chi)))
                    ** 0.5
                )
            )
        )

    return eta


def determine_coefficients_and_integration_points():
    """Determine the coefficients of all eta and the integration start and end points."""
    raise NotImplementedError(
        "The calculation for determining the coefficients and integration points has not been implemented."
    )


def get_charge_density(chi, x, omega_G, beta_G, spherical=True, maxwellian=True):
    r"""Calculate the charge density at all grid points.

    Calculate the charge density for a single species of particles.

    Parameters
    ----------
    chi : `numpy.ndarray`
        The normalized potential as defined in `~plasmapy.diagnostics.brl.normalizations.get_normalized_potential`.
    x : `numpy.ndarray`
        Normalized inverse radius calculated from `~plasmapy.diagnostics.brl.net_spacing.get_x_and_dx_ds`.
    omega_G, beta_G : `numpy.ndarray`
        Normalized angular momentum and energy of the locus of extrema as functions of `s`.
    spherical : `bool`, optional
        If `True` the probe will be treated as spherical. If `False` then the probe is cylindrical. Default is `True`.

    Returns
    -------
    eta : `numpy.ndarray`
    """
    if not maxwellian:
        return delta_function_charge_density(
            chi, x, omega_G, beta_G, spherical=spherical
        )
    else:
        raise NotImplementedError(
            "The calculation for charge density for a maxwellian distribution has not been implemented."
        )
