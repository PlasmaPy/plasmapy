"""Numerical solution of Hollweg's two-fluid dispersion relation."""

__all__ = ["hollweg"]

import warnings
from numbers import Real

import astropy.units as u
import numpy as np
from astropy.constants.si import c

from plasmapy.formulary.frequencies import gyrofrequency, plasma_frequency
from plasmapy.formulary.speeds import Alfven_speed, ion_sound_speed
from plasmapy.particles import ParticleLike, particle_input
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning

c_si_unitless = c.value


@validate_quantities(
    B={"can_be_negative": False},
    n_i={"can_be_negative": False},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
@particle_input
def hollweg(  # noqa: C901, PLR0912, PLR0915
    B: u.Quantity[u.T],
    ion: ParticleLike,
    k: u.Quantity[u.rad / u.m],
    n_i: u.Quantity[u.m**-3],
    theta: u.Quantity[u.rad],
    *,
    T_e: u.Quantity[u.K],
    T_i: u.Quantity[u.K],
    gamma_e: float = 1,
    gamma_i: float = 3,
    mass_numb: int | None = None,
    Z: float | None = None,
):
    r"""
    Calculate the two-fluid dispersion relation presented by
    :cite:t:`hollweg:1999`, and discussed by :cite:t:`bellan:2012`.

    This is a numerical solver of equation 3 in :cite:t:`bellan:2012`.
    See the **Notes** section below for additional details.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    ion : |particle-like|
        Representation of the ion species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.

    k : `~astropy.units.Quantity`
        Wavenumber in units convertible to rad/m.  Either single
        valued or 1-D array of length :math:`N`.

    n_i : `~astropy.units.Quantity`
        Ion number density in units convertible to m\ :sup:`-3`.

    theta : `~astropy.units.Quantity`
        The angle of propagation of the wave with respect to the
        magnetic field, :math:`\cos^{-1}(k_z / k)`, in units convertible
        to radians. Either single valued or 1-D array of size
        :math:`M`.

    T_e : `~astropy.units.Quantity`, |keyword-only|
        The electron temperature in units of K or eV.

    T_i : `~astropy.units.Quantity`, |keyword-only|
        The ion temperature in units of K or eV.

    gamma_e : real number, |keyword-only|, default: 1
        The adiabatic index for electrons. The default value
        assumes that the electrons are able to equalize their
        temperature rapidly enough that the electrons are
        effectively isothermal.

    gamma_i : real number, |keyword-only|, default: 3
        The adiabatic index for ions. The default value assumes
        that ion motion has only one degree of freedom, namely
        along magnetic field lines.

    mass_numb : integer, |keyword-only|, optional
        The mass number corresponding to ``ion``.

    Z : real number, |keyword-only|, optional
        The charge number corresponding to ``ion``.

    Returns
    -------
    omega : Dict[str, `~astropy.units.Quantity`]
        A dictionary of computed wave frequencies in units rad/s.
        The dictionary contains three keys: ``'fast_mode'`` for
        the fast mode, ``'alfven_mode'`` for the Alfvén mode, and
        ``'acoustic_mode'`` for the ion-acoustic mode.  The value
        for each key will be an :math:`N x M` array.

    Raises
    ------
    TypeError
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    TypeError
        If ``ion`` is not of type or convertible to
        `~plasmapy.particles.particle_class.Particle`.

    TypeError
        If ``gamma_e``, ``gamma_i``, or ``z_mean`` are not real
        numbers.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    ValueError
        If any of ``B``, ``k``, ``n_i``, ``T_e``, or ``T_i`` is
        negative.

    ValueError
        If ``k`` is negative or zero.

    ValueError
        If ``ion`` is not of category ion or element.

    ValueError
        If ``B``, ``n_i``, ``T_e``, or ``T_I`` are not single valued
        `astropy.units.Quantity` (i.e. an array).

    ValueError
        If ``k`` or ``theta`` are not single valued or a 1-D array.

    Warns
    -----
    : `~plasmapy.utils.exceptions.PhysicsWarning`
        When :math:`ω / ω_{\rm ci} > 0.1`, violation of the
        low-frequency assumption.

    : `~plasmapy.utils.exceptions.PhysicsWarning`
        When :math:`c_{\rm s} / v_{\rm A} > 0.1`, violation of low-β.

    : `~plasmapy.utils.exceptions.PhysicsWarning`
        When :math:`|θ - π/2| > 0.1`, violation of quasi-perpendicular
        propagation.

    Notes
    -----
    The dispersion relation presented in :cite:t:`hollweg:1999`
    (equation 3 in :cite:t:`bellan:2012`) is:

    .. math::
        \left( \frac{ω^2}{k_{\rm z}^2 v_{\rm A}^2} - 1 \right) &
        \left[
            ω^2 \left( ω^2 - k^2 v_{\rm A}^2 \right)
            - β k^2 v_{\rm A}^2 \left(
                ω^2 - k_{\rm z}^2 v_{\rm A}^2
            \right)
        \right] \\
        &= ω^2 \left(ω^2 - k^2 v_{\rm A}^2 \right) k_{\rm x}^2
        \left(
            \frac{c_{\rm s}^2}{ω_{\rm ci}^2}
            - \frac{c^2}{ω_{\rm pe}^2} \frac{ω^2}{k_{\rm z}^2v_{\rm A}^2}
        \right)

    where

    .. math::
        \mathbf{B}_0 &= B_0 \mathbf{\hat{z}} \\
        \cos θ &= \frac{k_z}{k} \\
        \mathbf{k} &= k_{\rm x} \hat{x} + k_{\rm z} \hat{z}

    :math:`ω` is the wave frequency, :math:`k` is the wavenumber,
    :math:`v_{\rm A}` is the Alfvén velocity, :math:`c_{\rm s}` is the
    sound speed, :math:`ω_{\rm ci}` is the ion gyrofrequency, and
    :math:`ω_{\rm pe}` is the electron plasma frequency. In the
    derivation of this relation Hollweg assumed low-frequency waves
    :math:`ω / ω_{\rm ci} ≪ 1`, no D.C. electric field
    :math:`\mathbf{E}_0=0`, and quasineutrality.

    :cite:t:`hollweg:1999` asserts this expression is valid for
    arbitrary :math:`c_{\rm s} / v_{\rm A}` (β) and
    :math:`k_{\rm z} / k` (:math:`θ`). Contrarily, :cite:t:`bellan:2012`
    states in §1.7 that due to the inconsistent retention of the
    :math:`ω / ω_{\rm ci} ≪ 1` terms the expression can
    only be valid if both :math:`c_{\rm s} ≪ v_{\rm A}` (low-β) and
    the wave propagation is nearly perpendicular to the magnetic field.

    This routine solves for :math:`ω` for given :math:`k` values
    by numerically solving for the roots of the above expression.

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.dispersion.numerical import hollweg_
    >>> inputs = {
    ...    "k": np.logspace(-7, -2, 2) * u.rad / u.m,
    ...    "theta": 88 * u.deg,
    ...    "n_i": 5 * u.cm ** -3,
    ...    "B": 2.2e-8 * u.T,
    ...    "T_e": 1.6e6 * u.K,
    ...    "T_i": 4.0e5 * u.K,
    ...    "ion": "p+",
    ... }
    >>> omegas = hollweg(**inputs)
    >>> omegas
    {'fast_mode': <Quantity [2.62911663e-02+0.j, 2.27876968e+03+0.j] rad / s>,
     'alfven_mode': <Quantity [7.48765909e-04+0.j, 2.13800404e+03+0.j] rad / s>,
     'acoustic_mode': <Quantity [0.00043295+0.j, 0.07358991+0.j] rad / s>}
    """

    # validate arguments
    for arg_name in ("B", "n_i", "T_e", "T_i"):
        val = locals()[arg_name].squeeze()
        if val.shape != ():
            raise ValueError(
                f"Argument '{arg_name}' must be single valued and not "
                f"an array of shape {val.shape}."
            )
        locals()[arg_name] = val

    # validate arguments
    for arg_name in ("gamma_e", "gamma_i"):
        if not isinstance(locals()[arg_name], Real):
            raise TypeError(
                f"Expected int or float for argument '{arg_name}', but "
                f"got {type(locals()[arg_name])}."
            )

    # validate argument k
    k = k.squeeze()
    if k.ndim not in {0, 1}:
        raise ValueError(
            f"Argument 'k' needs to be single valued or a 1D array "
            f"astropy Quantity, got array of shape {k.shape}."
        )
    if np.any(k <= 0):
        raise ValueError("Argument 'k' cannot be negative or have negative values.")

    # validate argument theta
    theta = theta.squeeze()
    if theta.ndim not in {0, 1}:
        raise ValueError(
            f"Argument 'theta' needs to be a single valued or 1D array astropy "
            f"Quantity, got array of shape {theta.shape}."
        )

    # Single k value case
    if np.isscalar(k.value):
        k = np.array([k.value]) * u.rad / u.m

    # Calc needed plasma parameters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PhysicsWarning)
        n_e = ion.charge_number * n_i
        c_s = ion_sound_speed(
            T_e=T_e,
            T_i=T_i,
            ion=ion,
            n_e=n_e,
            gamma_e=gamma_e,
            gamma_i=gamma_i,
        ).value
        v_A = Alfven_speed(B, n_i, ion=ion).value
        omega_ci = gyrofrequency(B=B, particle=ion, signed=False).value
        omega_pe = plasma_frequency(n=n_e, particle="e-").value

    cs_vA = c_s / v_A
    thetav, kv = np.meshgrid(theta.value, k.value)

    # Parameters kx and kz
    kz = np.cos(thetav) * kv
    kx = np.sin(thetav) * kv

    # Define helpful parameters
    beta = (c_s / v_A) ** 2
    alpha_A = (kv * v_A) ** 2
    alpha_s = (kv * c_s) ** 2
    sigma = (kz * v_A) ** 2
    D = (c_s / omega_ci) ** 2
    F = (c_si_unitless / omega_pe) ** 2

    # Polynomial coefficients: c3*x^3 + c2*x^2 + c1*x + c0 = 0
    c3 = F * kx**2 + 1
    c2 = -alpha_A * (1 + beta + F * kx**2) - sigma * (1 + D * kx**2)
    c1 = sigma * alpha_A * (1 + 2 * beta + D * kx**2)
    c0 = -alpha_s * sigma**2

    # Find roots to polynomial
    coefficients = np.array([c3, c2, c1, c0], ndmin=3)
    nroots = coefficients.shape[0] - 1  # 3
    nks = coefficients.shape[1]
    nthetas = coefficients.shape[2]
    roots = np.empty((nroots, nks, nthetas), dtype=np.complex128)
    for ii in range(nks):
        for jj in range(nthetas):
            roots[:, ii, jj] = np.roots(coefficients[:, ii, jj])

    roots = np.sqrt(roots)  # type: ignore[assignment]
    roots = np.sort(roots, axis=0)  # type: ignore[assignment]

    # Warn about NOT low-β
    if c_s / v_A > 0.1:
        warnings.warn(
            f"This solver is valid in the low-beta regime, "
            f"c_s/v_A ≪ 1 according to Bellan, 2012, Sec. 1.7 "
            f"(see documentation for DOI). A c_s/v_A value of {cs_vA:.2f} "
            f"was calculated which may affect the validity of the solution.",
            PhysicsWarning,
        )

    # Warn about theta not nearly perpendicular
    theta_diff_max = np.amax(np.abs(thetav - np.pi / 2))
    if theta_diff_max > 0.1:
        warnings.warn(
            f"This solver is valid in the regime where propagation is "
            f"nearly perpendicular to B according to Bellan, 2012, Sec. 1.7 "
            f"(see documentation for DOI). A |theta - pi/2| value of "
            f"{theta_diff_max:.2f} was calculated which may affect the "
            f"validity of the solution.",
            PhysicsWarning,
        )

    # dispersion relation is only valid in the regime ω ≪ ω_ci
    w_max = np.max(roots)
    w_wci_max = w_max / omega_ci
    if w_wci_max > 0.1:
        warnings.warn(
            f"This solver is valid in the regime ω/ω_ci ≪ 1. A ω "
            f"value of {w_max:.2f} and a ω/ω_ci value of "
            f"{w_wci_max:.2f} were calculated which may affect the "
            f"validity of the solution.",
            PhysicsWarning,
        )

    return {
        "fast_mode": roots[2, :].squeeze() * u.rad / u.s,
        "alfven_mode": roots[1, :].squeeze() * u.rad / u.s,
        "acoustic_mode": roots[0, :].squeeze() * u.rad / u.s,
    }
