"""Numerical solutions of the kinetic Alfvén wave dispersion relation."""

__all__ = ["kinetic_alfven"]

import warnings
from numbers import Real

import astropy.units as u
import numpy as np
from astropy.constants.si import c

from plasmapy.formulary import frequencies as pfp
from plasmapy.formulary import speeds as speed
from plasmapy.particles import ParticleLike, particle_input
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning

c_si_unitless = c.value


@particle_input
@validate_quantities(
    B={"can_be_negative": False},
    n_i={"can_be_negative": False},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def kinetic_alfven(  # noqa: C901, PLR0912
    B: u.Quantity[u.T],
    ion: ParticleLike,
    k: u.Quantity[u.rad / u.m],
    n_i: u.Quantity[u.m**-3],
    theta: u.Quantity[u.deg],
    *,
    T_e: u.Quantity[u.K],
    T_i: u.Quantity[u.K],
    gamma_e: float = 1,
    gamma_i: float = 3,
    mass_numb: int | None = None,
    Z: float | None = None,
):
    r"""Using the equation provided in :cite:t:`bellan:2012`, this function
    calculates the numerical solution to the kinetic Alfvén dispersion
    relation presented by :cite:t:`hirose:2004`.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to T.

    ion : |particle-like|
        Representation of the ion species (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.).

    k : `~astropy.units.Quantity`
        Wavenumber in units convertible to rad / m. Either single
        valued or 1-D array of length :math:`N`.

    n_i : `~astropy.units.Quantity`
        Ion number density in units convertible to m\ :sup:`-3`.

    T_e : `~astropy.units.Quantity`, |keyword-only|
        The electron temperature in units of K or eV.

    T_i : `~astropy.units.Quantity`, |keyword-only|
        The ion temperature in units of K or eV.

    theta : `~astropy.units.Quantity`
        The angle of propagation of the wave with respect to the
        magnetic field, :math:`\cos^{-1}(k_z / k)`, in units
        convertible to rad. Either single valued or 1-D array of size
        :math:`M`.

    gamma_e : real number, |keyword-only|, default: 1
        The adiabatic index for electrons. The default value assumes
        that the electrons are able to equalize their temperature
        rapidly enough that the electrons are effectively isothermal.

    gamma_i : real number, |keyword-only|, default: 3
        The adiabatic index for ions. The default value assumes that
        ion motion has only one degree of freedom, namely along
        magnetic field lines.

    mass_numb : integer, |keyword-only|, optional
        The mass number corresponding to ``ion``.

    Z : real number, |keyword-only|, optional
        The charge number corresponding to ``ion``.

    Returns
    -------
    omega : Dict[str, `~astropy.units.Quantity`]
        A dictionary of computed wave frequencies in units rad /
        s. The dictionary contains a key for each: ``theta`` value
        provided. The value for each key will be an :math:`N × M`
        array.

    Raises
    ------
    TypeError
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    TypeError
        If ``ion`` is not |particle-like|.

    TypeError
        If ``gamma_e``, ``gamma_i``, or ``Z`` are not real numbers.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    ValueError
        If any of ``B``, ``k``, ``n_i``, ``T_e``, or ``T_i`` is negative.

    ValueError
        If ``k`` is negative or zero.

    ValueError
        If ``ion`` is not of category ion or element.

    ValueError
        If ``B``, ``n_i``, ``T_e``, or ``T_i`` are not single valued
        `astropy.units.Quantity` (i.e. an array).

    ValueError
        If ``k`` or ``theta`` are not single valued or a 1-D array.

    Notes
    -----
    Using the 2 × 2 matrix approach method from :cite:t:`bellan:2012`,
    this function computes the corresponding wave frequencies in units
    of rad / s. This approach comes from :cite:t:`hasegawa:1982`,
    :cite:t:`morales:1997` and :cite:t:`william:1996`; who argued that
    a 3 × 3 matrix that describes warm plasma waves can be represented
    as a 2 × 2 matrix because the compressional (i.e., fast) mode can
    be factored out. The result is that the determinant, when in the
    limit of :math:`ω ≫ k_{z}^{2} c^{2}_{\rm s}`, reduces to the
    kinetic Alfvén dispersion relation.

    .. math::
        ω^2 = k_{\rm z}^2 v_{\rm A}^2 \left(1 + \frac{k_{\rm x}^2
        c_{\rm s}^2}{ω_{\rm ci}^2} \right)

    With :math:`c_{\rm s}` being the wave speed and :math:`ω_{\rm ci}`
    as the gyrofrequency of the respective ion.  The regions in which
    this is valid are :math:`ω ≪ ω_{\rm ci}` and :math:`ν_{\rm Te} ≫
    \frac{ω}{k_{z}} ≫ ν_{\rm Ti}`, with :math:`ν_{\rm Ti}`
    standing for the thermal speed of the respective ion. There is no
    restriction on propagation angle.

    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    >>> from plasmapy.particles import Particle
    >>> from plasmapy.dispersion.numerical import kinetic_alfven_
    >>> inputs = {
    ...     "B": 8.3e-9 * u.T,
    ...     "ion": Particle("p+"),
    ...     "k": np.logspace(-7, -2, 2) * u.rad / u.m,
    ...     "n_i": 5 * u.m**-3,
    ...     "T_e": 1.6e6 * u.K,
    ...     "T_i": 4.0e5 * u.K,
    ...     "theta": 30 * u.deg,
    ...     "gamma_e": 3,
    ...     "gamma_i": 3,
    ...     "Z": 1,
    ... }
    >>> kinetic_alfven(**inputs)
    {np.float64(30.0): <Quantity [1.24901116e+00, 3.45301796e+08] rad / s>}
    """

    # Validate arguments
    for arg_name in ("B", "n_i", "T_e", "T_i"):
        val = locals()[arg_name].squeeze()
        if val.shape != ():
            raise ValueError(
                f"Argument '{arg_name}' must be a single value and not "
                f"an array of shape '{val.shape}'."
            )
        locals()[arg_name] = val

    for arg_name in (gamma_e, gamma_i):
        if not isinstance(arg_name, Real):
            raise TypeError(
                f"Expected int or float for argument '{arg_name}', "
                f"instead got type {type(arg_name)}."
            )

    # Validate argument k
    k = k.value.squeeze()
    if k.ndim not in {0, 1}:
        raise ValueError(
            "Argument 'k' needs to be a single valued or 1D array "
            f"astropy Quantity, instead got array of shape {k.shape}."
        )
    elif np.isscalar(k):
        k = np.array([k])
    if np.any(k <= 0):
        raise ValueError("Argument 'k' can not be negative a or have negative values.")

    # Validate argument theta
    theta = theta.value.squeeze()
    if theta.ndim not in {0, 1}:
        raise ValueError(
            "Argument 'theta' needs to be a single valued or 1D array "
            f"astropy Quantity, instead got array of shape {theta.shape}."
        )
    elif np.isscalar(theta):
        theta = np.array([theta])

    Z = ion.charge_number
    n_e = Z * n_i
    c_s = speed.ion_sound_speed(
        T_e=T_e,
        T_i=T_i,
        ion=ion,
        n_e=n_e,
        gamma_e=gamma_e,
        gamma_i=gamma_i,
        Z=Z,
    )
    v_A = speed.Alfven_speed(B=B, density=n_i, ion=ion)
    omega_ci = pfp.gyrofrequency(B=B, particle=ion, signed=False)

    # parameters kz
    omega = {}
    for θ in theta:  # vectorize this?
        kz = np.cos(θ) * k
        kx = np.sqrt(k**2 - kz**2)

        # parameters sigma, D, and F to simplify equation 3
        A = (kz * v_A) ** 2
        F = ((kx * c_s) / omega_ci) ** 2

        omega[θ] = (np.sqrt(A.value * (1 + F.value))) * u.rad / u.s

        # thermal speeds for electrons and ions in plasma
        v_Te = speed.thermal_speed(T=T_e, particle="e-").value
        v_Ti = speed.thermal_speed(T=T_i, particle=ion).value

        # Maximum value of omega
        w_max = np.max(omega[θ])

        # Maximum and minimum values for ω/kz
        omega_kz = omega[θ] / kz

        omega_kz_max = np.max(omega_kz).value
        omega_kz_min = np.min(omega_kz).value

        # Maximum value for ω/kz test
        if omega_kz_max / v_Te > 0.1 or v_Ti / omega_kz_max > 0.1:
            warnings.warn(
                "This calculation produced one or more invalid ω/kz "
                "value(s), which violates the regime in which the "
                "dispersion relation is valid (v_Te ≫ ω/kz ≫ v_Ti)",
                PhysicsWarning,
            )

        # Minimum value for ω/kz test
        if omega_kz_min / v_Te > 0.1 or v_Ti / omega_kz_min > 0.1:
            warnings.warn(
                "This calculation produced one or more invalid ω/kz "
                "value(s) which violates the regime in which the "
                "dispersion relation is valid (v_Te ≫ ω/kz ≫ v_Ti)",
                PhysicsWarning,
            )

        # Dispersion relation is only valid in the regime ω << ω_ci
        if w_max / omega_ci > 0.1:
            warnings.warn(
                "The calculation produced a high-frequency wave, "
                "which violates the low frequency assumption (ω ≪ ω_ci)",
                PhysicsWarning,
            )

    return omega
