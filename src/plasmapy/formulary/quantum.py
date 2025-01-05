"""Quantum parameters, including electron degenerate plasmas and warm dense matter."""

__all__ = [
    "chemical_potential",
    "deBroglie_wavelength",
    "Fermi_energy",
    "quantum_theta",
    "Thomas_Fermi_length",
    "thermal_deBroglie_wavelength",
    "Wigner_Seitz_radius",
]
__aliases__ = ["Ef_", "lambdaDB_", "lambdaDB_th_"]

import astropy.units as u
import numpy as np
from astropy.constants.si import c, e, eps0, h, hbar, k_B, m_e
from lmfit import Parameters, minimize

from plasmapy.formulary import mathematics
from plasmapy.formulary.relativity import Lorentz_factor
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import RelativityError

__all__ += __aliases__


# TODO: Use @check_relativistic
@validate_quantities(
    V={"can_be_negative": True}, validations_on_return={"can_be_negative": False}
)
@particle_input
def deBroglie_wavelength(
    V: u.Quantity[u.m / u.s],
    particle: ParticleLike,
) -> u.Quantity[u.m]:
    r"""
    Return the de Broglie wavelength.

    The de Broglie wavelength (:math:`λ_{dB}`) of a particle is defined by

    .. math::

        λ_{dB} = \frac{h}{p} = \frac{h}{γ m V}

    where :math:`h` is the Planck constant, :math:`p` is the
    relativistic momentum of the particle, :math:`γ` is the
    Lorentz factor, :math:`m` is the mass of the particle, and
    :math:`V` is the velocity of the particle.

    **Aliases:** `lambdaDB_`

    Parameters
    ----------
    V : `~astropy.units.Quantity`
        Particle velocity in units convertible to meters per second.

    particle : `str`, `~plasmapy.particles.particle_class.Particle`, or |Quantity|
        An instance of `~plasmapy.particles.particle_class.Particle`, or
        an equivalent representation (e.g., ``'e-'``, ``'p+'``, ``'D+'``, or
        ``'He-4 1+'``), for the particle of interest, or the particle
        mass in units convertible to kg.  If a
        `~plasmapy.particles.particle_class.Particle` instance is given, then the
        particle mass is retrieved from the object.

    Returns
    -------
    lambda_dB : `~astropy.units.Quantity`
        The de Broglie wavelength in units of meters.

    Raises
    ------
    `TypeError`
        If the velocity is not a `~astropy.units.Quantity` and cannot be
        converted into a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the velocity is not in appropriate units.

    `~plasmapy.utils.exceptions.RelativityError`
        If the magnitude of ``V`` is larger than the speed of light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Examples
    --------
    >>> import astropy.units as u
    >>> velocity = 1.4e7 * u.m / u.s
    >>> deBroglie_wavelength(velocity, "e-")
    <Quantity 5.18997095e-11 m>
    >>> deBroglie_wavelength(V=0 * u.m / u.s, particle="D+")
    <Quantity inf m>
    """

    V = np.abs(V)

    if np.any(V >= c):
        raise RelativityError(
            "Velocity input in deBroglie_wavelength cannot "
            "be greater than or equal to the speed of "
            "light."
        )

    if V.size > 1:
        lambda_dBr = np.ones(V.shape) * np.inf * u.m
        indices = V.value != 0
        lambda_dBr[indices] = h / (
            particle.mass * V[indices] * Lorentz_factor(V[indices])
        )

    elif V == 0 * u.m / u.s:
        lambda_dBr = np.inf * u.m
    else:
        lambda_dBr = h / (Lorentz_factor(V) * particle.mass * V)

    return lambda_dBr


lambdaDB_ = deBroglie_wavelength
"""Alias to `~plasmapy.formulary.quantum.deBroglie_wavelength`."""


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    validations_on_return={"can_be_negative": False},
)
def thermal_deBroglie_wavelength(T_e: u.Quantity[u.K]) -> u.Quantity[u.m]:
    r"""
    Calculate the thermal de Broglie wavelength for electrons.

    **Aliases:** `lambdaDB_th_`

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        Electron temperature.

    Returns
    -------
    lambda_dbTh : `~astropy.units.Quantity`
        The thermal de Broglie wavelength for electrons in meters.

    Raises
    ------
    `TypeError`
        If argument is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If argument is in incorrect units.

    `ValueError`
        If argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The thermal de Broglie wavelength is approximately the average de Broglie
    wavelength for electrons in an ideal gas and is given by

    .. math::

       λ_{dbTh} = \frac{h}{\sqrt{2 π m_e k_B T_e}}

    Examples
    --------
    >>> import astropy.units as u
    >>> thermal_deBroglie_wavelength(1 * u.eV)
    <Quantity 6.9193675e-10 m>
    """
    return h / np.sqrt(2 * np.pi * m_e * k_B * T_e)


lambdaDB_th_ = thermal_deBroglie_wavelength
"""Alias to `~plasmapy.formulary.quantum.thermal_deBroglie_wavelength`."""


@validate_quantities(
    n_e={"can_be_negative": False}, validations_on_return={"can_be_negative": False}
)
def Fermi_energy(n_e: u.Quantity[u.m**-3]) -> u.Quantity[u.J]:
    r"""
    Calculate the kinetic energy in a degenerate electron gas.

    **Aliases:** `Ef_`

    Parameters
    ----------
    n_e : ~astropy.units.Quantity
        Electron number density.

    Returns
    -------
    energy_F : `~astropy.units.Quantity`
        The Fermi energy in joules.

    Raises
    ------
    `TypeError`
        If argument is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If argument is in incorrect units.

    `ValueError`
        If argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    See Also
    --------
    Thomas_Fermi_length

    Notes
    -----
    The Fermi energy is the kinetic energy in a degenerate electron gas
    and is given by

    .. math::

       E_F = \frac{π^2 ℏ^2}{2 m_e}
       \left( \frac{3 n_e}{π} \right)^{2/3}

    This quantity is often used in place of thermal energy for analysis
    of cold, dense plasmas (e.g. warm dense matter, condensed matter).

    Examples
    --------
    >>> import astropy.units as u
    >>> Fermi_energy(1e23 * u.cm**-3)
    <Quantity 1.2586761e-18 J>
    """
    coeff = (np.pi * hbar) ** 2 / (2 * m_e)
    return coeff * (3 * n_e / np.pi) ** (2 / 3)


Ef_ = Fermi_energy
"""Alias to `~plasmapy.formulary.quantum.Fermi_energy`."""


@validate_quantities(
    n_e={"can_be_negative": False}, validations_on_return={"can_be_negative": False}
)
def Thomas_Fermi_length(n_e: u.Quantity[u.m**-3]) -> u.Quantity[u.m]:
    r"""
    Calculate the exponential scale length for charge screening
    for cold and dense plasmas.

    Parameters
    ----------
    n_e : `~astropy.units.Quantity`
        Electron number density.

    Returns
    -------
    lambda_TF : `~astropy.units.Quantity`
        The Thomas-Fermi screening length in meters.

    Raises
    ------
    `TypeError`
        If argument is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If argument is in incorrect units.

    `ValueError`
        If argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    See Also
    --------
    ~plasmapy.formulary.quantum.Fermi_energy
    ~plasmapy.formulary.lengths.Debye_length

    Notes
    -----
    The Thomas-Fermi screening length is the exponential scale length for
    charge screening and is given by

    .. math::

       λ_{TF} = \sqrt{\frac{2 ε_0 E_F}{3 n_e e^2}}

    for an electron degenerate gas.

    This quantity is often used in place of the Debye length for analysis
    of cold, dense plasmas (e.g. warm dense matter, condensed matter).

    The electrical potential will drop by a factor of :math:`1/e` every
    Thomas-Fermi screening length.

    Plasmas will generally be quasineutral on length scales significantly
    larger than the Thomas-Fermi screening length.

    Examples
    --------
    >>> import astropy.units as u
    >>> Thomas_Fermi_length(1e23 * u.cm**-3)
    <Quantity 5.37991409e-11 m>
    """
    energy_F = Fermi_energy(n_e)
    return np.sqrt(2 * eps0 * energy_F / (3 * n_e * e**2))


@validate_quantities(
    n={"can_be_negative": False}, validations_on_return={"can_be_negative": False}
)
def Wigner_Seitz_radius(n: u.Quantity[u.m**-3]) -> u.Quantity[u.m]:
    r"""
    Calculate the Wigner-Seitz radius, which approximates the inter-particle
    spacing.

    This function returns the radius of a sphere whose volume is
    equal to the mean volume per atom in a solid. This parameter is
    often used to calculate the coupling parameter.
    When ion density is used, this is the ion sphere radius, i.e., the
    space occupied by a single ion with no other ions in that space. Higher
    density means less space for each ion, so the radius is smaller.

    Parameters
    ----------
    n : `~astropy.units.Quantity`
        Particle number density.

    Returns
    -------
    radius : `~astropy.units.Quantity`
        The Wigner-Seitz radius in meters.

    Raises
    ------
    `TypeError`
        If argument is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If argument is in incorrect units.

    `ValueError`
        If argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    See Also
    --------
    ~plasmapy.formulary.quantum.Fermi_energy

    Notes
    -----
    The Wigner-Seitz radius approximates the interparticle spacing.
    It is the radius of a sphere whose volume is equal to the mean
    volume per atom in a solid:

    .. math::
        r = \left(\frac{3}{4 π n}\right)^{1/3}

    Examples
    --------
    >>> import astropy.units as u
    >>> Wigner_Seitz_radius(1e29 * u.m**-3)
    <Quantity 1.33650462e-10 m>
    """
    return (3 / (4 * np.pi * n)) ** (1 / 3)


@validate_quantities(
    n_e={"can_be_negative": False},
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def chemical_potential(
    n_e: u.Quantity[u.m**-3], T: u.Quantity[u.K]
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the ideal chemical potential.

    Parameters
    ----------
    n_e : `~astropy.units.Quantity`
        Electron number density.

    T : `~astropy.units.Quantity`
        The temperature.

    Returns
    -------
    beta_mu : `~astropy.units.Quantity`
        The dimensionless ideal chemical potential. That is the ratio of
        the ideal chemical potential to the thermal energy.

    Raises
    ------
    `TypeError`
        If argument is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If argument is in incorrect units.

    `ValueError`
        If argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The ideal chemical potential is implicitly given by Eq. 1.2 in
    :cite:p:`bonitz:1998`\ :

    .. math::
        χ = nΛ^{3} = I_{1/2}(β μ^{ideal})

    where :math:`χ` is the degeneracy parameter, :math:`n` is the
    species number density, :math:`Λ` is the thermal de Broglie
    wavelength, :math:`I_{1/2}` is the Fermi integral with order 1/2,
    :math:`β` is the inverse thermal energy :math:`β = 1/(k_B T)`, and
    :math:`μ^{ideal}` is the ideal chemical potential.

    The definition for the ideal chemical potential is implicit, so it
    must be obtained numerically by solving for the Fermi integral for
    values of chemical potential approaching the degeneracy parameter.
    Since values returned from the
    `~plasmapy.formulary.mathematics.Fermi_integral` are complex, the
    Broyden–Fletcher–Goldfarb–Shanno algorithm is used to iteratively
    approach a value of :math:`μ^{ideal}` which minimizes
    :math:`\lvert I_{1/2}(β μ^{ideal}) - χ \rvert`.

    This function returns the dimensionless ideal chemical potential
    :math:`β μ^{ideal}`.

    Examples
    --------
    >>> import astropy.units as u
    >>> chemical_potential(n_e=1e25 * u.cm**-3, T=11000 * u.K)
    <Quantity 283.43506297>
    """

    # deBroglie wavelength
    lambdaDB = thermal_deBroglie_wavelength(T)
    # degeneracy parameter
    degen = (n_e * lambdaDB**3).to(u.dimensionless_unscaled)

    def residual(params, data):
        """Residual function for fitting parameters to Fermi_integral."""
        alpha = params["alpha"].value
        # note that alpha = mu / (k_B * T)
        model = mathematics.Fermi_integral(alpha, 0.5)
        complexResidue = abs(data - model)
        return complexResidue  # noqa: RET504

    # setting parameters for fitting along with bounds
    alphaGuess = 1 * u.dimensionless_unscaled
    params = Parameters()
    params.add("alpha", value=alphaGuess, min=0.0)
    # calling minimize function from lmfit to fit by minimizing the residual
    data = np.array(degen)  # result of Fermi_integral - degen should be zero
    minFit = minimize(residual, params, args=(data,), method="bfgsb")
    beta_mu = minFit.params["alpha"].value * u.dimensionless_unscaled

    return beta_mu  # noqa: RET504


def _chemical_potential_interp(n_e, T):
    r"""
    Fitting formula for interpolating chemical potential between classical
    and quantum regimes.

    See [1]_, [2]_ for more information.

    Parameters
    ----------
    n_e : `~astropy.units.Quantity`
        Electron number density.

    T : `~astropy.units.Quantity`
        Temperature in units of temperature or energy.

    Returns
    -------
    beta_mu : `~astropy.units.Quantity`
        The dimensionless chemical potential, which is a ratio of
        chemical potential energy to thermal kinetic energy.

    Raises
    ------
    `TypeError`
        If argument is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If argument is in incorrect units.

    `ValueError`
        If argument contains invalid values.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Notes
    -----
    The ideal chemical potential is given by [1]_:

    .. math::
        \frac{μ}{k_B T_e} = - \frac{3}{2} \ln Θ + \ln
        \frac{4}{3 \sqrt{π}} +
        \frac{A Θ^{-b - 1} + B Θ^{-(b + 1) / 2}}{1 + A Θ^{-b}}

    where

    .. math::
        Θ = \frac{k_B T_e}{E_F}

    is the degeneracy parameter, comparing the thermal energy to the
    Fermi energy, and the coefficients for the fitting formula are
    :math:`A = 0.25945`\ , :math:`B = 0.0072`\ , and :math:`b = 0.858`\ .

    References
    ----------
    .. [1] Ichimaru, Statistical Plasma Physics Addison-Wesley,
       Reading, MA, 1991.

    .. [2] Gregori, G., et al. "Theoretical model of x-ray scattering as a
       dense matter probe." Physical Review E 67.2 (2003): 026412.

    Examples
    --------
    >>> import astropy.units as u
    >>> _chemical_potential_interp(n_e=1e23 * u.cm**-3, T=11000 * u.K)
    <Quantity 8.17649>
    """
    A = 0.25945
    B = 0.072
    b = 0.858
    theta = k_B * T / Fermi_energy(n_e)
    term1 = -3 / 2 * np.log(theta)
    term2 = np.log(4 / (3 * np.sqrt(np.pi)))
    term3num = A * theta ** (-b - 1) + B * theta ** (-(b + 1) / 2)
    term3den = 1 + A * theta ** (-b)
    term3 = term3num / term3den
    beta_mu = term1 + term2 + term3
    return beta_mu.to(u.dimensionless_unscaled)


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def quantum_theta(
    T: u.Quantity[u.K], n_e: u.Quantity[u.m**-3]
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Compare Fermi energy to thermal kinetic energy to check if quantum
    effects are important.

    The quantum theta (:math:`θ`) of a plasma is defined by

    .. math::
        θ = \frac{E_T}{E_F}

    where :math:`E_T` is the thermal energy of the plasma
    and :math:`E_F` is the Fermi energy of the plasma.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The temperature of the plasma.

    n_e : `~astropy.units.Quantity`
          The electron number density of the plasma.

    Returns
    -------
    theta : `~astropy.units.Quantity`

    See Also
    --------
    ~plasmapy.formulary.quantum.Fermi_energy

    Notes
    -----
    The thermal energy of the plasma (:math:`E_T`) is defined by

    .. math::
        E_T = k_B T

    where :math:`k_B` is the Boltzmann constant
    and :math:`T` is the temperature of the plasma.

    Examples
    --------
    >>> import astropy.units as u
    >>> quantum_theta(1 * u.eV, 1e20 * u.m**-3)
    <Quantity 127290.619...>
    >>> quantum_theta(1 * u.eV, 1e16 * u.m**-3)
    <Quantity 59083071...>
    >>> quantum_theta(1 * u.eV, 1e26 * u.m**-3)
    <Quantity 12.72906...>
    >>> quantum_theta(1 * u.K, 1e26 * u.m**-3)
    <Quantity 0.00109...>
    """
    fermi_energy = Fermi_energy(n_e)
    thermal_energy = k_B * T
    return thermal_energy / fermi_energy
