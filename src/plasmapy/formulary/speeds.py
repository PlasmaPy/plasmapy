"""Fundamental speeds in plasmas."""

__all__ = [
    "Alfven_speed",
    "kappa_thermal_speed",
    "ion_sound_speed",
    "thermal_speed",
    "thermal_speed_coefficients",
    "thermal_speed_lite",
]
__aliases__ = ["cs_", "va_", "vth_", "vth_kappa_"]
__lite_funcs__ = ["thermal_speed_lite"]

import warnings
from numbers import Real
from typing import Literal

import astropy.units as u
import numpy as np
from astropy.constants.si import k_B, mu0

from plasmapy.formulary import lengths
from plasmapy.particles import electron
from plasmapy.particles.atomic import particle_mass
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.utils.decorators import (
    bind_lite_func,
    check_relativistic,
    preserve_signature,
    validate_quantities,
)
from plasmapy.utils.exceptions import PhysicsError, PhysicsWarning

__all__ += __aliases__ + __lite_funcs__

k_B_si_unitless = k_B.value


@particle_input
@check_relativistic
@validate_quantities(density={"can_be_negative": False})
def Alfven_speed(
    B: u.Quantity[u.T],
    density: (u.m**-3, u.kg / u.m**3),
    ion: ParticleLike | None = None,
    *,
    mass_numb: int | None = None,
    Z: float | None = None,
) -> u.Quantity[u.m / u.s]:
    r"""Calculate the Alfvén speed 🏎️💨.

    The Alfvén speed :math:`V_A` is the typical propagation speed of
    magnetic disturbances in a quasineutral plasma, and is given by:

    .. math::

        V_A = \frac{B}{\sqrt{μ_0 ρ}}

    where :math:`B` is the magnetic field and :math:`ρ = n_i m_i + n_e
    m_e` is the total mass density (:math:`n_i` is the ion number
    density, :math:`n_e ≡ Z n_i` is the electron number density
    assuming quasineutrality, :math:`m_i` is the ion mass, and
    :math:`m_e` is the electron mass) :cite:p:`alfven:1942`.

    **Aliases:** `va_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to tesla.

    density : `~astropy.units.Quantity`
        Either the ion number density :math:`n_i` in units convertible
        to m\ :sup:`-3` or the total mass density :math:`ρ` in units
        convertible to kg m\ :sup:`-3`\ .

    ion : `~plasmapy.particles.particle_class.Particle`, optional
        Representation of the ion species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 1+'`` for singly ionized
        helium-4, etc.). If the density is an ion number density, then
        this parameter is required in order to convert to mass
        density.

    mass_numb : `int`, |keyword-only|, optional
        The mass number corresponding to ``ion``.

    Z : `float`, |keyword-only|, optional
        The charge number corresponding to ``ion``. Note that the
        value of ``Z`` does not impact the Alfvén speed.

    Returns
    -------
    V_A : `~astropy.units.Quantity`
        The Alfvén speed in units of m/s.

    Raises
    ------
    `~plasmapy.utils.exceptions.RelativityError`
        If the Alfvén velocity is greater than or equal to the speed
        of light.

    `TypeError`
        If ``B`` and/or ``density`` are not of type
        `~astropy.units.Quantity`, or convertible.

    `TypeError`
        If ``ion`` is not |particle-like|.

    `TypeError`
        If ``z_mean`` is not of type `int` or `float`.

    `~astropy.units.UnitTypeError`
        If the magnetic field ``B`` does not have units equivalent to
        tesla.

    `~astropy.units.UnitTypeError`
        If the ``density`` does not have units equivalent to a number
        density or mass density.

    `ValueError`
        If ``density`` is negative.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the Alfvén velocity exceeds 5% of the speed of light.

    : `~astropy.units.UnitsWarning`
        If units are not provided for the magnetic field ``B``, units
        of tesla are assumed.

    Notes
    -----
    This expression does not account for relativistic effects, and
    loses validity when the resulting speed is a significant fraction
    of the speed of light.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.constants.si import m_p, m_e
    >>> B = 0.014 * u.T
    >>> n = 5e19 * u.m**-3
    >>> ion = "p+"
    >>> rho = n * (m_p + m_e)
    >>> Alfven_speed(B=B, density=n, ion=ion)
    <Quantity 43173.870... m / s>
    >>> Alfven_speed(B=B, density=rho)
    <Quantity 43173.870... m / s>
    >>> Alfven_speed(B=B, density=rho).to(u.cm / u.us)
    <Quantity 4.317387 cm / us>
    >>> Alfven_speed(B=B, density=n, ion="He-4 2+")
    <Quantity 21664.18... m / s>
    >>> Alfven_speed(B=B, density=n, ion="He", Z=1.8)
    <Quantity 21664.18... m / s>
    """

    if density.unit.physical_type == u.physical.mass_density and ion is not None:
        raise ValueError(
            "When calculating the Alfvén speed, an ion cannot be specified "
            "when the 'density' parameter is provided an argument with a "
            "physical type of mass density."
        )

    if density.unit.physical_type == u.physical.number_density and ion is None:
        raise ValueError(
            "When calculating the Alfvén speed, the ion must be specified "
            "when 'density' has a physical type of number density."
        )

    if density.unit.physical_type == u.physical.mass_density:
        rho = density
    else:
        rho = (ion.mass + ion.charge_number * electron.mass) * density

    return np.abs(B) / np.sqrt(mu0 * rho)


va_ = Alfven_speed
"""Alias to `~plasmapy.formulary.speeds.Alfven_speed`."""


@check_relativistic
@validate_quantities(
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False, "none_shall_pass": True},
    k={"can_be_negative": False, "none_shall_pass": True},
)
@particle_input
def ion_sound_speed(
    T_e: u.Quantity[u.K],
    T_i: u.Quantity[u.K],
    ion: ParticleLike,
    n_e: u.Quantity[u.m**-3] = None,
    k: u.Quantity[u.m**-1] = None,
    gamma_e: float = 1,
    gamma_i: float = 3,
    Z=None,
) -> u.Quantity[u.m / u.s]:
    r"""
    Return the ion sound speed for an electron-ion plasma.

    **Aliases:** `cs_`

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        Electron temperature in units of temperature or energy per
        particle. If this is not given, then the electron temperature
        is assumed to be zero.

    T_i : `~astropy.units.Quantity`
        Ion temperature in units of temperature or energy per
        particle.  If this is not given, then the ion temperature is
        assumed to be zero.

    ion : |particle-like|
        Representation of the ion species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, or ``'He-4 +1'`` for singly ionized
        helium-4).

    n_e : `~astropy.units.Quantity`
        Electron number density. If this is not given, then ion_sound_speed
        will be approximated in the non-dispersive limit
        (:math:`k^2 λ_{D}^2` will be assumed zero). If ``n_e`` is given,
        a value for ``k`` must also be given.

    k : `~astropy.units.Quantity`
        Wavenumber (in units of inverse length, e.g. m\ :sup:`-1`\ ). If this
        is not given, then ion_sound_speed will be approximated in the
        non-dispersive limit (:math:`k^2 λ_{D}^2` will be assumed zero).
        If ``k`` is given, a value for ``n_e`` must also be given.

    gamma_e : `float` or `int`
        The adiabatic index for electrons, which defaults to 1.  This
        value assumes that the electrons are able to equalize their
        temperature rapidly enough that the electrons are effectively
        isothermal.

    gamma_i : `float` or `int`
        The adiabatic index for ions, which defaults to 3.  This value
        assumes that ion motion has only one degree of freedom, namely
        along magnetic field lines.

    Z : `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) for a plasma where
        a macroscopic description is valid. If this quantity is not
        given then the charge number of the ion
        is used. This is effectively an average ion sound speed for the
        plasma where multiple charge states are present.

    mass_numb : positive integer, optional
        The mass number of an isotope corresponding to ``ion``.

    Returns
    -------
    V_S : `~astropy.units.Quantity`
        The ion sound speed in units of meters per second.

    Raises
    ------
    `TypeError`
        If any of the arguments are not entered as keyword arguments
        or are of an incorrect type.

    `ValueError`
        If the ion mass, adiabatic index, or temperature are invalid.

    `~plasmapy.utils.exceptions.PhysicsError`
        If an adiabatic index is less than one.

    `~astropy.units.UnitConversionError`
        If the temperature, electron number density, or wavenumber
        is in incorrect units.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the ion sound speed exceeds 5% of the speed of light.

    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    : `~plasmapy.utils.exceptions.PhysicsWarning`
        If only one of ``k`` or ``n_e`` is given, the non-dispersive
        limit is assumed.

    Notes
    -----
    The ion sound speed :math:`V_S` is given by

    .. math::

        V_S = \sqrt{\frac{γ_e Z k_B T_e + γ_i k_B T_i}{m_i (1 + k^2 λ_{D}^2)}}

    where :math:`γ_e` and :math:`γ_i` are the electron and
    ion adiabatic indices, :math:`k_B` is the Boltzmann constant,
    :math:`T_e` and :math:`T_i` are the electron and ion temperatures,
    :math:`Z` is the charge state of the ion, :math:`m_i` is the
    ion mass, :math:`λ_D` is the Debye length, and :math:`k` is the
    wavenumber.

    In the non-dispersive limit (:math:`k^2 λ_D^2` is small) the
    equation for :math:`V_S` is approximated (the denominator reduces
    to :math:`m_i`).

    When the electron temperature is much greater than the ion
    temperature, the ion sound velocity reduces to
    :math:`\sqrt{γ_e k_B T_e / m_i}`. Ion acoustic waves can
    therefore occur even when the ion temperature is zero.

    Examples
    --------
    >>> import astropy.units as u
    >>> n = 5e19 * u.m**-3
    >>> k_1 = 3e1 * u.m**-1
    >>> k_2 = 3e7 * u.m**-1
    >>> ion_sound_speed(
    ...     T_e=5e6 * u.K,
    ...     T_i=0 * u.K,
    ...     ion="p+",
    ...     gamma_e=1,
    ...     gamma_i=3,
    ... )
    <Quantity 203155... m / s>
    >>> ion_sound_speed(
    ...     T_e=5e6 * u.K,
    ...     T_i=0 * u.K,
    ...     n_e=n,
    ...     k=k_1,
    ...     ion="p+",
    ...     gamma_e=1,
    ...     gamma_i=3,
    ... )
    <Quantity 203155... m / s>
    >>> ion_sound_speed(
    ...     T_e=5e6 * u.K,
    ...     T_i=0 * u.K,
    ...     n_e=n,
    ...     k=k_2,
    ...     ion="p+",
    ...     gamma_e=1,
    ...     gamma_i=3,
    ... )
    <Quantity 310.31... m / s>
    >>> ion_sound_speed(T_e=5e6 * u.K, T_i=0 * u.K, n_e=n, k=k_1, ion="p+")
    <Quantity 203155... m / s>
    >>> ion_sound_speed(T_e=500 * u.eV, T_i=200 * u.eV, n_e=n, k=k_1, ion="D+")
    <Quantity 229585... m / s>
    """
    for gamma, species in zip([gamma_e, gamma_i], ["electrons", "ions"], strict=False):
        if not isinstance(gamma, Real):
            raise TypeError(
                f"The adiabatic index gamma for {species} must be a positive "
                f"number greater than one."
            )
        if gamma < 1:
            raise PhysicsError(
                f"The adiabatic index for {species} must be a positive "
                f"number greater than one."
            )

    # Assume non-dispersive limit if values for n_e (or k) are not specified
    klD2 = 0.0
    if (n_e is None) ^ (k is None):
        warnings.warn(
            "The non-dispersive limit has been assumed for "
            "ion_sound_speed. To prevent this, values must "
            "be specified for both n_e and k.",
            PhysicsWarning,
        )
    elif n_e is not None and k is not None:
        lambda_D = lengths.Debye_length(T_e, n_e)
        klD2 = (k * lambda_D) ** 2

    try:
        V_S_squared = (
            gamma_e * ion.charge_number * k_B * T_e + gamma_i * k_B * T_i
        ) / (ion.mass * (1 + klD2))
        V_S = np.sqrt(V_S_squared).to(u.m / u.s)
    except ValueError as ex:
        raise ValueError("Unable to find ion sound speed.") from ex

    return V_S


cs_ = ion_sound_speed
"""Alias to `~plasmapy.formulary.speeds.ion_sound_speed`."""


def thermal_speed_coefficients(method: str, ndim: int) -> float:
    r"""
    Get the thermal speed coefficient corresponding to the desired
    thermal speed definition.

    See the `~plasmapy.formulary.speeds.thermal_speed`
    :ref:`Notes <thermal-speed-notes>` section for further details of
    the various thermal speed definitions.

    Parameters
    ----------
    method : `str`
        Method to be used for calculating the thermal speed. Valid
        values are ``"most_probable"``, ``"rms"``, ``"mean_magnitude"``,
        and ``"nrl"``.

    ndim : `int`
        Dimensionality (1D, 2D, 3D) of space in which to calculate
        thermal speed. Valid values are ``1``, ``2``, or ``3``.

    Raises
    ------
    `ValueError`
        If ``method`` or ``ndim`` are not a valid value.

    Notes
    -----
    For a detailed explanation of the different coefficients used to
    calculate the thermal speed, then look to the
    :ref:`Notes <thermal-speed-notes>` section for
    `~plasmapy.formulary.speeds.thermal_speed`.  The possible return
    values are listed the following table:

    .. table:: Thermal speed :math:`v_{th}` coefficients.
       :widths: 2 1 1 1 1
       :width: 100%

       +--------------+------------+---------------+---------------+---------------+
       | ↓ **method** | **ndim** → | ``1``         | ``2``         | ``3``         |
       +--------------+------------+---------------+---------------+---------------+
       | ``"most_probable"``       | .. math::     | .. math::     | .. math::     |
       |                           |    0          |    1          |    \sqrt{2}   |
       +--------------+------------+---------------+---------------+---------------+
       | ``"rms"``                 | .. math::     | .. math::     | .. math::     |
       |                           |    1          |    \sqrt{2}   |    \sqrt{3}   |
       +--------------+------------+---------------+---------------+---------------+
       | ``"mean_magnitude"``      | .. math::     | .. math::     | .. math::     |
       |                           |    \sqrt{2/π} |    \sqrt{π/2} |    \sqrt{8/π} |
       +--------------+------------+---------------+---------------+---------------+
       | ``"nrl"``                 | .. math::                                     |
       |                           |    1                                          |
       +--------------+------------+---------------+---------------+---------------+

    Examples
    --------
    >>> thermal_speed_coefficients(method="most_probable", ndim=3)
    np.float64(1.414213...)
    """
    _coefficients = {
        (1, "most_probable"): 0,
        (2, "most_probable"): 1,
        (3, "most_probable"): np.sqrt(2),
        (1, "rms"): 1,
        (2, "rms"): np.sqrt(2),
        (3, "rms"): np.sqrt(3),
        (1, "mean_magnitude"): np.sqrt(2 / np.pi),
        (2, "mean_magnitude"): np.sqrt(np.pi / 2),
        (3, "mean_magnitude"): np.sqrt(8 / np.pi),
        (1, "nrl"): 1,
        (2, "nrl"): 1,
        (3, "nrl"): 1,
    }

    try:
        coeff = _coefficients[(ndim, method)]
    except KeyError as ex:
        raise ValueError(
            f"Value for (ndim, method) pair not valid, got '({ndim}, {method})'."
        ) from ex

    return coeff


@preserve_signature
def thermal_speed_lite(T: float, mass: float, coeff: float) -> float:
    r"""
    The :term:`lite-function` for
    `~plasmapy.formulary.speeds.thermal_speed`.  Performs the same
    thermal speed calculations as
    `~plasmapy.formulary.speeds.thermal_speed`, but is intended for
    computational use and, thus, has data conditioning safeguards
    removed.

    .. math::
        v_{th} = C_o \sqrt{\frac{k_B T}{m}}

    where :math:`T` is the temperature associated with the distribution,
    :math:`m` is the particle's mass, and :math:`C_o` is a constant of
    proportionality determined by the method in which :math:`v_{th}` is
    calculated and the dimensionality of the system (1D, 2D, 3D).  For
    further details see the :ref:`Notes <thermal-speed-notes>` section
    in the `~plasmapy.formulary.speeds.thermal_speed` documentation.

    Parameters
    ----------
    T : `float`
        The temperature of the particle distribution, in units of kelvin.

    mass : `float`
        Mass of the particle in kg.

    coeff : `float`
        The coefficient :math:`C_o` associated with the method used for
        calculating the thermal speed, see
        :ref:`Notes <thermal-speed-notes>` section in the
        `~plasmapy.formulary.speeds.thermal_speed` documentation.

    Returns
    -------
    vth : `float`
        Thermal speed of the Maxwellian distribution in units of m/s.

    Examples
    --------
    >>> from plasmapy.particles import Particle
    >>> mass = Particle("p").mass.value
    >>> coeff = thermal_speed_coefficients(method="most_probable", ndim=3)
    >>> thermal_speed_lite(T=1e6, mass=mass, coeff=coeff)
    np.float64(128486.57...)
    """
    return coeff * np.sqrt(k_B_si_unitless * T / mass)


@bind_lite_func(
    thermal_speed_lite,
    attrs={"coefficients": thermal_speed_coefficients},
)
@check_relativistic
@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    mass={"can_be_negative": False, "can_be_nan": True},
)
@particle_input
def thermal_speed(
    T: u.Quantity[u.K],
    particle: ParticleLike,
    method: Literal["most_probable", "rms", "mean_magnitude", "nrl"] = "most_probable",
    mass: u.Quantity[u.kg] = None,
    ndim: int = 3,
) -> u.Quantity[u.m / u.s]:
    r"""
    Calculate the speed of thermal motion for particles with a Maxwellian
    distribution.  (See the :ref:`Notes <thermal-speed-notes>` section for
    details.).

    **Aliases:** `~plasmapy.formulary.speeds.vth_`

    **Lite Version:** `~plasmapy.formulary.speeds.thermal_speed_lite`

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The temperature of the particle distribution, in units of kelvin or
        energy.

    particle : `~plasmapy.particles.particle_class.Particle`
        Representation of the particle species (e.g., ``"p"`` for protons,
        ``"D+"`` for deuterium, or ``"He-4 +1"`` for singly ionized helium-4).

    method : `str`, optional
        (Default ``"most_probable"``) Method to be used for calculating the
        thermal speed. Valid values are ``"most_probable"``, ``"rms"``,
        ``"mean_magnitude"``, and ``"nrl"``.

    mass : `~astropy.units.Quantity`
        Mass override in units convertible to kg.  If given, then ``mass`` will
        be used instead of the mass value associated with ``particle``.

    ndim : `int`
        (Default ``3``) Dimensionality (1D, 2D, 3D) of space in which to
        calculate thermal speed. Valid values are ``1``, ``2``, or ``3``.

    Returns
    -------
    vth : `~astropy.units.Quantity`
        Thermal speed of the Maxwellian distribution.

    Raises
    ------
    `TypeError`
        The particle temperature is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the particle temperature is not in units of temperature or
        energy per particle.

    `ValueError`
        The particle temperature is invalid or particle cannot be used to
        identify an isotope or particle.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the ion sound speed exceeds 5% of the speed of light.

    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.


    .. _thermal-speed-notes:

    Notes
    -----
    There are multiple methods (or definitions) for calculating the thermal
    speed, all of which give the expression

    .. math::
        v_{th} = C_o \sqrt{\frac{k_B T}{m}}

    where :math:`T` is the temperature associated with the distribution,
    :math:`m` is the particle's mass, and :math:`C_o` is a constant of
    proportionality determined by the method in which :math:`v_{th}` is
    calculated and the dimensionality of the system (1D, 2D, 3D).  The
    :math:`C_o` used for the ``thermal_speed`` calculation is determined from
    the input arguments ``method`` and ``ndim``, and the values can be seen in
    the table below:

    .. table:: Values for :math:`C_o`
       :widths: 2 1 1 1 1
       :width: 100%

       +--------------+------------+---------------+---------------+---------------+
       | ↓ **method** | **ndim** → | ``1``         | ``2``         | ``3``         |
       +--------------+------------+---------------+---------------+---------------+
       | ``"most_probable"``       | .. math::     | .. math::     | .. math::     |
       |                           |    0          |    1          |    \sqrt{2}   |
       +--------------+------------+---------------+---------------+---------------+
       | ``"rms"``                 | .. math::     | .. math::     | .. math::     |
       |                           |    1          |    \sqrt{2}   |    \sqrt{3}   |
       +--------------+------------+---------------+---------------+---------------+
       | ``"mean_magnitude"``      | .. math::     | .. math::     | .. math::     |
       |                           |    \sqrt{2/π} |    \sqrt{π/2} |    \sqrt{8/π} |
       +--------------+------------+---------------+---------------+---------------+
       | ``"nrl"``                 | .. math::                                     |
       |                           |    1                                          |
       +--------------+------------+---------------+---------------+---------------+

    The coefficients can be directly retrieved using
    `~plasmapy.formulary.speeds.thermal_speed_coefficients`.

        .. rubric:: The Methods

        In the following discussion the Maxwellian distribution
        :math:`f(\mathbf{v})` is assumed to be 3D, but similar expressions can
        be given for 1D and 2D.

        - **Most Probable** ``method = "most_probable"``

          This method expresses the thermal speed of the distribution by expressing
          it as the most probable speed a particle in the distribution may have.
          To do this we first define another function :math:`g(v)` given by

          .. math::
             \int_{0}^{∞} g(v) dv
                = \int_{-∞}^{∞} f(\mathbf{v}) d^3\mathbf{v}
                \quad \rightarrow \quad
                g(v) = 4 π v^2 f(v)

          then

          .. math::
             g^{\prime}(v_{th}) = \left.\frac{dg}{dv}\right|_{v_{th}} = 0\\
             \implies v_{th} = \sqrt{\frac{2 k_B T}{m}}

        - **Root Mean Square** ``method = "rms"``

          This method uses the root mean square to calculate an expression for
          the thermal speed of the particle distribution, which is given by

          .. math::
             v_{th} = \left[\int v^2 f(\mathbf{v}) d^3 \mathbf{v}\right]^{1/2}
                        = \sqrt{\frac{3 k_B T}{m}}

        - **Mean Magnitude** ``method = "mean_magnitude"``

          This method uses the mean speed of the particle distribution to
          calculate an expression for the thermal speed, which is given by

          .. math::
             v_{th} = \int |\mathbf{v}| f(\mathbf{v}) d^3 \mathbf{v}
                         = \sqrt{\frac{8 k_B T}{π m}}


        - **NRL Formulary** ``method = "nrl"``

          The NRL Plasma Formulary :cite:p:`nrlformulary`
          uses the square root of the Normal distribution's variance
          as the expression for thermal speed.

          .. math::
             v_{th} = σ = \sqrt{\frac{k_B T}{m}} \quad
             \text{where} \quad f(v) \sim e^{v^2 / 2 σ^2}

    Examples
    --------
    >>> import astropy.units as u
    >>> thermal_speed(5*u.eV, 'p+')
    <Quantity 30949.6... m / s>
    >>> thermal_speed(1e6*u.K, particle='p+')
    <Quantity 128486... m / s>
    >>> thermal_speed(5*u.eV, particle='e-')
    <Quantity 132620... m / s>
    >>> thermal_speed(1e6*u.K, particle='e-')
    <Quantity 550569... m / s>
    >>> thermal_speed(1e6*u.K, "e-", method="rms")
    <Quantity 674307... m / s>
    >>> thermal_speed(1e6*u.K, "e-", method="mean_magnitude")
    <Quantity 621251... m / s>

    For user convenience `~plasmapy.formulary.speeds.thermal_speed_coefficients`
    and `~plasmapy.formulary.speeds.thermal_speed_lite` are bound to this function
    and can be used as follows.

    >>> from plasmapy.particles import Particle
    >>> mass = Particle("p").mass.value
    >>> coeff = thermal_speed.coefficients(method="most_probable", ndim=3)
    >>> thermal_speed.lite(T=1e6, mass=mass, coeff=coeff)
    np.float64(128486.57...)
    """
    if mass is None:
        mass = particle_mass(particle)

    coeff = thermal_speed_coefficients(method=method, ndim=ndim)

    speed = thermal_speed_lite(T=T.value, mass=mass.value, coeff=coeff)
    return speed * u.m / u.s


vth_ = thermal_speed
"""Alias to :func:`~plasmapy.formulary.speeds.thermal_speed`."""


@check_relativistic
@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()}
)
@particle_input
def kappa_thermal_speed(
    T: u.Quantity[u.K],
    kappa,
    particle: ParticleLike,
    method: Literal["most_probable", "rms", "mean_magnitude"] = "most_probable",
    *,
    mass_numb: int | None = None,
    Z: float | None = None,
) -> u.Quantity[u.m / u.s]:
    r"""
    Return the most probable speed for a particle within a kappa
    distribution.

    **Aliases:** `vth_kappa_`

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The particle temperature in either kelvin or energy per particle

    kappa: `float`
        The ``kappa`` parameter is a dimensionless number which sets the slope
        of the energy spectrum of suprathermal particles forming the tail
        of the kappa velocity distribution function. ``kappa`` must be greater
        than 3/2.

    particle : |particle-like|
        Representation of the particle species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, or 'He-4 +1' for singly ionized helium-4).

    method : `str`, default: ``"most_probable"``
        Method to be used for calculating the thermal speed. Options are
        ``'most_probable'``, ``'rms'``, and ``'mean_magnitude'``.

    mass_numb : integer, optional
        The mass number corresponding to ``particle``.

    Z : real number, optional
        The charge number corresponding to ``particle``.

    Returns
    -------
    V : `~astropy.units.Quantity`
        Particle thermal speed.

    Raises
    ------
    `TypeError`
        The particle temperature is not a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If the particle temperature is not in units of temperature or
        energy per particle.

    `ValueError`
        The particle temperature is invalid or particle cannot be used to
        identify an isotope or particle.

    Warns
    -----
    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the particle thermal speed exceeds 5% of the speed of light.

    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    See Also
    --------
    ~plasmapy.formulary.speeds.kappa_thermal_speed
    ~plasmapy.formulary.distribution.kappa_velocity_1D

    Notes
    -----
    The particle thermal speed is given by:

    .. math::
        V_{th,i} = \sqrt{(2 κ - 3)\frac{2 k_B T_i}{κ m_i}}

    For more discussion on the ``'mean_magnitude'`` calculation method,
    see `PlasmaPy issue #186
    <https://github.com/PlasmaPy/PlasmaPy/issues/186>`__.

    Examples
    --------
    >>> import astropy.units as u
    >>> kappa_thermal_speed(5 * u.eV, 4, "p")  # defaults to most probable
    <Quantity 24467.87... m / s>
    >>> kappa_thermal_speed(5 * u.eV, 4, "p", "rms")
    <Quantity 37905.47... m / s>
    >>> kappa_thermal_speed(5 * u.eV, 4, "p", "mean_magnitude")
    <Quantity 34922.98... m / s>
    """
    # Checking thermal units
    if kappa <= 3 / 2:
        raise ValueError(
            f"Must have kappa > 3/2, instead of {kappa}, for "
            "kappa distribution function to be valid."
        )
    # different methods, as per https://en.wikipedia.org/wiki/Thermal_velocity
    vth = thermal_speed(T=T, particle=particle, method=method)

    coeff = np.sqrt((kappa - 3 / 2) / kappa) if method == "most_probable" else 1
    return vth * coeff


vth_kappa_ = kappa_thermal_speed
"""Alias to `~plasmapy.formulary.speeds.kappa_thermal_speed`."""
