"""
Calculation of Coulomb logarithms and cross-sections.

Coulomb collisions are collisions where the interaction force is
conveyed via the electric field, instead of any kind of contact force.
These collisions  usually result in relatively small deflections of
particle trajectories. However, given that there are many charged
particles in a plasma, one has to take into account the cumulative
effects of many such collisions.

This module includes functions for handling Coulomb collisions ranging
from weakly coupled (low density) to strongly coupled (high density)
regimes.
"""

__all__ = [
    "Coulomb_logarithm",
    "Coulomb_cross_section",
]

import warnings
from typing import Literal

import astropy.units as u
import numpy as np

from plasmapy import particles
from plasmapy.formulary.collisions import lengths
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import CouplingWarning


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    V={"none_shall_pass": True},
)
@particles.particle_input
def Coulomb_logarithm(
    T: u.Quantity[u.K],
    n_e: u.Quantity[u.m**-3],
    species: (particles.Particle, particles.Particle),
    z_mean: float = np.nan,
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    method: Literal[
        "classical",
        "GMS-1",
        "GMS-2",
        "GMS-3",
        "GMS-4",
        "GMS-5",
        "GMS-6",
        "hls_full_interp",
        "hls_max_interp",
        "hls_min_interp",
        "ls",
        "ls_clamp_mininterp",
        "ls_full_interp",
        "ls_min_interp",
    ] = "classical",
):
    r"""
    Compute the Coulomb logarithm.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Temperature in units of temperature or energy per particle,
        which is assumed to be equal for both the test particle and the
        target particle.

    n_e : `~astropy.units.Quantity`
        The electron number density in units convertible to m\ :sup:`-3`.

    species : `tuple`
        A tuple containing string representations of the test particle
        (listed first) and the target particle (listed second).

    z_mean : float, optional
        The average ionization (arithmetic mean) of a plasma for which a
        macroscopic description is valid. This parameter is used to
        compute the average ion density (given the average ionization
        and electron density) for calculating the ion sphere radius for
        non-classical impact parameters. ``z_mean`` is a required
        parameter if ``method`` is ``"ls_full_interp"``,
        ``"hls_max_interp"``, or ``"hls_full_interp"``.

    V : `~astropy.units.Quantity`, optional
        The relative velocity between particles. If not provided,
        thermal velocity is assumed: :math:`μ V^2 \sim 2 k_B T` where
        :math:`μ` is the reduced mass.

    method : `str`, optional
        The method by which to compute the Coulomb logarithm.  The
        default method is the classical straight-line Landau-Spitzer
        method (``"classical"`` or ``"ls"``). The other six supported
        methods are ``"ls_min_interp"``, ``"ls_full_interp"``,
        ``"ls_clamp_mininterp"``, ``"hls_min_interp"``,
        ``"hls_max_interp"``, and ``"hls_full_interp"``.  Please refer
        to the Notes section of this docstring for more information,
        including about abbreviated aliases of these names.

    Returns
    -------
    ln_Lambda : `float` or `numpy.ndarray`
        The dimensionless Coulomb logarithm.

    Raises
    ------
    `ValueError`
        If the mass or charge of either particle cannot be found, or any
        of the inputs contain incorrect values.

    `~astropy.units.UnitConversionError`
        If the units on any of the inputs are incorrect, or if any of
        ``n_e``, ``T``, or ``V`` is not a `~astropy.units.Quantity`.

    `~plasmapy.utils.exceptions.PhysicsError`
        If the result is smaller than 1.

    `~plasmapy.utils.exceptions.RelativityError`
        If the input velocity is same or greater than the speed of
        light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the input velocity is greater than 5% of the speed of light.

    See Also
    --------
    ~plasmapy.formulary.collisions.lengths.impact_parameter : Computes
        :math:`b_{min}` and :math:`b_{max}`.

    Notes
    -----
    **Summary of Supported Methods of Computing the Coulomb Logarithm**

    PlasmaPy supports seven methods of computing the Coulomb logarithm:

    1. ``"classical"`` or ``"ls"``
    2. ``"ls_min_interp"`` or ``"GMS-1"``
    3. ``"ls_full_interp"`` or ``"GMS-2"``
    4. ``"ls_clamp_mininterp"`` or ``"GMS-3"``
    5. ``"hls_min_interp"`` or ``"GMS-4"``
    6. ``"hls_max_interp"`` or ``"GMS-5"``
    7. ``"hls_full_interp"`` or ``"GMS-6"``

    Options 1–4 are straight-line Landau-Spitzer (``"ls..."``) methods
    in which the trajectory of a Coulomb collision is modeled as a
    straight line. For the straight-line Landau-Spitzer methods, the
    Coulomb logarithm (:math:`\ln{Λ}`) is defined to be:

    .. math::

        \ln{Λ} ≡ \ln\left( \frac{b_{max}}{b_{min}} \right)

    Options 5–7 are hyperbolic Landau-Spitzer (``"hls..."``) methods in
    which the trajectory of a Coulomb collision is modeled as a
    hyperbola. For the hyperbolic Landau-Spitzer methods, the Coulomb
    logarithm (:math:`\ln{Λ}`) is defined to be:

    .. math::

        \ln{Λ} ≡ \frac{1}{2} \ln\left(1 + \frac{b_{max}^2}{b_{min}^2} \right)

    For all 7 methods, :math:`b_{min}` and :math:`b_{max}` are the inner
    impact parameter and the outer impact parameter, respectively, for
    Coulomb collisions :cite:p:`spitzer:1962`\ ; :math:`b_{min}` and
    :math:`b_{max}` are each computed by
    `~plasmapy.formulary.collisions.lengths.impact_parameter`.

    The abbreviations of Options 2–7 (``"GMS-..."``) refer to the first
    initials of the three authors of :cite:t:`gericke:2002`.

    .. note::

        For strongly-coupled plasma, PlasmaPy recommends Option 7,
        ``"hls_full_interp"`` or ``"GMS-6"``, because of its high
        accuracy regardless of a plasma's strength of coupling.

    **Explanation of Supported Methods of Computing the Coulomb Logarithm**

    In this section, further information about each method, such as
    about interpolation and other special features, is documented.
    Please refer to :cite:t:`spitzer:1962` and :cite:t:`gericke:2002`
    for additional information about these methods.

    Option 1: ``"classical"`` or ``"ls"`` (Landau-Spitzer)

        The classical straight-line Landau-Spitzer method in which
        :math:`b_{min}` is defined to be the higher of the de Broglie
        wavelength (:math:`λ_{de Broglie}`) and the distance of closest
        approach (:math:`ρ_⟂`) if they are not equal (and either of the
        two if they are equal) and :math:`b_{max}` is defined to be the
        Debye length (:math:`λ_{Debye}`).

        .. math::

            \ln{Λ} ≡ \ln\left( \frac{b_{max}}{b_{min}} \right)

        .. math::

            b_{min} ≡
            \left\{
                \begin{array}{ll}
                           λ_{de Broglie} & \mbox{if } λ_{de Broglie} ≥ ρ_⟂ \\
                           ρ_⟂         & \mbox{if } ρ_⟂ ≥ λ_{de Broglie}
                \end{array}
            \right.

        .. math::

            b_{max} ≡ λ_{Debye}

        The inner impact parameter (:math:`b_{min}`) is the higher of
        :math:`λ_{de Broglie}` and :math:`ρ_⟂` because for impact
        parameters lower than :math:`λ_{de Broglie}`, quantum effects
        cause the collision to be non-Coulombic
        :cite:p:`chen:2016,fundamenski:2007`.

        The outer impact parameter (:math:`b_{max}`) is defined to be
        the Debye length (:math:`λ_{Debye}`) because at distances higher
        than the Debye length, the electric fields created by other
        particles are screened out by the electrons rearranging
        themselves.

        The uncertainty of the classical straight-line Landau-Spitzer
        method is on the order of its reciprocal.

        This method is invalid if :math:`\ln{Λ} < 2` because of the
        uncertainty of this method and is invalid if :math:`\ln{Λ} < 0`,
        which may be true if the coupling parameter is high (such as for
        nonideal, dense, cold plasmas).

        Please refer to :cite:t:`spitzer:1962` for additional
        information about this method.

    Option 2: ``"ls_min_interp"`` or ``"GMS-1"`` (Landau-Spitzer,
    interpolation of :math:`b_{min}`)

        A straight-line Landau-Spitzer method in which :math:`b_{min}`
        is interpolated between the de Broglie wavelength (:math:`λ_{de
        Broglie}`) and the distance of closest approach (:math:`ρ_⟂`)
        and :math:`b_{max}` is defined to be the Debye length
        (:math:`λ_{Debye}`).

        .. math::

            \ln{Λ} ≡ \ln\left( \frac{b_{max}}{b_{min}} \right)

        .. math::

            b_{min} ≡ \sqrt{λ_{de Broglie}^2 + ρ_⟂^2}

        .. math::

            b_{max} ≡ λ_{Debye}

        This method is invalid if :math:`\ln{Λ} < 0`, which may be true
        if the coupling parameter is high (such as for nonideal, dense,
        cold plasmas).

        Note: This is the first method in Table 1 of
        :cite:t:`gericke:2002`.

    Option 3: ``"ls_full_interp"`` or ``"GMS-2"`` (Landau-Spitzer,
    interpolation of :math:`b_{min}` and :math:`b_{max}`)

        A straight-line Landau-Spitzer method in which :math:`b_{min}`
        and :math:`b_{max}` are each interpolated. :math:`b_{min}` is
        interpolated between the de Broglie wavelength (:math:`λ_{de
        Broglie}`) and the distance of closest approach (:math:`ρ_⟂`).
        :math:`b_{max}` is interpolated between the Debye length
        (:math:`λ_{Debye}`) and the ion sphere radius (:math:`a_i`).

        .. math::

            \ln{Λ} ≡ \ln\left( \frac{b_{max}}{b_{min}} \right)

        .. math::

            b_{min} ≡ \sqrt{λ_{de Broglie}^2 + ρ_⟂^2}

        .. math::

            b_{max} ≡ \sqrt{λ_{Debye}^2 + a_i^2}

        This method is invalid if :math:`\ln{Λ} < 0`, which may be true
        if the coupling parameter is high (such as for nonideal, dense,
        cold plasmas).

        Note: This is the second method in Table 1 of
        :cite:t:`gericke:2002`.

    Option 4: ``"ls_clamp_mininterp"`` or ``"GMS-3"`` (Landau-Spitzer
    with a clamp, interpolation of :math:`b_{min}`)

        A straight-line Landau-Spitzer method in which the value of
        :math:`\ln{Λ}` is clamped at a minimum of :math:`2` so that it
        is impossible for :math:`\ln{Λ} < 0` (which is possible by the
        classical Landau-Spitzer method). :math:`b_{min}` is
        interpolated between the de Broglie wavelength (:math:`λ_{de
        Broglie}`) and the distance of closest approach
        (:math:`ρ_⟂`). :math:`b_{max}` is defined to be the Debye length
        (:math:`λ_{Debye}`).

        .. math::

            \ln{Λ} ≡
            \left\{
               \begin{array}{ll}
                  \ln\left(\frac{b_{max}}{b_{min}} \right)
                     & \mbox{if } \ln\left( \frac{b_{max}}{b_{min}}\right)
                  ≥ 2 \\
                  2 & \mbox{if } \ln\left( \frac{b_{max}}{b_{min}} \right)
                  ≤ 2
                \end{array}
            \right.

        .. math::

            b_{min} ≡ \sqrt{λ_{de Broglie}^2 + ρ_⟂^2}

        .. math::

            b_{max} ≡ λ_{Debye}

        This method is valid for any plasma because it is impossible for
        :math:`\ln{Λ} < 0` by this method, even if the coupling
        parameter is high (such as for nonideal, dense, cold plasmas).

        Note: This is the third method in Table 1 of
        :cite:t:`gericke:2002`.

    Option 5: ``"hls_min_interp"`` or ``"GMS-4"`` (Hyperbolic
    Landau-Spitzer, interpolation of :math:`b_{min}`)

        A hyperbolic Landau-Spitzer method in which :math:`b_{min}` is
        interpolated between the de Broglie wavelength (:math:`λ_{de
        Broglie}`) and the distance of closest approach (:math:`ρ_⟂`)
        and :math:`b_{max}` is defined to be the Debye length
        (:math:`λ_{Debye}`).

        .. math::

            \ln{Λ} ≡ \frac{1}{2} \ln\left(1 +
            \frac{b_{max}^2}{b_{min}^2} \right)

        .. math::

            b_{min} ≡ \sqrt{λ_{de Broglie}^2 + ρ_⟂^2}

        .. math::

            b_{max} ≡ λ_{Debye}

        This method is valid for any plasma because it is impossible for
        :math:`\ln{Λ} < 0` by this method, even if the coupling
        parameter is high (such as for nonideal, dense, cold plasmas).

        Note: This is the fourth method in Table 1 of
        :cite:t:`gericke:2002`.

    Option 6: ``"hls_max_interp"`` or ``"GMS-5"`` (Hyperbolic
    Landau-Spitzer, interpolation of :math:`b_{max}`)

        A hyperbolic Landau-Spitzer method in which :math:`b_{max}` is
        interpolated between the Debye length (:math:`λ_{Debye}`) and
        the ion sphere radius (:math:`a_i`) and :math:`b_{min}` is
        defined to be the distance of closest approach (:math:`ρ_⟂`).

        .. math::

            \ln{Λ} ≡ \frac{1}{2} \ln\left(1 +
            \frac{b_{max}^2}{b_{min}^2} \right)

        .. math::

            b_{min} ≡ ρ_⟂

        .. math::

            b_{max} ≡ \sqrt{λ_{Debye}^2 + a_i^2}

        This method is valid for any plasma because it is impossible for
        :math:`\ln{Λ} < 0` by this method, even if the coupling
        parameter is high (such as for nonideal, dense, cold plasmas).

        Caution: This method overestimates :math:`\ln{Λ}` at high
        temperatures.

        Note: This is the fifth method in Table 1 of
        :cite:t:`gericke:2002`.

    Option 7: ``"hls_full_interp"`` or ``"GMS-6"`` (Hyperbolic
    Landau-Spitzer, interpolation of :math:`b_{min}` and
    :math:`b_{max}`)

        A hyperbolic Landau-Spitzer method in which :math:`b_{min}` and
        :math:`b_{max}` are each interpolated. :math:`b_{min}` is
        interpolated between the de Broglie wavelength (:math:`λ_{de
        Broglie}`) and the distance of closest approach (:math:`ρ_⟂`).
        :math:`b_{max}` is interpolated between the Debye length
        (:math:`λ_{Debye}`) and the ion sphere radius (:math:`a_i`).

        .. math::

            \ln{Λ} ≡ \frac{1}{2} \ln\left(1 +
            \frac{b_{max}^2}{b_{min}^2} \right)

        .. math::

            b_{min} ≡ \sqrt{λ_{de Broglie}^2 + ρ_⟂^2}

        .. math::

            b_{max} ≡ \sqrt{λ_{Debye}^2 + a_i^2}

        This method is valid for any plasma because it is impossible for
        :math:`\ln{Λ} < 0` by this method, even if the coupling
        parameter is high (such as for nonideal, dense, cold plasmas).

        Note: This is the sixth method in Table 1 of
        :cite:t:`gericke:2002`.

    Examples
    --------
    >>> import astropy.units as u
    >>> n_e = 1e19 * u.m**-3
    >>> T = 1e6 * u.K
    >>> Coulomb_logarithm(T, n_e, ('e-', 'p+'))
    np.float64(14.545527556...)
    >>> Coulomb_logarithm(T, n_e, ('e-', 'p+'), V = 1e6 * u.m / u.s)
    np.float64(11.363478378...)

    See Also
    --------
    ~plasmapy.formulary.collisions.lengths.impact_parameter : Computes
        :math:`b_{min}` and :math:`b_{max}`.
    """
    # fetching impact min and max impact parameters
    bmin, bmax = lengths.impact_parameter(
        T=T, n_e=n_e, species=species, z_mean=z_mean, V=V, method=method
    )

    if method in {
        "classical",
        "ls",
        "ls_min_interp",
        "GMS-1",
        "ls_full_interp",
        "GMS-2",
    }:
        ln_Lambda = np.log(bmax / bmin)
    elif method in {"ls_clamp_mininterp", "GMS-3"}:
        ln_Lambda = np.log(bmax / bmin)
        if np.any(ln_Lambda < 2):
            if np.isscalar(ln_Lambda.value):
                ln_Lambda = 2 * u.dimensionless_unscaled
            else:
                ln_Lambda[ln_Lambda < 2] = 2 * u.dimensionless_unscaled
    elif method in {
        "hls_min_interp",
        "GMS-4",
        "hls_max_interp",
        "GMS-5",
        "hls_full_interp",
        "GMS-6",
    }:
        ln_Lambda = 0.5 * np.log(1 + bmax**2 / bmin**2)
    else:
        raise ValueError(
            'Unknown method. Choose from "classical", "ls_min_interp", '
            '"ls_full_interp", "ls_clamp_mininterp", "hls_min_interp", '
            '"hls_max_interp", "hls_full_interp", and their aliases. '
            "Please refer to the documentation of this function for "
            "more information."
        )

    ln_Lambda = ln_Lambda.to(u.dimensionless_unscaled).value

    min_ln_Lambda = np.nanmin(ln_Lambda)

    if min_ln_Lambda < 2 and method in {
        "classical",
        "ls",
        "ls_min_interp",
        "GMS-1",
        "ls_full_interp",
        "GMS-2",
    }:
        warnings.warn(
            f"The calculation of the Coulomb logarithm has found a value of "
            f"min(ln Λ) = {min_ln_Lambda:.4f} which is likely to be inaccurate "
            f"due to strong coupling effects, in particular because "
            f"{method = } assumes weak coupling.",
            CouplingWarning,
        )
    elif min_ln_Lambda < 4:
        warnings.warn(
            f"The calculation of the Coulomb logarithm has found a value of "
            f"min(ln Λ) = {min_ln_Lambda:.4f}. Coulomb logarithms of ≲ 4 may "
            f"have increased uncertainty due to strong coupling effects.",
            CouplingWarning,
        )

    return ln_Lambda


@validate_quantities(impact_param={"can_be_negative": False})
def Coulomb_cross_section(impact_param: u.Quantity[u.m]) -> u.Quantity[u.m**2]:
    r"""
    Cross-section for a large angle Coulomb collision.

    Parameters
    ----------
    impact_param : `~astropy.units.Quantity`
        Impact parameter for the collision.

    Returns
    -------
    sigma : `~astropy.units.Quantity`
        The Coulomb collision cross section area.

    Notes
    -----
    The `collisional cross-section
    <https://en.wikipedia.org/w/index.php?title=Cross_section_(physics)&oldid=1037954726#Collision_among_gas_particles>`__
    for a 90° Coulomb collision is obtained by

    .. math::

        σ = π (2 * ρ_⟂)^2

    where :math:`ρ_⟂` is the distance of the closest approach for a 90°
    Coulomb collision. This function is a generalization of that
    calculation. Please note that it is not guaranteed to return the
    correct results for small angle collisions.

    Examples
    --------
    >>> Coulomb_cross_section(7e-10 * u.m)
    <Quantity 6.157...e-18 m2>
    >>> Coulomb_cross_section(0.5 * u.m)
    <Quantity 3.141... m2>
    """
    return np.pi * (2 * impact_param) ** 2
