"""Plasma dielectric parameters."""

__all__ = [
    "cold_plasma_permittivity_SDP",
    "cold_plasma_permittivity_LRP",
    "permittivity_1D_Maxwellian",
    "RotatingTensorElements",
    "StixTensorElements",
]
__lite_funcs__ = ["permittivity_1D_Maxwellian_lite"]

from collections import namedtuple
from collections.abc import Sequence

import astropy.units as u
import numpy as np

from plasmapy.dispersion.dispersion_functions import plasma_dispersion_func_deriv
from plasmapy.formulary.frequencies import gyrofrequency, plasma_frequency
from plasmapy.formulary.speeds import thermal_speed
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.particles.particle_collections import ParticleListLike
from plasmapy.utils.decorators import (
    bind_lite_func,
    preserve_signature,
    validate_quantities,
)

__all__ += __lite_funcs__

r"""
Values should be returned as a `~astropy.units.Quantity` in SI units.
"""

StixTensorElements = namedtuple("StixTensorElements", ["sum", "difference", "plasma"])
"""Output type for `~plasmapy.formulary.dielectric.cold_plasma_permittivity_SDP`."""


RotatingTensorElements = namedtuple(
    "RotatingTensorElements", ["left", "right", "plasma"]
)
"""Output type for `~plasmapy.formulary.dielectric.cold_plasma_permittivity_LRP`."""


@validate_quantities(B={"can_be_negative": False}, omega={"can_be_negative": False})
def cold_plasma_permittivity_SDP(
    B: u.Quantity[u.T],
    species: ParticleListLike,
    n: Sequence[u.Quantity[u.m**-3]] | u.Quantity[u.m**-3],
    omega: u.Quantity[u.rad / u.s],
) -> StixTensorElements:
    r"""
    Magnetized cold plasma dielectric permittivity tensor elements.

    Elements (S, D, P) are given in the "Stix" frame, i.e. with
    :math:`B ∥ \hat{z}` :cite:p:`stix:1992`.

    The :math:`\exp(-i ω t)` time-harmonic convention is assumed.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        Magnetic field magnitude in units convertible to tesla.

    species : |particle-list-like|
        List of the plasma particle species,
        e.g.: ``['e-', 'D+']`` or ``['e-', 'D+', 'He+']``.

    n : `list` of `~astropy.units.Quantity`
        `list` of species density in units convertible to per cubic meter
        The order of the species densities should follow species.

    omega : `~astropy.units.Quantity`
        Electromagnetic wave frequency in rad/s.

    Returns
    -------
    sum : `~astropy.units.Quantity`
        The "sum" dielectric tensor element, :math:`S`.

    difference : `~astropy.units.Quantity`
        The "difference" dielectric tensor element, :math:`D`.

    plasma : `~astropy.units.Quantity`
        The "plasma" dielectric tensor element, :math:`P`.

    Notes
    -----
    The dielectric permittivity tensor is expressed in the Stix frame
    with the :math:`\exp(-i ω t)` time-harmonic convention as
    :math:`ε = ε_0 A`, with :math:`A` being

    .. math::

        ε = ε_0 \left(\begin{matrix}  S & -i D & 0 \\
                              +i D & S & 0 \\
                              0 & 0 & P \end{matrix}\right)

    where:

    .. math::
        S = 1 - \sum_s \frac{ω_{p,s}^2}{ω^2 - Ω_{c,s}^2}

        D = \sum_s \frac{Ω_{c,s}}{ω}
            \frac{ω_{p,s}^2}{ω^2 - Ω_{c,s}^2}

        P = 1 - \sum_s \frac{ω_{p,s}^2}{ω^2}

    where :math:`ω_{p,s}` is the plasma frequency and
    :math:`Ω_{c,s}` is the signed version of the cyclotron frequency
    for the species :math:`s`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from numpy import pi
    >>> B = 2*u.T
    >>> species = ['e-', 'D+']
    >>> n = [1e18*u.m**-3, 1e18*u.m**-3]
    >>> omega = 3.7e9*(2*pi)*(u.rad/u.s)
    >>> permittivity = S, D, P = cold_plasma_permittivity_SDP(B, species, n, omega)
    >>> S
    <Quantity 1.02422...>
    >>> permittivity.sum   # namedtuple-style access
    <Quantity 1.02422...>
    >>> D
    <Quantity 0.39089...>
    >>> P
    <Quantity -4.8903...>
    """
    S, D, P = 1, 0, 1

    for s, n_s in zip(species, n, strict=False):
        omega_c = gyrofrequency(B=B, particle=s, signed=True)
        omega_p = plasma_frequency(n=n_s, particle=s)

        S += -(omega_p**2) / (omega**2 - omega_c**2)
        D += omega_c / omega * omega_p**2 / (omega**2 - omega_c**2)
        P += -(omega_p**2) / omega**2
    return StixTensorElements(S, D, P)


@validate_quantities(B={"can_be_negative": False}, omega={"can_be_negative": False})
def cold_plasma_permittivity_LRP(
    B: u.Quantity[u.T],
    species: ParticleListLike,
    n: list[u.Quantity[u.m**-3]] | u.Quantity[u.m**-3],
    omega: u.Quantity[u.rad / u.s],
) -> RotatingTensorElements:
    r"""
    Magnetized cold plasma dielectric permittivity tensor elements.

    Elements (L, R, P) are given in the "rotating" basis, i.e. in the basis
    :math:`(\mathbf{u}_{+}, \mathbf{u}_{-}, \mathbf{u}_z)`,
    where the tensor is diagonal and with :math:`B ∥ z`\ .

    The :math:`\exp(-i ω t)` time-harmonic convention is assumed.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        Magnetic field magnitude in units convertible to tesla.

    species : (k,) |particle-list-like|
        The plasma particle species (e.g.: ``['e-', 'D+']`` or
        ``['e-', 'D+', 'He+']``.

    n : (k,) `list` of `~astropy.units.Quantity`
        `list` of species density in units convertible to per cubic meter.
        The order of the species densities should follow species.

    omega : `~astropy.units.Quantity`
        Electromagnetic wave frequency in rad/s.

    Returns
    -------
    left : `~astropy.units.Quantity`
        L ("Left") Left-handed circularly polarization tensor element.

    right : `~astropy.units.Quantity`
        R ("Right") Right-handed circularly polarization tensor element.

    plasma : `~astropy.units.Quantity`
        P ("Plasma") dielectric tensor element.

    Notes
    -----
    In the rotating frame defined by
    :math:`(\mathbf{u}_{+}, \mathbf{u}_{-}, \mathbf{u}_z)`
    with :math:`\mathbf{u}_{±}=(\mathbf{u}_x ± \mathbf{u}_y)/\sqrt{2}`,
    the dielectric tensor takes a diagonal form with elements L, R, P with:

    .. math::
        L = 1 - \sum_s
                \frac{ω_{p,s}^2}{ω\left(ω - Ω_{c,s}\right)}

        R = 1 - \sum_s
                \frac{ω_{p,s}^2}{ω\left(ω + Ω_{c,s}\right)}

        P = 1 - \sum_s \frac{ω_{p,s}^2}{ω^2}

    where :math:`ω_{p,s}` is the plasma frequency and
    :math:`Ω_{c,s}` is the signed version of the cyclotron frequency
    for the species :math:`s` :cite:p:`stix:1992`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from numpy import pi
    >>> B = 2 * u.T
    >>> species = ["e-", "D+"]
    >>> n = [1e18 * u.m**-3, 1e18 * u.m**-3]
    >>> omega = 3.7e9 * (2 * pi) * (u.rad / u.s)
    >>> L, R, P = permittivity = cold_plasma_permittivity_LRP(B, species, n, omega)
    >>> L
    <Quantity 0.63333...>
    >>> permittivity.left  # namedtuple-style access
    <Quantity 0.63333...>
    >>> R
    <Quantity 1.41512...>
    >>> P
    <Quantity -4.8903...>
    """
    L, R, P = 1, 1, 1

    for s, n_s in zip(species, n, strict=False):
        omega_c = gyrofrequency(B=B, particle=s, signed=True)
        omega_p = plasma_frequency(n=n_s, particle=s)

        L += -(omega_p**2) / (omega * (omega - omega_c))
        R += -(omega_p**2) / (omega * (omega + omega_c))
        P += -(omega_p**2) / omega**2
    return RotatingTensorElements(L, R, P)


@preserve_signature
def permittivity_1D_Maxwellian_lite(omega, kWave, vth, wp):
    r"""
    The :term:`lite-function` for
    `~plasmapy.formulary.dielectric.permittivity_1D_Maxwellian`.
    Performs the same calculations as
    `~plasmapy.formulary.dielectric.permittivity_1D_Maxwellian`, but is
    intended for computational use and, thus, has data conditioning
    safeguards removed.

    Parameters
    ----------
    omega : |array_like| of real positive values
        The frequency, in rad/s, of the electromagnetic wave propagating
        through the plasma.

    kWave : |array_like| of real values
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    vth : `float`
        The 3D, most probable thermal speed, in m/s. (i.e. it includes
        the factor of :math:`\sqrt{2}`, see
        :ref:`thermal speed notes <thermal-speed-notes>`)

    wp : `float`
        The plasma frequency, in rad/s.

    Returns
    -------
    chi : |array_like| of complex values
        The ion or the electron dielectric permittivity of the plasma.
        This is a dimensionless quantity.

    See Also
    --------
    ~plasmapy.formulary.dielectric.permittivity_1D_Maxwellian

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.formulary import thermal_speed, plasma_frequency
    >>> T = 30 * u.eV
    >>> n = 1e18 * u.cm**-3
    >>> particle = "Ne"
    >>> Z = 8
    >>> omega = 3.541e15  # in rad/s
    >>> vth = thermal_speed(T=T, particle=particle).value
    >>> wp = plasma_frequency(n=n, particle=particle, Z=Z).value
    >>> k_wave = omega / vth
    >>> permittivity_1D_Maxwellian_lite(omega, k_wave, vth=vth, wp=wp)
    np.complex128(-6.72794...e-08+5.76024...e-07j)
    """

    # scattering parameter alpha.
    # explicitly removing factor of sqrt(2) to be consistent with Froula
    alpha = np.sqrt(2) * wp / (kWave * vth)
    # The dimensionless phase velocity of the propagating EM wave.
    zeta = omega / (kWave * vth)
    return -0.5 * (alpha**2) * plasma_dispersion_func_deriv(zeta)


@bind_lite_func(permittivity_1D_Maxwellian_lite)
@validate_quantities(
    kWave={"none_shall_pass": True}, validations_on_return={"can_be_complex": True}
)
def permittivity_1D_Maxwellian(
    omega: u.Quantity[u.rad / u.s],
    kWave: u.Quantity[u.rad / u.m],
    T: u.Quantity[u.K],
    n: u.Quantity[u.m**-3],
    particle: ParticleLike,
    z_mean: float | None = None,
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Compute the classical dielectric permittivity for a 1D Maxwellian
    plasma.

    This function can calculate both the ion and electron
    permittivities.  No additional effects are considered (e.g.
    magnetic fields, relativistic effects, strongly coupled regime, etc.).

    Parameters
    ----------
    omega : `~astropy.units.Quantity`
        The frequency, in rad/s, of the electromagnetic wave propagating
        through the plasma.

    kWave : `~astropy.units.Quantity`
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    T : `~astropy.units.Quantity`
        The plasma temperature — this can be either the electron or the
        ion temperature, but should be consistent with density and
        particle.

    n : `~astropy.units.Quantity`
        The plasma density — this can be either the electron or the ion
        density, but should be consistent with temperature and particle.

    particle : |particle-like|
        The plasma particle species.

    z_mean : `float`
        The average ionization of the plasma. This is only required for
        calculating the ion permittivity.

    Returns
    -------
    chi : `~astropy.units.Quantity`
        The ion or the electron dielectric permittivity of the plasma.
        This is a dimensionless quantity.

    Notes
    -----
    The dielectric permittivities for a Maxwellian plasma are described
    by the following equations (see p. 106 of :cite:t:`froula:2011`):

    .. math::
        χ_e(k, ω) = - \frac{α_e^2}{2} Z'(x_e)

        χ_i(k, ω) = - \frac{α_i^2}{2}\frac{Z}{} Z'(x_i)

        α = \frac{ω_p}{k v_{Th}}

        x = \frac{ω}{k v_{Th}}

    :math:`χ_e` and :math:`χ_i` are the electron and ion permittivities,
    respectively. :math:`Z'` is the derivative of the plasma dispersion
    function. :math:`α` is the scattering parameter which delineates
    the difference between the collective and non-collective Thomson
    scattering regimes. :math:`x` is the dimensionless phase velocity
    of the electromagnetic wave propagating through the plasma.

    Examples
    --------
    >>> import astropy.units as u
    >>> from numpy import pi
    >>> from plasmapy.formulary import thermal_speed
    >>> T = 30 * 11600 * u.K
    >>> n = 1e18 * u.cm**-3
    >>> particle = "Ne"
    >>> Z = 8
    >>> vth = thermal_speed(T, particle, method="most_probable")
    >>> omega = 5.635e14 * 2 * pi * u.rad / u.s
    >>> k_wave = omega / vth
    >>> permittivity_1D_Maxwellian(omega, k_wave, T, n, particle, Z)
    <Quantity -6.72955...e-08+5.76163...e-07j>

    For user convenience
    `~plasmapy.formulary.dielectric.permittivity_1D_Maxwellian_lite`
    is bound to this function and can be used as follows:

    >>> from plasmapy.formulary import plasma_frequency
    >>> wp = plasma_frequency(n, particle, Z=Z)
    >>> permittivity_1D_Maxwellian.lite(
    ...     omega.value, k_wave.value, vth=vth.value, wp=wp.value
    ... )
    np.complex128(-6.72955...e-08+5.76163...e-07j)
    """
    vth = thermal_speed(T=T, particle=particle, method="most_probable").value
    wp = plasma_frequency(n=n, particle=particle, Z=z_mean).value

    chi = permittivity_1D_Maxwellian_lite(
        omega.value,
        kWave.value,
        vth,
        wp,
    )
    return chi * u.dimensionless_unscaled
