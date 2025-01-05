"""Analytical solutions to the Stix cold plasma function."""

__all__ = ["stix"]

import astropy.units as u
import numpy as np
from astropy.constants.si import c

from plasmapy.formulary.frequencies import gyrofrequency, plasma_frequency
from plasmapy.particles import Particle, ParticleList
from plasmapy.utils.decorators import validate_quantities

c_si_unitless = c.value


@validate_quantities(
    B={"can_be_negative": False},
    n_i={"can_be_negative": False},
    w={"can_be_negative": False, "can_be_zero": False},
)
def stix(  # noqa: C901, PLR0912, PLR0915
    B: u.Quantity[u.T],
    w: u.Quantity[u.rad / u.s],
    ions: Particle,
    n_i: u.Quantity[u.m**-3],
    theta: u.Quantity[u.rad],
):
    r"""
    Calculate the cold plasma dispersion function presented by
    :cite:t:`stix:1992`, and discussed by :cite:t:`bellan:2012`.

    This is an analytical solution of equation 8 in
    :cite:t:`bellan:2012`.  See the **Notes** section below for
    additional details.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to T.

    w : `~astropy.units.Quantity`, single valued or 1-D array
        Wavefrequency in units convertible to rad/s.  Either singled
        valued or 1-D array of length :math:`N`.

    ions: a single or `list` of |particle-like| object(s)
        A list or single instance of |particle-like| objects
        representing the ion species (e.g., ``"p"`` for protons,
        ``"D+"`` for deuterium, ``["H+", "He+"]`` for hydrogen and
        helium, etc.).  All ions must be positively charged.

    n_i: `~astropy.units.Quantity`, single valued or 1-D array
        Ion number density in units convertible to m\ :sup:`-3`.  Must
        be single valued or equal length to ``ions``.

    theta: `~astropy.units.Quantity`, single valued or 1-D array
        The angle of propagation of the wave with respect to the
        magnetic field, :math:`\cos^{-1}(k_z / k)`, in units convertible
        to radians.  Either single valued or 1-D array of size
        :math:`M`.

    Returns
    -------
    k : `~astropy.units.Quantity` of shape ``(N, M, 4)``
        An array of wavenubmers in units rad/m (shape
        :math:`N × M × 4`).  The first dimension maps to the
        ``w`` array, the second dimension maps to the ``theta`` array,
        and the third dimension maps to the four roots of the Stix
        polynomial.

        * ``k[0]`` is the square root of the positive quadratic solution
        * ``k[1] = -k[0]``
        * ``k[2]`` is the square root of the negative quadratic solution
        * ``k[3] = -k[2]``

    Raises
    ------
    `TypeError`
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    `ValueError`
        Particles in ``ions`` are not positively charged ions.

    `ValueError`
        The size of ``n_i`` is not the same as the length of ``ions``.

    ValueError
        If of ``B`` or ``n_i`` is negative.

    ValueError
        If ``w`` is negative or zero.

    ValueError
        If ``w`` or ``theta`` are not single valued or a 1-D array.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    Notes
    -----
    The cold plasma dispersion function is defined by
    :cite:t:`stix:1992` in section 1-3 (and present by
    :cite:t:`bellan:2012` in equation 8) to be

    .. math::
        (S\sin^{2}(θ) + P\cos^{2}(θ))(ck/ω)^{4}
            - [
                RL\sin^{2}(θ) + PS(1 + \cos^{2}(θ))
            ](ck/ω)^{2} + PRL = 0

    where,

    .. math::
        \mathbf{B_o} &= B_{o} \mathbf{\hat{z}} \\
        \cos θ &= \frac{k_z}{k} \\
        \mathbf{k} &= k_{\rm x} \hat{x} + k_{\rm z} \hat{z}

    .. math::
        S &= 1 - \sum_{s} \frac{ω^{2}_{p,s}}{ω^{2} -
            ω^{2}_{c,s}}\\
        P &= 1 - \sum_{s} \frac{ω^{2}_{p,s}}{ω^{2}}\\
        D &= \sum_{s}
            \frac{ω_{c,s}}{ω}
            \frac{ω^{2}_{p,s}}{ω^{2} -
            ω_{c,s}^{2}}

    .. math::
        R = S + D \hspace{1cm} L = S - D

    :math:`ω` is the wave frequency, :math:`k` is the wavenumber,
    :math:`θ` is the wave propagation angle with respect to the
    background magntic field :math:`\mathbf{B_o}`, :math:`s` corresponds
    to plasma species :math:`s`, :math:`ω_{p,s}` is the plasma
    frequency of species :math:`s`, and :math:`ω_{c,s}` is the
    gyrofrequency of species :math:`s`. The derivation of this
    dispersion relation assumed:

    * zero temperature for all plasma species (:math:`T_{s}=0`)
    * quasi-neutrality
    * a uniform background magntic field
      :math:`\mathbf{B_o} = B_{o} \mathbf{\hat{z}}`
    * no D.C. electric field :math:`\mathbf{E_o}=0`
    * zero-order quantities for all plasma parameters (densities,
      electric-field, magnetic field, particle speeds, etc.) are
      constant in time and space
    * first-order perturbations in plasma parameters vary like
      :math:`\sim e^{\left [ i (\textbf{k}\cdot\textbf{r} - ω t)\right ]}`

    Due to the cold plasma assumption, this equation is valid for all
    :math:`ω` and :math:`k` given
    :math:`\frac{ω}{k_{z}} \gg v_{Th}` for all thermal speeds
    :math:`v_{Th}` of all plasma species and :math:`k_{x} r_{L} \ll 1`
    for all gyroradii :math:`r_{L}` of all plasma species.

    The relation predicts :math:`k \to 0` when any one of P, R or L
    vanish (cutoffs) and :math:`k \to ∞` for perpendicular
    propagation during wave resonance :math:`S \to 0`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.particles import Particle
    >>> from plasmapy.dispersion.analytical.stix_ import stix
    >>> inputs = {
    ...     "B": 8.3e-9 * u.T,
    ...     "w": 0.001 * u.rad / u.s,
    ...     "ions": [Particle("H+"), Particle("He+")],
    ...     "n_i": [4.0e5,2.0e5] * u.m**-3,
    ...     "theta": 30 * u.deg,
    ... }
    >>> stix(**inputs)
    <Quantity [ 6.03817661e-09-0.j, -6.03817661e-09+0.j,
        6.97262784e-09-0.j, -6.97262784e-09+0.j] rad / m>
    """

    # Validate ions argument
    if not isinstance(ions, list | tuple | ParticleList):
        ions = [ions]
    ions = ParticleList(ions)

    if not all(failed := [ion.is_ion and ion.charge_number > 0 for ion in ions]):
        raise ValueError(
            "Particle(s) passed to 'ions' must be a positively charged"
            " ion. The following particle(s) is(are) not allowed "
            f"{[ion for ion, fail in zip(ions, failed, strict=False) if not fail]}"
        )

    # Validate n_i argument
    if n_i.ndim not in {0, 1}:
        raise ValueError(
            "Argument 'n_i' must be a single valued or a 1D array of "
            f"size 1 or {len(ions)}, instead got shape of {n_i.shape}"
        )
    elif n_i.ndim == 1 and n_i.size != len(ions):
        raise ValueError(
            "Argument 'n_i' and 'ions' need to be the same length, got"
            f" value of shape {len(ions)} and {len(n_i.shape)}."
        )

    n_i = n_i.value
    if n_i.ndim == 0:
        n_i = np.array([n_i] * len(ions))
    elif n_i.size == 1:
        n_i = np.repeat(n_i, len(ions))

    species = [*ions, Particle("e-")]
    densities = np.zeros(n_i.size + 1)
    densities[:-1] = n_i
    densities[-1] = np.sum(n_i * ions.charge_number)

    # Validate B argument
    B = B.squeeze()
    if B.ndim != 0:
        raise ValueError(
            "Argument 'B' must be single valued and not an array of"
            f" shape  {B.shape}."
        )

    # Validate w argument and dimension
    w = w.value.squeeze()
    if w.ndim not in {0, 1}:
        raise ValueError(
            "Argument 'w' needs to be a single value or a 1D array "
            f" astropy Quantity, got a value of shape {w.shape}."
        )
    elif np.isscalar(w):
        w = np.array([w])

    # Validate theta value
    theta = theta.value.squeeze()
    if theta.ndim not in {0, 1}:
        raise TypeError(
            "Argument 'theta' needs to be a single value or 1D array "
            f" astropy Quantity, got array of shape {theta.shape}."
        )
    elif np.isscalar(theta):
        theta = np.array([theta])

    # Generate mesh grid of w x theta
    w, theta = np.meshgrid(w, theta, indexing="ij")

    # Generate the plasma parameters needed
    wps = []
    wcs = []
    for par, dens in zip(species, densities, strict=False):
        wps.append(plasma_frequency(n=dens * u.m**-3, particle=par).value)
        wcs.append(gyrofrequency(B=B, particle=par, signed=True).value)

    # Stix method implemented
    S = np.ones_like(w, dtype=np.float64)
    P = np.ones_like(S)
    D = np.zeros_like(S)
    for wc, wp in zip(wcs, wps, strict=False):
        S -= (wp**2) / (w**2 - wc**2)
        P -= (wp / w) ** 2
        D += ((wp**2) / (w**2 - wc**2)) * (wc / w)

    R = S + D
    L = S - D

    # Generate coefficients to solve, a * k**4 + b * k**2 + c = 0
    a = (S * np.sin(theta) ** 2) + (P * np.cos(theta) ** 2)
    b = -((R * L * np.sin(theta) ** 2) + (P * S * (1 + np.cos(theta) ** 2)))
    c = P * R * L

    # Solve for k values
    k = np.empty((*w.shape, 4), dtype=np.complex128)
    k[..., 0] = (
        np.emath.sqrt((-b + np.emath.sqrt(b**2 - 4 * a * c)) / (2 * a))
        * w
        / c_si_unitless
    )
    k[..., 1] = -k[..., 0]
    k[..., 2] = (
        np.emath.sqrt((-b - np.emath.sqrt(b**2 - 4 * a * c)) / (2 * a))
        * w
        / c_si_unitless
    )
    k[..., 3] = -k[..., 2]

    return k.squeeze() * u.rad / u.m
