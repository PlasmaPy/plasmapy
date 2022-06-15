"""Functions related to ionization states and the properties thereof."""

__all__ = ["ionization_balance", "Saha", "Z_bal_", "thomas_fermi_ionization_state"]
__aliases__ = ["Z_bal_"]

import astropy.units as u

from astropy.constants import a0, k_B, m_p
from numpy import exp, log, pi, sqrt

from plasmapy.utils.decorators import validate_quantities


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()}
)
def ionization_balance(n: u.m**-3, T_e: u.K) -> u.dimensionless_unscaled:
    r"""
    Return the average ionization state of ions in a plasma assuming that
    the numbers of ions in each state are equal.

    Z_bal is the estimate average ionization level of a plasma in thermal equilibrium
    that balances the number density of ions in two different ionization states.
    Z_bal is derived from the Saha equation with the assumptions that
    the atoms are of a single species,
    are either hydrogenic or completely ionized,
    and that there is a balance between ionization and recombination,
    meaning that the number of atoms in either state are equal.
    The Saha equation and therefore Z_bal are more accurate when the plasma
    is at a high density and temperature.

    .. math::

        Z\_bal = \sqrt{\frac{k_B T_e}{E_H}} \sqrt{\ln{\frac{1}{4 n a_{0}^3}
        (\frac{k_B T_e}{π E_H})^{3/2}}} - \frac{1}{2}

    Where :math:`k_B` is the Boltzmann constant,
    :math:`a_0` is the Bohr radius, and
    :math:`E_H` is the ionization energy of Hydrogen

    **Aliases:** `Z_bal_`

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        The electron temperature.

    n : `~astropy.units.Quantity`
        The electron number density of the plasma.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Raises
    ------
    `TypeError`
        If either of ``T_e`` or ``n`` is not a `~astropy.units.Quantity`
        and cannot be converted into a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If either of ``T_e`` or ``n`` is not in appropriate units.

    Examples
    --------
    >>> import astropy.units as u
    >>> T_e = 5000 * u.K
    >>> n = 1e19 * u.m ** -3
    >>> ionization_balance(n, T_e)
    <Quantity 0.274...>
    >>> T_e = 50 * u.eV
    >>> n = 1e10 * u.m ** -3
    >>> ionization_balance(n, T_e)
    <Quantity 12.615...>

    Returns
    -------
    Z : `~astropy.units.Quantity`
        The average ionization state of the ions in the plasma
        assuming that the numbers of ions in each state are equal.

    """
    E_H = 1 * u.Ry

    A = sqrt(k_B * T_e / E_H)
    B = log(1 / (4 * n * a0**3) * (k_B * T_e / (pi * E_H)) ** (3 / 2))

    return A * sqrt(B) - 1 / 2


Z_bal_ = ionization_balance
"""Alias for `~plasmapy.formulary.ionization.ionization_balance`."""


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()}
)
def Saha(g_j, g_k, n_e: u.m**-3, E_jk: u.J, T_e: u.K) -> u.dimensionless_unscaled:
    r"""
    Return the ratio of populations of two ionization states.

    The Saha equation, derived in statistical mechanics, gives an
    approximation of the ratio of population of ions in two different
    ionization states in a plasma. This approximation applies to plasmas
    in thermodynamic equilibrium where ionization and recombination of
    ions with electrons are balanced.

    .. math::
        \frac{N_j}{N_k} = \frac{1}{n_e} \frac{g_j}{4 g_k a_0^{3}} \left(
                              \frac{k_B T_e}{π E_H}
                          \right)^{\frac{3}{2}}
                          \exp\left( \frac{-E_{jk}}{k_B T_e} \right)

    Where :math:`k_B` is the Boltzmann constant, :math:`a_0` is the Bohr
    radius, :math:`E_H` is the ionization energy of hydrogen, :math:`N_j`
    and :math:`N_k` are the population of ions in the :math:`j` and
    :math:`k` states respectively. This function is equivalent to Eq.
    3.47 in `Drake`_.

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        The electron temperature in units of temperature or thermal
        energy per particle.

    g_j : `int`
        The degeneracy of ionization state :math:`j`\ .

    g_k : `int`
        The degeneracy of ionization state :math:`k`\ .

    E_jk : `~astropy.units.Quantity`
        The energy difference between ionization states :math:`j` and
        :math:`k`\ .

    n_e : `~astropy.units.Quantity`
        The electron number density of the plasma.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Raises
    ------
    `TypeError`
        If any of ``T_e``, ``E_jk``, or ``n_e`` is not a `~astropy.units.Quantity`
        and cannot be converted into a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If any of ``T_e``, ``E_jk``, or ``n`` is not in appropriate units.

    Returns
    -------
    `~astropy.units.Quantity`
        The ratio of population of ions in ionization state :math:`j` to
        state :math:`k`\ .

    Examples
    --------
    >>> import astropy.units as u
    >>> T_e = 5000 * u.K
    >>> n = 1e19 * u.m ** -3
    >>> g_j = 2
    >>> g_k = 2
    >>> E_jk = 1 * u.Ry
    >>> Saha(g_j, g_k, n, E_jk, T_e)
    <Quantity 3.299...e-06>
    >>> T_e = 1 * u.Ry
    >>> n = 1e23 * u.m ** -3
    >>> Saha(g_j, g_k, n, E_jk, T_e)
    <Quantity 1114595.586...>

    Notes
    -----
    For reference to this function and for more information regarding
    the Saha equation, see chapter 3 of R. Paul Drake's book,
    "High-Energy-Density Physics: Foundation of Inertial Fusion and
    Experimental Astrophysics" (`DOI: 10.1007/978-3-319-67711-8_3`_).

    .. _`Drake`: https://doi.org/10.1007/978-3-319-67711-8
    .. _`DOI: 10.1007/978-3-319-67711-8_3`: https://doi.org/10.1007/978-3-319-67711-8_3
    """
    E_h = 1 * u.Ry

    degeneracy_factor = (1 / n_e) * g_j / (4 * g_k * a0**3)
    physical_constants = (k_B * T_e / (pi * E_h)) ** (3 / 2)
    boltzmann_factor = exp(-E_jk / (k_B * T_e))

    return degeneracy_factor * physical_constants * boltzmann_factor


@validate_quantities(
    n_e={"can_be_negative": False},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def thomas_fermi_ionization_state(
    Z, n_e: u.m ** (-3), T_e: u.K
) -> u.dimensionless_unscaled:
    r"""
    Return the finite temperature Thomas-Fermi mean ionization state using fit provided in Table IV of
    :cite:p:`more:1985`.

    Parameters
    ----------
    Z : int
        Ion charge number.

    n_e : `~astropy.units.Quantity`
        Electron number density in units convertible to m\ :sup:`-3`\ .

    T_e : `~astropy.units.Quantity`
        Electron temperature in K or energy per electron.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    Raises
    ------
    `TypeError`
        If either of ``T_e`` or ``n_e`` is not a `~astropy.units.Quantity`
        and cannot be converted into a `~astropy.units.Quantity`.

    `~astropy.units.UnitConversionError`
        If either of ``T_e`` or ``n_e`` is not in appropriate units.

    Examples
    --------
    >>> import astropy.units as u
    >>> T_e = 1.16e7*u.K
    >>> n_e = 1e29*u.m**(-3)
    >>> Z = 6
    >>> thomas_fermi_ionization_state(Z, n_e, T_e)
    <Quantity 5.84965421>
    >>> T_e = 1e3 * u.eV
    >>> thomas_fermi_ionization_state(Z, n_e, T_e)
    <Quantity 5.84971587>

    Returns
    -------
    `~astropy.units.Quantity`
        The Thomas-Fermi ionization state of the ion in the plasma.

    Notes
    -----
    In hot dense plasmas, the combined effects of degeneracy and continuum lowering prevent the use of the Saha equation
    for the calculation of the mean ionization state, $\langle Z \rangle$. This is because the Saha equation is
    obtained from a balance of free energies of ideal gases with well-defined chemical compositions. In the strong
    coupling regime the exact chemical abundances of the elements and their ionization thresholds are unknown,
    therefore, it is more convenient to imagine the plasma as a continuum medium of electrons and positive nuclei.
    The most common physical picture employed in the calculation of static properties of hot dense plasmas is the
    so-called Average Atom (AA) model. This model considers a nucleus with charge number $Z$ embedded in a spherical cavity
    in a continuous background of positive charge. The radius of the sphere is $a_i$ determined by the plasma density.
    :cite:t:`more:1985`. Several AA models have been formulated for the calculation of $\langle Z \rangle$ and reviewed in
    :cite:t:`more:1985`, :cite:t:`murillo:2013`. The simplest model that includes thermal and pressure ionization effects is
    the Thomas-Fermi model which calculates the mean ionization state from

    .. math::
        \langle Z \rangle_{\rm TF} = Z_{\rm nuc} - \frac{4\pi a_i^3}{3} n(a_i)

    where :math:`Z_{\rm nuc}` is the bare nuclear charge of the ion, :math:`a_i` is the Wigner-Seitz radius,
    see :func:`plasmapy.formulary.quantum.Wigner_Seitz_radius`, and :math:`n(a_i)` is the electron density at the edge
    of the spherical cavity. This density is obtained by solving nonlinear equation obtained from Thomas-Fermi
    Density Functional Theory (TF-DFT). More information and details can be found in :cite:t:`murillo:2013`.

    The `thomas_fermi_ionization_state` function, instead of solving nonlinear equations, uses fits provided by R. More
    to calculate the mean ionization state :cite:t:`more:1985`.
    Table IV of :cite:t:`more:1985` has been reproduced in :cite:t:`stanton:2016` for convenience.

    """

    alpha = 14.3139
    beta = 0.6624
    a1 = 0.003323
    a2 = 0.9718
    a3 = 9.26148e-5
    a4 = 3.10165
    b0 = -1.7630
    b1 = 1.43175
    b2 = 0.31546
    c1 = -0.366667
    c2 = 0.983333

    # Convert density to g/cc and temperature to eV
    n_cc_u = n_e.to(u.cm ** (-3)) * m_p.cgs
    n_cc = n_cc_u.value
    T_eV = T_e.to(u.eV, equivalencies=u.temperature_energy()).value

    R = n_cc / Z
    T0 = T_eV / Z ** (4.0 / 3.0)
    Tf = T0 / (1 + T0)

    A = a1 * T0 ** a2 + a3 * T0 ** a4
    B = -exp(b0 + b1 * Tf + b2 * Tf ** 7)
    C = c1 * Tf + c2
    Q1 = A * R ** B
    Q = (R ** C + Q1 ** C) ** (1 / C)
    x = alpha * Q ** beta

    return Z * x / (1 + x + sqrt(1 + 2.0 * x)) * u.dimensionless_unscaled
