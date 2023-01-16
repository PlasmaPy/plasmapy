"""Functions related to ionization states and the properties thereof."""

__all__ = ["ionization_balance", "Saha", "Z_bal_"]
__aliases__ = ["Z_bal_"]

import astropy.units as u

from astropy.constants import a0, k_B
from numpy import exp, log, pi, sqrt

from plasmapy.utils.decorators import validate_quantities


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()}
)
def ionization_balance(
    n: u.Quantity[u.m**-3], T_e: u.Quantity[u.K]
) -> u.Quantity[u.dimensionless_unscaled]:
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
def Saha(
    g_j, g_k, n_e: u.Quantity[u.m**-3], E_jk: u.Quantity[u.J], T_e: u.Quantity[u.K]
) -> u.Quantity[u.dimensionless_unscaled]:
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
    ratio : `~astropy.units.Quantity`
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
