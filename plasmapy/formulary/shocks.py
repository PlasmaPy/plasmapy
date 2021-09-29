""" This module gathers functions relating to shocks and the properties thereof.
"""
__all__ = ["entropy_jump_polytropic"]


from astropy import units as u
from astropy.constants import c, a0, k_B
from numpy import pi, exp, sqrt, log, nan, isclose
from warnings import warn

from plasmapy.utils.exceptions import PhysicsWarning
from plasmapy.utils.decorators import (
    angular_freq_to_hz,
    check_relativistic,
    validate_quantities,
)


@validate_quantities(
    rho_1={"can_be_negative": False, "pass_equivalent_units": True},
    rho_2={"can_be_negative": False, "pass_equivalent_units": True},
    p_1={"can_be_negative": True, "pass_equivalent_units": True},
    p_2={"can_be_negative": True, "pass_equivalent_units": True},
    c_v={"can_be_negative": False},
)
def entropy_jump_polytropic(
    c_v: u.J / u.K,
    p_1: u.Pa,
    p_2: u.Pa,
    rho_1: u.kg / u.m ** 3,
    rho_2: u.kg / u.m ** 3,
    gamma=5 / 3,
) -> u.J / u.K:
    r"""
    Entropy is not conserved across a shock, since a shock is
    an irreversible, non-adiabatic process. This equation is based on the specific entropy
    of a polytropic gas before and after the shock has passed. The shock
    is an isolated steady, perpendicular shock, flowing through a
    non magnetized, polytropic gas.

    .. math::

         s_2 - s_1 = c_v
                     \ln{\left[
                         \frac{p_2}{p_1}
                         \left( \frac{\rho_1}{\rho_2} \right)^{\gamma}
                     \right]}

    Here variables indexed by 1 and 2 are upstream (pre shock)
    and downstream (post shock) respectively.
    This function is equivalent to Eq. 4.24 in `Drake`_.

    Parameters
    ----------
    c_v : `~astropy.units.Quantity`
        Heat capacity a constant volume for the gas.

    p_1 : `~astropy.units.Quantity`
        Upstream pressure of the gas.

    p_2 : `~astropy.units.Quantity`
        Downstream pressure of the gas.

    rho_1 : `~astropy.units.Quantity`
        Upstream mass density of the gas.

    rho_2 : `~astropy.units.Quantity`
        Downstream mass density of the gas.

    gamma : `float`
        Polytropic index of the gas.

    Returns
    -------
    ds: `~astropy.units.Quantity`
        The change of entropy due to the shock on the gas.

    Examples
    --------
    >>> import astropy.units as u
    >>> c_v = 15 * u.J / u.K
    >>> gamma = 5 / 3 #ideal gas approx
    >>> p_1 = 101325 * u.Pa
    >>> p_2 = 2 * p_1
    >>> rho_1 = 2 * u.kg / u.m ** 3
    >>> rho_2 = 3 * u.kg / u.m ** 3
    >>> entropy_jump_polytropic(c_v, p_1, p_2, rho_1, rho_2, gamma)
    <Quantity 0.2605... J / K>

    Notes
    -----
    For reference to this function and for more information regarding shocks and rarefactions,
    see chapter 4 of R Paul Drake's book, "High-Energy-Density Physics: Foundation of
    Inertial Fusion and Experimental Astrophysics" (`DOI: 10.1007/978-3-319-67711-8_3`_).

    .. _`Drake`: https://doi.org/10.1007/978-3-319-67711-8
    .. _`DOI: 10.1007/978-3-319-67711-8_3`: https://doi.org/10.1007/978-3-319-67711-8_3

    """
    pressure_ratio = (p_2 / p_1).to(u.dimensionless_unscaled)
    density_ratio = (rho_1 / rho_2).to(u.dimensionless_unscaled)

    ds = c_v * log(pressure_ratio * density_ratio ** gamma)

    if isclose(abs(ds), 0.0, rtol=1e-4, atol=1e-8):
        warn(
            """Entropy change cannot be 0, as this would imply that
            a shock is a reversible process. Shocks are
            irreversible, so the entropy change must be nonzero.""",
            PhysicsWarning,
        )
        return u.Quantity(nan, unit=c_v.unit)

    elif ds < 0:
        warn(
            "Entropy change cannot be negative, perhaps regions 1 and 2 are switched",
            PhysicsWarning,
        )
        return u.Quantity(nan, unit=c_v.unit)

    else:
        return ds
