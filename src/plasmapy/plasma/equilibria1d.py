"""One-dimensional plasma equilibria."""

__all__ = ["HarrisSheet"]

import astropy.constants as const
import astropy.units as u
import numpy as np

from plasmapy.utils.decorators.validators import validate_quantities


class HarrisSheet:
    r"""
    Define a Harris Sheet Equilibrium.

    Magnetic field will be in the :math:`±x` direction and the current
    density will be in the :math:`±z` direction in a :math:`\hat{x} ×
    \hat{y} = \hat{z}` coordinate system.

    Parameters
    ----------
    B0 : `~astropy.units.Quantity`
         Magnitude of magnetic field in the limit of :math:`y → ∞` in
         units convertible to teslas.

    delta : `~astropy.units.Quantity`
        The thickness of the current sheet in units convertible to
        meters.

    P0 : `~astropy.units.Quantity`
        The plasma pressure in the limit of :math:`y → ∞` in units
        convertible to pascals.

    Notes
    -----
    A current sheet is current limited to a surface.

    A Harris sheet is a 1D ideal MHD equilibrium. In resistive MHD if
    there is any resistivity, it won't be a true equilibrium since the
    resistivity will gradually smooth the profile out over time.

    A Harris sheet is often used as the initial condition for
    simulations of magnetic reconnection.

    Examples
    --------
    >>> import astropy.units as u
    >>> harris_sheet = HarrisSheet(delta=3 * u.m, B0=2 * u.T)
    >>> harris_sheet.magnetic_field(y=5 * u.m)
    <Quantity 1.8622... T>
    """

    def __init__(self, B0, delta, P0=0 * u.Pa) -> None:
        self.B0 = B0
        self.delta = delta
        self.P0 = P0

    @validate_quantities
    def magnetic_field(self, y: u.Quantity[u.m]) -> u.Quantity[u.T]:
        r"""
        Compute the magnetic field.

        In this equation, :math:`B_0` is the asymptotic magnitude of the
        magnetic field for :math:`y → ±∞` and :math:`δ` is the thickness
        of the sheet.

        .. math::

            B_x(y) = B_0 \tanh \left( \frac{y}{δ} \right)

        Parameters
        ----------
        y : `~astropy.units.Quantity`
           Orthogonal distance from the current sheet center.

        Examples
        --------
        >>> import astropy.units as u
        >>> B0 = 1 * u.T
        >>> delta = 1 * u.m
        >>> P0 = 0 * u.Pa
        >>> hs = HarrisSheet(B0, delta, P0)
        >>> y = [-2, 0, 2] * u.m
        >>> hs.magnetic_field(y)
        <Quantity [-0.9640...,  0.        ,  0.9640...] T>
        """
        return self.B0 * np.tanh(u.rad * y / self.delta)

    @validate_quantities
    def current_density(self, y: u.Quantity[u.m]) -> u.Quantity[u.A / u.m**2]:
        r"""
        Compute the current density.

        .. math::

          J_z(y) = - \frac{B_0}{δ μ_0) \mathrm{sech}^2 \left( \frac{y}{δ} \right)

        Parameters
        ----------
        y : `~astropy.units.Quantity`
          Orthogonal distance from the current sheet center.

        Examples
        --------
        >>> import astropy.units as u
        >>> B0 = 1 * u.T
        >>> delta = 1 * u.m
        >>> P0 = 0 * u.Pa
        >>> hs = HarrisSheet(B0, delta, P0)
        >>> y = [-2, 0, 2] * u.m
        >>> hs.current_density(y)
        <Quantity [ -56222.14... , -795774.71...,  -56222.14... ] A / m2>
        """
        return (
            -self.B0 / (self.delta * const.mu0) * np.cosh(u.rad * y / self.delta) ** -2
        )

    @validate_quantities
    def plasma_pressure(self, y: u.Quantity[u.m]) -> u.Quantity[u.Pa]:
        r"""
        Compute plasma pressure.

        .. math::

            p(y) = \frac{B_0^2}{2 μ_0} \mathrm{sech}^2 \left( \frac{y}{δ} \right) + p_0

        Parameters
        ----------
        y : `~astropy.units.Quantity`
          Orthogonal distance from the current sheet center.

        Examples
        --------
        >>> import astropy.units as u
        >>> B0 = 1 * u.T
        >>> delta = 1 * u.m
        >>> P0 = 0 * u.Pa
        >>> hs = HarrisSheet(B0, delta, P0)
        >>> y = [-2, 0, 2] * u.m
        >>> hs.plasma_pressure(y)
        <Quantity [ 28111.07..., 397887.35...,  28111.07...] Pa>
        """
        return (
            self.B0**2 / (2 * const.mu0) * (np.cosh(u.rad * y / self.delta) ** -2)
            + self.P0
        )
