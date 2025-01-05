"""Plasma equilibria in cylindrical geometry."""

import numpy as np
import scipy.special


class ForceFreeFluxRope:
    r"""
    Representation of the analytical Lundquist solution for
    force-free magnetic flux ropes :cite:p:`lundquist:1950`.

    Parameters
    ----------
    B0 : `~astropy.units.Quantity`
        Magnetic field strength in units convertible to tesla.

    alpha : `~astropy.units.Quantity`
        Eigenvalue to make :math:`\mathbf{J} × \mathbf{B} = 0`, in units
        convertible to inverse length.

    Notes
    -----
    The Lundquist solution [also known as the Bessel Function Model (BFM)]
    is a cylindrically symmetric force-free equilibrium which is often used
    to approximate the magnetic structure of interplanetary coronal mass
    ejections (ICMEs).
    """

    def __init__(self, B0, alpha: float) -> None:
        self.B0 = B0
        self.alpha = alpha

    def B_theta(self, r):
        r"""
        Compute the component of the magnetic field in the azimuthal
        direction.

        .. math::

            B_θ(r) = B_0 J_1(α r)

        where :math:`α` is the eigenvalue and :math:`J_1` is the Bessel
        function of the first kind of order 1.

        Parameters
        ----------
        r : `~astropy.units.Quantity`
             Radial distance from flux rope axis in units convertible
             to meters.

        Returns
        -------
        `~astropy.units.Quantity`
        """
        return self.B0 * scipy.special.j1(self.alpha * r)

    def B_z(self, r):
        r"""
        Compute the axial component of the magnetic field.

        .. math::

            B_z(r) = B_0 J_0(α r)

        where :math:`α` is the eigenvalue and :math:`J_0` is the Bessel
        function of the first kind of order 0.

        Parameters
        ----------
        r : `~astropy.units.Quantity`
             Radial distance from flux rope axis in units convertible
             to meters.

        Returns
        -------
         `~astropy.units.Quantity`

        """
        return self.B0 * scipy.special.j0(self.alpha * r)

    def B_magnitude(self, r):
        r"""
        Compute the total magnetic field.

        The magnitude of the magnetic field is given by

        .. math::

            B(r) = \sqrt{B_z(r)^2 + B_θ(r)^2}.

        Parameters
        ----------
        r : `~astropy.units.Quantity`
             Radial distance from flux rope axis in units convertible
             to meters.

        Returns
        -------
        `~astropy.units.Quantity`
        """
        return np.sqrt(self.B_z(r) ** 2 + self.B_theta(r) ** 2)
