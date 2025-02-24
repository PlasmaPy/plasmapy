import math
import numpy as np
import scipy.special
import sympy.functions.special.bessel

from sympy import Derivative


class solution:
    """
    Define Analytical solution for spheromak equilibria.

    Parameters
    ----------
    B0: `float`
    magnetic field.

    j1 : `float`
    spherical bessel function.

    r : 'float'
    A surface where the radial magnetic field vanishes.

    lamb : 'float'
    eigenvalue to make j cross B = 0.

    Notes
    -----
    A spheromak is an arrangement of plasma that is formed smilar to a smoke ring. The plasma uses its own properties to creat a torodial shape.


    """

    def __init__(self, B0, a, lamb):
        self.B0 = B0
        self.a = a
        self.lamb = lamb
        self.r = r
        self.z = z

    def B_radial(self, r, theta):
        r"""
        Compute the magnetic field in the radial direction.

        .. math::

            2*B_0*(a/r)*j1*(\lambda*r)*\cos(\theta)

        Parameters
        ----------
        lamb : `float`
          eigenvalue to make J cross B = 0.

        """

        return (
            2
            * self.B0
            * (self.a / r)
            * scipy.special.j1(self.lamb * r)
            * math.cos(theta)
        )

    def B_theta(self, r, j1, theta):
        r"""
        Compute the magnetic field for theta.

        .. math::

            -1*B_0*(a/r)*Derivative[r*j1*(\lambda*r)]*\sin(\theta)

        Parameters
        ----------
        lamb : `float`
          eigenvalue to make J cross B = 0.

        """
        return (
            -1
            * self.B0
            * (self.a / r)
            * Derivative[r * sympy.functions.special.bessel.ji(self.lamb * r), r]
            * math.sin(theta)
        )

    def B_phi(self, r, theta):
        r"""
        Compute the magnetic field for phi.

        .. math::

        lamb*a*B_0*j1*(\lambda*r)*\sin(\theta)

        Parameters
        ----------
        lamb : `float`
          eigenvalue to make J cross B = 0.

        """
        return (
            self.lamb
            * self.a
            * self.B0
            * scipy.special.j1(self.lamb * r)
            * math.sin(theta)
        )

    def pshi(self, r, z):
        beta = self.lamb * (np.sqrt(r**2 + z**2))

        return ((2 * np.pi * (r) ** 2) / (self.lamb * (r**2 + z**2))) * (
            (np.sin(beta) / beta) - np.cos(beta)
        )
