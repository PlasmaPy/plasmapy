"""
Define MagneticStatics class to calculate common static magnetic fields
as first raised in issue #100.
"""
__all__ = [
    "CircularWire",
    "FiniteStraightWire",
    "GeneralWire",
    "InfiniteStraightWire",
    "MagneticDipole",
    "MagnetoStatics",
    "Wire",
]

import abc
import astropy.units as u
import numbers
import numpy as np
import scipy.special

from astropy import constants

from plasmapy.utils.decorators import validate_quantities


class MagnetoStatics(abc.ABC):
    """Abstract class for all kinds of magnetic static fields"""

    @abc.abstractmethod
    def magnetic_field(self, p: u.m) -> u.T:
        """
        Calculate magnetic field generated by this wire at position ``p``.

        Parameters
        ----------
        p : `~astropy.units.Quantity`
            Three-dimensional position vector.

        Returns
        -------
        B : `~astropy.units.Quantity`
            Magnetic field at the specified position.
        """


class MagneticDipole(MagnetoStatics):
    r"""
    Simple magnetic dipole — two nearby opposite point charges.

    Parameters
    ----------
    moment: `~astropy.units.Quantity`
        Magnetic moment vector, in units of A m\ :sup:`2`\ .

    p0: `~astropy.units.Quantity`
        Position of the dipole.
    """

    @validate_quantities
    def __init__(self, moment: u.A * u.m ** 2, p0: u.m):
        self.moment = moment.value
        self._moment_u = moment.unit
        self.p0 = p0.value
        self._p0_u = p0.unit

    def __repr__(self):
        name = self.__class__.__name__
        moment = self.moment
        p0 = self.p0
        moment_u = self._moment_u
        p0_u = self._p0_u
        return f"{name}(moment={moment}{moment_u}, p0={p0}{p0_u})"

    def magnetic_field(self, p: u.m) -> u.T:
        r"""
        Calculate magnetic field generated by this wire at position ``p``.

        Parameters
        ----------
        p : `~astropy.units.Quantity`
            Three-dimensional position vector.

        Returns
        -------
        B : `~astropy.units.Quantity`
            Magnetic field at the specified position.
        """
        r = p - self.p0
        m = self.moment
        B = (
            constants.mu0.value
            / 4
            / np.pi
            * (
                3 * r * np.dot(m, r) / np.linalg.norm(r) ** 5
                - m / np.linalg.norm(r) ** 3
            )
        )
        return B * u.T


class Wire(MagnetoStatics):
    """Abstract wire class for concrete wires to be inherited from."""


class GeneralWire(Wire):
    r"""
    General wire class described by its parametric vector equation

    Parameters
    ----------
    parametric_eq: Callable
        A vector-valued (with units of position) function of a single real
        parameter.

    t1: `float`
        Lower bound of the parameter, smaller than ``t2``.

    t2: `float`
        Upper bound of the parameter, larger than ``t1``.

    current: `~astropy.units.Quantity`
        Electric current.
    """

    @validate_quantities
    def __init__(self, parametric_eq, t1, t2, current: u.A):
        if callable(parametric_eq):
            self.parametric_eq = parametric_eq
        else:
            raise ValueError("Argument parametric_eq should be a callable")
        if t1 < t2:
            self.t1 = t1
            self.t2 = t2
        else:
            raise ValueError(f"t1={t1} is not smaller than t2={t2}")
        self.current = current.value
        self._current_u = current.unit

    def __repr__(self):
        name = self.__class__.__name__
        parametric_eq = self.parametric_eq.__name__
        t1 = self.t1
        t2 = self.t2
        current = self.current
        current_u = self._current_u
        return (
            f"{name}(parametric_eq={parametric_eq}, t1={t1}, t2={t2}, "
            f"current={current}{current_u})"
        )

    def magnetic_field(self, p: u.m, n: numbers.Integral = 1000) -> u.T:
        r"""
        Calculate magnetic field generated by this wire at position `p`

        Parameters
        ----------
        p : `~astropy.units.Quantity`
            Three-dimensional position vector.

        n : `int`, optional
            Number of segments for Wire calculation (defaults to 1000).

        Returns
        -------
        B : `astropy.units.Quantity`
            Magnetic field at the specified position.

        Notes
        -----
        For simplicity, we segment the wire into ``n`` equal pieces,
        and assume each segment is straight. Default ``n`` is 1000.

        .. math::

            \vec B
            \approx \frac{μ_0 I}{4π} \sum_{i=1}^{n}
            \frac{[\vec l(t_{i}) - \vec l(t_{i-1})] \times
            \left[\vec p - \frac{\vec l(t_{i}) + \vec l(t_{i-1})}{2}\right]}
            {\left|\vec p - \frac{\vec l(t_{i}) + \vec l(t_{i-1})}{2}\right|^3},
            \quad \text{where}\, t_i = t_{\min}+i/n*(t_{\max}-t_{\min})
        """

        p1 = self.parametric_eq(self.t1)
        step = (self.t2 - self.t1) / n
        t = self.t1
        B = 0
        for i in range(n):
            t = t + step
            p2 = self.parametric_eq(t)
            dl = p2 - p1
            p1 = p2
            R = p - (p2 + p1) / 2
            B += np.cross(dl, R) / np.linalg.norm(R) ** 3
        B = B * constants.mu0.value / 4 / np.pi * self.current
        return B * u.T


class FiniteStraightWire(Wire):
    """
    Finite length straight wire class.

    The ``p1`` to ``p2`` direction is the positive current direction.

    Parameters
    ----------
    p1 : `~astropy.units.Quantity`
        Three-dimensional Cartesian coordinate of one end of the straight wire.

    p2 : `~astropy.units.Quantity`
        Three-dimensional Cartesian coordinate of another end of the straight wire.

    current : `astropy.units.Quantity`
        Electric current.
    """

    @validate_quantities
    def __init__(self, p1: u.m, p2: u.m, current: u.A):
        self.p1 = p1.value
        self.p2 = p2.value
        self._p1_u = p1.unit
        self._p2_u = p2.unit
        if np.all(p1 == p2):
            raise ValueError("p1, p2 should not be the same point.")
        self.current = current.value
        self._current_u = current.unit

    def __repr__(self):
        name = self.__class__.__name__
        p1 = self.p1
        p2 = self.p2
        current = self.current
        p1_u = self._p1_u
        p2_u = self._p2_u
        current_u = self._current_u
        return f"{name}(p1={p1}{p1_u}, p2={p2}{p2_u}, current={current}{current_u})"

    def magnetic_field(self, p) -> u.T:
        r"""
        Calculate magnetic field generated by this wire at position `p`

        Parameters
        ----------
        p : `astropy.units.Quantity`
            Three-dimensional position vector

        Returns
        -------
        B : `astropy.units.Quantity`
            Magnetic field at the specified position

        Notes
        -----
        Let :math:`P_f` be the foot of perpendicular, :math:`θ_1`
        (\ :math:`θ_2`\ ) be the angles between :math:`\overrightarrow{PP_1}`
        (\ :math:`\overrightarrow{PP_2}`\ ) and :math:`\overrightarrow{P_2P_1}`\ .

        .. math:
            \vec B = \frac{(\overrightarrow{P_2P_1}\times\overrightarrow{PP_f})^0}
                     {|\overrightarrow{PP_f}|}
                     \frac{μ_0 I}{4π} (\cos θ_1 - \cos θ_2)
        """
        # foot of perpendicular
        p1, p2 = self.p1, self.p2
        p2_p1 = p2 - p1
        ratio = np.dot(p - p1, p2_p1) / np.dot(p2_p1, p2_p1)
        pf = p1 + p2_p1 * ratio

        # angles: theta_1 = <p - p1, p2 - p1>, theta_2 = <p - p2, p2 - p1>
        cos_theta_1 = (
            np.dot(p - p1, p2_p1) / np.linalg.norm(p - p1) / np.linalg.norm(p2_p1)
        )
        cos_theta_2 = (
            np.dot(p - p2, p2_p1) / np.linalg.norm(p - p2) / np.linalg.norm(p2_p1)
        )

        B_unit = np.cross(p2_p1, p - pf)
        B_unit = B_unit / np.linalg.norm(B_unit)

        B = (
            B_unit
            / np.linalg.norm(p - pf)
            * (cos_theta_1 - cos_theta_2)
            * constants.mu0.value
            / 4
            / np.pi
            * self.current
        )

        return B * u.T

    def to_GeneralWire(self):
        """Convert this `Wire` into a `GeneralWire`."""
        p1, p2 = self.p1, self.p2
        return GeneralWire(lambda t: p1 + (p2 - p1) * t, 0, 1, self.current * u.A)


class InfiniteStraightWire(Wire):
    """
    Infinite straight wire class.

    Parameters
    ----------
    direction:
        Three-dimensional direction vector of the wire, also the
        positive current direction.

    p0 : `~astropy.units.Quantity`
        One point on the wire.

    current : `~astropy.units.Quantity`
        Electric current.
    """

    @validate_quantities
    def __init__(self, direction, p0: u.m, current: u.A):
        self.direction = direction / np.linalg.norm(direction)
        self.p0 = p0.value
        self._p0_u = p0.unit
        self.current = current.value
        self._current_u = current.unit

    def __repr__(self):
        name = self.__class__.__name__
        direction = self.direction
        p0 = self.p0
        current = self.current
        p0_u = self._p0_u
        current_u = self._current_u
        return (
            f"{name}(direction={direction}, p0={p0}{p0_u}, "
            f"current={current}{current_u})"
        )

    def magnetic_field(self, p) -> u.T:
        r"""
        Calculate magnetic field generated by this wire at position `p`

        Parameters
        ----------
        p : `astropy.units.Quantity`
            Three-dimensional position vector.

        Returns
        -------
        B : `astropy.units.Quantity`
            Magnetic field at the specified position.

        Notes
        -----
        .. math:
            \vec B = \frac{μ_0 I}{2π r}*(\vec l^0\times \vec{PP_0})^0,
            \text{where}\, \vec l^0\, \text{is the unit vector of current direction},
            r\, \text{is the perpendicular distance between} P_0 \text{and the infinite wire}
        """
        r = np.cross(self.direction, p - self.p0)
        B_unit = r / np.linalg.norm(r)
        r = np.linalg.norm(r)

        return B_unit / r * constants.mu0.value / 2 / np.pi * self.current * u.T


class CircularWire(Wire):
    """
    Circular wire(coil) class.

    Parameters
    ----------
    normal :
        Three-dimensional normal vector of the circular coil.

    center : `~astropy.units.Quantity`
        Three-dimensional position vector of the circular coil's center.

    radius: `~astropy.units.Quantity`
        Radius of the circular coil.

    current: `~astropy.units.Quantity`
        Electric current.
    """

    def __repr__(self):
        name = (self.__class__.__name__,)
        normal = (self.normal,)
        center = (self.center,)
        radius = (self.radius,)
        current = (self.current,)
        center_u = (self._center_u,)
        radius_u = (self._radius_u,)
        current_u = (self._current_u,)
        return (
            f"{name}(normal={normal}, center={center}{center_u}, "
            f"radius={radius}{radius_u}, current={current}{current_u})"
        )

    @validate_quantities
    def __init__(self, normal, center: u.m, radius: u.m, current: u.A, n=300):
        self.normal = normal / np.linalg.norm(normal)
        self.center = center.value
        self._center_u = center.unit
        if radius > 0:
            self.radius = radius.value
            self._radius_u = radius.unit
        else:
            raise ValueError("Radius should bu larger than 0")
        self.current = current.value
        self._current_u = current.unit

        # parametric equation
        # find other two axises in the disc plane
        z = np.array([0, 0, 1])
        axis_x = np.cross(z, self.normal)
        axis_y = np.cross(self.normal, axis_x)

        if np.linalg.norm(axis_x) == 0:
            axis_x = np.array([1, 0, 0])
            axis_y = np.array([0, 1, 0])
        else:
            axis_x = axis_x / np.linalg.norm(axis_x)
            axis_y = axis_y / np.linalg.norm(axis_y)

        self.axis_x = axis_x
        self.axis_y = axis_y

        def curve(t):
            if isinstance(t, np.ndarray):
                t = np.expand_dims(t, 0)
                axis_x_mat = np.expand_dims(axis_x, 1)
                axis_y_mat = np.expand_dims(axis_y, 1)
                return self.radius * (
                    np.matmul(axis_x_mat, np.cos(t)) + np.matmul(axis_y_mat, np.sin(t))
                ) + np.expand_dims(self.center, 1)
            else:
                return (
                    self.radius * (np.cos(t) * axis_x + np.sin(t) * axis_y)
                    + self.center
                )

        self.curve = curve

        self.roots_legendre = scipy.special.roots_legendre(n)
        self.n = n

    def magnetic_field(self, p) -> u.T:
        r"""
        Calculate magnetic field generated by this wire at position ``p``.

        Parameters
        ----------
        p : `~astropy.units.Quantity`
            Three-dimensional position vector.

        Returns
        -------
        B : `~astropy.units.Quantity`
            Magnetic field at the specified position.

        Notes
        -----
        .. math:
            \vec B
            = \frac{μ_0 I}{4π}
            \int \frac{d\vec l\times(\vec p - \vec l(t))}{|\vec p - \vec l(t)|^3}\\
            = \frac{μ_0 I}{4π} \int_{-π}^{π} {(-r\sin θ \hat x + r\cos θ \hat y)}
            \times \frac{\vec p - \vec l(t)}{|\vec p - \vec l(t)|^3} dθ

        We use ``n`` points using Gauss-Legendre quadrature to compute
        the integral. The default ``n`` is 300.
        """
        x, w = self.roots_legendre
        t = x * np.pi
        pt = self.curve(t)
        dl = self.radius * (
            -np.matmul(np.expand_dims(self.axis_x, 1), np.expand_dims(np.sin(t), 0))
            + np.matmul(np.expand_dims(self.axis_y, 1), np.expand_dims(np.cos(t), 0))
        )  # (3, n)

        r = np.expand_dims(p, 1) - pt  # (3, n)
        r_norm_3 = np.linalg.norm(r, axis=0) ** 3
        ft = np.cross(dl, r, axisa=0, axisb=0) / np.expand_dims(r_norm_3, 1)  # (n, 3)

        return (
            np.pi
            * np.matmul(np.expand_dims(w, 0), ft).squeeze(0)
            * constants.mu0.value
            / 4
            / np.pi
            * self.current
            * u.T
        )

    def to_GeneralWire(self):
        """Convert this `Wire` into a `GeneralWire`."""
        return GeneralWire(self.curve, -np.pi, np.pi, self.current * u.A)
