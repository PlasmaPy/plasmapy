"""
Particle movement integrators.

These do not have `astropy.units` support, choosing instead to limit
overhead and increase performance.

They act in-place on position and velocity arrays to reduce memory
allocation.
"""

__all__ = ["AbstractIntegrator", "BorisIntegrator", "RelativisticBorisIntegrator"]

from abc import ABC, abstractmethod

import astropy.constants as const
import numpy as np

_c = const.c


class AbstractIntegrator(ABC):
    """Outlines the necessary methods to define a particle integrator."""

    @property
    @abstractmethod
    def is_relativistic(self):
        r"""
        Property representing whether an integrator incorporates relativistic
        corrections.
        """
        ...

    @staticmethod
    @abstractmethod
    def push(x, v, B, E, q, m, dt):
        r"""
        The method for applying a push to the specified ensemble of particles.

        The passed parameters should in general not be `~astropy.units.Quantity`
        objects, but rather their SI values. Specific user implementations may vary.
        """
        ...


class BorisIntegrator(AbstractIntegrator):
    """The explicit Boris pusher."""

    @property
    def is_relativistic(self) -> bool:
        r"""
        The explicit Boris pusher is not relativistic.
        """
        return False

    @staticmethod
    def push(x, v, B, E, q, m, dt):
        r"""
        Parameters
        ----------
        x : `~numpy.ndarray`
            Particle position at full timestep, in SI (meter) units.

        v : `~numpy.ndarray`
            Particle velocity at half timestep, in SI (meter/second) units.

        B : `~numpy.ndarray`
            Magnetic field at full timestep, in SI (tesla) units.

        E : `float`
            Electric field at full timestep, in SI (V/m) units.

        q : `float`
            Particle charge, in SI (coulomb) units.

        m : `float`
            Particle mass, in SI (kg) units.

        dt : `float`
            Timestep, in SI (second) units.

        Returns
        -------
        x : `~numpy.ndarray`
            Particle position x after Boris push algorithm, in SI (meter) units.

        v : `~numpy.ndarray`
            Particle velocity after Boris push algorithm, in SI (meter/second) units.

        Examples
        --------
        >>> B = np.array([[0.0, 0.0, 5.0]])
        >>> E = np.array([[0.0, 0.0, 0.0]])
        >>> x_t0 = np.array([[0.0, 0.0, 0.0]])
        >>> v_t0 = np.array([[5.0, 0.0, 0.0]])
        >>> x_t1, v_t1 = BorisIntegrator.push(
        ...     x=x_t0, v=v_t0, B=B, E=E, q=1.0, m=1.0, dt=0.01
        ... )
        >>> x_t1
        array([[ 0.04993754, -0.00249844,  0.        ]])
        >>> v_t1
        array([[ 4.9937539 , -0.24984385,  0.        ]])
        >>> x_t1, v_t1 = BorisIntegrator.push(
        ...     x=x_t0, v=v_t0, B=B, E=E, q=1.0, m=1.0, dt=0.01
        ... )
        >>> x_t1
        array([[ 0.04993754, -0.00249844,  0.        ]])
        >>> v_t1
        array([[ 4.9937539 , -0.24984385,  0.        ]])
        >>> # B parallel to v
        >>> B = np.array([[0.0, 0.0, 5.0]])
        >>> v_t0 = np.array([[0.0, 0.0, 5.0]])
        >>> x_t0 = np.array([[0.0, 0.0, 0.0]])
        >>> x_t1, v_t1 = BorisIntegrator.push(
        ...     x=x_t0, v=v_t0, B=B, E=E, q=1.0, m=1.0, dt=0.01
        ... )
        >>> # no rotation of vector v
        >>> v_t1
        array([[0., 0., 5.]])
        >>> x_t1
        array([[0.  , 0.  , 0.05]])
        >>> # B perpendicular to v
        >>> B = np.array([[5.0, 0.0, 0.0]])
        >>> v_t0 = np.array([[0.0, 5.0, 0.0]])
        >>> x_t0 = np.array([[0.0, 0.0, 0.0]])
        >>> x_t1, v_t1 = BorisIntegrator.push(
        ...     x=x_t0, v=v_t0, B=B, E=E, q=1.0, m=1.0, dt=0.01
        ... )
        >>> # rotation of vector v
        >>> v_t1
        array([[ 0.        ,  4.9937539 , -0.24984385]])
        >>> x_t1
        array([[ 0.        ,  0.04993754, -0.00249844]])
        >>> # nonzero E and zero B
        >>> E = np.array([[1.0, 1.0, 1.0]])
        >>> B = np.array([[0.0, 0.0, 0.0]])
        >>> v_t0 = np.array([[0.0, 0.0, 5.0]])
        >>> x_t0 = np.array([[0.0, 0.0, 0.0]])
        >>> x_t1, v_t1 = BorisIntegrator.push(
        ...     x=x_t0, v=v_t0, B=B, E=E, q=1.0, m=1.0, dt=0.01
        ... )
        >>> v_t1
        array([[0.01, 0.01, 5.01]])
        >>> x_t1
        array([[0.0001, 0.0001, 0.0501]])
        >>> x_t1, v_t1 = BorisIntegrator.push(
        ...     x=x_t0, v=v_t0, B=B, E=E, q=1.0, m=1.0, dt=0.01
        ... )
        >>> v_t1
        array([[0.02, 0.02, 5.02]])
        >>> x_t1
        array([[0.0001, 0.0001, 0.0501]])

        Notes
        -----
        The Boris algorithm :cite:p:`boris:1970` is the standard energy
        conserving algorithm for particle movement in plasma physics. See
        :cite:t:`birdsall:2004` for more details, and this `page on the
        Boris method <https://www.particleincell.com/2011/vxb-rotation>`__
        for a nice overview.

        Conceptually, the algorithm has three phases:

        1. Add half the impulse from electric field.
        2. Rotate the particle velocity about the direction of the magnetic
           field.
        3. Add the second half of the impulse from the electric field.

        This ends up causing the magnetic field action to be properly "centered" in
        time, and the algorithm, being a symplectic integrator, conserves energy.
        """
        hqmdt = 0.5 * dt * q / m
        vminus = v + hqmdt * E

        # rotate to add magnetic field
        t = B * hqmdt
        s = 2 * t / (1 + (t * t).sum(axis=1, keepdims=True))
        vprime = vminus + np.cross(vminus, t)
        vplus = vminus + np.cross(vprime, s)

        # add second half of electric impulse
        v = vplus + hqmdt * E
        return x + v * dt, v


class RelativisticBorisIntegrator(AbstractIntegrator):
    """The explicit Boris pusher, including relativistic corrections."""

    @property
    def is_relativistic(self) -> bool:
        r"""
        The push implementation of the Boris push algorithm includes relativistic
        corrections.
        """
        return True

    @staticmethod
    def push(x, v, B, E, q, m, dt):
        r"""
        Parameters
        ----------
        x : `~numpy.ndarray`
            particle position at full timestep, in SI (meter) units.
        v : `~numpy.ndarray`
            particle velocity at half timestep, in SI (meter/second) units.
        B : `~numpy.ndarray`
            magnetic field at full timestep, in SI (tesla) units.
        E : float
            electric field at full timestep, in SI (V/m) units.
        q : float
            particle charge, in SI (Coulomb) units.
        m : float
            particle mass, in SI (kg) units.
        dt : float
            timestep, in SI (second) units.

        Notes
        -----
        For the basic overview of this algorithm, see `BorisIntegrator`. This
        version, based on [1]_, applies relativistic corrections such as
        the proper velocity and proper time transformations. These corrections
        affect the leapfrog scheme timestep. The addition of the impulse from
        the electric field must now account for the relativistic mass, resulting
        in an impulse reduced by a factor of :math:`\gamma` relative to the classical
        result. A similar adjustment occurs in the rotation due to the magnetic
        field, as the angle of rotation is roughly inversely proportional to the
        mass of the charged particle. More precisely, it is proportional to :math:`\arctan{(1 / m)}`.
        This means the angle of rotation will also decrease approximately by a
        factor of :math:`\gamma`.

        Keep in mind that the non-relativistic version will be slightly
        faster if you don't encounter velocities in relativistic regimes.

        References
        ----------
        .. [1] C. K. Birdsall, A. B. Langdon, "Plasma Physics via Computer
               Simulation", 2004, p. 58-63
        """
        γ = 1 / np.sqrt(
            1 - (np.linalg.norm(v, axis=1, keepdims=True) / _c.si.value) ** 2
        )
        uvel = v * γ

        uvel_minus = uvel + q * E * dt / (2 * m)

        γ1 = np.sqrt(
            1 + (np.linalg.norm(uvel_minus, axis=1, keepdims=True) / _c.si.value) ** 2
        )

        t = q * B * dt / (2 * γ1 * m)
        s = 2 * t / (1 + (t * t).sum(axis=1, keepdims=True))

        uvel_prime = uvel_minus + np.cross(uvel_minus, t)
        uvel_plus = uvel_minus + np.cross(uvel_prime, s)
        uvel_new = uvel_plus + q * E * dt / (2 * m)

        # You can show that this expression is equivalent to calculating
        # v_new  then calculating γnew using the usual formula
        γ2 = np.sqrt(
            1 + (np.linalg.norm(uvel_new, axis=1, keepdims=True) / _c.si.value) ** 2
        )

        v = uvel_new / γ2

        return x + v * dt, v
