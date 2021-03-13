"""
Class representing a group of particles.
"""
__all__ = ["ParticleTracker"]

import astropy.units as u
import numpy as np
import scipy.interpolate as interp
import warnings

from astropy import constants

from plasmapy.particles import atomic
from plasmapy.utils.decorators import validate_quantities

from . import particle_integrators


class ParticleTracker:
    """
    Object representing a species of particles: ions, electrons, or simply
    a group of particles with a particular initial velocity distribution.

    Parameters
    ----------
    plasma : `Plasma`
        plasma from which fields can be pulled
    type : str
        particle type. See `plasmapy.particles.atomic` for suitable arguments.
        The default is a proton.
    n_particles : int
        number of macroparticles. The default is a single particle.
    scaling : float
        number of particles represented by each macroparticle.
        The default is 1, which means a 1:1 correspondence between particles
        and macroparticles.
    dt : `astropy.units.Quantity`
        length of timestep
    nt : int
        number of timesteps

    Attributes
    ----------
    x : `astropy.units.Quantity`
    v : `astropy.units.Quantity`
        Current position and velocity, respectively. Shape (n, 3).
    position_history : `astropy.units.Quantity`
    velocity_history : `astropy.units.Quantity`
        History of position and velocity. Shape (nt, n, 3).
    q : `astropy.units.Quantity`
    m : `astropy.units.Quantity`
        Charge and mass of particle.
    eff_q : `astropy.units.Quantity`
    eff_m : `astropy.units.Quantity`
        Total charge and mass of macroparticle.

    Examples
    ----------
    See `Particle Stepper Notebook`_.

    .. _`Particle Stepper Notebook`: ../notebooks/particle_stepper.ipynb
    """

    integrators = {
        "explicit_boris": particle_integrators.boris_push,
    }

    # Work In Progress integrators (throws a warning if used)
    _wip_integrators = {
        "explicit_boris_relativistic": particle_integrators.boris_push_relativistic,
    }

    _all_integrators = dict(**integrators, **_wip_integrators)

    @validate_quantities(dt=u.s)
    def __init__(
        self,
        plasma,
        particle_type="p",
        n_particles=1,
        scaling=1,
        dt=np.inf * u.s,
        nt=np.inf,
        integrator="explicit_boris",
    ):

        if np.isinf(dt) and np.isinf(nt):  # coverage: ignore
            raise ValueError("Both dt and nt are infinite.")

        self.q = atomic.integer_charge(particle_type) * constants.e.si
        self.m = atomic.particle_mass(particle_type)
        self.N = int(n_particles)
        self.scaling = scaling
        self.eff_q = self.q * scaling
        self.eff_m = self.m * scaling

        self.plasma = plasma

        self.dt = dt
        self.NT = int(nt)
        self.t = np.arange(nt) * dt

        self.x = np.zeros((n_particles, 3), dtype=float) * u.m
        self.v = np.zeros((n_particles, 3), dtype=float) * (u.m / u.s)
        self.name = particle_type

        self.position_history = np.zeros((self.NT, *self.x.shape), dtype=float) * u.m
        self.velocity_history = np.zeros((self.NT, *self.v.shape), dtype=float) * (
            u.m / u.s
        )
        # create intermediate array of dimension (nx,ny,nz,3) in order to allow
        # interpolation on non-equal spatial domain dimensions
        _B = np.moveaxis(self.plasma.magnetic_field.si.value, 0, -1)
        _E = np.moveaxis(self.plasma.electric_field.si.value, 0, -1)

        if integrator not in self.integrators.keys():
            warnings.warn(
                f"The {integrator} integrator is still under development. "
                "Use at your own risk."
            )

        self.integrator = self._all_integrators[integrator]

        self._B_interpolator = interp.RegularGridInterpolator(
            (self.plasma.x.si.value, self.plasma.y.si.value, self.plasma.z.si.value),
            _B,
            method="linear",
            bounds_error=True,
        )

        self._E_interpolator = interp.RegularGridInterpolator(
            (self.plasma.x.si.value, self.plasma.y.si.value, self.plasma.z.si.value),
            _E,
            method="linear",
            bounds_error=True,
        )

    def _interpolate_fields(self):
        interpolated_b = self._B_interpolator(self.x.si.value) * u.T
        interpolated_e = self._E_interpolator(self.x.si.value) * u.V / u.m
        return interpolated_b, interpolated_e

    @property
    def kinetic_energy_history(self):
        r"""
        Calculates the kinetic energy history for each particle.

        Returns
        --------
        ~astropy.units.Quantity
            Array of kinetic energies, shape (nt, n).
        """
        return (self.velocity_history ** 2).sum(axis=-1) * self.eff_m / 2

    def boris_push(self, init=False):
        r"""
        Implements the Boris algorithm for moving particles and updating their
        velocities.

        Arguments
        ----------
        init : bool (optional)
            If `True`, does not change the particle positions and sets dt
            to -dt/2.

        Notes
        ----------
        The Boris algorithm is the standard energy conserving algorithm for
        particle movement in plasma physics. See [1]_ for more details.

        Conceptually, the algorithm has three phases:

        1. Add half the impulse from electric field.
        2. Rotate the particle velocity about the direction of the magnetic
           field.
        3. Add the second half of the impulse from the electric field.

        This ends up causing the magnetic field action to be properly
        "centered" in time, and the algorithm conserves energy.

        References
        ----------
        .. [1] C. K. Birdsall, A. B. Langdon, "Plasma Physics via Computer
               Simulation", 2004, p. 58-63
        """
        b, e = self._interpolate_fields()

        if init:
            self.integrator(
                self.x.copy(), self.v, b, e, self.q, self.m, -0.5 * self.dt,
            )  # we don't want to change position here
        else:
            self.integrator(
                self.x, self.v, b, e, self.q, self.m, self.dt,
            )

    def run(self):
        r"""
        Runs a simulation instance.
        """
        self.boris_push(init=True)
        self.position_history[0] = self.x
        self.velocity_history[0] = self.v
        for i in range(1, self.NT):
            self.boris_push()
            self.position_history[i] = self.x
            self.velocity_history[i] = self.v

    def __repr__(self, *args, **kwargs):
        return (
            f"Species(q={self.q:.4e},m={self.m:.4e},N={self.N},"
            f'name="{self.name}",NT={self.NT})'
        )

    def __str__(self):  # coverage: ignore
        return (
            f"{self.N} {self.scaling:.2e}-{self.name} with "
            f"q = {self.q:.2e}, m = {self.m:.2e}, "
            f"{self.saved_iterations} saved history "
            f"steps over {self.NT} iterations"
        )

    def plot_trajectories(self):  # coverage: ignore
        r"""Draws trajectory history."""
        import matplotlib.pyplot as plt

        from astropy.visualization import quantity_support
        from mpl_toolkits.mplot3d import Axes3D

        quantity_support()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        for p_index in range(self.N):
            r = self.position_history[:, p_index]
            x, y, z = r.T
            ax.plot(x, y, z)
        ax.set_title(self.name)
        ax.set_xlabel("$x$ position")
        ax.set_ylabel("$y$ position")
        ax.set_zlabel("$z$ position")
        plt.show()

    def plot_time_trajectories(self, plot="xyz"):  # coverage: ignore
        r"""
        Draws position history versus time.

        Parameters
        ----------
        plot : str (optional)
            Enable plotting of position component x, y, z for each of these
            letters included in `plot`.
        """
        import matplotlib.pyplot as plt

        from astropy.visualization import quantity_support
        from mpl_toolkits.mplot3d import Axes3D

        quantity_support()
        fig, ax = plt.subplots()
        for p_index in range(self.N):
            r = self.position_history[:, p_index]
            x, y, z = r.T
            if "x" in plot:
                ax.plot(self.t, x, label=f"x_{p_index}")
            if "y" in plot:
                ax.plot(self.t, y, label=f"y_{p_index}")
            if "z" in plot:
                ax.plot(self.t, z, label=f"z_{p_index}")
        ax.set_title(self.name)
        ax.legend(loc="best")
        ax.grid()
        plt.show()

    def test_kinetic_energy(self):
        r"""Test conservation of kinetic energy."""
        assert np.allclose(
            self.kinetic_energy_history,
            self.kinetic_energy_history.mean(),
            atol=3 * self.kinetic_energy_history.std(),
        ), "Kinetic energy is not conserved!"
