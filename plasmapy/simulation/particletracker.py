"""Class representing a group of particles moving in a plasma's fields."""

__all__ = ["ParticleTracker", "ParticleTrackerAccessor"]

import numpy as np
import pathlib
import tqdm.auto
import warnings
import xarray

from astropy import units as u
from typing import Iterable, Union

from plasmapy import formulary, particles
from plasmapy.particles import particle_input
from plasmapy.plasma import BasePlasma
from plasmapy.utils.decorators import validate_quantities

from . import particle_integrators


@xarray.register_dataset_accessor("particletracker")
class ParticleTrackerAccessor:
    # TODO turn these into functions
    """Custom accessor for PlasmaPy particle simulations."""

    def __init__(self, xarray_obj: xarray.Dataset):
        self._obj = xarray_obj
        # TODO handle CustomParticles on the `Particle` layer!
        self.particle = particles.Particle(xarray_obj.attrs["particle"])

    def vector_norm(self, array: xarray.DataArray, order: int = None):
        """
        Calculates the norm of a vector quantity.

        Parameters
        ----------
        array : xarray.DataArray
            Array to calculate vector norm of.
        order : int
            Order of vector norm passed to numpy.linalg.norm
        """
        return xarray.apply_ufunc(
            np.linalg.norm,
            self._obj[array],
            input_core_dims=[["dimension"]],
            kwargs={"ord": order, "axis": -1},
        )

    def plot_trajectories(self, *args, **kwargs):  # coverage: ignore
        r"""Draws trajectory history."""
        import matplotlib.pyplot as plt

        from astropy.visualization import quantity_support
        from mpl_toolkits.mplot3d import Axes3D

        quantity_support()
        fig = plt.figure()
        axis = fig.add_subplot(111, projection="3d")
        for p_index in range(self._obj.particle.size):
            r = self._obj.position.isel(particle=p_index)
            x, y, z = r.T
            axis.plot(x, y, z, *args, **kwargs)
        axis.set_xlabel("$x$ position")
        axis.set_ylabel("$y$ position")
        axis.set_zlabel("$z$ position")
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
        fig, axis = plt.subplots()
        for p_index in range(self._obj.particle.size):
            r = self._obj.position.isel(particle=p_index)
            x, y, z = r.T
            if "x" in plot:
                axis.plot(self._obj.time, x, label=f"x_{p_index}")
            if "y" in plot:
                axis.plot(self._obj.time, y, label=f"y_{p_index}")
            if "z" in plot:
                axis.plot(self._obj.time, z, label=f"z_{p_index}")
        axis.legend(loc="best")
        axis.grid()
        axis.set_xlabel(f"Time $t$ [{u.s}]")
        axis.set_ylabel(f"Position [{u.m}]")
        plt.show()

    def visualize(
        self, figure=None, particles=(0,), stride=1, plasma=None, name=None
    ):  # coverage: ignore
        """Plot the trajectory using PyVista."""
        import pyvista as pv

        if figure is None:
            fig = pv.Plotter()
            fig.add_axes()
        else:
            fig = figure
        for i in particles:
            points = self._obj.position.sel(particle=i)[::stride].values
            trajectory = pv.Spline(
                points,
                max((1000, self._obj.sizes["time"] // 100)),  # TODO clean this up!
            )
            fig.add_mesh(trajectory, name=name)

        if plasma is not None:
            if hasattr(plasma, "visualize"):
                plasma.visualize(fig)
            else:
                warnings.warn(
                    f"Your plasma={plasma} has no `visualize` method, there's nothing to display!"
                )

        return fig

    def animate(
        self,
        filename: Union[str, pathlib.Path],
        particles: Iterable[int] = None,
        nframes: int = 50,
        plasma: BasePlasma = None,
        notebook_display: bool = False,
        plot_trajectories: bool = True,
        plot_arrows: bool = True,
    ):
        """
        Produces a 3D animation of particle trajectories using the pyvista library.

        Parameters
        ----------

        filename: str or pathlib.Path
            File path for the saved animation (.mp4).
        particles: iterable of int
            indices of particles to display. By default, displays all particles.
            This may be resource intensive.
        nframes: int
            Number of frames to display. By default, 50.
        plasma: `plasmapy.classes.BasePlasma`
            Optional plasmapy Plasma object; if it has a `visualize` method,
            that is used to display the environment the particles move in
        notebook_display: bool
            If True, displays the animation from the saved file.
            Use for interactive work in Jupyter notebooks.
        plot_trajectories: bool
            If True, draws trajectories as curves.
        plot_arrows: bool
            If True, draws arrows representing current particle positions
            and velocities.

        """
        import pyvista as pv

        if particles is None:
            particles = (0,)

        fig = pv.Plotter(off_screen=True)
        fig.open_movie(str(filename))
        if plasma is not None:
            if hasattr(plasma, "visualize"):
                plasma.visualize(fig)
            else:
                warnings.warn(
                    f"Plasma object {plasma} passed to animate, but it has no visualize method!"
                )
        for renderer in fig.renderers:
            if not renderer.camera_set:
                renderer.camera_position = renderer.get_default_cam_pos()
                renderer.ResetCamera()
        fig.write_frame()
        abs_vel = self.vector_norm("velocity")
        self._obj["|v|"] = (("time", "particle"), abs_vel)
        if plot_arrows:
            vectors = self._obj["velocity"] / (10 * self._obj["|v|"])

        for i in tqdm.auto.trange(1, nframes):
            fig.clear()
            if plasma is not None:
                if hasattr(plasma, "visualize"):
                    plasma.visualize(fig)
                else:
                    warnings.warn(
                        f"Plasma object {plasma} passed to animate, but it has no visualize method!"
                    )
            frame_max = self._obj.sizes["time"] // nframes * i
            if plot_trajectories:
                for particle in particles:
                    trajectories = (
                        self._obj.position.sel(particle=particle)
                        .isel(time=range(0, frame_max + 1))
                        .values
                    )
                    trajectory = pv.Spline(trajectories)
                    fig.add_mesh(trajectory)

            points = (
                self._obj.position.sel(particle=list(particles))
                .isel(time=frame_max)
                .values
            )

            point_cloud = pv.PolyData(points)
            point_cloud["velocity"] = self._obj["|v|"].isel(time=frame_max)
            velocity_range = [
                self._obj["|v|"].min().item(),
                self._obj["|v|"].max().item(),
            ]
            if plot_arrows:
                point_cloud["vectors"] = vectors.isel(time=frame_max)
                point_vectors = point_cloud.glyph(
                    orient="vectors", scale=False, factor=0.15
                )
                fig.add_mesh(point_vectors)
            fig.add_mesh(
                point_cloud,
                scalars="velocity",
                clim=velocity_range,
                render_points_as_spheres=True,
            )
            fig.write_frame()
        fig.close()
        if notebook_display:
            from IPython.display import display, Video

            display(Video(str(filename), embed=True))


class ParticleTracker:
    """
    A group of particles moving in a plasma's electromagnetic field.

    Parameters
    ----------
    plasma : `Plasma`
        plasma from which fields can be pulled
    type : str
        particle type. See `plasmapy.particles.atomic` for suitable arguments.
        The default is a proton.
    x: `astropy.units.Quantity`
    v: `astropy.units.Quantity`
        Initial conditions for position and velocity, with a shape of (N, 3),
        N being the number of particles in your simulation.

    Attributes
    ----------
    _x : `np.ndarray`
    _v : `np.ndarray`
        Current position and velocity, without units. Shape (N, 3).
    q : `astropy.units.Quantity`
    m : `astropy.units.Quantity`
        Charge and mass of particle.

    Examples
    ----------
    See `Particle Stepper Notebook`_.

    .. _`Particle Stepper Notebook`: ../notebooks/simulation/particle_stepper.ipynb
    """

    integrators = {"explicit_boris": particle_integrators.boris_push}

    _wip_integrators = {}

    _all_integrators = dict(**integrators, **_wip_integrators)

    @particle_input
    @validate_quantities
    def __init__(
        self,
        plasma,
        x: u.m = None,
        v: u.m / u.s = None,
        particle_type: particles.Particle = "p",
    ):

        if x is not None and v is not None:
            pass
        elif x is None and v is None:
            x = u.Quantity(np.zeros((1, 3)), u.m)
            v = u.Quantity(np.zeros((1, 3)), u.m / u.s)
        elif v is not None:
            x = u.Quantity(np.zeros((v.shape)), u.m)
        elif x is not None:
            v = u.Quantity(np.zeros((x.shape)), u.m / u.s)
        self.x = x
        self.v = v
        self.particle = particle_type
        self.plasma = plasma

        assert v.shape == x.shape
        self.N, dims = x.shape
        assert dims == 3

        self._check_field_size()

    def _check_field_size(self):
        b = self.plasma.interpolate_B(self.x)
        e = self.plasma.interpolate_E(self.x)
        if b.shape != self.x.shape:
            raise ValueError(
                f"""Invalid shape {b.shape} for the magnetic field array!
                `plasma.interpolate_B` must return an array of shape (N, 3),
                where N is the number of particles in the simulation, currently {self.N}."""
            )
        if e.shape != self.x.shape:
            raise ValueError(
                f"""Invalid shape {e.shape} for the electric field array!
                `plasma.interpolate_E` must return an array of shape (N, 3),
                where N is the number of particles in the simulation, currently {self.N}."""
            )

    @validate_quantities()
    def run(
        self,
        total_time: u.s,
        dt: u.s = None,
        progressbar=True,
        pusher="explicit_boris",
        snapshot_steps=1000,
    ):
        r"""Run a simulation instance.

        dt: u.s = np.inf * u.s,
        nt: int = np.inf,
        """
        integrator = self._all_integrators[pusher]
        if dt is None:
            b = np.linalg.norm(self.plasma.interpolate_B(self.x), axis=-1)
            gyroperiod = (1 / formulary.gyrofrequency(b, self.particle, to_hz=True)).to(
                u.s
            )
            dt = gyroperiod.min() / 10
            warnings.warn(
                f"Set timestep to {dt:.3e}, 1/10 of smallest gyroperiod", UserWarning
            )

        _dt = dt.si.value
        _q = self.particle.charge.si.value
        _m = self.particle.mass.si.value

        _total_time = total_time.si.value
        _time = 0.0
        actual_timesteps = _total_time // _dt
        if snapshot_steps == None:
            snapshot_steps = 1
        else:
            snapshot_steps = min([snapshot_steps, actual_timesteps])
        _snapshot_timestep = _total_time / snapshot_steps
        next_snapshot_update_time = _time + _snapshot_timestep
        _times = [_time]
        _timesteps = [_dt]
        _x = self.x.si.value.copy()
        _v = self.v.si.value.copy()
        i = 0

        init_kinetic = self.kinetic_energy(_v, _m)
        if init_kinetic:
            reldelta = self.kinetic_energy(_v, _m) / init_kinetic - 1
        else:
            delta = self.kinetic_energy(_v, _m)

        with np.errstate(all="raise"):
            b = self.plasma._interpolate_B(_x)
            e = self.plasma._interpolate_E(_x)

            integrator(
                _x.copy(), _v, b, e, _q, _m, -0.5 * _dt
            )  # we don't want to change position here

            _position_history = [_x.copy()]
            _velocity_history = [_v.copy()]
            _b_history = [b.copy()]
            _e_history = [e.copy()]
            if hasattr(
                self.plasma, "potentials"
            ):  # FIXME this is a hack for inter-particle interactions
                potential_history = [self.plasma.potentials.copy()]
            else:
                potential_history = None

            if progressbar:
                pbar = tqdm.auto.tqdm(
                    total=snapshot_steps,
                    bar_format="{l_bar}{bar}| [{elapsed}<{remaining}, "
                    "{rate_fmt}{postfix}]",
                )

            while _time < _total_time:
                _time += _dt
                i += 1
                b = self.plasma._interpolate_B(_x)
                e = self.plasma._interpolate_E(_x)
                integrator(_x, _v, b, e, _q, _m, _dt)

                if _time > next_snapshot_update_time:
                    next_snapshot_update_time += _snapshot_timestep
                    _position_history.append(_x.copy())
                    _velocity_history.append(_v.copy())
                    _b_history.append(b.copy())
                    _e_history.append(e.copy())
                    if hasattr(self.plasma, "potentials"):
                        potential_history.append(self.plasma.potentials.copy())
                    _times.append(_time)
                    _timesteps.append(_dt)

                    diagnostics = dict(i=i, dt=_dt)
                    if init_kinetic:
                        reldelta = self.kinetic_energy(_v, _m) / init_kinetic - 1
                        diagnostics["Relative kinetic energy change"] = reldelta
                    else:
                        delta = self.kinetic_energy(_v, _m)
                        diagnostics["Kinetic energy change"] = delta
                    pbar.set_postfix(diagnostics)
                    pbar.update()
        if progressbar:
            pbar.close()

        solution = self._create_xarray(
            u.Quantity(_position_history, u.m),
            u.Quantity(_velocity_history, u.m / u.s),
            u.Quantity(_times, u.s),
            u.Quantity(_b_history, u.T),
            u.Quantity(_e_history, u.V / u.m),
            u.Quantity(_timesteps, u.s),
            total_iterations=i,
            potentials=u.Quantity(potential_history, u.J)
            if potential_history is not None
            else None,
        )
        return solution

    def kinetic_energy(self, _v=None, _m=None):
        if _v is None:
            _v = self.v
        if _m is None:
            _m = self.particle.mass
        return (_v ** 2).sum() * _m / 2

    def __repr__(self, *args, **kwargs):
        return (
            f"ParticleTracker(plasma={self.plasma}, particle_type={self.particle},"
            f" N = {self.x.shape[0]})"
        )

    @validate_quantities
    def _create_xarray(
        self,
        position_history: u.m,
        velocity_history: u.m / u.s,
        times: u.s,
        b_history: u.T,
        e_history: u.V / u.m,
        timesteps: u.s,
        total_iterations: int,
        dimensions="xyz",
        potentials=None,
    ):
        # TODO replace scheme with OpenPMD! I can't believe I forgot about it!
        data_vars = {}
        assert position_history.shape == velocity_history.shape
        data_vars["position"] = (("time", "particle", "dimension"), position_history)
        data_vars["velocity"] = (("time", "particle", "dimension"), velocity_history)
        data_vars["B"] = (("time", "particle", "dimension"), b_history)
        data_vars["E"] = (("time", "particle", "dimension"), e_history)
        data_vars["timestep"] = (("time",), timesteps)
        kinetic_energy = (velocity_history ** 2).sum(axis=-1) * self.particle.mass / 2
        data_vars["kinetic_energy"] = (("time", "particle"), kinetic_energy)
        if potentials is not None:
            data_vars["potential_energy"] = (("time", "particle"), potentials)

        data = xarray.Dataset(
            data_vars=data_vars,
            coords={
                "time": times,
                "particle": range(position_history.shape[1]),
                "dimension": list(dimensions),
            },
        )
        for index, quantity in [
            ("position", position_history),
            ("velocity", velocity_history),
            ("time", times),
            ("B", b_history),
            ("E", e_history),
            ("timestep", timesteps),
            ("kinetic_energy", kinetic_energy),
            ("potential_energy", potentials),
        ]:
            if index in data:
                data[index].attrs["unit"] = str(quantity.unit)

        data.attrs["particle"] = str(self.particle)
        data.attrs["total_iterations"] = total_iterations
        return data
