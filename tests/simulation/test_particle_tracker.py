"""
Tests for particle_tracker.py
"""

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import fsolve

from plasmapy.formulary.collisions.misc import Mott_Bethe_mean_squared_scattering
from plasmapy.formulary.frequencies import gyrofrequency
from plasmapy.formulary.lengths import gyroradius
from plasmapy.formulary.relativity import RelativisticBody
from plasmapy.particles import particle_input
from plasmapy.particles.particle_class import CustomParticle, Particle, ParticleLike
from plasmapy.plasma import Plasma
from plasmapy.plasma.grids import CartesianGrid
from plasmapy.simulation.particle_integrators import BorisIntegrator
from plasmapy.simulation.particle_tracker.particle_tracker import ParticleTracker
from plasmapy.simulation.particle_tracker.save_routines import IntervalSaveRoutine
from plasmapy.simulation.particle_tracker.termination_conditions import (
    AllParticlesOffGridTerminationCondition,
    NoParticlesOnGridsTerminationCondition,
    TimeElapsedTerminationCondition,
)
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning, RelativityWarning

rng = np.random.default_rng()


@pytest.fixture()
def no_particles_on_grids_instantiated():
    return NoParticlesOnGridsTerminationCondition()


@pytest.fixture()
def time_elapsed_termination_condition_instantiated():
    return TimeElapsedTerminationCondition(1 * u.s)


@pytest.fixture()
def disk_interval_save_routine_instantiated(tmp_path):
    return IntervalSaveRoutine(1 * u.s, output_directory=tmp_path)


@pytest.fixture()
def memory_interval_save_routine_instantiated():
    return IntervalSaveRoutine(1 * u.s)


@pytest.fixture()
def grid_with_inf_entry():
    grid = CartesianGrid(-1 * u.m, 1 * u.m)
    entry = np.full(grid.shape, np.nan) * u.V / u.m
    grid.add_quantities(E_x=entry)

    return grid


@pytest.mark.parametrize(
    ("grids", "termination_condition", "save_routine", "kwargs", "expected_exception"),
    [
        # Old ParticleTracker construction deprecation error
        (
            Plasma(
                domain_x=np.linspace(-1, 1, 10) * u.m,
                domain_y=np.linspace(-1, 1, 10) * u.m,
                domain_z=np.linspace(-1, 1, 10) * u.m,
            ),
            "no_particles_on_grids_instantiated",
            None,
            {},
            TypeError,
        ),
        # Unrecognized grid type
        (42, "time_elapsed_termination_condition_instantiated", None, {}, TypeError),
        # Unrecognized termination condition
        (CartesianGrid(-1 * u.m, 1 * u.m), ("lorem ipsum",), None, {}, TypeError),
        (
            CartesianGrid(-1 * u.m, 1 * u.m),
            "time_elapsed_termination_condition_instantiated",
            ("lorem ipsum",),
            {},
            TypeError,
        ),
        # Specifies dt and dt_range error
        (
            CartesianGrid(-1 * u.m, 1 * u.m),
            "no_particles_on_grids_instantiated",
            None,
            {"dt": 1e-2 * u.s, "dt_range": [1e-2 * u.s, 5e-2 * u.s]},
            ValueError,
        ),
        # Infinite/NaN entry in grid object
        (
            "grid_with_inf_entry",
            "no_particles_on_grids_instantiated",
            None,
            {},
            ValueError,
        ),
        # Invalid field weighting
        (
            CartesianGrid(-1 * u.m, 1 * u.m),
            "no_particles_on_grids_instantiated",
            None,
            {"field_weighting": "lorem ipsum"},
            ValueError,
        ),
    ],
)
def test_particle_tracker_constructor_errors(
    request, grids, termination_condition, save_routine, kwargs, expected_exception
) -> None:
    if isinstance(grids, str):
        grids = request.getfixturevalue(grids)

    if termination_condition is not None and isinstance(termination_condition, str):
        termination_condition = request.getfixturevalue(termination_condition)

    if save_routine is not None and isinstance(save_routine, str):
        save_routine = request.getfixturevalue(save_routine)

    with pytest.raises(expected_exception):
        ParticleTracker(grids, termination_condition, save_routine, **kwargs)


@pytest.mark.parametrize(
    ("grids", "termination_condition", "save_routine", "kwargs"),
    [
        (
            [CartesianGrid(-2 * u.m, 1 * u.m), CartesianGrid(-1 * u.m, 2 * u.m)],
            "no_particles_on_grids_instantiated",
            None,
            {},
        ),
        (
            CartesianGrid(-1 * u.m, 1 * u.m),
            "no_particles_on_grids_instantiated",
            None,
            {"req_quantities": ["rho"]},
        ),
    ],
)
def test_particle_tracker_construction(
    request, grids, termination_condition, save_routine, kwargs
) -> None:
    termination_condition = request.getfixturevalue(termination_condition)

    if save_routine is not None:
        save_routine = request.getfixturevalue(save_routine)

    ParticleTracker(grids, termination_condition, save_routine, **kwargs)


def test_particle_tracker_load_particles_shape_error(
    no_particles_on_grids_instantiated,
) -> None:
    """Inconsistent shape for x and v error"""
    grid = CartesianGrid(-1 * u.m, 1 * u.m)

    simulation = ParticleTracker(grid, no_particles_on_grids_instantiated)

    with pytest.raises(ValueError):
        simulation.load_particles(
            [[0, 0, 0]] * u.m, [[0, 0, 0], [0, 0, 0]] * u.m / u.s, Particle("p+")
        )


class TestParticleTrackerGyroradius:
    v_x = rng.integers(1, 10, size=100) * u.m / u.s

    v = np.array([[v_x_element.value, 0, 0] for v_x_element in v_x]) * u.m / u.s

    B_strength = rng.integers(1, 10) * u.T

    point_particle = CustomParticle(1 * u.kg, 1 * u.C)

    # Set the initial position to the gyroradius
    # This means the particle will orbit the origin
    R_L = gyroradius(B_strength, point_particle, Vperp=v_x)
    x = np.array([[0, R_L_element.value, 0] for R_L_element in R_L]) * u.m

    L = 1e6 * u.km
    num = 2
    grid = CartesianGrid(-L, L, num=num)
    grid_shape = (num,) * 3

    Bz = np.full(grid_shape, B_strength) * u.T
    grid.add_quantities(B_z=Bz)

    termination_time = 5 * np.max(1 / gyrofrequency(Bz, point_particle)).to(
        u.s, equivalencies=u.dimensionless_angles()
    )
    termination_condition = TimeElapsedTerminationCondition(termination_time)
    save_routine = IntervalSaveRoutine(termination_time / 10)

    simulation = ParticleTracker(grid, termination_condition, save_routine)
    simulation.setup_adaptive_time_step(time_steps_per_gyroperiod=100)
    simulation.load_particles(x, v, point_particle)

    simulation.run()

    def test_gyroradius(self) -> None:
        """Test to ensure particles maintain their gyroradius over time"""
        positions = self.save_routine.results["x"]
        distances = np.linalg.norm(positions, axis=-1)

        assert np.isclose(distances, self.R_L, rtol=5e-2).all()

    def test_kinetic_energy(self) -> None:
        """Test to ensure particles maintain their gyroradius over time"""

        initial_kinetic_energies = 0.5 * self.point_particle.mass * self.v_x**2

        velocities = self.save_routine.results["v"]
        speeds = np.linalg.norm(velocities, axis=-1)
        simulation_kinetic_energies = 0.5 * self.point_particle.mass * speeds**2

        assert np.isclose(initial_kinetic_energies, simulation_kinetic_energies).all()


@pytest.mark.slow()
@given(st.integers(1, 10), st.integers(1, 10), st.integers(1, 10), st.integers(1, 10))
@settings(deadline=2e4, max_examples=10)
def test_particle_tracker_potential_difference(
    request, E_strength, L, mass, charge
) -> None:
    # Apply appropriate units to the random inputs
    E_strength = E_strength * u.V / u.m
    L = L * u.m
    mass = mass * u.kg
    charge = charge * u.C

    num = 2
    dt = 1e-2 * u.s

    grid = CartesianGrid(-L, L, num=num)
    grid_shape = (num,) * 3

    Ex = np.full(grid_shape, E_strength) * u.V / u.m
    grid.add_quantities(E_x=Ex)

    point_particle = CustomParticle(mass, charge)

    x = [[0, 0, 0]] * u.m
    v = [[0, 0, 0]] * u.m / u.s

    termination_condition = request.getfixturevalue(
        "no_particles_on_grids_instantiated"
    )
    save_routine = request.getfixturevalue("memory_interval_save_routine_instantiated")

    simulation = ParticleTracker(
        grid,
        termination_condition,
        save_routine,
        dt=dt,
        field_weighting="nearest neighbor",
    )
    simulation.load_particles(x, v, point_particle)

    simulation.run()

    velocities = save_routine.results["v"]
    velocities_particle = velocities[:, 0]

    speeds = np.linalg.norm(velocities_particle, axis=-1)

    # Final energy is given by the product of the charge and potential difference
    final_expected_energy = (E_strength * L * point_particle.charge).to(u.J)
    final_simulated_energy = (0.5 * point_particle.mass * speeds[-1] ** 2).to(u.J)

    assert np.isclose(
        final_expected_energy, final_simulated_energy, atol=0.5, rtol=5e-2
    )


def test_asynchronous_time_step(no_particles_on_grids_instantiated) -> None:
    E_strength = 1 * u.V / u.m
    L = 1 * u.m
    mass = 1 * u.kg
    charge = 1 * u.C

    num = 2

    grid = CartesianGrid(-L, L, num=num)
    grid_shape = (num,) * 3

    Ex = np.full(grid_shape, E_strength) * u.V / u.m
    grid.add_quantities(E_x=Ex)

    point_particle = CustomParticle(mass, charge)

    x = [[0, 0, 0], [0, 0, 0]] * u.m
    v = [[0, 0, 1], [1, 0, 0]] * u.m / u.s

    termination_condition = no_particles_on_grids_instantiated

    simulation = ParticleTracker(grid, termination_condition)

    # Particles not loaded error
    with pytest.raises(ValueError):
        simulation.run()

    simulation.load_particles(x, v, point_particle)

    assert simulation.is_adaptive_time_step
    assert not simulation.is_synchronized_time_step

    simulation.run()

    with pytest.raises(RuntimeError):
        simulation.load_particles(x, v, point_particle)


def test_asynchronous_time_step_error(
    memory_interval_save_routine_instantiated, no_particles_on_grids_instantiated
) -> None:
    E_strength = 1 * u.V / u.m
    L = 1 * u.m

    num = 2

    grid = CartesianGrid(-L, L, num=num)
    grid_shape = (num,) * 3

    Ex = np.full(grid_shape, E_strength) * u.V / u.m
    grid.add_quantities(E_x=Ex)

    termination_condition = no_particles_on_grids_instantiated
    save_routine = memory_interval_save_routine_instantiated

    with pytest.raises(ValueError):
        ParticleTracker(
            grid, termination_condition, save_routine, dt=[1e-2, 2e-2] * u.s
        )


def test_nearest_neighbor_interpolation(
    time_elapsed_termination_condition_instantiated,
) -> None:
    E_strength = 1 * u.V / u.m
    L = 1 * u.m
    mass = 1 * u.kg
    charge = 1 * u.C

    num = 2

    grid = CartesianGrid(-L, L, num=num)
    grid_shape = (num,) * 3

    Ex = np.full(grid_shape, E_strength) * u.V / u.m
    grid.add_quantities(E_x=Ex)

    point_particle = CustomParticle(mass, charge)

    x = [[0, 0, 0]] * u.m
    v = [[0, 0, 0]] * u.m / u.s

    termination_condition = time_elapsed_termination_condition_instantiated

    simulation = ParticleTracker(
        grid, termination_condition, field_weighting="nearest neighbor"
    )
    simulation.load_particles(x, v, point_particle)

    simulation.run()


def test_setup_adaptive_time_step(
    time_elapsed_termination_condition_instantiated,
) -> None:
    E_strength = 1 * u.V / u.m
    L = 1 * u.m
    mass = 1 * u.kg
    charge = 1 * u.C

    num = 2

    grid = CartesianGrid(-L, L, num=num)
    grid_shape = (num,) * 3

    Ex = np.full(grid_shape, E_strength) * u.V / u.m
    grid.add_quantities(E_x=Ex)

    point_particle = CustomParticle(mass, charge)

    x = [[0, 0, 0]] * u.m
    v = [[0, 0, 0]] * u.m / u.s

    termination_condition = time_elapsed_termination_condition_instantiated

    simulation = ParticleTracker(grid, termination_condition)
    simulation.load_particles(x, v, point_particle)

    simulation.setup_adaptive_time_step(
        time_steps_per_gyroperiod=25, Courant_parameter=0.25
    )

    simulation.run()


def test_particle_tracker_stop_particles(request) -> None:
    E_strength = 1 * u.V / u.m
    L = 1 * u.m
    mass = 1 * u.kg
    charge = 1 * u.C

    num = 2
    dt = 1e-2 * u.s

    grid = CartesianGrid(-L, L, num=num)
    grid_shape = (num,) * 3

    Ex = np.full(grid_shape, E_strength) * u.V / u.m
    grid.add_quantities(E_x=Ex)

    point_particle = CustomParticle(mass, charge)

    x = [[0, 0, 0]] * u.m
    v = [[0, 0, 0]] * u.m / u.s

    termination_condition = request.getfixturevalue(
        "no_particles_on_grids_instantiated"
    )
    save_routine = request.getfixturevalue("memory_interval_save_routine_instantiated")

    simulation = ParticleTracker(grid, termination_condition, save_routine, dt=dt)
    simulation.load_particles(x, v, point_particle)

    # Not an adaptive time step error
    with pytest.raises(ValueError):
        simulation.setup_adaptive_time_step(time_steps_per_gyroperiod=15)

    simulation.run()
    simulation._stop_particles([True])

    # Number of entries in mask must be equal to number of particles
    with pytest.raises(ValueError):
        simulation._stop_particles([True, True])

    assert np.isnan(simulation.v[0, :]).all()
    assert not np.isnan(simulation.x[0, :]).all()

    simulation._remove_particles([True])

    # Number of entries in mask must be equal to number of particles
    with pytest.raises(ValueError):
        simulation._remove_particles([True, True])

    assert np.isnan(simulation.x[0, :]).all()


# @pytest.mark.parametrize(
#     ("kwargs", "expected_error", "match_string"),
#     [
#         (
#             {"method": "NIST", "target": [Particle("Al")], "target_rho": [2.69890e00] * u.g / u.cm**3},
#             ValueError,
#             "Please provide an array of length ngrids for the materials.",
#         ),
#         (
#             {
#                 "method": "NIST",
#                 "materials": ["ALUMINUM", "ALUMINUM"],
#                 "target": [Particle("Al")],
#                 "target_rho": [2.69890e00] * u.g / u.cm**3
#             },
#             ValueError,
#             "Please provide an array of length ngrids for the materials.",
#         ),
#         (
#             {
#                 "method": "Lorem Ipsum",
#                 "target": [Particle("Al")],
#                 "target_rho": [2.69890e00] * u.g / u.cm**3
#             },
#             ValueError,
#             "Please provide one of 'NIST' or 'Bethe' for the method keyword.",
#         ),
#         (
#             {"method": "Bethe", "target": [Particle("Al")], "target_rho": [2.69890e00] * u.g / u.cm**3},
#             ValueError,
#             "Please provide an array of length ngrids for the mean excitation energy.",
#         ),
#         (
#             {"method": "Bethe", "target": [Particle("Al")], "target_rho": [2.69890e00] * u.g / u.cm**3, "I": [0, 0] * u.eV},
#             ValueError,
#             "Please provide an array of length ngrids for the mean excitation energy.",
#         ),
#     ],
# )
# def test_particle_tracker_add_stopping_errors(
#     no_particles_on_grids_instantiated,
#     memory_interval_save_routine_instantiated,
#     kwargs,
#     expected_error,
#     match_string,
# ) -> None:
#     E_strength = 1 * u.V / u.m
#     L = 1 * u.m
#
#     num = 2
#     dt = 1e-2 * u.s
#
#     grid = CartesianGrid(-L, L, num=num)
#     grid_shape = (num,) * 3
#
#     Ex = np.full(grid_shape, E_strength) * u.V / u.m
#     grid.add_quantities(E_x=Ex)
#
#     x = [[0, 0, 0]] * u.m
#     v = [[0, 0, 0]] * u.m / u.s
#
#     termination_condition = no_particles_on_grids_instantiated
#     save_routine = memory_interval_save_routine_instantiated
#
#     simulation = ParticleTracker(grid, termination_condition, save_routine, dt=dt)
#     simulation.load_particles(x, v, Particle("p+"))
#
#     with pytest.raises(expected_error, match=re.escape(match_string)):
#         simulation.add_stopping(**kwargs)
#
#     with pytest.warns(
#         RuntimeWarning, match="The density is not defined on any of the provided grids!"
#     ):
#         simulation.add_stopping(method="NIST", target=["ALUMINUM"])


def test_particle_tracker_Bethe_warning(
    no_particles_on_grids_instantiated,
    memory_interval_save_routine_instantiated,
) -> None:
    L = 1 * u.m

    num = 20

    grid = CartesianGrid(-L, L, num=num)

    x = [[0, 0, 0]] * u.m
    v = [[10, 0, 0]] * u.m / u.s

    termination_condition = no_particles_on_grids_instantiated
    save_routine = memory_interval_save_routine_instantiated

    simulation = ParticleTracker(
        grid, termination_condition, save_routine, dt=1e-5 * u.s
    )
    simulation.load_particles(x, v, Particle("p+"))

    target = [Particle("Al-27")]

    simulation.add_stopping(
        method="Bethe",
        target=target,
        target_rho=[2.69890e00] * u.g / u.cm**3,
        I=[166] * u.eV,
    )

    with pytest.warns(
        PhysicsWarning, match="The Bethe model is only valid for high energy particles."
    ):
        simulation.run()


# def test_particle_tracker_scattering_warning(
#     no_particles_on_grids_instantiated, memory_interval_save_routine_instantiated
# ):
#     L = 1 * u.m
#
#     num = 2
#
#     grid = CartesianGrid(-L, L, num=num)
#     grid_shape = (num,) * 3
#
#     x = [[0, 0, 0]] * u.m
#     v = [[0, 0, 0]] * u.m / u.s
#
#     termination_condition = no_particles_on_grids_instantiated
#     save_routine = memory_interval_save_routine_instantiated
#
#     simulation = ParticleTracker(grid, termination_condition, save_routine)
#     simulation.load_particles(x, v, Particle("p+"))
#
#     with pytest.warns(
#         RuntimeWarning,
#         match="The electron number density is not defined on any of the provided grids!",
#     ):
#         simulation.add_scattering("Al-27 0+", lambda: 42, "mean square rate")
#
#     n_e = np.full(grid_shape, 1) * u.m**-3
#     grid.add_quantities(n_e=n_e)
#
#     with pytest.warns(
#         RuntimeWarning,
#         match="The electron temperature is not defined on any of the provided grids!",
#     ):
#         simulation.add_scattering("Al-27 0+", lambda: 42, "mean square rate")
#
#     T_e = np.full(grid_shape, 1) * u.K
#     grid.add_quantities(T_e=T_e)
#
#     with pytest.raises(
#         ValueError,
#         match="Please pass one of 'differential cross-section' or 'mean square rate'",
#     ):
#         simulation.add_scattering("Al-27 0+", lambda: 42, "Lorem ipsum")


PARTICLES_PER_ENERGY = 100


@pytest.mark.parametrize(
    (
        "target_material",
        "target_density",
        "target_temperature",
        "NIST_material_string",
        "beam_energies",
        "stopping_ranges",
        "expected_lateral_scattering",
    ),
    [
        (
            "Al-27",
            [
                2.69890e00,
            ]
            * u.g
            / u.cm**3,
            298 * u.K,
            "ALUMINUM",
            [1, 5, 10] * u.MeV,
            [16.33, 215.93, 709.23] * u.um,
            [0.9972, 10.29, 31.70] * u.um,
        )
    ],
)
@particle_input
@validate_quantities
def test_particle_tracker_ranges_straggling(
    target_material: ParticleLike,
    target_density: u.Quantity[u.kg / u.m**3],
    target_temperature: u.Quantity[u.K],
    NIST_material_string: str,
    beam_energies: u.Quantity[u.J],
    stopping_ranges: u.Quantity[u.m],
    expected_lateral_scattering: u.Quantity[u.m],
):
    """
    Test both the stopping range and scattering distribution of the particle
    tracker for a variety of materials and energies.

    """
    # TODO: add implementation-specific details to this docstring

    beam_particle = Particle("p+")

    # Let the width of the grid be dynamic between materials based on the
    # maximum stopping range
    width = np.max(stopping_ranges)
    grid = CartesianGrid(
        [-0.2, 0.0, -0.2] * u.cm, [0.2, width.to(u.cm).value, 0.2] * u.cm
    )

    # Construct the quantity arrays for the grid
    grid_shape_identity = np.ones(grid.shape)
    temperature = grid_shape_identity * target_temperature
    grid.add_quantities(T_i=temperature)

    # Create the simulation
    termination_condition = AllParticlesOffGridTerminationCondition()
    sim = ParticleTracker(grid, termination_condition=termination_condition, seed=42)

    # Load particles as a beam source collinear with the y-axis
    nparticles = beam_energies.shape[0] * PARTICLES_PER_ENERGY

    x0 = np.zeros(shape=(nparticles, 3))
    v0 = np.zeros(shape=(beam_energies.shape[0], PARTICLES_PER_ENERGY, 3))

    initial_speed = RelativisticBody(
        particle=beam_particle, kinetic_energy=beam_energies
    ).velocity

    # The speed value should be copied into the y-component along the second
    # axis
    # TODO: figure out a cleaner way of doing this
    for i, speed in enumerate(initial_speed):
        v0[i, :, 1] = speed.si.value

    v0 = np.reshape(v0, newshape=(nparticles, 3))
    sim.load_particles(x0 * u.m, v0 * u.m / u.s, particle=beam_particle)

    # Add stopping and scattering to the simulation
    sim.add_stopping("NIST", target_material, target_density, NIST_material_string)
    sim.add_scattering(
        target_material,
        target_density,
        scatter_routine=Mott_Bethe_mean_squared_scattering,
        scatter_routine_type="mean square rate",
    )

    sim.run()

    # Extract the component of the position that is collinear with the detector normal
    # Reshape the result into an array of shape [energies, particles_per_energy]
    x_final = (
        np.reshape(sim.x[:, 1], newshape=(beam_energies.shape[0], PARTICLES_PER_ENERGY))
        * u.m
    )

    assert np.isclose(np.mean(x_final, axis=-1), stopping_ranges, rtol=0.15).all()

    xloc, yloc = sim.x[:, 0], sim.x[:, 2]
    xloc, yloc = (
        np.reshape(xloc, newshape=(beam_energies.shape[0], PARTICLES_PER_ENERGY)),
        np.reshape(yloc, newshape=(beam_energies.shape[0], PARTICLES_PER_ENERGY)),
    )

    not_nan_mask = ~(np.isnan(xloc) | np.isnan(yloc))

    lat_straggling = np.sqrt(
        np.nansum(((np.abs(xloc) + np.abs(yloc)) / 2) ** 2, axis=1)
        / np.sum(not_nan_mask, axis=1)
    )
    lat_straggling *= u.m

    assert np.isclose(lat_straggling.T, expected_lateral_scattering, rtol=0.25).all()


class TestParticleTrajectory:
    @staticmethod
    def laboratory_time_case_one(𝜏, vd, γd, ν):
        """
        fsolve optimization function
        Eq. 72 in the Friedman paper

        """
        return (
            γd.si.value**2
            / ν.si.value
            * (
                ν.si.value * 𝜏
                - vd.si.value**2 / const.c.si.value**2 * np.sin(ν.si.value * 𝜏)
            )
        )

    @classmethod
    def ExB_trajectory_case_one(  # noqa: ANN206
        cls,
        t,
        E,
        B,
        q=const.e.si,
        m=const.m_p.si,
        is_relativistic: bool = True,
    ):
        """
        Calculates the relativistically-correct ExB drift trajectory for a
        particle starting at x=v=0 at t=0 with E in the x direction and
        B in the y direction and ExB in the z direction. The motion is 2D in the
        xz plane.

        From https://journals.aps.org/pre/abstract/10.1103/PhysRevE.72.026603

        """

        if E >= const.c.si * B:
            raise ValueError("Currently this function only works for E<cB")

        # Just after Eq. 41
        vd = (E / B).to(u.m / u.s)

        # Eq. 34
        ν = q * np.sqrt((const.c.si * B) ** 2 - E**2) / (m * const.c.si)

        if is_relativistic:
            # Eq. 45
            γd = 1 / np.sqrt(1 - vd**2 / const.c.si**2)

            # Numerically invert Eq. 72 to calculate the proper time for each time
            # t in the lab frame
            𝜏 = (
                fsolve(
                    lambda 𝜏: t.si.value - cls.laboratory_time_case_one(𝜏, vd, γd, ν),
                    t.si.value,
                )
                * u.s
            )

            # Calculate the positions
            # Eq. 71
            a = γd * vd / ν
            x = (a * γd * (ν * 𝜏 - np.sin(ν.si.value * 𝜏.si.value))).to(u.m)
            z = (a * (np.cos(ν.si.value * 𝜏.si.value) - 1)).to(u.m)
        else:
            # Calculate the positions using the non-relativistic expression
            # Eq. 79
            a = m * vd / (q * B)
            x = (a * (ν * t - np.sin(ν.si.value * t.si.value))).to(u.m)
            z = (a * (np.cos(ν.si.value * t.si.value) - 1)).to(u.m)

        return x, z

    @staticmethod
    def construct_field(
        grid, magnitude, direction
    ) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
        # add third dimension to account for the fact that we are dealing with a vector field
        field = (
            np.full(fill_value=magnitude * direction, shape=(*grid.shape, 3))
            * magnitude.unit
        )
        f_x, f_y, f_z = np.moveaxis(field, -1, 0)

        return f_x, f_y, f_z

    trajectory_tolerance_parameters = {"rtol": 0.03}

    @classmethod
    @pytest.mark.parametrize(
        ("regime", "particle"),
        [
            (0.01, Particle("p+")),
            (0.01, Particle("e-")),
            (0.5, Particle("p+")),
            (0.5, Particle("e-")),
            (0.9, Particle("p+")),
            (0.9, Particle("e-")),
        ],
    )
    @pytest.mark.slow()
    def test_relativistic_Boris_integrator_fitting(cls, regime, particle) -> None:
        """
        Fit the results of the relativistic Boris integrator using
        relativistic models developed in https://www.sciencedirect.com/science/article/pii/S163107211400148X
        """

        N_PERIODS_RECORDED = 5
        B_0 = 10 * u.T
        E_0 = regime * const.c * B_0
        B_dir = np.asarray([0, 0, 1])
        E_dir = np.asarray([0, 1, 0])

        q = particle.charge
        m = particle.mass
        vd = E_0 / B_0
        ν = q * np.sqrt((const.c * B_0) ** 2 - E_0**2) / (m * const.c)
        γd = 1 / np.sqrt(1 - vd**2 / const.c**2)

        # Convert period in proper time to the time elapsed in the laboratory frame
        proper_period = np.abs(2 * np.pi / ν).to(
            u.s, equivalencies=u.dimensionless_angles()
        )
        period = cls.laboratory_time_case_one(proper_period.si.value, vd, γd, ν) * u.s

        L = vd * period * (N_PERIODS_RECORDED + 1)
        fields = CartesianGrid(-L, L)
        E_x, E_y, E_z = cls.construct_field(fields, E_0, E_dir)
        B_x, B_y, B_z = cls.construct_field(fields, B_0, B_dir)
        fields.add_quantities(
            E_x=E_x,
            E_y=E_y,
            E_z=E_z,
            B_x=B_x,
            B_y=B_y,
            B_z=B_z,
        )

        """
        ----------
        Simulation
        ----------
        """
        termination_condition = TimeElapsedTerminationCondition(
            N_PERIODS_RECORDED * period
        )

        save_routine = IntervalSaveRoutine(period / 10)

        simulation = ParticleTracker(
            grids=fields,
            save_routine=save_routine,
            termination_condition=termination_condition,
        )

        simulation.load_particles(
            x=[np.zeros(3)] * u.m,
            v=[np.zeros(3)] * u.m / u.s,
            particle=particle,
        )

        simulation.run()

        """
        --------------
        Theory Fitting
        --------------
        """
        relativistic_theory_x, relativistic_theory_z = cls.ExB_trajectory_case_one(
            save_routine.results["time"],
            E_0,
            B_0,
            q=particle.charge,
            m=particle.mass,
        )

        # Discard the first five points due to large relative error.
        assert np.isclose(
            relativistic_theory_x[5:],
            save_routine.results["x"][5:, 0, 0],
            equal_nan=True,
            rtol=0.05,
        ).all()

        # Ensure that a non-relativistic analytic solution does not appear to fit
        # the relativistic trajectory within the defined tolerance parameters.
        if regime >= 0.5:
            # Calculate the classical analytic trajectory with all else being equal
            calculate_theory_x, calculate_theory_z = cls.ExB_trajectory_case_one(
                save_routine.results["time"],
                E_0,
                B_0,
                q=particle.charge,
                m=particle.mass,
                is_relativistic=False,
            )

            # Calculate the sum of the squares of the residuals for both the classical
            # and relativistic analytic trajectories as compared against the simulated
            # trajectory
            classical_r_squared = np.sum(
                (save_routine.results["x"][:, 0, 0] - calculate_theory_x) ** 2
            )
            relativistic_r_squared = np.sum(
                (save_routine.results["x"][:, 0, 0] - relativistic_theory_x) ** 2
            )

            assert classical_r_squared.si.value / relativistic_r_squared.si.value > 100

    @classmethod
    @pytest.mark.parametrize(
        ("regime", "particle"),
        [
            (0.01, Particle("p+")),
            (0.01, Particle("e-")),
            (0.1, Particle("p+")),
            (0.1, Particle("e-")),
        ],
    )
    def test_classical_Boris_integrator_fitting(cls, regime, particle) -> None:
        """
        Fit the results of the non-relativistic Boris integrator using
        relativistic models developed in https://www.sciencedirect.com/science/article/pii/S163107211400148X
        """

        N_PERIODS_RECORDED = 5
        B_0 = 10 * u.T
        E_0 = regime * const.c * B_0
        B_dir = np.asarray([0, 0, 1])
        E_dir = np.asarray([0, 1, 0])

        q = particle.charge
        m = particle.mass
        vd = E_0 / B_0
        ν = q * np.sqrt((const.c * B_0) ** 2 - E_0**2) / (m * const.c)
        γd = 1 / np.sqrt(1 - vd**2 / const.c**2)

        # Convert period in proper time to the time elapsed in the laboratory frame
        proper_period = np.abs(2 * np.pi / ν).to(
            u.s, equivalencies=u.dimensionless_angles()
        )
        period = cls.laboratory_time_case_one(proper_period.si.value, vd, γd, ν) * u.s

        L = vd * period * (N_PERIODS_RECORDED + 1)
        fields = CartesianGrid(-L, L)
        E_x, E_y, E_z = cls.construct_field(fields, E_0, E_dir)
        B_x, B_y, B_z = cls.construct_field(fields, B_0, B_dir)
        fields.add_quantities(
            E_x=E_x,
            E_y=E_y,
            E_z=E_z,
            B_x=B_x,
            B_y=B_y,
            B_z=B_z,
        )

        """
        ----------
        Simulation
        ----------
        """
        termination_condition = TimeElapsedTerminationCondition(
            N_PERIODS_RECORDED * period
        )

        save_routine = IntervalSaveRoutine(period / 10)

        simulation = ParticleTracker(
            grids=fields,
            save_routine=save_routine,
            termination_condition=termination_condition,
            particle_integrator=BorisIntegrator,
        )

        simulation.load_particles(
            x=[np.zeros(3)] * u.m,
            v=[np.zeros(3)] * u.m / u.s,
            particle=particle,
        )

        simulation.run()

        """
        --------------
        Theory Fitting
        --------------
        """
        classical_theory_x, classical_theory_z = cls.ExB_trajectory_case_one(
            save_routine.results["time"],
            E_0,
            B_0,
            q=particle.charge,
            m=particle.mass,
            is_relativistic=False,
        )

        # Discard the first five points due to large relative error.
        assert np.isclose(
            classical_theory_x[5:],
            save_routine.results["x"][5:, 0, 0],
            equal_nan=True,
            rtol=0.05,
        ).all()

    @classmethod
    @pytest.mark.parametrize(
        ("regime", "particle"),
        [
            (0.3, Particle("p+")),
            (0.3, Particle("e-")),
            (0.5, Particle("p+")),
            (0.5, Particle("e-")),
            (0.9, Particle("p+")),
            (0.9, Particle("e-")),
        ],
    )
    @pytest.mark.slow()
    def test_relativity_warning(cls, regime, particle) -> None:
        """
        Fit the results of the non-relativistic Boris integrator using
        relativistic models developed in https://www.sciencedirect.com/science/article/pii/S163107211400148X
        """

        N_PERIODS_RECORDED = 5
        B_0 = 10 * u.T
        E_0 = regime * const.c * B_0
        B_dir = np.asarray([0, 0, 1])
        E_dir = np.asarray([0, 1, 0])

        q = particle.charge
        m = particle.mass
        vd = E_0 / B_0
        ν = q * np.sqrt((const.c * B_0) ** 2 - E_0**2) / (m * const.c)
        γd = 1 / np.sqrt(1 - vd**2 / const.c**2)

        # Convert period in proper time to the time elapsed in the laboratory frame
        proper_period = np.abs(2 * np.pi / ν).to(
            u.s, equivalencies=u.dimensionless_angles()
        )
        period = cls.laboratory_time_case_one(proper_period.si.value, vd, γd, ν) * u.s

        L = vd * period * (N_PERIODS_RECORDED + 1)
        fields = CartesianGrid(-L, L)
        E_x, E_y, E_z = cls.construct_field(fields, E_0, E_dir)
        B_x, B_y, B_z = cls.construct_field(fields, B_0, B_dir)
        fields.add_quantities(
            E_x=E_x,
            E_y=E_y,
            E_z=E_z,
            B_x=B_x,
            B_y=B_y,
            B_z=B_z,
        )

        """
        ----------
        Simulation
        ----------
        """
        termination_condition = TimeElapsedTerminationCondition(
            N_PERIODS_RECORDED * period
        )

        save_routine = IntervalSaveRoutine(period / 10)

        simulation = ParticleTracker(
            grids=fields,
            save_routine=save_routine,
            termination_condition=termination_condition,
            particle_integrator=BorisIntegrator,
        )

        simulation.load_particles(
            x=[np.zeros(3)] * u.m,
            v=[np.zeros(3)] * u.m / u.s,
            particle=particle,
        )

        with pytest.warns(RelativityWarning):
            simulation.run()
