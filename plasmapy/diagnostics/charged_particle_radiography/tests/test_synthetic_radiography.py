"""
Tests for proton radiography functions
"""
import astropy.units as u
import numpy as np
import os
import pytest

from scipy.special import erf

from plasmapy.diagnostics.charged_particle_radiography import (
    synthetic_radiography as cpr,
)
from plasmapy.plasma.grids import CartesianGrid
from plasmapy.utils.data import downloader


def _test_grid(
    name,
    L=1 * u.mm,
    num=100,
    B0=10 * u.T,
    E0=5e8 * u.V / u.m,
    phi0=1.4e5 * u.V,
    a=None,
    b=None,
):
    r"""
    Generates grids representing some common physical scenarios for testing
    and illustration. Valid example names are:
    * axially_magnetized_cylinder : A cylinder of radius L/4 magnetized in the
        Z-direction (like a solenoid, but without the fringe fields).
    * electrostatic_discontinuity : A discontinuity in the electric field at z=0
        with a radial gaussian profile in the xy plane.
    * electrostatic_gaussian_sphere : An electric field created by a sphere
        of potential of radius L/2 with a radial Gaussian distribution.
    Parameters
    ----------
    name : str
        Name of example to load (from list above)
    L : `~u.Quantity` (or array of three of the same)
        Length scale (or scales). -L and L are passed to the grid constructor
        as start and stop respectively. The default is 1 cm.
    num : int or list of three ints
        The number of points in each direction (or list of one for each dimension).
        Passed to the grid constructor as the num argument. The default is 100.

    E0, B0, phi0 : u.Quantities
        Scaling quantities used in the various examples

    a, b : u.Quantities
        Two length scales used in the various examples

    Returns
    -------
    grid : CartesianGrid
        A CartesianGrid object containing quantity arrays representing
        the chosen example.
    """

    grid = CartesianGrid(-L, L, num=num)

    # If an array was provided to the constructor, reduce to a single
    # length scale now.
    if L.size > 1:
        L = np.max(L)

    if name == "empty":
        pass

    elif name == "constant_bz":
        Bz = np.ones(grid.shape) * B0
        grid.add_quantities(B_z=Bz)

    elif name == "constant_ex":
        Ex = np.ones(grid.shape) * E0
        grid.add_quantities(E_x=Ex)

    elif name == "axially_magnetized_cylinder":
        if a is None:
            a = L / 4
        radius = np.linalg.norm(grid.grid[..., 0:2] * grid.unit, axis=3)
        Bz = np.where(radius < a, B0, 0 * u.T)
        grid.add_quantities(B_z=Bz)

    elif name == "electrostatic_discontinuity":
        if a is None:
            a = L / 2
        delta = a / 120

        radius = np.linalg.norm(grid.grid[..., 0:2] * grid.unit, axis=3)
        z = grid.grids[2]

        potential = (1 - erf(z / delta)) * np.exp(-((radius / a) ** 2)) * u.V

        Ex, Ey, Ez = np.gradient(potential, grid.dax0, grid.dax1, grid.dax2)

        grid.add_quantities(E_x=Ex, E_y=Ey, E_z=Ez, phi=potential)

    elif name == "electrostatic_gaussian_sphere":
        if a is None:
            a = L / 3
        if b is None:
            b = L / 2
        radius = np.linalg.norm(grid.grid, axis=3)
        arg = (radius / a).to(u.dimensionless_unscaled)
        potential = phi0 * np.exp(-(arg**2))

        Ex, Ey, Ez = np.gradient(potential, grid.dax0, grid.dax1, grid.dax2)

        Ex = np.where(radius < b, Ex, 0)
        Ey = np.where(radius < b, Ey, 0)
        Ez = np.where(radius < b, Ez, 0)

        grid.add_quantities(E_x=-Ex, E_y=-Ey, E_z=-Ez, phi=potential)

    else:
        raise ValueError(
            "No example corresponding to the provided name " f"({name}) exists."
        )

    # If any of the following quantities are missing, add them as empty arrays
    req_quantities = ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z"]

    for q in req_quantities:
        if q not in list(grid.ds.data_vars):
            unit = grid.recognized_quantities[q].unit
            arg = {q: np.zeros(grid.shape) * unit}
            grid.add_quantities(**arg)

    return grid


def test_multiple_grids():
    """
    Test that a case with two grids runs.

    TODO: automate test by including two fields with some obvious analytical
    solution??
    """

    grid1 = _test_grid("constant_bz", L=3 * u.cm, num=50, B0=0.7 * u.T)
    grid2 = _test_grid("electrostatic_gaussian_sphere", L=1 * u.mm, num=50)
    grids = [grid1, grid2]

    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    sim = cpr.Tracker(grids, source, detector, verbose=True)

    sim.create_particles(1e5, 15 * u.MeV, max_theta=8 * u.deg)

    sim.run(field_weighting="nearest neighbor")

    size = np.array([[-1, 1], [-1, 1]]) * 5 * u.cm
    bins = [100, 100]
    hax, vax, values = cpr.synthetic_radiograph(sim, size=size, bins=bins)

    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh(hax.to(u.cm).value, vax.to(u.cm).value, values.T)
    """


def run_1D_example(name):
    """
    Run a simulation through an example with parameters optimized to
    sum up to a lineout along x. The goal is to run a relatively fast
    sim with a quasi-1D field grid that can then be summed to get good
    enough statistics to use as a test.
    """
    grid = _test_grid(name, L=1 * u.mm, num=50)

    # Cartesian
    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    # Expect warnings because these fields aren't well-behaved at the edges
    with pytest.warns(
        RuntimeWarning, match="Fields should go to zero at edges of grid to avoid "
    ):
        sim = cpr.Tracker(grid, source, detector, verbose=False)
    sim.create_particles(1e4, 3 * u.MeV, max_theta=0.1 * u.deg)
    sim.run()

    size = np.array([[-1, 1], [-1, 1]]) * 10 * u.cm
    bins = [200, 60]
    hax, vax, values = cpr.synthetic_radiograph(sim, size=size, bins=bins)

    values = np.mean(values[:, 20:40], axis=1)

    return hax, values


def run_mesh_example(
    location=np.array([0, -2, 0]) * u.mm,
    extent=(2 * u.mm, 1.5 * u.mm),
    nwires=9,
    wire_diameter=20 * u.um,
    mesh_hdir=None,
    mesh_vdir=None,
    nparticles=1e4,
    problem="electrostatic_gaussian_sphere",
):
    """
    Takes all of the add_wire_mesh parameters and runs a standard example problem
    simulation using them.

    Returns the sim object for use in additional tests
    """

    grid = _test_grid(problem, num=100)
    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    sim = cpr.Tracker(grid, source, detector, verbose=False)

    sim.add_wire_mesh(
        location,
        extent,
        nwires,
        wire_diameter,
        mesh_hdir=mesh_hdir,
        mesh_vdir=mesh_vdir,
    )

    sim.create_particles(nparticles, 3 * u.MeV, max_theta=10 * u.deg)
    sim.run(field_weighting="nearest neighbor")

    return sim


@pytest.mark.slow
def test_1D_deflections():
    # Check B-deflection
    hax, lineout = run_1D_example("constant_bz")
    loc = hax[np.argmax(lineout)]
    assert np.isclose(loc.si.value, 0.0165, 0.005)

    # Check E-deflection
    hax, lineout = run_1D_example("constant_ex")
    loc = hax[np.argmax(lineout)]
    assert np.isclose(loc.si.value, 0.0335, 0.005)


@pytest.mark.slow
def test_coordinate_systems():
    """
    Check that specifying the same point in different coordinate systems
    ends up with identical source and detector vectors.
    """

    grid = _test_grid("empty")

    # Cartesian
    source = (-7.07 * u.mm, -7.07 * u.mm, 0 * u.mm)
    detector = (70.07 * u.mm, 70.07 * u.mm, 0 * u.mm)
    sim1 = cpr.Tracker(grid, source, detector, verbose=True)

    # Cylindrical
    source = (-1 * u.cm, 45 * u.deg, 0 * u.mm)
    detector = (10 * u.cm, 45 * u.deg, 0 * u.mm)
    sim2 = cpr.Tracker(grid, source, detector, verbose=False)

    # In spherical
    source = (-0.01 * u.m, 90 * u.deg, 45 * u.deg)
    detector = (0.1 * u.m, 90 * u.deg, 45 * u.deg)
    sim3 = cpr.Tracker(grid, source, detector, verbose=False)

    assert np.allclose(sim1.source, sim2.source, atol=1e-2)
    assert np.allclose(sim2.source, sim3.source, atol=1e-2)
    assert np.allclose(sim1.detector, sim2.detector, atol=1e-2)
    assert np.allclose(sim2.detector, sim3.detector, atol=1e-2)


@pytest.mark.slow
def test_input_validation():
    """
    Intentionally raise a number of errors.
    """

    # ************************************************************************
    # During initialization
    # ************************************************************************

    grid = _test_grid("electrostatic_gaussian_sphere")
    source = (-10 * u.mm, 90 * u.deg, 45 * u.deg)
    detector = (100 * u.mm, 90 * u.deg, 45 * u.deg)

    # Check that an error is raised when an input grid has a nan or infty value
    # First check NaN
    Ex = grid["E_x"]
    Ex[0, 0, 0] = np.nan * u.V / u.m
    grid.add_quantities(E_x=Ex)
    with pytest.raises(ValueError):
        sim = cpr.Tracker(grid, source, detector, verbose=False)
    Ex[0, 0, 0] = 0 * u.V / u.m

    Ex[0, 0, 0] = np.inf * u.V / u.m  # Reset element for the rest of the tests
    grid.add_quantities(E_x=Ex)
    with pytest.raises(ValueError):
        sim = cpr.Tracker(grid, source, detector, verbose=False)
    Ex[0, 0, 0] = 0 * u.V / u.m

    # Check what happens if a value is large relative to the rest of the array
    Ex[0, 0, 0] = 0.5 * np.max(Ex)
    grid.add_quantities(E_x=Ex)
    # with pytest.raises(ValueError):
    with pytest.warns(RuntimeWarning):
        sim = cpr.Tracker(grid, source, detector, verbose=False)
    Ex[0, 0, 0] = 0 * u.V / u.m

    # Raise error when source-to-detector vector doesn't pass through the
    # field grid
    source_bad = (10 * u.mm, -10 * u.mm, 0 * u.mm)
    detector_bad = (10 * u.mm, 100 * u.mm, 0 * u.mm)
    with pytest.raises(ValueError):
        sim = cpr.Tracker(grid, source_bad, detector_bad, verbose=False)

    # Test raises warning when one (or more) of the required fields is missing
    grid_bad = CartesianGrid(-1 * u.mm, 1 * u.mm, num=50)
    with pytest.warns(RuntimeWarning, match="is not specified for the provided grid."):
        sim = cpr.Tracker(grid_bad, source, detector, verbose=True)

    # ************************************************************************
    # During create_particles
    # ************************************************************************
    sim = cpr.Tracker(grid, source, detector, verbose=False)
    sim.create_particles(1e3, 15 * u.MeV, max_theta=0.99 * np.pi / 2 * u.rad)

    # ************************************************************************
    # During runtime
    # ************************************************************************

    sim = cpr.Tracker(grid, source, detector, verbose=False)
    sim.create_particles(1e3, 15 * u.MeV)

    # Test an invalid field weighting keyword
    with pytest.raises(ValueError):
        sim.run(field_weighting="not a valid field weighting")

    # ************************************************************************
    # During runtime
    # ************************************************************************
    # SYNTHETIC RADIOGRAPH ERRORS
    sim.run()

    # Choose a very small synthetic radiograph size that misses most of the
    # particles
    with pytest.warns(
        RuntimeWarning, match="of the particles are shown on this synthetic radiograph."
    ):
        size = np.array([[-1, 1], [-1, 1]]) * 1 * u.mm
        hax, vax, values = cpr.synthetic_radiograph(sim, size=size)


@pytest.mark.slow
def test_init():
    grid = _test_grid("electrostatic_gaussian_sphere", num=50)

    # Cartesian
    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    sim = cpr.Tracker(grid, source, detector, verbose=False)

    # Test manually setting hdir and vdir
    hdir = np.array([1, 0, 1])
    sim = cpr.Tracker(grid, source, detector, verbose=False, detector_hdir=hdir)

    # Test special case hdir == [0,0,1]
    source = (0 * u.mm, 0 * u.mm, -10 * u.mm)
    detector = (0 * u.mm, 0 * u.mm, 200 * u.mm)
    sim = cpr.Tracker(grid, source, detector, verbose=False)
    assert all(sim.det_hdir == np.array([1, 0, 0]))

    # Test that hdir is calculated correctly if src-det axis is anti-parallel to z
    source = (0 * u.mm, 0 * u.mm, 10 * u.mm)
    detector = (0 * u.mm, 0 * u.mm, -200 * u.mm)
    sim = cpr.Tracker(grid, source, detector, verbose=False)
    assert all(sim.det_hdir == np.array([1, 0, 0]))


@pytest.mark.slow
def test_create_particles():
    grid = _test_grid("electrostatic_gaussian_sphere", num=50)

    # Cartesian
    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    sim = cpr.Tracker(grid, source, detector, verbose=False)

    sim.create_particles(
        1e3, 15 * u.MeV, max_theta=0.1 * u.rad, distribution="monte-carlo"
    )

    sim.create_particles(1e3, 15 * u.MeV, max_theta=0.1 * u.rad, distribution="uniform")

    # Test specifying particle
    sim.create_particles(1e3, 15 * u.MeV, particle="e")


@pytest.mark.slow
def test_load_particles():

    grid = _test_grid("electrostatic_gaussian_sphere", num=50)

    # Cartesian
    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    sim = cpr.Tracker(grid, source, detector, verbose=False)
    sim.create_particles(1e3, 15 * u.MeV, max_theta=0.1 * u.rad, distribution="uniform")

    # Test adding unequal numbers of particles
    x = np.zeros([100, 3]) * u.m
    v = np.ones([150, 3]) * u.m / u.s
    with pytest.raises(ValueError):
        sim.load_particles(x, v)

    # Test creating particles with explicit keywords
    x = sim.x * u.m
    v = sim.v * u.m / u.s

    # Try setting particles going the wrong direction
    with pytest.warns(RuntimeWarning):
        sim.load_particles(x, -v)

    # Try specifying a larger ion (not a proton or electron)
    sim.load_particles(x, v, particle="C-12 +3")

    # Run the tracker to make sure everything works
    sim.run(field_weighting="nearest neighbor")


@pytest.mark.slow
def test_run_options():
    grid = _test_grid("electrostatic_gaussian_sphere", num=50)

    # Cartesian
    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)
    sim = cpr.Tracker(grid, source, detector, verbose=True)

    # Test that trying to call run() without creating particles
    # raises an exception
    with pytest.raises(ValueError):
        sim.run()

    sim = cpr.Tracker(grid, source, detector, verbose=True)
    sim.create_particles(1e4, 3 * u.MeV, max_theta=10 * u.deg)

    # Try running with nearest neighbor interpolator
    # Test manually setting a timestep
    sim.run(field_weighting="nearest neighbor", dt=1e-12 * u.s)

    # Test max_deflections
    sim.max_deflection

    # Test way too big of a max_theta
    sim = cpr.Tracker(grid, source, detector, verbose=True)
    sim.create_particles(1e4, 3 * u.MeV, max_theta=89 * u.deg)
    with pytest.warns(RuntimeWarning, match="of " "particles entered the field grid"):
        sim.run(field_weighting="nearest neighbor", dt=1e-12 * u.s)

    # Test extreme deflections -> warns user
    # This requires instantiating a whole new example field with a really
    # big B-field
    grid = _test_grid("constant_bz", num=50, B0=250 * u.T)
    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    # Expect warnings because these fields aren't well-behaved at the edges
    with pytest.warns(
        RuntimeWarning, match="Fields should go to zero at edges of grid to avoid "
    ):
        sim = cpr.Tracker(grid, source, detector, verbose=False)
    sim.create_particles(1e4, 3 * u.MeV, max_theta=0.1 * u.deg)
    with pytest.warns(
        RuntimeWarning,
        match="particles have been " "deflected away from the detector plane",
    ):
        sim.run(field_weighting="nearest neighbor", dt=1e-12 * u.s)
    # Calc max deflection: should be between 0 and pi/2
    # Note: that's only true because max_theta is very small
    # More generally, max_deflection can be a bit bigger than pi/2 for
    # particles that begin at an angle then deflect all the way around.
    assert 0 < sim.max_deflection.to(u.rad).value < np.pi / 2


def create_tracker_obj():
    # CREATE A RADIOGRAPH OBJECT
    grid = _test_grid("electrostatic_gaussian_sphere", num=50)
    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    sim = cpr.Tracker(grid, source, detector, verbose=False)
    sim.create_particles(int(1e4), 3 * u.MeV, max_theta=10 * u.deg)
    return sim


tracker_obj_simulated = create_tracker_obj().run(field_weighting="nearest neighbor")


@pytest.mark.slow
class TestSyntheticRadiograph:
    """
    Tests for
    `plasmapy.diagnostics.charged_particle_radiography.synthetic_radiograph`.
    """

    tracker_obj_not_simulated = create_tracker_obj()
    tracker_obj_simulated = create_tracker_obj()
    tracker_obj_simulated.run(field_weighting="nearest neighbor")
    sim_results = tracker_obj_simulated.results_dict.copy()

    @pytest.mark.parametrize(
        "args, kwargs, _raises",
        [
            # obj wrong type
            ((5,), {}, TypeError),
            # size wrong type
            ((tracker_obj_simulated,), {"size": "not a Quantity"}, TypeError),
            # size not convertible to meters
            ((sim_results,), {"size": 5 * u.ms}, ValueError),
            # size wrong shape
            ((sim_results,), {"size": [-1, 1] * u.cm}, ValueError),
            # simulation was never run
            ((tracker_obj_not_simulated,), {}, RuntimeError),
        ],
    )
    def test_raises(self, args, kwargs, _raises):
        """Test scenarios the raise an Exception."""
        with pytest.raises(_raises):
            cpr.synthetic_radiograph(*args, **kwargs)

    def test_warns(self):
        """
        Test warning when less than half the particles reach the detector plane.
        """
        sim_results = self.sim_results.copy()
        sim_results["nparticles"] = 3 * sim_results["nparticles"]
        with pytest.warns(RuntimeWarning):
            cpr.synthetic_radiograph(sim_results)

    @pytest.mark.parametrize(
        "args, kwargs, expected",
        [
            (
                # From a Tracker object
                (tracker_obj_simulated,),
                {},
                {
                    "xrange": [-0.036934532101889815, 0.03702186098974771] * u.m,
                    "yrange": [-0.03697401811598356, 0.037007901161403144] * u.m,
                    "bins": (200, 200),
                },
            ),
            (
                # From a dict
                (sim_results,),
                {},
                {
                    "xrange": [-0.036934532101889815, 0.03702186098974771] * u.m,
                    "yrange": [-0.03697401811598356, 0.037007901161403144] * u.m,
                    "bins": (200, 200),
                },
            ),
            (
                # From a dict
                (sim_results,),
                {"size": np.array([[-1, 1], [-1, 1]]) * 30 * u.cm, "bins": (200, 60)},
                {
                    "xrange": [-0.3, 0.3] * u.m,
                    "yrange": [-0.3, 0.3] * u.m,
                    "bins": (200, 60),
                },
            ),
        ],
    )
    def test_intensity_histogram(self, args, kwargs, expected):
        """Test several valid use cases."""
        results = cpr.synthetic_radiograph(*args, **kwargs)

        assert len(results) == 3

        x = results[0]
        assert isinstance(x, u.Quantity)
        assert x.unit == u.m
        assert x.shape == (expected["bins"][0],)
        assert np.isclose(np.min(x), expected["xrange"][0], rtol=1e4)
        assert np.isclose(np.max(x), expected["xrange"][1], rtol=1e4)

        y = results[1]
        assert isinstance(y, u.Quantity)
        assert y.unit == u.m
        assert y.shape == (expected["bins"][1],)
        assert np.isclose(np.min(y), expected["yrange"][0], rtol=1e4)
        assert np.isclose(np.max(y), expected["yrange"][1], rtol=1e4)

        histogram = results[2]
        assert isinstance(histogram, np.ndarray)
        assert histogram.shape == expected["bins"]

    def test_optical_density_histogram(self):
        """
        Test the optical density calculation is correct and stuffed
        with numpy.inf when the intensity is zero.
        """
        bins = (200, 60)
        size = np.array([[-1, 1], [-1, 1]]) * 30 * u.cm

        intensity_results = cpr.synthetic_radiograph(
            self.sim_results, size=size, bins=bins
        )
        od_results = cpr.synthetic_radiograph(
            self.sim_results, size=size, bins=bins, optical_density=True
        )

        assert np.allclose(intensity_results[0], od_results[0])
        assert np.allclose(intensity_results[1], od_results[1])

        intensity = intensity_results[2]
        zero_mask = intensity == 0
        initial_intensity = np.mean(intensity[~zero_mask])
        optical_density = -np.log10(intensity / initial_intensity)

        assert np.allclose(optical_density[~zero_mask], od_results[2][~zero_mask])
        assert np.all(np.isposinf(od_results[2][zero_mask]))


@pytest.mark.slow
def test_saving_output(tmp_path):
    """Test behavior of Tracker.save_results."""

    sim = create_tracker_obj()

    # Test that output cannot be saved prior to running
    with pytest.raises(RuntimeError):
        _ = sim.results_dict

    sim.run(field_weighting="nearest neighbor")

    results_1 = sim.results_dict

    # Save result
    path = tmp_path / "temp.npz"
    sim.save_results(path)

    # Load result
    results_2 = dict(np.load(path, "r", allow_pickle=True))

    assert set(results_1.keys()) == set(results_2.keys())
    for key in results_1.keys():
        assert np.allclose(results_1[key], results_2[key])


@pytest.mark.slow
@pytest.mark.parametrize(
    "case",
    ["creating particles", "loading particles", "adding a wire mesh"],
)
def test_cannot_modify_simulation_after_running(case):
    """
    Test that a Tracker objection can not be modified after it is
    run (Tracker.run).
    """

    sim = create_tracker_obj()
    sim.run(field_weighting="nearest neighbor")

    # Error from creating particles
    with pytest.raises(RuntimeError):
        if case == "creating particles":
            sim.create_particles(1e4, 3 * u.MeV, max_theta=10 * u.deg)
        elif case == "loading particles":
            sim.load_particles(sim.x, sim.v)
        elif case == "adding a wire mesh":
            sim.add_wire_mesh(
                np.array([0, -2, 0]) * u.mm,
                (2 * u.mm, 1.5 * u.mm),
                9,
                20 * u.um,
            )
        else:
            pytest.fail(f"Unrecognized test case '{case}'.")


@pytest.mark.slow
def test_gaussian_sphere_analytical_comparison():
    """
    This test runs a known example problem and compares to a theoretical
    model for small deflections.

    Still under construction (comparing the actual form of the radiograph
    is possible but tricky to implement).
    """

    # The Gaussian sphere problem for small deflection potentials
    # is solved in Kugland2012relation, and the equations referenced
    # below are from that paper.
    # https://doi.org/10.1063/1.4750234

    a = (1 * u.mm / 3).to(u.mm).value
    phi0 = 1.4e5
    W = 15e6

    l = 10  # noqa: E741
    L = 200

    # Define and run the problem
    # Setting b to be much larger than the problem so that the field is not
    # cut off at the edges. This is required to be directly
    # comparable to the theoretical result.
    grid = _test_grid(
        "electrostatic_gaussian_sphere",
        num=100,
        phi0=phi0 * u.V,
        a=a * u.mm,
        b=20 * u.mm,
    )
    source = (0 * u.mm, -l * u.mm, 0 * u.mm)
    detector = (0 * u.mm, L * u.mm, 0 * u.mm)

    with pytest.warns(
        RuntimeWarning, match="Fields should go to zero at edges of grid to avoid "
    ):
        sim = cpr.Tracker(grid, source, detector, verbose=False)

    sim.create_particles(1e3, W * u.eV, max_theta=12 * u.deg)
    sim.run()

    size = np.array([[-1, 1], [-1, 1]]) * 4 * u.cm
    bins = [100, 100]
    h, v, i = cpr.synthetic_radiograph(sim, size=size, bins=bins)
    h = h.to(u.mm).value / sim.mag
    v = v.to(u.mm).value / sim.mag
    r0 = h

    # Calculate a lineout across the center of the plane (y=0)
    v0 = np.argmin(np.abs(v))

    line = np.mean(i[:, v0 - 6 : v0 + 6], axis=1)
    # Zero the edge of the radiograph
    line += -np.mean(line)
    line *= 1 / np.max(np.abs(line))

    # Calculate the theoretical deflection angles (Eq. 28)
    theory = phi0 / W * np.sqrt(np.pi) * (r0 / a) * np.exp(-((r0 / a) ** 2))

    max_deflection = np.max(np.abs(theory))
    mu = np.sqrt(np.pi) * (phi0 / W) * (l / a)

    # sim_mu = sim.max_deflection.to(u.rad).value*(l/a)

    # Calculate the theoretical inversion (Eq. 31 )
    theory_deflect = -2 * mu * (1 - (r0 / a) ** 2) * np.exp(-((r0 / a) ** 2))
    theory_deflect *= 1 / np.max(np.abs(theory_deflect))

    # Uncomment for debug
    """
    print(f"Theory max deflection: {max_deflection:.6f}")
    print(f"Theory mu: {mu:.3f}")
    print(f"Sim max deflection: {sim.max_deflection.to(u.rad).value:.6f}")
    print(f"Sim mu: {sim_mu:.3f}")

    import matplotlib.pyplot as plt
    print(f"Theory max deflection: {max_deflection:.6f}")
    print(f"Theory mu: {mu:.3f}")
    print(f"Sim max deflection: {sim.max_deflection.to(u.rad).value:.6f}")
    print(f"Sim mu: {sim_mu:.3f}")

    fig, ax = plt.subplots()
    ax.pcolormesh(h, v, i.T, shading='auto', cmap='Blues_r')
    ax.set_aspect('equal')

    fig, ax = plt.subplots()
    ax.plot(h, line )
    ax.plot(h, theory_deflect)
    """

    assert np.isclose(max_deflection, sim.max_deflection.to(u.rad).value, atol=1e-3)


@pytest.mark.slow
def test_add_wire_mesh():

    # ************************************************************
    # Test various input configurations
    # ************************************************************

    # Test a circular mesh
    run_mesh_example(extent=1 * u.mm)

    # Test providing hdir
    run_mesh_example(mesh_hdir=np.array([0.5, 0, 0.5]))

    # Test providing hdir and vdir
    run_mesh_example(mesh_hdir=np.array([0.5, 0, 0.5]), mesh_vdir=np.array([0, 0.1, 1]))

    # ************************************************************
    # Test invalid inputs
    # ************************************************************

    # Test invalid extent (too many elements)
    with pytest.raises(ValueError):
        run_mesh_example(extent=(1 * u.mm, 2 * u.mm, 3 * u.mm))

    # Test wire mesh completely blocks all particles (in this case because
    # the wire diameter is absurdly large)
    with pytest.raises(ValueError):
        run_mesh_example(wire_diameter=5 * u.mm)

    # Test if wire mesh is not between the source and object
    with pytest.raises(ValueError):
        run_mesh_example(location=np.array([0, 3, 0]) * u.mm)

    # ************************************************************
    # Test that mesh is the right size in the detector plane, and that
    # the wire spacing images correctly.
    # This is actually a good overall test of the whole proton radiography
    # particle tracing algorithm.
    # ************************************************************
    loc = np.array([0, -2, 0]) * u.mm
    extent = (1 * u.mm, 1 * u.mm)
    wire_diameter = 30 * u.um
    nwires = 9
    sim = run_mesh_example(
        problem="empty",
        nparticles=1e5,
        location=loc,
        extent=extent,
        wire_diameter=wire_diameter,
        nwires=nwires,
    )

    # Calculate the width that the grid SHOULD have on the image plane
    src_to_mesh = np.linalg.norm(loc.si.value - sim.source)
    mesh_to_det = np.linalg.norm(sim.detector - loc.si.value)
    mag = 1 + mesh_to_det / src_to_mesh
    true_width = mag * extent[0].to(u.mm).value
    true_spacing = true_width / (nwires - 1)

    # Create a synthetic radiograph
    size = np.array([[-1, 1], [-1, 1]]) * 2 * u.cm
    bins = [100, 50]
    # Expect a warning because many particles are off the radiograph
    # (Chose max_theta so corners are covered)
    with pytest.warns(RuntimeWarning):
        h, v, i = cpr.synthetic_radiograph(sim, size=size, bins=bins)

    # Sum up the vertical direction
    line = np.sum(i, axis=1)

    # Determine the points that are on gridlines: where 1/line is above the
    # median by a lot
    ind = np.argwhere(1 / line > 2 * np.median(1 / line))
    hwhere = h.to(u.mm).value[ind]
    measured_width = np.max(hwhere) - np.min(hwhere)

    # Calculate the max spatial frequency (should be close to the grid spacing)
    dx = np.abs(size[0][1] - size[0][0]).to(u.mm).value / bins[0]
    fnyquist = int(bins[0] / 2)
    freqs = np.fft.fftfreq(h.size, d=dx)
    freqs = freqs[:fnyquist]
    # Calculate the positive frequency power spectrum
    pspect = np.abs(np.fft.fft(1 / line)) ** 2
    pspect = pspect[:fnyquist]
    pspect = np.where(np.abs(freqs) < 0.1, 0, pspect)  # Mask the low frequencies

    # Measured spacing is the inverse of the maximum spatial frequency
    measured_spacing = 1 / freqs[np.argmax(pspect)]

    # This test is somewhat tricky, so here's a matplotlib plot
    # that can be uncommented for debugging
    """
    fig, ax = plt.subplots(nrows=3, figsize=(4,15))
    ax[0].pcolormesh(h.to(u.mm).value, v.to(u.mm).value, i.T, cmap='Blues_r')
    ax[0].set_aspect('equal')
    ax[0].axvline(x=np.max(hwhere), color='red')
    ax[0].axvline(x=np.min(hwhere), color='red')

    ax[1].plot(h.to(u.mm).value, 1/line)
    ax[1].axhline(y=np.median(1/line))
    ax[1].axvline(x=np.max(hwhere), color='red')
    ax[1].axvline(x=np.min(hwhere), color='red')

    ax[2].plot(freqs, pspect)
    """

    # Verify that the edges of the mesh are imaged correctly
    assert np.isclose(measured_width, true_width, 1)

    # Verify that the spacing is correct by checking the FFT
    assert np.isclose(measured_spacing, true_spacing, 0.5)


def test_multiple_grids2():
    """
    Test that a case with two grids runs.
    TODO: automate test by including two fields with some obvious analytical
    solution??
    """

    grid1 = _test_grid("constant_bz", L=3 * u.cm, num=50, B0=0.7 * u.T)
    grid2 = _test_grid("electrostatic_gaussian_sphere", L=1 * u.mm, num=50)
    grids = [grid1, grid2]

    source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 200 * u.mm, 0 * u.mm)

    sim = cpr.Tracker(grids, source, detector, verbose=True)

    sim.create_particles(1e5, 15 * u.MeV, max_theta=8 * u.deg)

    sim.run(field_weighting="nearest neighbor")

    size = np.array([[-1, 1], [-1, 1]]) * 5 * u.cm
    bins = [100, 100]
    hax, vax, values = cpr.synthetic_radiograph(sim, size=size, bins=bins)

    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.pcolormesh(hax.to(u.cm).value, vax.to(u.cm).value, values.T)
    """


def test_proton_scattering():
    """
    Test proton scattering
    """

    # grid = _test_grid("dense_cylinder", L=0.5 * u.mm, num=200, rho0=1e3*u.kg/u.m**3)
    rho0 = 1e3 * u.kg / u.m**3
    energy = 15 * u.MeV

    # Create the cylinder
    grid1 = CartesianGrid(
        [-0.5 * u.mm, -0.5 * u.mm, -1.5 * u.mm],
        [0.5 * u.mm, 0.5 * u.mm, 1.5 * u.mm],
        num=(200, 200, 50),
    )

    a = 290 * u.um
    b = a - 20 * u.um
    radius = np.linalg.norm(grid1.grid[..., 0:2], axis=3)

    rho = np.where(np.logical_and(radius < a, radius > b), rho0, 0 * u.kg / u.m**3)
    grid1.add_quantities(rho=rho)

    # Create the coils
    rho0 = 1e5 * u.kg / u.m**3
    grid2 = CartesianGrid(
        [-2.5 * u.mm, -2.5 * u.mm, -5 * u.mm],
        [2.5 * u.mm, 2.5 * u.mm, 5 * u.mm],
        num=(100, 100, 100),
    )

    radius = np.linalg.norm(grid2.grid[..., 0:2], axis=3)
    z = grid2.grid[..., 2]
    rho = 1e3 * u.kg / u.m**3

    # Create a cylinder
    rho = np.where(radius < 2.3 * u.mm, rho0, 0 * u.kg / u.m**3)
    # Cut out the center
    rho = np.where(np.abs(z) < 2.125 * u.mm, 0 * u.kg / u.m**3, rho)
    grid2.add_quantities(rho=rho)

    grids = [grid1, grid2]

    mass_L_rad = [63.04 * u.g / u.cm**2] * 2

    source = (-11 * u.mm, (90 - 26.57) * u.deg, 0 * u.deg)
    detector = (270 * u.mm, (90 - 26.57) * u.deg, 0 * u.deg)

    sim = cpr.Tracker(grids, source, detector, verbose=True, mass_L_rad=mass_L_rad)

    sim.create_particles(1e4, energy, max_theta=12 * u.deg)

    sim.run(field_weighting="nearest neighbor")

    size = np.array([[-1, 1], [-1, 1]]) * 5 * u.cm
    bins = [150, 150]
    hax, vax, values = cpr.synthetic_radiograph(sim, size=size, bins=bins)

    mag = 1 + 270 / 11

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.pcolormesh(hax.to(u.mm).value / mag, vax.to(u.mm).value / mag, values.T)
    ax.set_xlabel("X (mm, obj)")
    ax.set_ylabel("Y (mm, obj)")
    ax.set_title(f"{energy}")


def test_particle_stopping():

    # Create the cylinder
    grid1 = CartesianGrid(
        [-0.5 * u.mm, -0.5 * u.mm, -1.5 * u.mm],
        [0.5 * u.mm, 0.5 * u.mm, 1.5 * u.mm],
        num=(200, 200, 50),
    )
    rho0 = 1e6 * u.kg / u.m**3
    a = 290 * u.um
    radius = np.linalg.norm(grid1.grid[..., 0:2], axis=3)

    rho = np.where(radius < a, rho0, 0 * u.kg / u.m**3)
    grid1.add_quantities(rho=rho)

    source = (-11 * u.mm, (90 - 26.57) * u.deg, 0 * u.deg)
    detector = (270 * u.mm, (90 - 26.57) * u.deg, 0 * u.deg)

    aluminum_path = downloader.get_file(
        "NIST_PSTAR_aluminum.txt", directory=os.getcwd()
    )
    arr = np.loadtxt(aluminum_path, skiprows=8)
    eaxis = arr[:, 0] * u.MeV
    mass_stopping_power = arr[:, 1] * u.MeV * u.cm**2 / u.g

    mass_stopping_power = (eaxis, mass_stopping_power)

    sim = cpr.Tracker(
        [
            grid1,
        ],
        source,
        detector,
        verbose=True,
        mass_stopping_power=mass_stopping_power,
    )

    sim.create_particles(1e4, 15 * u.MeV, max_theta=12 * u.deg)

    sim.run(field_weighting="nearest neighbor")

    size = np.array([[-1, 1], [-1, 1]]) * 5 * u.cm
    bins = [150, 150]
    hax, vax, values = cpr.synthetic_radiograph(sim, size=size, bins=bins)

    mag = 1 + 270 / 11

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.pcolormesh(hax.to(u.mm).value / mag, vax.to(u.mm).value / mag, values.T)
    ax.set_xlabel("X (mm, obj)")
    ax.set_ylabel("Y (mm, obj)")


if __name__ == "__main__":
    # test_input_validation()
    # test_run_options()
    # run_mesh_example()
    # test_add_wire_mesh()
    # test_multiple_grids()
    test_proton_scattering()
    # test_particle_stopping()
