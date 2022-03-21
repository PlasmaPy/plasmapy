import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pytest

from plasmapy.diagnostics.path_integrated_diagnostic import (
    Interferometer,
    LineIntegratedDiagnostic,
    LineIntegrateScalarQuantities,
    PathIntegratedDiagnostic,
)
from plasmapy.plasma.grids import CartesianGrid, NonUniformCartesianGrid


@pytest.fixture
def cartesian_density_cylinder_grid():
    """
    A Cartesian grid with a cylinder of constant density aligned with
    the z axis.
    """
    ax = np.linspace(-2, 2, num=50) * u.mm
    xarr, yarr, zarr = np.meshgrid(ax, ax, ax, indexing="ij")
    radius = np.sqrt(xarr ** 2 + yarr ** 2)
    field = np.where(radius < 1 * u.mm, 1, 0) * u.kg / u.m ** 3
    grid = CartesianGrid(xarr, yarr, zarr)
    grid.add_quantities(rho=field)
    return grid


@pytest.fixture
def cartesian_ne_cylinder_grid():
    """

    A Cartesian grid with a cylinder of constant electron density aligned
    with the z-axis. The density is 3e19 cm^-3 and the cylinder radius
    is 1 mm.
    """

    ax = np.linspace(-2, 2, num=50) * u.mm
    xarr, yarr, zarr = np.meshgrid(ax, ax, ax, indexing="ij")
    radius = np.sqrt(xarr ** 2 + yarr ** 2)
    n_e = np.where(radius < 1 * u.mm, 1, 0) * 3e18 / u.cm ** 3
    grid = CartesianGrid(xarr, yarr, zarr)

    grid.add_quantities(n_e=n_e)

    return grid


def test_abstract_line_integrated_diagnostic(cartesian_density_cylinder_grid):
    grid = cartesian_density_cylinder_grid
    source = (0 * u.mm, 0 * u.mm, -5 * u.mm)
    detector = (0 * u.mm, 0 * u.mm, 5 * u.mm)

    # Test that you can't instantiate the base class
    with pytest.raises(TypeError):
        obj = LineIntegratedDiagnostic(grid, source, detector)


def test_integrate_scalar_quantities(cartesian_density_cylinder_grid):
    grid = cartesian_density_cylinder_grid
    source = (0 * u.mm, 0 * u.mm, -5 * u.mm)
    detector = (0 * u.mm, 0 * u.mm, 5 * u.mm)

    # Erorr should be raised if quantity requested is not defined on the grid
    with pytest.raises(ValueError):
        obj = LineIntegrateScalarQuantities(grid, source, detector, "B_x")

    obj = LineIntegrateScalarQuantities(grid, source, detector, "rho")
    h, v, i = obj.evaluate(25)


def test_constant_cylinder(cartesian_density_cylinder_grid):
    """
    Test that LineIntegratedScalarQuantities works on a simple known test
    grid.
    """
    grid = cartesian_density_cylinder_grid
    source = (-5 * u.mm, 0 * u.mm, 0 * u.mm)
    detector = (5 * u.mm, 0 * u.mm, 0 * u.mm)

    obj = LineIntegrateScalarQuantities(grid, source, detector, ["rho"], verbose=True)

    # Test that line-integrating with collimated = False yields
    hax, vax, integral = obj.evaluate(50, bins=[50, 10], collimated=True)
    hax = hax.to(u.mm).value
    vax = vax.to(u.mm).value
    integral = integral.to(u.kg / u.m ** 2).value

    line = np.mean(integral, axis=1)

    height = np.nanmax(line)
    wi = np.argmax(np.where(line > 0.00010, 1, 0))
    halfwidth = np.abs(hax[wi])

    assert np.isclose(height, 0.002, atol=0.01)
    assert np.isclose(halfwidth, 1, atol=0.25)


def test_non_collimated(cartesian_density_cylinder_grid):
    """
    Tests line-integration with a point source by checking that the
    width of the feature scales with the magnification.
    """
    grid = cartesian_density_cylinder_grid
    source = (0 * u.mm, -5 * u.mm, 0 * u.mm)
    detector = (0 * u.mm, 10 * u.mm, 0 * u.mm)
    obj = LineIntegrateScalarQuantities(grid, source, detector, "rho", verbose=False)

    size = np.array([[-1, 1], [-1, 1]]) * 5 * u.mm
    hax, vax, integral = obj._line_integral(
        100, size=size, bins=[100, 100], collimated=False
    )

    # Scaling hax/mag gives hax in the object plane
    # That will now agree with the theory
    hax = hax.to(u.mm).value / obj.mag
    vax = vax.to(u.mm).value

    integral = integral.to(u.kg / u.m ** 2).value
    line = np.mean(integral, axis=1)

    # Theoretical expected answer for line-integral through a cylinder
    theory = np.where(np.abs(hax) < 0.999, 2 * np.sqrt(np.abs(1 - hax ** 2)) * 1e-3, 0)

    assert np.allclose(line, theory, atol=5e-4)


@pytest.mark.parametrize("unwrapped", [(True), (False)])
def test_interferogram_density_cylinder(unwrapped, cartesian_ne_cylinder_grid):

    grid = cartesian_ne_cylinder_grid

    source = (-5 * u.mm, 0 * u.mm, 0 * u.mm)
    detector = (5 * u.mm, 0 * u.mm, 0 * u.mm)
    obj = Interferometer(grid, source, detector, verbose=False)

    size = np.array([[-2, 2], [-2, 2]]) * u.mm
    bins = [50, 5]

    fprobe = 1.14e15 * u.Hz
    hax, vax, phase = obj.evaluate(
        fprobe,
        100,
        size=size,
        bins=bins,
        unwrapped=unwrapped,
        collimated=True,
    )

    if unwrapped:
        # Assert the known value of the max
        int_ne = (
            (
                -2
                * const.c
                * const.eps0.si
                * const.m_e
                / const.e.si ** 2
                * fprobe
                * phase
            )
            .to(u.cm ** -2)
            .value
        )

        # Average along one axis
        int_ne = np.mean(int_ne, axis=-1)

        max_ne = np.max(np.abs(int_ne))

        # Assert density is close to the known theoretical value
        assert np.isclose(max_ne, 6e17, atol=1e17)

    else:
        # Assert that the phase is less than pi

        assert np.max(np.abs(phase)) < 1.1 * np.pi
