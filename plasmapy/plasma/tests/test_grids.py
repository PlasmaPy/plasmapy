"""
Tests for grids.py
"""

import astropy.units as u
import numpy as np
import pytest
import warnings

from plasmapy.plasma import grids as grids


def test_AbstractGrid():
    grid = grids.AbstractGrid(-1 * u.cm, 1 * u.cm, num=(10, 20, 5))

    array = grid.grid
    units = grid.units

    pts0, pts1, pts2 = grid.pts0, grid.pts1, grid.pts2

    # Test wrong number of positional arguments: 1 or more than 3
    with pytest.raises(TypeError):
        grid = grids.AbstractGrid(1 * u.cm, num=10)
    with pytest.raises(TypeError):
        grid = grids.AbstractGrid(-1 * u.cm, 1 * u.cm, 1 * u.cm, 1 * u.cm)

    # Test unequal lengths of arguments raises error
    with pytest.raises(ValueError):
        grid = grids.AbstractGrid(-1 * u.m, [2 * u.m, 3 * u.m], num=10)

    with pytest.raises(ValueError):
        grid = grids.AbstractGrid(
            np.random.randn(2, 5, 3) * u.m,
            np.random.randn(2, 5, 3) * u.m,
            np.random.randn(2, 5, 4) * u.m,
        )

    # Test incompatible units
    with pytest.raises(ValueError):
        grid = grids.AbstractGrid(1 * u.cm, 1 * u.eV, num=10)

    # Test adding a quantity
    q = np.random.randn(10, 20, 5) * u.T
    grid.add_quantity("B_x", q)

    # Test adding a quantity with wrong units
    q = np.random.randn(10, 20, 5) * u.kg
    with pytest.raises(ValueError):
        grid.add_quantity("B_x", q)

    # Testing adding a quantity with an unrecognized key name
    with pytest.warns(UserWarning):
        grid.add_quantity("not_a_recognized_key", q)

    # Test adding a quantity of incompatible size
    q = np.random.randn(5, 20, 5) * u.T
    with pytest.raises(ValueError):
        grid.add_quantity("B_x", q)

    # Test adding multiple quantites at once
    q = np.random.randn(10, 20, 5) * u.T
    grid.add_quantities(['B_x', 'B_y', 'B_z'], [q,q,q])

    # Test unequal numbers of keys and quantities
    with pytest.raises(ValueError):
        grid.add_quantities(['B_x', 'B_y', 'B_z'], [q,q])


def test_CartesianGrid():

    grid = grids.CartesianGrid(-1 * u.cm, 1 * u.cm, num=10)

    x_arr, y_arr, z_arr = grid.grids
    x_axis, y_axis, z_axis = grid.ax0, grid.ax1, grid.ax2
    d_x, d_y, d_z = grid.dax0, grid.dax1, grid.dax2
    is_uniform_grid = grid.is_uniform_grid
    shape = grid.shape
    unit = grid.units

    # Grid should be uniform
    assert grid.is_uniform_grid == True

    # Test initializing with a provided grid
    grid2 = grids.CartesianGrid(grid.grids[0], grid.grids[1], grid.grids[2],)

    # Units not all consistent
    with pytest.raises(ValueError):
        grid = grids.CartesianGrid(
            [-1 * u.m, -1 * u.rad, -1 * u.m], [1 * u.m, 1 * u.rad, 1 * u.m]
        )


def test_interpolate_indices():
    # Create grid
    grid = grids.CartesianGrid(-1 * u.cm, 1 * u.cm, num=25)

    # One position
    pos = np.array([0.1, -0.3, 0]) * u.cm
    i = grid.interpolate_indices(pos)[0]
    # Assert that nearest grid cell was found
    pout = grid.grid[int(i[0]), int(i[1]), int(i[2])]*grid.unit
    assert np.allclose(pos, pout, atol=0.1)

    # Two positions
    pos = np.array([[0.1, -0.3, 0], [0.1, -0.3, 0]]) * u.cm
    i = grid.interpolate_indices(pos)[0]

    # Contains out-of-bounds values (index array should contain NaNs)
    pos = np.array([5, -0.3, 0]) * u.cm
    i = grid.interpolate_indices(pos)[0]
    assert np.sum(np.isnan(i)) > 0


def test_nearest_neighbor_interpolator():
    # Create grid
    grid = grids.CartesianGrid(-1 * u.cm, 1 * u.cm, num=25)
    # Add some data to the grid
    grid.add_quantity("x", grid.grids[0])
    grid.add_quantity("y", grid.grids[1])

    # One position
    pos = np.array([0.1, -0.3, 0]) * u.cm
    pout = grid.nearest_neighbor_interpolator(pos, "x")
    assert np.allclose(pos[0], pout, atol=0.1)

    # Two positions, two quantities
    pos = np.array([[0.1, -0.3, 0], [0.1, -0.3, 0]]) * u.cm
    pout = grid.nearest_neighbor_interpolator(pos, "x", "y")

    # Contains out-of-bounds values (must handle NaNs correctly)
    pos = np.array([5, -0.3, 0]) * u.cm
    pout = grid.nearest_neighbor_interpolator(pos, "x")
    assert np.allclose(pout, 0*u.cm, atol=0.1)

def test_volume_averaged_interpolator():
    # Create grid
    grid = grids.CartesianGrid(-1 * u.cm, 1 * u.cm, num=25)
    # Add some data to the grid
    grid.add_quantity("x", grid.grids[0])
    grid.add_quantity("y", grid.grids[1])

    # One position
    pos = np.array([0.1, -0.3, 0]) * u.cm
    pout = grid.volume_averaged_interpolator(pos, "x")
    assert np.allclose(pos[0], pout, atol=0.1)

    # Two positions, two quantities
    pos = np.array([[0.1, -0.3, 0], [0.1, -0.3, 0]]) * u.cm
    pout = grid.volume_averaged_interpolator(pos, "x", "y")

    # Contains out-of-bounds values (must handle NaNs correctly)
    pos = np.array([5, -0.3, 0]) * u.cm
    pout = grid.volume_averaged_interpolator(pos, "x")
    assert np.allclose(pout, 0*u.cm, atol=0.1)



def test_NonUniformCartesianGrid():
    grid = grids.NonUniformCartesianGrid(-1 * u.cm, 1 * u.cm, num=10)

    pts0, pts1, pts2 = grid.grids
    shape = grid.shape
    units = grid.units

    # Grid should be non-uniform
    assert grid.is_uniform_grid == False

    # Test assigning a quantity
    q1 = np.random.randn(10, 10, 10) * u.kg/u.cm**3
    grid.add_quantity("rho", q1)

test_AbstractGrid()
test_CartesianGrid()
test_interpolate_indices()
test_nearest_neighbor_interpolator()
test_volume_averaged_interpolator()
test_NonUniformCartesianGrid()