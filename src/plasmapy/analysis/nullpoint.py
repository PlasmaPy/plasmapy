"""
Locating and analyzing 3D magnetic null points.

.. attention::

   |expect-api-changes|
"""

__all__ = [
    "MultipleNullPointWarning",
    "NonZeroDivergence",
    "NullPoint",
    "NullPointError",
    "NullPointWarning",
    "Point",
    "null_point_find",
    "trilinear_approx",
    "uniform_null_point_find",
]

import warnings
from collections.abc import Callable

import numpy as np

# Declare Constants & global variables
_EQUALITY_ATOL = 1e-10
_MAX_RECURSION_LEVEL = 10
_recursion_level = 0


class NullPointError(Exception):
    """
    A class for handling the exceptions of the null point finder
    functionality.

    .. note::

       This functionality is still under development and the API may
       change in future releases.
    """


class NullPointWarning(UserWarning):
    """
    A class for handling the warnings of the null point finder
    functionality.

    .. note::

       This functionality is still under development and the API may
       change in future releases.
    """


class NonZeroDivergence(NullPointError):  # noqa: N818
    """
    A class for handling the exception raised by passing in a magnetic
    field that violates the zero divergence constraint.

    .. note::

       This functionality is still under development and the API may
       change in future releases.
    """

    def __init__(self) -> None:
        super().__init__(
            "The divergence constraint does not hold for the provided magnetic field."
        )


class MultipleNullPointWarning(NullPointWarning):
    """
    A class for handling the warning raised by passing in a magnetic
    field grid that may contain multiple null points in close proximity due to low
    resolution.

    .. note::

       This functionality is still under development and the API may
       change in future releases.
    """


class Point:
    """
    Abstract class for defining a point in 3D space.

    .. note::

       This functionality is still under development and the API may
       change in future releases.
    """

    def __init__(self, loc) -> None:
        self._loc = loc

    def get_loc(self):
        r"""
        Returns the coordinates of the point object.
        """
        return self._loc

    loc = property(get_loc)


class NullPoint(Point):
    """
    A class for defining a null point in 3D space.

    .. note::

       This functionality is still under development and the API may
       change in future releases.
    """

    def __init__(self, null_loc, classification) -> None:
        super().__init__(null_loc)
        self._classification = classification

    def get_classification(self):
        r"""
        Returns the type of the null point object.
        """
        return self._classification

    def __eq__(self, point):
        r"""
        Returns `True` if two null point objects have the same
        coordinates. `False` otherwise.
        """
        d = np.sqrt(
            (self.loc[0] - point.loc[0]) ** 2
            + (self.loc[1] - point.loc[1]) ** 2
            + (self.loc[2] - point.loc[2]) ** 2
        )
        return np.isclose(d, 0, atol=_EQUALITY_ATOL)

    classification = property(get_classification)


def _vector_space(
    x_arr=None,
    y_arr=None,
    z_arr=None,
    x_range=(0, 1),
    y_range=(0, 1),
    z_range=(0, 1),
    u_arr=None,
    v_arr=None,
    w_arr=None,
    func=(lambda x, y, z: [x, y, z]),
    precision=(0.05, 0.05, 0.05),
):
    r"""
    Returns a vector space in the form of a multi-dimensional array.

    Parameters
    ----------
    x_arr : |array_like|
        The array representing the coordinates in the x-dimension. If
        not given, then range values are used to construct a uniform
        array on that interval.

    y_arr : |array_like|
        The array representing the coordinates in the y-dimension. If
        not given, then range values are used to construct a uniform
        array on that interval.

    z_arr : |array_like|
        The array representing the coordinates in the z-dimension. If
        not given, then range values are used to construct a uniform
        array on that interval.

    x_range : |array_like|, default: ``[0, 1]``
        A 1 by 2 array containing the range of x-values for the vector
        spaces.

    y_range : |array_like|, default: ``[0, 1]``
        A 1 by 2 array containing the range of y-values for the vector
        spaces.

    z_range : |array_like|, default: ``[0, 1]``
        A 1 by 2 array containing the range of z-values for the vector
        spaces.

    u_arr : |array_like|
        A 3D array containing the x-component of the vector values for
        the vector space. If not given, the vector values are generated
        over the vector space using the function func.

    v_arr : |array_like|
        A 3D array containing the y-component of the vector values for
        the vector space. If not given, the vector values are generated
        over the vector space using the function func.

    w_arr : |array_like|
        A 3D array containing the z-component of the vector values for
        the vector space. If not given, the vector values are generated
        over the vector space using the function func.

    func : function
        A function that takes in 3 arguments, respectively representing
        an :math:`(x, y, z)` coordinate of a point and returns the
        vector value for that point in the form of a 1 by 3 array.

    precision : |array_like|, default: ``[0.05, 0.05, 0.05]``
        A 1 by 3 array containing the approximate precision values for
        each dimension, in the case where uniform arrays are being used.

    Returns
    -------
    ndarray
        A 1 by 3 array with the first element containing the
        coordinates, the second element containing the vector values,
        and the third element containing the delta values for each
        dimension.
    """
    # Constructing the Meshgrid

    if x_arr is not None and y_arr is not None and z_arr is not None:
        x, y, z = np.meshgrid(
            x_arr,
            y_arr,
            z_arr,
            indexing="ij",
        )
        dx = np.diff(x_arr)
        dy = np.diff(y_arr)
        dz = np.diff(z_arr)
    else:
        x_den = int(np.around((x_range[1] - x_range[0]) / precision[0]) + 1)
        y_den = int(np.around((y_range[1] - y_range[0]) / precision[1]) + 1)
        z_den = int(np.around((z_range[1] - z_range[0]) / precision[2]) + 1)
        dx = np.diff(np.linspace(x_range[0], x_range[1], x_den))
        dy = np.diff(np.linspace(y_range[0], y_range[1], y_den))
        dz = np.diff(np.linspace(z_range[0], z_range[1], z_den))
        x, y, z = np.meshgrid(
            np.linspace(x_range[0], x_range[1], x_den),
            np.linspace(y_range[0], y_range[1], y_den),
            np.linspace(z_range[0], z_range[1], z_den),
            indexing="ij",
        )
    # Calculating the vector values
    if u_arr is not None and v_arr is not None and w_arr is not None:
        u = u_arr
        v = v_arr
        w = w_arr
    else:
        u, v, w = func(x, y, z)
    return np.array([x, y, z]), np.array([u, v, w]), [dx, dy, dz]


def _trilinear_coeff_cal(vspace, cell):
    r"""
    Return the coefficients for the trilinear approximation function on
    a given grid cell in a given vector space.

    Parameters
    ----------
    vspace: |array_like|
        The vector space as constructed by the vector_space function
        which is a 1 by 3 array with the first element containing the
        coordinates, the second element containing the vector values,
        and the third element containing the delta values for each
        dimension.

    cell: |array_like| of integers
        A grid cell, represented by a 1 by 3 array of integers, which
        correspond to a grid cell in the vector space.

    Returns
    -------
    ndarray
        Returns a 1 by 3 array with the first element containing the
        coefficients for the trilinear approximation function for the
        x-component of the vector space, the second element containing
        the coefficients for the trilinear approximation function for
        the y-component of the vector space, and the third element
        containing the coefficients for the trilinear approximation
        function for the z-component of the vector space.
    """
    u, v, w = vspace[1]
    deltax, deltay, deltaz = vspace[2]
    f000 = cell
    f001 = [cell[0], cell[1], cell[2] + 1]
    f010 = [cell[0], cell[1] + 1, cell[2]]
    f011 = [cell[0], cell[1] + 1, cell[2] + 1]
    f100 = [cell[0] + 1, cell[1], cell[2]]
    f101 = [cell[0] + 1, cell[1], cell[2] + 1]
    f110 = [cell[0] + 1, cell[1] + 1, cell[2]]
    f111 = [cell[0] + 1, cell[1] + 1, cell[2] + 1]
    x0 = float(vspace[0][0][f000[0]][f000[1]][f000[2]])
    x1 = float(x0 + deltax[cell[0]])
    y0 = float(vspace[0][1][f000[0]][f000[1]][f000[2]])
    y1 = float(y0 + deltay[cell[1]])
    z0 = float(vspace[0][2][f000[0]][f000[1]][f000[2]])
    z1 = float(z0 + deltaz[cell[2]])
    A = np.array(
        [
            [1, x0, y0, z0, x0 * y0, x0 * z0, y0 * z0, x0 * y0 * z0],
            [1, x1, y0, z0, x1 * y0, x1 * z0, y0 * z0, x1 * y0 * z0],
            [1, x0, y1, z0, x0 * y1, x0 * z0, y1 * z0, x0 * y1 * z0],
            [1, x1, y1, z0, x1 * y1, x1 * z0, y1 * z0, x1 * y1 * z0],
            [1, x0, y0, z1, x0 * y0, x0 * z1, y0 * z1, x0 * y0 * z1],
            [1, x1, y0, z1, x1 * y0, x1 * z1, y0 * z1, x1 * y0 * z1],
            [1, x0, y1, z1, x0 * y1, x0 * z1, y1 * z1, x0 * y1 * z1],
            [1, x1, y1, z1, x1 * y1, x1 * z1, y1 * z1, x1 * y1 * z1],
        ]
    )
    sx = np.array(
        [
            [u[f000[0]][f000[1]][f000[2]]],
            [u[f100[0]][f100[1]][f100[2]]],
            [u[f010[0]][f010[1]][f010[2]]],
            [u[f110[0]][f110[1]][f110[2]]],
            [u[f001[0]][f001[1]][f001[2]]],
            [u[f101[0]][f101[1]][f101[2]]],
            [u[f011[0]][f011[1]][f011[2]]],
            [u[f111[0]][f111[1]][f111[2]]],
        ]
    )
    sy = np.array(
        [
            [v[f000[0]][f000[1]][f000[2]]],
            [v[f100[0]][f100[1]][f100[2]]],
            [v[f010[0]][f010[1]][f010[2]]],
            [v[f110[0]][f110[1]][f110[2]]],
            [v[f001[0]][f001[1]][f001[2]]],
            [v[f101[0]][f101[1]][f101[2]]],
            [v[f011[0]][f011[1]][f011[2]]],
            [v[f111[0]][f111[1]][f111[2]]],
        ]
    )
    sz = np.array(
        [
            [w[f000[0]][f000[1]][f000[2]]],
            [w[f100[0]][f100[1]][f100[2]]],
            [w[f010[0]][f010[1]][f010[2]]],
            [w[f110[0]][f110[1]][f110[2]]],
            [w[f001[0]][f001[1]][f001[2]]],
            [w[f101[0]][f101[1]][f101[2]]],
            [w[f011[0]][f011[1]][f011[2]]],
            [w[f111[0]][f111[1]][f111[2]]],
        ]
    )

    ax, bx, cx, dx, ex, fx, gx, hx = np.linalg.solve(A, sx).reshape(1, 8)[0]
    ay, by, cy, dy, ey, fy, gy, hy = np.linalg.solve(A, sy).reshape(1, 8)[0]
    az, bz, cz, dz, ez, fz, gz, hz = np.linalg.solve(A, sz).reshape(1, 8)[0]

    return np.array(
        [
            [ax, bx, cx, dx, ex, fx, gx, hx],
            [ay, by, cy, dy, ey, fy, gy, hy],
            [az, bz, cz, dz, ez, fz, gz, hz],
        ]
    )


def trilinear_approx(vspace, cell):
    r"""
    Return a function whose input is a coordinate within a given grid
    cell and returns the trilinearly approximated vector value at that
    particular coordinate in that grid cell.

    .. note::

       This functionality is still under development and the API may
       change in future releases.

    Parameters
    ----------
    vspace: |array_like|
        The vector space as constructed by the vector_space function
        which is a 1 by 3 array with the first element containing the
        coordinates, the second element containing the vector values,
        and the third element containing the delta values for each
        dimension.

    cell: |array_like| of integers
        A grid cell, represented by a 1 by 3 array of integers, which
        correspond to a grid cell in the vector space.

    Returns
    -------
    function
        A function whose input is a coordinate within a given grid cell
        and returns the trilinearly approximated vector value at that
        particular coordinate in that grid cell.

    """
    # Calculating coefficients
    ax, bx, cx, dx, ex, fx, gx, hx = _trilinear_coeff_cal(vspace, cell)[0]
    ay, by, cy, dy, ey, fy, gy, hy = _trilinear_coeff_cal(vspace, cell)[1]
    az, bz, cz, dz, ez, fz, gz, hz = _trilinear_coeff_cal(vspace, cell)[2]

    def approx_func(xInput, yInput, zInput):
        Bx = (
            ax
            + bx * xInput
            + cx * yInput
            + dx * zInput
            + ex * xInput * yInput
            + fx * xInput * zInput
            + gx * yInput * zInput
            + hx * xInput * yInput * zInput
        )
        By = (
            ay
            + by * xInput
            + cy * yInput
            + dy * zInput
            + ey * xInput * yInput
            + fy * xInput * zInput
            + gy * yInput * zInput
            + hy * xInput * yInput * zInput
        )
        Bz = (
            az
            + bz * xInput
            + cz * yInput
            + dz * zInput
            + ez * xInput * yInput
            + fz * xInput * zInput
            + gz * yInput * zInput
            + hz * xInput * yInput * zInput
        )
        return np.array([Bx, By, Bz])

    return approx_func


def _trilinear_jacobian(vspace, cell):
    r"""
    Returns a function whose input is a coordinate within a given grid
    cell and returns the trilinearly approximated jacobian matrix for
    that particular coordinate in that grid cell.

    Parameters
    ----------
    vspace: |array_like|
        The vector space as constructed by the vector_space function
        which is a 1 by 3 array with the first element containing the
        coordinates, the second element containing the vector values,
        and the third element containing the delta values for each
        dimension.

    cell: |array_like| of integers
        A grid cell, represented by a 1 by 3 array of integers, which
        correspond to a grid cell in the vector space.

    Returns
    -------
    A function whose input is a coordinate within a given grid cell and
    returns the trilinearly approximated jacobian matrix for that
    particular coordinate in that grid cell.
    """
    # Calculating coefficients
    ax, bx, cx, dx, ex, fx, gx, hx = _trilinear_coeff_cal(vspace, cell)[0]
    ay, by, cy, dy, ey, fy, gy, hy = _trilinear_coeff_cal(vspace, cell)[1]
    az, bz, cz, dz, ez, fz, gz, hz = _trilinear_coeff_cal(vspace, cell)[2]

    def jacobian_func(xInput, yInput, zInput):
        dBxdx = bx + ex * yInput + fx * zInput + hx * yInput * zInput
        dBxdy = cx + ex * xInput + gx * zInput + hx * xInput * yInput
        dBxdz = dx + fx * xInput + gx * yInput + hx * xInput * yInput
        dBydx = by + ey * yInput + fy * zInput + hy * yInput * zInput
        dBydy = cy + ey * xInput + gy * zInput + hy * xInput * yInput
        dBydz = dy + fy * xInput + gy * yInput + hy * xInput * yInput
        dBzdx = bz + ez * yInput + fz * zInput + hz * yInput * zInput
        dBzdy = cz + ez * xInput + gz * zInput + hz * xInput * yInput
        dBzdz = dz + fz * xInput + gz * yInput + hz * xInput * yInput

        def get_scalar(v: float | np.ndarray):
            """
            If a `float`, return the argument. If a `numpy.ndarray` with
            a single element, return that element.
            """
            if not hasattr(v, "__len__"):
                return v
            if len(v) != 1:
                raise ValueError("Unable to get scalar from multi-element array.")
            return v[0]  # type: ignore[index]

        return np.array(
            [
                get_scalar(dBxdx),
                get_scalar(dBxdy),
                get_scalar(dBxdz),
                get_scalar(dBydx),
                get_scalar(dBydy),
                get_scalar(dBydz),
                get_scalar(dBzdx),
                get_scalar(dBzdy),
                get_scalar(dBzdz),
            ]
        ).reshape(3, 3)

    return jacobian_func


def _reduction(vspace, cell):
    r"""
    Return a true or false based on whether a grid cell passes the
    reduction phase, meaning that they potentially contain a null point.

    Parameters
    ----------
    vspace: |array_like|
        The vector space as constructed by the vector_space function
        which is a 1 by 3 array with the first element containing the
        coordinates, the second element containing the vector values,
        and the third element containing the delta values for each
        dimension.

    cell: |array_like| of integers
        A grid cell, represented by a 1 by 3 array of integers, which
        correspond to a grid cell in the vector space.

    Returns
    -------
    bool
        `True` if a grid cell passes the reduction phase. `False`,
        otherwise.

    Notes
    -----
    Depending on the grid resolution, a cell containing more than one
    null point may not pass reduction, so it would not be detected.
    """
    u, v, w = vspace[1]
    f000 = cell
    f001 = [cell[0], cell[1], cell[2] + 1]
    f010 = [cell[0], cell[1] + 1, cell[2]]
    f011 = [cell[0], cell[1] + 1, cell[2] + 1]
    f100 = [cell[0] + 1, cell[1], cell[2]]
    f101 = [cell[0] + 1, cell[1], cell[2] + 1]
    f110 = [cell[0] + 1, cell[1] + 1, cell[2]]
    f111 = [cell[0] + 1, cell[1] + 1, cell[2] + 1]
    corners = [f000, f001, f010, f011, f100, f101, f110, f111]
    passX = False
    passY = False
    passZ = False
    # Check reduction criteria
    sign_x = np.sign(u[cell[0]][cell[1]][cell[2]])
    sign_y = np.sign(v[cell[0]][cell[1]][cell[2]])
    sign_z = np.sign(w[cell[0]][cell[1]][cell[2]])
    for point in corners:
        if (
            u[point[0]][point[1]][point[2]] == 0
            or np.sign(u[point[0]][point[1]][point[2]]) != sign_x
        ):
            passX = True
        if (
            v[point[0]][point[1]][point[2]] == 0
            or np.sign(v[point[0]][point[1]][point[2]]) != sign_y
        ):
            passY = True
        if (
            w[point[0]][point[1]][point[2]] == 0
            or np.sign(w[point[0]][point[1]][point[2]]) != sign_z
        ):
            passZ = True

    return passX and passY and passZ


def _bilinear_root(a1, b1, c1, d1, a2, b2, c2, d2):  # noqa: C901, PLR0911, PLR0912
    r"""
    Return the roots of a pair of bilinear equations of the following
    format.

    .. math::
        a_1 + b_1 x + c_1 y + d_1 x y = 0 \\
        a_2 + b_2 x + c_2 y + d_2 x y = 0

    Parameters
    ----------
    a1: float
    b1: float
    c1: float
    d1: float
    a2: float
    b2: float
    c2: float
    d2: float

    Returns
    -------
    roots : |array_like| of floats
        A 1 by 2 array containing the two roots.
    """
    m1 = np.array([[a1, a2], [c1, c2]])
    m2 = np.array([[a1, a2], [d1, d2]])
    m3 = np.array([[b1, b2], [c1, c2]])
    m4 = np.array([[b1, b2], [d1, d2]])

    a = np.linalg.det(m4)
    b = np.linalg.det(m2) + np.linalg.det(m3)
    c = np.linalg.det(m1)

    if np.isclose(a, 0, atol=_EQUALITY_ATOL):
        if np.isclose(b, 0, atol=_EQUALITY_ATOL):
            return np.array([])
        x1 = (-1.0 * c) / b
        x2 = (-1.0 * c) / b

    else:
        if (b**2 - 4.0 * a * c) < 0:
            return np.array([])
        x1 = (-1.0 * b + (b**2 - 4.0 * a * c) ** 0.5) / (2.0 * a)
        x2 = (-1.0 * b - (b**2 - 4.0 * a * c) ** 0.5) / (2.0 * a)

    y1 = None
    y2 = None
    if not (np.isclose((c1 + d1 * x1), 0, atol=_EQUALITY_ATOL)):
        y1 = (-a1 - b1 * x1) / (c1 + d1 * x1)
    elif not (np.isclose((c2 + d2 * x1), 0, atol=_EQUALITY_ATOL)):
        y1 = (-a2 - b2 * x1) / (c2 + d2 * x1)
    if not (np.isclose((c1 + d1 * x2), 0, atol=_EQUALITY_ATOL)):
        y2 = (-a1 - b1 * x2) / (c1 + d1 * x2)
    elif not (np.isclose((c2 + d2 * x2), 0, atol=_EQUALITY_ATOL)):
        y2 = (-a2 - b2 * x2) / (c2 + d2 * x2)

    if y1 is y2 is None:
        return np.array([])
    elif y1 is None:
        return np.array([(x2, y2)])
    elif y2 is None:
        return np.array([(x1, y1)])
    else:
        d = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if np.isclose(d, 0, atol=_EQUALITY_ATOL):
            return np.array([(x1, y1)])
        else:
            return np.array([(x1, y1), (x2, y2)])


def _trilinear_analysis(vspace, cell):  # noqa: C901, PLR0915
    r"""
    Return a true or false value based on whether a grid cell which has
    passed the reduction step, contains a null point, using trilinear
    analysis.

    Parameters
    ----------
    vspace: |array_like|
        The vector space as constructed by the vector_space function
        which is a 1 by 3 array with the first element containing the
        coordinates, the second element containing the vector values,
        and the third element containing the delta values for each
        dimension.

    cell: |array_like| of integers
        A grid cell, represented by a 1 by 3 array of integers, which
        correspond to a grid cell in the vector space.

    Returns
    -------
    bool
        `True` if a grid cell contains a null point using trilinear
        analysis. `False`, otherwise.

    Raises
    ------
    This function does not raise any exceptions.

    Warns
    -----
    :`UserWarning`
        If there is a possible lack of grid resolution, so
        that a grid cell may contain more than one null point.
    """

    # Critical Cell Corners
    f000 = cell
    f111 = [cell[0] + 1, cell[1] + 1, cell[2] + 1]

    # Calculating coefficients
    ax, bx, cx, dx, ex, fx, gx, hx = _trilinear_coeff_cal(vspace, cell)[0]
    ay, by, cy, dy, ey, fy, gy, hy = _trilinear_coeff_cal(vspace, cell)[1]
    az, bz, cz, dz, ez, fz, gz, hz = _trilinear_coeff_cal(vspace, cell)[2]

    # Initial Position of the cell corner
    initial = np.array(
        [
            vspace[0][0][f000[0]][f000[1]][f000[2]],
            vspace[0][1][f000[0]][f000[1]][f000[2]],
            vspace[0][2][f000[0]][f000[1]][f000[2]],
        ]
    )

    # Arrays holding the endpoints
    BxByEndpoints = []
    BxBzEndpoints = []
    ByBzEndpoints = []

    # Check if the null point already exists in root list
    def is_root_in_list(root, arr) -> bool:
        for r in arr:
            x_close = np.isclose(root[0], r[0], atol=_EQUALITY_ATOL)
            y_close = np.isclose(root[1], r[1], atol=_EQUALITY_ATOL)
            z_close = np.isclose(root[2], r[2], atol=_EQUALITY_ATOL)
            if x_close and y_close and z_close:
                return True
        return False

    # Front Surface
    yConst1 = vspace[0][1][f000[0]][f000[1]][
        f000[2]
    ]  # y-coordinate of the front surface
    # Bx=By=0 Curve Endpoint

    root_list = _bilinear_root(
        ax + cx * yConst1,
        bx + ex * yConst1,
        dx + gx * yConst1,
        fx + hx * yConst1,
        ay + cy * yConst1,
        by + ey * yConst1,
        dy + gy * yConst1,
        fy + hy * yConst1,
    )

    BxByEndpoints.extend(
        [
            (root[0], yConst1, root[1])
            for root in root_list
            if not is_root_in_list((root[0], yConst1, root[1]), BxByEndpoints)
        ]
    )

    # Bx=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + cx * yConst1,
        bx + ex * yConst1,
        dx + gx * yConst1,
        fx + hx * yConst1,
        az + cz * yConst1,
        bz + ez * yConst1,
        dz + gz * yConst1,
        fz + hz * yConst1,
    )

    BxBzEndpoints.extend(
        [
            (root[0], yConst1, root[1])
            for root in root_list
            if not is_root_in_list((root[0], yConst1, root[1]), BxBzEndpoints)
        ]
    )

    # By=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ay + cy * yConst1,
        by + ey * yConst1,
        dy + gy * yConst1,
        fy + hy * yConst1,
        az + cz * yConst1,
        bz + ez * yConst1,
        dz + gz * yConst1,
        fz + hz * yConst1,
    )

    ByBzEndpoints.extend(
        [
            (root[0], yConst1, root[1])
            for root in root_list
            if not is_root_in_list((root[0], yConst1, root[1]), ByBzEndpoints)
        ]
    )

    # Back Surface
    yConst2 = vspace[0][1][f111[0]][f111[1]][
        f111[2]
    ]  # y-coordinate of the front surface
    # Bx=By=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + cx * yConst2,
        bx + ex * yConst2,
        dx + gx * yConst2,
        fx + hx * yConst2,
        ay + cy * yConst2,
        by + ey * yConst2,
        dy + gy * yConst2,
        fy + hy * yConst2,
    )

    BxByEndpoints.extend(
        [
            (root[0], yConst2, root[1])
            for root in root_list
            if not is_root_in_list((root[0], yConst2, root[1]), BxByEndpoints)
        ]
    )

    # Bx=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + cx * yConst2,
        bx + ex * yConst2,
        dx + gx * yConst2,
        fx + hx * yConst2,
        az + cz * yConst2,
        bz + ez * yConst2,
        dz + gz * yConst2,
        fz + hz * yConst2,
    )

    BxBzEndpoints.extend(
        [
            (root[0], yConst2, root[1])
            for root in root_list
            if not is_root_in_list((root[0], yConst2, root[1]), BxBzEndpoints)
        ]
    )

    # By=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ay + cy * yConst2,
        by + ey * yConst2,
        dy + gy * yConst2,
        fy + hy * yConst2,
        az + cz * yConst2,
        bz + ez * yConst2,
        dz + gz * yConst2,
        fz + hz * yConst2,
    )

    ByBzEndpoints.extend(
        [
            (root[0], yConst2, root[1])
            for root in root_list
            if not is_root_in_list((root[0], yConst2, root[1]), ByBzEndpoints)
        ]
    )

    # Right Surface
    xConst1 = vspace[0][0][f111[0]][f111[1]][f111[2]]
    # Bx=By=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + bx * xConst1,
        cx + ex * xConst1,
        dx + fx * xConst1,
        gx + hx * xConst1,
        ay + by * xConst1,
        cy + ey * xConst1,
        dy + fy * xConst1,
        gy + hy * xConst1,
    )

    BxByEndpoints.extend(
        [
            (xConst1, root[0], root[1])
            for root in root_list
            if not is_root_in_list((xConst1, root[0], root[1]), BxByEndpoints)
        ]
    )

    # Bx=BZ=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + bx * xConst1,
        cx + ex * xConst1,
        dx + fx * xConst1,
        gx + hx * xConst1,
        az + bz * xConst1,
        cz + ez * xConst1,
        dz + fz * xConst1,
        gz + hz * xConst1,
    )

    BxBzEndpoints.extend(
        [
            (xConst1, root[0], root[1])
            for root in root_list
            if not is_root_in_list((xConst1, root[0], root[1]), BxBzEndpoints)
        ]
    )

    # By=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ay + by * xConst1,
        cy + ey * xConst1,
        dy + fy * xConst1,
        gy + hy * xConst1,
        az + bz * xConst1,
        cz + ez * xConst1,
        dz + fz * xConst1,
        gz + hz * xConst1,
    )

    ByBzEndpoints.extend(
        [
            (xConst1, root[0], root[1])
            for root in root_list
            if not is_root_in_list((xConst1, root[0], root[1]), ByBzEndpoints)
        ]
    )

    # Left Surface
    xConst2 = vspace[0][0][f000[0]][f000[1]][f000[2]]
    # Bx=By=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + bx * xConst2,
        cx + ex * xConst2,
        dx + fx * xConst2,
        gx + hx * xConst2,
        ay + by * xConst2,
        cy + ey * xConst2,
        dy + fy * xConst2,
        gy + hy * xConst2,
    )

    BxByEndpoints.extend(
        [
            (xConst2, root[0], root[1])
            for root in root_list
            if not is_root_in_list((xConst2, root[0], root[1]), BxByEndpoints)
        ]
    )

    # Bx=BZ=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + bx * xConst2,
        cx + ex * xConst2,
        dx + fx * xConst2,
        gx + hx * xConst2,
        az + bz * xConst2,
        cz + ez * xConst2,
        dz + fz * xConst2,
        gz + hz * xConst2,
    )

    BxBzEndpoints.extend(
        [
            (xConst2, root[0], root[1])
            for root in root_list
            if not is_root_in_list((xConst2, root[0], root[1]), BxBzEndpoints)
        ]
    )

    # By=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ay + by * xConst2,
        cy + ey * xConst2,
        dy + fy * xConst2,
        gy + hy * xConst2,
        az + bz * xConst2,
        cz + ez * xConst2,
        dz + fz * xConst2,
        gz + hz * xConst2,
    )

    ByBzEndpoints.extend(
        [
            (xConst2, root[0], root[1])
            for root in root_list
            if not is_root_in_list((xConst2, root[0], root[1]), ByBzEndpoints)
        ]
    )

    # Up Surface
    zConst1 = vspace[0][2][f111[0]][f111[1]][f111[2]]
    # Bx=By=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + dx * zConst1,
        bx + fx * zConst1,
        cx + gx * zConst1,
        ex + hx * zConst1,
        ay + dy * zConst1,
        by + fy * zConst1,
        cy + gy * zConst1,
        ey + hy * zConst1,
    )

    BxByEndpoints.extend(
        [
            (root[0], root[1], zConst1)
            for root in root_list
            if not is_root_in_list((root[0], root[1], zConst1), BxByEndpoints)
        ]
    )

    # Bx=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + dx * zConst1,
        bx + fx * zConst1,
        cx + gx * zConst1,
        ex + hx * zConst1,
        az + dz * zConst1,
        bz + fz * zConst1,
        cz + gz * zConst1,
        ez + hz * zConst1,
    )

    BxBzEndpoints.extend(
        [
            (root[0], root[1], zConst1)
            for root in root_list
            if not is_root_in_list((root[0], root[1], zConst1), BxBzEndpoints)
        ]
    )

    # By=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ay + dy * zConst1,
        by + fy * zConst1,
        cy + gy * zConst1,
        ey + hy * zConst1,
        az + dz * zConst1,
        bz + fz * zConst1,
        cz + gz * zConst1,
        ez + hz * zConst1,
    )

    ByBzEndpoints.extend(
        [
            (root[0], root[1], zConst1)
            for root in root_list
            if not is_root_in_list((root[0], root[1], zConst1), ByBzEndpoints)
        ]
    )

    # Down Surface
    zConst2 = vspace[0][2][f000[0]][f000[1]][f000[2]]
    # Bx=By=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + dx * zConst2,
        bx + fx * zConst2,
        cx + gx * zConst2,
        ex + hx * zConst2,
        ay + dy * zConst2,
        by + fy * zConst2,
        cy + gy * zConst2,
        ey + hy * zConst2,
    )

    BxByEndpoints.extend(
        [
            (root[0], root[1], zConst2)
            for root in root_list
            if not is_root_in_list((root[0], root[1], zConst2), BxByEndpoints)
        ]
    )

    # Bx=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ax + dx * zConst2,
        bx + fx * zConst2,
        cx + gx * zConst2,
        ex + hx * zConst2,
        az + dz * zConst2,
        bz + fz * zConst2,
        cz + gz * zConst2,
        ez + hz * zConst2,
    )

    BxBzEndpoints.extend(
        [
            (root[0], root[1], zConst2)
            for root in root_list
            if not is_root_in_list((root[0], root[1], zConst2), BxBzEndpoints)
        ]
    )

    # By=Bz=0 Curve Endpoint
    root_list = _bilinear_root(
        ay + dy * zConst2,
        by + fy * zConst2,
        cy + gy * zConst2,
        ey + hy * zConst2,
        az + dz * zConst2,
        bz + fz * zConst2,
        cz + gz * zConst2,
        ez + hz * zConst2,
    )

    ByBzEndpoints.extend(
        [
            (root[0], root[1], zConst2)
            for root in root_list
            if not is_root_in_list((root[0], root[1], zConst2), ByBzEndpoints)
        ]
    )

    xbound = vspace[0][0][f111[0]][f111[1]][f111[2]]
    ybound = vspace[0][1][f111[0]][f111[1]][f111[2]]
    zbound = vspace[0][2][f111[0]][f111[1]][f111[2]]

    def bound(epoint):
        """
        Checks if the endpoints are located in or on the cube.
        """
        is_x_in_bound = (
            initial[0] < epoint[0]
            or np.isclose(initial[0], epoint[0], atol=_EQUALITY_ATOL)
        ) and (epoint[0] < xbound or np.isclose(epoint[0], xbound, atol=_EQUALITY_ATOL))
        is_y_in_bound = (
            initial[1] < epoint[1]
            or np.isclose(initial[1], epoint[1], atol=_EQUALITY_ATOL)
        ) and (epoint[1] < ybound or np.isclose(epoint[1], ybound, atol=_EQUALITY_ATOL))
        is_z_in_bound = (
            initial[2] < epoint[2]
            or np.isclose(initial[2], epoint[2], atol=_EQUALITY_ATOL)
        ) and (epoint[2] < zbound or np.isclose(epoint[2], zbound, atol=_EQUALITY_ATOL))
        return is_x_in_bound and is_y_in_bound and is_z_in_bound

    BxByEndpoints = list(filter(bound, BxByEndpoints))
    BxBzEndpoints = list(filter(bound, BxBzEndpoints))
    ByBzEndpoints = list(filter(bound, ByBzEndpoints))
    tlApprox = trilinear_approx(vspace, cell)

    # Check on the Surfaces
    for p in BxByEndpoints:
        if np.linalg.norm(tlApprox(p[0], p[1], p[2])) < _EQUALITY_ATOL:
            # print(cell)
            # print(tlApprox(p[0], p[1], p[2]))
            return True
    for p in BxBzEndpoints:
        if np.linalg.norm(tlApprox(p[0], p[1], p[2])) < _EQUALITY_ATOL:
            # print(cell)
            # print(tlApprox(p[0], p[1], p[2]))
            return True
    for p in ByBzEndpoints:
        if np.linalg.norm(tlApprox(p[0], p[1], p[2])) < _EQUALITY_ATOL:
            # print(cell)
            # print(tlApprox(p[0], p[1], p[2]))
            return True

    # Check Grid Resolution
    if len(BxByEndpoints) == len(BxBzEndpoints) == len(ByBzEndpoints) == 0:
        warnings.warn(
            "Multiple null points suspected. Trilinear method may not work as intended.",
            MultipleNullPointWarning,
        )
        return False

    if len(BxByEndpoints) != 2 or len(BxBzEndpoints) != 2 or len(ByBzEndpoints) != 2:
        return False

    def endpoint_sign_check(curve_endpoints, curve_name: str):
        if curve_name == "x":
            index = 0
        elif curve_name == "y":
            index = 1
        elif curve_name == "z":
            index = 2

        first_endpoint = tlApprox(
            curve_endpoints[0][0], curve_endpoints[0][1], curve_endpoints[0][2]
        )[index]
        second_endpoint = tlApprox(
            curve_endpoints[1][0], curve_endpoints[1][1], curve_endpoints[1][2]
        )[index]
        if np.isclose(first_endpoint, 0, atol=_EQUALITY_ATOL) or np.isclose(
            second_endpoint, 0, atol=_EQUALITY_ATOL
        ):
            return True
        return np.sign(first_endpoint) * np.sign(second_endpoint) <= 0

    opposite_sign_z = endpoint_sign_check(BxByEndpoints, "z")
    opposite_sign_y = endpoint_sign_check(BxBzEndpoints, "y")
    opposite_sign_x = endpoint_sign_check(ByBzEndpoints, "x")
    return bool(opposite_sign_x and opposite_sign_y and opposite_sign_z)


def _locate_null_point(vspace, cell, n, err):
    r"""
    Return the coordinates of a null point within a given grid cell in
    a vector space using the Newton-Rapshon method.

    Multiple initial positions are tried until either one converges
    inside a the grid cell, or the maximum iteration is reached. If
    neither occurs, more starting positions are tried, by breaking up
    the cell into 8 smaller sub-grid cells, until one starting position
    does converge or stop inside the grid cell. This process is repeated
    a finite amount of times, after which the function returns `None`.

    Parameters
    ----------
    vspace : |array_like|
        The vector space as constructed by the vector_space function
        which is a 1 by 3 array with the first element containing the
        coordinates, the second element containing the vector values,
        and the third element containing the delta values for each
        dimension.

    cell : |array_like| of integers
        A grid cell, represented by a 1 by 3 array of integers, which
        correspond to a grid cell in the vector space.

    n : int
        The maximum number of times the iterative step of the
        Newton-Raphson method is repeated.

    err : float
        The threshold/error that determines if convergence has occurred
        using the Newton-Raphson method.

    Returns
    -------
    |array_like| of floats
        A 1 by 3 array containing the converged coordinates of the null
        point.

    NoneType
        `None` if the coordinates of the null point could not be
        converged at a point inside the grid cell.

    Warns
    -----
    :`UserWarning`
        If the maximum number of iteration has been reached, but
        convergence has not occurred.
    """
    global _recursion_level  # noqa: PLW0602
    # Calculating the Jacobian and trilinear approximation functions for the cell
    tlApprox = trilinear_approx(vspace, cell)
    jcb = _trilinear_jacobian(vspace, cell)
    # Calculating the deltas
    deltax, deltay, deltaz = vspace[2]
    deltax = deltax[cell[0]]
    deltay = deltay[cell[1]]
    deltaz = deltaz[cell[2]]

    f000 = cell
    f001 = [cell[0], cell[1], cell[2] + 1]
    f010 = [cell[0], cell[1] + 1, cell[2]]
    f011 = [cell[0], cell[1] + 1, cell[2] + 1]
    f100 = [cell[0] + 1, cell[1], cell[2]]
    f101 = [cell[0] + 1, cell[1], cell[2] + 1]
    f110 = [cell[0] + 1, cell[1] + 1, cell[2]]
    f111 = [cell[0] + 1, cell[1] + 1, cell[2] + 1]
    corners = [f000, f001, f010, f011, f100, f101, f110, f111]

    # Critical Coordinates
    pos_000 = np.array(
        [
            vspace[0][0][f000[0]][f000[1]][f000[2]],
            vspace[0][1][f000[0]][f000[1]][f000[2]],
            vspace[0][2][f000[0]][f000[1]][f000[2]],
        ]
    )
    pos_111 = np.array(
        [
            vspace[0][0][f111[0]][f111[1]][f111[2]],
            vspace[0][1][f111[0]][f111[1]][f111[2]],
            vspace[0][2][f111[0]][f111[1]][f111[2]],
        ]
    )

    def in_bound(pos):
        """
        Checks if the estimated position is located inside the cube.
        """
        pos = pos.reshape(1, 3)[0]
        is_x_in_bound = (
            np.isclose(pos_000[0], pos[0], atol=_EQUALITY_ATOL) or pos_000[0] < pos[0]
        ) and (
            np.isclose(pos[0], pos_111[0], atol=_EQUALITY_ATOL) or pos[0] < pos_111[0]
        )
        is_y_in_bound = (
            np.isclose(pos_000[1], pos[1], atol=_EQUALITY_ATOL) or pos_000[1] < pos[1]
        ) and (
            np.isclose(pos[1], pos_111[1], atol=_EQUALITY_ATOL) or pos[1] < pos_111[1]
        )
        is_z_in_bound = (
            np.isclose(pos_000[2], pos[2], atol=_EQUALITY_ATOL) or pos_000[2] < pos[2]
        ) and (
            np.isclose(pos[2], pos_111[2], atol=_EQUALITY_ATOL) or pos[2] < pos_111[2]
        )
        return is_x_in_bound and is_y_in_bound and is_z_in_bound

    starting_pos = []
    # Adding the Corners
    starting_pos = [
        [
            vspace[0][0][point[0]][point[1]][point[2]],
            vspace[0][1][point[0]][point[1]][point[2]],
            vspace[0][2][point[0]][point[1]][point[2]],
        ]
        for point in corners
    ]
    # Adding the Mid Point
    starting_pos.append(
        [
            vspace[0][0][f000[0]][f000[1]][f000[2]] + deltax / 2.0,
            vspace[0][1][f000[0]][f000[1]][f000[2]] + deltay / 2.0,
            vspace[0][2][f000[0]][f000[1]][f000[2]] + deltaz / 2.0,
        ]
    )
    # Newton Iteration
    for x0 in starting_pos:
        x0 = np.array(x0)  # noqa: PLW2901
        x0 = x0.reshape(3, 1)  # noqa: PLW2901
        for _i in range(n):
            locx = tlApprox(x0[0], x0[1], x0[2])[0]
            locy = tlApprox(x0[0], x0[1], x0[2])[1]
            locz = tlApprox(x0[0], x0[1], x0[2])[2]
            Bx0 = np.array([locx, locy, locz])
            Bx0 = Bx0.reshape(3, 1)
            prev_norm = np.linalg.norm(x0)
            # Too many null points if the determinant of the Jacobian is zero
            if np.isclose(
                np.linalg.det(jcb(x0[0], x0[1], x0[2])), 0, atol=_EQUALITY_ATOL
            ):
                warnings.warn(
                    "Multiple null points suspected. Trilinear method may not work as intended.",
                    MultipleNullPointWarning,
                )
                if (
                    np.isclose(locx, 0, atol=_EQUALITY_ATOL)
                    and np.isclose(locy, 0, atol=_EQUALITY_ATOL)
                    and np.isclose(locz, 0, atol=_EQUALITY_ATOL)
                ):
                    return x0
                else:
                    break
            # Adjust position
            x0 = np.subtract(  # noqa: PLW2901
                x0, np.matmul(np.linalg.inv(jcb(x0[0], x0[1], x0[2])), Bx0)
            )
            norm = np.linalg.norm(x0)
            if np.abs((norm - prev_norm) / (prev_norm + 1e-10)) < err and in_bound(x0):
                return x0
        if in_bound(x0):
            warnings.warn("Max Iterations Reached without Convergence")
            if (
                np.isclose(locx, 0, atol=_EQUALITY_ATOL)
                and np.isclose(locy, 0, atol=_EQUALITY_ATOL)
                and np.isclose(locz, 0, atol=_EQUALITY_ATOL)
            ):
                return x0
    warnings.warn("Various starting points did not locate possible null point.")
    # Generate new starting points localized into 8 small cells?
    return None


def _classify_null_point(vspace, cell, loc):
    r"""
    Return the coordinates of a null point within a given grid cell in a
    vector space using the Newton-Rapshon method.

    Multiple initial positions are tried until either one converges
    inside a the grid cell, or the maximum iteration is reached. If
    neither occurs, more starting positions are tried, by breaking up
    the cell into 8 smaller sub-grid cells, until one starting position
    does converge or stop inside the grid cell. This process is repeated
    a finite amount of times, after which the function returns `None`.

    Parameters
    ----------
    vspace: |array_like|
        The vector space as constructed by the vector_space function
        which is a 1 by 3 array with the first element containing the
        coordinates, the second element containing the vector values,
        and the third element containing the delta values for each
        dimension.

    cell: |array_like| of integers
        A grid cell, represented by a 1 by 3 array of integers, which
        correspond to a grid cell in the vector space.

    Returns
    -------
    str
        A string describing the null point type.

    Raises
    ------
    NonZeroDivergence
        If the divergence of the given vector space is not sufficiently
        close to zero at the null point.

    Notes
    -----
    This method is described by :cite:t:`parnell:1996`.
    """
    jcb = _trilinear_jacobian(vspace, cell)
    M = jcb(loc[0], loc[1], loc[2])
    if not np.isclose(np.trace(M), 0, atol=_EQUALITY_ATOL):
        raise NonZeroDivergence
    eigen_vals, eigen_vectors = np.linalg.eig(M)
    # using the notation from Parnell et al. (1996)
    R = -1.0 * np.linalg.det(M)
    Q = -0.5 * np.trace(np.matmul(M, M))

    discriminant = (Q**3 / 27.0) + (R**2 / 4.0)
    determinant = -1.0 * R
    if np.isclose(discriminant, 0, atol=_EQUALITY_ATOL):
        if np.allclose(M, M.T, atol=_EQUALITY_ATOL):  # Checking if M is symmetric
            null_point_type = "Proper radial null"
        else:
            null_point_type = (
                "Anti-parallel lines with null plane OR Planes of parabolae with null line"
                if np.isclose(determinant, 0, atol=_EQUALITY_ATOL)
                else "Critical spiral null"
            )
    elif discriminant < 0:
        if np.allclose(M, M.T, atol=_EQUALITY_ATOL):
            null_point_type = (
                "Continuous potential X-points"
                if np.isclose(determinant, 0, atol=_EQUALITY_ATOL)
                else "Improper radial null"
            )
        elif np.isclose(determinant, 0, atol=_EQUALITY_ATOL):
            null_point_type = "Continuous X-points"
        else:
            null_point_type = "Skewed improper null"
    elif np.isclose(determinant, 0, atol=_EQUALITY_ATOL):
        null_point_type = "Continuous concentric ellipses"
    else:
        null_point_type = "Spiral null"
    return null_point_type


def _vspace_iterator(vspace, maxiter: int = 500, err: float = 1e-10):
    r"""
    Returns an array of null point objects, representing the null points
    of the given vector space.

    Parameters
    ----------
    vspace : |array_like|
        The vector space as constructed by the ``_vector_space``
        function which is a 1 by 3 array with the first element
        containing the coordinates, the second element containing the
        vector values, and the third element containing the delta values
        for each dimension.

    maxiter : int, default: 500
        The maximum iterations of the Newton-Raphson method.

    err : float, default: ``1e-10``
        The threshold/error that determines if convergence has occurred
        using the Newton-Raphson method.

    Returns
    -------
    |array_like| of `~plasmapy.analysis.nullpoint.NullPoint`
        An array of `~plasmapy.analysis.nullpoint.NullPoint` objects
        representing the null points of the given vector space.
    """
    nullpoints = []
    for i in range(len(vspace[0][0]) - 1):
        for j in range(len(vspace[0][0][0]) - 1):
            for k in range(len(vspace[0][0][0][0]) - 1):
                if _reduction(vspace, [i, j, k]) and _trilinear_analysis(
                    vspace, [i, j, k]
                ):
                    loc = _locate_null_point(vspace, [i, j, k], maxiter, err)
                    if loc is not None:
                        null_type = _classify_null_point(vspace, [i, j, k], loc)
                        p = NullPoint(loc, null_type)
                        if p not in nullpoints:
                            nullpoints.append(p)
    return nullpoints


def null_point_find(
    x_arr=None,
    y_arr=None,
    z_arr=None,
    u_arr=None,
    v_arr=None,
    w_arr=None,
    maxiter: int = 500,
    err: float = 1e-10,
):
    r"""
    Returns an array of `~plasmapy.analysis.nullpoint.NullPoint` object,
    representing the null points of the given vector space.

    .. note::

       This functionality is still under development and the API may
       change in future releases.

    Parameters
    ----------
    x_arr: |array_like|
        The array representing the coordinates in the x-dimension.
        If not given, then range values are used to construct a
        uniform array on that interval.

    y_arr: |array_like|
        The array representing the coordinates in the y-dimension. If
        not given, then range values are used to construct a uniform
        array on that interval.

    z_arr: |array_like|
        The array representing the coordinates in the z-dimension. If
        not given, then range values are used to construct a uniform
        array on that interval.

    u_arr: |array_like|
        A 3D array containing the x-component of the vector values for
        the vector space. If not given, the vector values are generated
        over the vector space using the function func.

    v_arr: |array_like|
        A 3D array containing the y-component of the vector values for
        the vector space. If not given, the vector values are generated
        over the vector space using the function func.

    w_arr: |array_like|
        A 3D array containing the z-component of the vector values for
        the vector space. If not given, the vector values are generated
        over the vector space using the function func.

    maxiter: int, default: 500
        The maximum iterations of the Newton-Raphson method.

    err: float, default: ``1e-10``
        The threshold/error that determines if convergence has occurred
        using the Newton-Raphson method.

    Returns
    -------
    |array_like| of `~plasmapy.analysis.nullpoint.NullPoint`
        An array of `~plasmapy.analysis.nullpoint.NullPoint` objects
        representing the null points of the given vector space.

    Notes
    -----
    This method is described by :cite:t:`haynes:2007`.
    """
    # Constructing the vspace
    vspace = _vector_space(
        x_arr,
        y_arr,
        z_arr,
        None,
        None,
        None,
        u_arr,
        v_arr,
        w_arr,
        None,
        None,
    )
    return _vspace_iterator(vspace, maxiter, err)


def uniform_null_point_find(
    x_range,
    y_range,
    z_range,
    func: Callable,
    precision=(0.05, 0.05, 0.05),
    maxiter: int = 500,
    err: float = 1e-10,
):
    r"""
    Return an array of `~plasmapy.analysis.nullpoint.NullPoint` objects,
    representing the null points of the given vector space.

    .. note::

       This functionality is still under development and the API may
       change in future releases.

    Parameters
    ----------
    x_range: |array_like|, default: ``[0, 1]``
        A 1 by 2 array containing the range of x-values for the vector
        spaces.

    y_range: |array_like|, default: ``[0, 1]``
        A 1 by 2 array containing the range of y-values for the vector
        spaces.

    z_range: |array_like|, default: ``[0, 1]``
        A 1 by 2 array containing the range of z-values for the vector
        spaces.

    func: function
        A function that takes in 3 arguments, respectively representing
        an :math:`(x, y, z)` coordinate of a point and returns the
        vector value for that point in the form of a 1 by 3 array.

    precision: |array_like|, default: ``[0.05, 0.05, 0.05]``
        A 1 by 3 array containing the approximate precision values for
        each dimension, in the case where uniform arrays are being used.

    Returns
    -------
    |array_like| of `~plasmapy.analysis.nullpoint.NullPoint`
        An array of `~plasmapy.analysis.nullpoint.NullPoint` objects
        representing the null points of the given vector space.

    Notes
    -----
    This method is described by :cite:t:`haynes:2007`.
    """
    vspace = _vector_space(
        None,
        None,
        None,
        x_range,
        y_range,
        z_range,
        None,
        None,
        None,
        func,
        precision,
    )
    return _vspace_iterator(vspace, maxiter, err)
