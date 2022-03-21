"""
Defines the PathIntegratedDiagnostic base class and subclasses, including
LineIntegratedDiagnostic.
"""

__all__ = [
    "PathIntegratedDiagnostic",
    "LineIntegratedDiagnostic",
    "LineIntegrateScalarQuantities",
    "Interferometer",
]

import astropy.constants as const
import astropy.units as u
import numpy as np

from abc import ABC, abstractmethod
from typing import Union


class PathIntegratedDiagnostic(ABC):
    def __init__(
        self,
        grid,
        source,
        detector,
        detector_hdir=None,
        verbose=True,
    ):
        """
        An abstract class representing a path-integrated diagnostic described
        by a source, grid (object), and detector plane. This abstract class
        contains methods common to this problem geometry.

        Parameters
        ----------

        grid : `~plasmapy.plasma.grids.AbstractGrid` or subclass thereof
            A Grid object containing scalar quantities.

        source : `~astropy.units.Quantity`, shape (3)
            A vector pointing from the origin of the grid to the location
            of the particle source. This vector will be interpreted as
            being in either cartesian, cylindrical, or spherical coordinates
            based on its units. Valid geometries are:

            * Cartesian (x,y,z) : (meters, meters, meters)
            * cylindrical (r, theta, z) : (meters, radians, meters)
            * spherical (r, theta, phi) : (meters, radians, radians)

            In spherical coordinates theta is the polar angle.

        detector : `~astropy.units.Quantity`, shape (3)
            A vector pointing from the origin of the grid to the center
            of the detector plane. The vector from the source point to this
            point defines the normal vector of the detector plane. This vector
            can also be specified in cartesian, cylindrical, or spherical
            coordinates (see the `source` keyword).

        detector_hdir : `numpy.ndarray`, shape (3), optional
            A unit vector (in Cartesian coordinates) defining the horizontal
            direction on the detector plane. By default, the horizontal axis in the
            detector plane is defined to be perpendicular to both the
            source-to-detector vector and the z-axis (unless the source-to-detector axis
            is parallel to the z axis, in which case the horizontal axis is the x-axis).

            The detector vertical axis is then defined
            to be orthogonal to both the source-to-detector vector and the
            detector horizontal axis.

        verbose : bool, optional
            If true, updates on the status of the program will be printed
            into the standard output while running.

        """

        # self.grid is the grid object
        self.grid = grid

        # self.grid_arr is the grid positions in si. This is created here
        # so that it isn't continously called later
        self.grid_arr = grid.grid.to(u.m).value

        self.verbose = verbose

        self.source = self._coerce_to_cartesian_si(source)
        self.detector = self._coerce_to_cartesian_si(detector)
        self._log(f"Source: {self.source} m")
        self._log(f"Detector: {self.detector} m")

        # Calculate normal vectors (facing towards the grid origin) for both
        # the source and detector planes
        self.src_n = -self.source / np.linalg.norm(self.source)
        self.det_n = -self.detector / np.linalg.norm(self.detector)
        # Vector directly from source to detector
        self.src_det = self.detector - self.source

        # Experiment axis is the unit vector from the source to the detector
        self.src_det_n = self.src_det / np.linalg.norm(self.src_det)

        self.mag = 1 + np.linalg.norm(self.detector) / np.linalg.norm(self.source)
        self._log(f"Magnification: {self.mag}")

        # Calculate the scale of the grid
        # (used for auto-choosing the size keyword when making histogram images)
        self.grid_scale_length = np.max(
            [
                np.max(np.abs(self.grid.pts0.to(u.m).value)),
                np.max(np.abs(self.grid.pts1.to(u.m).value)),
                np.max(np.abs(self.grid.pts2.to(u.m).value)),
            ]
        )

        # Check that source-detector vector actually passes through the grid
        if not self.grid.vector_intersects(self.source * u.m, self.detector * u.m):
            raise ValueError(
                "The vector between the source and the detector "
                "does not intersect the grid provided!"
            )

        # ************************************************************************
        # Define the detector plane
        # ************************************************************************

        # Load or calculate the detector hdir
        if detector_hdir is not None:
            self.det_hdir = detector_hdir / np.linalg.norm(detector_hdir)
        else:
            self.det_hdir = self._default_detector_hdir()

        # Calculate the detector vdir
        ny = np.cross(self.det_hdir, self.det_n)
        self.det_vdir = -ny / np.linalg.norm(ny)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _coerce_to_cartesian_si(self, pos):
        """
        Takes a tuple of `~astropy.unit.Quantity` values representing a position
        in space in either Cartesian, cylindrical, or spherical coordinates, and
        returns a numpy array representing the same point in Cartesian
        coordinates and units of meters.
        """
        # Auto-detect geometry based on units
        geo_units = [x.unit for x in pos]
        if geo_units[2].is_equivalent(u.rad):
            geometry = "spherical"
        elif geo_units[1].is_equivalent(u.rad):
            geometry = "cylindrical"
        else:
            geometry = "cartesian"

        # Convert geometrical inputs between coordinates systems
        pos_out = np.zeros(3)
        if geometry == "cartesian":
            x, y, z = pos
            pos_out[0] = x.to(u.m).value
            pos_out[1] = y.to(u.m).value
            pos_out[2] = z.to(u.m).value

        elif geometry == "cylindrical":
            r, t, z = pos
            r = r.to(u.m)
            t = t.to(u.rad).value
            z = z.to(u.m)
            pos_out[0] = (r * np.cos(t)).to(u.m).value
            pos_out[1] = (r * np.sin(t)).to(u.m).value
            pos_out[2] = z.to(u.m).value

        elif geometry == "spherical":
            r, t, p = pos
            r = r.to(u.m)
            t = t.to(u.rad).value
            p = p.to(u.rad).value

            pos_out[0] = (r * np.sin(t) * np.cos(p)).to(u.m).value
            pos_out[1] = (r * np.sin(t) * np.sin(p)).to(u.m).value
            pos_out[2] = (r * np.cos(t)).to(u.m).value

        return pos_out

    def _default_detector_hdir(self):
        """
        Calculates the default horizontal unit vector for the detector plane
        (see __init__ description for details)
        """
        # Create unit vectors that define the detector plane
        # Define plane  horizontal axis
        if np.allclose(np.abs(self.det_n), np.array([0, 0, 1])):
            nx = np.array([1, 0, 0])
        else:
            nx = np.cross(np.array([0, 0, 1]), self.det_n)
        nx = nx / np.linalg.norm(nx)
        return nx


class LineIntegratedDiagnostic(PathIntegratedDiagnostic):
    def _line_integral(
        self,
        num,
        size=None,
        bins=[200, 200],
        collimated=True,
    ):
        """
        Calculates the line-integral of the integrand function through the
        provided grid. This is accomplished through the following steps:

        1) Create an array of points in the detector plane using the detector
        location and the size and bins keywords.

        2) For each cell of detector grid, create an array of points of
        separation ds from there to the source point (or, when collimated=True,
        the source plane).

        3) Evaluate the integrand function at each point.

        4) Integrate along the line-integrated dimension to obtain the
        line-integrated quantity in the detector plane.

        Parameters
        ----------
        num : int
            Number of integration points along the line (within the grid region).

        size : `~astropy.units.Quantity` array of shape [2,2], optional
            The bounds of the detector region. By default the size will be
            chosen to be slightly larger than the image of the grid on the
            detector plane.

        bins : list of two ints, [N,], optional
            Number of bins in each direction of the detector region. The
            default is [200, 200].

        collimated : Boolean, optional
            If True, the source will be assumed to be collimated. If False,
            a point source will be used. The default is True (collimated).

        Returns
        -------
        xax : `~astropy.units.Quantity` array (Nh,)
            The horizontal axis of the detector plane

        yax : `~astropy.units.Quantity` array of shape (Nv,)
            The vertical axis of the detector plane

        integral : `~astropy.units.Quantity` array
           The line-integrated values in the detector plane.

        *integral : `~astropy.units.Quantity` array
            If multiple quantities are specified, multiple integral arrays
            will be returned.

        """

        if size is None:
            w = self.mag * self.grid_scale_length
            # Factor of 1.5 provides border on image
            size = 1.5 * np.array([[-w, w], [-w, w]]) * u.m

        # Create arrays of detector grid points
        xax = np.linspace(
            size[0][0].to(u.m).value, size[0][1].to(u.m).value, num=int(bins[0])
        )
        yax = np.linspace(
            size[1][0].to(u.m).value, size[1][1].to(u.m).value, num=int(bins[1])
        )
        x_offset, y_offset = np.meshgrid(xax, yax, indexing="ij")

        # Shift those points in space to be in the detector plane
        det_pts = (
            np.outer(x_offset, self.det_hdir)
            + np.outer(y_offset, self.det_vdir)
            + self.detector
        )
        det_pts = np.reshape(det_pts, [bins[0], bins[1], 3])

        # Create 2D grids of source points
        if collimated:
            src_pts = det_pts - self.src_det
        else:
            # If not collimated, assume a point source
            src_pts = np.outer(np.ones([bins[0], bins[1]]), self.source)
            src_pts = np.reshape(src_pts, (bins[0], bins[1], 3))

        # Determine where the grid begins and ends as fractions of the
        # source-to-detector vector
        source_to_det = np.linalg.norm(self.src_det)
        source_to_grid = np.min(np.linalg.norm(self.grid_arr - self.source, axis=-1))
        grid_to_det = np.min(np.linalg.norm(self.grid_arr - self.detector, axis=-1))

        # Paramter for parameteric equation
        start = source_to_grid / source_to_det
        stop = 1 - grid_to_det / source_to_det
        i = np.linspace(start, stop, num=num)
        ds = (stop - start) * source_to_det / num

        # Create an array of points along lines between points in src_pts
        # and det_pts of the equation m*i + b where
        # m = det_pts - src_pts
        # Integrating along the lines corresponds to summation along ax=2
        mi = np.outer(det_pts - src_pts, i)
        mi = np.reshape(mi, [bins[0], bins[1], 3, num])

        b = np.outer(src_pts, np.ones(num))
        b = np.reshape(b, [bins[0], bins[1], 3, num])

        pts = (mi + b) * u.m
        pts = np.moveaxis(pts, 2, 3)

        # Evaluate the integrands
        integrands = self._integrand(pts)

        # If a single integrand is returned, put it in a list
        if not isinstance(integrands, tuple):
            integrands = (integrands,)

        # Integrate
        integral = []
        for integrand in integrands:

            # Convert NaNs (off-grid values from interpolator) to zeros
            integrand = np.nan_to_num(integrand, nan=0.0)

            # Integrate
            integral.append(np.trapz(integrand, axis=2) * (ds * u.m))

        return (xax * u.m, yax * u.m, *integral)

    @abstractmethod
    def _integrand(self, pts):
        """
        Returns the integrand at a particular position.

        This method is over-written by subclasses to reflect actual
        diagnostic physics, and can pull plasma parameter information from
        the parameters dict provided.

        Parameters
        ----------

        pts: `~astropy.units.Quantity` (nx*ny*nz, 3)
            Positions at which the integrand will be
            evaluated.

        Returns
        -------

        arr: `~astropy.units.Quantity` (nx*ny*nz) or list of same
            Integrand value at each of the points provided. Some integrand
            functions may return multiple arrays of interpolated values as
            a list, each of which will then be integrated separately.
        """
        ...

    @abstractmethod
    def evaluate(self):
        """
        Runs the line-integration routine, then performs some calculation
        on top of the result to return a synthetic diagnostic image.

        This method is over-written by subclasses to reflect actual
        diagnostic physics.
        """
        ...


class LineIntegrateScalarQuantities(LineIntegratedDiagnostic):
    def __init__(
        self,
        grid,
        source,
        detector,
        quantities: Union[str, list, tuple],
        verbose=True,
    ):

        """
        A subclass of LineIntegratedDiagnostic that integrates one or more
        scalar quantities on the grid provided.

        Parameters
        ----------

        grid : `~plasmapy.plasma.grids.AbstractGrid` or subclass thereof
            A Grid object containing scalar quantities.

        source : `~astropy.units.Quantity`, shape (3)
            A vector pointing from the origin of the grid to the location
            of the particle source. This vector will be interpreted as
            being in either cartesian, cylindrical, or spherical coordinates
            based on its units. Valid geometries are:

            * Cartesian (x,y,z) : (meters, meters, meters)
            * cylindrical (r, theta, z) : (meters, radians, meters)
            * spherical (r, theta, phi) : (meters, radians, radians)

            In spherical coordinates theta is the polar angle.

        detector : `~astropy.units.Quantity`, shape (3)
            A vector pointing from the origin of the grid to the center
            of the detector plane. The vector from the source point to this
            point defines the normal vector of the detector plane. This vector
            can also be specified in cartesian, cylindrical, or spherical
            coordinates (see the `source` keyword).

        detector_hdir : `numpy.ndarray`, shape (3), optional
            A unit vector (in Cartesian coordinates) defining the horizontal
            direction on the detector plane. By default, the horizontal axis in the
            detector plane is defined to be perpendicular to both the
            source-to-detector vector and the z-axis (unless the source-to-detector axis
            is parallel to the z axis, in which case the horizontal axis is the x-axis).

            The detector vertical axis is then defined
            to be orthogonal to both the source-to-detector vector and the
            detector horizontal axis.

        verbose : bool, optional
            If true, updates on the status of the program will be printed
            into the standard output while running.

        """

        # Validate the quantities input
        if isinstance(quantities, str):
            quantities = [
                quantities,
            ]

        for quantity in quantities:
            if quantity not in grid.quantities:
                raise ValueError(
                    f"quantity {quantity} is not defined on the " "provided grid."
                )

        self.quantities = quantities

        # Continue with the rest of the parent class init
        super().__init__(grid, source, detector, verbose=verbose)

    def _integrand(self, pts, interpolator="nearest_neighbor"):
        r"""
        Returns the scalar value of the quantity or quantities specified
        at each point.

        Parameters
        ----------

        pts: `~astropy.units.Quantity` (nx*ny*nz, 3)
            Positions at which the integrand will be evaluated.

        interpolator: str, optional
            Determines which interpolator will be used. Must correspond to
            one of the interpolators defined on the given grid, eg.
            * ``nearest_neighbor``
            * ``volume_averaged``

        Returns
        -------

        arr: `~astropy.units.Quantity` (nx*ny*nz) or list of same
            Integrand value at each of the points provided. Some integrand
            functions may return multiple arrays of interpolated values as
            a list, each of which will then be integrated separately.


        """
        # Reshape the pts array from grid shape (nx, ny, nz, 3) to a list
        # of points (nx*ny*nz, 3) as required by the grids interpolators
        nx, ny, nz, ndim = pts.shape
        pts = np.reshape(pts, (nx * ny * nz, ndim))

        # If volume_averaged_interpolator not defined for this grid
        # (eg. for non-uniform grids)
        # use the nearest neighbor interpolator instead

        if interpolator == "nearest_neighbor":
            integrand = self.grid.nearest_neighbor_interpolator(pts, *self.quantities)
        elif interpolator == "volume_averaged":
            integrand = self.grid.volume_averaged_interpolator(pts, *self.quantities)
        else:
            raise ValueError(f"Interpolator specified does not exist: {interpolator}")

        # Reshape the integrands from (nx*ny*nz) to (nx, ny, nz)
        integrand = np.reshape(integrand, (nx, ny, nz))

        return integrand

    def evaluate(
        self,
        num,
        size=None,
        bins=[200, 200],
        collimated=True,
    ):
        r"""
        Evaluates the line integral through the
        provided grid. This is accomplished through the following steps:

        1. Create an array of points in the detector plane using the detector
           location and the size and bins keywords.

        2. For each cell of detector grid, create an array of points of
           separation ds from there to the source point (or,
           when collimated=True, the source plane).

        3. Evaluate the integrand function at each point.

        4. Integrate along the line-integrated dimension to obtain the
           line-integrated quantity in the detector plane.

        Parameters
        ----------
        num : int
            Number of integration points along the line (within the grid region).

        size : `~astropy.units.Quantity` array of shape [2,2], optional
            The bounds of the detector region. By default the size will be
            chosen to be slightly larger than the image of the grid on the
            detector plane.

        bins : list of two ints, [N,], optional
            Number of bins in each direction of the detector region. The
            default is [200, 200].

        collimated : Boolean, optional
            If True, the source will be assumed to be collimated. If False,
            a point source will be used. The default is True (collimated).

        Returns
        -------
        xax : `~astropy.units.Quantity` array (Nh,)
            The horizontal axis of the detector plane

        yax : `~astropy.units.Quantity` array of shape (Nv,)
            The vertical axis of the detector plane

        integral : `~astropy.units.Quantity` array or list of arrays of shape (Nh, Ny)
           The line-integrated values in the detector plane.

        """
        return self._line_integral(
            num,
            size=size,
            bins=bins,
            collimated=collimated,
        )


class Interferometer(LineIntegrateScalarQuantities):
    def __init__(self, grid, source, detector, verbose=False):
        super().__init__(grid, source, detector, quantities="n_e", verbose=verbose)

    def evaluate(
        self,
        probe_freq: u.Hz,
        num,
        size=None,
        bins=[200, 200],
        collimated=True,
        unwrapped=True,
    ):
        r"""
        Creates an interferogram by calculating the line integral through
        the electron number density :math:`n_e` provided on the grid. The phase shift
        is

        .. math:: \\Delta \\phi = -\\frac{\omega_{probe}}{2 c n_c} \\int n_e dl

        where :math:`\omega_{probe}` is the probe beam frequency, :math:`c`
        is the speed of light, :math:`\int n_e dl` is the line-integrated
        electron density, and :math:`n_c` is the critical density

        .. math:: n_c = \\frac{\\epsilon_0 m_e}{e^2} \\omega_{probe}^2

        Parameters
        ----------
        probe_freq : `~astropy.units.Quantity`
            Frequency of the probe beam, in units convertable to Hz.

        num : int
            Number of integration points along the line (within the grid region).

        size : `~astropy.units.Quantity` array of shape [2,2], optional
            The bounds of the detector region. By default the size will be
            chosen to be slightly larger than the image of the grid on the
            detector plane.

        bins : list of two ints, [N,], optional
            Number of bins in each direction of the detector region. The
            default is [200, 200].

        collimated : bool, optional
            If True, the source will be assumed to be collimated. If False,
            a point source will be used. The default is True (collimated).

        unwrapped : bool, optional
            If True, the total phase shift will be returned (without
            :math:`\pi/2` discontinuities). If False, the phase shift with
            :math:`\pi/2` discontinuities (the value measured experimentally)
            will be returned. the default is True (unwrapped).


        Returns
        -------
        hax : `~astropy.units.Quantity` array (Nh,)
            The horizontal axis of the detector plane

        vax : `~astropy.units.Quantity` array of shape (Nv,)
            The vertical axis of the detector plane

        phase_shift : `~astropy.units.Quantity` array or list of arrays of shape (Nh, Ny)
           The phase shift measured in the detector plane.

        """

        # TODO: implement an actual critical density function for PlasmaPy
        # Critical density in cm^-3
        n_c = (const.eps0.si * const.m_e / const.e.si ** 2) * (
            2 * np.pi * probe_freq ** 2
        )
        n_c = n_c.to(u.cm ** -3)

        hax, vax, int_ne = self._line_integral(
            size=size, bins=bins, collimated=collimated, num=num
        )

        phase_shift = (-np.pi * probe_freq / (const.c.si * n_c)) * int_ne

        phase_shift = phase_shift.to(u.dimensionless_unscaled).value

        if not unwrapped:
            # Thanks to helpful stack exchange answer for this compact expression
            # https://stackoverflow.com/a/15927914
            phase_shift = (phase_shift + np.pi) % (2 * np.pi) - np.pi

        return hax, vax, phase_shift
