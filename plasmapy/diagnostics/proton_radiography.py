"""
Routines for the analysis of proton radiographs. These routines can be broadly
classified as either creating synthetic radiographs from prescribed fields or
methods of 'inverting' experimentally created radiographs to reconstruct the
original fields (under some set of assumptions).
"""

__all__ = [
    "SyntheticProtonRadiograph",
]

import astropy.constants as const
import astropy.units as u
import numpy as np
import sys
import warnings

from tqdm import tqdm

from plasmapy import particles
from plasmapy.formulary.mathematics import rot_a_to_b
from plasmapy.particles import Particle
from plasmapy.plasma.grids import AbstractGrid
from plasmapy.simulation.particle_integrators import boris_push


def _coerce_to_cartesian_si(pos):
    """
    Takes a tuple of `astropy.unit.Quantity` values representing a position
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


class SyntheticProtonRadiograph:
    r"""
    Represents a charged particle radiography experiment with simulated or
    calculated E and B fields given at positions defined by a grid of spatial
    coordinates. The particle source and detector plane are defined by vectors
    from the origin of the grid.

    Parameters
    ----------
    grid : `~plasmapy.plasma.grids.AbstractGrid` or subclass thereof
        A Grid object containing the required quantities [E_x, E_y, E_z, B_x, B_y, B_z].
        If any of these quantities are missing, a warning will be given and that
        quantity will be assumed to be zero everywhere.

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

    def __init__(
        self,
        grid: AbstractGrid,
        source: u.m,
        detector: u.m,
        detector_hdir=None,
        verbose=True,
    ):

        # self.grid is the grid object
        self.grid = grid
        # self.grid_arr is the grid positions in si units. This is created here
        # so that it isn't continously called later
        self.grid_arr = grid.grid.to(u.m).value

        self.verbose = verbose

        # A list of wire meshes added to the grid with add_wire_mesh
        # Particles that would hit these meshes will be removed at runtime
        # by _apply_wire_mesh
        self.mesh_list = []

        # ************************************************************************
        # Setup the source and detector geometries
        # ************************************************************************

        self.source = _coerce_to_cartesian_si(source)
        self.detector = _coerce_to_cartesian_si(detector)
        self._log(f"Source: {self.source} m")
        self._log(f"Detector: {self.detector} m")

        # Calculate normal vectors (facing towards the grid origin) for both
        # the source and detector planes
        self.src_n = -self.source / np.linalg.norm(self.source)
        self.det_n = -self.detector / np.linalg.norm(self.detector)

        # Vector directly from source to detector
        self.src_det = self.detector - self.source

        # Magnification
        self.mag = 1 + np.linalg.norm(self.detector) / np.linalg.norm(self.source)
        self._log(f"Magnification: {self.mag}")

        # Check that source-detector vector actually passes through the grid
        if not self.grid.vector_intersects(self.source * u.m, self.detector * u.m):
            raise ValueError(
                "The vector between the source and the detector "
                "does not intersect the grid provided!"
            )

        # Determine the angle above which particles will not hit the grid
        # these particles can be ignored until the end of the simulation,
        # then immediately advanced to the detector grid with their original
        # velocities
        self.max_theta_hit_grid = self._max_theta_hit_grid()

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

        # ************************************************************************
        # Validate the E and B fields
        # ************************************************************************

        req_quantities = ["E_x", "E_y", "E_z", "B_x", "B_y", "B_z"]

        self.grid.require_quantities(req_quantities, replace_with_zeros=True)

        for rq in req_quantities:

            # Check that there are no infinite values
            if not np.isfinite(self.grid[rq].value).all():
                raise ValueError(
                    f"Input arrays must be finite: {rq} contains "
                    "either NaN or infinite values."
                )

            # Check that the max values on the edges of the arrays are
            # small relative to the maximum values on that grid
            #
            # Array must be dimensionless to re-assemble it into an array
            # of max values like this
            arr = np.abs(self.grid[rq]).value
            edge_max = np.max(
                np.array(
                    [
                        np.max(arr[0, :, :]),
                        np.max(arr[-1, :, :]),
                        np.max(arr[:, 0, :]),
                        np.max(arr[:, -1, :]),
                        np.max(arr[:, :, 0]),
                        np.max(arr[:, :, -1]),
                    ]
                )
            )

            if edge_max > 1e-3 * np.max(arr):
                unit = grid.recognized_quantities[rq].unit
                warnings.warn(
                    "Fields should go to zero at edges of grid to avoid "
                    f"non-physical effects, but a value of {edge_max:.2E} {unit} was "
                    f"found on the edge of the {rq} array. Consider applying a "
                    "envelope function to force the fields at the edge to go to "
                    "zero.",
                    RuntimeWarning,
                )

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

    def _max_theta_hit_grid(self):
        r"""
        Using the grid and the source position, compute the maximum particle
        theta that will impact the grid. This value can be used to determine
        which particles are worth tracking.
        """
        ind = 0
        theta = np.zeros([8])
        for x in [0, -1]:
            for y in [0, -1]:
                for z in [0, -1]:
                    # Source to grid corner vector
                    vec = self.grid_arr[x, y, z, :] - self.source

                    # Calculate angle between vec and the source-to-detector
                    # axis, which is the central axis of the particle beam
                    theta[ind] = np.arccos(
                        np.dot(vec, self.src_det)
                        / np.linalg.norm(vec)
                        / np.linalg.norm(self.src_det)
                    )
                    ind += 1
        return np.max(theta)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # Define some constants so they don't get constantly re-evaluated
    _c = const.c.si.value

    # *************************************************************************
    # Create mesh
    # *************************************************************************

    def add_wire_mesh(
        self, location, extent, nwires, wire_diameter, mesh_hdir=None, mesh_vdir=None
    ):
        """
        Add a wire mesh grid between the particle source and the object grid
        that blocks particles whose paths intersect the wires.

        Parameters
        ----------
        location : `~astropy.units.Quantity`, shape (3)
            A vector pointing from the origin of the grid to the center of the
            mesh grid. This location must be between the source and the
            object grid.

            This vector will be interpreted as
            being in either cartesian, cylindrical, or spherical coordinates
            based on its units. Valid geometries are:

            * Cartesian (x,y,z) : (meters, meters, meters)
            * cylindrical (r, theta, z) : (meters, radians, meters)
            * spherical (r, theta, phi) : (meters, radians, radians)

            In spherical coordinates theta is the polar angle.

        extent : Tuple of 1 or 2 `~astropy.units.Quantity`
            The size of the mesh grid (in the mesh plane). If one value
            is provided, the mesh is circular and the value provided is
            interpreted as the diameter. If two values are provided, the
            mesh is rectangular and they the values are interpreted as the
            width and height respectively.

        nwires : Tuple of 1 or 2 ints, or a single int
            The number of wires in the horizontal and vertical directions. If
            only one value is provided, the number in the two directions is
            assumed to be equal. Note that a wire will cross the center of the
            mesh only when nwires is odd.

        wire_diameter : `~astropy.units.Quantity`
            The diameter of the wires.

        mesh_hdir : `numpy.ndarray`, shape (3), optional
            A unit vector (in Cartesian coordinates) defining the horizontal
            direction on the mesh plane. Modifying this vector can rotate the
            mesh in the plane or tilt the mesh plane relative to the
            source-detector axis. By default, `mesh_hdir` is set equal to
            `detector_hdir` (see `detector_hdir` keyword in `__init__`).

        mesh_vdir : `numpy.ndarray`, shape (3), optional
            A unit vector (in Cartesian coordinates) defining the vertical
            direction on the mesh plane. Modifying this vector can tilt the
            mesh relative to the source-detector axis. By default, `mesh_vdir`
            is defined to be perpendicular to `mesh_hdir` and the detector
            plane normal (such that the mesh is parallel to the detector plane).

        Raises
        ------
        ValueError
            Raises a ValueError if the provided mesh location is not
            between the source and the object grid.

        """

        location = _coerce_to_cartesian_si(location)
        wire_radius = wire_diameter.si.value / 2

        if not isinstance(extent, tuple):
            extent = (extent,)

        if len(extent) == 1:
            radius = 0.5 * extent[0].si.value
            width = extent[0].si.value
            height = extent[0].si.value
        elif len(extent) == 2:
            radius = None
            width = extent[0].si.value
            height = extent[1].si.value
        else:
            raise ValueError(
                "extent must be a tuple of 1 or 2 elements, but "
                f"{len(extent)} elements were provided."
            )

        if not isinstance(nwires, tuple):
            nwires = (nwires,)

        if len(nwires) != 2:
            nwires = (nwires[0], nwires[0])

        # If no hdir/vdir is specified, calculate a default value
        # If one is specified, make sure it is normalized
        if mesh_hdir is None:
            # Re-calculate the default here, in case the user
            # specified a different det_hdir
            mesh_hdir = self._default_detector_hdir()
        else:
            mesh_hdir = mesh_hdir / np.linalg.norm(mesh_hdir)

        if mesh_vdir is None:
            mesh_vdir = np.cross(mesh_hdir, self.det_n)
            mesh_vdir = -mesh_vdir / np.linalg.norm(mesh_vdir)
        else:
            mesh_vdir = mesh_vdir / np.linalg.norm(mesh_vdir)

        # Raise exception if mesh is AFTER the field grid
        if np.linalg.norm(location - self.source) > np.linalg.norm(self.source):
            raise ValueError(
                f"The specified mesh location, {location},"
                "is not between the source and the origin."
            )

        mesh_entry = {
            "location": location,
            "wire_radius": wire_radius,
            "radius": radius,
            "width": width,
            "height": height,
            "nwires": nwires,
            "mesh_hdir": mesh_hdir,
            "mesh_vdir": mesh_vdir,
        }

        self.mesh_list.append(mesh_entry)

    def _apply_wire_mesh(
        self,
        location=None,
        wire_radius=None,
        radius=None,
        width=None,
        height=None,
        nwires=None,
        mesh_hdir=None,
        mesh_vdir=None,
    ):
        """
        Apply wire meshes that were added to self.mesh_list
        """
        x = self._coast_to_plane(location, mesh_hdir, mesh_vdir)

        # Particle positions in 2D on the mesh plane
        xloc = np.dot(x - location, mesh_hdir)
        yloc = np.dot(x - location, mesh_vdir)

        # Create an array in which True indicates that a particle has hit a wire
        # and False indicates that it has not
        hit = np.zeros(self.nparticles, dtype=bool)

        # Mark particles that overlap vertical or horizontal position with a wire
        h_centers = np.linspace(-width / 2, width / 2, num=nwires[0])
        for c in h_centers:
            hit |= np.isclose(xloc, c, atol=wire_radius)

        v_centers = np.linspace(-height / 2, height / 2, num=nwires[1])
        for c in v_centers:
            hit |= np.isclose(yloc, c, atol=wire_radius)

        # Put back any particles that are outside the mesh boundaries
        # First handle the case where the mesh is rectangular
        if radius is None:
            # Replace particles outside the x-boundary
            hit[
                np.logical_or(
                    xloc > np.max(h_centers) + wire_radius,
                    xloc < np.min(h_centers) - wire_radius,
                )
            ] = False
            # Replace particles outside the y-boundary
            hit[
                np.logical_or(
                    yloc > np.max(v_centers) + wire_radius,
                    yloc < np.min(v_centers) - wire_radius,
                )
            ] = False
        # Handle the case where the mesh is circular
        else:
            loc_rad = np.sqrt(xloc ** 2 + yloc ** 2)
            hit[loc_rad > radius] = False

            # In the case of a circular mesh, also create a round wire along the
            # outside edge
            hit[np.isclose(loc_rad, radius, atol=wire_radius)] = True

        # Identify the particles that have hit something, then remove them from
        # all of the arrays
        keep_these_particles = ~hit
        number_kept_particles = keep_these_particles.sum()
        nremoved = self.nparticles - number_kept_particles

        if self.nparticles - nremoved <= 0:
            raise ValueError(
                "The specified mesh is blocking all of the particles. "
                f"The wire diameter ({2*wire_radius}) may be too large."
            )

        self.x = self.x[keep_these_particles, :]
        self.v = self.v[keep_these_particles, :]
        self.theta = self.theta[
            keep_these_particles
        ]  # Important to apply here to get correct grid_ind
        self.nparticles = number_kept_particles

    # *************************************************************************
    # Particle creation methods
    # *************************************************************************

    def _angles_monte_carlo(self):
        """
        Generates angles for each particle randomly such that the flux
        per solid angle is uniform.
        """
        # Create a probability vector for the theta distribution
        # Theta must follow a sine distribution in order for the particle
        # flux per solid angle to be uniform.
        arg = np.linspace(0, self.max_theta, num=int(1e5))
        prob = np.sin(arg)
        prob *= 1 / np.sum(prob)

        # Randomly choose theta's weighted with the sine probabilities
        theta = np.random.choice(arg, size=self.nparticles, replace=True, p=prob)

        # Also generate a uniform phi distribution
        phi = np.random.uniform(high=2 * np.pi, size=self.nparticles)

        return theta, phi

    def _angles_uniform(self):
        """
        Generates angles for each particle such that their velocities are
        uniformly distributed on a grid in theta and phi. This method
        requires that `nparticles` be a perfect square. If it is not,
        `nparticles` will be set as the largest perfect square smaller
        than the provided `nparticles`.
        """
        # Calculate the approximate square root
        n_per = np.floor(np.sqrt(self.nparticles)).astype(np.int32)

        # Set new nparticles to be a perfect square
        self.nparticles = n_per ** 2

        # Create an imaginary grid positioned 1 unit from the source
        # and spanning max_theta at the corners
        extent = np.sin(self.max_theta) / np.sqrt(2)
        arr = np.linspace(-extent, extent, num=n_per)
        harr, varr = np.meshgrid(arr, arr, indexing="ij")

        # calculate the angles from the source for each point in
        # the grid.
        theta = np.arctan(np.sqrt(harr ** 2 + varr ** 2))
        phi = np.arctan2(varr, harr)

        return theta.flatten(), phi.flatten()

    @particles.particle_input
    def create_particles(
        self,
        nparticles,
        particle_energy,
        max_theta=None,
        particle: Particle = Particle("p+"),
        distribution="monte-carlo",
    ):
        r"""
        Generates the angular distributions about the Z-axis, then
        rotates those distributions to align with the source-to-detector axis.

        By default, particles are generated over almost the entire pi/2. However,
        if the detector is far from the source, many of these particles will
        never be observed. The max_theta keyword allows these extraneous
        particles to be neglected to focus computational resources on the
        particles who will actually hit the detector.

        nparticles : integer
            The number of particles to include in the simulation. The default
            is 1e5.

        particle_energy : `~astropy.units.Quantity`
            The energy of the particle, in units convertible to eV.
            All particles are given the same energy.

        max_theta : `~astropy.units.Quantity`, optional
            The largest velocity vector angle (measured from the
            source-to-detector axis) for which particles should be generated.
            Decreasing this angle can eliminate particles that would never
            reach the detector region of interest. If no value is given, a
            guess will be made based on the size of the grid.
            Units must be convertible to radians.

        particle : ~plasmapy.particles.Particle or string representation of same, optional
            Representation of the particle species as either a `Particle` object
            or a string representation. The default particle is protons.

        distribution: str
            A keyword which determines how particles will be distributed
            in velocity space. Options are:

                - 'monte-carlo': velocities will be chosen randomly,
                    such that the flux per solid angle is uniform.

                - 'uniform': velocities will be distrbuted such that,
                   left unperturbed,they will form a uniform pattern
                   on the detection plane. This method
                   requires that `nparticles` be a perfect square. If it is not,
                   `nparticles` will be set as the largest perfect square smaller
                   than the provided `nparticles`.

            Simulations run in the `uniform` mode will imprint a grid pattern
            on the image, but will well-sample the field grid with a
            smaller number of particles. The default is `monte-carlo`


        """
        self._log("Creating Particles")

        # Load inputs
        self.nparticles = int(nparticles)
        self.particle_energy = particle_energy.to(u.eV).value
        self.q = particle.charge.to(u.C).value
        self.m = particle.mass.to(u.kg).value

        # If max_theta is not specified, make a guess based on the grid size
        if max_theta is None:
            self.max_theta = np.clip(
                1.5 * self.max_theta_hit_grid, 0.01, 0.99 * np.pi / 2
            )
        else:
            self.max_theta = max_theta.to(u.rad).value

        # Calculate the velocity corresponding to the particle energy
        ER = self.particle_energy * 1.6e-19 / (self.m * self._c ** 2)
        v0 = self._c * np.sqrt(1 - 1 / (ER + 1) ** 2)

        if distribution == "monte-carlo":
            theta, phi = self._angles_monte_carlo()
        elif distribution == "uniform":
            theta, phi = self._angles_uniform()

        # Temporarily save theta to later determine which particles
        # should be tracked
        self.theta = theta

        # Construct the velocity distribution around the z-axis
        self.v = np.zeros([self.nparticles, 3])
        self.v[:, 0] = v0 * np.sin(theta) * np.cos(phi)
        self.v[:, 1] = v0 * np.sin(theta) * np.sin(phi)
        self.v[:, 2] = v0 * np.cos(theta)

        # Calculate the rotation matrix that rotates the z-axis
        # onto the source-detector axis
        a = np.array([0, 0, 1])
        b = self.detector - self.source
        rot = rot_a_to_b(a, b)

        # Apply rotation matrix to calculated velocity distribution
        self.v = np.matmul(self.v, rot)

        # Place particles at the source
        self.x = np.tile(self.source, (self.nparticles, 1))

    @particles.particle_input
    def load_particles(
        self, x, v, particle: Particle = Particle("p+"),
    ):
        r"""
        Load arrays of particle positions and velocities

        x : `~astropy.units.Quantity`, shape (N,3)
            Positions for N particles

        v: `~astropy.units.Quantity`, shape (N,3)
            Velocities for N particles

        particle : ~plasmapy.particles.Particle or string representation of same, optional
            Representation of the particle species as either a `Particle` object
            or a string representation. The default particle is protons.

        distribution: str
            A keyword which determines how particles will be distributed
            in velocity space. Options are:

                - 'monte-carlo': velocities will be chosen randomly,
                    such that the flux per solid angle is uniform.

                - 'uniform': velocities will be distrbuted such that,
                   left unpreturbed,they will form a uniform pattern
                   on the detection plane.

            Simulations run in the `uniform` mode will imprint a grid pattern
            on the image, but will well-sample the field grid with a
            smaller number of particles. The default is `monte-carlo`


        """
        self.q = particle.charge.to(u.C).value
        self.m = particle.mass.to(u.kg).value

        if x.shape[0] != v.shape[0]:
            raise ValueError(
                "Provided x and v arrays have inconsistent numbers "
                " of particles "
                f"({x.shape[0]} and {v.shape[0]} respectively)."
            )
        else:
            self.nparticles = x.shape[0]

        self.x = x.to(u.m).value
        self.v = v.to(u.m / u.s).value

        self.theta = np.arccos(
            np.inner(self.v, self.src_n) / np.linalg.norm(self.v, axis=-1)
        )

        n_wrong_way = np.sum(np.where(self.theta > np.pi / 2, 1, 0))
        if n_wrong_way > 1:
            warnings.warn(
                f"{100*n_wrong_way/self.nparticles:.2f}% of particles "
                "initialized are heading away from the grid. Check the orientation "
                " of the provided velocity vectors.",
                RuntimeWarning,
            )

    # *************************************************************************
    # Run/push loop methods
    # *************************************************************************

    def _adaptive_dt(self, Ex, Ey, Ez, Bx, By, Bz):
        r"""
        Calculate the appropriate dt based on a number of considerations
        including the local grid resolution (ds) and the gyroperiod of the
        particles in the current fields.
        """
        # If dt was explicitly set, skip the rest of this function
        if self.dt.size == 1:
            return self.dt

        # Compute the timestep indicated by the grid resolution
        ds = self.grid.grid_resolution.to(u.m).value
        gridstep = 0.5 * (np.min(ds) / self.vmax)

        # If not, compute a number of possible timesteps
        # Compute the cyclotron gyroperiod
        Bmag = np.max(np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)).to(u.T).value

        # Compute the gyroperiod
        if Bmag == 0:
            gyroperiod = np.inf
        else:
            gyroperiod = 2 * np.pi * self.m / (self.q * np.max(Bmag))

        # TODO: introduce a minimum timestep based on electric fields too!

        # Create an array of all the possible time steps we computed
        candidates = np.array([gyroperiod / 12, gridstep])

        # Enforce limits on dt
        candidates = np.clip(candidates, self.dt[0], self.dt[1])

        # dt is the min of the remaining candidates
        return np.min(candidates)

    def _coast_to_grid(self):
        r"""
        Coasts all particles to the timestep when the first particle should
        be entering the grid. Doing in this in one step (rather than pushing
        the particles through zero fields) saves computation time.
        """
        # Distance from the source to the nearest gridpoint
        dist = np.min(np.linalg.norm(self.grid_arr - self.source, axis=3))

        # Find the particle with the highest speed towards the grid
        vmax = np.max(np.dot(self.v, self.src_n))

        # Time for fastest possible particle to reach the grid.
        t = dist / vmax

        # Coast the particles to the advanced position
        self.x = self.x + self.v * t

    def _coast_to_plane(self, center, hdir, vdir, x=None):
        """
        Calculates the positions where the current trajectories of each
        particle impact a plane, described by the plane's center and
        horizontal and vertical unit vectors.

        Returns an [nparticles, 3] array of the particle positions in the plane

        By default this function does not alter self.x. The optional keyword
        x can be used to pass in an output array that will used to hold
        the positions in the plane. This can be used to directly update self.x
        as follows:

        self._coast_to_plane(self.detector, self.det_hdir, self.det_vdir, x = self.x)

        """

        normal = np.cross(hdir, vdir)

        # Calculate the time required to evolve each particle into the
        # plane
        t = np.inner(center[np.newaxis, :] - self.x, normal) / np.inner(self.v, normal)

        # Calculate particle positions in the plane
        if x is None:
            # If no output array is provided, preallocate
            x = np.empty_like(self.x)

        x[...] = self.x + self.v * t[:, np.newaxis]

        # Check that all points are now in the plane
        # (Eq. of a plane is nhat*x + d = 0)
        plane_eq = np.dot(x - center, normal)
        assert np.allclose(plane_eq, 0, atol=1e-6)

        return x

    def _remove_deflected_particles(self):
        r"""
        Removes any particles that have been deflected away from the detector
        plane (eg. those that will never hit the grid)
        """
        dist_remaining = np.dot(self.x, self.det_n) + np.linalg.norm(self.detector)

        v_towards_det = np.dot(self.v, -self.det_n)

        # If particles have not yet reached the detector plane and are moving
        # away from it, they will never reach the detector.
        # So, we can remove them from the arrays

        # Find the indices of all particles that we should keep:
        # i.e. those still moving towards the detector.
        ind = np.logical_not((v_towards_det < 0) & (dist_remaining > 0)).nonzero()[0]

        # Drop the other particles
        self.x = self.x[ind, :]
        self.v = self.v[ind, :]
        self.v_init = self.v_init[ind, :]
        self.nparticles_grid = self.x.shape[0]

        # Store the number of particles deflected
        self.fract_deflected = (self.nparticles - ind.size) / self.nparticles

        # Warn the user if a large number of particles are being deflected
        if self.fract_deflected > 0.05:
            warnings.warn(
                f"{100*self.fract_deflected:.1f}% particles have been "
                "deflected away from the detector plane. The fields "
                "provided may be too high to successfully radiograph "
                "with this particle energy.",
                RuntimeWarning,
            )

    def _push(self):
        r"""
        Advance particles using an implementation of the time-centered
        Boris algorithm
        """
        # Get a list of positions (input for interpolator)
        pos = self.x[self.grid_ind, :] * u.m

        # Update the list of particles on and off the grid
        self.on_grid = self.grid.on_grid(pos)
        # entered_grid is zero at the end if a particle has never
        # entered the grid
        self.entered_grid += self.on_grid

        # Estimate the E and B fields for each particle
        # Note that this interpolation step is BY FAR the slowest part of the push
        # loop. Any speed improvements will have to come from here.
        if self.field_weighting == "volume averaged":
            Ex, Ey, Ez, Bx, By, Bz = self.grid.volume_averaged_interpolator(
                pos, "E_x", "E_y", "E_z", "B_x", "B_y", "B_z", persistent=True,
            )
        elif self.field_weighting == "nearest neighbor":
            Ex, Ey, Ez, Bx, By, Bz = self.grid.nearest_neighbor_interpolator(
                pos, "E_x", "E_y", "E_z", "B_x", "B_y", "B_z", persistent=True,
            )

        # Create arrays of E and B as required by push algorithm
        E = np.array(
            [Ex.to(u.V / u.m).value, Ey.to(u.V / u.m).value, Ez.to(u.V / u.m).value]
        )
        E = np.moveaxis(E, 0, -1)
        B = np.array([Bx.to(u.T).value, By.to(u.T).value, Bz.to(u.T).value])
        B = np.moveaxis(B, 0, -1)

        # Calculate the adaptive timestep from the fields currently experienced
        # by the particles
        # If user sets dt explicitly, that's handled in _adpative_dt
        dt = self._adaptive_dt(Ex, Ey, Ez, Bx, By, Bz)

        # TODO: Test v/c and implement relativistic Boris push when required
        # vc = np.max(v)/_c

        x = self.x[self.grid_ind, :]
        v = self.v[self.grid_ind, :]
        boris_push(x, v, B, E, self.q, self.m, dt)
        self.x[self.grid_ind, :] = x
        self.v[self.grid_ind, :] = v

    def _stop_condition(self):
        r"""
        The stop condition is that most of the particles have entered the grid
        and almost all have now left it.
        """
        # Count the number of particles who have entered, which is the
        # number of non-zero entries in entered_grid
        self.num_entered = np.nonzero(self.entered_grid)[0].size

        # How many of the particles have entered the grid
        self.fract_entered = np.sum(self.num_entered) / self.nparticles_grid

        # Of the particles that have entered the grid, how many are currently
        # on the grid?
        # if/else avoids dividing by zero
        if np.sum(self.num_entered) > 0:
            still_on = np.sum(self.on_grid) / np.sum(self.num_entered)
        else:
            still_on = 0.0

        if self.fract_entered > 0.1 and still_on < 0.001:
            # Warn user if < 10% of the particles ended up on the grid
            if self.num_entered < 0.1 * self.nparticles:
                warnings.warn(
                    f"Only {100*self.num_entered/self.nparticles:.2f}% of "
                    "particles entered the field grid: consider "
                    "decreasing the max_theta to increase this "
                    "number.",
                    RuntimeWarning,
                )

            return True
        else:
            return False

    def run(
        self, dt=None, field_weighting="volume averaged",
    ):
        r"""
        Runs a particle-tracing simulation.
        Timesteps are adaptively calculated based on the
        local grid resolution of the particles and the electric and magnetic
        fields they are experiencing. After all particles
        have left the grid, they are advanced to the
        detector plane where they can be used to construct a synthetic
        diagnostic image.

        Parameters
        ----------

        dt : `~astropy.units.Quantity`, optional
            An explicitly set timestep in units convertable to seconds.
            Setting this optional keyword overrules the adaptive time step
            capability and forces the use of this timestep throughout. If a tuple
            of timesteps is provided, the adaptive timstep will be clamped
            between the first and second values.

        field_weighting : str
            String that selects the field weighting algorithm used to determine
            what fields are felt by the particles. Options are:

            * 'nearest neighbor': Particles are assigned the fields on
                the grid vertex closest to them.

            * 'volume averaged' : The fields experienced by a particle are a
                volume-average of the eight grid points surrounding them.

            The default is 'volume averaged'.

        Returns
        -------
        None.

        """

        # Load and validate inputs
        field_weightings = ["volume averaged", "nearest neighbor"]
        if field_weighting in field_weightings:
            self.field_weighting = field_weighting
        else:
            raise ValueError(
                f"{field_weighting} is not a valid option for ",
                "field_weighting. Valid choices are",
                f"{field_weightings}",
            )

        if dt is None:
            # Set dt as an infinite range by default (auto dt with no restrictions)
            self.dt = np.array([0.0, np.inf]) * u.s
        else:
            self.dt = dt
        self.dt = (self.dt).to(u.s).value

        # Check to make sure particles have already been generated
        if not hasattr(self, "x"):
            raise ValueError(
                "Either the create_particles or load_particles method must be "
                "called before running the particle tracing algorithm."
            )

        # If meshes have been added, apply them now
        for mesh in self.mesh_list:
            self._apply_wire_mesh(**mesh)

        # Store a copy of the initial velocity distribution in memory
        # This will be used later to calculate the maximum deflection
        self.v_init = np.copy(self.v)

        # Calculate the maximum velocity
        # Used for determining the grid crossing maximum timestep
        self.vmax = np.max(np.linalg.norm(self.v, axis=-1))

        # Determine which particles should be tracked
        # This array holds the indices of all particles that WILL hit the grid
        # Only these particles will actually be pushed through the fields
        self.grid_ind = np.where(self.theta < self.max_theta_hit_grid)[0]
        self.nparticles_grid = len(self.grid_ind)
        self.fract_tracked = self.nparticles_grid / self.nparticles

        # Create flags for tracking when particles during the simulation
        # on_grid -> zero if the particle is off grid, 1
        self.on_grid = np.zeros([self.nparticles_grid])
        # Entered grid -> non-zero if particle EVER entered the grid
        self.entered_grid = np.zeros([self.nparticles_grid])

        # Generate a null distribution of points (the result in the absence of
        # any fields) for statistical comparison
        self.x0 = self._coast_to_plane(self.detector, self.det_hdir, self.det_vdir)

        # Advance the particles to the near the start of the grid
        self._coast_to_grid()

        # TODO from here...
        # Initialize a "progress bar" (really more of a meter)
        # Setting sys.stdout lets this play nicely with regular print()
        pbar = tqdm(
            initial=0,
            total=self.nparticles_grid + 1,
            disable=not self.verbose,
            desc="Particles on grid",
            unit="particles",
            bar_format="{l_bar}{bar}{n:.1e}/{total:.1e} {unit}",
            file=sys.stdout,
        )

        # Push the particles until the stop condition is satisfied
        # (no more particles on the simulation grid)
        while not self._stop_condition():
            n_on_grid = np.sum(self.on_grid)
            pbar.n = n_on_grid
            pbar.last_print_n = n_on_grid
            pbar.update()

            self._push()
        pbar.close()

        # TODO to here, effectively
        # Remove particles that will never reach the detector
        self._remove_deflected_particles()

        # Advance the particles to the image plane
        self._coast_to_plane(self.detector, self.det_hdir, self.det_vdir, x=self.x)

        # Log a summary of the run

        self._log("Run completed")

        self._log("Fraction of particles tracked: " f"{self.fract_tracked*100:.1f}%")

        self._log(
            "Fraction of tracked particles that entered the grid: "
            f"{self.fract_entered*100:.1f}%"
        )

        self._log(
            "Fraction of tracked particles deflected away from the "
            "detector plane: "
            f"{self.fract_deflected*100}%"
        )

    @property
    def max_deflection(self):
        """
        The maximum deflection experienced by one of the particles, determined
        by comparing their initial and final velocitiy vectors.

        This value can be used to determine the charged particle radiography regime
        using the dimensionless number defined by Kugland et al. 2012

        Returns
        -------
        max_deflection : float
            The maximum deflection in radians

        """
        # Normalize the initial and final velocities
        v_norm = self.v / np.linalg.norm(self.v, axis=1, keepdims=True)
        v_init_norm = self.v_init / np.linalg.norm(self.v_init, axis=1, keepdims=True)
        # Compute the dot product
        proj = np.sum(v_norm * v_init_norm, axis=1)
        # In case of numerical errors, make sure the output is within the domain of
        # arccos
        proj = np.where(proj > 1, 1, proj)
        max_deflection = np.max(np.arccos(proj))

        return max_deflection * u.rad

    # *************************************************************************
    # Synthetic diagnostic methods (creating output)
    # *************************************************************************

    def synthetic_radiograph(
        self, size=None, bins=[200, 200], ignore_grid=False, optical_density=False
    ):
        r"""
        Calculate a "synthetic radiograph" (particle count histogram in the
        image plane).

        Parameters
        ----------
        size : `~astropy.units.Quantity`, shape (2,2)
            The size of the detector array, specified as the minimum
            and maximum values included in both the horizontal and vertical
            directions in the detector plane coordinates. Shape is
            [[hmin,hmax], [vmin, vmax]]. Units must be convertable to meters.

        bins : array of integers, shape (2)
            The number of bins in each direction in the format [hbins, vbins].
            The default is [200,200].

        ignore_grid: bool
            If True, returns the intensity in the image plane in the absence
            of simulated fields.

        optical_density: bool
            If True, return the optical density rather than the intensity

            .. math::
                OD = -log_{10}(Intensity/I_0)

            where I_O is the intensity on the detector plane in the absence of
            simulated fields. Default is False.

        Returns
        -------
        hax : `~astropy.units.Quantity` array shape (hbins,)
            The horizontal axis of the synthetic radiograph in meters.

        vax : `~astropy.units.Quantity` array shape (vbins, )
            The vertical axis of the synthetic radiograph in meters.

        intensity : ndarray, shape (hbins, vbins)
            The number of particles counted in each bin of the histogram.
        """

        # Note that, at the end of the simulation, all particles were moved
        # into the image plane.

        # If ignore_grid is True, use the predicted positions in the absence of
        # simulated fields
        if ignore_grid:
            x = self.x0
        else:
            x = self.x

        # Determine locations of points in the detector plane using unit
        # vectors
        xloc = np.dot(x - self.detector, self.det_hdir)
        yloc = np.dot(x - self.detector, self.det_vdir)

        if size is None:
            # If a detector size is not given, choose lengths based on the
            # dimensions of the grid
            w = self.mag * np.max(
                [
                    np.max(np.abs(self.grid.pts0.to(u.m).value)),
                    np.max(np.abs(self.grid.pts1.to(u.m).value)),
                    np.max(np.abs(self.grid.pts2.to(u.m).value)),
                ]
            )

            # The factor of 5 here is somewhat arbitrary: we just want a
            # region a few times bigger than the image of the grid on the
            # detector, since particles could be deflected out
            size = 5 * np.array([[-w, w], [-w, w]]) * u.m

        # Generate the histogram
        intensity, h, v = np.histogram2d(
            xloc, yloc, range=size.to(u.m).value, bins=bins
        )

        # h, v are the bin edges: compute the centers to produce arrays
        # of the right length (then trim off the extra point)
        h = ((h + np.roll(h, -1)) / 2)[0:-1]
        v = ((v + np.roll(v, -1)) / 2)[0:-1]

        # Throw a warning if < 50% of the particles are included on the
        # histogram
        percentage = np.sum(intensity) / self.nparticles
        if percentage < 0.5:
            warnings.warn(
                f"Only {percentage:.2%} of the particles are shown "
                "on this synthetic radiograph. Consider increasing "
                "the size to include more.",
                RuntimeWarning,
            )

        if optical_density:
            # Generate the null radiograph
            x, y, I0 = self.synthetic_radiograph(size=size, bins=bins, ignore_grid=True)

            # Calculate I0 as the mean of the non-zero values in the null
            # histogram. Zeros are just outside of the illuminate area.
            I0 = np.mean(I0[I0 != 0])

            # Overwrite any zeros in intensity to avoid log10(0)
            intensity[intensity == 0] = 1

            # Calculate the optical_density
            intensity = -np.log10(intensity / I0)

        return h * u.m, v * u.m, intensity
