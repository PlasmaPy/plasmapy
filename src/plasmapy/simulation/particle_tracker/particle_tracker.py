"""
Module containing the definition for the general particle tracker.
"""

__all__ = [
    "ParticleTracker",
]

import collections
import sys
import typing
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Literal

import astropy.constants as const
import astropy.units as u
import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from tqdm import tqdm

from plasmapy.formulary.collisions.misc import Bethe_stopping_lite
from plasmapy.formulary.lengths import Debye_length
from plasmapy.particles.atomic import reduced_mass, stopping_power
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.particle_class import Particle, ParticleLike
from plasmapy.particles.particle_collections import ParticleList, ParticleListLike
from plasmapy.plasma.grids import AbstractGrid
from plasmapy.plasma.plasma_base import BasePlasma
from plasmapy.simulation.particle_integrators import (
    AbstractIntegrator,
    RelativisticBorisIntegrator,
)
from plasmapy.simulation.particle_tracker.save_routines import (
    AbstractSaveRoutine,
    DoNotSaveSaveRoutine,
)
from plasmapy.simulation.particle_tracker.termination_conditions import (
    AbstractTerminationCondition,
)
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning, RelativityWarning

_c = const.c
_m_p = const.m_p


class ParticleTracker:
    r"""A particle tracker for particles in electric and magnetic fields without inter-particle interactions.

    Particles are instantiated and pushed through a grid of provided E and
    B fields using the Boris push algorithm. These fields are specified as
    part of a grid which are then interpolated to determine the local field
    acting on each particle.

    The time step used in the push routine can be specified, or an adaptive
    time step will be determined based off the gyroradius of the particle.
    Some save routines involve time stamping the location and velocities of
    particles at fixed intervals. In order for this data to be coherent, it
    is required that all particles follow the same time step. This is
    referred to as a synchronized time step. If no time step is specified
    and the provided save routine does not require a synchronized time step,
    then an adaptive time step is calculated independently for each particle.

    The simulation will push particles through the provided fields until a
    condition is met. This termination condition is provided as an instance
    of the `~plasmapy.simulation.particle_tracker.termination_conditions.AbstractTerminationCondition`
    class as arguments to the simulation constructor. The results of a simulation
    can be exported by specifying an instance of the `~plasmapy.simulation.particle_tracker.save_routines.AbstractSaveRoutine`
    class to the ``run`` method.

    Parameters
    ----------
    grids : An instance of `~plasmapy.plasma.grids.AbstractGrid`
        A Grid object or list of grid objects containing the required quantities.
        The list of required quantities varies depending on other keywords.

    termination_condition : `~plasmapy.simulation.particle_tracker.termination_conditions.AbstractTerminationCondition`
        An subclass of `~plasmapy.simulation.particle_tracker.termination_conditions.AbstractTerminationCondition` which determines when the simulation has finished.
        See `~plasmapy.simulation.particle_tracker.termination_conditions.AbstractTerminationCondition` for more details.

    save_routine : `~plasmapy.simulation.particle_tracker.save_routines.AbstractSaveRoutine`, optional
        An subclass of `~plasmapy.simulation.particle_tracker.save_routines.AbstractSaveRoutine` which determines which
        time steps of the simulation to save. The default is `~plasmapy.simulation.particle_tracker.save_routines.DoNotSaveSaveRoutine`.
        See `~plasmapy.simulation.particle_tracker.save_routines.AbstractSaveRoutine` for more details.

    particle_integrator : `~plasmapy.simulation.particle_integrators.AbstractIntegrator`, optional
        An subclass of `~plasmapy.simulation.particle_integrators.AbstractIntegrator` that is responsible for implementing the push behavior
        of the simulation when provided the electric and magnetic fields. The default value is set to `~plasmapy.simulation.particle_integrators.RelativisticBorisIntegrator`.
        See `~plasmapy.simulation.particle_integrators.AbstractIntegrator` for more information on how to implement custom push routines.

    dt : `~astropy.units.Quantity`, optional
        An explicitly set time step in units convertible to seconds.
        Setting this optional keyword overrules the adaptive time step
        capability and forces the use of this time step throughout.

    dt_range : tuple of shape (2,) of `~astropy.units.Quantity`, optional
        If specified, the calculated adaptive time step will be clamped
        between the first and second values.

    field_weighting : str
        String that selects the field weighting algorithm used to determine
        what fields are felt by the particles. Options are:

        * 'nearest neighbor': Particles are assigned the fields on
            the grid vertex closest to them.
        * 'volume averaged' : The fields experienced by a particle are a
            volume-average of the eight grid points surrounding them.

        The default is 'volume averaged'.

    req_quantities : `list` of `str`, default : `None`
        A list of quantity keys required to be specified on the Grid object.
        The base particle pushing simulation requires the quantities
        [E_x, E_y, E_z, B_x, B_y, B_z]. This keyword is for specifying
        quantities in addition to these six. If any required
        quantities are missing, those quantities will be assumed to be zero
        everywhere. A warning will be raised if any of the additional
        required quantities are missing and are set to zero.

    seed : int, optional, default : `None`
        The seed used to create an instance of the `~numpy.random.Generator`
        object. If no value is provided then fresh entropy will be pulled from
        the OS. The resulting `~numpy.random.Generator` is used to sample
        distributions for stopping and scattering functionalities.

    verbose : bool, optional
        If true, updates on the status of the program will be printed
        into the standard output while running. The default is `True`.

    Warns
    -----
    `~plasmapy.utils.exceptions.RelativityWarning`
        The provided integrator does not account for relativistic corrections
        and particles have reached a non-negligible fraction of the speed of
        light.

    Notes
    -----
    We adopt the convention of ``NaN`` values to represent various states of a particle.

    If the particle's position and velocity are not ``NaN``, the particle is being tracked and evolved.
    If the particle's position is not ``NaN``, but the velocity is ``NaN``, the particle has been stopped
    (i.e. it is still in the simulation but is no longer evolved.)
    If both the particle's position and velocity are set to ``NaN``, then the particle has been removed from the simulation.

    Example
    -----
    >>> from plasmapy.particles import Particle
    >>> from plasmapy.plasma.grids import CartesianGrid
    >>> from plasmapy.simulation.particle_tracker.particle_tracker import (
    ...     ParticleTracker,
    ... )
    >>> from plasmapy.simulation.particle_tracker.save_routines import (
    ...     IntervalSaveRoutine,
    ... )
    >>> from plasmapy.simulation.particle_tracker.termination_conditions import (
    ...     TimeElapsedTerminationCondition,
    ... )
    >>> import astropy.units as u
    >>> import numpy as np
    >>> example_particle = Particle("p+")
    >>> grid = CartesianGrid(-1e6 * u.m, 1e6 * u.m, num=2)
    >>> grid_shape = (2, 2, 2)
    >>> Bz = np.full(grid_shape, 1) * u.T
    >>> grid.add_quantities(B_z=Bz)
    >>> x = [[0, 0, 0]] * u.m
    >>> v = [[1, 0, 0]] * u.m / u.s
    >>> termination_condition = TimeElapsedTerminationCondition(6.28 * u.s)
    >>> save_routine = IntervalSaveRoutine(6.28 * u.s / 10)
    >>> simulation = ParticleTracker(
    ...     grid,
    ...     termination_condition,
    ...     dt=1e-2 * u.s,
    ...     save_routine=save_routine,
    ...     field_weighting="nearest neighbor",
    ... )
    >>> simulation.load_particles(x, v, example_particle)
    >>> simulation.run()
    >>> print(
    ...     simulation.save_routine.results["time"][-1],
    ...     simulation.save_routine.results["x"][-1],
    ... )
    6.29999999999991 s [[-1.73302335e-08  1.31539878e-05  0.00000000e+00]] m
    """

    # Some quantities are necessary for the particle tracker to function
    # regardless of other configurations
    _REQUIRED_QUANTITIES: typing.ClassVar[set[str]] = {
        "E_x",
        "E_y",
        "E_z",
        "B_x",
        "B_y",
        "B_z",
    }

    def __init__(
        self,
        grids: AbstractGrid | Iterable[AbstractGrid],
        termination_condition: AbstractTerminationCondition | None = None,
        save_routine: AbstractSaveRoutine | None = None,
        particle_integrator: type[AbstractIntegrator] | None = None,
        dt=None,
        dt_range=None,
        field_weighting: str = "volume averaged",
        req_quantities=None,
        seed: int | None = None,
        verbose=True,
    ) -> None:
        # Instantiate the integrator object for use in the _push() method
        self._integrator: AbstractIntegrator = (
            RelativisticBorisIntegrator()
            if not particle_integrator
            else particle_integrator()
        )

        # self.grid is the grid object
        self.grids = self._grid_factory(grids)

        # Errors for unsupported grid types are raised in the validate constructor inputs method

        # Instantiate the "do not save" save routine if no save routine was specified
        if save_routine is None:
            save_routine = DoNotSaveSaveRoutine()

        # Validate inputs to the run function
        self._validate_constructor_inputs(
            grids, termination_condition, save_routine, field_weighting
        )

        self._set_time_step_attributes(dt, termination_condition, save_routine)

        if dt_range is not None and not self._is_adaptive_time_step:
            raise ValueError(
                "Specifying a time step range is only possible for an adaptive time step."
            )

        self._rng = np.random.default_rng(seed)

        self.verbose = verbose

        # This flag records whether the simulation has been run
        self._has_run = False

        # Should the tracker update particle energies after every time step to
        # reflect stopping?
        self._do_stopping = False
        self._do_scattering = False
        self._raised_relativity_warning = False

        # Raise a ValueError if a synchronized dt is required by termination condition or save routine but one is
        # not given. This is only the case if an array with differing entries is specified for dt
        if self._require_synchronized_time and not self._is_synchronized_time_step:
            raise ValueError(
                "Please specify a synchronized time step to use the simulation with this configuration!"
            )

        # Ensure that the grids have defined the necessary quantities or at least zeroed
        self._required_quantities = self._REQUIRED_QUANTITIES.copy()
        self._preprocess_grids(req_quantities)

        # self.grid_arr is the grid positions in si units. This is created here
        # so that it isn't continuously called later
        self.grids_arr = [grid.grid.to(u.m).value for grid in self.grids]

        self.dt = dt.to(u.s).value if dt is not None else None

        dt_range = [0, np.inf] * u.s if dt_range is None else dt_range
        self.dt_range = dt_range.to(u.s).value

        # Update the `tracker` attribute so that the stop condition & save routine can be used
        termination_condition.tracker = self

        save_routine.tracker = self

        self.termination_condition = termination_condition
        self.save_routine = save_routine

    @staticmethod
    def _grid_factory(grids):
        """
        Take the user provided argument for grids and convert it into the proper type.
        """

        if isinstance(grids, AbstractGrid):
            return [
                grids,
            ]
        elif isinstance(grids, collections.abc.Iterable):
            return grids
        else:
            return None

    def _set_time_step_attributes(
        self, dt, termination_condition, save_routine
    ) -> None:
        """Determines whether the simulation will follow a synchronized or adaptive time step."""

        self._require_synchronized_time = (
            termination_condition.require_synchronized_dt
            or (save_routine is not None and save_routine.require_synchronized_dt)
        )

        if isinstance(dt, u.Quantity):
            if isinstance(dt.value, np.ndarray):
                # If an array is specified for the time step, a synchronized time step is implied if all
                # the entries are equal
                self._is_synchronized_time_step = bool(
                    np.all(dt.value[0] == dt.value[:])
                )
            else:
                self._is_synchronized_time_step = True

            self._is_adaptive_time_step = False
        elif dt is None:
            self._is_synchronized_time_step = self._require_synchronized_time
            self._is_adaptive_time_step = True

        if self._is_adaptive_time_step:
            # Initialize default values for time steps per gyroperiod and Courant parameter
            self.setup_adaptive_time_step()

    def setup_adaptive_time_step(
        self,
        time_steps_per_gyroperiod: int | None = 12,
        Courant_parameter: float | None = 0.5,
    ) -> None:
        """Set parameters for the adaptive time step candidates.

        Parameters
        ----------
        time_steps_per_gyroperiod : int, optional
            The minimum ratio of the particle gyroperiod to the timestep. Higher numbers
            correspond to higher temporal resolution. The default is twelve.

        Courant_parameter : float, optional
            The Courant parameter is the minimum ratio of the timestep to the grid crossing time,
            grid cell length / particle velocity. Lower Courant numbers correspond to higher temporal resolution.


        Notes
        -----
        Two candidates are calculated for the adaptive time step: a time step based on the gyroradius
        of the particle and a time step based on the resolution of the grid. The candidate associated
        with the gyroradius of the particle takes a ``time_steps_per_gyroperiod`` parameter that specifies
        how many times the orbit of a gyrating particles will be subdivided. The other candidate,
        associated with the spatial resolution of the grid object, calculates a time step using the time
        it would take the fastest particle to cross some fraction of a grid cell length. This fraction is the Courant number.
        """

        if not self._is_adaptive_time_step:
            raise ValueError(
                "The setup adaptive time step method only applies to adaptive time steps!"
            )

        self._steps_per_gyroperiod = time_steps_per_gyroperiod
        self._Courant_parameter = Courant_parameter

    def _validate_constructor_inputs(
        self, grids, termination_condition, save_routine, field_weighting: str
    ) -> None:
        """
        Ensure the specified termination condition and save routine are actually
        a termination routine class and save routine, respectively.
        """

        if isinstance(grids, BasePlasma):
            raise TypeError(
                "It appears you may be trying to access an older version of the ParticleTracker class."
                "This class has been deprecated."
                "Please revert to PlasmaPy version 2023.5.1 to use this version of ParticleTracker."
            )
        # The constructor did not recognize the provided grid object
        elif self.grids is None:
            raise TypeError("Type of argument `grids` not recognized.")

        if not isinstance(termination_condition, AbstractTerminationCondition):
            raise TypeError("Please specify a valid termination condition.")

        if not isinstance(save_routine, AbstractSaveRoutine):
            raise TypeError("Please specify a valid save routine.")

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

    def _preprocess_grids(self, additional_required_quantities) -> None:
        """Add required quantities to grid objects.

        Grids lacking the default required quantities will be filled with zeros.
        This method is not only called during instantiation of the |ParticleTracker|
        but is also called when enabling stopping.
        """

        # Convert the required quantities set to a list to appease type hinting
        quantities_list = list(self._REQUIRED_QUANTITIES)

        for grid in self.grids:
            # Require the field quantities - do not warn if they are absent
            # and are replaced with zeros
            grid.require_quantities(
                quantities_list,
                replace_with_zeros=True,
                warn_on_replace_with_zeros=False,
            )

            # "additional_required_quantities" can also refer to explicitly specified
            # fields that are covered as default field requirements
            if additional_required_quantities is not None:
                # Require the additional quantities - in this case, do warn
                # if they are set to zeros
                grid.require_quantities(
                    additional_required_quantities, replace_with_zeros=True
                )

        if additional_required_quantities is not None:
            # Add additional required quantities based off simulation configuration
            # this must be done after the above processing to properly get warnings
            self._required_quantities.update(additional_required_quantities)

        for grid in self.grids:
            for rq in self._required_quantities:
                # Check that there are no infinite values
                if not np.isfinite(grid[rq].value).all():
                    raise ValueError(
                        f"Input arrays must be finite: {rq} contains "
                        "either NaN or infinite values."
                    )

                # Check that the max values on the edges of the arrays are
                # small relative to the maximum values on that grid
                #
                # Array must be dimensionless to re-assemble it into an array
                # of max values like this
                arr = np.abs(grid[rq]).value
                edge_max = np.max(
                    np.array(
                        [
                            np.max(a)
                            for a in (
                                arr[0, :, :],
                                arr[-1, :, :],
                                arr[:, 0, :],
                                arr[:, -1, :],
                                arr[:, :, 0],
                                arr[:, :, -1],
                            )
                        ]
                    )
                )

                if edge_max > 1e-3 * np.max(arr):
                    unit = grid.recognized_quantities[rq].unit
                    warnings.warn(
                        "Quantities should go to zero at edges of grid to avoid "
                        f"non-physical effects, but a value of {edge_max:.2E} {unit} was "
                        f"found on the edge of the {rq} array. Consider applying a "
                        "envelope function to force the quantities at the edge to go to "
                        "zero.",
                        RuntimeWarning,
                    )

    @property
    def num_grids(self) -> int:
        """The number of grids specified at instantiation."""
        return len(self.grids)

    def _log(self, msg) -> None:
        if self.verbose:
            print(msg)  # noqa: T201

    @particle_input
    def load_particles(
        self,
        x,
        v,
        particle: Particle,
    ) -> None:
        r"""
        Load arrays of particle positions and velocities.

        Parameters
        ----------
        x : `~astropy.units.Quantity`, shape (N,3)
            Positions for N particles

        v : `~astropy.units.Quantity`, shape (N,3)
            Velocities for N particles

        particle : `~plasmapy.particles.particle_class.ParticleLike`
            Representation of the particle species as either a |Particle| object
            or a string representation.
        """
        # Raise an error if the run method has already been called.
        self._enforce_order()

        self.q = particle.charge.to(u.C).value
        self.m = particle.mass.to(u.kg).value
        self._particle = particle

        if x.shape[0] != v.shape[0]:
            raise ValueError(
                "Provided x and v arrays have inconsistent numbers "
                " of particles "
                f"({x.shape[0]} and {v.shape[0]} respectively)."
            )
        else:
            self.nparticles: int = x.shape[0]

        self.x = x.to(u.m).value
        self.v = v.to(u.m / u.s).value

    def _is_quantity_defined_on_one_grid(self, quantity: str) -> bool:
        r"""
        Check to ensure the provided quantity string is defined on at least one grid.

        Returns ``True`` if the quantity is defined on at least one grid and
        ``False`` if none of the grids have defined the specified quantity.
        """

        return any(quantity in grid.quantities for grid in self.grids)

    def _validate_stopping_inputs(
        self,
        method: Literal["NIST", "Bethe"],
        target: ParticleList | None,
        target_rho: u.Quantity[u.kg / u.m**2],
        NIST_target_string: Sequence[str] | None,
        I: u.Quantity[u.J] | None,  # noqa: E741
    ) -> bool:
        r"""
        Validate inputs to the `add_stopping` method. Raises errors if the
        proper keyword arguments are not provided for a given method.
        """

        if len({len(self.grids), len(target), target_rho.shape[0]}) != 1:
            raise ValueError(
                "Please provide an array of length ngrids for the materials and densities."
            )

        match method:
            case "NIST":
                if NIST_target_string is None:
                    raise ValueError(
                        "Please specify the associated NIST material string to use NIST stopping!"
                    )

                if len(NIST_target_string) != len(self.grids):
                    raise ValueError(
                        "Please specify a list of NIST material strings with length equal to the "
                        "number of grids."
                    )
            case "Bethe":
                # TODO: remove this when we implement the iterable decorator
                I_shape = I.shape if I.shape != () else (1,)

                if I_shape[0] != len(self.grids):
                    raise ValueError(
                        "Please specify a mean excitations energy array with length equal to the"
                        "number of grids."
                    )
            case _:
                raise ValueError(
                    f"Please provide one of 'NIST' or 'Bethe' for the method keyword. (Got: {method})"
                )

        return True

    def _set_target_attributes(
        self, target: ParticleList, target_rho: u.Quantity[u.kg / u.m**3]
    ):
        self._target = target

        # Cast inputs to one dimensional arrays
        self._target_rho = np.reshape(target_rho, newshape=(self.num_grids, 1))
        self._target_number_density = np.reshape(
            [(target_rho / ind_target.mass).si.value for ind_target in target],
            newshape=(self.num_grids, 1),
        )

    # TODO: add NIST PSTAR & "composition of materials used in STAR databases" references
    @particle_input(require="isotope")
    @validate_quantities
    def add_stopping(
        self,
        method: Literal["NIST", "Bethe"],
        target: ParticleListLike,
        target_rho: u.Quantity[u.kg / u.m**3],
        NIST_target_string: str | Sequence[str] | None = None,
        I: u.Quantity[u.J] | None = None,  # noqa: E741
    ):
        r"""
        Enable particle stopping using experimental stopping powers.

        Interpolation of stopping powers is conducted using data from the NIST
        PSTAR database. This information is combined with the mass density
        quantity provided in the grids to calculate the energy loss over the
        distance travelled during a timestep.

        Parameters
        ----------
        method : str
            The routine to be used to calculate the stopping power. Acceptable
            options include ``NIST``, which draws data from the NIST PSTAR database,
            and ``Bethe``, which uses the Bethe formula to calculate the stopping
            power for the material.

        target : `~plasmapy.particles.particle_collections.ParticleListLike`
            A `~plasmapy.particles.particle_collections.ParticleListLike` representation of the target material(s).
            The atomic mass of each particle must be specified (i.e. ``Al-27``).

        target_rho : `~astropy.units.Quantity`
            The density of the material in units convertible to kg/m\ :sup:`3`.

        NIST_target_string : str | Sequence[str] | None, optional
            A string representation of the material(s) in the NIST database. This is
            required if ``method`` is ``NIST``. For a complete list of materials,
            see the NIST database.

        I : `astropy.units.Quantity`
            The mean excitation energy of the material(s) in units convertible to
            Joules. This is required if ``method`` is ``Bethe``. Common values for
            the mean excitation energy can be found in the NIST materials
            composition database.

        """

        # If the target string is just a string, cast it to a list
        if isinstance(NIST_target_string, str):
            NIST_target_string = [NIST_target_string]

        # If the passed quantity does not have an array value, cast it to one
        if not isinstance(target_rho.value, np.ndarray):
            target_rho = np.asarray([target_rho.value]) * target_rho.unit

        if not isinstance(I.value, np.ndarray):
            I = np.asarray([I.value]) * I.unit  # noqa: E741

        # Check inputs for user error and raise respective exceptions/warnings if
        # necessary. Returns `True` if no errors were detected. If a "falsy" value
        # is returned then the add_stopping method will abort.
        if not self._validate_stopping_inputs(
            method, target, target_rho, NIST_target_string, I
        ):
            return

        self._set_target_attributes(target, target_rho)

        match method:
            case "NIST":
                stopping_power_interpolators = [
                    stopping_power(
                        self._particle, NIST_material, return_interpolator=True
                    )
                    for NIST_material in NIST_target_string
                ]

            case "Bethe":
                self._raised_energy_warning = False

                # These functions are used to represent that the mean excitation energy
                # and ion number density does not change over space for a given grid.
                def wrapped_Bethe_stopping(I_grid, n_i):
                    def inner_Bethe_stopping(v):
                        return Bethe_stopping_lite(
                            I_grid, n_i, v, self._particle.charge_number
                        )

                    return inner_Bethe_stopping

                # TODO: make this compatible with a `ParticleList` object of targets
                target_electron_number_density = (
                    const.N_A
                    * target[0].atomic_number
                    * target_rho
                    / (target[0].mass_number * u.kg / u.mol)
                )

                stopping_power_interpolators = [
                    wrapped_Bethe_stopping(I_grid.si.value, n_i)
                    for I_grid, n_i in zip(
                        I, target_electron_number_density.value, strict=True
                    )
                ]

        self._do_stopping = True
        self._stopping_method = method
        self._stopping_power_interpolators = stopping_power_interpolators

    # TODO: create a method to validate that these quantities have shape ngrids
    # TODO: is this signature too much?
    # TODO: find a way to create a type for the signature of the Callable parameters
    #  (preferably using only one expression)
    @particle_input(require="isotope")
    @validate_quantities
    def add_scattering(
        self,
        target: ParticleListLike,
        target_rho: u.Quantity[u.kg / u.m**3],
        scatter_routine: Callable[
            [
                u.Quantity[u.rad] | None,  # Theta
                u.Quantity[u.m / u.s],  # Speed
                ParticleLike,  # Beam
                ParticleLike,  # Target
                u.Quantity[u.K],  # Electron Temperature
                u.Quantity[1 / u.m**3],  # Electron Number Density
            ],
            u.Quantity[u.m**2 / u.sr]
            | u.Quantity[
                u.rad**2 / u.s
            ],  # Differential cross-section or mean-square scatter rate in radians^2 per second
        ],
        scatter_routine_type: Literal["differential cross-section", "mean square rate"],
    ):
        r"""
        Enable particle scattering by numeric integration of the provided
        differential cross-section.

        Parameters
        ----------
        target : `~plasmapy.particles.particle_collections.ParticleListLike`
            A `~plasmapy.particles.particle_collections.ParticleListLike` representation of the target material(s).
            The atomic mass of each particle must be specified (i.e. ``Al-27``).

        scatter_routine : `callable`
            A function implementation used to calculate the scattering distribution.
            Relevant keyword arguments are passed to the provided function, where
            the defined keywords depend on the value of ``scatter_routine_type``. If
            ``scatter_routine_type`` is set to ``differential cross-section``
            then the function receives the following keyword arguments: ``theta``,
            ``speed``, ``beam_material``, ``target_material``, ``T_s``, and ``n_s``.
            When ``scatter_routine_type`` is set to ``mean square rate``,
            the passed function receives the following keyword arguments: ``speed``,
            ``beam_material``, ``target_material``, ``T_s``, and ``n_s``.

        scatter_routine_type : `str`
            Determines the role of the function passed to ``scatter_routine``.
            Must be either ``differential cross-section`` or ``mean square rate``.
            If set to ``differential cross-section``, the provided function must return
            a `~astropy.units.Quantity` object with units convertible to square meters
            per steradian.
        """

        # For scattering calculations, a multidimensional interpolator must be
        # constructed to obtain the mean-square scattering rate as a function of
        # velocity and number density.
        if len({len(self.grids), len(target), target_rho.shape[0]}) != 1:
            raise ValueError(
                "Please provide an array of length ngrids for the materials and densities."
            )

        if not self._is_quantity_defined_on_one_grid("T_i"):
            warnings.warn(
                "The ion temperature is not defined on any of the provided grids! "
                "Particle scattering will not be calculated.",
                RuntimeWarning,
            )

            # Don't set `_do_scattering`. The push loop does not have to do scattering
            # calculations
            return

        for grid in self.grids:
            grid.require_quantities(["T_i"], replace_with_zeros=True)
            self._required_quantities.update({"T_i"})

        if scatter_routine_type not in [
            "differential cross-section",
            "mean square rate",
        ]:
            raise ValueError(
                "Please pass one of 'differential cross-section' or 'mean square rate' "
                "for the scatter routine keyword."
            )

        self._scattering_method = scatter_routine_type
        self._scattering_routine = scatter_routine

        self._set_target_attributes(target, target_rho)
        self._do_scattering = True

        # For now the integrals will be evaluated every time step
        # The below code constructs an interpolator for the scattering integral
        # # TODO: make sure these expressions are working
        # # TODO: make sure that this method of interpolation is accurate
        # #  (i.e. as opposed to something like a log spline)
        # #  include profiling & accuracy analysis on PR
        # max_electron_temperature = np.log10(
        #     np.max([grid["T_e"] for grid in self.grids])
        # )
        # max_number_density = np.log10(np.max([grid["n_e"] for grid in self.grids]))
        #
        # electron_temperature_space = np.logspace(0, max_electron_temperature) * u.K
        # number_density_space = np.logspace(0, max_number_density+1) / u.m**3
        # velocity_space = np.logspace(0, np.log(const.c.si.value)) * u.m / u.s
        #
        # mu = reduced_mass(self._particle, target)
        #
        # b_0 = (
        #     target.charge
        #     * self._particle.charge
        #     / (4 * np.pi * const.eps0 * mu * velocity_space**2)
        # )
        # lambda_D = Debye_length(electron_temperature_space, number_density_space)
        # theta_min_space = 2 * b_0 / lambda_D
        #
        # def mean_squared_integrand(theta, v, T_e, n_e):
        #     return (
        #         theta**2
        #         * differential_cross_section(
        #             theta,
        #             v,
        #             self._particle,
        #             target,
        #             T_e,
        #             n_e,
        #         )
        #     ).si.value
        #
        # evaluated_mean_square_rate = [
        #     (
        #         2
        #         * np.pi
        #         * velocity
        #         * quad(
        #             mean_squared_integrand, theta_min, np.pi, args=(velocity, T_e, n_e)
        #         )[0]
        #     ).si.value
        #     for (theta_min, velocity, T_e, n_e) in zip(
        #         theta_min_space,
        #         velocity_space,
        #         electron_temperature_space,
        #         number_density_space,
        #     )
        # ]
        #
        # self._do_scattering = True
        # self._scattering_interpolator = LinearNDInterpolator(  # sigma(n_e, v)
        #     list(zip(number_density_space.si.value, velocity_space.si.value)),
        #     evaluated_mean_square_rate,
        # )

    def run(self) -> None:
        r"""
        Runs a particle-tracing simulation.
        Time steps are adaptively calculated based on the local grid resolution
        of the particles and the electric and magnetic fields they are
        experiencing.

        Returns
        -------
        None

        """

        self._enforce_particle_creation()

        # Keep track of how many push steps have occurred for trajectory tracing
        # This number is independent of the current "time" of the simulation
        self.iteration_number = 0

        # The time state of a simulation with synchronized time step can be described
        # by a single number. Otherwise, a time value is required for each particle.
        self.time: NDArray[np.float64] | float = (
            np.zeros((self.nparticles, 1)) if not self.is_synchronized_time_step else 0
        )

        # Entered grid -> non-zero if particle EVER entered any grid
        self.entered_grid: NDArray[np.bool_] = np.zeros([self.nparticles]).astype(
            np.bool_
        )

        # Initialize a "progress bar" (really more of a meter)
        # Setting sys.stdout lets this play nicely with regular print()
        pbar = tqdm(
            initial=0,
            total=self.termination_condition.total,
            disable=not self.verbose,
            desc=self.termination_condition.progress_description,
            unit=self.termination_condition.units_string,
            bar_format="{l_bar}{bar}{n:.1e}/{total:.1e} {unit}",
            file=sys.stdout,
        )

        # Push the particles until the termination condition is satisfied
        # or the number of particles being evolved is zero
        is_finished = False
        while not (is_finished or self.nparticles_tracked == 0):
            is_finished = self.termination_condition.is_finished
            progress = min(
                self.termination_condition.progress, self.termination_condition.total
            )

            pbar.n = progress
            pbar.last_print_n = progress
            pbar.update(0)

            self._push()

            # The state of a step is saved after each time step by calling `post_push_hook`
            # The save routine may choose to do nothing with this information
            self.save_routine.post_push_hook()

        # Simulation has finished running
        self._has_run = True

        # Force save of the final state of the simulation
        self.save_routine.save()

        pbar.close()

        self._log("Run completed")

    @property
    def num_entered(self):
        """Count the number of particles that have entered the grids.
        This number is calculated by summing the number of non-zero entries in the
        entered grid array.
        """

        return (self.entered_grid > 0).sum()

    @property
    def fract_entered(self):
        """The fraction of particles that have entered any grid.

        The denominator of this fraction is based off the number of total
        particles.
        """
        return self.num_entered / self.nparticles

    def _remove_grid_values(self, particles_to_remove) -> None:
        """Remove the specified particles from the internal grid
        values object.

        The ``particles_to_remove`` array must be of length ``nparticles``.

        This must be done whenever particles are stopped or removed during an
        intermediate step in the push algorithm. Otherwise, subsequent readouts
        of tracked velocities will not match the shape of the grid values.
        """

        # The dictionary is not defined at the beginning of the simulation
        if not hasattr(self, "_total_grid_values"):
            return

        # Construct masked array with all `True` of length nparticles
        keep_grid_value_mask = np.full(shape=self.nparticles, fill_value=True)

        # Mark the indicated particles for removal from the grid values
        keep_grid_value_mask[particles_to_remove] = False

        # Iterate through the `self._total_grid_values` dictionary, keeping only
        # the necessary values
        self._total_grid_values = {
            key: np.compress(
                keep_grid_value_mask[self._tracked_particle_mask], grid_value
            )
            for key, grid_value in self._total_grid_values.items()
        }

    def _stop_particles(self, particles_to_stop_mask) -> None:
        """Stop tracking the particles specified by the stop mask.

        This is represented by setting the particle's velocity to NaN.
        This method also takes care of removing the specified particles
        from the grid values dictionary.
        """

        if len(particles_to_stop_mask) != self.x.shape[0]:
            raise ValueError(
                f"Expected mask of size {self.x.shape[0]}, got {len(particles_to_stop_mask)}"
            )

        self._remove_grid_values(particles_to_stop_mask)

        self.v[particles_to_stop_mask] = np.nan

    def _remove_particles(self, particles_to_remove_mask) -> None:
        """Remove the specified particles from the simulation.

        For the sake of keeping consistent array lengths, the position and
        velocities of the removed particles are set to NaN.
        """

        if len(particles_to_remove_mask) != self.x.shape[0]:
            raise ValueError(
                f"Expected mask of size {self.x.shape[0]}, got {len(particles_to_remove_mask)}"
            )

        self._remove_grid_values(particles_to_remove_mask)

        self.x[particles_to_remove_mask] = np.nan
        self.v[particles_to_remove_mask] = np.nan

    # *************************************************************************
    # Run/push loop methods
    # *************************************************************************

    def _adaptive_dt(
        self,
        Ex,  # noqa: ARG002
        Ey,  # noqa: ARG002
        Ez,  # noqa: ARG002
        Bx,
        By,
        Bz,
    ) -> NDArray[np.float64] | float:
        r"""
        Calculate the appropriate dt for each grid based on a number of
        considerations including the local grid resolution (ds) and the
        gyroperiod of the particles in the current fields.
        """

        # candidate time steps includes one per grid (based on the grid resolution)
        # plus additional candidates based on the field at each particle
        candidates = np.ones([self.nparticles, self.num_grids + 1]) * np.inf

        # Compute the time step indicated by the grid resolution
        ds = np.array([grid.grid_resolution.to(u.m).value for grid in self.grids])
        gridstep = self._Courant_parameter * (ds / self.vmax)

        # Wherever a particle is on a grid, include that grid's grid step
        # in the list of candidate time steps
        for i, _grid in enumerate(self.grids):  # noqa: B007
            candidates[:, i] = np.where(
                self.particles_on_grid[:, i] > 0, gridstep[i], np.inf
            )

        # If not, compute a number of possible time steps
        # Compute the cyclotron gyroperiod
        Bmag = np.max(np.sqrt(Bx**2 + By**2 + Bz**2)).to(u.T).value
        # Compute the gyroperiod
        if Bmag == 0:
            gyroperiod = np.inf
        else:
            gyroperiod = (
                2 * np.pi * self.m / (np.abs(self.q) * np.max(Bmag))
            )  # Account for negative charges!

        # Subdivide the gyroperiod into a provided number of steps
        # Use the result as the candidate associated with gyration in B field
        candidates[:, self.num_grids] = gyroperiod / self._steps_per_gyroperiod

        # TODO: introduce a minimum time step based on electric fields too!

        # Enforce limits on dt
        candidates = np.clip(candidates, self.dt_range[0], self.dt_range[1])

        if not self._is_synchronized_time_step:
            # dt is the min of all the candidates for each particle
            # a separate dt is returned for each particle
            dt = np.min(candidates, axis=-1)

            # dt should never actually be infinite, so replace any infinities
            # with the largest gridstep
            dt[dt == np.inf] = np.max(gridstep)
        else:
            # a single value for dt is returned
            # this is the time step used for all particles
            dt = np.min(candidates)

        return dt

    @property
    def particles_on_grid(self):
        r"""
        Returns a boolean mask of shape [nparticles, ngrids] corresponding to
        whether or not the particle is on the associated grid.
        """

        all_particles = np.array([grid.on_grid(self.x * u.m) for grid in self.grids]).T
        all_particles[~self._tracked_particle_mask] = False

        return all_particles

    @property
    def _particle_kinetic_energy(self):
        r"""
        Return the relativistic kinetic energy of the particles.
        """
        current_speeds = np.linalg.norm(self.v, axis=-1, keepdims=True)

        beta = current_speeds / _c.si.value
        gamma = np.sqrt(1 / (1 - beta**2))

        return (gamma - 1) * self.m * _c.si.value**2

    def _interpolate_grid(self):
        # Get a list of positions (input for interpolator)
        tracked_mask = self._tracked_particle_mask

        self.iteration_number += 1

        pos_tracked = self.x[tracked_mask]

        # entered_grid is zero at the end if a particle has never
        # entered any grid
        self.entered_grid += np.sum(self.particles_on_grid, axis=-1).astype(np.bool_)

        # TODO: how should we handle unrecognized quantities?

        # Construct a dictionary to store the interpolation results
        # This is accomplished using a dictionary comprehension. Each quantity is
        # initialized as a zeros array with its respective units
        total_grid_values = {
            field_name: np.zeros(self.nparticles_tracked)
            * AbstractGrid.recognized_quantities[field_name].unit
            for field_name in self._required_quantities
        }

        for grid in self.grids:
            # TODO: maybe convert this to a ternary operator for now?
            match self.field_weighting:
                case "volume averaged":
                    interpolation_method = grid.volume_averaged_interpolator
                case "nearest neighbor":
                    interpolation_method = grid.nearest_neighbor_interpolator

            # Use the keys of `total_grid_values` as input quantity strings to the interpolator
            grid_values = interpolation_method(
                pos_tracked * u.m, *total_grid_values.keys(), persistent=True
            )

            # Iterate through the interpolated fields and add them to the running sum
            # NaN values are zeroed
            for grid_value, grid_name in zip(
                grid_values,
                total_grid_values.keys(),
                strict=True,
            ):
                total_grid_values[grid_name] += np.nan_to_num(
                    grid_value,
                    0.0 * AbstractGrid.recognized_quantities[grid_name].unit,
                )

        return total_grid_values

    def _update_time(self):
        r"""
        Calculate an appropriate time step for the simulation with respect to
        user configuration. Returns the calculated time step.
        """
        # Calculate the adaptive time step from the fields currently experienced
        # by the particles
        # If user sets dt explicitly, that's handled in _adaptive_dt
        if self._is_adaptive_time_step:
            dt = self._adaptive_dt(
                self._total_grid_values["E_x"],
                self._total_grid_values["E_y"],
                self._total_grid_values["E_z"],
                self._total_grid_values["B_x"],
                self._total_grid_values["B_y"],
                self._total_grid_values["B_z"],
            )
        else:
            dt = self.dt

        # Make sure the time step can be multiplied by a [nparticles, 3] shape field array
        if isinstance(dt, np.ndarray) and dt.size > 1:
            dt = dt[:, np.newaxis]

            # Increment the tracked particles' time by dt
            self.time[self._tracked_particle_mask] += dt[self._tracked_particle_mask]
        else:
            self.time += dt

        return dt

    def _update_position(self):
        r"""
        Update the positions and velocities of the simulated particles using the
        integrator provided at instantiation.
        """

        # Create arrays of E and B as required by push algorithm
        E = np.array(
            [
                self._total_grid_values["E_x"].to(u.V / u.m).value,
                self._total_grid_values["E_y"].to(u.V / u.m).value,
                self._total_grid_values["E_z"].to(u.V / u.m).value,
            ]
        )
        E = np.moveaxis(E, 0, -1)
        B = np.array(
            [
                self._total_grid_values["B_x"].to(u.T).value,
                self._total_grid_values["B_y"].to(u.T).value,
                self._total_grid_values["B_z"].to(u.T).value,
            ]
        )
        B = np.moveaxis(B, 0, -1)

        pos_tracked = self.x[self._tracked_particle_mask]
        vel_tracked = self.v[self._tracked_particle_mask]

        # If `dt` is an array then only take the tracked particles
        # otherwise, if it's just a scalar then we can use dt directly
        if isinstance(self.dt, np.ndarray) and self.dt.size > 1:
            dt = self.dt[self._tracked_particle_mask]
        else:
            dt = self.dt

        x_results, v_results = self._integrator.push(
            pos_tracked,
            vel_tracked,
            B,
            E,
            self.q,
            self.m,
            dt,
        )

        tracked_particle_mask = self._tracked_particle_mask

        self.x[tracked_particle_mask], self.v[tracked_particle_mask] = (
            x_results,
            v_results,
        )

    def _multi_grid_check(self):
        # Ensure there are no particles on multiple grids
        if (np.sum(self.particles_on_grid, axis=-1) > 1).any():
            warnings.warn(
                "Detected particles existing in the domain of multiple grids. This will "
                "lead to undefined behavior in scattering results.",
                category=RuntimeWarning,
            )

    def _update_velocity_stopping(self):
        r"""
        Apply stopping to the simulated particles using the provided stopping
        routine. The stopping is applied to the simulation by calculating the
        new energy values of the particles, and then updating the particles'
        velocity to match these energies.
        """

        current_speeds = np.linalg.norm(
            self.v[self._tracked_particle_mask], axis=-1, keepdims=True
        )
        velocity_unit_vectors = np.multiply(
            1 / current_speeds, self.v[self._tracked_particle_mask]
        )

        if isinstance(self.dt, np.ndarray) and self.dt.size > 1:
            dt = self.dt[self._tracked_particle_mask]
        else:
            dt = self.dt

        dx = np.multiply(current_speeds, dt)

        relevant_kinetic_energy = (
            self._particle_kinetic_energy[self._tracked_particle_mask] * u.J
        )

        # The total energy loss per unit length can be summed over according to Bragg's rule
        total_energy_loss_per_length = np.zeros(shape=(self.nparticles_tracked, 1))

        match self._stopping_method:
            case "NIST":
                for i, interpolator in enumerate(self._stopping_power_interpolators):
                    # Determine which particles are on this grid
                    particles_on_grid = self.particles_on_grid[
                        self._tracked_particle_mask, i
                    ]

                    # Perform interpolation only for particles on the grid
                    total_energy_loss_per_length[particles_on_grid] += (
                        (interpolator(relevant_kinetic_energy[particles_on_grid]))
                        * self._target_rho[i, np.newaxis]
                    ).si.value
            case "Bethe":
                for i, interpolator in enumerate(self._stopping_power_interpolators):
                    # Determine which particles are on this grid
                    particles_on_grid = self.particles_on_grid[
                        self._tracked_particle_mask, i
                    ]

                    total_energy_loss_per_length[particles_on_grid] += interpolator(
                        current_speeds[particles_on_grid]
                    )

        dE = -total_energy_loss_per_length * dx

        # Update the velocities of the particles using the new energy values
        # TODO: again, figure out how to differentiate relativistic and classical cases
        E = self._particle_kinetic_energy[self._tracked_particle_mask] + dE

        if (
            self._stopping_method == "Bethe"
            and not self._raised_energy_warning
            and np.nanmin(E * u.J).to(u.MeV).value < 1
        ):
            self._raised_energy_warning = True

            warnings.warn(
                "The Bethe model is only valid for high energy particles. Consider using "
                "NIST stopping if you require accurate stopping powers at lower energies.",
                category=PhysicsWarning,
            )

        particles_to_be_stopped_mask = np.full(
            shape=self._tracked_particle_mask.shape, fill_value=False
        )
        tracked_particles_to_be_stopped_mask = (
            E < 0
        ).flatten()  # A subset of the tracked particles!
        # Of the tracked particles, stop the ones indicated by the subset mask
        particles_to_be_stopped_mask[self._tracked_particle_mask] = (
            tracked_particles_to_be_stopped_mask
        )

        # Set negative energies equal to zero to prevent sqrt warnings
        E[E < 0] = 0

        rest_energy = self.m * _c.si.value**2
        new_speeds = _c.si.value * np.sqrt(1 - ((E / rest_energy) + 1) ** (-2))

        self.v[self._tracked_particle_mask] = np.multiply(
            new_speeds, velocity_unit_vectors
        )

        self._stop_particles(particles_to_be_stopped_mask)

    def _update_velocity_scattering(self):
        speeds = np.linalg.norm(self.v[self._tracked_particle_mask], axis=-1)

        match self._scattering_method:
            # Numerically integrate the scattering integral to construct the mean square scatter rate
            case "differential cross section":
                # `theta_prime` dummy variable introduced to prevent global shadow of `theta`
                def mean_squared_integrand(theta_prime, speed, T_i, n_i):
                    return (
                        theta_prime**2
                        * self._scattering_routine(
                            theta=theta_prime,
                            speed=speed,
                            beam_particle=self._particle,
                            target_particle=self._target,
                            T_s=T_i,
                            n_s=n_i,
                        )
                    ).si.value

                # Calculate the minimum scattering angle based on screening
                mu = reduced_mass(self._target, self._particle)
                b_0 = (
                    self._particle.atomic_number
                    * self._target.atomic_number
                    * const.e.si**2
                    / (4 * np.pi * const.eps0 * mu * speeds**2)
                )
                lambda_D = Debye_length(
                    self._total_grid_values["T_i"], self._total_grid_values["n_i"]
                )
                theta_min = 2 * b_0 / lambda_D

                # Numerically integrate the classical scattering integral
                mean_square_scatter_rate = [
                    (
                        2
                        * np.pi
                        * n_i
                        * speed
                        * quad(
                            mean_squared_integrand,
                            lower_bound,
                            np.pi,
                            args=(speed, T_i, n_i),
                        )[0]
                    )
                    for (T_i, n_i, speed, lower_bound) in zip(
                        self._total_grid_values["T_i"],
                        self._total_grid_values["n_i"],
                        speeds,
                        theta_min.value,
                        strict=False,
                    )
                ]  # Quad returns a tuple, we only care about the result
            case "mean square rate":
                # Ensure there are no particles on multiple grids
                self._multi_grid_check()

                mean_square_scatter_rate = np.zeros(shape=self.nparticles_tracked)

                for i, _grid in enumerate(self.grids):  # noqa: B007
                    particles_on_grid = self.particles_on_grid[
                        self._tracked_particle_mask, i
                    ]

                    # Passing an empty array to the scattering routine can cause an error to
                    # be emitted
                    if particles_on_grid.sum() == 0:
                        continue

                    # TODO: figure out additivity rule for scattering
                    mean_square_scatter_rate[particles_on_grid] += (
                        self._scattering_routine(
                            speed=speeds[particles_on_grid],
                            beam_particle=self._particle,
                            target_particle=self._target[i],
                            T_s=self._total_grid_values["T_i"][
                                particles_on_grid
                            ].si.value,
                            n_s=self._target_number_density[i],
                        )
                    )

        mean_square_scatter_rate = np.reshape(
            mean_square_scatter_rate, newshape=(self.nparticles_tracked, 1)
        )

        # The (polar) angle resulting from scattering
        theta = self._rng.normal(
            scale=np.sqrt(
                mean_square_scatter_rate * self.dt[self._tracked_particle_mask]
            ),
            size=(self.nparticles_tracked, 1),
        )

        theta = np.nan_to_num(theta, 0)

        # Generate a vector that lies somewhere in the plane perpendicular to the velocity
        k = np.cross(
            self.v[self._tracked_particle_mask],
            self._rng.uniform(low=-1, high=1, size=(self.nparticles_tracked, 3)),
        )
        # Normalize
        k_norm = k / np.linalg.norm(k, axis=-1, keepdims=True)
        # Scale to the perpendicular component of the change in velocities
        delta_v = speeds[:, np.newaxis] * np.sin(theta) * k_norm

        # Begin by constructing the new velocities by adding the change in
        # velocity to the current velocities. The direction of this vector will be
        # correct but will need to be "rescaled" to ensure that the speed remains
        # consistent.
        v_new_unscaled = self.v[self._tracked_particle_mask] + delta_v

        # Normalize and rescale
        self.v[self._tracked_particle_mask] = (
            v_new_unscaled
            / np.linalg.norm(v_new_unscaled, axis=-1, keepdims=True)
            * speeds[:, np.newaxis]
        )

    def _push(self) -> None:
        r"""
        Advance particles using an implementation of the time-centered
        Boris algorithm.
        """
        # Interpolate fields at particle positions
        self._total_grid_values = self._interpolate_grid()

        # Calculate an appropriate timestep (uniform, synchronized)
        self.dt = self._update_time()

        # Update the position and velocities of the particles using timestep
        # calculations as well as the magnitude of E and B fields
        self._update_position()

        if not self._integrator.is_relativistic and not self._raised_relativity_warning:
            beta_max = self.vmax / const.c.si.value

            if beta_max >= 0.001:
                self._raised_relativity_warning = True

                warnings.warn(
                    f"Particles have reached {beta_max}% of the speed of light. Consider using "
                    f"a relativistic integrator for more accurate results.",
                    RelativityWarning,
                )

        if self.nparticles_tracked == 0:
            return

        # Update velocities to reflect stopping
        if self._do_stopping:
            self._update_velocity_stopping()

        # It is possible for all particles to have been stopped. This could cause
        # an error for certain numpy operations if the event loop were to continue
        if self.nparticles_tracked == 0:
            return

        if self._do_scattering:
            self._update_velocity_scattering()

    @property
    def on_any_grid(self) -> NDArray[np.bool_]:
        """
        Binary array for each particle indicating whether it is currently
        on ANY grid.
        """
        return np.sum(self.particles_on_grid, axis=-1) > 0

    @property
    def vmax(self) -> float:
        """The maximum velocity of any particle in the simulation.

        This quantity is used for determining the grid crossing maximum time step.
        """
        tracked_mask = self._tracked_particle_mask

        return float(np.max(np.linalg.norm(self.v[tracked_mask], axis=-1)))

    @property
    def _tracked_particle_mask(self) -> NDArray[np.bool_]:
        """
        Calculates a boolean mask corresponding to particles that have not been stopped or removed.
        """
        # See Class docstring for definition of `stopped` and `removed`
        return ~np.logical_or(np.isnan(self.x[:, 0]), np.isnan(self.v[:, 0]))

    @property
    def nparticles_tracked(self) -> int:
        """Return the number of particles currently being tracked.
        That is, they do not have NaN position or velocity.
        """
        return int(self._tracked_particle_mask.sum())

    @property
    def is_adaptive_time_step(self) -> bool:
        """Return whether the simulation is calculating an adaptive time step or using the user-provided time step."""
        return self._is_adaptive_time_step

    @property
    def is_synchronized_time_step(self) -> bool:
        """Return if the simulation is applying the same time step across all particles."""
        return self._is_synchronized_time_step

    def _enforce_particle_creation(self) -> None:
        """Ensure the array position array `x` has been populated."""

        # Check to make sure particles have already been generated
        if not hasattr(self, "x"):
            raise ValueError(
                "The create_particles method must be called before running the particle "
                "tracing algorithm."
            )

    def _enforce_order(self) -> None:
        r"""
        The `Tracker` methods could give strange results if setup methods
        are used again after the simulation has run. This method
        raises an error if the simulation has already been run.

        """

        if self._has_run:
            raise RuntimeError(
                "Modifying the `Tracker` object after running the "
                "simulation is not supported. Create a new `Tracker` "
                "object for a new simulation."
            )
