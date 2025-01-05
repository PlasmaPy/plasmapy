"""Termination conditions for the |ParticleTracker|."""

__all__ = [
    "AbstractTerminationCondition",
    "TimeElapsedTerminationCondition",
    "NoParticlesOnGridsTerminationCondition",
    "AllParticlesOffGridTerminationCondition",
]

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
import numpy.typing as npt


class AbstractTerminationCondition(ABC):
    """Abstract base class containing the necessary methods for a ParticleTracker termination condition."""

    @property
    def tracker(self):
        """Return the |ParticleTracker| object for this termination condition."""
        return self._particle_tracker

    @tracker.setter
    def tracker(self, particle_tracker) -> None:
        self._particle_tracker = particle_tracker

    @property
    @abstractmethod
    def require_synchronized_dt(self) -> bool:
        """Return if this termination condition requires a synchronized time step."""
        ...

    @property
    @abstractmethod
    def progress_description(self) -> str:
        """Return a small string describing the relevant quantity shown on the meter."""
        ...

    @property
    @abstractmethod
    def units_string(self) -> str:
        """Return a string representation of the units of `total`."""
        ...

    @property
    @abstractmethod
    def is_finished(self) -> bool:
        """Return `True` if the simulation has finished."""
        ...

    @property
    @abstractmethod
    def progress(self) -> float:
        """Return a number representing the progress of the simulation (compared to total).
        This number represents the numerator of the completion percentage.
        """
        ...

    @property
    @abstractmethod
    def total(self) -> float:
        """Return a number representing the total number of steps in a simulation.
        This number represents the denominator of the completion percentage.
        """
        ...


class TimeElapsedTerminationCondition(AbstractTerminationCondition):
    """Termination condition corresponding to the elapsed time of a ParticleTracker."""

    def __init__(self, termination_time: u.Quantity) -> None:
        self.termination_time: float = termination_time.to(u.s).value

    @property
    def require_synchronized_dt(self) -> bool:
        """The elapsed time termination condition requires a synchronized step
        to stop all particles at the same time.
        """
        return True

    @property
    def progress_description(self) -> str:
        """The time elapsed termination condition depends on elapsed time,
        therefore the relevant quantity is time remaining.
        """
        return "Time remaining"

    @property
    def units_string(self) -> str:
        """The units for the time elapsed condition have the units of seconds."""

        return "seconds"

    @property
    def is_finished(self) -> bool:
        """Conclude the simulation if all particles have been tracked over the specified termination time."""

        return bool(float(self.tracker.time) >= self.termination_time)

    @property
    def progress(self) -> float:
        """Return the current time step of the simulation."""
        return float(self.tracker.time)

    @property
    def total(self) -> float:
        """Return the total amount of time over which the particles are tracked."""
        return self.termination_time


class NoParticlesOnGridsTerminationCondition(AbstractTerminationCondition):
    """Termination condition corresponding to stopping the simulation when all particles have exited the grid."""

    def __init__(self) -> None:
        pass

    @property
    def require_synchronized_dt(self) -> bool:
        """The no field termination condition does not require a synchronized time step."""
        return False

    @property
    def progress_description(self) -> str:
        """The progress meter is described in terms of the fraction of particles still on the grid."""
        return "Number of particles still on grid"

    @property
    def units_string(self) -> str:
        """The progress meter is described in terms of the fraction of particles still on the grid."""
        return "Particles"

    @property
    def is_finished(self) -> bool:
        """The simulation is finished when no more particles are on any grids."""
        is_not_on_grid = self.tracker.on_any_grid == False  # noqa: E712

        return is_not_on_grid.all() and self.tracker.iteration_number > 0

    @property
    def progress(self) -> float:
        """The progress of the simulation is measured by how many particles are no longer on a grid."""
        is_not_on_grid: npt.NDArray[np.bool_] = self.tracker.particles_on_grid == 0

        return float(is_not_on_grid.sum())

    @property
    def total(self) -> float:
        """The progress of the simulation is measured against the total number of particles in the simulation."""
        return float(self.tracker.nparticles)


class AllParticlesOffGridTerminationCondition(AbstractTerminationCondition):
    """Termination condition corresponding to stopping the simulation when a specified
    proportion of particles have entered and exited the grids.

    Parameters
    ----------
    fraction_exited_threshold: float, optional
        The fraction of particles that must leave the grids to terminate the simulation.
        This does not include particles that have never entered the grids.
    """

    def __init__(
        self,
        fraction_exited_threshold: float = 0.99,
    ) -> None:
        super().__init__()

        self.fraction_exited_threshold = fraction_exited_threshold

    @property
    def require_synchronized_dt(self) -> bool:
        """The termination condition does not require a synchronized time step."""
        return False

    @property
    def progress_description(self) -> str:
        """The termination condition tracks the number of particles remaining in the grids."""
        return "Fraction of particles remaining"

    @property
    def units_string(self) -> str:
        """The termination condition tracks particles."""

        return "Particles"

    @property
    def is_finished(self) -> bool:
        r"""
        Check to see if the proportion of particles that have entered and exited the grid meet thresholds.
        """

        # Of the particles that have entered the grid, how many are currently
        # on the grid?
        # if/else avoids dividing by zero
        if self._particle_tracker.num_entered > 0:
            # Normalize to the number that have entered a grid
            still_on = (
                np.sum(self._particle_tracker.on_any_grid)
                / self._particle_tracker.num_entered
            )
        else:
            still_on = 0.0

        proportion_exited = 1 - still_on

        return (
            self._particle_tracker.num_entered > 0
            and self.fraction_exited_threshold <= proportion_exited
        )

    @property
    def progress(self) -> float:
        """The progress of the simulation is defined in the context of how many particles are on the grids."""
        return self._particle_tracker.on_any_grid.sum()

    @property
    def total(self) -> float:
        """The progress of the simulation is compared to the number of particles tracked."""
        return self._particle_tracker.nparticles_tracked
