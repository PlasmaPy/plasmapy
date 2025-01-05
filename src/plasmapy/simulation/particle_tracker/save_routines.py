"""Save routines for the |ParticleTracker|."""

__all__ = [
    "AbstractSaveRoutine",
    "DoNotSaveSaveRoutine",
    "SaveOnceOnCompletion",
    "IntervalSaveRoutine",
]

from abc import ABC, abstractmethod
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np


class AbstractSaveRoutine(ABC):
    """Abstract base class containing the necessary methods for a
    `~plasmapy.simulation.particle_tracker.particle_tracker.ParticleTracker` save routine.

    The save routine class is responsible for defining the conditions and hooks
    for saving.

    Parameters
    ----------
    output_directory : `~pathlib.Path`, optional
        Output for objects that are saved to disk. If a directory is not specified
        then a memory save routine is used.

    output_basename : `str`, optional
        Optional string basename for saved files.


    Notes
    -----
    After every push, the `post_push_hook` method is called with the
    respective `~plasmapy.simulation.particle_tracker.particle_tracker.ParticleTracker` object passed as a parameter.
    Then, the hook calls `save_now` to determine whether or not the simulation state should be saved.
    """

    def __init__(
        self, output_directory: Path | None = None, output_basename: str = "output"
    ) -> None:
        self.output_directory = output_directory
        self.output_basename = output_basename

        self._results = {}
        self._quantities = {
            "time": (u.s, "dataset"),
            "x": (u.m, "dataset"),
            "v": (u.m / u.s, "dataset"),
        }

        self._particle_tracker = None

    @property
    def tracker(self):
        """Return the |ParticleTracker| object for this stop condition."""
        return self._particle_tracker

    @tracker.setter
    def tracker(self, particle_tracker) -> None:
        self._particle_tracker = particle_tracker

    @property
    @abstractmethod
    def require_synchronized_dt(self) -> bool:
        """Return if this save routine requires a synchronized time step."""
        ...

    @property
    @abstractmethod
    def save_now(self) -> bool:
        """Determine if to save on the current push step."""
        ...

    @property
    def results(self) -> dict[str, u.Quantity]:
        """Return the results of the simulation.
        The quantities returned depend on those defined in the body of the save routine.
        """
        return self._apply_units_to_results()

    def save(self) -> None:
        """Save the current state of the simulation to memory.

        If an output directory is specified then the state will also be saved
        to the disk.
        """

        self._save_to_memory()

        if self.output_directory is not None:
            self._save_to_disk()

    def _save_to_disk(self) -> None:
        """Save a hdf5 file containing simulation positions and velocities."""

        path = (
            self.output_directory
            / f"{self.output_basename}_iter{self.tracker.iteration_number}.h5"
        )

        with h5py.File(path, "w") as output_file:
            for key, (_units, data_type) in self._quantities.items():
                match data_type:
                    case "attribute":
                        output_file.attrs.create(key, self._results[key])
                    case "dataset":
                        output_file.create_dataset(key, data=self._results[key])

    # TODO: Find a better name for this method
    def _save_to_memory(self) -> None:
        """Update the results dictionary with the current state of the simulation."""

        for quantity in self._quantities:
            quantity_history = self._results.get(quantity, [])
            current_quantity = np.copy(getattr(self._particle_tracker, quantity, 0))

            quantity_history.append(current_quantity)
            self._results[quantity] = quantity_history

    def _apply_units_to_results(self):
        """Apply units to the results dictionary.

        This function is called anytime the results dictionary is retrieved
        """
        results_copy = self._results.copy()

        for quantity, (units, _data_type) in self._quantities.items():
            # Only apply units if they are specified
            # otherwise assume the quantity is dimensionless
            if units is not None:
                results_copy[quantity] = results_copy[quantity] * units

        return results_copy

    def post_push_hook(self) -> None:
        """Function called after a push step.

        This function is responsible for handling two steps of save routine, namely:
            - Deciding to save on the current time step
            - How the simulation data is saved (i.e. to disk or memory)

        """

        # Update the result dictionary
        if self.save_now:
            self.save()


class DoNotSaveSaveRoutine(AbstractSaveRoutine):
    """The default save routine for the `~plasmapy.simulation.particle_tracker.particle_tracker.ParticleTracker` class.

    This save routine is a placeholder and will not save the state of the particle tracker.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def require_synchronized_dt(self) -> bool:
        """The do not save save routine does not require a synchronized time step."""
        return False

    @property
    def save_now(self) -> bool:
        """The do not save save routine will never save by definition."""
        return False


class SaveOnceOnCompletion(AbstractSaveRoutine):
    """Save only once the `~plasmapy.simulation.particle_tracker.particle_tracker.ParticleTracker` has finished.

    This works by taking advantage of the fact that the ``save()`` method
    is called directly at the conclusion of a simulation, effectively
    bypassing the ``save_now()`` criteria.
    """

    def __init__(
        self, output_directory: Path | None = None, output_basename: str = "output"
    ) -> None:
        super().__init__(output_directory, output_basename)

    @property
    def save_now(self) -> bool:
        """Never save during the simulation."""
        return False

    @property
    def require_synchronized_dt(self) -> bool:
        """A synchronized time step is not required for this save routine to make sense."""
        return False


class IntervalSaveRoutine(AbstractSaveRoutine):
    """Abstract class describing a save routine that saves every given interval."""

    def __init__(self, interval: u.Quantity, **kwargs) -> None:
        super().__init__(**kwargs)

        self._quantities = {
            "time": (u.s, "dataset"),
            "x": (u.m, "dataset"),
            "v": (u.m / u.s, "dataset"),
        }

        self.save_interval: float = interval.to(u.s).value
        self.time_of_last_save: float = 0

    @property
    def require_synchronized_dt(self) -> bool:
        """Save output only makes sense for synchronized time steps."""
        return True

    @property
    def save_now(self) -> bool:
        """Save at every interval given in instantiation."""

        return bool(self.tracker.time - self.time_of_last_save >= self.save_interval)

    def save(self) -> None:
        """Save the current state of the simulation.
        Sets the time of last save attribute and log the timestamp.
        """
        super().save()

        self.time_of_last_save = self.tracker.time
