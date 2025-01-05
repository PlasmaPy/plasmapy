"""
Reading in HDF5 files following the |OpenPMD| standard.

.. attention::

   |expect-api-changes|
"""

from types import TracebackType

__all__ = ["HDF5Reader"]

from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from packaging.version import Version

from plasmapy.plasma.exceptions import DataStandardError
from plasmapy.plasma.plasma_base import GenericPlasma

_OUTDATED_VERSION = "1.1.0"
_NEWER_VERSION = "2.0.0"

# This is the order what OpenPMD uses to store unit
# dimensions for a record.
_UNITS = (u.meter, u.kilogram, u.second, u.ampere, u.Kelvin, u.mol, u.candela)


def _fetch_units(openPMD_dims):
    """Converts a collection of OpenPMD dimensions to astropy.units."""

    units = u.dimensionless_unscaled
    for factor, unit in zip(openPMD_dims, _UNITS, strict=False):
        units *= unit**factor
    units, *_ = units.compose()
    return units


def _valid_version(openPMD_version, outdated=_OUTDATED_VERSION, newer=_NEWER_VERSION):
    """Checks if the passed version is supported or not."""

    parsed_version = Version(openPMD_version)
    outdated_version = Version(outdated)
    newer_version = Version(newer)
    return outdated_version <= parsed_version < newer_version


class HDF5Reader(GenericPlasma):
    """
    Core class for accessing various attributes on HDF5 files that
    are based on `OpenPMD <https://www.openpmd.org>`__ standards.

    Parameters
    ----------
    hdf5 : `str`
        Path to HDF5 file.

    **kwargs
        Any keyword accepted by `~plasmapy.plasma.plasma_base.GenericPlasma`.
    """

    def __init__(self, hdf5, **kwargs) -> None:
        super().__init__(**kwargs)

        if not Path(hdf5).is_file():
            raise FileNotFoundError(f"Could not find file: '{hdf5}'")

        h5 = h5py.File(hdf5, "r")
        self.h5 = h5

        self._check_valid_openpmd_version()

        self.subname = next(iter(self.h5["data"]))

    def __enter__(self):
        return self.h5

    def close(self) -> None:
        self.h5.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.h5.close()

    def _check_valid_openpmd_version(self) -> bool:
        try:
            openPMD_version = self.h5.attrs["openPMD"].decode("utf-8")
            if _valid_version(openPMD_version):
                return True
            else:
                raise DataStandardError(
                    f"We currently only support HDF5 versions"
                    f"starting from v{_OUTDATED_VERSION} and "
                    f"lower than v{_NEWER_VERSION}. You can "
                    f"however convert your HDF5 to a supported "
                    f"version. For more information; see "
                    f"https://github.com/openPMD/openPMD-updater"
                )
        except KeyError as ex:
            raise DataStandardError(
                "Input HDF5 file does not go on with standards defined by OpenPMD"
            ) from ex

    @property
    def electric_field(self):
        """
        An :math:`(x, y, z)` array containing electric field data.

        Returns
        -------
        `~astropy.units.Quantity`
        """
        path = f"data/{self.subname}/fields/E"
        if path in self.h5:
            units = _fetch_units(self.h5[path].attrs["unitDimension"])
            axes = [self.h5[path][axis] for axis in self.h5[path]]
            return np.array(axes) * units
        else:
            raise AttributeError("No electric field data available in HDF5 file")

    @property
    def charge_density(self):
        """
        An array containing charge density data.

        Returns
        -------
        `~astropy.units.Quantity`
        """
        path = f"data/{self.subname}/fields/rho"
        if path in self.h5:
            units = _fetch_units(self.h5[path].attrs["unitDimension"])
            return np.array(self.h5[path]) * units
        else:
            raise AttributeError("No charge density data available in HDF5 file")

    @property
    def magnetic_field(self):
        """
        An array containing magnetic field data.

        Returns
        -------
        `~astropy.units.Quantity`
        """
        path = f"data/{self.subname}/fields/B"
        if path in self.h5:
            units = _fetch_units(self.h5[path].attrs["unitDimension"])
            axes = [self.h5[path][axis] for axis in self.h5[path]]
            return np.array(axes) * units
        else:
            raise AttributeError("No magnetic field data available in HDF5 file")

    @property
    def electric_current(self):
        """
        An array containing electric current data.

        Returns
        -------
        `~astropy.units.Quantity`
        """
        path = f"data/{self.subname}/fields/J"
        if path in self.h5:
            units = _fetch_units(self.h5[path].attrs["unitDimension"])
            axes = [self.h5[path][axis] for axis in self.h5[path]]
            return np.array(axes) * units
        else:
            raise AttributeError("No electric current data available in HDF5 file")

    @classmethod
    def is_datasource_for(cls, **kwargs):
        if "hdf5" not in kwargs:
            return False

        hdf5 = kwargs.get("hdf5")
        openPMD = kwargs.get("openPMD")

        if not Path(hdf5).is_file():
            raise FileNotFoundError(f"Could not find file: '{hdf5}'")

        if "openPMD" not in kwargs:
            h5 = h5py.File(hdf5, "r")
            try:
                openPMD = h5.attrs["openPMD"]
            except KeyError:
                openPMD = False

        return openPMD
