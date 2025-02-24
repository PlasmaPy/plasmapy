import astropy.units as u
from plasmapy.analysis.magnetics import compute_bfield
from plasmapy.utils.decorators import validate_quantities
from typing import Tuple


class Bdot:
    """_summary_

    """
    _settings = {}
    _signal = None
    _time = None
    _area = None

    @validate_quantities
    def __init__(
        self,
        signal: u.volt,
        time: u.s,
        area: u.m**2,
        *,
        nloops,
        gain,
    ):
        self._signal = signal.value
        self._time = time.value
        self._area = area.value
        self._settings = {
            "nloops": nloops,
            "gain": gain,
        }

    @validate_quantities
    def calc_bfield(self) -> u.T:
        """_summary_

        """
        bfield = compute_bfield(
            self._signal,
            self._time,
            self._area,
            #    ** self._settings,
        )
        return bfield * u.T
