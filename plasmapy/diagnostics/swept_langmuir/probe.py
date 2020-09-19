"""
Defines the `SweptLangmuirProbe` class which is used to fully defined the
physical parameters of a Langmuir Probe.
"""
__all__ = ['SweptLangmuirProbe']

import astropy.units as u

from plasmapy.diagnostics import AbstractProbe
from plasmapy.utils.decorators import validate_quantities


class SweptLangmuirProbe(AbstractProbe):
    """Class to defined the physical parameters of a Langmuir probe used for a
    swept analysis.
    """
    @validate_quantities
    def __init__(self, area: u.cm ** 2):
        self._area = area

    @property
    def area(self) -> u.cm ** 2:
        return self._area

    @area.setter
    @validate_quantities(value={'can_be_negative': False,
                                'can_be_complex': False,
                                'none_shall_pass': False})
    def area(self, value: u.cm ** 2):
        self._area = value
