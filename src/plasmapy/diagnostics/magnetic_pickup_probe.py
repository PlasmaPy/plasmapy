"""Defines the basic magnetic pickup probe (Bdot probe) analysis
   
"""
__all__ = []

from warnings import warn
import scipy.integrate as sp
from typing import Tuple
import numpy as np


import astropy.units as u
from plasmapy.utils.decorators import validate_quantities
from typing import Tuple

@validate_quantities
def compute_bfield(bdot_voltage: u.Quantity[u.volt], time_array: u.Quantity[u.s], loop_area: u.Quantity[u.m**2], num_loop: int = 1, gain: float = 1.0) -> u.Quantity[u.T]:
    r"""
    Takes the voltage output of a magnetic pickup probe (Bdot probe) and time base and 
    returns the associated array of the magnetic field as a function of time

    Parameters
    ----------
    bdot_voltage: `astropy.units.Quantity`
        A data array containing voltage fluctuations from a bdot probe. The
        array values are proportional to time changing magnetic field via Faradays Law
        The input values are assumed to be in volts.

    time_array: `astropy.units.Quantity`
        The time series to the data collection in units of seconds.

    loop_area: 'astropy.units.Quantity'
        The area through which the changing flux is measured in units of meters squared.

    num_loop: int
        The number of loops of the Bdot probe (assumed to be all the same area)
        
    gain: float
        Any gains that need to be applied
        
    Returns
    -------
    magnetic_field_array: `astropy.units.Quantity`
        The array containing the magnetic field in Tesla as a function of time in seconds. 
        
    Notes
    -----
    The integral form of Faraday's Law states that the time change in flux through an area generates a voltage around a loop,
    .. math::
       V = -N\frac{d\Phi}{dt}
    where :math:`\Phi` is the magnetic flux
    .. math::
        \Phi = \int{BdA}
    and N is the number of loops.
    A magnetic pickup probe works by providing a fixed area loop or loops using a length of conducting wire through 
    which magnetic fields penetrate. As these fields change in time, a voltage is generated that can be measured
    by some kind of data aquisition device (such as an ossciloscope). We can also assume that the loop is small 
    enough that the magnetic field through the loop is constant everywhere in the loop (and only changes in time).
    Given this, we can write
    .. math::
        V = -A\frac{dB}{dt}
    where the voltage becomes preportional only to the change in magnetic field and the area of the loop. Magnetic
    field as a function of time can be recovered by time integrating :math:`V(t)/NA`.
    
    Examples
    --------
    >>> import numpy as np
    >>> import astropy.units as u
    time_array = np.linspace(0,4*np.pi,100) #create an time array
    bdot_voltage = np.sin(time_array) #create a sine wave representing a voltage oscillation
    loop_area = 1 #define a loop with area that will be in meters squared
    num_loop = 1
    gain = 1
    bdot_voltage = bdot_voltage * u.volt
    time_array = time_array * u.s
    loop_area = loop_area * u.m**2
    bfield = compute_bfield(
                bdot_voltage,
                time_array,
                loop_area,
                num_loop=num_loop,
                gain=gain)
    """
    if len(bdot_voltage) != len(time_array):
        raise Exception("length of time and voltage arrays in not equal\n")
    
    bdot_voltage = bdot_voltage.value
    loop_area = loop_area.value
    time_array = time_array.value
    bdot_voltage /= (loop_area*num_loop*gain)
    magnetic_field_array = sp.cumtrapz(bdot_voltage, time_array, initial=0)  # Tesla

    return magnetic_field_array * u.T