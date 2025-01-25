import configparser
import math
import numpy
import os
import pickle

from scipy import integrate

from plasmapy.reaction_rates import cross_section
from plasmapy.reaction_rates.config import config
from plasmapy.reaction_rates.exceptions import (
    ReactivityTemperatureTooHighError,
    ReactivityTemperatureTooLowError,
)


def reactivity(T_i, reaction, method="parameterized", zero_low_temperatures=False):
    """Return the fusion reactivity <σv> of a given fusion reaction in m^3/s

    Keyword arguments:
    T_i -- ion temperature in keV
    reaction -- string identifying the reaction. Supported reactions are:
                From Bosch and Hale 1992
                "T(d,n)4He": D + T --> n + α
                "D(d,p)T": D + D --> p + T
                "D(d,n)3He": D + D --> n + 3He
                "3He(d,p)4He": D + 3He --> p + α

                From Nevins and Swain 2000
                "11B(p,4He)4He4He": p + 11B --> 3α
    method -- 'parameterized' or 'integrated'.

             'parameterized' uses a fast parameterization from the literature.
              Note the limited temperature ranges of the parameterized
              reactivities in reactions.ini which are enforced here.

              'integrated' *Experimental* uses the cross section parameterization to
              numerically integrate the average over a maxwellian velocity
              distribution and is quite slow. However the cached vales are fast
              to retreive. Integrated is not supported for p-11B because the energy
              range of the cross section may not be wide enough. Note that at
              low and high temperatures this may give incorrect results due to
              limited cross section data.

    zero_low_temperatures -- Boolean. Return a reactivity of 0 if the requested temperature
                             is below the supported minimum temperature of the reactivity
                             parameterization. This is needed when integrating
                             over temperature profiles where the temperature drops to zero near
                             the edge. For example, calculating $lambda_F$ over a parabolic
                             temeprature profile.

    #TODO Consider adding Sikora Weller 2016 for updated 11B(p,4He)4He4He reactivity
    """
    # Enforce only parameterized reactivity for p11b
    if reaction == "11B(p,4He)4He4He" and method == "integrated":
        raise ValueError(
            "11B(p,4He)4He4He does not support integrated reactivity, only parameterized"
        )

    # Calculate reactivity
    try:
        if reaction in ["T(d,n)4He", "D(d,p)T", "D(d,n)3He", "3He(d,p)4He"]:
            if method == "parameterized":
                sigma_v = _reactivity_bosch_hale(T_i, reaction)
            elif method == "integrated":
                sigma_v = _reactivity_integrated(T_i, reaction)
        elif reaction == "11B(p,4He)4He4He":
            sigma_v = _reactivity_nevins_swain_pb11(T_i)
        else:
            raise ValueError("reaction %s not recognized" % reaction)
    except ReactivityTemperatureTooLowError as ex:
        if zero_low_temperatures == True:
            sigma_v = 0
        else:
            raise ex
    return sigma_v


def _reactivity_integrated(T_i, reaction):
    """Return the reactivity in m^3/s by performing numerical integral to
    average σv over a maxwellian velocity distribution.

    There are no hard range checks on the ion temperature since the integration
    goes from the minimum to maximum center of mass energies for which the
    cross sections are valid and that range is quite wide.
    #TODO However, some thought should go into what the appropriate allowed ion
    temperature range here.

    Keyword arguments:
    T_i -- ion temperature
    reaction -- string identifying the reaction
    """
    coefficients = config[reaction]
    c = 299792458  # m/s
    m1 = float(coefficients["m1_keV"]) / (c * c)
    m2 = float(coefficients["m2_keV"]) / (c * c)
    mu = (m1 * m2) / (m1 + m2)  # reduced mass
    # NOTE 1:
    # The cross section MUST be integrated in units of millibarns (or simillar
    # order of magnitude). If it's in units of m^2 the integrator accumulates
    # small floating point errors and gives an incorrect result.

    # NOTE 2:
    # The cross sections given by Bosch and Hale and Nevins and Swain
    # are center of mass cross sections. To get the thermal reactivities we
    # I've transformed Eq. 1.3.2 of Wesson (p.6) from the frame of the
    # bombarding particle to the CM frame with the intention of comparing it to
    # Atzeni and Meyer-Ter-Vehn Eq. 1.50 (p. 15). Unfortunately, Atzeni and
    # Meyer-Ter-Vehn Eq. 1.50 (p. 15) includes an incorrect factor of pi.

    min_E_keV = cross_section.cross_section_energy_ranges[reaction][0]
    max_E_keV = cross_section.cross_section_energy_ranges[reaction][1]

    # Evaluate integral expression from the limits of the known cross sections
    integral = integrate.quad(
        lambda E: E
        * cross_section.cross_section_cm(E, reaction)
        * math.exp((-1 * E) / (T_i)),
        min_E_keV,
        max_E_keV,
    )
    # Calculate reactivity after transforming Wesson 1.3.2 to CM frame
    reactivity = (4 / ((2 * math.pi * mu) ** (0.5))) * ((T_i) ** (-1.5)) * integral[0]

    # Incorrect Atzeni Eq. 1.50 (p. 15), note additional, incorrect factor of pi
    # reactivity = ((4*math.pi)/((2*math.pi*mu)**(0.5))) * ((T_i)**(-1.5)) * integral[0]

    # finally convert milibarns to m^2
    reactivity = reactivity * 1e-28 * 1e-3
    return reactivity


def _reactivity_nevins_swain_pb11(T_i):
    """Return the reactivity <σv> of p-B11 fusion in units of m^3/s
    based on Nevins and Swain 2000's parameterization.

    T_i - Effective ion temperature

    #TODO currently only the high temperature portion of the parameterization
    works. The low temperature portion gives incorrect results likely due to an
    error in the reduced mass Mr or some other error. It is not currently used
    in calculations.
    """
    coefficients = config["11B(p,4He)4He4He"]
    C0 = coefficients.getfloat("C0")
    C1 = coefficients.getfloat("C1")
    C2 = coefficients.getfloat("C2")
    EG = coefficients.getfloat("EG")
    Mrc2 = coefficients.getfloat("Mrc2")

    P1 = coefficients.getfloat("P1")
    P2 = coefficients.getfloat("P2")
    P3 = coefficients.getfloat("P3")
    P4 = coefficients.getfloat("P4")
    P5 = coefficients.getfloat("P5")
    P6 = coefficients.getfloat("P6")
    P7 = coefficients.getfloat("P7")

    c = 299792458  # m/s

    # Enforce maximum and minimum temperature values specified implied by
    # Nevis and Swain and recorded in reactions.ini
    min_T_i = coefficients.getfloat("min_Ti_keV_reactivity")
    max_T_i = coefficients.getfloat("max_Ti_keV_reactivity")

    if T_i < min_T_i:
        raise ReactivityTemperatureTooLowError(
            reaction="11B(p,4He)4He4He",
            temperature=T_i,
            allowed_range=[min_T_i, max_T_i],
        )
    elif T_i > max_T_i:
        raise ReactivityTemperatureTooHighError(
            reaction="11B(p,4He)4He4He",
            temperature=T_i,
            allowed_range=[min_T_i, max_T_i],
        )

    # In the paper, reactivity is the sum of the non resonant and
    # resonant component.
    # <σv> = <σv>NR + <σv>R
    # The non resonant component is different at high and low temps
    # The resonant component is the same for both

    # <σv>R
    R = 5.41e-21 * T_i ** (-3 / 2) * math.exp(-148 / T_i)

    # <σv>NR
    # TODO properly handle low vs high temp ranges
    range = "high"

    # Low T
    # Here we have to convert EG from (MeV) to (keV)
    E0 = ((EG * 1000) / 4.0) ** (1.0 / 3.0) * T_i ** (2.0 / 3.0)
    DE0 = 4 * (T_i * E0 / 3) ** (0.5)
    tau = (3 * E0) / T_i
    # Here we have to convert Seff from (MeV b) to (keV b)
    Seff = 1000 * (
        C0 * (1 + 5 / (12 * tau))
        + C1 * (E0 + (35 / 36) * T_i)
        + C2 * (E0**2 + (89 / 36) * E0 * T_i)
    )
    c = 3e8  # m/s
    NR_low_T = (
        (2 * T_i / (Mrc2 / c**2)) ** (0.5)
        * (DE0 * Seff / (T_i**2))
        * math.exp(-1 * tau)
    )

    # High T
    theta = T_i / (
        1.0
        - (
            (T_i * (P2 + (T_i * (P4 + (T_i * P6)))))
            / (1.0 + T_i * (P3 + (T_i * (P5 + (T_i * P7)))))
        )
    )

    xi = (EG * 1000 / (4 * theta)) ** (1 / 3)

    NR_high_T = P1 * theta * (xi / (Mrc2 * T_i**3)) ** (0.5) * math.exp(-3 * xi)

    if range == "low":
        return NR_low_T + R
    if range == "high":
        return NR_high_T + R


def _reactivity_bosch_hale(T_i, reaction):
    """Return the parameterized reactivity in units of m^3 s^-1.

    This function uses the paremterized of fusion reactivities <sigma v>
    assuming maxwellian distribution of velocities from Bosch and Hale,
    'Improved Formulas For Fusion Cross-Sections and Thermal Reactivities'
    Nuclear Fusion Vol 32 No. 4 (1992).

    Note that these parameterized reactivities have very limited temperature
    ranges which are detailed in the reactions.ini file.
    To utilize higher maximum temperetures use the reactivity function with
    method='integrated' which numerically integrates the cross section data.

    Keyword arguments:
    T_i -- ion temperature in keV
    reaction -- string identifying the reaction

    Supported values of reaction are:
    'T(d,n)4He'
    '3He(d,p)4He'
    'D(d,p)T'
    'D(d,n)3He'
    """
    coefficients = config[reaction]

    # Check to ensure that temperature is in range of parameterizaiton
    min_T_i = coefficients.getfloat("min_Ti_keV_reactivity")
    max_T_i = coefficients.getfloat("max_Ti_keV_reactivity")
    if T_i < min_T_i:
        raise ReactivityTemperatureTooLowError(
            reaction=reaction, temperature=T_i, allowed_range=[min_T_i, max_T_i]
        )
    elif T_i > max_T_i:
        raise ReactivityTemperatureTooHighError(
            reaction=reaction, temperature=T_i, allowed_range=[min_T_i, max_T_i]
        )

    # Assign coeficients
    BG = float(coefficients["BG"])
    mcsquared = float(coefficients["mcsquared"])
    C1 = float(coefficients["C1"])
    C2 = float(coefficients["C2"])
    C3 = float(coefficients["C3"])
    C4 = float(coefficients["C4"])
    C5 = float(coefficients["C5"])
    C6 = float(coefficients["C6"])
    C7 = float(coefficients["C7"])

    # Calculate reactivity per Equations 12, 13, and 14 from
    # Bosch and Hale 1992
    theta = T_i / (
        1.0
        - (
            (T_i * (C2 + (T_i * (C4 + (T_i * C6)))))
            / (1.0 + T_i * (C3 + (T_i * (C5 + (T_i * C7)))))
        )
    )
    xi = ((BG**2) / (4 * theta)) ** (1.0 / 3.0)
    r_cm3_s = (
        C1 * theta * math.sqrt(xi / (mcsquared * (T_i**3))) * math.exp(-3.0 * xi)
    )
    # As is, this is in cm^3/s so convert to m^3/s
    r_m3_s = r_cm3_s * 1.0e-6
    return r_m3_s
