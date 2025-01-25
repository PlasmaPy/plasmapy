import math

from plasmapy.reaction_rates.config import config
from plasmapy.reaction_rates.exceptions import CrossSectionEnergyRangeError

cross_section_energy_ranges = {
    "T(d,n)4He": [
        config["T(d,n)4He"].getfloat("min_E_keV_xsec"),
        config["T(d,n)4He_HIGH_ENERGY"].getfloat("max_E_keV_xsec"),
    ],
    "3He(d,p)4He": [
        config["3He(d,p)4He"].getfloat("min_E_keV_xsec"),
        config["3He(d,p)4He_HIGH_ENERGY"].getfloat("max_E_keV_xsec"),
    ],
    "D(d,p)T": [
        config["D(d,p)T"].getfloat("min_E_keV_xsec"),
        config["D(d,p)T"].getfloat("max_E_keV_xsec"),
    ],
    "D(d,n)3He": [
        config["D(d,n)3He"].getfloat("min_E_keV_xsec"),
        config["D(d,n)3He"].getfloat("max_E_keV_xsec"),
    ],
    "11B(p,4He)4He4He": [
        config["11B(p,4He)4He4He"].getfloat("min_E_keV_xsec"),
        config["11B(p,4He)4He4He"].getfloat("max_E_keV_xsec"),
    ],
}


def cross_section_cm(E, reaction):
    """Return the center of mass cross section in millibarns.

    Keyword arguments:
    E -- Center of mass frame energy in keV
    reaction -- string identifying the reaction. Supported reactions are:
                From Bosch and Hale 1992
                "T(d,n)4He": D + T --> n + α
                "D(d,p)T": D + D --> p + T
                "D(d,n)3He": D + D --> n + 3He
                "3He(d,p)4He": D + 3He --> p + α

                From Nevins and Swain 2000
                "11B(p,4He)4He4He": p + 11B --> 3α

    The CM cross section is used to calculate reactivity in `reactivity.py` by
    averaging over a maxwellian velocity distribution.

    The reason this function return milibarns and not barns (or m^2) is to
    reduce accumulated errors when performing numerical integration to calculate
    the reactivity due to additions of many very small floating point numbers.
    You really don't want to change this to bars and especially not m^2.

    Potential #TODO would be to add Sikora Weller 2016 for updated
    11B(p,4He)4He4He cross section
    """
    if reaction in ["T(d,n)4He", "D(d,p)T", "D(d,n)3He", "3He(d,p)4He"]:
        sigma = _cross_section_cm_bosch_hale(E, reaction)
    elif reaction == "11B(p,4He)4He4He":
        sigma = _cross_section_cm_nevins_swain_pb11(E)
    else:
        raise ValueError("reaction %s not recognized" % reaction)
    return sigma


def _cross_section_cm_bosch_hale(E, reaction):
    """Private function returning reaction cross section in milibarns (mb) as a
    function of CM energy based on Bosch and Hale 1992. Enforces maximum and
    minimum energy values specified by Boch and Hale and recorded in
    reactions.ini

    Keyword arguments:
    E -- Center of mass frame energy in keV
    reaction -- string identifying the reaction. Supported reactions are:
                "T(d,n)4He": D + T --> n + α
                "D(d,p)T": D + D --> p + T
                "D(d,n)3He": D + D --> n + 3He
                "3He(d,p)4He": D + 3He --> p + α
    """
    allowed_range = cross_section_energy_ranges[reaction]
    if not allowed_range[0] <= E <= allowed_range[1]:
        raise CrossSectionEnergyRangeError(
            reaction=reaction, energy=E, allowed_range=allowed_range
        )

    # Bosch and Hale recommends a cutover at 530 keV for high energy D-T
    # CM cross section. See page 622 top of right column of Bosch and Hale.
    if reaction == "T(d,n)4He" and E >= 530:
        reaction = "T(d,n)4He_HIGH_ENERGY"

    # Bosch and Hale recommends a cutover at 900 keV for high energy D-3He
    # CM cross section. See page 622 top of right column.
    if reaction == "3He(d,p)4He" and E >= 900:
        reaction = "3He(d,p)4He_HIGH_ENERGY"

    coefficients = config[reaction]

    BG = float(coefficients["BG"])
    A1 = float(coefficients["A1"])
    A2 = float(coefficients["A2"])
    A3 = float(coefficients["A3"])
    A4 = float(coefficients["A4"])
    A5 = float(coefficients["A5"])
    B1 = float(coefficients["B1"])
    B2 = float(coefficients["B2"])
    B3 = float(coefficients["B3"])
    B4 = float(coefficients["B4"])

    # First evalueate "Astrophysical factor" S
    S = (A1 + E * (A2 + E * (A3 + E * (A4 + (E * A5))))) / (
        1 + E * (B1 + E * (B2 + E * (B3 + E * B4)))
    )

    # Evaluate CM cross section
    sigma = (1.0 / E) * S * math.exp((-1.0 * BG) / (E ** (0.5)))  # millibarns

    return sigma


def _cross_section_cm_nevins_swain_pb11(E):
    """Private function returning cross section in milibarns (mb) of pb11
    reaction as a function of CM energy based on Nevins and Swain 2000.
    Enforces maximum and minimum energy values specified by Nevins and Swain
    and recorded in reactions.ini

    Keyword arguments:
    E -- Center of mass frame energy in keV
    """
    # Return the p-B11 cross section in units of millibarns (mb)
    # Use the cross section parameterization from Nevins Swain 2000

    coefficients = config["11B(p,4He)4He4He"]
    C0 = float(coefficients["C0"])
    C1 = float(coefficients["C1"])
    C2 = float(coefficients["C2"])
    AL = float(coefficients["AL"])
    EL = float(coefficients["EL"])
    D0 = float(coefficients["D0"])
    D1 = float(coefficients["D1"])
    D2 = float(coefficients["D2"])
    D5 = float(coefficients["D5"])

    delEL = float(coefficients["delEL"])

    A0 = coefficients.getfloat("A0")
    A1 = coefficients.getfloat("A1")
    A2 = coefficients.getfloat("A2")
    A3 = coefficients.getfloat("A3")

    E0 = coefficients.getfloat("E0")
    E1 = coefficients.getfloat("E1")
    E2 = coefficients.getfloat("E2")
    E3 = coefficients.getfloat("E3")

    delE0 = coefficients.getfloat("delE0")
    delE1 = coefficients.getfloat("delE1")
    delE2 = coefficients.getfloat("delE2")
    delE3 = coefficients.getfloat("delE3")

    B = coefficients.getfloat("B")

    EG = float(coefficients["EG"])

    # Enforce maximum and minimum energy values specified by Boch and Hale
    # and recorded in reactions.ini
    reaction = "11B(p,4He)4He4He"
    allowed_range = cross_section_energy_ranges[reaction]
    if not allowed_range[0] <= E <= allowed_range[1]:
        raise ValueError(
            f"{E} keV is out of range. Supported range for {reaction} is {cross_section_energy_ranges[reaction][0]} to {cross_section_energy_ranges[reaction][1]} keV."
        )

    # First calculate astrophysical factor S depending on the energy range
    S = 0
    if E <= 400:  # keV
        S = C0 + C1 * E + C2 * (E**2) + AL / ((E - EL) ** 2 + (delEL) ** 2)
    elif 400 < E < 642:  # keV
        S = (
            D0
            + D1 * ((E - 400) / 100)
            + D2 * ((E - 400) / 100) ** 2
            + D5 * ((E - 400) / 100) ** 5
        )
    elif E >= 642:
        S = (
            A0 / ((E - E0) ** 2 + (delE0) ** 2)
            + A1 / ((E - E1) ** 2 + (delE1) ** 2)
            + A2 / ((E - E2) ** 2 + (delE2) ** 2)
            + A3 / ((E - E3) ** 2 + (delE3) ** 2)
            + B
        )
    # Next calculate CM cross section sigma
    # Note that EG is in units of MeV.
    # We convert astrophysical factor from MeV to keV
    # And EG from MeV to keV since E is assumed in keV.
    sigma = ((S * 1000) / E) * math.exp(-1.0 * ((EG * 1000) / E) ** (0.5))
    # Convert to millibarns for consistency with other functions
    # in this library
    sigma = 1000 * sigma
    return sigma
