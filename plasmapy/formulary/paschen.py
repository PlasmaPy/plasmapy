"""Functionality for determining the Paschen breakdown voltage """
__all__ = ["get_paschen_constants", "breakdown_voltage", "minimum_breakdown_voltage"]


import numpy as np


def get_paschen_constants(gas, electrode):

    r"""

     Return the constants A and B and the second Townsend coefficient to calculate the Paschen breakdown voltage.


     Parameters
     ----------

     gas :        `str`
          The gas for which to find the coefficients.

     electrode :  `str`
          The electrode material.

     Return
     ------

     Dictionary containing the constants ``A``, ``B`` and ``townsend_gamma`` for calculation
     of the breakdown voltage.



     References
     ---------
     Paschen_constants contains the coefficents A and B  for the estimation of the
     First Townsend Ionization Coefficent
     (exponential fit to the First Townsend Ionization coefficient)
     as adapted from
     E.Nasser, Fundamentals of Gaseous Ionization and Plasma Electronics,
     Wiley-Interscience, New York 1971

     format: paschen_constants dir {"gas":[A,B]}
     units: A in [Ionisation/(Pa m)] and B in [V/(Pa m)]


     `townsend_gamma` is the Second Townsend Ionization coefficient as given by
     A.Beroual and I. Fonfana, Discharge in Long Air Gap Modeling and Application
     IOP Publishing Ltd 2016
     ISBN 978-0-7503-1236-3 (ebook)
     ISBN 978-0-7503-1237-0 (print)



     Examples
     --------

     >>> get_paschen_constants("Ar", "Ni")
     {'A': 11, 'B': 135, 'gam': 0.058}

     If electrode material is not found a default value of 0.01 is taken

     >>> get_paschen_constants("Ar", "zz")
     {'A': 11, 'B': 135, 'gam': 0.01}

     If ``gas`` is not found, `None` is returned.

     >>> get_paschen_constants("Zz", "Ni")


 """

    # Supported gases

    gases = ["Air", "N2", "H2", "He", "Ne", "Ar", "Kr", "Xe"]

    paschen_constants = {
        "Air": [11, 274],
        "N2": [9.0, 257],
        "H2": [3.8, 104],
        "N2": [9.0, 257],
        "He": [2.3, 26],
        "Ne": [3.0, 75],
        "Ar": [11, 135],
        "Kr": [13, 180],
        "Xe": [20, 263],
    }

    # Supported electrode materials

    materials = ["Al", "Cu", "Ni", "Pt", "C", "W", "Fe"]

    townsend_gamma = {
        "Air": {
            "Al": 0.035,
            "Cu": 0.025,
            "Ni": 0.036,
            "Pt": 0.017,
            "C": None,
            "W": None,
            "Fe": 0.02,
        },
        "N2": {
            "Al": 0.1,
            "Cu": 0.066,
            "Ni": 0.077,
            "Pt": 0.59,
            "C": None,
            "W": None,
            "Fe": 0.059,
        },
        "H2": {
            "Al": 0.095,
            "Cu": 0.05,
            "Ni": 0.053,
            "Pt": 0.02,
            "C": 0.014,
            "W": None,
            "Fe": 0.061,
        },
        "He": {
            "Al": 0.021,
            "Cu": None,
            "Ni": 0.015,
            "Pt": 0.01,
            "C": None,
            "W": None,
            "Fe": 0.015,
        },
        "Ne": {
            "Al": 0.053,
            "Cu": 0.02,
            "Ni": 0.031,
            "Pt": 0.023,
            "C": None,
            "W": 0.045,
            "Fe": 0.022,
        },
        "Ar": {
            "Al": 0.12,
            "Cu": 0.058,
            "Ni": 0.058,
            "Pt": 0.058,
            "C": None,
            "W": None,
            "Fe": 0.058,
        },
        "Kr": {
            "Al": None,
            "Cu": None,
            "Ni": None,
            "Pt": None,
            "C": None,
            "W": None,
            "Fe": None,
        },
        "Xe": {
            "Al": None,
            "Cu": None,
            "Ni": None,
            "Pt": None,
            "C": None,
            "W": None,
            "Fe": None,
        },
    }

    # Check if the asked gas and electrode material is supported
    resg = gas in gases
    rese = electrode in materials

    # If the gas is supported get the constants A and B
    if resg:
        A = paschen_constants[gas][0]
        B = paschen_constants[gas][1]

        #   Get the townsend_gamma coefficient for the the gas/electrode combination
        if rese:
            gam = townsend_gamma[gas]
            gn = gam[electrode]
            #       Test if townsend_gamma exists for the demanded gas/electrode configuration
            #       If not a default townsend_gamma value of 0.01 is taken

            if gn is None:
                gn = 0.01
        else:

            #       If the electrode material is not supportes set townsend_gamma to default = 0.01
            gn = 0.01

        #   Create output dir {const}
        const = {"A": A, "B": B, "gam": gn}
        return const

    # If gas is not supported set const=None
    else:
        const = None
        return const


def breakdown_voltage(distance, pressure, A, B, gam):
    r"""
    Calculate the breakdown voltage V according to the Paschen law

                         𝑉=𝐵𝑝𝑑/𝑙𝑛(𝐴𝑝𝑑/𝑙𝑛(1+1/𝛾))


    Parameters
    ----------

    distance:  floating
    electrode distance

    pressure:  list
    Gas pressure

    A:         floating
    Paschen constant A

    B:         floating
    Paschen constant B

    gam:       floating
    Second Townsend coefficient

    The parameters A,B and gam can be obtain for some typical gases and electrode materials
    from the function get_paschen_constants, other values can be also be introduced


    Return
    ------
    [pd,breakdown_voltage] :  list

    [p*d value, Paschen breakdown voltage]


    Examples
    --------

    >>> breakdown_voltage(0.1,[100,200],11,135,0.058)
    [(10.0, 371.44065672260245), (20.0, 623.8960061470598)]

    """

    #  Calculate breakdown voltage according to the Paschen law
    x = list(distance * p for p in pressure)
    g = np.log(1 + (1 / gam))
    vb = list((pd, B * pd / (np.log((A * pd) / g))) for pd in x)
    return vb


def minimum_breakdown_voltage(A, B, gam):

    r"""
    Calculate the minimum breakdown voltage and the corresponding pd value from the Paschen law

    Parameters
    ----------
    A:     floating
    B:     floating
    gam:   floating

    Return
    ------
    vmin:  floating
    Minimum breakdown voltage

    pdmin: floating
    pd value of the minimum breakdown voltage

    Example
    -------
    >>> minimum_breakdown_voltage(11,135,0.058)
    (96.85926603680262, 0.7174760447170564)

    """
    # Calculate vmin and pdmin
    g = np.log(1 + (1 / gam))
    vmin = 2.718 * (B / A) * g
    pdmin = 2.718 * (g / A)
    return (vmin, pdmin)
