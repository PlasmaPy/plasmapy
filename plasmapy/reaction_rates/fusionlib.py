import configparser
import math
import numpy
import sys

from math import exp
from scipy import integrate

from plasmapy.reaction_rates import reactivity

# Constants
# Bremstrahlung Constant:
# From 2004 NRL Plasma Formulary p. 58 Eq. (30) Cbr = 1.69e-32 cm^3 J eV^-0.5 s^-1
Cbr = 1.69e-32
Cbr = Cbr * 1e-6  # Convert cm^3 --> m^3
Cbr = Cbr * 6.24e18  # Convert J --> eV
Cbr = Cbr / math.sqrt(1000)  # Convert ev^0.5 to keV^0.5

# Import the reactivity coefficients from config file
reactions_config = configparser.ConfigParser()

# Get config values once at import time
reactions_config.read("config/reactions.ini")


def optimal_mix_triple_product(
    T_i,
    reaction,
    Q_fuel,
    xi=1,
    density="ion",
    bremsstrahlung_frac=1,
    relativistic_mode="nonrelativistic",
    method="integrated",
):

    tp = T_i * optimal_mix_lawson_parameter(
        T_i=T_i,
        reaction=reaction,
        Q_fuel=Q_fuel,
        xi=xi,
        density=density,
        bremsstrahlung_frac=bremsstrahlung_frac,
        relativistic_mode=relativistic_mode,
        method=method,
    )
    return tp


def optimal_mix_lawson_parameter(
    T_i,
    reaction,
    Q_fuel,
    xi=1,
    density="ion",
    bremsstrahlung_frac=1,
    relativistic_mode="nonrelativistic",
    method="integrated",
):
    """Return the required confinement parameter n_i * tau_E*
    necessary to achieve a particular Q given at any given value of Ti
    and value of xi where xi = Te/Ti given a single reaction.

    Assumes optimal mix of reactants: k1 = 1/(2*Z1) and k2 = 1/(2*Z2),
    where k1 is relative fraction of ion species 1 (k1 = n1/ne) and
    where k2 is relative fraction of ion species 2 (k2 = n2/ne).

    Arguments
    Ti -- ion temperature
    reaction -- string identifying the reaction. Supported reactions are:
                From Bosch and Hale 1992
                "T(d,n)4He": D + T --> n + α
                "D(d,p)T": D + D --> p + T
                "D(d,n)3He": D + D --> n + 3He
                "3He(d,p)4He": D + 3He --> p + α

                From Nevins and Swain 2000
                "11B(p,4He)4He4He": p + 11B --> 3α

                Note that choosing a single D-D reactions, while allowed,
                will give a nonphysical result since both reactions occur
                with roughly 50/50 probability. See the catalyzed_DD functions
                below.

    Q_fuel -- Fuel gain P_F / P_E
    xi -- Allow for different ion and electron temperatures. xi = Te/Ti.
    density -- Default is 'ion'. Changes the returned density in the triple
               product. The default returns 'n_i T_i tau_E'. Setting
               density = 'electron' returns 'n_e T_i tau_E'. This has a
               relatively minor effect when dealing with non hydrogen isotope
               plasmas.
    bremsstrahlung_frac -- Allows for arbitrary altering of the bremsstrahlung
                           constant Cbr, by replacing Cbr with
                           bremsstrahlung_frac * Cbr. Typically used to
                           demonstrate the effect of reduced bremsstrahlung
                           losses due to an optically thick plasma.
                           Default is 1.
    relativistic_mode -- Options for including relativistic bremsstraulung.
                         One of 'putvinski', 'ryder', or 'nonrelativistic'

    """
    # Collect parameters of specified reaction
    reaction_info = reactions_config[reaction]
    Ef = float(reaction_info["energy_per_reaction_total_keV"])
    fraction_charged = float(reaction_info["fraction_charged"])
    Z1 = reaction_info.getfloat("Z1")
    Z2 = reaction_info.getfloat("Z2")
    are_identical_particles = reaction_info.getboolean("are_identical_particles")

    # Evaluate kronecker delta
    if are_identical_particles is True:
        kronecker_delta = 1
    elif are_identical_particles is False:
        kronecker_delta = 0
    else:
        raise ValueError

    # Calculate optimal relative ion fractions
    # Maximum fusion power at constant electron density
    k1 = 1 / (2 * Z1)
    k2 = 1 / (2 * Z2)

    # TODO show that under optimal mix, Zeff is simply (Z1 + Z2)/2
    # Evaluate Zeff
    Zeff = (Z1 + Z2) / 2

    # Calculate triple product
    numerator = (3 / 2) * (k1 + k2 + xi) * T_i
    # Made adjustment for the ion density as part of the triple product
    if density == "ion":
        numerator = numerator * (k1 + k2)
    elif density == "electron":
        pass
    else:
        raise (ValueError)

    denominator = (
        (fraction_charged + (1.0 / Q_fuel))
        * ((k1 * k2) / (1 + kronecker_delta))
        * Ef
        * reactivity.reactivity(T_i, reaction, method)
    ) - (
        bremsstrahlung_frac
        * Cbr
        * ((xi * T_i) ** 0.5)
        * relativistic_bremsstrahlung_correction(
            xi * T_i, Zeff, relativistic_mode=relativistic_mode
        )
    )
    # print(f'{denominator}')
    # Need to avoid divide by zero and nonphysical negative values of
    # confinement parameter when bremstrahlung power is greater than
    # fusion power. In this case we return infinity.
    if denominator > 0:
        triple_product = numerator / denominator
    else:
        triple_product = float("inf")

    return triple_product


def DT_adjusted_triple_product(
    T_i,
    Q_avg,
    lambda_F,
    lambda_B,
    lambda_kappa,
    xi,
    Zeff=1,
    Zbar=1,
    brem_frac=1,
    Pin_frac=1,
):
    """Return the required triple product n T tau_E in units of m^-3 keV s

    Keyword Arguments:
    T_i -- Ion temperature
    Q_avg -- The adjusted volume average Q for given lambda parameters
    lambda_F -- Fusion power density correciton factor
    lambda_B -- Bremsstrahlung power density correction factor
    lambda_kappa -- Thermal conduction power density correction factor
    xi -- Allow for different ion and electron temperatures. xi = Te/Ti.
    Zeff -- Zeff
    Zbar -- ne/ni
    brem_frac -- Multiplier on Cb. Set to zero for ICF where bremsstrahlung losses are zero
    Pin_frac -- Fraction of externally applied power which is absorbed by fuel. 1 for MCF, much smaller for ICF.
    """
    # Get reaction information
    reaction = "T(d,n)4He"
    reaction_info = reactions_config[reaction]

    Ef = float(reaction_info["energy_per_reaction_total_keV"])
    fraction_charged = float(reaction_info["fraction_charged"])
    denominator = (
        lambda_F
        * (fraction_charged + ((Q_avg / Pin_frac) ** (-1)))
        * reactivity.reactivity(T_i, reaction)
        * Ef
        * (1 / 4)
        - lambda_B * brem_frac * Cbr * Zeff * Zbar**2 * (xi * T_i) ** 0.5
    )
    # Need to avoid divide by zero and nonphysical negative values of
    # triple product when bremstrahlung power is greater than fusion power
    if denominator > 0:
        adjusted_tp = (
            (3 / 2) * lambda_kappa * T_i**2 * (1 + (Zbar * xi)) / denominator
        )
    else:
        adjusted_tp = float("inf")
    return adjusted_tp


def DT_adjusted_lawson_parameter(
    T_i0,
    Q_fuel,
    lambda_F,
    lambda_B,
    lambda_kappa,
    xi,
    Z_eff=1,
    Z_bar=1,
    brem_frac=1,
    self_heating=True,
):
    """Return the required triple product n T tau_E in units of m^-3 keV s

    Keyword Arguments:
    T_i0 -- Peak ion temperature
    Q_fuel -- Fusion power density / P_in
    lambda_F -- Fusion power density correciton factor
    lambda_B -- Bremsstrahlung power density correction factor
    lambda_kappa -- Thermal conduction power density correction factor
    xi -- Ratio of electron temperature to ion temperature i.e., Te/Ti.
    Z_eff -- Z_eff
    Z_bar -- ne/ni
    brem_frac -- Multiplier on Cb. Set to zero for ICF where bremsstrahlung losses are zero. Set to 1 for MCF.
    """
    # Get reaction information for D-T
    reaction = "T(d,n)4He"
    reaction_info = reactions_config[reaction]

    Ef = float(reaction_info["energy_per_reaction_total_keV"])
    if self_heating == True:
        fraction_charged = float(reaction_info["fraction_charged"])
    elif self_heating == False:
        fraction_charged = 0
    else:
        raise ValueError
    denominator = (
        lambda_F
        * (fraction_charged + ((Q_fuel) ** (-1)))
        * reactivity.reactivity(T_i0, reaction)
        * Ef
        * (1 / 4)
        - lambda_B * brem_frac * Cbr * Z_eff * Z_bar**2 * (xi * T_i0) ** 0.5
    )
    # Need to avoid divide by zero and nonphysical negative values of
    # triple product when bremstrahlung power is greater than fusion power
    if denominator > 0:
        adjusted_confinement_parameter = (
            (3 / 2) * lambda_kappa * T_i0 * (1 + (Z_bar * xi)) / denominator
        )
    else:
        adjusted_confinement_parameter = float("inf")
    return adjusted_confinement_parameter


def catalyzed_DD_triple_product(
    T_i, Q_fuel, xi=1, density="ion", relativistic_mode="putvinski", method="integrated"
):
    tp = T_i * catalyzed_DD_lawson_parameter(
        T_i=T_i,
        Q_fuel=Q_fuel,
        xi=xi,
        density=density,
        relativistic_mode=relativistic_mode,
        method=method,
    )
    return tp


def catalyzed_DD_lawson_parameter(
    T_i, Q_fuel, xi=1, density="ion", relativistic_mode="putvinski", method="integrated"
):
    """Return the required triple product necessary to achieve a particular
    Qfuel given at any given value of ion temperature T_i for the
    catalyzed D-D reaction.
    In steady state the catalyzed D-D reaction involves both branches of the
    D + D reaction and their subsequent follow on reactions as illustrated
    below:

    D + D --> T + p
              |
              v
          D + T --> α + n

    D + D --> 3He + n
               |
               v
          D + 3He --> p + α

    Assumptions:
    1) Te = xi Ti (electron and ion temperature may be different)
    2) In steady state subsequent D-3He and D-T reactions occur at the same
       rate as their respective parent D-D reactions. This sets exact ratios
       on the equilibrium concentrations of D, 3He, and T.
    3) Alpha particle and proton ash is immediately removed and has zero
       equilibrium concentration.

    Calculations:
    See Sam's paper calculations done 10 April 2021 and 29 December 2021
    for explaination of the various factors in the result.

    Arguments
    T_i -- Ion temperature
    Qfuel -- Fuel gain P_F / P_abs (ratio of fusion power to absorbed heating
             power)
    density -- Default is 'ion'. Changes the returned density in the lawson
               parameter. The default returns 'ni tau_E'. Setting
               density = 'electron' returns 'ne tau_E'.
    xi -- Ratio of electron temperature to ion temperature i.e., Te/Ti.
    density -- Default is 'ion'. Changes the returned density in the triple
               product. The default returns 'n_i T_i tau_E'. Setting
               density = 'electron' returns 'n_e T_i tau_E'. This has a
               relatively minor effect when dealing with non hydrogen isotope
               plasmas.
    relativistic_mode -- Options for including relativistic bremsstraulung.
                         One of 'putvinski', 'ryder', or 'nonrelativistic'
    """
    # Get the relevant reactivities

    DD_T_reactivity = reactivity.reactivity(T_i, "D(d,p)T", method=method)
    DD_T_reaction_info = reactions_config["D(d,p)T"]
    DD_T_E_total = DD_T_reaction_info.getfloat("energy_per_reaction_total_keV")

    DT_reactivity = reactivity.reactivity(T_i, "T(d,n)4He", method=method)
    DT_reaction_info = reactions_config["T(d,n)4He"]
    DT_E_total = DT_reaction_info.getfloat("energy_per_reaction_total_keV")

    DD_3He_reactivity = reactivity.reactivity(T_i, "D(d,n)3He", method=method)
    DD_3He_reaction_info = reactions_config["D(d,n)3He"]
    DD_3He_E_total = DD_3He_reaction_info.getfloat("energy_per_reaction_total_keV")

    D3He_reactivity = reactivity.reactivity(T_i, "3He(d,p)4He", method=method)
    D3He_reaction_info = reactions_config["3He(d,p)4He"]
    D3He_E_total = D3He_reaction_info.getfloat("energy_per_reaction_total_keV")

    # The steady state number densities of 3He, and T, are related to the
    # D number densities by the following relationships:

    # n_3He = (1/2) * n_D * Xh and
    # n_T = (1/2) * n_D * Xt where,

    Xh = (1 / 2) * (DD_3He_reactivity / D3He_reactivity)
    Xt = (1 / 2) * (DD_T_reactivity / DT_reactivity)

    fraction_charged = 0.617

    Zeff = (1 + 4 * Xh + Xt) / (1 + 2 * Xh + Xt)

    # Calculate lawson parameter

    # Numerator depends on whether we want n_i T tau_E or n_e T tau_E
    if density == "electron":
        numerator = (
            (3 / 2)
            * ((xi + 1) + ((2 * xi) + 1) * Xh + (xi + 1) * Xt)
            * (1 + 2 * Xh + Xt)
            * T_i
        )
    elif density == "ion":
        numerator = (
            (3 / 2)
            * ((xi + 1) + ((2 * xi) + 1) * Xh + (xi + 1) * Xt)
            * (1 + Xh + Xt)
            * T_i
        )
    else:
        raise ValueError

    # TODO Note that the factor of 1/2 in the first term of the denominator IS consistent with the factor of
    # 1/4 in the paper, note the different reactivities that are used. Below is more precise but consider changing to
    # more closely match the paper's estimate of the equivilance of the two D-D reactivities.
    denominator = (fraction_charged + (1.0 / Q_fuel)) * (1 / 2) * (
        DD_T_reactivity * (DD_T_E_total + DT_E_total)
        + DD_3He_reactivity * (DD_3He_E_total + D3He_E_total)
    ) - (
        (1 + 2 * Xh + Xt) ** 2
        * Cbr
        * (xi * T_i) ** 0.5
        * relativistic_bremsstrahlung_correction(
            xi * T_i, Zeff, relativistic_mode=relativistic_mode
        )
    )

    # Need to avoid divide by zero and nonphysical negative values of
    # confinement parameter when bremstrahlung power is greater than
    # fusion power. In this case we return a Lawson parameter of infinity.
    if denominator > 0:
        triple_product = numerator / denominator
    else:
        triple_product = float("inf")

    return triple_product


def power_density_bremsstrahlung(
    T_e,
    n_e,
    Zeff=1,
    return_units="keV s^-1 m^-3",
    relativistic_mode="nonrelativistic",
):
    """Return the radiated bremsstrahlung power density in specified units

    Keyword arguments:
    T_e -- electron temperature in keV
    n_e -- electron number density
    Zeff -- Effective Z: Number of electrons per ion on average
    return_units -- Units of power density to return. One of 'keV s^-1 m^-3' or 'W/m^3'
    relativistic -- Include relativistic correction. One of 'putvinski', 'ryder', or 'nonrelativistic'.
    """
    if return_units == "W/m^3":
        # From 2004 NRL Plasma Formulary p. 58 Eq. (30) Cbr = 1.69e-32 cm^3 J eV^-0.5 s^-1
        Cb = 1.69e-32 * 1e-6 * math.sqrt(1000)
    elif return_units == "keV s^-1 m^-3":
        # From 2004 NRL Plasma Formulary p. 58 Eq. (30) Cbr = 1.69e-32 cm^3 J eV^-0.5 s^-1
        Cb = 1.69e-32
        Cb = Cb * 1e-6  # Convert cm^3 --> m^3
        Cb = Cb * 6.24e18  # Convert J --> eV
        Cb = Cb / math.sqrt(1000)  # Convert ev^0.5 to keV^0.5
    else:
        raise (ValueError)

    # Initial calculation of bremsstrahlung power density
    p = (
        Cb * n_e * n_e * (T_e**0.5)
    )  # Zeff is handled in next step for all cases (relativistic and non-relativistic)

    # This is simply p = p * Zeff if relativistic_mode = 'nonrelativistic'
    p = p * relativistic_bremsstrahlung_correction(
        T_e, Zeff, relativistic_mode=relativistic_mode
    )

    return p


def relativistic_bremsstrahlung_correction(
    T_e, Zeff, relativistic_mode="nonrelativistic"
):
    """Return the unitless correction to bremsstrahlung power density. Includes the Zeff term.

    Keyword arguments:
    T_e -- electron temperature in keV
    Zeff -- Effective Z: Number of electrons per ion on average
    mode -- Include relativistic correction. One of 'putvinski', 'ryder', or 'nonrelativistic'.
    """
    if relativistic_mode == "putvinski":
        # See Putvinski 2019 for details.
        t = T_e / 511  # T_e / m_e c^2
        correction = Zeff * (1 + 1.78 * t**1.34) + 2.12 * t * (
            1 + 1.1 * t + t**2 - 1.25 * t**2.5
        )
    elif relativistic_mode == "ryder":
        t = T_e / 511  # T_e / m_e c^2
        correction = Zeff * (1 + 0.7936 * t + 1.874 * t**2) + (3 / (2**0.5)) * t
    elif relativistic_mode == "nonrelativistic":
        correction = Zeff
    else:
        raise (ValueError)
    return correction


def Q_vs_T_and_ntau(T, ntau):
    """Return the fusion gain Q as a function of temperature T and Lawson
    parameter ntau assuming a pure, hydrogenic DT plasma. Used to recreate
    Lawson's original plots from his 1955 and 1957 papers to demonstrate
    the 'severe' conditions required for achieving a useful value of
    energy gain.

    Keyword arguments:
    T -- temperature in keV
    ntau -- Lawson parameter, product of density and confinement time
    """
    reaction_info = reactions_config["T(d,n)4He"]
    dt_energy_per_reaction_total_keV = float(
        reaction_info["energy_per_reaction_total_keV"]
    )

    Q = (
        reactivity.reactivity(T_i=T, reaction="T(d,n)4He", method="parameterized")
        * dt_energy_per_reaction_total_keV
        / (12 * T)
    ) / ((Cbr / (3 * (T**0.5))) + (1 / ntau))

    return Q


def Q_vs_T_and_ntau_eff_generalized(T, ntau_eff):
    """Return the fusion gain Q as a function of temperature T and Lawson
    parameter ntau assuming a pure, hydrogenic DT plasma. Used in the
    generalization of Lawson's second, pulsed scenario which includes self
    heating from charged fusion products. To match the paper we change the
    variable ntau --> ntau_eff, though this is effectively cosmetic here.

    Keyword arguments:
    T -- temperature in keV
    ntau_eff -- Lawson parameter, product of density and effective
                charactaristic time
    """
    reaction_info = reactions_config["T(d,n)4He"]
    dt_energy_per_reaction_total_keV = float(
        reaction_info["energy_per_reaction_total_keV"]
    )
    fraction_charged = float(reaction_info["fraction_charged"])

    Q = (
        reactivity.reactivity(T, "T(d,n)4He")
        * dt_energy_per_reaction_total_keV
        / (12 * T)
    ) / (
        (Cbr / (3 * (T**0.5)))
        - (
            fraction_charged
            * reactivity.reactivity(T, "T(d,n)4He")
            * dt_energy_per_reaction_total_keV
        )
        / (12 * T)
        + (1 / ntau_eff)
    )
    # Needed to handle unphysical range
    if Q <= 0:
        Q = float("inf")

    return Q
