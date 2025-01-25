import numpy

from scipy import integrate

from plasmapy.reaction_rates import reactivity
from plasmapy.reaction_rates.exceptions import ReactivityTemperatureTooLowError


class BaseProfile:
    def __init__(self):
        """Base class inhertited by all profiles

        Includes functions to evaluate values of lambda_F, lambda_B, lambda_kappa,
        and peaking values nu_T and nu_n
        """
        pass

    def lambda_F_of_T(self, T_i0):
        """Calculate and return lambda_F as a function of peak ion temperature.

        Keyword argument:
        T_i0 -- Max ion temperature at origin and in the uniform case

        Lambda_F is the ratio of the average fusion power density in the profile case
        to that of the spatially uniform case.

        Here we assume a cylindrical geometry or a circular torus geometry.
        We also asume a purely D-T reaction.
        """
        uniform_power_density = reactivity.reactivity(T_i0, "T(d,n)4He")
        profile_avg_power_density = integrate.quad(
            lambda x: ((self.n_i(x)) ** 2)
            * reactivity.reactivity(
                T_i=T_i0 * self.T_i(x),
                reaction="T(d,n)4He",
                method="parameterized",
                zero_low_temperatures=True,
            )
            * 2
            * x,
            0,
            1,
        )[0]
        lambda_F = profile_avg_power_density / uniform_power_density
        return lambda_F

    def lambda_B(self):
        """Calculate and return lambda_B

        Lambda_B is the ratio of the volume average bremsstrahlung power density
        in the profile case to the that of the spatially uniform case.

        Here we assume a cylindrical geometry or a circular torus geometry.
        """
        lambda_B = integrate.quad(
            lambda x: (self.n_i(x)) ** 2 * (self.T_i(x)) ** (1 / 2) * 2 * x, 0, 1
        )[0]
        return lambda_B

    def lambda_kappa(self):
        """Calculate and return lambda_kappa.

        Lambda_kappa is the ratio of the average thermal conduction power density
        in the profile case to that of the spatially uniform case.

        Here we assume a cylindrical geometry or a circular torus geometry.
        """
        lambda_kappa = integrate.quad(
            lambda x: self.n_i(x) * self.T_i(x) * 2 * x, 0, 1
        )[0]
        return lambda_kappa

    def peaking_temperature(self):
        """Calculate the temperature peaking $\nu_T$.
        Note that this is only defined for plasmas that do not extend beyond x=1.
        (Excludes Bennet profiles)
        """
        T_avg = integrate.quad(lambda x: self.T_i(x) * 2 * x, 0, 1)[0]
        peaking_T = self.T_i(0) / T_avg
        return peaking_T

    def peaking_density(self):
        """Calculate the density peaking $\nu_n$
        Note that this is only defined for plasmas that do not extend beyond x=1.
        (Excludes Bennet profiles)
        """
        n_avg = integrate.quad(lambda x: self.n_i(x) * 2 * x, 0, 1)[0]
        peaking_n = self.n_i(0) / n_avg
        return peaking_n

    def __str__(self):
        s = """
            name: {}
            lambda_B: {}
            lambda_kappa: {}
            peaking_temperature: {}
            peaking_density: {}
            """.format(
            self.name,
            self.lambda_B(),
            self.lambda_kappa(),
            self.peaking_temperature(),
            self.peaking_density(),
        )
        return s


class UniformProfile(BaseProfile):
    def __init__(self):
        self.name = "uniform"
        self.displayname = "Uniform profile"

    def n_i(self, x):
        return 1

    def T_i(self, x):
        return 1


class ParabolicProfile(BaseProfile):
    def __init__(self):
        self.name = "parabolic"
        self.displayname = "Parabolic profile"

    def n_i(self, x):
        return (1 - x**2) ** (1)

    def T_i(self, x):
        return (1 - x**2) ** (1)


class PeakedAndBroadProfile(BaseProfile):
    def __init__(self):
        self.name = "peaked_and_broad"
        self.displayname = "Peaked and broad profile"

    def n_i(self, x):
        return (1 - x**2) ** (0.2)

    def T_i(self, x):
        return (1 - x**2) ** (3)


class BennettProfile(BaseProfile):
    def __init__(self):
        self.name = "bennett"
        self.displayname = "Bennett profile"

    def n_i(self, x):
        return (1 + x**2) ** -2

    def T_i(self, x):
        return 1


class AdvancedTokamakProfile(BaseProfile):
    def __init__(self):
        self.name = "advanced_tokamak"
        self.displayname = "Advanced tokamak profile"

    def n_i(self, x):
        return (1 - x**2) ** 0.5

    def T_i(self, x):
        return 0.2 + (1 - 0.2) * (1 - x**5) ** 8


class TokamakProfile(BaseProfile):
    def __init__(self):
        self.name = "tokamak"
        self.displayname = "Tokamak profile"
        self.citations = ["Angioni_2009"]

    def n_i(self, x):
        return (1 - x**2) ** 0.5

    def T_i(self, x):
        return 1 - x**2


class SphericalTokamakProfile(BaseProfile):
    def __init__(self):
        """See p.8 of Buxton_2019, "On the energy confinement time in spherical
        tokamaks: implications for the design of pilot plants and fusion
        reactiors" for source of density and temperature profiles for a ST.

        Don't change, this is used to calculate density and temperature peaking.
        """
        self.name = "spherical_tokamak"
        self.displayname = "Spherical tokamak profile"
        self.citations = ["Buxton_2019"]

    def n_i(self, x):
        return (1 - x**2) ** 0.7

    def T_i(self, x):
        return (1 - x**2) ** 1.1


class SPARCProfile(BaseProfile):
    def __init__(self):
        self.name = "sparc"
        self.displayname = "SPARC profile"

    def n_i(self, x):
        return 0.8 + (1 - 0.8) * (1 - x**1) ** 1

    def T_i(self, x):
        return 0.23 + (1 - 0.23) * (1 - x**2) ** 2


class StellaratorProfile(BaseProfile):
    def __init__(self):
        self.name = "stellarator"
        self.displayname = "Stellarator profile"
        self.citations = ["Sheffield_2016"]

    def n_i(self, x):
        return 1

    def T_i(self, x):
        return (1 - x**2) ** 2
