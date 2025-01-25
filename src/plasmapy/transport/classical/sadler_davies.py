"""
Implementation of the magnetized transport coefficients simultaneously proposed
by Sadler et al. (10.1103/physrevlett.126.075001)
and Davies et al. (10.1063/5.0023445) in 2021.

Coefficents are taken from the supplemntal materials to the Sadler et al.
paper.
"""

__all__ = [
    "SadlerDavies",
]

import numpy as np

from collections import namedtuple

from plasmapy.transport.classical.base import AbstractClassicalTransportCoefficients

coefs = namedtuple("coefs", ["A", "B", "C", "D", "E", "F"])

coef_table = {}
coef_table["_apara"] = coefs(A=0.129, B=0.886, C=0.295, D=1.59, E=None, F=None)
coef_table["_aperp0"] = coefs(A=2.49, B=1.41, C=0.119, D=-0.01, E=None, F=None)
coef_table["_aperp1"] = coefs(A=4.94, B=6.44, C=1.71, D=2.10, E=None, F=None)
coef_table["_aperp2"] = coefs(A=8.94, B=23.0, C=5.95, D=2.78, E=None, F=None)

coef_table["_awdge0"] = coefs(A=0.252, B=-0.455, C=3.02, D=0.983, E=10.5, F=7.49)
coef_table["_awdge1"] = coefs(A=0.919, B=4.57, C=1.79, D=1.53, E=3.22, F=1.25)
coef_table["_awdge2"] = coefs(A=37.6, B=17.1, C=12.6, D=7.47, E=4.34, F=6.51)
coef_table["_awdge3"] = coefs(A=21.3, B=78.7, C=20.5, D=13.5, E=6.02, F=1.35)

coef_table["_bpara"] = coefs(A=0.461, B=22.3, C=11.3, D=10.9, E=29.1, F=7.55)
coef_table["_bperp1"] = coefs(A=-173, B=1230, C=68.0, D=690, E=78.6, F=2.68)
coef_table["_bperp2"] = coefs(A=12.3, B=563, C=3350, D=973, E=340, F=19.6)
coef_table["_bperp3"] = coefs(A=3.24, B=0.109, C=78.5, D=277, E=105, F=6.36)

coef_table["_bwdge0"] = coefs(A=8.5, B=-18.1, C=30.7, D=12.1, E=24.2, F=11.3)
coef_table["_bwdge1"] = coefs(A=-22.7, B=7140, C=1470, D=1.5, E=5340, F=980)
coef_table["_bwdge2"] = coefs(A=72.1, B=-60, C=47.1, D=16.7, E=31.6, F=12.3)
coef_table["_bwdge3"] = coefs(A=13.4, B=37.5, C=65.6, D=17.5, E=17.2, F=10.3)

coef_table["_kpara"] = coefs(A=39.3, B=142, C=5.18, D=44.8, E=11.9, F=0.398)
coef_table["_kperp1"] = coefs(A=51, B=46.1, C=4.24, D=43.5, E=5.41, F=0.411)
coef_table["_kperp2"] = coefs(A=-385, B=969, C=122, D=126, E=8.02, F=0.734)
coef_table["_kperp3"] = coefs(A=62.3, B=-96.1, C=57.8, D=3.13, E=10, F=1.13)

coef_table["_kwdge0"] = coefs(A=45600, B=-89800, C=73800, D=168, E=4350, F=327)
coef_table["_kwdge1"] = coefs(A=1.3, B=12.8, C=2.38, D=2.72, E=4.19, F=1.31)
coef_table["_kwdge2"] = coefs(A=31800, B=-42200, C=44000, D=11.4, E=20600, F=1920)
coef_table["_kwdge3"] = coefs(A=19700, B=-34300, C=30200, D=71.2, E=4250, F=316)


class SadlerDavies(AbstractClassicalTransportCoefficients):
    def __getattr__(self, name):

        eq3_list = ["_apara", "_aperp0", "_aperp1", "_aperp2"]

        eq6_list = [
            "_bpara",
            "_bperp1",
            "_bperp2",
            "_bperp3",
            "_kpara",
            "_kperp1",
            "_kperp2",
            "_kperp3",
        ]

        eq10_list = [
            "_awdge0",
            "_awdge1",
            "_awdge2",
            "_awdge3",
            "_bwdge0",
            "_bwdge1",
            "_bwdge2",
            "_bwdge3",
            "_kwdge0",
            "_kwdge1",
            "_kwdge2",
            "_kwdge3",
        ]

        Z = self.Z

        # Equation 3
        if name in eq3_list:
            c = coef_table[name]
            return (c.A + c.B * Z + c.C * Z**2) / (c.D * Z + Z**2)

        # Equation 6
        elif name in eq6_list:
            c = coef_table[name]
            return (c.A * Z + c.B * Z**2 + c.C * Z**3) / (
                1 + c.D * Z + c.E * Z**2 + c.F * Z**3
            )

        # Equation 10
        elif name in eq10_list:
            c = coef_table[name]
            return (c.A + c.B * Z + c.C * Z**2 + c.D * Z**3) / (
                1 + c.E * Z + c.F * Z**2 + Z**3
            )
        else:
            raise KeyError("Unrecognized attribute: {name}")

    @property
    def norm_alpha_para(self):
        return self._apara * np.ones(self.chi_e.shape)

    @property
    def norm_alpha_perp(self):
        X = self.chi_e
        return self._apara + (X**2 + (1 - self._apara) * X**3) / (
            self._aperp0 + self._aperp1 * X + self._aperp2 * X**2 + X**3
        )

    @property
    def norm_alpha_cross(self):
        X = self.chi_e
        return (self._awdge0 * X + self._awdge1 * X**2) / (
            1 + self._awdge2 * X + self._awdge3 * X**2 + X**3
        ) ** (8 / 9)

    @property
    def norm_beta_para(self):
        return self._bpara * np.ones(self.chi_e.shape)

    @property
    def norm_beta_perp(self):
        X = self.chi_e
        return (
            self._bpara
            * (1 + (8 / 9) * self._bperp1 * X)
            / (1 + self._bperp1 * X + self._bperp2 * X**2 + self._bperp3 * X**3)
            ** (8 / 9)
        )

    @property
    def norm_beta_cross(self):
        X = self.chi_e
        return (self._bwdge0 * X + self._bwdge1 * X**2) / (
            1 + self._bwdge2 * X + self._bwdge3 * X**2 + X**3
        )

    @property
    def norm_kappa_e_para(self):
        return self._kpara * np.ones(self.chi_e.shape)

    @property
    def norm_kappa_e_perp(self):
        X = self.chi_e
        return (
            self._kpara
            * (1 + self._kperp1 * X)
            / (1 + self._kperp1 * X + self._kperp2 * X**2 + self._kperp3 * X**3)
        )

    @property
    def norm_kappa_e_cross(self):
        X = self.chi_e
        return (self._kwdge0 * X + self._kwdge1 * X**2) / (
            1 + self._kwdge2 * X + self._kwdge3 * X**2 + X**3
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    chi = np.linspace(-2, 2, num=50)
    chi = 10**chi

    coef = SadlerDavies.dimensionless(chi_e=chi, Z=1)

    data = coef.norm_gamma[1, :]

    print(data.shape)

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.plot(chi, data)
