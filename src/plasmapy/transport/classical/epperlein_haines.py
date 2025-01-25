"""
Implementation of the magnetized transport coefficients presented by
Epperlein and Haines in their 1986 paper "Plasma transport coefficients in a magnetic field by direct numerical solution of
the Fokker–Planck equation" (Phys. Fluids), doi: 10.1063/1.865901
"""

__all__ = [
    "EpperleinHaines",
]

import numpy as np
import warnings

from plasmapy.transport.classical.base import AbstractClassicalTransportCoefficients

coef_table = {}
coef_table["Z"] = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 20, 30, 60, np.inf])

# TABLE II
coef_table["alpha0"] = np.array(
    [
        0.5061,
        0.4295,
        0.3950,
        0.3750,
        0.3618,
        0.3524,
        0.3454,
        0.3399,
        0.3319,
        0.3263,
        0.3221,
        0.3144,
        0.3081,
        0.3015,
        0.2945,
    ]
)
coef_table["alpha0p"] = np.array(
    [
        1.37,
        1.58,
        1.68,
        1.74,
        1.78,
        1.8,
        1.82,
        1.84,
        1.87,
        1.88,
        1.9,
        3.53,
        5.49,
        7.61,
        9.17,
    ]
)
coef_table["alpha1p"] = np.array(
    [
        3.03,
        3.21,
        3.17,
        3.15,
        3.14,
        3.13,
        3.12,
        3.11,
        3.1,
        3.1,
        3.09,
        3.52,
        3.97,
        4.41,
        4.73,
    ]
)
coef_table["a0p"] = np.array(
    [
        2.77,
        2.78,
        2.78,
        2.78,
        2.78,
        2.79,
        2.79,
        2.79,
        2.79,
        2.8,
        2.8,
        5.14,
        7.94,
        10.9,
        13,
    ]
)
coef_table["a1p"] = np.array(
    [
        6.72,
        6.7,
        6.47,
        6.37,
        6.33,
        6.29,
        6.26,
        6.23,
        6.21,
        6.2,
        6.19,
        7.97,
        10,
        12.2,
        13.8,
    ]
)
coef_table["alpha0x"] = np.array(
    [
        0.1989,
        0.3285,
        0.41457,
        0.4794,
        0.5285,
        0.5676,
        0.5996,
        0.6264,
        0.6686,
        0.7004,
        0.7254,
        0.7757,
        0.8209,
        0.8727,
        0.9328,
    ]
)
coef_table["alpha0pp"] = np.array(
    [
        2.66e2,
        4.91e2,
        6.3e2,
        7.06e2,
        7.23e2,
        7.57e2,
        7.94e2,
        8.17e2,
        8.89e2,
        9.62e2,
        9.88e2,
        1.06e3,
        1.17e3,
        1.3e3,
        1.33e3,
    ]
)
coef_table["alpha1pp"] = np.array([2.53] * 15)
coef_table["a0pp"] = np.array(
    [
        3280,
        3720,
        3790,
        3670,
        3370,
        3280,
        3250,
        3200,
        3270,
        3390,
        3360,
        3360,
        3520,
        3720,
        3540,
    ]
)
coef_table["a1pp"] = np.array(
    [
        3460,
        6690,
        8990,
        10200,
        10600,
        11100,
        11700,
        12200,
        13300,
        14700,
        15100,
        16200,
        18500,
        20800,
        21900,
    ]
)
coef_table["a2pp"] = np.array(
    [366, 554, 675, 735, 744, 769, 793, 825, 863, 925, 940, 985, 1080, 1180, 1180]
)

# TABLE III
# β0(Z=20) is clearly incorrect; replaced with fitted value
coef_table["beta0"] = np.array(
    [
        0.7029,
        0.9054,
        1.018,
        1.092,
        1.146,
        1.186,
        1.218,
        1.244,
        1.283,
        1.312,
        1.334,
        1.3801,
        1.414,
        1.455,
        1.5,
    ]
)
coef_table["beta0p"] = np.array(
    [
        1.05e3,
        1.38e3,
        1.55e3,
        1.64e3,
        1.71e3,
        1.74e3,
        1.73e3,
        1.79e3,
        1.92e3,
        1.89e3,
        1.92e3,
        2.01e3,
        2.09e3,
        2.16e3,
        2.20e3,
    ]
)
coef_table["beta1p"] = np.array([6.33] * 15)
coef_table["b0p"] = np.array(
    [
        3710,
        3800,
        3800,
        3740,
        3720,
        3640,
        3530,
        3570,
        3740,
        3580,
        3580,
        3620,
        3680,
        3690,
        3650,
    ]
)
coef_table["b1p"] = np.array(
    [
        4110,
        7050,
        8750,
        9730,
        10700,
        11100,
        11300,
        11700,
        12900,
        13000,
        13300,
        14300,
        15200,
        16000,
        16800,
    ]
)
coef_table["b2p"] = np.array(
    [515, 642, 690, 707, 731, 735, 729, 734, 772, 762, 765, 789, 808, 822, 809]
)
coef_table["beta0x"] = np.array(
    [
        0.8831,
        1.812,
        2.589,
        3.236,
        3.78,
        4.243,
        4.64,
        4.986,
        5.556,
        6.007,
        6.372,
        7.144,
        7.874,
        8.756,
        9.844,
    ]
)
coef_table["beta0pp"] = np.array(
    [
        2.54,
        4.40,
        3.77,
        3.43,
        3.20,
        3.05,
        2.92,
        2.82,
        2.70,
        2.62,
        2.55,
        2.43,
        2.31,
        2.26,
        2.15,
    ]
)
coef_table["beta1pp"] = np.array([1.5] * 15)
coef_table["b0pp"] = np.array(
    [
        2.87,
        2.43,
        1.46,
        1.06,
        0.848,
        0.718,
        0.629,
        0.565,
        0.486,
        0.436,
        0.401,
        0.341,
        0.294,
        0.258,
        0.219,
    ]
)
coef_table["b1pp"] = np.array(
    [
        3.27,
        5.18,
        4.34,
        3.92,
        3.66,
        3.48,
        3.33,
        3.21,
        3.09,
        3.00,
        2.93,
        2.81,
        2.68,
        2.64,
        2.53,
    ]
)
coef_table["b2pp"] = np.array(
    [
        7.09,
        9.34,
        8.65,
        8.27,
        8.02,
        7.83,
        7.68,
        7.55,
        7.41,
        7.30,
        7.22,
        7.07,
        6.91,
        6.84,
        6.72,
    ]
)

# TABLE IV
coef_table["gamma0"] = np.array(
    [
        3.203,
        4.931,
        6.115,
        6.995,
        7.68,
        8.231,
        8.685,
        9.067,
        9.673,
        10.13,
        10.5,
        11.23,
        11.9,
        12.67,
        13.58,
    ]
)
coef_table["gamma0p"] = np.array(
    [
        6.18,
        9.30,
        10.2,
        9.14,
        8.60,
        8.57,
        8.84,
        7.93,
        7.44,
        7.32,
        7.08,
        6.79,
        6.74,
        6.36,
        6.21,
    ]
)
coef_table["gamma1p"] = np.array(
    [
        4.66,
        3.96,
        3.72,
        3.6,
        3.53,
        3.49,
        3.49,
        3.43,
        3.39,
        3.37,
        3.35,
        3.32,
        3.3,
        3.27,
        3.25,
    ]
)
coef_table["c0p"] = np.array(
    [
        1.93,
        1.89,
        1.66,
        1.31,
        1.12,
        1.04,
        1.02,
        0.875,
        0.77,
        0.722,
        0.674,
        0.605,
        0.566,
        0.502,
        0.457,
    ]
)
coef_table["c1p"] = np.array(
    [
        2.31,
        3.78,
        4.76,
        4.63,
        4.62,
        4.83,
        5.19,
        4.74,
        4.63,
        4.7,
        4.64,
        4.65,
        4.81,
        4.71,
        4.81,
    ]
)
coef_table["c2p"] = np.array(
    [
        5.3,
        7.78,
        8.88,
        8.8,
        8.8,
        8.96,
        9.24,
        8.84,
        8.71,
        8.73,
        8.65,
        8.6,
        8.66,
        8.52,
        8.53,
    ]
)
coef_table["gamma0x"] = np.array(
    [
        6.071,
        15.75,
        25.65,
        34.95,
        43.45,
        51.12,
        58.05,
        64.29,
        75.04,
        83.93,
        91.38,
        107.8,
        124.3,
        145.2,
        172.7,
    ]
)
coef_table["gamma0pp"] = np.array(
    [
        4.01,
        2.46,
        1.13,
        0.628,
        0.418,
        0.319,
        0.268,
        0.238,
        0.225,
        0.212,
        0.202,
        0.200,
        0.194,
        0.189,
        0.186,
    ]
)
coef_table["gamma1pp"] = np.array([2.5] * 15)
coef_table["c0pp"] = np.array(
    [
        0.661,
        0.156,
        0.0442,
        0.018,
        0.00963,
        0.00625,
        0.00461,
        0.00371,
        0.003,
        0.00252,
        0.00221,
        0.00185,
        0.00156,
        0.0013,
        0.00108,
    ]
)
coef_table["c1pp"] = np.array(
    [
        0.931,
        0.398,
        0.175,
        0.101,
        0.0702,
        0.0551,
        0.0465,
        0.041,
        0.0354,
        0.0317,
        0.0291,
        0.0256,
        0.0228,
        0.0202,
        0.018,
    ]
)
coef_table["c2pp"] = np.array(
    [
        2.5,
        1.71,
        1.05,
        0.775,
        0.646,
        0.578,
        0.539,
        0.515,
        0.497,
        0.482,
        0.471,
        0.461,
        0.45,
        0.44,
        0.43,
    ]
)


class EpperleinHaines(AbstractClassicalTransportCoefficients):
    @property
    def _c(self):
        return coef_table

    def _find_nearest_Z(self, Z):
        """
        Finds the nearest Z-value to the given Z value in the coefficient tables.
        Prints a warning if the Z found is not equal to the Z requested.

        Parameters
        ----------
        Z : float
            An integer charge

        Returns
        -------
        i : int
            The index of the closest Z in the tables

        """
        if Z == np.inf:
            return -1

        i = np.argmin(np.abs(self._c["Z"] - Z))
        if self._c["Z"][i] != Z:
            warnings.warn(
                f"Value Z = {Z} is not in the coefficient table. "
                f"Using the nearest value, Z = {self._c['Z'][i]}. "
                f"The values in the table are {self._c['Z']}.",
                RuntimeWarning,
            )
        return i

    @property
    def norm_alpha_para(self):
        i = self._find_nearest_Z(self.Z)
        return self._c["alpha0"][i] * np.ones(self.chi_e.shape)

    @property
    def norm_alpha_perp(self):
        i = self._find_nearest_Z(self.Z)
        return 1 - (self._c["alpha1p"][i] * self.chi_e + self._c["alpha0p"][i]) / (
            self.chi_e**2 + self._c["a1p"][i] * self.chi_e + self._c["a0p"][i]
        )

    @property
    def norm_alpha_cross(self):
        i = self._find_nearest_Z(self.Z)
        return (
            self.chi_e
            * (self._c["alpha1pp"][i] * self.chi_e + self._c["alpha0pp"][i])
            / (
                self.chi_e**3
                + self._c["a2pp"][i] * self.chi_e**2
                + self._c["a1pp"][i] * self.chi_e
                + self._c["a0pp"][i]
            )
            ** (8 / 9)
        )

    @property
    def norm_beta_para(self):
        i = self._find_nearest_Z(self.Z)
        return self._c["beta0"][i]  * np.ones(self.chi_e.shape)

    @property
    def norm_beta_perp(self):
        i = self._find_nearest_Z(self.Z)
        return (self._c["beta1p"][i] * self.chi_e + self._c["beta0p"][i]) / (
            self.chi_e**3
            + self._c["b2p"][i] * self.chi_e**2
            + self._c["b1p"][i] * self.chi_e
            + self._c["b0p"][i]
        ) ** (8 / 9)

    # TODO: Note that this function doesn't match the EH paper Fig. 1 in the
    # chi -> inf side. The coefficients and polynomial are right...
    # this might be a mistake in the EH tables?
    @property
    def norm_beta_cross(self):
        i = self._find_nearest_Z(self.Z)
        return (
            self.chi_e
            * (self._c["beta1pp"][i] * self.chi_e + self._c["beta0pp"][i])
            / (
                self.chi_e**3 * self._c["b2pp"][i] * self.chi_e**2
                + self._c["b1pp"][i] * self.chi_e
                + self._c["b0pp"][i]
            )
        )

    @property
    def norm_kappa_e_para(self):
        i = self._find_nearest_Z(self.Z)
        return self._c["gamma0"][i] * np.ones(self.chi_e.shape)

    @property
    def norm_kappa_e_perp(self):
        i = self._find_nearest_Z(self.Z)
        return (self._c["gamma1p"][i] * self.chi_e + self._c["gamma0p"][i]) / (
            self.chi_e**3
            + self._c["c2p"][i] * self.chi_e**2
            + self._c["c1p"][i] * self.chi_e
            + self._c["c0p"][i]
        )

    @property
    def norm_kappa_e_cross(self):
        i = self._find_nearest_Z(self.Z)
        return (
            self.chi_e
            * (self._c["gamma1pp"][i] * self.chi_e + self._c["gamma0pp"][i])
            / (
                self.chi_e**3
                + self._c["c2pp"][i] * self.chi_e**2
                + self._c["c1pp"][i] * self.chi_e
                + self._c["c0pp"][i]
            )
        )


if __name__ == "__main__":

    import astropy.units as u

    coef = EpperleinHaines.dimensional(
        particle="H+", B=10 * u.T, n_e=1e19 * u.cm**-3, T_e=100 * u.eV
    )

    import matplotlib.pyplot as plt

    chi = np.linspace(-2, 2, num=100)
    chi = 10**chi

    # Instantiate the object
    coef1 = EpperleinHaines.dimensionless(chi_e=chi, Z=1)
    coef2 = EpperleinHaines.dimensionless(chi_e=chi, Z=np.inf)

    fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), sharex=True)

    for i in range(3):
        for j in range(2):
            ax = axarr[i][j]
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-2, 1e2)

    axarr[-1][0].set_xlabel("$\\chi$")
    axarr[-1][1].set_xlabel("$\\chi$")

    ax = axarr[0][0]
    ax.set_title("$\\alpha_\\perp$")
    ax.set_ylim(0.2, 1)
    ax.plot(chi, coef1.norm_alpha[1], label="Z=1")
    ax.plot(chi, coef2.norm_alpha[1], label="Z=Infinite")
    ax.legend()

    ax = axarr[1][0]
    ax.set_title("$\\alpha_\\wedge$")
    ax.set_ylim(0.01, 1)
    ax.plot(chi, coef1.norm_alpha[2])
    ax.plot(chi, coef2.norm_alpha[2])

    ax = axarr[2][0]
    ax.set_title("$\\beta_\\perp$")
    ax.set_ylim(0.001, 10)
    ax.plot(chi, coef1.norm_beta[1])
    ax.plot(chi, coef2.norm_beta[1])

    ax = axarr[0][1]
    ax.set_title("$\\beta_\\wedge$")
    ax.set_ylim(0.01, 1)
    ax.plot(chi, coef1.norm_beta[2])
    ax.plot(chi, coef2.norm_beta[2])

    ax = axarr[1][1]
    ax.set_title("$\\kappa_\\perp$")
    ax.set_ylim(0.001, 100)
    ax.plot(chi, coef1.norm_kappa_e[1])
    ax.plot(chi, coef2.norm_kappa_e[1])

    ax = axarr[2][1]
    ax.set_title("$\\kappa_\\wedge$")
    ax.set_ylim(0.01, 10)
    ax.plot(chi, coef1.norm_kappa_e[2])
    ax.plot(chi, coef2.norm_kappa_e[2])

    import astropy.units as u

    chi = np.linspace(-2, 2, num=5)
    chi = 10**chi
    coef = EpperleinHaines.dimensionless(chi_e=chi, Z=1)

    print(coef.norm_alpha.shape)

    import matplotlib.pyplot as plt

    chi = np.linspace(-1, 1, num=50)
    chi = 10**chi
    # Instantiate the object
    coef = EpperleinHaines.dimensionless(chi_e=chi, Z=1)
    para, perp, wedge = coef.norm_kappa_e

    mag_heatflux = np.sqrt(para**2 + perp**2 + wedge**2)

    fig, ax = plt.subplots()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\chi_e$")
    ax.set_ylabel("Normalized Transport Coefficient")

    ax.plot(chi, para, label="$\\kappa_\\parallel$")
    ax.plot(chi, perp, label="$\\kappa_\\perp$")
    ax.plot(chi, wedge, label="$\\kappa_\\wedge$")
    ax.plot(chi, mag_heatflux, label="|$q_e$| (norm.)")

    ax.legend()
