"""
Implementation of the magnetized transport coefficients presented by
Braginskii in his 1965 review article "Transport Processes in a Plasma"
"""

__all__ = [
    "Braginskii",
]

import numpy as np
import os
import warnings

from plasmapy.transport.classical.base import AbstractClassicalTransportCoefficients

# Get the absolute path to the data files
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

coef_table = {}
coef_table["Z"] = np.array([1, 2, 3, 4, np.inf])

# TABLE 2
coef_table["alpha0"] = np.array([0.5129, 0.4408, 0.3965, 0.3752, 0.2949])
coef_table["beta0"] = np.array([0.7110, 0.9052, 1.016, 1.090, 1.521])
coef_table["gamma0"] = np.array([3.1616, 4.890, 6.064, 6.920, 12.471])

coef_table["delta0"] = np.array([3.7703, 1.0465, 0.5814, 0.4106, 0.0961])
coef_table["delta1"] = np.array([14.79, 10.80, 9.618, 9.055, 7, 482])

coef_table["alpha1p"] = np.array([6.416, 5.523, 5.226, 5.077, 4.63])
coef_table["alpha0p"] = np.array([1.837, 0.5956, 0.3515, 0.2566, 0.0678])
coef_table["alpha1pp"] = np.array([1.704] * 5)
coef_table["alpha0pp"] = np.array([0.7796, 0.3439, 0.2400, 0.1957, 0.0940])

coef_table["beta1p"] = np.array([5.101, 4.450, 4.233, 4.124, 3.798])
coef_table["beta0p"] = np.array([2.681, 0.9473, 0.5905, 0.4478, 0.1461])
coef_table["beta1pp"] = np.array([1.5] * 5)
coef_table["beta0pp"] = np.array([3.053, 1.784, 1.442, 1.285, 0.877])

coef_table["gamma1p"] = np.array([4.664, 3.957, 3.721, 3.604, 3.25])
coef_table["gamma0p"] = np.array([11.92, 5.118, 3.525, 2.841, 1.20])
coef_table["gamma1pp"] = np.array([2.5] * 5)
coef_table["gamma0pp"] = np.array([21.67, 15.37, 13.53, 12.65, 10.23])


# TODO: Note that the Braginskii Beta coefficient is normalized differently than in EH:
# Compare Braginskii Eq 4.30-4.33 to EH Eq. 7a-7b. Factors of ne and Te?
# The docs need to include a very clear indication that of what these coefficients
# are!
# The EH normalization is at the top of Pg.5 right
# The Braginskii normalization is just implicit in the equtions


class Braginskii(AbstractClassicalTransportCoefficients):
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

    def _Delta(self, i):
        return (
            self.chi_e**4
            + self._c["delta1"][i] * self.chi_e**2
            + self._c["delta0"][i]
        )

    @property
    def norm_alpha_para(self):
        i = self._find_nearest_Z(self.Z)
        return self._c["alpha0"][i] * np.ones(self.chi_e.size)

    @property
    def norm_alpha_perp(self):
        i = self._find_nearest_Z(self.Z)
        return 1 - (
            self._c["alpha1p"][i] * self.chi_e**2 + self._c["alpha0p"][i]
        ) / self._Delta(i)

    @property
    def norm_alpha_cross(self):
        i = self._find_nearest_Z(self.Z)
        return (
            self.chi_e
            * (self._c["alpha1pp"][i] * self.chi_e**2 + self._c["alpha0pp"][i])
            / self._Delta(i)
        )

    @property
    def norm_beta_para(self):
        i = self._find_nearest_Z(self.Z)
        return self._c["beta0"][i] * np.ones(self.chi_e.size)

    @property
    def norm_beta_perp(self):
        i = self._find_nearest_Z(self.Z)
        return (
            self._c["beta1p"][i] * self.chi_e**2 + self._c["beta0p"][i]
        ) / self._Delta(i)

    @property
    def norm_beta_cross(self):
        i = self._find_nearest_Z(self.Z)
        return (
            self.chi_e
            * (self._c["beta1pp"][i] * self.chi_e**2 + self._c["beta0pp"][i])
            / self._Delta(i)
        )

    @property
    def norm_kappa_e_para(self):
        i = self._find_nearest_Z(self.Z)
        return self._c["gamma0"][i] * np.ones(self.chi_e.size)

    @property
    def norm_kappa_e_perp(self):
        i = self._find_nearest_Z(self.Z)
        return (
            self._c["gamma1p"][i] * self.chi_e**2 + self._c["gamma0p"][i]
        ) / self._Delta(i)

    @property
    def norm_kappa_e_cross(self):
        i = self._find_nearest_Z(self.Z)
        return (
            self.chi_e
            * (self._c["gamma1pp"][i] * self.chi_e**2 + self._c["gamma0pp"][i])
            / self._Delta(i)
        )


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    """
    chi = np.linspace(-2, 2, num=50)
    chi = 10**chi

    coef = BraginskiiInterpolated.dimensionless(chi_e=chi, Z=5)

    data = coef.norm_alpha[1, :]

    # print(data)

    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.plot(chi, data)

    """
    chi = np.linspace(-2, 2, num=100)
    chi = 10**chi

    # Instantiate the object
    coef1 = Braginskii.dimensionless(chi_e=chi, Z=1)
    coef2 = Braginskii.dimensionless(chi_e=chi, Z=np.inf)

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
