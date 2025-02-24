__all__ = [
    "AbstractClassicalTransportCoefficients",
    "validate_attributes_not_none",
]

import functools
from abc import ABC

import astropy.constants as const
import astropy.units as u
import numpy as np

from plasmapy import particles
from plasmapy.formulary.collisions import (
    fundamental_electron_collision_freq,
    fundamental_ion_collision_freq,
)
from plasmapy.formulary.frequencies import gyrofrequency
from plasmapy.particles.particle_class import ParticleLike

m_e = const.m_e.si


def validate_attributes_not_none(attributes=()):
    """
    This decorator for methods of a class validates that a list of
    attributes on that class are not None.

    Expand to a real separate decorator in the util package?
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            missing = []

            for a in attributes:
                if getattr(self, a) is None:
                    missing.append(a)

            if len(missing) > 0:
                raise ValueError(
                    f"Attributes {attributes} must be set to calculate "
                    f"{self.__class__.__name__}.{func.__name__}(), but "
                    f"the following keywords were not provided {missing}."
                )
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator


# This format string is used for cleaner NotImplemented errors for
# coefficients that are not included in a particular model
not_implemented = (
    "The {0} property is not implemented in {1}. "
    "The {0} coefficient may not be defined as part of this model."
)


class AbstractClassicalTransportCoefficients(ABC):
    r"""
    An abstract class representing classical transport coefficients. Subclasses
    representing different classical transport models re-implement
    the transport coefficient methods of this abstract class.

    """

    def __init__(
        self,
    ):
        self._dimensional = None
        self.chi_e = None
        self.chi_i = None
        self.Z = None
        self.particle = None
        self.B = None
        self.n_e = None
        self.n_i = None
        self.T_e = None
        self.T_i = None
        self.e_collision_freq = None
        self.i_collision_freq = None

    @classmethod
    def dimensionless(
        cls,
        Z: int | float | np.ndarray,
        chi_e: np.ndarray | None = None,
        chi_i: np.ndarray | None = None,
    ):
        """
        Construct object with dimensionless parameters.

        Parameters (dimensionless)
        --------------------------

        chi_e : `numpy.ndarray` (N,)
            The electron Hall parameter

        chi_i : `numpy.ndarray` (N,)
            The ion Hall parameter

        Z : float
            The plasma mean ionization
        """

        obj = cls()
        obj._dimensional = False

        # TODO: Ensure support for arrays of Z
        if not isinstance(Z, (int, float, np.ndarray)):
            raise ValueError("Z must be a dimensionless number or " "`~numpy.ndarray`.")
        else:
            obj.Z = Z

        # chi_e and chi_i must be a number or None, but both cannot be None
        if not (isinstance(chi_e, np.ndarray) or chi_e is None):
            raise ValueError(
                "chi_e must be either a dimensionless " "`~numpy.ndarray` or None."
            )
        else:
            obj.chi_e = chi_e

        if not (isinstance(chi_i, np.ndarray) or chi_i is None):
            raise ValueError(
                "chi_i must be either a dimensionless " "`~numpy.ndarray` or None."
            )
        else:
            obj.chi_i = chi_i

        if (obj.chi_e is None) and (obj.chi_i is None):
            raise ValueError(
                "chi_e and chi_i cannot both be None when "
                "using the dimensionless mode."
            )

        return obj

    @classmethod
    @particles.particle_input()
    def dimensional(
        cls,
        *,
        particle: ParticleLike | None = None,
        B: u.T = None,
        n_e: u.cm**-3 = None,
        n_i: u.cm**-3 = None,
        T_e: u.K = None,
        T_i: u.K = None,
        e_collision_freq: u.Hz = None,
        i_collision_freq: u.Hz = None,
    ):
        """
        Parameters
        ----------
        particle : `~plasmapy.particles.Particle` instance or str
            The ion species

        B : `~astropy.units.Quantity`
            Magnetic field strength in units convertible to Tesla.

        n_e : `~astropy.units.Quantity`
            Electron number density, in units convertible to cm^-3.

        n_i : `~astropy.units.Quantity`
            Ion number density, in units convertible to cm^-3.

        T_e : `~astropy.units.Quantity`
            Electron temperature, in units convertible to eV.

        T_i : `~astropy.units.Quantity`
            Ion temperature, in units convertible to eV.

        e_collision_freq : `~astropy.units.Quantity`
            The fundamental electron collision frequency, in units convertible
            to Hz. If not provided, this will be calculated using
            `~plasmapy.formulary.collisions.fundamental_electron_collision_freq()`

        i_collision_freq : `~astropy.units.Quantity`
            The fundamental ion collision frequency, in units convertible
            to Hz. If not provided, this will be calculated using
            `~plasmapy.formulary.collisions.fundamental_ion_collision_freq()`
        """

        obj = cls()
        obj._dimensional = True

        # particle and B keyword arguments are always required
        if particle is None:
            raise ValueError(
                "The 'particle' keyword is required in dimensional " "mode"
            )
        else:
            obj.particle = particle
            obj.Z = obj.particle.charge_number

        if B is None:
            raise ValueError("The 'B' keyword is required in dimensional " "mode.")
        else:
            obj.B = B

        # Validate that input arrays have the same shape
        shape = obj.B.shape
        array_quantities = {
            "n_e": n_e,
            "n_i": n_i,
            "T_e": T_e,
            "T_i": T_i,
            "e_collision_freq": e_collision_freq,
            "i_collision_freq": i_collision_freq,
        }
        for key, val in array_quantities.items():
            if val is not None and val.shape != shape:
                raise ValueError(
                    "Shape of arrays must be equal, but "
                    f"B: {B.shape} and "
                    f"{key}:{val.shape} are not."
                )

        obj.n_e = n_e
        obj.n_i = n_i
        obj.T_e = T_e
        obj.T_i = T_i
        obj.e_collision_freq = e_collision_freq
        obj.i_collision_freq = i_collision_freq

        # If all the electron quantities are provided, calculate chi_e
        if all(v is not None for v in [obj.n_e, obj.T_e]):
            # If collision frequency is not provided, calculate it from the
            # provided parameters
            if obj.e_collision_freq is None:
                obj.e_collision_freq = fundamental_electron_collision_freq(
                    obj.T_e, obj.n_e, obj.particle
                )
            wce = gyrofrequency(obj.B, "e-")
            obj.chi_e = (wce / obj.e_collision_freq).to(u.rad).value

        else:
            obj.chi_e = None

        # If all the ion quantities are provided, calculate chi_i
        if all(v is not None for v in [obj.n_i, obj.T_i]):
            # If collision frequency is not provided, calculate it from the
            # provided parameters
            if obj.i_collision_freq is None:
                obj.i_collision_freq = fundamental_ion_collision_freq(
                    obj.T_i, obj.n_i, obj.particle
                )
            wci = gyrofrequency(obj.B, obj.particle)

            obj.chi_i = (wci / obj.i_collision_freq).to(u.rad).value
        else:
            obj.chi_i = None

        return obj

    # **********************************************************************
    # Normalization Constants
    # (defaults, overwritable by child classes)
    # Normalizations are defined such that multiplying the dimensionless
    # quantity by the normalization constant will return the dimensional
    # quantity
    # **********************************************************************
    @property
    def alpha_normalization(self):
        """
        The normalization constant for alpha.

        Defined such that multiplying the dimensionless quantity by the
        normalization constnat will return the dimensional quantity.
        """
        if self._dimensional:
            return self.particle.mass * self.n_e * self.e_collision_freq
        else:
            return None

    @property
    def beta_normalization(self):
        """
        The normalization constant for beta.

        Defined such that multiplying the dimensionless quantity by the
        normalization constnat will return the dimensional quantity.
        """
        if self._dimensional:
            return 1.0
        else:
            return None

    @property
    def kappa_e_normalization(self):
        """
        The normalization constant for kappa_e.

        Defined such that multiplying the dimensionless quantity by the
        normalization constnat will return the dimensional quantity.
        """
        if self._dimensional:
            return self.n_e * self.T_e / self.e_collision_freq / m_e
        else:
            return None

    @property
    def eta_e_normalization(self):
        """
        The normalization constant for eta_e.

        Defined such that multiplying the dimensionless quantity by the
        normalization constnat will return the dimensional quantity.
        """
        if self._dimensional:
            return self.n_e * self.T_e / self.e_collision_freq
        else:
            return None

    @property
    def kappa_i_normalization(self):
        """
        The normalization constant for kappa_i.

        Defined such that multiplying the dimensionless quantity by the
        normalization constnat will return the dimensional quantity.
        """
        if self._dimensional:
            return self.n_i * self.T_i / self.i_collision_freq / self.particle.mass
        else:
            return None

    @property
    def eta_i_normalization(self):
        """
        The normalization constant for eta_i.

        Defined such that multiplying the dimensionless quantity by the
        normalization constnat will return the dimensional quantity.
        """
        if self._dimensional:
            return self.n_i * self.T_i / self.i_collision_freq
        else:
            return None

    @property
    def gamma_normalization(self):
        """
        The normalization constant for gamma.

        Defined such that multiplying the dimensionless quantity by the
        normalization constnat will return the dimensional quantity.
        """
        # Walsh 2020 Eq. A6, gamma =  gamma_dimensionless * tau_e/m_e
        if self._dimensional:
            return 1 / (self.e_collision_freq * const.m_e.si)
        else:
            return None

    # **********************************************************************
    # Resistivity (alpha)
    # **********************************************************************
    @property
    def norm_alpha_para(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    def norm_alpha_perp(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    def norm_alpha_cross(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    @validate_attributes_not_none(attributes=("chi_e", "Z"))
    def norm_alpha(self):
        """
        Calculates the normalized alpha coefficients in terms of the
        dimensionless Hall parameter and the ionization fraction.

        Returns
        -------
        norm_alpha, `~numpy.ndarray` of `~u.Quantity` instances (3, N)
            The resistivity coefficients:
                [alpha_para, alpha_perp, alpha_cross]
        """
        return np.array(
            [self.norm_alpha_para, self.norm_alpha_perp, self.norm_alpha_cross]
        )

    @property
    def alpha_para(self):
        return self.norm_alpha * self.norm_alpha_para

    @property
    def alpha_perp(self):
        return self.norm_alpha * self.norm_alpha_perp

    @property
    def alpha_cross(self):
        return self.norm_alpha * self.norm_alpha_cross

    @property
    @validate_attributes_not_none(attributes=("n_e", "T_e", "B", "particle"))
    def alpha(self):
        return self.norm_alpha * self.alpha_normalization

    # **********************************************************************
    # Thermoelectric Coefficient (beta)
    # **********************************************************************
    @property
    def norm_beta_para(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    def norm_beta_perp(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    def norm_beta_cross(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    @validate_attributes_not_none(attributes=("chi_e", "Z"))
    def norm_beta(self):
        """
        Calculates the normalized beta coefficients in terms of the
        dimensionless Hall parameter and the ionization fraction.


        Returns
        -------
        norm_beta, `~numpy.ndarray` of `~u.Quantity` instances (3, N)
            The thermoelectric coefficients:
                [beta_para, beta_perp, beta_cross]

        """
        return np.array(
            [self.norm_beta_para, self.norm_beta_perp, self.norm_beta_cross]
        )

    @property
    def beta_para(self):
        return self.norm_beta * self.norm_beta_para

    @property
    def beta_perp(self):
        return self.norm_beta * self.norm_beta_perp

    @property
    def beta_cross(self):
        return self.norm_beta * self.norm_beta_cross

    @property
    @validate_attributes_not_none(attributes=())
    def beta(self):
        return self.norm_beta * self.beta_normalization

    # **********************************************************************
    # Electron Thermal Conductivity (kappa_e)
    # **********************************************************************
    @property
    def norm_kappa_e_para(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    def norm_kappa_e_perp(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    def norm_kappa_e_cross(self):
        raise NotImplementedError(
            "This coefficient is not implemented by the " "current class"
        )

    @property
    @validate_attributes_not_none(attributes=("chi_e", "Z"))
    def norm_kappa_e(self):
        """
        Calculates the normalized kappa_e coefficients in terms of the
        dimensionless Hall parameter and the ionization fraction.

        Returns
        -------
        norm_kapp_e, `~numpy.ndarray` of `~u.Quantity` instances (3, N)
            The electron thermal conductivity coefficients:
                [kappa_e_para, kappa_e_perp, kappa_e_cross]

        """
        return np.array(
            [self.norm_kappa_e_para, self.norm_kappa_e_perp, self.norm_kappa_e_cross]
        )

    @property
    def kappa_e_para(self):
        return self.norm_kappa_e * self.norm_kappa_e_para

    @property
    def kappa_e_perp(self):
        return self.norm_kappa_e * self.norm_kappa_e_perp

    @property
    def kappa_e_cross(self):
        return self.norm_kappa_e * self.norm_kappa_e_cross

    @property
    @validate_attributes_not_none(attributes=("n_e", "T_e", "B", "particle"))
    def kappa_e(self):
        # newaxis is necessary if n_e, B etc. are not scalars
        return self.norm_kappa_e * self.kappa_e_normalization[..., np.newaxis]

    # **********************************************************************
    # Ion Thermal Conductivity (kappa_i)
    # **********************************************************************

    @property
    def norm_kappa_i(self):
        raise NotImplementedError(
            not_implemented.format("kappa_i", self.__class__.__name__)
        )

    @property
    @validate_attributes_not_none(attributes=("n_i", "T_i", "B", "particle"))
    def kappa_i(self):
        return self.norm_kappa_i * self.kappa_i_normalization[..., np.newaxis]

    # **********************************************************************
    # Electron Viscosity (eta_e)
    # Note: returns an array [eta0, eta1, eta2, eta3, eta4]
    # **********************************************************************
    @property
    def norm_eta_e(self):
        raise NotImplementedError(
            not_implemented.format("eta_e", self.__class__.__name__)
        )

    @property
    @validate_attributes_not_none(attributes=("n_e", "T_e", "B", "particle"))
    def eta_e(self):
        return self.norm_eta_e * self.eta_e_normalization

    # **********************************************************************
    # Ion Viscosity (eta_i)
    # Note: returns an array [eta0, eta1, eta2, eta3, eta4]
    # **********************************************************************
    @property
    def norm_eta_i(self):
        raise NotImplementedError(
            not_implemented.format("eta_i", self.__class__.__name__)
        )

    @property
    @validate_attributes_not_none(attributes=("n_i", "T_i", "B", "particle"))
    def eta_i(self):
        return self.norm_eta_i * self.eta_i_normalization

    # **********************************************************************
    # Resistive Velocity (delta)
    # "Symmetric" coefficient formulism of Sadler and Davies
    # **********************************************************************
    @property
    def norm_delta(self):
        """
        The normalized symmetric transport coefficient delta

        Returns
        -------
        coef : `~numpy.ndarray` (2,)
            An array of
            [norm_delta_perp, norm_delta_cross]
        """
        perp = self.norm_alpha[2] / self.chi_e
        cross = (self.norm_alpha[1] - self.norm_alpha[0]) / self.chi_e

        return np.array([perp, cross])

    @property
    @validate_attributes_not_none(attributes=("n_e", "T_e", "B", "particle"))
    def delta(self):
        return self.norm_delta * self.alpha_normalization

    # **********************************************************************
    # Nernst Coefficient (gamma)
    # "Symmetric" coefficient formulism of Sadler and Davies
    # **********************************************************************

    @property
    def norm_gamma_perp(self):
        return self.norm_beta_cross / self.chi_e

    @property
    def norm_gamma_cross(self):
        return (self.norm_beta_para - self.norm_beta_perp) / self.chi_e

    @property
    def norm_gamma(self):
        """
        The normalized symmetric transport coefficient gamma

        Returns
        -------
        coef : `~numpy.ndarray` (2,)
            An array of
            [norm_gamma_perp, norm_gamma_cross]
        """
        return np.array([self.norm_gamma_perp, self.norm_gamma_cross])

    @property
    @validate_attributes_not_none(attributes=())
    def gamma(self):
        return self.norm_gamma * self.gamma_normalization


if __name__ == "__main__":
    chi = np.linspace(-2, 2, num=50)
    chi = 10**chi
    x = AbstractClassicalTransportCoefficients.dimensionless(chi_e=chi, Z=1)
    print(x.norm_beta)
