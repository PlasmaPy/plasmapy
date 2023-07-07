"""
Objects for storing magnetohydrodynamic wave parameters.
"""
__all__ = [
    "AbstractMHDWave",
    "AlfvenWave",
    "FastMagnetosonicWave",
    "SlowMagnetosonicWave",
    "mhd_waves",
]

import astropy.units as u
import numpy as np
import warnings

from abc import ABC, abstractmethod
from astropy.constants.si import k_B
from numbers import Integral, Real
from typing import Optional, Union

from plasmapy.formulary.frequencies import gyrofrequency
from plasmapy.formulary.speeds import Alfven_speed
from plasmapy.particles import electron, particle_input, ParticleLike
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning


class AbstractMHDWave(ABC):
    """
    Core class for magnetohydrodynamic waves.
    """

    @particle_input
    @validate_quantities(
        B={"can_be_negative": False},
        density={"can_be_negative": False},
        T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    )
    def __init__(
        self,
        B: u.T,
        density: (u.m**-3, u.kg / u.m**3),
        ion: ParticleLike,
        *,
        T: u.K = 0 * u.K,
        gamma: Union[float, int] = 5 / 3,
        mass_numb: Optional[Integral] = None,
        Z: Optional[Real] = None,
    ):
        # validate arguments
        for arg_name in ("B", "density", "T"):
            val = locals()[arg_name].squeeze()
            if val.shape != ():
                raise ValueError(
                    f"Argument '{arg_name}' must be a single value and not an array of "
                    f"shape {val.shape}."
                )
            locals()[arg_name] = val

        # validate gamma
        if not isinstance(gamma, Real):
            raise TypeError(
                f"Expected int or float for argument 'gamma', but got "
                f"{type(gamma)}."
            )

        if density.unit.physical_type == u.physical.mass_density:
            _rho = density
        else:
            _rho = (ion.mass + ion.charge_number * electron.mass) * density

        # Alfvén speed
        self._v_a = Alfven_speed(B, _rho)
        # sound speed
        self._c_s = np.sqrt(gamma * k_B * T / ion.mass).to(u.m / u.s)
        # magnetosonic speed
        self._c_ms = np.sqrt(self._v_a**2 + self._c_s**2)
        # gyrofrequency
        self._oc = gyrofrequency(B, ion)

    @property
    def alfven_speed(self):
        """The Alfvén speed of the plasma."""
        return self._v_a

    @property
    def sound_speed(self) -> u.m / u.s:
        """The sound speed of the plasma."""
        return self._c_s

    @property
    def magnetosonic_speed(self) -> u.m / u.s:
        r"""
        The magnetosonic speed of the plasma.

        Defined as :math:`c_{ms} = \sqrt{v_a^2 + c_s^2}` where
        :math:`v_a` is the Alfvén speed and :math:`c_s` is the sound speed
        """
        return self._c_ms

    @staticmethod
    @validate_quantities
    def _validate_k_theta(k: u.rad / u.m, theta: u.rad):
        """Validate and return wavenumber and angle."""
        # validate argument k
        k = k.squeeze()
        if k.ndim not in [0, 1]:
            raise ValueError(
                f"Argument 'k' needs to be a single-valued or 1D array astropy Quantity,"
                f" got array of shape {k.shape}."
            )
        if np.any(k <= 0):
            raise ValueError("Argument 'k' can not be a or have negative values.")

        # validate argument theta
        theta = theta.squeeze()
        if theta.ndim not in [0, 1]:
            raise ValueError(
                f"Argument 'theta' needs to be a single-valued or 1D array astropy "
                f"Quantity, got array of shape {k.shape}."
            )

        # return theta and k as coordinate arrays
        return np.meshgrid(theta, k)

    @validate_quantities
    def _validate_angular_frequency(self, omega: u.rad / u.s):
        """Validate and return angular frequency."""
        omega_oc_max = np.max(omega / self._oc)
        if omega_oc_max > 0.1:
            warnings.warn(
                f"The calculation produced a high-frequency wave (ω/ω_c == "
                f"{omega_oc_max:.3f}), which violates the low-frequency (ω/ω_c << 1) "
                f"assumption of the dispersion relation.",
                PhysicsWarning,
            )
        return np.squeeze(omega)

    @abstractmethod
    def angular_frequency(self, k: u.rad / u.m, theta: u.rad):
        r"""
        Calculate the angular frequency of magnetohydrodynamic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`, single valued or 1-D array
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of length :math:`N`.
        theta : `~astropy.units.Quantity`, single valued or 1-D array
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units must be
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        omega : `~astropy.units.Quantity`
            An :math:`N × M` array of computed wave frequencies in units
            rad/s.

        Raises
        ------
        ~astropy.units.UnitTypeError
            If applicable arguments do not have units convertible to the
            expected units.

        ValueError
            If ``k`` is negative or zero.

        ValueError
            If ``k`` or ``theta`` are not single valued or a 1-D array.

        Warns
        -----
        : `~plasmapy.utils.exceptions.PhysicsWarning`
            When the computed wave frequencies violate the low-frequency
            (:math:`ω/ω_c ≪ 1`) assumption of the dispersion relation.
        """

    def phase_velocity(self, k: u.rad / u.m, theta: u.rad):
        r"""
        Calculate the phase velocities of magnetohydrodynamic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`, single valued or 1-D array
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of length :math:`N`.
        theta : `~astropy.units.Quantity`, single valued or 1-D array
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units must be
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        phase_velocity : `~astropy.units.Quantity`
            An :math:`N × M` array of computed phase velocities in units
            m/s.

        Raises
        ------
        ~astropy.units.UnitTypeError
            If applicable arguments do not have units convertible to the
            expected units.

        ValueError
            If ``k`` is negative or zero.

        ValueError
            If ``k`` or ``theta`` are not single valued or a 1-D array.

        Warns
        -----
        : `~plasmapy.utils.exceptions.PhysicsWarning`
            When the computed wave frequencies violate the low-frequency
            (:math:`ω/ω_c ≪ 1`) assumption of the dispersion relation.
        """
        return self.angular_frequency(k, theta) / k


class AlfvenWave(AbstractMHDWave):
    r"""
    A class to represent magnetohydrodynamic Alfvén waves.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to T.
    density : `~astropy.units.Quantity`
        Either the ion number density :math:`n_i` in units convertible
        to m\ :sup:`-3` or the total mass density :math:`ρ` in units
        convertible to kg m\ :sup:`-3`\ .
    ion : `str` or `~plasmapy.particles.particle_class.Particle`
        Representation of the ion species (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.
    T : `~astropy.units.Quantity`, |keyword-only|, optional
        The plasma temperature in units of K or eV, which defaults
        to zero.
    gamma : `float` or `int`, |keyword-only|, optional
        The adiabatic index for the plasma, which defaults to 3/5.
    mass_numb : integer, |keyword-only|, optional
        The mass number corresponding to ``ion``.
    Z : `float` or int, |keyword-only|, optional
        The charge number corresponding to ``ion``.

    Raises
    ------
    TypeError
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    TypeError
        If ``ion`` is not |particle-like|.

    TypeError
        If ``gamma`` or ``Z`` are not of type `int` or `float`.

    TypeError
        If ``mass_numb`` is not of type `int`.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    ValueError
        If any of ``B``, ``density``, or ``T`` is negative.

    ValueError
        If ``ion`` is not of category ion or element.

    ValueError
        If ``B``, ``density``, or ``T`` are not single valued
        `astropy.units.Quantity` (i.e. an array).

    See Also
    --------
    ~plasmapy.dispersion.analytical.mhd_wave_class.FastMagnetosonicWave
    ~plasmapy.dispersion.analytical.mhd_wave_class.SlowMagnetosonicWave

    Examples
    --------
    >>> from astropy import units as u
    >>> from plasmapy.dispersion.analytical import AlfvenWave
    >>> alfven = AlfvenWave(1e-3 * u.T, 1e16 * u.m ** -3, "p+")
    >>> alfven.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 2.18060973 rad / s>
    >>> alfven.phase_velocity(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 218060.97295233 m / s>
    >>> alfven.alfven_speed
    <Quantity 218060.97295233 m / s>
    """

    def angular_frequency(self, k: u.rad / u.m, theta: u.rad):
        r"""
        Calculate the angular frequency of magnetohydrodynamic
        Alfvén waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`, single valued or 1-D array
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of length :math:`N`.
        theta : `~astropy.units.Quantity`, single valued or 1-D array
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units must be
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        omega : `~astropy.units.Quantity`
            An :math:`N × M` array of computed wave frequencies in units
            rad/s.

        Raises
        ------
        ~astropy.units.UnitTypeError
            If applicable arguments do not have units convertible to the
            expected units.

        ValueError
            If ``k`` is negative or zero.

        ValueError
            If ``k`` or ``theta`` are not single valued or a 1-D array.

        Warns
        -----
        : `~plasmapy.utils.exceptions.PhysicsWarning`
            When the computed wave frequencies violate the low-frequency
            (:math:`ω/ω_c ≪ 1`) assumption of the dispersion relation.

        Notes
        -----
        The angular frequency :math:`\omega` of a magnetohydrodynamic
        Alfvén wave is given by

        .. math::

            \omega = k v_A \cos\theta

        where :math:`k` is the wavenumber, :math:`v_A` is the Alfvén
        speed, and :math:`theta` is the angle between the wavevector and
        the equilibrium magnetic field.

        Examples
        --------
        >>> from astropy import units as u
        >>> from plasmapy.dispersion.analytical import AlfvenWave
        >>> alfven = AlfvenWave(1e-3 * u.T, 1e16 * u.m ** -3, "p+")
        >>> alfven.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
        <Quantity 2.18060973 rad / s>
        >>> alfven.angular_frequency([1e-5, 2e-4] * (u.rad / u.m), 0 * u.deg)
        <Quantity [ 2.18060973, 43.61219459] rad / s>
        >>> alfven.angular_frequency(1e-5 * u.rad / u.m, [0, 45, 90] * u.deg)
        <Quantity [2.18060973e+00, 1.54192393e+00, 1.33523836e-16] rad / s>
        >>> alfven.angular_frequency([1e-5, 2e-4] * (u.rad / u.m), [0, 45, 90] * u.deg)
        <Quantity [[2.18060973e+00, 1.54192393e+00, 1.33523836e-16],
                   [4.36121946e+01, 3.08384785e+01, 2.67047673e-15]] rad / s>
        """
        theta, k = super()._validate_k_theta(k, theta)
        omega = k * self._v_a * np.cos(theta)
        return super()._validate_angular_frequency(omega)


class FastMagnetosonicWave(AbstractMHDWave):
    r"""
    A class to represent fast magnetosonic waves.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to T.
    density : `~astropy.units.Quantity`
        Either the ion number density :math:`n_i` in units convertible
        to m\ :sup:`-3` or the total mass density :math:`ρ` in units
        convertible to kg m\ :sup:`-3`\ .
    ion : `str` or `~plasmapy.particles.particle_class.Particle`
        Representation of the ion species (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.
    T : `~astropy.units.Quantity`, |keyword-only|, optional
        The plasma temperature in units of K or eV, which defaults
        to zero.
    gamma : `float` or `int`, |keyword-only|, optional
        The adiabatic index for the plasma, which defaults to 3/5.
    mass_numb : integer, |keyword-only|, optional
        The mass number corresponding to ``ion``.
    Z : `float` or int, |keyword-only|, optional
        The charge number corresponding to ``ion``.

    Raises
    ------
    TypeError
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    TypeError
        If ``ion`` is not |particle-like|.

    TypeError
        If ``gamma`` or ``Z`` are not of type `int` or `float`.

    TypeError
        If ``mass_numb`` is not of type `int`.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    ValueError
        If any of ``B``, ``density``, or ``T`` is negative.

    ValueError
        If ``ion`` is not of category ion or element.

    ValueError
        If ``B``, ``density``, or ``T`` are not single-valued
        `astropy.units.Quantity` (i.e. an array).

    See Also
    --------
    ~plasmapy.dispersion.analytical.mhd_wave_class.AlfvenWave
    ~plasmapy.dispersion.analytical.mhd_wave_class.SlowMagnetosonicWave

    Examples
    --------
    >>> from astropy import units as u
    >>> from plasmapy.dispersion.analytical import FastMagnetosonicWave
    >>> fast = FastMagnetosonicWave(1e-3 * u.T, 1e16 * u.m ** -3, "p+", T=2.5e6 * u.K)
    >>> fast.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 2.18060973 rad / s>
    >>> fast.phase_velocity(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 218060.97295233 m / s>
    >>> fast.alfven_speed
    <Quantity 218060.97295233 m / s>
    """

    def angular_frequency(self, k: u.rad / u.m, theta: u.rad):
        r"""
        Calculate the angular frequency of a fast magnetosonic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`, single valued or 1-D array
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of length :math:`N`.
        theta : `~astropy.units.Quantity`, single valued or 1-D array
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units must be
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        omega : `~astropy.units.Quantity`
            An :math:`N × M` array of computed wave frequencies in units
            rad/s.

        Raises
        ------
        ~astropy.units.UnitTypeError
            If applicable arguments do not have units convertible to the
            expected units.

        ValueError
            If ``k`` is negative or zero.

        ValueError
            If ``k`` or ``theta`` are not single valued or a 1-D array.

        Warns
        -----
        : `~plasmapy.utils.exceptions.PhysicsWarning`
            When the computed wave frequencies violate the low-frequency
            (:math:`ω/ω_c ≪ 1`) assumption of the dispersion relation.

        Notes
        -----
        The angular frequency :math:`\omega` of a fast magnetosonic wave
        is given by the equation

        .. math::

            \omega^2 = \frac{k^2}{2} \left(c_{ms}^2 + \sqrt{c_{ms}^4 - 4 v_A^2 c_s^2 \cos^2 \theta}\right)

        where :math:`k` is the wavenumber, :math:`v_A` is the Alfvén
        speed, :math:`c_s` is the ideal sound speed,
        :math:`c_{ms} = \sqrt{v_A^2 + c_s^2}` is the magnetosonic speed,
        and :math:`theta` is the angle between the wavevector and the
        equilibrium magnetic field.

        Examples
        --------
        >>> from astropy import units as u
        >>> from plasmapy.dispersion.analytical import FastMagnetosonicWave
        >>> fast = FastMagnetosonicWave(1e-3 * u.T, 1e16 * u.m ** -3, "p+", T=2.5e6 * u.K)
        >>> fast.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
        <Quantity 2.18060973 rad / s>
        >>> fast.angular_frequency([1e-5, 2e-4] * (u.rad / u.m), 0 * u.deg)
        <Quantity [ 2.18060973, 43.61219459] rad / s>
        >>> fast.angular_frequency(1e-5 * u.rad / u.m, [0, 45, 90] * u.deg)
        <Quantity [2.18060973, 2.65168984, 2.86258485] rad / s>
        >>> fast.angular_frequency([1e-5, 2e-4] * (u.rad / u.m), [0, 45, 90] * u.deg)
        <Quantity [[ 2.18060973,  2.65168984,  2.86258485],
                   [43.61219459, 53.03379678, 57.251697  ]] rad / s>
        """
        theta, k = super()._validate_k_theta(k, theta)
        omega = k * np.sqrt(
            (
                self._c_ms**2
                + np.sqrt(
                    (self._c_ms**2 + 2 * self._v_a * self._c_s * np.cos(theta))
                    * (self._c_ms**2 - 2 * self._v_a * self._c_s * np.cos(theta))
                )
            )
            / 2
        )
        return super()._validate_angular_frequency(omega)


class SlowMagnetosonicWave(AbstractMHDWave):
    r"""
    A class to represent slow magnetosonic waves.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to T.
    density : `~astropy.units.Quantity`
        Either the ion number density :math:`n_i` in units convertible
        to m\ :sup:`-3` or the total mass density :math:`ρ` in units
        convertible to kg m\ :sup:`-3`\ .
    ion : `str` or `~plasmapy.particles.particle_class.Particle`
        Representation of the ion species (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.
    T : `~astropy.units.Quantity`, |keyword-only|, optional
        The plasma temperature in units of K or eV, which defaults
        to zero.
    gamma : `float` or `int`, |keyword-only|, optional
        The adiabatic index for the plasma, which defaults to 3/5.
    mass_numb : integer, |keyword-only|, optional
        The mass number corresponding to ``ion``.
    Z : `float` or int, |keyword-only|, optional
        The charge number corresponding to ``ion``.

    Raises
    ------
    TypeError
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    TypeError
        If ``ion`` is not |particle-like|.

    TypeError
        If ``gamma`` or ``Z`` are not of type `int` or `float`.

    TypeError
        If ``mass_numb`` is not of type `int`.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    ValueError
        If any of ``B``, ``density``, or ``T`` is negative.

    ValueError
        If ``ion`` is not of category ion or element.

    ValueError
        If ``B``, ``density``, or ``T`` are not single-valued
        `astropy.units.Quantity` (i.e. an array).

    See Also
    --------
    ~plasmapy.dispersion.analytical.mhd_wave_class.AlfvenWave
    ~plasmapy.dispersion.analytical.mhd_wave_class.FastMagnetosonicWave

    Examples
    --------
    >>> from astropy import units as u
    >>> from plasmapy.dispersion.analytical import SlowMagnetosonicWave
    >>> slow = SlowMagnetosonicWave(1e-3 * u.T, 1e16 * u.m ** -3, "p+", T=2.5e6 * u.K)
    >>> slow.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 1.85454394 rad / s>
    >>> slow.phase_velocity(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 185454.39417735 m / s>
    >>> slow.sound_speed
    <Quantity 185454.39417735 m / s>
    """

    def angular_frequency(self, k: u.rad / u.m, theta: u.rad):
        r"""
        Calculate the angular frequency of slow magnetosonic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`, single valued or 1-D array
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of length :math:`N`.
        theta : `~astropy.units.Quantity`, single valued or 1-D array
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units must be
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        omega : `~astropy.units.Quantity`
            An :math:`N × M` array of computed wave frequencies in units
            rad/s.

        Raises
        ------
        ~astropy.units.UnitTypeError
            If applicable arguments do not have units convertible to the
            expected units.

        ValueError
            If ``k`` is negative or zero.

        ValueError
            If ``k`` or ``theta`` are not single valued or a 1-D array.

        Warns
        -----
        : `~plasmapy.utils.exceptions.PhysicsWarning`
            When the computed wave frequencies violate the low-frequency
            (:math:`ω/ω_c ≪ 1`) assumption of the dispersion relation.

        Notes
        -----
        The angular frequency :math:`\omega` of a slow magnetosonic wave
        is given by the equation

        .. math::

            \omega^2 = \frac{k^2}{2} \left(c_{ms}^2 - \sqrt{c_{ms}^4 - 4 v_A^2 c_s^2 \cos^2 \theta}\right)

        where :math:`k` is the wavenumber, :math:`v_A` is the Alfvén
        speed, :math:`c_s` is the ideal sound speed,
        :math:`c_{ms} = \sqrt{v_A^2 + c_s^2}` is the magnetosonic speed,
        and :math:`theta` is the angle between the wavevector and the
        equilibrium magnetic field.

        Examples
        --------
        >>> from astropy import units as u
        >>> from plasmapy.dispersion.analytical import SlowMagnetosonicWave
        >>> slow = SlowMagnetosonicWave(1e-3 * u.T, 1e16 * u.m ** -3, "p+", T=2.5e6 * u.K)
        >>> slow.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
        <Quantity 1.85454394 rad / s>
        >>> slow.angular_frequency([1e-5, 2e-4] * (u.rad / u.m), 0 * u.deg)
        <Quantity [ 1.85454394, 37.09087884] rad / s>
        >>> slow.angular_frequency(1e-5 * u.rad / u.m, [0, 45, 90] * u.deg)
        <Quantity [1.85454394, 1.07839372, 0.        ] rad / s>
        >>> slow.angular_frequency([1e-5, 2e-4] * (u.rad / u.m), [0, 45, 90] * u.deg)
        <Quantity [[ 1.85454394,  1.07839372,  0.        ],
                   [37.09087884, 21.56787445,  0.        ]] rad / s>
        """
        theta, k = super()._validate_k_theta(k, theta)
        omega = k * np.sqrt(
            (
                self._c_ms**2
                - np.sqrt(
                    (self._c_ms**2 + 2 * self._v_a * self._c_s * np.cos(theta))
                    * (self._c_ms**2 - 2 * self._v_a * self._c_s * np.cos(theta))
                )
            )
            / 2
        )
        return super()._validate_angular_frequency(omega)


def mhd_waves(*args, **kwargs):
    r"""
    Returns a dictionary containing objects of the three
    magnetohydrodynamic waves with identical parameters.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to T.
    density : `~astropy.units.Quantity`
        Either the ion number density :math:`n_i` in units convertible
        to m\ :sup:`-3` or the total mass density :math:`ρ` in units
        convertible to kg m\ :sup:`-3`\ .
    ion : `str` or `~plasmapy.particles.particle_class.Particle`
        Representation of the ion species (e.g., ``'p'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.
    T : `~astropy.units.Quantity`, |keyword-only|, optional
        The plasma temperature in units of K or eV, which defaults
        to zero.
    gamma : `float` or `int`, |keyword-only|, optional
        The adiabatic index for the plasma, which defaults to 3/5.
    mass_numb : integer, |keyword-only|, optional
        The mass number corresponding to ``ion``.
    Z : `float` or int, |keyword-only|, optional
        The charge number corresponding to ``ion``.

    Returns
    -------
    mhd_waves : Dict[str, `~plasmapy.dispersion.analytical.mhd_wave_class.AlfvenWave` or `~plasmapy.dispersion.analytical.mhd_wave_class.FastMagnetosonicWave` or `~plasmapy.dispersion.analytical.mhd_wave_class.SlowMagnetosonicWave`]
        A dictionary of magnetohydrodynamic-wave objects. The
        dictionary contains three keys: ``'alfven'`` for the Alfvén
        mode, ``'fast'`` for the fast magnetosonic mode, and
        ``'slow'`` for the slow magnetosonic mode.  The value for
        each key will be of type

    Raises
    ------
    TypeError
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    TypeError
        If ``ion`` is not |particle-like|.

    TypeError
        If ``gamma`` or ``Z`` are not of type `int` or `float`.

    TypeError
        If ``mass_numb`` is not of type `int`.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    ValueError
        If any of ``B``, ``density``, or ``T`` is negative.

    ValueError
        If ``ion`` is not of category ion or element.

    ValueError
        If ``B``, ``density``, or ``T`` are not single-valued
        `astropy.units.Quantity` (i.e. an array).
    """
    return {
        "alfven": AlfvenWave(*args, **kwargs),
        "fast": FastMagnetosonicWave(*args, **kwargs),
        "slow": SlowMagnetosonicWave(*args, **kwargs),
    }