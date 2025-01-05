"""Magnetohydrodynamic (MHD) wave dispersion relations."""

__all__ = [
    "AbstractMHDWave",
    "AlfvenWave",
    "FastMagnetosonicWave",
    "SlowMagnetosonicWave",
    "mhd_waves",
]

import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from numbers import Real

import astropy.units as u
import numpy as np
from astropy.constants.si import k_B

from plasmapy.formulary.dimensionless import beta
from plasmapy.formulary.frequencies import gyrofrequency, plasma_frequency
from plasmapy.formulary.speeds import Alfven_speed
from plasmapy.particles import ParticleLike, electron, particle_input
from plasmapy.utils.decorators import check_relativistic, validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning


class AbstractMHDWave(ABC):
    """Abstract base class for magnetohydrodynamic waves."""

    @particle_input
    @validate_quantities(
        B={"can_be_negative": False},
        density={"can_be_negative": False},
        T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    )
    def __init__(
        self,
        B: u.Quantity[u.T],
        density: (u.m**-3, u.kg / u.m**3),
        ion: ParticleLike,
        *,
        T: u.Quantity[u.K] = 0 * u.K,
        gamma: float = 5 / 3,
        mass_numb: int | None = None,
        Z: float | None = None,
    ) -> None:
        # validate arguments
        for arg_name in ("B", "density", "T"):
            val = locals()[arg_name].squeeze()
            if val.shape != ():
                raise ValueError(
                    f"Argument '{arg_name}' must be a single value and not an array of "
                    f"shape {val.shape}."
                )
            locals()[arg_name] = val

        if not isinstance(gamma, Real):
            raise TypeError(
                f"Expected int or float for argument 'gamma', but got "
                f"{type(gamma)}."
            )

        if density.unit.physical_type == u.physical.mass_density:
            _n = density / (ion.mass + ion.charge_number * electron.mass)
            _rho = density
        else:
            _n = density
            _rho = (ion.mass + ion.charge_number * electron.mass) * density

        self._Alfven_speed = Alfven_speed(B, _rho)
        self._sound_speed = np.sqrt(gamma * k_B * T / ion.mass).to(u.m / u.s)
        self._magnetosonic_speed = np.sqrt(self._Alfven_speed**2 + self._sound_speed**2)
        self._beta = beta(T, _n, B)
        self._gyrofrequency = gyrofrequency(B, ion)
        self._plasma_frequency = plasma_frequency(_n, ion)

    @property
    def alfven_speed(self) -> u.Quantity[u.m / u.s]:
        """The Alfvén speed of the plasma."""
        return self._Alfven_speed

    @property
    def sound_speed(self) -> u.Quantity[u.m / u.s]:
        r"""
        The sound speed of the plasma.

        Defined as :math:`c_s = \sqrt{γ k_B T / m_i}` where
        :math:`gamma` is the adiabatic index of the fluid,
        :math:`k_B` is the Boltzmann constant, :math:`T` is the
        temperature of the fluid, and :math:`m_i` is the mass of
        the ion species in the fluid.
        """
        return self._sound_speed

    @property
    def magnetosonic_speed(self) -> u.Quantity[u.m / u.s]:
        r"""
        The magnetosonic speed of the plasma.

        Defined as :math:`c_{ms} = \sqrt{v_A^2 + c_s^2}` where
        :math:`v_A` is the Alfvén speed and :math:`c_s` is the sound speed.
        """
        return self._magnetosonic_speed

    @property
    def beta(self):
        """The ratio of thermal pressure to magnetic pressure."""
        return self._beta

    @staticmethod
    @validate_quantities
    def _validate_k_theta(
        k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]
    ) -> list[u.Quantity]:
        """Validate and return wavenumber and angle."""
        # validate argument k
        k = k.squeeze()
        if k.ndim not in {0, 1}:
            raise ValueError(
                f"Argument 'k' needs to be a single-valued or 1D array astropy Quantity,"
                f" got array of shape {k.shape}."
            )
        if np.any(k <= 0):
            raise ValueError("Argument 'k' cannot be a or have negative values.")

        # validate argument theta
        theta = theta.squeeze()
        if theta.ndim not in {0, 1}:
            raise ValueError(
                f"Argument 'theta' needs to be a single-valued or 1D array astropy "
                f"Quantity, got array of shape {k.shape}."
            )

        # return theta and k as coordinate arrays
        return np.meshgrid(theta, k)

    @validate_quantities
    def _validate_angular_frequency(self, omega: u.Quantity[u.rad / u.s]):
        """Validate and return angular frequency."""
        omega_gyrofrequency_max = np.max(omega / self._gyrofrequency)
        omega_plasma_frequency_max = np.max(omega / self._plasma_frequency)
        if omega_gyrofrequency_max > 0.1 or omega_plasma_frequency_max > 0.1:
            warnings.warn(
                f"The calculation produced a high-frequency wave (ω/ω_c == {omega_gyrofrequency_max:.3f} "
                f"and ω/ω_c == {omega_plasma_frequency_max:.3f}), which violates the low-frequency "
                f"assumption of the dispersion relation (ω/ω_c ≪ 1 and ω/ω_p ≪ 1).",
                PhysicsWarning,
            )
        return np.squeeze(omega)

    @abstractmethod
    def angular_frequency(
        self,
        k: u.Quantity[u.rad / u.m],
        theta: u.Quantity[u.rad],
    ) -> u.Quantity[u.rad / u.s]:
        r"""
        Calculate the angular frequency of magnetohydrodynamic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of length :math:`N`.

        theta : `~astropy.units.Quantity`
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units
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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.
        """

    @abstractmethod
    @check_relativistic
    @validate_quantities
    def group_velocity(
        self, k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]
    ) -> u.Quantity[u.m / u.s]:
        r"""
        Calculate the group velocities of magnetohydrodynamic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of length :math:`N`.

        theta : `~astropy.units.Quantity`
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        group_velocity : `~astropy.units.Quantity` of shape ``(2, N, M)``
            An array of group_velocities in units m/s with shape
            :math:`2 × N × M`. The first dimension maps to the
            two coordinate arrays in the direction of ``k`` and in
            the direction of increasing ``theta``, the second
            dimension maps to the ``k`` array, and the third dimension
            maps to the ``theta`` array.

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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.

        Notes
        -----
        The group velocity :math:`\mathbf{v}_g` is given by

        .. math::

        \mathbf{v}_g = \frac{dω}{d\mathbf{k}}
            = \hat{\mathbf{k}} \frac{∂ω}{∂ k}
                + \hat{\mathbf{θ}} \frac{∂ v_{ph}}{∂θ}

        where :math:`ω` is the angular frequency, :math:`\mathbf{k}` is
        the wavevector, :math:`θ` is the angle between :math:`\mathbf{k}`
        and the unperturbed magnetic field, and :math:`v_{ph}` is the
        phase velocity.
        """

    @check_relativistic
    @validate_quantities
    def phase_velocity(
        self, k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]
    ) -> u.Quantity[u.m / u.s]:
        r"""
        Calculate the phase velocities of magnetohydrodynamic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of size :math:`N`.

        theta : `~astropy.units.Quantity`
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        phase_velocity : `~astropy.units.Quantity`
            An :math:`N × M` array of computed phase velocities in units
            of m/s.

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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.
        """
        angular_frequency = self.angular_frequency(k, theta)
        theta, k = self._validate_k_theta(k, theta)
        return np.squeeze(angular_frequency / k)


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

    ion : |particle-like|
        Representation of the ion species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.

    T : `~astropy.units.Quantity`, |keyword-only|, optional
        The plasma temperature in units of K or eV, which defaults
        to zero.

    gamma : `float` or `int`, |keyword-only|, default: 5/3
        The adiabatic index for the plasma.

    mass_numb : `int`, |keyword-only|, optional
        The mass number corresponding to ``ion``.

    Z : `float` or `int`, |keyword-only|, optional
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
    ~plasmapy.dispersion.analytical.mhd_waves_.FastMagnetosonicWave
    ~plasmapy.dispersion.analytical.mhd_waves_.SlowMagnetosonicWave

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.dispersion.analytical import AlfvenWave
    >>> alfven = AlfvenWave(1e-3 * u.T, 1e16 * u.m**-3, "p+")
    >>> alfven.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 2.18060973 rad / s>
    >>> alfven.phase_velocity(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 218060.97295233 m / s>
    >>> alfven.alfven_speed
    <Quantity 218060.97295233 m / s>
    """

    def angular_frequency(self, k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]):
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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.

        Notes
        -----
        The angular frequency :math:`ω` of a magnetohydrodynamic
        Alfvén wave is given by

        .. math::

            ω = k v_A \cosθ

        where :math:`k` is the wavenumber, :math:`v_A` is the Alfvén
        speed, and :math:`θ` is the angle between the wavevector and
        the equilibrium magnetic field.

        Examples
        --------
        >>> import astropy.units as u
        >>> from plasmapy.dispersion.analytical import AlfvenWave
        >>> alfven = AlfvenWave(1e-3 * u.T, 1e16 * u.m**-3, "p+")
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
        omega = k * self._Alfven_speed * np.abs(np.cos(theta))
        return super()._validate_angular_frequency(omega)

    def group_velocity(self, k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]):
        r"""
        Calculate the group velocities of magnetohydrodynamic Alfvén
        waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`, single valued or 1-D array
            Wavenumber in units convertible to rad/m`. Either single
            valued or 1-D array of length :math:`N`.

        theta : `~astropy.units.Quantity`
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        group_velocity : `~astropy.units.Quantity` of shape ``(2, N, M)``
            An array of group_velocities in units m/s with shape
            :math:`2 × N × M`. The first dimension maps to the
            two coordinate arrays in the direction of ``k`` and in
            the direction of increasing ``theta``, the second
            dimension maps to the ``k`` array, and the third dimension
            maps to the ``theta`` array.

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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.

        Notes
        -----
        The group velocity :math:`\mathbf{v}_g` is given by

        .. math::

        \mathbf{v}_g = \frac{dω}{d\mathbf{k}}
            = ± \hat{\mathbf{B}} v_{A}

        where :math:`\hat{\mathbf{B}}` is the unit vector in the
        direction of the unperturbed magnetic field and :math:`v_A` is
        the Alfvén speed.
        """
        phase_velocity = self.phase_velocity(k, theta)
        theta, k = super()._validate_k_theta(k, theta)
        return [
            phase_velocity,
            -phase_velocity * np.tan(theta),
        ]


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

    ion : |particle-like|
        Representation of the ion species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.

    T : `~astropy.units.Quantity`, |keyword-only|, optional
        The plasma temperature in units of K or eV, which defaults
        to zero.

    gamma : `float` or `int`, |keyword-only|, default: 5/3
        The adiabatic index for the plasma.

    mass_numb : `int`, |keyword-only|, optional
        The mass number corresponding to ``ion``.

    Z : `float` or `int`, |keyword-only|, optional
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
    ~plasmapy.dispersion.analytical.mhd_waves_.AlfvenWave
    ~plasmapy.dispersion.analytical.mhd_waves_.SlowMagnetosonicWave

    Notes
    -----
    Fast magnetosonic waves are also referred to as fast magnetoacoustic
    waves.

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.dispersion.analytical import FastMagnetosonicWave
    >>> fast = FastMagnetosonicWave(1e-3 * u.T, 1e16 * u.m**-3, "p+", T=2.5e6 * u.K)
    >>> fast.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 2.18060973 rad / s>
    >>> fast.phase_velocity(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 218060.97295233 m / s>
    >>> fast.alfven_speed
    <Quantity 218060.97295233 m / s>
    """

    def angular_frequency(self, k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]):
        r"""
        Calculate the angular frequency of a fast magnetosonic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`
            Wavenumber in units convertible to rad/m`.  Either single
            valued or 1-D array of length :math:`N`.

        theta : `~astropy.units.Quantity`
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units
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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.

        Notes
        -----
        The angular frequency :math:`ω` of a fast magnetosonic wave
        is given by the equation

        .. math::

            ω^2 = \frac{k^2}{2} \left(
               c_{ms}^2 + \sqrt{c_{ms}^4 - 4 v_A^2 c_s^2 \cos^2 θ}
            \right)

        where :math:`k` is the wavenumber, :math:`v_A` is the Alfvén
        speed, :math:`c_s` is the ideal sound speed,
        :math:`c_{ms} = \sqrt{v_A^2 + c_s^2}` is the magnetosonic speed,
        and :math:`θ` is the angle between the wavevector and the
        equilibrium magnetic field.

        Examples
        --------
        >>> import astropy.units as u
        >>> from plasmapy.dispersion.analytical import FastMagnetosonicWave
        >>> fast = FastMagnetosonicWave(1e-3 * u.T, 1e16 * u.m**-3, "p+", T=2.5e6 * u.K)
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
                self._magnetosonic_speed**2
                + np.sqrt(
                    (
                        self._magnetosonic_speed**2
                        + 2 * self._Alfven_speed * self._sound_speed * np.cos(theta)
                    )
                    * (
                        self._magnetosonic_speed**2
                        - 2 * self._Alfven_speed * self._sound_speed * np.cos(theta)
                    )
                )
            )
            / 2
        )
        return super()._validate_angular_frequency(omega)

    def group_velocity(self, k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]):
        r"""
        Calculate the group velocities of fast magnetosonic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`
            Wavenumber in units convertible to rad/m`. Either single
            valued or 1-D array of length :math:`N`.

        theta : `~astropy.units.Quantity`
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        group_velocity : `~astropy.units.Quantity` of shape ``(2, N, M)``
            An array of group_velocities in units m/s with shape
            :math:`2 × N × M`. The first dimension maps to the
            two coordinate arrays in the direction of ``k`` and in
            the direction of increasing ``theta``, the second
            dimension maps to the ``k`` array, and the third dimension
            maps to the ``theta`` array.

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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.

        Notes
        -----
        The group velocity :math:`\mathbf{v}_g` is given by

        .. math::

        \mathbf{v}_g = \frac{dω}{d\mathbf{k}}
            = \hat{\mathbf{k}} v_{ph}
                + \hat{\mathbf{θ}} \frac{∂ v_{ph}}{∂θ}

        where :math:`ω` is the angular frequency, :math:`\mathbf{k}` is
        the wavevector, :math:`θ` is the angle between :math:`\mathbf{k}`
        and the unperturbed magnetic field, and :math:`v_{ph}` is the
        phase velocity.
        """
        phase_velocity = self.phase_velocity(k, theta)
        theta, k = super()._validate_k_theta(k, theta)
        return [
            phase_velocity,
            np.squeeze(
                self._Alfven_speed**2
                * self._sound_speed**2
                * np.sin(theta)
                * np.cos(theta)
                / (
                    phase_velocity
                    * (2 * phase_velocity**2 - self._magnetosonic_speed**2)
                )
            ),
        ]


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
    ion : |particle-like|
        Representation of the ion species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.
    T : `~astropy.units.Quantity`, |keyword-only|, optional
        The plasma temperature in units of K or eV, which defaults
        to zero.
    gamma : `float` or `int`, |keyword-only|, optional
        The adiabatic index for the plasma, which defaults to 3/5.
    mass_numb : `int`, |keyword-only|, optional
        The mass number corresponding to ``ion``.
    Z : `float` or `int`, |keyword-only|, optional
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
    ~plasmapy.dispersion.analytical.mhd_waves_.AlfvenWave
    ~plasmapy.dispersion.analytical.mhd_waves_.FastMagnetosonicWave

    Notes
    -----
    Slow magnetosonic waves are also referred to as slow magnetoacoustic
    waves.

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.dispersion.analytical import SlowMagnetosonicWave
    >>> slow = SlowMagnetosonicWave(1e-3 * u.T, 1e16 * u.m**-3, "p+", T=2.5e6 * u.K)
    >>> slow.angular_frequency(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 1.85454394 rad / s>
    >>> slow.phase_velocity(1e-5 * u.rad / u.m, 0 * u.deg)
    <Quantity 185454.39417735 m / s>
    >>> slow.sound_speed
    <Quantity 185454.39417735 m / s>
    """

    def angular_frequency(self, k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]):
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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.

        Notes
        -----
        The angular frequency :math:`ω` of a slow magnetosonic wave
        is given by the equation

        .. math::

            ω^2 = \frac{k^2}{2} \left(c_{ms}^2 - \sqrt{c_{ms}^4 - 4 v_A^2 c_s^2 \cos^2 θ}\right)

        where :math:`k` is the wavenumber, :math:`v_A` is the Alfvén
        speed, :math:`c_s` is the ideal sound speed,
        :math:`c_{ms} = \sqrt{v_A^2 + c_s^2}` is the magnetosonic speed,
        and :math:`θ` is the angle between the wavevector and the
        equilibrium magnetic field.

        Examples
        --------
        >>> import astropy.units as u
        >>> from plasmapy.dispersion.analytical import SlowMagnetosonicWave
        >>> slow = SlowMagnetosonicWave(1e-3 * u.T, 1e16 * u.m**-3, "p+", T=2.5e6 * u.K)
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
                self._magnetosonic_speed**2
                - np.sqrt(
                    (
                        self._magnetosonic_speed**2
                        + 2 * self._Alfven_speed * self._sound_speed * np.cos(theta)
                    )
                    * (
                        self._magnetosonic_speed**2
                        - 2 * self._Alfven_speed * self._sound_speed * np.cos(theta)
                    )
                )
            )
            / 2
        )
        return super()._validate_angular_frequency(omega)

    def group_velocity(self, k: u.Quantity[u.rad / u.m], theta: u.Quantity[u.rad]):
        r"""
        Calculate the group velocities of slow magnetosonic waves.

        Parameters
        ----------
        k : `~astropy.units.Quantity`
            Wavenumber in units convertible to rad/m`. Either single
            valued or 1-D array of length :math:`N`.

        theta : `~astropy.units.Quantity`
            The angle of propagation of the wave with respect to the
            magnetic field, :math:`\cos^{-1}(k_z / k)`, in units
            convertible to radians. Either single valued or 1-D array of
            size :math:`M`.

        Returns
        -------
        group_velocity : `~astropy.units.Quantity` of shape ``(2, N, M)``
            An array of group_velocities in units m/s with shape
            :math:`2 × N × M`. The first dimension maps to the
            two coordinate arrays in the direction of ``k`` and in
            the direction of increasing ``theta``, the second
            dimension maps to the ``k`` array, and the third dimension
            maps to the ``theta`` array.

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
            (:math:`ω ≪ ω_c,ω_p`) assumption of the dispersion relation.

        Notes
        -----
        The group velocity :math:`\mathbf{v}_g` is given by

        .. math::

        \mathbf{v}_g = \frac{dω}{d\mathbf{k}}
            = \hat{\mathbf{k}} v_{ph}
                + \hat{\mathbf{θ}} \frac{∂ v_{ph}}{∂θ}

        where :math:`ω` is the angular frequency, :math:`\mathbf{k}` is
        the wavevector, :math:`θ` is the angle between :math:`\mathbf{k}`
        and the unperturbed magnetic field, and :math:`v_{ph}` is the
        phase velocity.
        """
        phase_velocity = self.phase_velocity(k, theta)

        theta, k = super()._validate_k_theta(k, theta)
        group_velocity = np.ones(k.shape) * (0 * u.m / u.s)
        np.squeeze(
            np.divide(
                self._Alfven_speed**2
                * self._sound_speed**2
                * np.sin(theta)
                * np.cos(theta),
                phase_velocity * (2 * phase_velocity**2 - self._magnetosonic_speed**2),
                out=group_velocity,
                where=phase_velocity != 0,
            )
        )

        return [
            phase_velocity,
            group_velocity,
        ]


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

    ion : |particle-like|
        Representation of the ion species (e.g., ``'p+'`` for protons,
        ``'D+'`` for deuterium, ``'He-4 +1'`` for singly ionized
        helium-4, etc.). If no charge state information is provided,
        then the ions are assumed to be singly ionized.

    T : `~astropy.units.Quantity`, |keyword-only|, default: 0 K
        The plasma temperature in units of K or eV.

    gamma : `float` or `int`, |keyword-only|, default: 5/3
        The adiabatic index for the plasma.

    mass_numb : 'int', |keyword-only|, optional
        The mass number corresponding to ``ion``.

    Z : `float` or 'int', |keyword-only|, optional
        The charge number corresponding to ``ion``.

    Returns
    -------
    mhd_waves : namedtuple[str, `~plasmapy.dispersion.analytical.mhd_waves_.AlfvenWave` or `~plasmapy.dispersion.analytical.mhd_waves_.FastMagnetosonicWave` or `~plasmapy.dispersion.analytical.mhd_waves_.SlowMagnetosonicWave`]
        A named tuple of magnetohydrodynamic-wave objects. It
        contains three keys: ``'alfven'`` for the Alfvén
        mode, ``'fast'`` for the fast magnetosonic mode, and
        ``'slow'`` for the slow magnetosonic mode.

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
    MHD_Waves = namedtuple(
        "MHD_Waves", ["alfven_wave", "fast_magnetosonic_wave", "slow_magnetosonic_wave"]
    )
    return MHD_Waves(
        alfven_wave=AlfvenWave(*args, **kwargs),
        fast_magnetosonic_wave=FastMagnetosonicWave(*args, **kwargs),
        slow_magnetosonic_wave=SlowMagnetosonicWave(*args, **kwargs),
    )
