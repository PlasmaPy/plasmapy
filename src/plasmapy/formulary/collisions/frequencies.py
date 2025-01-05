"""Frequencies related to particle collisions."""

__all__ = [
    "SingleParticleCollisionFrequencies",
    "MaxwellianCollisionFrequencies",
    "collision_frequency",
    "fundamental_electron_collision_freq",
    "fundamental_ion_collision_freq",
]

from functools import cached_property

import astropy.units as u
import numpy as np
import scipy
from astropy.constants.si import e, k_B, m_e

from plasmapy import particles
from plasmapy.formulary.collisions import coulomb, lengths, misc
from plasmapy.formulary.speeds import thermal_speed
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.utils.decorators import deprecated, validate_quantities
from plasmapy.utils.exceptions import PhysicsError, PlasmaPyFutureWarning


class SingleParticleCollisionFrequencies:
    r"""
    Compute collision frequencies between test particles (labeled 'a')
    and field particles (labeled 'b').

    Parameters
    ----------
    test_particle : |particle-like|
        The test particle streaming through a background of field
        particles.

    field_particle : |particle-like|
        The background particle that ``test_particle`` is interacting
        with.

    v_drift : `~astropy.units.Quantity`, |keyword-only|
        The relative drift velocity between the test and field
        particles. Cannot be negative.

    T_b : `~astropy.units.Quantity`, |keyword-only|
        The temperature of the background field particles in units
        convertible to kelvin or eV per particle.

    n_b : `~astropy.units.Quantity`, |keyword-only|
        The number density of ``field_particle`` in units convertible to
        m\ :sup:`-3`.

    Coulomb_log : `~astropy.units.Quantity`, |keyword-only|
        The value of the Coulomb logarithm for the interaction.

    Raises
    ------
    `ValueError`
        If the specified ``v_drift`` and ``n_b`` arrays do not have
        equal size.

    Notes
    -----
    The frequency of collisions between a test particle (subscript
    :math:`α`) and a field particle (subscript :math:`β`) each with mass
    :math:`m` and charge :math:`e` are given by four differential
    equations:

        momentum loss: :math:`\frac{d\mathbf{v}_α}{dt}
        = - ν_{s}^{α \backslash β} \mathbf{v}_α`

        transverse diffusion: :math:`\frac{d}{dt}
        \left( \mathbf{v}_α - \overline{\mathbf{v}}_α \right)_⊥^2
        = ν_⊥^{α \backslash β} v_α^2`

        parallel diffusion: :math:`\frac{d}{dt}
        \left(\mathbf{v}_α-\overline{\mathbf{v}}_α\right)_∥^2
        = ν_∥^{α \backslash β} v_α^2`

        energy loss: :math:`\frac{d}{dt} v_α^2
        = -ν_ϵ^{α \backslash β} v_α^2`

    These equations yield the exact formulas:

        momentum loss: :math:`ν_{s}^{α \backslash β}
        = \left( 1 + \frac{m_α}{m_β} \right) ψ
        \left( x^{α \backslash β} \right) ν_{0}^{α \backslash β}`

        transverse diffusion: :math:`ν_⊥^{α \backslash β}
        = 2 \left[ \left( 1-\frac{1}{2x^{α \backslash β}} \right)
        ψ \left( x^{α \backslash β} \right) \
        + ψ' \left( x^{α \backslash β} \right) \right]
        ν_0^{α \backslash β}`

        parallel diffusion: :math:`ν_{∥}^{α \backslash β}
        = \left[ \frac{ψ \left( x^{α \backslash β} \right)
        }{x^{α \backslash β}} \right] ν_{0}^{α \backslash β}`

        energy loss: :math:`ν_ϵ^{α \backslash β}
        = 2\left[ \left( \frac{m_α}{m_β} \right)
        ψ \left( x^{α \backslash β} \right)
        - ψ' \left(x^{α \backslash β} \right)
        \right] ν_0^{α \backslash β}`

    where,

        :math:`ν_{0}^{α \backslash β}
        = \frac{4π e_α^2 e_β^2 λ_{αβ} n_β}{m_α^2 v_α^3}`,

        :math:`x^{α \backslash β} = \frac{m_β v_α^2}{2 k_B T_β}`,

        :math:`ψ \left( x \right) = \frac{2}{\sqrt{π}}
        \int_0^x t^{ \frac{1}{2} } e^{-t} dt`,

        :math:`ψ' \left( x \right) = \frac{dψ}{dx}`,

    and :math:`λ_{α β}` is the Coulomb logarithm for the
    collisions, :math:`n_β` is the number density of the field
    particles, :math:`v_α` is the speed of the test particles relative
    to the field particles, :math:`k_B` is Boltzmann's constant, and
    :math:`T_β` is the temperature of the field particles.

    For values of :math:`x ≪ 1` (the 'slow' or 'thermal' limit) or
    :math:`x ≫ 1` (the 'fast' or 'beam' limit), :math:`ψ` asymptotes to
    zero or one respectively. For simplified expressions in these
    limits, please refer to page 31 of :cite:t:`nrlformulary`.

    Examples
    --------
    >>> import astropy.units as u
    >>> v_drift = 1e5 * u.m / u.s
    >>> n_b = 1e26 * u.m**-3
    >>> T_b = 1e3 * u.eV
    >>> Coulomb_log = 10 * u.dimensionless_unscaled
    >>> frequencies = SingleParticleCollisionFrequencies(
    ...     "e-", "e-", v_drift=v_drift, n_b=n_b, T_b=T_b, Coulomb_log=Coulomb_log
    ... )
    >>> frequencies.energy_loss
    <Quantity -9.69828719e+15 Hz>

    See Also
    --------
    ~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm
    """

    @particles.particle_input
    @validate_quantities(
        v_drift={"can_be_negative": False},
        T_b={
            "can_be_negative": False,
            "equivalencies": u.temperature_energy(),
        },
        n_b={"can_be_negative": False},
    )
    def __init__(
        self,
        test_particle: ParticleLike,
        field_particle: ParticleLike,
        *,
        v_drift: u.Quantity[u.m / u.s],
        T_b: u.Quantity[u.K],
        n_b: u.Quantity[u.m**-3],
        Coulomb_log,
    ) -> None:
        # Note: This class uses CGS units internally to coincide
        #       with our references.  Input is taken in MKS units and
        #       then converted as necessary. Output is in MKS units.

        if (
            isinstance(v_drift, np.ndarray)
            and isinstance(n_b, np.ndarray)
            and v_drift.shape != n_b.shape
        ):
            raise ValueError("Please specify arrays of equal length.")

        self.test_particle = test_particle
        self.field_particle = field_particle
        self.v_drift = v_drift
        self.T_b = T_b
        self.n_b = n_b
        self.Coulomb_log = (
            Coulomb_log
            if isinstance(Coulomb_log, u.Quantity)
            else Coulomb_log * u.dimensionless_unscaled
        )

    @cached_property
    def _mass_ratio(self):
        return self.test_particle.mass / self.field_particle.mass

    @cached_property
    def momentum_loss(self):
        """
        The momentum loss rate due to collisions.
        """
        return (1 + self._mass_ratio) * self.phi * self.Lorentz_collision_frequency

    @cached_property
    def transverse_diffusion(self):
        """The rate of transverse diffusion due to collisions."""
        return (
            2
            * ((1 - 1 / (2 * self.x)) * self.phi + self._phi_prime)
            * self.Lorentz_collision_frequency
        )

    @cached_property
    def parallel_diffusion(self):
        """
        The rate of parallel diffusion due to collisions.
        """
        return (self.phi / self.x) * self.Lorentz_collision_frequency

    @cached_property
    def energy_loss(self):
        """
        The energy loss rate due to collisions.
        """
        return (
            2
            * (self._mass_ratio * self.phi - self._phi_prime)
            * self.Lorentz_collision_frequency
        )

    @cached_property
    def Lorentz_collision_frequency(self):
        r"""
        The Lorentz collision frequency.

        The Lorentz collision frequency (see Ch. 5 of
        :cite:t:`chen:2016`) is given by

        .. math::

            ν = n σ v \ln{Λ}

        where :math:`n` is the particle density, :math:`σ` is the
        collisional cross-section, :math:`v` is the inter-particle
        velocity, and :math:`\ln{Λ}` is the Coulomb logarithm
        accounting for small angle collisions.

        See Equation (2.86) in :cite:t:`callen:unpublished`.

        The Lorentz collision frequency is equivalent to the variable
        :math:`ν_0^{α/β}` on p. 31 of :cite:t:`nrlformulary`.

        This form of the Lorentz collision frequency differs from the
        form found in
        `~plasmapy.formulary.collisions.frequencies.MaxwellianCollisionFrequencies`
        in that :math:`v` is the drift velocity (as opposed to the mean
        thermal velocity between species).
        """

        return (
            4
            * np.pi
            * (self.test_particle.charge_number * e.esu) ** 2
            * (self.field_particle.charge_number * e.esu) ** 2
            * self.Coulomb_log
            * self.n_b
            / (self.test_particle.mass**2 * self.v_drift**3)
        ).to(u.Hz)

    @cached_property
    def x(self) -> u.Quantity[u.dimensionless_unscaled]:
        """
        The ratio of kinetic energy in the test particle to the thermal
        energy of the field particle.

        This parameter determines the regime in which the collision
        falls.

        See Also
        --------
        |SingleParticleCollisionFrequencies|
        """

        x = self.field_particle.mass * self.v_drift**2 / (2 * k_B.cgs * self.T_b)
        return x.to(u.dimensionless_unscaled)

    @staticmethod
    def _phi_integrand(t: u.Quantity[u.dimensionless_unscaled]):  # noqa: ANN205
        """
        The phi integrand used in calculating phi.
        """
        return np.sqrt(t) * np.exp(-t)

    def _phi_explicit(self, x: float):
        """The non-vectorized method for evaluating the integral for phi."""
        integral, _ = scipy.integrate.quad(self._phi_integrand, 0, x)
        return integral

    @cached_property
    def phi(self):
        """
        The parameter phi used in calculating collision frequencies
        calculated using the default error tolerances of
        `~scipy.integrate.quad`.

        For more information refer to page 31 of
        :cite:t:`nrlformulary`.
        """
        vectorized_integral = np.vectorize(self._phi_explicit)

        return 2 / np.pi**0.5 * vectorized_integral(self.x.value)

    @cached_property
    def _phi_prime(self):
        """
        The derivative of phi evaluated at x.
        """
        return 2 / np.pi**0.5 * self._phi_integrand(self.x)


class MaxwellianCollisionFrequencies:
    r"""
    Compute collision frequencies between two slowly flowing Maxwellian
    populations.

    The condition of "slowly flowing", outlined by Eq. 2.133 in
    :cite:t:`callen:unpublished` requires that

    .. math::

        v_{drift} ≪ \sqrt{ v_{T_a}^2 + v_{T_b}^2 }

    where :math:`v_{drift}` is the relative drift between the two
    species, :math:`v_{T_a}` is the thermal velocity of species
    :math:`a`, and :math:`v_{T_b}` is the thermal velocity of species
    :math:`b`.

    Parameters
    ----------
    test_particle : |particle-like|
        The test particle streaming through a background of field
        particles.

    field_particle : |particle-like|
        The background particle being interacted with.

    v_drift : `~astropy.units.Quantity`, default: 0 m/s
        The relative drift between the test and field particles.

    T_a : `~astropy.units.Quantity`
        The temperature of the test particles in units convertible to
        kelvin or eV per particle.

    n_a : `~astropy.units.Quantity`
        The number density of the test particles in units convertible to
        m\ :sup:`-3`\ .

    T_b : `~astropy.units.Quantity`
        The temperature of the background field particles in units
        convertible to kelvin or eV per particle.

    n_b : `~astropy.units.Quantity`
        The number density of the background field particles in units
        convertible to m\ :sup:`-3`\ .

    Coulomb_log : `~astropy.units.Quantity`
        The value of the Coulomb logarithm for the interaction.

    Raises
    ------
    `ValueError`
        If the specified ``v_drift`` and ``T_a`` arrays do not have
        equal size.

    See Also
    --------
    ~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm
    """

    @particles.particle_input
    @validate_quantities(
        v_drift={"can_be_negative": False},
        T_a={
            "can_be_negative": False,
            "equivalencies": u.temperature_energy(),
        },
        n_a={"can_be_negative": False},
        T_b={
            "can_be_negative": False,
            "equivalencies": u.temperature_energy(),
        },
        n_b={"can_be_negative": False},
    )
    def __init__(
        self,
        test_particle: ParticleLike,
        field_particle: ParticleLike,
        *,
        v_drift: u.Quantity[u.m / u.s] = 0 * u.m / u.s,
        T_a: u.Quantity[u.K],
        n_a: u.Quantity[u.m**-3],
        T_b: u.Quantity[u.K],
        n_b: u.Quantity[u.m**-3],
        Coulomb_log: u.Quantity[u.dimensionless_unscaled],
    ) -> None:
        if (
            isinstance(v_drift, np.ndarray)
            and isinstance(T_a, np.ndarray)
            and v_drift.shape != T_a.shape
        ):
            raise ValueError("Please specify arrays of equal length.")

        self.test_particle = test_particle
        self.field_particle = field_particle
        self.v_drift = v_drift
        self.T_a = T_a
        self.n_a = n_a
        self.T_b = T_b
        self.n_b = n_b
        self.Coulomb_log = (
            Coulomb_log
            if isinstance(Coulomb_log, u.Quantity)
            else Coulomb_log * u.dimensionless_unscaled
        )

        self.v_T_a = thermal_speed(self.T_a, self.test_particle)
        self.v_T_b = thermal_speed(self.T_b, self.field_particle)

    @cached_property
    def _mean_thermal_velocity(self):
        """
        Parameter used in enforcing the definition of "slowly flowing"
        Maxwellian particles. See Eq. 2.133 in
        :cite:t:`callen:unpublished`.
        """

        return (self.v_T_a**2 + self.v_T_b**2) ** 0.5

    @cached_property
    def _is_slowly_flowing(self):
        """
        Criteria used in determining whether
        `Maxwellian_avg_ei_collision_freq` and
        `Maxwellian_avg_ii_collision_freq` can be applied to the
        specified species.
        """

        return self.v_drift / self._mean_thermal_velocity < 0.1

    @cached_property
    def Lorentz_collision_frequency(self):
        r"""
        The Lorentz collision frequency.

        The Lorentz collision frequency (see Ch. 5 of
        :cite:t:`chen:2016`) is given by

        .. math::

            ν = n σ v \ln{Λ}

        where :math:`n` is the particle number density, :math:`σ` is the
        collisional cross-section, :math:`v` is the mean thermal
        velocity between particle species (see Equation 2.133 in
        :cite:t:`callen:unpublished`), and :math:`\ln{Λ}` is the Coulomb
        logarithm accounting for small angle collisions.

        See Equation (2.86) in :cite:t:`callen:unpublished`.

        This form of the Lorentz collision frequency differs from the
        form found in |SingleParticleCollisionFrequencies| in that
        :math:`v` is the mean thermal velocity between particle species
        in this method (as opposed to the drift velocity between
        species).
        """

        return (
            4
            * np.pi
            * (self.test_particle.charge_number * e.esu) ** 2
            * (self.field_particle.charge_number * e.esu) ** 2
            * self.Coulomb_log
            * self.n_b
            / (self.test_particle.mass**2 * self._mean_thermal_velocity**3)
        ).to(u.Hz)

    @cached_property
    def Maxwellian_avg_ei_collision_freq(self):
        r"""
        Average momentum relaxation rate for a slowly flowing
        Maxwellian distribution of electrons, relative to a population
        of stationary ions.

        This function assumes that both populations are Maxwellian, and
        :math:`T_i ≲ T_e`.

        :cite:t:`callen:unpublished` provides a derivation of this as an
        average collision frequency between electrons and ions for a
        Maxwellian distribution. It is thus a special case of the
        collision frequency with an averaging factor, and is on many
        occasions in transport theory the most relevant collision
        frequency that has to be considered. It commonly occurs in
        relation to diffusion and resistivity in plasmas.

        Raises
        ------
        `~plasmapy.utils.exceptions.PhysicsError`
            If the test particles are not 'slowly flowing' relative to
            the field particles (see notes).

        `ValueError`
            If the specified interaction is not electron-ion.

        Examples
        --------
        >>> import astropy.units as u
        >>> v_drift = 1 * u.m / u.s
        >>> n_a = 1e26 * u.m**-3
        >>> T_a = 1 * u.eV
        >>> n_b = 1e26 * u.m**-3
        >>> T_b = 1e3 * u.eV
        >>> Coulomb_log = 10 * u.dimensionless_unscaled
        >>> electron_ion_collisions = MaxwellianCollisionFrequencies(
        ...     "e-",
        ...     "Na+",
        ...     v_drift=v_drift,
        ...     n_a=n_a,
        ...     T_a=T_a,
        ...     n_b=n_b,
        ...     T_b=T_b,
        ...     Coulomb_log=Coulomb_log,
        ... )
        >>> electron_ion_collisions.Maxwellian_avg_ei_collision_freq
        <Quantity 2.8053078...e+15 Hz>
        """

        if not self.test_particle.is_electron or not self.field_particle.is_ion:
            raise ValueError(
                "Please specify an electron-ion interaction to use the "
                "Maxwellian_avg_ei_collision_freq attribute."
            )

        if not self._is_slowly_flowing:
            raise PhysicsError(
                "This frequency is only defined for slowly flowing "
                "species.  (see MaxwellianCollisionFrequencies class "
                "documentation for further details)"
            )

        coeff = 4 / (3 * np.sqrt(np.pi))

        return coeff * self.Lorentz_collision_frequency

    @cached_property
    def Maxwellian_avg_ii_collision_freq(self):
        r"""
        Average momentum relaxation rate for a slowly flowing Maxwellian
        distribution of ions, relative to a population of stationary
        ions.

        This function assumes that both populations are Maxwellian, and
        :math:`T_i ≲ T_e`.

        :cite:t:`callen:unpublished` provides a derivation of this as an
        average collision frequency between ions and ions for a
        Maxwellian distribution. It is thus a special case of the
        collision frequency with an averaging factor.

        Raises
        ------
        `~plasmapy.utils.exceptions.PhysicsError`
            The test particles are not 'slowly flowing' relative to the
            field particles (see notes).

        `ValueError`
            If the specified interaction is not ion-ion.

        Examples
        --------
        >>> import astropy.units as u
        >>> v_drift = 1 * u.m / u.s
        >>> n_a = 1e26 * u.m**-3
        >>> T_a = 1e3 * u.eV
        >>> n_b = 1e26 * u.m**-3
        >>> T_b = 1e3 * u.eV
        >>> Coulomb_log = 10 * u.dimensionless_unscaled
        >>> ion_ion_collisions = MaxwellianCollisionFrequencies(
        ...     "Na+",
        ...     "Na+",
        ...     v_drift=v_drift,
        ...     n_a=n_a,
        ...     T_a=T_a,
        ...     n_b=n_b,
        ...     T_b=T_b,
        ...     Coulomb_log=Coulomb_log,
        ... )
        >>> ion_ion_collisions.Maxwellian_avg_ii_collision_freq
        <Quantity 1.1223822...e+08 Hz>
        """

        if not self.test_particle.is_ion or not self.field_particle.is_ion:
            raise ValueError(
                "Please specify an ion-ion interaction to use the "
                "Maxwellian_avg_ii_collision_freq attribute"
            )

        if not self._is_slowly_flowing:
            raise PhysicsError(
                "This frequency is only defined for slowly flowing "
                "species.  (see MaxwellianCollisionFrequencies class "
                "documentation for further details)"
            )

        coeff = 4 / (3 * np.sqrt(2 * np.pi))

        return coeff * self.Lorentz_collision_frequency


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n={"can_be_negative": False},
)
def collision_frequency(
    T: u.Quantity[u.K],
    n: u.Quantity[u.m**-3],
    species,
    z_mean: float = np.nan,
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    method: str = "classical",
) -> u.Quantity[u.Hz]:
    r"""
    Collision frequency of particles in a plasma.

    .. deprecated::
        The `~plasmapy.formulary.collisions.frequencies.collision_frequency`
        function has been replaced by the more general
        |SingleParticleCollisionFrequencies|
        class.  To replicate the functionality of
        `~plasmapy.formulary.collisions.frequencies.collision_frequency`, create a
        |SingleParticleCollisionFrequencies|
        class and access the ``Lorentz_collision_frequency`` attribute.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Temperature in units of temperature. This should be the electron
        temperature for electron-electron and electron-ion collisions,
        and the ion temperature for ion-ion collisions.

    n : `~astropy.units.Quantity`
        The density in units convertible to per cubic meter. This should
        be the electron density for electron-electron collisions, and
        the ion density for electron-ion and ion-ion collisions.

    species : `tuple`
        A tuple containing string representations of the test particle
        (listed first) and the target particle (listed second).

    z_mean : `~astropy.units.Quantity`, optional
        The average ionization (arithmetic mean) of a plasma for which a
        macroscopic description is valid. This parameter is used to
        compute the average ion density (given the average ionization
        and electron density) for calculating the ion sphere radius for
        non-classical impact parameters. ``z_mean`` is a required
        parameter if ``method`` is ``"ls_full_interp"``,
        ``"hls_max_interp"``, or ``"hls_full_interp"``.

    V : `~astropy.units.Quantity`, optional
        The relative velocity between particles. If not provided,
        thermal velocity is assumed: :math:`μ V^2 \sim 2 k_B T` where
        :math:`μ` is the reduced mass.

    method : `str`, optional
        The method by which to compute the Coulomb logarithm.  The
        default method is the classical straight-line Landau-Spitzer
        method (``"classical"`` or ``"ls"``). The other 6 supported
        methods are ``"ls_min_interp"``, ``"ls_full_interp"``,
        ``"ls_clamp_mininterp"``, ``"hls_min_interp"``,
        ``"hls_max_interp"``, and ``"hls_full_interp"``.  Please refer
        to the docstring of
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for more
        information about these methods.

    Returns
    -------
    freq : float or numpy.ndarray
        The collision frequency of particles in a plasma.

    Raises
    ------
    `ValueError`
        If the mass or charge of either particle cannot be found, or any
        of the inputs contain incorrect values.

    `~astropy.units.UnitConversionError`
        If the units on any of the inputs are incorrect.

    `TypeError`
        If any of ``n_e``, ``T``, or ``V`` is not a
        `~astropy.units.Quantity`.

    `~plasmapy.utils.exceptions.RelativityError`
        If the input velocity is same or greater than the speed of
        light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed

    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the input velocity is greater than 5% of the speed of light.

    Notes
    -----
    The collision frequency (see Ch. 5 of :cite:t:`chen:2016`) is given
    by

    .. math::

        ν = n σ v \ln{Λ}

    where :math:`n` is the particle density, :math:`σ` is the
    collisional cross-section, :math:`v` is the inter-particle velocity
    (typically taken as the thermal velocity), and :math:`\ln{Λ}` is the
    Coulomb logarithm accounting for small angle collisions.

    See Equation (2.14) in :cite:t:`callen:unpublished`.

    Examples
    --------
    >>> import astropy.units as u
    >>> n = 1e19 * u.m**-3
    >>> T = 1e6 * u.K
    >>> species = ("e", "p")
    >>> collision_frequency(T, n, species)
    <Quantity 70249... Hz>

    See Also
    --------
    ~plasmapy.formulary.collisions.frequencies.SingleParticleCollisionFrequencies
    """

    deprecated(
        since="0.9.0",
        warning_type=PlasmaPyFutureWarning,
        message=(
            "The collision_frequency function has been replaced by the "
            "more general SingleParticleCollisionFrequencies class. To "
            "replicate the functionality of collision_frequency, create"
            " a SingleParticleCollisionFrequencies class and access "
            "the `Lorentz_collision_frequency` attribute."
        ),
    )

    T, masses, charges, reduced_mass, V_r = misc._process_inputs(  # noqa: SLF001
        T=T, species=species, V=V
    )
    # using a more descriptive name for the thermal velocity using
    # reduced mass
    V_reduced = V_r

    if species[0] in {"e", "e-"} and species[1] in {"e", "e-"}:
        # electron-electron collision
        # if a velocity was passed, we use that instead of the reduced
        # thermal velocity
        V = misc._replace_nan_velocity_with_thermal_velocity(  # noqa: SLF001
            V,
            T,
            reduced_mass,
        )
        # impact parameter for 90° collision
        bPerp = lengths.impact_parameter_perp(T=T, species=species, V=V_reduced)
    elif species[0] in {"e", "e-"} or species[1] in {"e", "e-"}:
        # electron-ion collision
        # Need to manually pass electron thermal velocity to obtain
        # correct perpendicular collision radius
        # we ignore the reduced velocity and use the electron thermal
        # velocity instead
        V = misc._replace_nan_velocity_with_thermal_velocity(V, T, m_e)  # noqa: SLF001
        # need to also correct mass in collision radius from reduced
        # mass to electron mass
        bPerp = (
            lengths.impact_parameter_perp(T=T, species=species, V=V)
            * reduced_mass
            / m_e
        )
        # !!! may also need to correct Coulomb logarithm to be
        # electron-electron version !!!
    else:
        # ion-ion collision
        # if a velocity was passed, we use that instead of the reduced
        # thermal velocity
        V = misc._replace_nan_velocity_with_thermal_velocity(  # noqa: SLF001
            V,
            T,
            reduced_mass,
        )
        bPerp = lengths.impact_parameter_perp(T=T, species=species, V=V)
    cou_log = coulomb.Coulomb_logarithm(T, n, species, z_mean, V=V, method=method)
    # collisional cross section
    sigma = coulomb.Coulomb_cross_section(bPerp)
    # collision frequency where Coulomb logarithm accounts for
    # small angle collisions, which are more frequent than large
    # angle collisions.
    return n * sigma * V * cou_log


@validate_quantities(
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def fundamental_electron_collision_freq(
    T_e: u.Quantity[u.K],
    n_e: u.Quantity[u.m**-3],
    ion,
    coulomb_log=None,
    V=None,
    coulomb_log_method: str = "classical",
) -> u.Quantity[u.s**-1]:
    r"""
    Average momentum relaxation rate for a slowly flowing Maxwellian
    distribution of electrons.

    .. note::
        The `~plasmapy.formulary.collisions.frequencies.fundamental_electron_collision_freq`
        function has been replaced by the more general
        `~plasmapy.formulary.collisions.frequencies.MaxwellianCollisionFrequencies`
        class.  To replicate the functionality of
        `~plasmapy.formulary.collisions.frequencies.fundamental_electron_collision_freq`,
        create a
        `~plasmapy.formulary.collisions.frequencies.MaxwellianCollisionFrequencies`
        class and access the ``Maxwellian_avg_ei_collision_freq``
        attribute.

    :cite:t:`braginskii:1965` provides a derivation of this as an
    average collision frequency between electrons and ions for a
    Maxwellian distribution. It is thus a special case of the collision
    frequency with an averaging factor, and is on many occasions
    in transport theory the most relevant collision frequency that has
    to be considered. It commonly occurs in relation to diffusion and
    resistivity in plasmas.

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        The electron temperature of the Maxwellian test electrons.

    n_e : `~astropy.units.Quantity`
        The number density of the Maxwellian test electrons.

    ion : `str`
        String signifying a particle type of the field ions, including
        charge state information.

    V : `~astropy.units.Quantity`, optional
        The relative velocity between particles.  If not provided,
        thermal velocity is assumed: :math:`μ V^2 \sim 2 k_B T` where
        :math:`μ` is the reduced mass.

    coulomb_log : `float` or dimensionless `~astropy.units.Quantity`, optional
        Option to specify a Coulomb logarithm of the electrons on the
        ions.  If not specified, the Coulomb log will is calculated
        using the `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm`
        function.

    coulomb_log_method : `str`, optional
        The method by which to compute the Coulomb logarithm.  The
        default method is the classical straight-line Landau-Spitzer
        method (``"classical"`` or ``"ls"``). The other 6 supported
        methods are ``"ls_min_interp"``, ``"ls_full_interp"``,
        ``"ls_clamp_mininterp"``, ``"hls_min_interp"``,
        ``"hls_max_interp"``, and ``"hls_full_interp"``.  Please refer
        to the docstring of
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for
        more information about these methods.

    Returns
    -------
    nu_e : `~astropy.units.Quantity`

    Notes
    -----
    Equations (2.17) and (2.120) in :cite:t:`callen:unpublished` provide
    the original source used to implement this formula, however, the
    simplest form that connects our average collision frequency to the
    general collision frequency is this (from 2.17):

    .. math::

        ν_e = \frac{4}{3 \sqrt{π}} ν(v_{Te})

    Where :math:`ν` is the general collision frequency and
    :math:`v_{Te}` is the electron thermal velocity (the average, for a
    Maxwellian distribution).

    This implementation of the average collision frequency is
    equivalent to:

    * :math:`1/τ_e` from equation (2.5e) on page 215 of
      :cite:t:`braginskii:1965`
    * :math:`ν_e` from page 33 of :cite:t:`nrlformulary`

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.constants import c
    >>> fundamental_electron_collision_freq(0.1 * u.eV, 1e6 / u.m**3, "p")
    <Quantity 0.001801... 1 / s>
    >>> fundamental_electron_collision_freq(1e6 * u.K, 1e6 / u.m**3, "p")
    <Quantity 1.07221...e-07 1 / s>
    >>> fundamental_electron_collision_freq(100 * u.eV, 1e20 / u.m**3, "p")
    <Quantity 3935958.7... 1 / s>
    >>> fundamental_electron_collision_freq(
    ...     100 * u.eV, 1e20 / u.m**3, "p", coulomb_log_method="GMS-1"
    ... )
    <Quantity 3872815.5... 1 / s>
    >>> fundamental_electron_collision_freq(0.1 * u.eV, 1e6 / u.m**3, "p", V=c / 100)
    <Quantity 5.6589...e-07 1 / s>
    >>> fundamental_electron_collision_freq(
    ...     100 * u.eV, 1e20 / u.m**3, "p", coulomb_log=20
    ... )
    <Quantity 5812633... 1 / s>

    See Also
    --------
    ~plasmapy.formulary.collisions.frequencies.collision_frequency
    ~plasmapy.formulary.collisions.frequencies.fundamental_ion_collision_freq
    """

    deprecated(
        since="0.9.0",
        warning_type=PlasmaPyFutureWarning,
        message=(
            "The `fundamental_electron_collision_freq` function has been"
            "replaced by the more general `MaxwellianCollisionFrequencies` "
            "class.  To replicate the functionality of "
            "`fundamental_electron_collision_freq`, create a"
            "`MaxwellianCollisionFrequencies` class and access the "
            "`Maxwellian_avg_ei_collision_freq` attribute."
        ),
    )

    # specify to use electron thermal velocity (most probable), not based on reduced mass
    V = misc._replace_nan_velocity_with_thermal_velocity(V, T_e, m_e)  # noqa: SLF001

    species = [ion, "e-"]
    Z_i = particles.charge_number(ion) * u.dimensionless_unscaled
    n_i = n_e / Z_i
    nu = collision_frequency(
        T_e, n_i, species, z_mean=Z_i, V=V, method=coulomb_log_method
    )
    coeff = 4 / np.sqrt(np.pi) / 3

    # accounting for when a Coulomb logarithm value is passed
    if np.any(coulomb_log):
        cLog = coulomb.Coulomb_logarithm(
            T_e, n_e, species, z_mean=Z_i, V=V, method=coulomb_log_method
        )
        # dividing out by typical Coulomb logarithm value implicit in
        # the collision frequency calculation and replacing with
        # the user defined Coulomb logarithm value
        nu_mod = nu * coulomb_log / cLog
        return coeff * nu_mod

    return coeff * nu


@validate_quantities(
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_i={"can_be_negative": False},
)
def fundamental_ion_collision_freq(
    T_i: u.Quantity[u.K],
    n_i: u.Quantity[u.m**-3],
    ion,
    coulomb_log=None,
    V=None,
    coulomb_log_method: str = "classical",
) -> u.Quantity[u.s**-1]:
    r"""
    Average momentum relaxation rate for a slowly flowing Maxwellian
    distribution of ions.

    .. note::
        The `~plasmapy.formulary.collisions.frequencies.fundamental_ion_collision_freq`
        function has been replaced by the more general
        `~plasmapy.formulary.collisions.frequencies.MaxwellianCollisionFrequencies`
        class.  To replicate the functionality of
        `~plasmapy.formulary.collisions.frequencies.fundamental_ion_collision_freq`,
        create a
        `~plasmapy.formulary.collisions.frequencies.MaxwellianCollisionFrequencies`
        class and access the ``Maxwellian_avg_ii_collision_freq``
        attribute.


    :cite:t:`braginskii:1965` provides a derivation of this as an
    average collision frequency between ions and ions for a Maxwellian
    distribution. It is thus a special case of the collision frequency
    with an averaging factor.

    Parameters
    ----------
    T_i : `~astropy.units.Quantity`
        The electron temperature of the Maxwellian test ions.

    n_i : `~astropy.units.Quantity`
        The number density of the Maxwellian test ions.

    ion : `str`
        String signifying a particle type of the test and field ions,
        including charge state information. This function assumes the
        test and field ions are the same species.

    V : `~astropy.units.Quantity`, optional
        The relative velocity between particles.  If not provided,
        thermal velocity is assumed: :math:`μ V^2 \sim 2 k_B T` where
        :math:`μ` is the reduced mass.

    coulomb_log : `float` or dimensionless `~astropy.units.Quantity`, optional
        Option to specify a Coulomb logarithm of the electrons on the
        ions.  If not specified, the Coulomb log will is calculated
        using the
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm`
        function.

    coulomb_log_method : `str`, optional
        The method by which to compute the Coulomb logarithm.  The
        default method is the classical straight-line Landau-Spitzer
        method (``"classical"`` or ``"ls"``). The other 6 supported
        methods are ``"ls_min_interp"``, ``"ls_full_interp"``,
        ``"ls_clamp_mininterp"``, ``"hls_min_interp"``,
        ``"hls_max_interp"``, and ``"hls_full_interp"``.  Please refer
        to the docstring of
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for
        more information about these methods.

    Returns
    -------
    nu_i : `~astropy.units.Quantity`

    Notes
    -----
    Equations (2.36) and (2.122) in :cite:t:`callen:unpublished` provide
    the original source used to implement this formula, however, in our
    implementation we use the very same process that leads to the
    fundamental electron collision rate (2.17), gaining simply a
    different coefficient:

    .. math::

        ν_i = \frac{8}{3 × 4 × \sqrt{π}} ν(v_{Ti})

    Where :math:`ν` is the general collision frequency and
    :math:`v_{Ti}` is the ion thermal velocity (the average, for a
    Maxwellian distribution).

    Note that in the derivation, it is assumed that electrons are
    present in such numbers as to establish quasineutrality, but the
    effects of the test ions colliding with them are not considered
    here. This is a very typical approximation in transport theory.

    This result is an ion momentum relaxation rate, and is used in many
    classical transport expressions. It is equivalent to:

    * :math:`1/τ_i` from equation (2.5i) on page 215 of
      :cite:t:`braginskii:1965`
    * :math:`ν_i` from page 33 of :cite:t:`nrlformulary`

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.constants import c
    >>> fundamental_ion_collision_freq(0.1 * u.eV, 1e6 / u.m**3, "p")
    <Quantity 2.868...e-05 1 / s>
    >>> fundamental_ion_collision_freq(1e6 * u.K, 1e6 / u.m**3, "p")
    <Quantity 1.741...e-09 1 / s>
    >>> fundamental_ion_collision_freq(100 * u.eV, 1e20 / u.m**3, "p")
    <Quantity 63087.5... 1 / s>
    >>> fundamental_ion_collision_freq(
    ...     100 * u.eV, 1e20 / u.m**3, "p", coulomb_log_method="GMS-1"
    ... )
    <Quantity 63085.1... 1 / s>
    >>> fundamental_ion_collision_freq(100 * u.eV, 1e20 / u.m**3, "p", V=c / 100)
    <Quantity 9.111... 1 / s>
    >>> fundamental_ion_collision_freq(100 * u.eV, 1e20 / u.m**3, "p", coulomb_log=20)
    <Quantity 95918.7... 1 / s>

    See Also
    --------
    ~plasmapy.formulary.collisions.frequencies.collision_frequency
    ~plasmapy.formulary.collisions.frequencies.fundamental_electron_collision_freq
    """

    deprecated(
        since="0.9.0",
        warning_type=PlasmaPyFutureWarning,
        message=(
            "The `fundamental_ion_collision_freq` function has been"
            "replaced by the more general `MaxwellianCollisionFrequencies`"
            " class.  To replicate the functionality of "
            "`fundamental_ion_collision_freq`, create a"
            "`MaxwellianCollisionFrequencies` class and access the "
            "`Maxwellian_avg_ii_collision_freq` attribute."
        ),
    )

    m_i = particles.particle_mass(ion)
    species = [ion, ion]

    # specify to use ion thermal velocity (most probable), not based on reduced mass
    V = misc._replace_nan_velocity_with_thermal_velocity(V, T_i, m_i)  # noqa: SLF001

    Z_i = particles.charge_number(ion) * u.dimensionless_unscaled

    nu = collision_frequency(
        T_i, n_i, species, z_mean=Z_i, V=V, method=coulomb_log_method
    )
    # factor of 4 due to reduced mass in bperp and the rest is
    # due to differences in definitions of collisional frequency
    coeff = np.sqrt(8 / np.pi) / 3 / 4

    # accounting for when a Coulomb logarithm value is passed
    if np.any(coulomb_log):
        cLog = coulomb.Coulomb_logarithm(
            T_i, n_i, species, z_mean=Z_i, V=V, method=coulomb_log_method
        )
        # dividing out by typical Coulomb logarithm value implicit in
        # the collision frequency calculation and replacing with
        # the user defined Coulomb logarithm value
        nu_mod = nu * coulomb_log / cLog
        return coeff * nu_mod

    return coeff * nu
