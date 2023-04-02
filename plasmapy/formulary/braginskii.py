"""
Functions to calculate classical transport coefficients.

.. nbgallery::

    /notebooks/formulary/braginskii

Introduction
============

Classical transport theory is derived by using kinetic theory to close
the plasma two-fluid (electron and ion fluid) equations in the
collisional limit. The first complete model in this form was done by
:cite:t:`braginskii:1965`.

As described in the next section, this module uses fitting functions
from the literature
:cite:p:`braginskii:1965,spitzer:1953,spitzer:1962,epperlein:1986,ji:2013`
to calculate the transport coefficients, which are the resistivity,
thermoelectric conductivity, thermal conductivity, and viscosity.

Keep in mind the following assumptions under which the transport equations
are derived:

1. The plasma is fully ionized, only consisting of ions and electrons.
   Neutral atoms are neglected.
2. Turbulent transport does not dominate.
3. The velocity distribution is close to Maxwellian. This implies:

    a) Collisional mean free path ≪ gradient scale length along field.
    b) Gyroradius ≪ gradient scale length perpendicular to field.

4. The plasma is highly collisional: collisional frequency ≫ gyrofrequency.

When classical transport is not valid, e.g. due to the presence of strong
gradients or turbulent transport, the transport is significantly increased
by these other effects. Thus classical transport often serves as a lower
bound on the losses / transport encountered in a plasma.

Transport Variables
===================
For documentation on the individual transport variables, please take
the following links to documentation of methods of |ClassicalTransport|.

* `~plasmapy.formulary.braginskii.ClassicalTransport.resistivity`
* `~plasmapy.formulary.braginskii.ClassicalTransport.thermoelectric_conductivity`
* `~plasmapy.formulary.braginskii.ClassicalTransport.ion_thermal_conductivity`
* `~plasmapy.formulary.braginskii.ClassicalTransport.electron_thermal_conductivity`
* `~plasmapy.formulary.braginskii.ClassicalTransport.ion_viscosity`
* `~plasmapy.formulary.braginskii.ClassicalTransport.electron_viscosity`

Using the module
================
Given that many of the transport variables share a lot of the same computation
and many are often needed to be calculated simultaneously, this module provides
a |ClassicalTransport| class that can be initialized once with all of the
variables necessary for calculation. It then provides all of the functionality
as methods (please refer to its documentation).

If you only wish to calculate a single transport variable (or if just don't
like object-oriented interfaces), we have also provided wrapper functions in
the main module namespace that use |ClassicalTransport| under the hood (see below,
in the Functions section).

.. warning::

    The API for this package is not yet stable.

Classical transport models
==========================
In this section, we present a broad overview of classical transport models
implemented within this module.

Braginskii :cite:p:`braginskii:1965`
------------------------------------

The original Braginskii treatment as presented in the highly cited review
paper from 1965. Coefficients are found from expansion of the kinetic
equation in Laguerre polynomials, truncated at the second term in their
series expansion (\ :math:`k = 2`\ ). This theory allows for arbitrary Hall parameter
and include results for Z = 1, 2, 3, 4, and infinity (the case of Lorentz
gas completely stripped of electrons, and the stationary ion approximation).

Spitzer-Harm :cite:p:`spitzer:1953,spitzer:1962`
------------------------------------------------

These coefficients were obtained from a numerical solution of the
Fokker-Planck equation. They give one of the earliest and most accurate
(in the Fokker-Planck sense) results for electron transport in simple
plasma. They principally apply in the unmagnetized / parallel field
case, although for resistivity Spitzer also calculated a famous result
for a strong perpendicular magnetic field. Results are for Z = 1, 2, 4,
16, and infinity (Lorentz gas / stationary ion approximation).

Epperlein-Haines :cite:p:`epperlein:1986`
-----------------------------------------

Not yet implemented.

Ji-Held :cite:p:`ji:2013`
-------------------------

This is a modern treatment of the classical transport problem that has been
carried out with laudable care. It allows for arbitrary hall parameter and
arbitrary :math:`Z` for all coefficients. Similar to the Epperlein-Haines model,
it corrects some known inaccuracies in the original Braginskii results,
notably the asymptotic behavior of alpha-cross and beta_perp as Hall →
+infinity. It also studies effects of electron collisions in the ion
terms, which all other treatments have not. To neglect electron-electron
collisions, leave :math:`μ = 0`\ . To consider them, specify mu and theta.
"""
__all__ = [
    "ClassicalTransport",
    "resistivity",
    "thermoelectric_conductivity",
    "ion_thermal_conductivity",
    "electron_thermal_conductivity",
    "ion_viscosity",
    "electron_viscosity",
]

import numpy as np
import warnings

from astropy import units as u
from astropy.constants.si import e, k_B, m_e

from plasmapy import particles, utils
from plasmapy.formulary.collisions import (
    Coulomb_logarithm,
    fundamental_electron_collision_freq,
    fundamental_ion_collision_freq,
)
from plasmapy.formulary.dimensionless import Hall_parameter
from plasmapy.formulary.misc import _grab_charge
from plasmapy.particles.atomic import _is_electron
from plasmapy.particles.exceptions import InvalidParticleError
from plasmapy.utils import PhysicsError
from plasmapy.utils.decorators import validate_quantities


class ClassicalTransport:
    """
    Classical transport coefficients (e.g. Braginskii, 1965).

    Notes
    -----
    Given that many of the transport variables share a lot of the same
    computation and many are often needed to be calculated
    simultaneously, this class can be initialized once with all of the
    variables necessary for calculation. It then provides all of the
    functionality as methods (please refer to their documentation).

    Parameters
    ----------
    T_e : `~astropy.units.Quantity`
        Electron temperature in units of temperature or energy per
        particle.

    n_e : `~astropy.units.Quantity`
        The electron number density in units convertible to per cubic
        meter.

    T_i : `~astropy.units.Quantity`
        Ion temperature in units of temperature or energy per particle.

    n_i : `~astropy.units.Quantity`
        The ion number density in units convertible to per cubic meter.

    ion : `str`
        Representation of the ion species (e.g., ``'p'`` for protons,
        ``'e'`` for electrons, ``'D+'`` for deuterium, or ``'He-4 +1'``
        for singly ionized helium-4). If no charge state information is
        provided, then the particles are assumed to be singly charged.

    Z : `int` or `numpy.inf`, optional
        The ion charge state. Overrides particle charge state if
        included.  Different theories support different values of
        ``Z``. For the original Braginskii model, ``Z`` can be any of
        [1, 2, 3, 4, infinity]. The Ji-Held model supports arbitrary
        ``Z``. Average ionization states ``Z_mean`` can be input using
        this input and the Ji-Held model, although doing so may neglect
        effects caused by multiple ion populations.

    B : `~astropy.units.Quantity`, optional
        The magnetic field strength in units convertible to tesla.
        Defaults to zero.

    model: `str`
        Indication of whose formulation from literature to use. Allowed
        values are:

        * ``"Braginskii"`` :cite:p:`braginskii:1965`
        * ``"Spitzer-Harm"`` :cite:p:`spitzer:1953,spitzer:1962`
        * ``"Epperlein-Haines"`` (not yet implemented) :cite:p:`epperlein:1986`
        * ``"Ji-Held"`` :cite:p:`ji:2013`

    field_orientation : `str`, defaults to ``'parallel'``
        Either of ``'parallel'``, ``'par'``, ``'perpendicular'``,
        ``'perp'``, ``'cross'``, or ``'all'``, indicating the cardinal
        orientation of the magnetic field with respect to the transport
        direction of interest. Note that ``'perp'`` refers to transport
        perpendicular to the field direction (in the direction of the
        temperature gradient), while ``'cross'`` refers to the direction
        perpendicular to B and the gradient of temperature
        (:math:`B × ∇T`\ ). The option ``'all'`` will return a
        `numpy.array` of all three, ``np.array((par, perp, cross))``.
        Does not apply to viscosities.

    coulomb_log_ei : `float` or dimensionless `~astropy.units.Quantity`, optional
        Force a particular value to be used for the electron-ion Coulomb
        logarithm (test electrons on field ions). If `None`,
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` will
        be used.  Useful for comparing calculations.

    V_ei : `~astropy.units.Quantity`, optional
       The relative velocity between particles.  Supplied to
       `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm`
       function, not otherwise used.  If not provided, thermal velocity
       is assumed: :math:`μ V^2 \sim 2 k_B T` where :math:`μ` is the
       reduced mass.

    coulomb_log_ii : `float` or dimensionless `~astropy.units.Quantity`, optional
        Force a particular value to be used for the ion-ion Coulomb
        logarithm (test ions on field ions). If `None`, the PlasmaPy
        function
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` will
        be used.  Useful for comparing calculations.

    V_ii : `~astropy.units.Quantity`, optional
       The relative velocity between particles.  Supplied to
       `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm`
       function, not otherwise used. If not provided, thermal velocity
       is assumed: :math:`μ V^2 \sim 2 k_B T` where :math`μ` is the
       reduced mass.

    hall_e : `float` or dimensionless `~astropy.units.Quantity`, optional
        Force a particular value to be used for the electron Hall
        parameter. If `None`,
        `~plasmapy.formulary.dimensionless.Hall_parameter` will be
        used. Useful for comparing calculations.

    hall_i : `float` or dimensionless `~astropy.units.Quantity`, optional
        Force a particular value to be used for the ion Hall parameter.
        If `None`, `~plasmapy.formulary.dimensionless.Hall_parameter`
        will be used. Useful for comparing calculations.

    mu : `float` or dimensionless `~astropy.units.Quantity`, optional
        Ji-Held model only, may be used to include ion-electron effects
        on the ion transport coefficients. Defaults to zero, thus
        disabling these effects.

    theta : `float` or dimensionless `~astropy.units.Quantity`, optional
        Ji-Held model only, may be used to include ion-electron effects
        on the ion transport coefficients. Defaults to
        :math:`T_e / T_i`\ .  Only has effect if ``mu`` is non-zero.

    coulomb_log_method : `str`, optional
        The method by which to compute the Coulomb logarithm.
        The default method is the classical straight-line
        Landau-Spitzer method (``"classical"`` or ``"ls"``). The other
        6 supported methods are ``"ls_min_interp"``,
        ``"ls_full_interp"``, ``"ls_clamp_mininterp"``,
        ``"hls_min_interp"``, ``"hls_max_interp"``, and
        ``"hls_full_interp"``.  Please refer to the docstring of
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for
        more information about these methods.

    Raises
    ------
    `ValueError`
        On incorrect or unknown values of arguments.

    `~plasmapy.utils.exceptions.PhysicsError`
        If input or calculated values for Coulomb logarithms are nonphysical.

    Examples
    --------
    >>> from astropy import units as u
    >>> t = ClassicalTransport(1*u.eV, 1e20/u.m**3,
    ...                         1*u.eV, 1e20/u.m**3, 'p')
    >>> t.resistivity
    <Quantity 0.0003670... m Ohm>
    >>> t.thermoelectric_conductivity
    <Quantity 0.71108...>
    >>> t.ion_thermal_conductivity
    <Quantity 0.01552... W / (K m)>
    >>> t.electron_thermal_conductivity
    <Quantity 0.38064... W / (K m)>
    >>> t.ion_viscosity
    <Quantity [4.621297...e-07, 4.607248...e-07, 4.607248...e-07, 0.000000...e+00,
               0.000000...e+00] Pa s>
    >>> t.electron_viscosity
    <Quantity [5.822738...e-09, 5.820820...e-09, 5.820820...e-09, 0.000000...e+00,
               0.000000...e+00] Pa s>
    """

    @validate_quantities(
        T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
        T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
        m_i={"can_be_negative": False},
    )
    def __init__(
        self,
        T_e: u.K,
        n_e: u.m**-3,
        T_i: u.K,
        n_i: u.m**-3,
        ion,
        m_i: u.kg = None,
        Z=None,
        B: u.T = 0.0 * u.T,
        model="Braginskii",
        field_orientation="parallel",
        coulomb_log_ei=None,
        V_ei=None,
        coulomb_log_ii=None,
        V_ii=None,
        hall_e=None,
        hall_i=None,
        mu=None,
        theta=None,
        coulomb_log_method="classical",
    ):
        # check the model
        self.model = model.lower()  # string inputs should be case-insensitive
        valid_models = ["braginskii", "spitzer", "spitzer-harm", "ji-held"]
        if self.model not in valid_models:
            raise ValueError(f"Unknown transport model '{self.model}'")

        # check the field orientation
        self.field_orientation = field_orientation.lower()
        valid_fields = ["parallel", "par", "perpendicular", "perp", "cross", "all"]
        is_valid_field = self.field_orientation in valid_fields
        if not is_valid_field:
            raise ValueError(f"Unknown field orientation '{self.field_orientation}'")

        # values and units have already been checked by decorator
        self.T_e = T_e
        self.T_i = T_i
        self.n_e = n_e
        self.n_i = n_i

        # get ion mass and charge state
        if m_i is None:
            try:
                self.m_i = particles.particle_mass(ion)
            except InvalidParticleError as ex:
                raise ValueError(
                    f"Unable to find mass of particle: {ion} in ClassicalTransport"
                ) from ex
        else:
            self.m_i = m_i
        self.Z = _grab_charge(ion, Z) * u.dimensionless_unscaled
        if self.Z < 0:
            raise ValueError("Z is not allowed to be negative!")  # TODO remove?

        # decide on the particle string for the electrons
        self.e_particle = "e"
        self.ion = ion

        # save other arguments
        self.B = B
        self.V_ei = V_ei
        self.V_ii = V_ii

        # calculate Coulomb logs if not forced in input
        if coulomb_log_ei is not None:
            self.coulomb_log_ei = coulomb_log_ei
        else:
            self.coulomb_log_ei = Coulomb_logarithm(
                T_e, n_e, (self.e_particle, self.ion), V_ei, method=coulomb_log_method
            )

        if self.coulomb_log_ei < 1:
            # TODO discuss whether this is not too strict
            raise PhysicsError(
                f"Coulomb logarithm is {coulomb_log_ei} (below 1),"
                "this is probably not physical!"
            )
        elif self.coulomb_log_ei < 4:
            warnings.warn(
                f"Coulomb logarithm is {coulomb_log_ei},"
                f" you might have strong coupling effects",
                utils.CouplingWarning,
            )

        if coulomb_log_ii is not None:
            self.coulomb_log_ii = coulomb_log_ii
        else:
            self.coulomb_log_ii = Coulomb_logarithm(
                T_i,
                n_e,  # this is not a typo!
                (self.ion, self.ion),
                V_ii,
                method=coulomb_log_method,
            )

        if self.coulomb_log_ii < 1:
            # TODO discuss whether this is not too strict
            raise PhysicsError(
                f"Coulomb logarithm is {coulomb_log_ii} (below 1),"
                "this is probably not physical!"
            )
        elif self.coulomb_log_ii < 4:
            warnings.warn(
                f"Coulomb logarithm is {coulomb_log_ii},"
                f" you might have strong coupling effects",
                utils.CouplingWarning,
            )

        # calculate Hall parameters if not forced in input
        if hall_e is not None:
            self.hall_e = hall_e
        else:
            self.hall_e = Hall_parameter(
                n_e,
                T_e,
                B,
                self.ion,
                self.e_particle,
                coulomb_log_ei,
                V_ei,
                coulomb_log_method=coulomb_log_method,
            )
        if hall_i is not None:
            self.hall_i = hall_i
        else:
            self.hall_i = Hall_parameter(
                n_i,
                T_i,
                B,
                self.ion,
                self.ion,
                coulomb_log_ii,
                V_ii,
                coulomb_log_method=coulomb_log_method,
            )
        # set up the ion non-dimensional coefficients for the Ji-Held model
        self.mu = 0 if mu is None else mu  # disable the JH special features by default
        # self.mu = m_e / self.m_i  # enable the JH special features
        self.theta = self.T_e / self.T_i if theta is None else theta

    @property
    @validate_quantities
    def resistivity(self) -> u.Ohm * u.m:
        """
        Calculate the resistivity.

        The resistivity (:math:`α`) of a plasma is defined by

        .. math::
            α = \frac{\hat{α}}{n_e e^2 \frac{τ_e}{m_e}}

        where :math:`\hat{α}` is the non-dimensional resistivity of the plasma,
        :math:`n_e` is the electron number density of the plasma,
        :math:`e` is Euler's number,
        :math:`τ_e` is the fundamental electron collision period of the plasma,
        and :math:`m_e` is the mass of an electron.

        Notes
        -----
        The resistivity here is defined similarly to solid conductors, and thus
        represents the classical plasmas' property to resist the flow of
        electrical current. The result is in units of ohm meters, so if you
        assume where the current is flowing in the plasma (length and
        cross-sectional area), you could calculate a DC resistance of the
        plasma in ohms as resistivity × length / cross-sectional area.

        Experimentalists with plasma discharges may observe different
        :math:`V = IR` Ohm's law behavior than suggested by the
        resistance calculated here, for reasons such as the occurrence
        of plasma sheath layers at the electrodes or the plasma not
        satisfying the classical assumptions.

        Returns
        -------
        `~astropy.units.quantity.Quantity`

        """
        alpha_hat = _nondim_resistivity(
            self.hall_e, self.Z, self.e_particle, self.model, self.field_orientation
        )
        tau_e = 1 / fundamental_electron_collision_freq(
            self.T_e, self.n_e, self.ion, self.coulomb_log_ei, self.V_ei
        )

        alpha = alpha_hat / (self.n_e * e**2 * tau_e / m_e)
        return alpha

    @property
    def thermoelectric_conductivity(self):
        """
        Calculate the thermoelectric conductivity.

        .. todo::
            The thermoelectric conductivity (:math:`\hat{β}`) of a plasma
            is defined by...

        Notes
        -----
        To be improved.

        Returns
        -------
        `~astropy.units.quantity.Quantity`

        """
        beta_hat = _nondim_te_conductivity(
            self.hall_e, self.Z, self.e_particle, self.model, self.field_orientation
        )
        return u.Quantity(beta_hat)

    @property
    @validate_quantities
    def ion_thermal_conductivity(self) -> u.W / u.m / u.K:
        """
        Calculate the thermal conductivity for ions.

        The ion thermal conductivity (:math:`κ`) of a plasma is defined by

        .. math::
            κ = \hat{κ} \frac{n_i k_B^2 T_i τ_i}{m_i}

        where :math:`\hat{κ}` is the non-dimensional ion thermal
        conductivity of the plasma,
        :math:`n_i` is the ion number density of the plasma,
        :math:`k_B` is the Boltzmann constant,
        :math:`T_i` is the ion temperature of the plasma,
        :math:`τ_i` is the fundamental ion collision period of the plasma,
        and :math:`m_i` is the mass of an ion of the plasma.

        Notes
        -----
        This is the classical plasma ions' ability to conduct energy and heat,
        defined similarly to other materials. The result is a conductivity in
        units of W / m / K, so if you assume you know where the heat is flowing
        (temperature gradient, cross-sectional area) you can calculate the
        energy transport in watts as conductivity × cross-sectional area ×
        temperature gradient. In lab plasmas, typically the energy is flowing
        out of your high-temperature plasma to something else, like the walls
        of your device, and you are sad about this.

        Returns
        -------
        `~astropy.units.quantity.Quantity`

        See Also
        --------
        ion_thermal_conductivity

        """
        kappa_hat = _nondim_thermal_conductivity(
            self.hall_i,
            self.Z,
            self.ion,
            self.model,
            self.field_orientation,
            self.mu,
            self.theta,
        )
        tau_i = 1 / fundamental_ion_collision_freq(
            self.T_i, self.n_i, self.ion, self.coulomb_log_ii, self.V_ii
        )
        kappa = kappa_hat * (self.n_i * k_B**2 * self.T_i * tau_i / self.m_i)
        return kappa

    @property
    @validate_quantities
    def electron_thermal_conductivity(self) -> u.W / u.m / u.K:
        """
        Calculate the thermal conductivity for electrons.

        The electron thermal conductivity (:math:`κ`) of a plasma is defined by

        .. math::
            κ = \hat{κ} \frac{n_e k_B^2 T_e τ_e}{m_e}

        where :math:`\hat{κ}` is the non-dimensional electron thermal
        conductivity of the plasma,
        :math:`n_e` is the electron number density of the plasma,
        :math:`k_B` is the Boltzmann constant,
        :math:`T_e` is the electron temperature of the plasma,
        :math:`τ_e` is the fundamental electron collision period of the plasma,
        and :math:`m_e` is the mass of an electron.

        Notes
        -----
        This is quite similar to the ion thermal conductivity, except that it's
        for the plasma electrons. In a typical unmagnetized plasma, the
        electron thermal conductivity is much higher than the ions and will
        dominate, due to the electrons' low mass and fast speeds.

        In a strongly magnetized plasma, following the classical transport
        analysis, you calculate that the perpendicular-field thermal
        conductivity becomes greatly reduced for the ions and electrons, with
        the electrons actually being restrained even more than the ions due to
        their low mass and small gyroradius. In reality, the electrons and ions
        are pulling on each other strongly due to their opposing charges, so
        you have the situation of ambipolar diffusion.

        This situation has been likened to an energetic little child (the
        electrons) not wanting to be pulled away from the playground (the
        magnetic field) by the parents (the ions).

        The ultimate rate must typically be in between the individual rates for
        electrons and ions, so at least you can get some bounds from this type
        of analysis.

        Returns
        -------
        `~astropy.units.quantity.Quantity`

        See Also
        --------
        ion_thermal_conductivity

        """
        kappa_hat = _nondim_thermal_conductivity(
            self.hall_e,
            self.Z,
            self.e_particle,
            self.model,
            self.field_orientation,
            self.mu,
            self.theta,
        )
        tau_e = 1 / fundamental_electron_collision_freq(
            self.T_e, self.n_e, self.ion, self.coulomb_log_ei, self.V_ei
        )
        kappa = kappa_hat * (self.n_e * k_B**2 * self.T_e * tau_e / m_e)
        return kappa

    @property
    @validate_quantities
    def ion_viscosity(self) -> u.Pa * u.s:
        """
        Calculate the ion viscosity.

        .. todo::
            The ion viscosity (:math:`\eta`) of a plasma is defined by...

        Notes
        -----
        This is the dynamic viscosity that you find for ions in the classical
        plasma, similar to the viscosity of air or water or honey. The big
        effect is the :math:`T^{5/2}` dependence, so as classical plasmas
        get hotter they become dramatically more viscous. The ion
        viscosity typically dominates over the electron viscosity.

        Returns
        -------
        `~astropy.units.quantity.Quantity`

        See Also
        --------
        electron_viscosity

        """
        eta_hat = _nondim_viscosity(
            self.hall_i,
            self.Z,
            self.ion,
            self.model,
            self.field_orientation,
            self.mu,
            self.theta,
        )
        tau_i = 1 / fundamental_ion_collision_freq(
            self.T_i, self.n_i, self.ion, self.coulomb_log_ii, self.V_ii
        )
        common_factor = self.n_i * k_B * self.T_i * tau_i
        eta1 = np.array(eta_hat) * common_factor
        if not np.isclose(self.hall_i, 0, rtol=1e-8):
            eta1[1:3] /= self.hall_i**2
            eta1[3:] /= self.hall_i
        if eta1[0].unit == eta1[2].unit == eta1[4].unit:
            unit_val = eta1[0].unit
            eta = eta1.value * unit_val
        return eta

    @property
    @validate_quantities
    def electron_viscosity(self) -> u.Pa * u.s:
        """
        Calculate the electron viscosity.

        .. todo::
            The electron viscosity (:math:`\eta`) of a plasma is defined by...

        Notes
        -----
        This is the dynamic viscosity that you find for electrons in the
        classical plasma, similar to the viscosity of air or water or honey.
        The big effect is the :math:`T^{5/2}` dependence, so as classical
        plasmas get hotter they become dramatically more viscous. The
        ion viscosity typically dominates over the electron viscosity.

        Returns
        -------
        `~astropy.units.quantity.Quantity`

        See Also
        --------
        ~plasmapy.formulary.braginskii.ClassicalTransport.ion_viscosity
        """
        eta_hat = _nondim_viscosity(
            self.hall_e,
            self.Z,
            self.e_particle,
            self.model,
            self.field_orientation,
            self.mu,
            self.theta,
        )
        tau_e = 1 / fundamental_electron_collision_freq(
            self.T_e, self.n_e, self.ion, self.coulomb_log_ei, self.V_ei
        )
        common_factor = self.n_e * k_B * self.T_e * tau_e
        if np.isclose(self.hall_e, 0, rtol=1e-8):
            eta1 = (
                eta_hat[0] * common_factor,
                eta_hat[1] * common_factor,
                eta_hat[2] * common_factor,
                eta_hat[3] * common_factor,
                eta_hat[4] * common_factor,
            )
        else:
            eta1 = (
                eta_hat[0] * common_factor,
                eta_hat[1] * common_factor / self.hall_e**2,
                eta_hat[2] * common_factor / self.hall_e**2,
                eta_hat[3] * common_factor / self.hall_e,
                eta_hat[4] * common_factor / self.hall_e,
            )
        if eta1[0].unit == eta1[2].unit == eta1[4].unit:
            unit_val = eta1[0].unit
            eta = (
                np.array(
                    (
                        eta1[0].value,
                        eta1[1].value,
                        eta1[2].value,
                        eta1[3].value,
                        eta1[4].value,
                    )
                )
                * unit_val
            )
        return eta

    @property
    def all_variables(self) -> dict:
        """
        Return all transport variables as a dictionary.

        Returns
        -------
        dict

        """
        d = {
            "resistivity": self.resistivity,
            "thermoelectric conductivity": self.thermoelectric_conductivity,
            "electron thermal conductivity": self.electron_thermal_conductivity,
            "electron viscosity": self.electron_viscosity,
        }

        if self.model != "spitzer":
            d["ion thermal conductivity"] = self.ion_thermal_conductivity
            d["ion viscosity"] = self.ion_viscosity
        return d


@validate_quantities
def resistivity(
    T_e,
    n_e,
    T_i,
    n_i,
    ion,
    m_i=None,
    Z=None,
    B: u.T = 0.0 * u.T,
    model="Braginskii",
    field_orientation="parallel",
    mu=None,
    theta=None,
    coulomb_log_method="classical",
) -> u.Ohm * u.m:
    """
    Calculate the resistivity.

    The resistivity (:math:`α`) of a plasma is defined by

    .. math::

        α = \frac{\hat{α}}{n_e e^2 \frac{τ_e}{m_e}}

    where :math:`\hat{α}` is the non-dimensional resistivity of the plasma,
    :math:`n_e` is the electron number density of the plasma,
    :math:`e` is Euler's number,
    :math:`τ_e` is the fundamental electron collision period of the plasma,
    and :math:`m_e` is the mass of an electron.

    Notes
    -----
    The resistivity here is defined similarly to solid conductors, and thus
    represents the classical plasmas' property to resist the flow of
    electrical current. The result is in units of ohm meters, so if you
    assume where the current is flowing in the plasma (length and
    cross-sectional area), you could calculate a DC resistance of the
    plasma in ohms as resistivity × length / cross-sectional area.

    Experimentalists with plasma discharges may observe different
    :math:`V = IR` Ohm's law behavior than suggested by the resistance
    calculated here, for reasons such as the occurrence of plasma sheath
    layers at the electrodes or the plasma not satisfying the classical
    assumptions.

    Returns
    -------
    `~astropy.units.quantity.Quantity`

    """
    ct = ClassicalTransport(
        T_e,
        n_e,
        T_i,
        n_i,
        ion,
        m_i,
        Z=Z,
        B=B,
        model=model,
        field_orientation=field_orientation,
        mu=mu,
        theta=theta,
        coulomb_log_method=coulomb_log_method,
    )
    return ct.resistivity


@validate_quantities
def thermoelectric_conductivity(
    T_e,
    n_e,
    T_i,
    n_i,
    ion,
    m_i=None,
    Z=None,
    B: u.T = 0.0 * u.T,
    model="Braginskii",
    field_orientation="parallel",
    mu=None,
    theta=None,
    coulomb_log_method="classical",
):
    """
    Calculate the thermoelectric conductivity.

    .. todo::

        The thermoelectric conductivity (:math:`\hat{β}`) of a plasma is
        defined by...
    """
    ct = ClassicalTransport(
        T_e,
        n_e,
        T_i,
        n_i,
        ion,
        m_i,
        Z=Z,
        B=B,
        model=model,
        field_orientation=field_orientation,
        mu=mu,
        theta=theta,
        coulomb_log_method=coulomb_log_method,
    )
    return ct.thermoelectric_conductivity


@validate_quantities
def ion_thermal_conductivity(
    T_e,
    n_e,
    T_i,
    n_i,
    ion,
    m_i=None,
    Z=None,
    B: u.T = 0.0 * u.T,
    model="Braginskii",
    field_orientation="parallel",
    mu=None,
    theta=None,
    coulomb_log_method="classical",
) -> u.W / u.m / u.K:
    """
    Calculate the thermal conductivity for ions.

    The ion thermal conductivity (:math:`κ`) of a plasma is defined by

    .. math::

        κ = \hat{κ} \frac{n_i k_B^2 T_i τ_i}{m_i}

    where :math:`\hat{κ}` is the non-dimensional ion thermal conductivity
    of the plasma,
    :math:`n_i` is the ion number density of the plasma,
    :math:`k_B` is the Boltzmann constant,
    :math:`T_i` is the ion temperature of the plasma,
    :math:`τ_i` is the fundamental ion collision period of the plasma,
    and :math:`m_i` is the mass of an ion of the plasma.

    Notes
    -----
    This is the classical plasma ions' ability to conduct energy and heat,
    defined similarly to other materials. The result is a conductivity in units
    of W / m / K, so if you assume you know where the heat is flowing
    (temperature gradient, cross-sectional area) you can calculate the energy
    transport in watts as conductivity × cross-sectional area × temperature
    gradient. In laboratory plasmas, typically the energy is flowing out of your
    high-temperature plasma to something else, like the walls of your device,
    and you are sad about this.

    Returns
    -------
    `~astropy.units.quantity.Quantity`

    See Also
    --------
    ion_thermal_conductivity

    """
    ct = ClassicalTransport(
        T_e,
        n_e,
        T_i,
        n_i,
        ion,
        m_i,
        Z=Z,
        B=B,
        model=model,
        field_orientation=field_orientation,
        mu=mu,
        theta=theta,
        coulomb_log_method=coulomb_log_method,
    )
    return ct.ion_thermal_conductivity


@validate_quantities
def electron_thermal_conductivity(
    T_e,
    n_e,
    T_i,
    n_i,
    ion,
    m_i=None,
    Z=None,
    B: u.T = 0.0 * u.T,
    model="Braginskii",
    field_orientation="parallel",
    mu=None,
    theta=None,
    coulomb_log_method="classical",
) -> u.W / u.m / u.K:
    """
    Calculate the thermal conductivity for electrons.

    The electron thermal conductivity (:math:`κ`) of a plasma is defined by

    .. math::

        κ = \hat{κ} \frac{n_e k_B^2 T_e τ_e}{m_e}

    where :math:`\hat{κ}` is the non-dimensional electron thermal
    conductivity of the plasma,
    :math:`n_e` is the electron number density of the plasma,
    :math:`k_B` is the Boltzmann constant,
    :math:`T_e` is the electron temperature of the plasma,
    :math:`τ_e` is the fundamental electron collision period of the
    plasma, and :math:`m_e` is the mass of an electron.

    Notes
    -----
    This is quite similar to the ion thermal conductivity, except that it's for
    the plasma electrons. In a typical unmagnetized plasma, the electron
    thermal conductivity is much higher than the ions and will dominate, due to
    the electrons' low mass and fast speeds.

    In a strongly magnetized plasma, following the classical transport
    analysis, you calculate that the perpendicular-field thermal conductivity
    becomes greatly reduced for the ions and electrons, with the electrons
    actually being restrained even more than the ions due to their low mass and
    small gyroradius. In reality, the electrons and ions are pulling on each
    other strongly due to their opposing charges, so you have the situation of
    ambipolar diffusion.

    This situation has been likened to an energetic little child (the
    electrons) not wanting to be pulled away from the playground (the magnetic
    field) by the parents (the ions).

    The ultimate rate must typically be in between the individual rates for
    electrons and ions, so at least you can get some bounds from this type of
    analysis.

    Returns
    -------
    `~astropy.units.quantity.Quantity`

    See Also
    --------
    ion_thermal_conductivity
    """
    ct = ClassicalTransport(
        T_e,
        n_e,
        T_i,
        n_i,
        ion,
        m_i,
        Z=Z,
        B=B,
        model=model,
        field_orientation=field_orientation,
        mu=mu,
        theta=theta,
        coulomb_log_method=coulomb_log_method,
    )
    return ct.electron_thermal_conductivity


@validate_quantities
def ion_viscosity(
    T_e,
    n_e,
    T_i,
    n_i,
    ion,
    m_i=None,
    Z=None,
    B: u.T = 0.0 * u.T,
    model="Braginskii",
    field_orientation="parallel",
    mu=None,
    theta=None,
    coulomb_log_method="classical",
) -> u.Pa * u.s:
    """
    Calculate the ion viscosity.

    .. todo::
        The ion viscosity (:math:`\eta`) of a plasma is defined by...

    Notes
    -----
    This is the dynamic viscosity that you find for ions in the classical
    plasma, similar to the viscosity of air or water or honey. The big
    effect is the :math:`T^{5/2}` dependence, so as classical plasmas get hotter they
    become dramatically more viscous. The ion viscosity typically dominates
    over the electron viscosity.

    Returns
    -------
    `~astropy.units.quantity.Quantity`

    See Also
    --------
    electron_viscosity

    """
    ct = ClassicalTransport(
        T_e,
        n_e,
        T_i,
        n_i,
        ion,
        m_i,
        Z=Z,
        B=B,
        model=model,
        field_orientation=field_orientation,
        mu=mu,
        theta=theta,
        coulomb_log_method=coulomb_log_method,
    )
    return ct.ion_viscosity


@validate_quantities
def electron_viscosity(
    T_e,
    n_e,
    T_i,
    n_i,
    ion,
    m_i=None,
    Z=None,
    B: u.T = 0.0 * u.T,
    model="Braginskii",
    field_orientation="parallel",
    mu=None,
    theta=None,
    coulomb_log_method="classical",
) -> u.Pa * u.s:
    """
    Calculate the electron viscosity.

    .. todo::
        The electron viscosity (:math:`\eta`) of a plasma is defined by...

    Notes
    -----
    This is the dynamic viscosity that you find for electrons in the
    classical plasma, similar to the viscosity of air or water or honey.
    The big effect is the :math:`T^{5/2}` dependence, so as classical plasmas get
    hotter they become dramatically more viscous. The ion viscosity
    typically dominates over the electron viscosity.

    Returns
    -------
    `~astropy.units.quantity.Quantity`

    See Also
    --------
    ion_viscosity

    """
    ct = ClassicalTransport(
        T_e,
        n_e,
        T_i,
        n_i,
        ion,
        m_i,
        Z=Z,
        B=B,
        model=model,
        field_orientation=field_orientation,
        mu=mu,
        theta=theta,
        coulomb_log_method=coulomb_log_method,
    )
    return ct.electron_viscosity


def _nondim_thermal_conductivity(
    hall, Z, particle, model, field_orientation, mu=None, theta=None
):
    """
    Calculate dimensionless classical thermal conductivity coefficients.

    This function is a switchboard / wrapper that calls the appropriate
    model-specific functions depending on which model is specified and which
    type of particle (electron or ion) is input. Non-electrons are assumed to
    be ions.
    """
    if _is_electron(particle):
        if model in ("spitzer-harm", "spitzer"):
            kappa_hat = _nondim_tc_e_spitzer(Z)
        elif model == "braginskii":
            kappa_hat = _nondim_tc_e_braginskii(hall, Z, field_orientation)
        elif model == "ji-held":
            kappa_hat = _nondim_tc_e_ji_held(hall, Z, field_orientation)
        else:
            raise ValueError(
                f"Unrecognized model '{model}' in _nondim_thermal_conductivity"
            )
    elif model == "braginskii":
        kappa_hat = _nondim_tc_i_braginskii(hall, field_orientation)
    elif model == "ji-held":
        kappa_hat = _nondim_tc_i_ji_held(hall, Z, mu, theta, field_orientation)
    elif model in ("spitzer-harm", "spitzer"):
        raise NotImplementedError(
            "Ion thermal conductivity is not implemented in the Spitzer model."
        )
    else:
        raise ValueError(
            f"Unrecognized model '{model}' in _nondim_thermal_conductivity"
        )
    return kappa_hat


def _nondim_viscosity(hall, Z, particle, model, field_orientation, mu=None, theta=None):
    """
    Calculate dimensionless classical viscosity coefficients.

    This function is a switchboard / wrapper that calls the appropriate
    model-specific functions depending on which model is specified and which
    type of particle (electron or ion) is input. Non-electrons are assumed to
    be ions.
    """
    if _is_electron(particle):
        if model == "braginskii":
            eta_hat = _nondim_visc_e_braginskii(hall, Z)
        elif model == "ji-held":
            eta_hat = _nondim_visc_e_ji_held(hall, Z)
        else:
            raise ValueError(f"Unrecognized model '{model}' in _nondim_viscosity")
    elif model == "braginskii":
        eta_hat = _nondim_visc_i_braginskii(hall)
    elif model == "ji-held":
        eta_hat = _nondim_visc_i_ji_held(hall, Z, mu, theta)
    elif model in ("spitzer-harm", "spitzer"):
        raise NotImplementedError(
            "Ion viscosity is not implemented in the Spitzer model."
        )
    else:
        raise ValueError(f"Unrecognized model '{model}' in _nondim_viscosity")
    return eta_hat


def _nondim_resistivity(hall, Z, particle, model, field_orientation):
    """
    Calculate dimensionless classical resistivity coefficients.

    This function is a switchboard / wrapper that calls the appropriate
    model-specific functions depending on which model is specified.
    """
    if model in ("spitzer-harm", "spitzer"):
        alpha_hat = _nondim_resist_spitzer(Z, field_orientation)
    elif model == "braginskii":
        alpha_hat = _nondim_resist_braginskii(hall, Z, field_orientation)
    elif model == "ji-held":
        alpha_hat = _nondim_resist_ji_held(hall, Z, field_orientation)
    else:
        raise ValueError(f"Unrecognized model '{model}' in _nondim_resistivity")
    return alpha_hat


def _nondim_te_conductivity(hall, Z, particle, model, field_orientation):
    """
    Calculate dimensionless classical thermoelectric coefficients.

    This function is a switchboard / wrapper that calls the appropriate
    model-specific functions depending on which model is specified.
    """
    if model in ("spitzer-harm", "spitzer"):
        beta_hat = _nondim_tec_spitzer(Z)
    elif model == "braginskii":
        beta_hat = _nondim_tec_braginskii(hall, Z, field_orientation)
    elif model == "ji-held":
        beta_hat = _nondim_tec_ji_held(hall, Z, field_orientation)
    else:
        raise ValueError(f"Unrecognized model '{model}' in _nondim_te_conductivity")
    return beta_hat


def _check_Z(allowed_Z, Z):
    """Determine if the input Z value is okay given the list of allowed_Z."""
    # first, determine if arbitrary Z values are allowed in the theory
    arbitrary_Z_allowed = False
    the_arbitrary_idx = np.nan
    for idx, allowed_Z_val in enumerate(allowed_Z):
        if allowed_Z_val == "arbitrary":
            arbitrary_Z_allowed = True
            the_arbitrary_idx = idx
    # next, search the allowed_Z for a match to the current Z
    Z_idx = np.nan
    for idx, allowed_Z_val in enumerate(allowed_Z):
        if Z == allowed_Z_val:
            Z_idx = idx
    # at this point we have looped through allowed_Z and either found a match
    # or not. If we haven't found a match and arbitrary Z aren't allowed, break
    if np.isnan(Z_idx):
        if arbitrary_Z_allowed:
            # return a Z_idx pointing to the 'arbitrary'
            Z_idx = the_arbitrary_idx
        else:
            raise utils.PhysicsError(f"{Z} is not an allowed Z value")
    # we have got the Z_idx we want. return
    return Z_idx


def _get_spitzer_harm_coeffs(Z):
    """
    Return numerical coefficients from Spitzer-Harm '53.

    Table III, Spitzer and Harm, Phys. Rev. Vol 89, 5, 1953
    """
    allowed_Z = [1, 2, 4, 16, np.inf]
    Z_idx = _check_Z(allowed_Z, Z)
    gamma_E = [0.5816, 0.6833, 0.7849, 0.9225, 1.0000]
    gamma_T = [0.2727, 0.4137, 0.5714, 0.8279, 1.0000]
    delta_E = [0.4652, 0.5787, 0.7043, 0.8870, 1.0000]
    delta_T = [0.2252, 0.3563, 0.5133, 0.7907, 1.0000]
    return gamma_E[Z_idx], gamma_T[Z_idx], delta_E[Z_idx], delta_T[Z_idx]


def _nondim_tc_e_spitzer(Z):
    """
    Dimensionless electron thermal conductivity — Spitzer.

    This result is for parallel field or unmagnetized plasma only.
    """
    (gamma_E, gamma_T, delta_E, delta_T) = _get_spitzer_harm_coeffs(Z)
    kappa = (64 / np.pi) * delta_T * (5 / 3 - (gamma_T * delta_E) / (delta_T * gamma_E))
    return kappa


def _nondim_resist_spitzer(Z, field_orientation):
    """
    Dimensionless resistivity — Spitzer.

    These are results for both parallel-field / unmagnetized plasmas as well
    as perpendicular-field / strongly magnetized plasmas. Summary description
    in Physics of Fully Ionized Gases, Spitzer.
    """
    alpha_perp = 1
    if field_orientation in ("perpendicular", "perp"):
        return alpha_perp

    (gamma_E, gamma_T, delta_E, delta_T) = _get_spitzer_harm_coeffs(Z)
    alpha_par = (3 * np.pi / 32) * (1 / gamma_E)
    if field_orientation in ("parallel", "par"):
        return alpha_par
    #        alpha_par = 0.5064 # Z = 1

    if field_orientation == "all":
        return alpha_par, alpha_perp


def _nondim_tec_spitzer(Z):
    """
    Dimensionless thermoelectric conductivity — Spitzer.

    This result is for parallel field or unmagnetized plasma only.
    """
    (gamma_E, gamma_T, delta_E, delta_T) = _get_spitzer_harm_coeffs(Z)
    beta = 5 / 2 * (8 / 5 * (delta_E / gamma_E) - 1)
    return beta


def _nondim_tc_e_braginskii(hall, Z, field_orientation):
    """
    Dimensionless electron thermal conductivity — Braginskii.

    Braginskii, S. I. "Transport processes in a plasma." Reviews of plasma
    physics 1 (1965): 205.
    """
    allowed_Z = [1, 2, 3, 4, np.inf]
    Z_idx = _check_Z(allowed_Z, Z)

    # fixing overflow errors when exponentiating hall by making a float
    # instead of an int
    hall = float(hall)

    delta_0 = [3.7703, 1.0465, 0.5814, 0.4106, 0.0961]
    delta_1 = [14.79, 10.80, 9.618, 9.055, 7.482]
    gamma_1_prime = [4.664, 3.957, 3.721, 3.604, 3.25]
    gamma_0_prime = [11.92, 5.118, 3.525, 2.841, 1.20]
    gamma_1_doubleprime = [2.500, 2.500, 2.500, 2.500, 2.500]
    gamma_0_doubleprime = [21.67, 15.37, 13.53, 12.65, 10.23]

    gamma_0 = gamma_0_prime[Z_idx] / delta_0[Z_idx]
    Delta = hall**4 + delta_1[Z_idx] * hall**2 + delta_0[Z_idx]

    if field_orientation in ("parallel", "par"):
        kappa_par = gamma_0
        return kappa_par

    if field_orientation in ("perpendicular", "perp"):
        kappa_perp = (gamma_1_prime[Z_idx] * hall**2 + gamma_0_prime[Z_idx]) / Delta
        return kappa_perp

    if field_orientation == "cross":
        kappa_cross = (
            gamma_1_doubleprime[Z_idx] * hall**3 + gamma_0_doubleprime[Z_idx] * hall
        ) / Delta
        return kappa_cross

    if field_orientation == "all":
        kappa_par = gamma_0

        kappa_perp = (gamma_1_prime[Z_idx] * hall**2 + gamma_0_prime[Z_idx]) / Delta

        kappa_cross = (
            gamma_1_doubleprime[Z_idx] * hall**3 + gamma_0_doubleprime[Z_idx] * hall
        ) / Delta
        return np.array((kappa_par, kappa_perp, kappa_cross))


def _nondim_tc_i_braginskii(hall, field_orientation):
    """
    Dimensionless ion thermal conductivity — Braginskii.

    Braginskii, S. I. "Transport processes in a plasma." Reviews of plasma
    physics 1 (1965): 205.
    """
    # fixing overflow errors when exponentiating hall by making a float
    # instead of an int
    hall = float(hall)

    if field_orientation in ("parallel", "par"):
        kappa_par_coeff_0 = 3.906
        kappa_par = kappa_par_coeff_0
        return kappa_par

    delta_1 = 2.70
    delta_0 = 0.677
    Delta = hall**4 + delta_1 * hall**2 + delta_0

    if field_orientation in ("perpendicular", "perp"):
        kappa_perp_coeff_2 = 2.0
        kappa_perp_coeff_0 = 2.645
        kappa_perp = (kappa_perp_coeff_2 * hall**2 + kappa_perp_coeff_0) / Delta
        return kappa_perp

    if field_orientation == "cross":
        kappa_cross_coeff_3 = 2.5
        kappa_cross_coeff_1 = 4.65
        kappa_cross = (
            kappa_cross_coeff_3 * hall**3 + kappa_cross_coeff_1 * hall
        ) / Delta
        return kappa_cross

    if field_orientation == "all":
        kappa_par_coeff_0 = 3.906
        kappa_par = kappa_par_coeff_0

        kappa_perp_coeff_2 = 2.0
        kappa_perp_coeff_0 = 2.645
        kappa_perp = (kappa_perp_coeff_2 * hall**2 + kappa_perp_coeff_0) / Delta

        kappa_cross_coeff_3 = 2.5
        kappa_cross_coeff_1 = 4.65
        kappa_cross = (
            kappa_cross_coeff_3 * hall**3 + kappa_cross_coeff_1 * hall
        ) / Delta
        return np.array((kappa_par, kappa_perp, kappa_cross))


def _nondim_visc_e_braginskii(hall, Z):
    """
    Dimensionless electron viscosity — Braginskii.

    Braginskii, S. I. "Transport processes in a plasma." Reviews of plasma
    physics 1 (1965): 205.
    """
    # fixing overflow errors when exponentiating hall by making a float
    # instead of an int
    hall = float(hall)
    allowed_Z = [1]
    _check_Z(allowed_Z, Z)
    eta_prime_0 = 0.733
    eta_doubleprime_2 = 2.05
    eta_doubleprime_0 = 8.50
    eta_tripleprime_2 = 1.0
    eta_tripleprime_0 = 7.91
    delta_1 = 13.8
    delta_0 = 11.6
    eta_0_e = eta_prime_0

    def eta_2(hall):
        Delta = hall**4 + delta_1 * hall**2 + delta_0
        return (eta_doubleprime_2 * hall**2 + eta_doubleprime_0) / Delta

    eta_2_e = eta_2(hall)
    eta_1_e = eta_2(2 * hall)

    def f_eta_4(hall):
        Delta = hall**4 + delta_1 * hall**2 + delta_0
        return (eta_tripleprime_2 * hall**3 + eta_tripleprime_0 * hall) / Delta

    eta_4_e = f_eta_4(hall)
    eta_3_e = f_eta_4(2 * hall)
    return np.array((eta_0_e, eta_1_e, eta_2_e, eta_3_e, eta_4_e))


def _nondim_visc_i_braginskii(hall):
    """
    Dimensionless ion viscosity — Braginskii.

    Braginskii, S. I. "Transport processes in a plasma." Reviews of plasma
    physics 1 (1965): 205.
    """
    eta_prime_0 = 0.96
    eta_doubleprime_2 = 6 / 5
    eta_doubleprime_0 = 2.23
    eta_tripleprime_2 = 1.0
    eta_tripleprime_0 = 2.38
    delta_1 = 4.03
    delta_0 = 2.33
    eta_0_i = eta_prime_0

    # fixing overflow errors when exponentiating hall by making a float
    # instead of an int
    hall = float(hall)

    def f_eta_2(hall):
        Delta = hall**4 + delta_1 * hall**2 + delta_0
        return (eta_doubleprime_2 * hall**2 + eta_doubleprime_0) / Delta

    eta_2_i = f_eta_2(hall)
    eta_1_i = f_eta_2(2 * hall)

    def f_eta_4(hall):
        Delta = hall**4 + delta_1 * hall**2 + delta_0
        return (eta_tripleprime_2 * hall**3 + eta_tripleprime_0 * hall) / Delta

    eta_4_i = f_eta_4(hall)
    eta_3_i = f_eta_4(2 * hall)
    return np.array((eta_0_i, eta_1_i, eta_2_i, eta_3_i, eta_4_i))


def _nondim_resist_braginskii(hall, Z, field_orientation):
    """
    Dimensionless resistivity — Braginskii.

    Braginskii, S. I. "Transport processes in a plasma." Reviews of plasma
    physics 1 (1965): 205.
    """
    allowed_Z = [1, 2, 3, 4, np.inf]
    Z_idx = _check_Z(allowed_Z, Z)

    # fixing overflow errors when exponentiating hall by making a float
    # instead of an int
    hall = float(hall)

    #    alpha_0 = 0.5129
    delta_0 = [3.7703, 1.0465, 0.5814, 0.4106, 0.0961]
    delta_1 = [14.79, 10.80, 9.618, 9.055, 7.482]
    alpha_1_prime = [6.416, 5.523, 5.226, 5.077, 4.63]
    alpha_0_prime = [1.837, 0.5956, 0.3515, 0.2566, 0.0678]
    alpha_1_doubleprime = [1.704, 1.704, 1.704, 1.704, 1.704]
    alpha_0_doubleprime = [0.7796, 0.3439, 0.2400, 0.1957, 0.0940]

    alpha_0 = 1 - alpha_0_prime[Z_idx] / delta_0[Z_idx]
    Delta = hall**4 + delta_1[Z_idx] * hall**2 + delta_0[Z_idx]

    if field_orientation in ("parallel", "par"):
        alpha_par = alpha_0
        return alpha_par

    if field_orientation in ("perpendicular", "perp"):
        alpha_perp = (
            1 - (alpha_1_prime[Z_idx] * hall**2 + alpha_0_prime[Z_idx]) / Delta
        )
        return alpha_perp

    if field_orientation == "cross":
        alpha_cross = (
            alpha_1_doubleprime[Z_idx] * hall**3 + alpha_0_doubleprime[Z_idx] * hall
        ) / Delta
        return alpha_cross

    if field_orientation == "all":
        alpha_par = alpha_0

        alpha_perp = (
            1 - (alpha_1_prime[Z_idx] * hall**2 + alpha_0_prime[Z_idx]) / Delta
        )

        alpha_cross = (
            alpha_1_doubleprime[Z_idx] * hall**3 + alpha_0_doubleprime[Z_idx] * hall
        ) / Delta
        return np.array((alpha_par, alpha_perp, alpha_cross))


def _nondim_tec_braginskii(hall, Z, field_orientation):
    """
    Dimensionless thermoelectric conductivity — Braginskii.

    Braginskii, S. I. "Transport processes in a plasma." Reviews of plasma
    physics 1 (1965): 205.
    """
    allowed_Z = [1, 2, 3, 4, np.inf]
    Z_idx = _check_Z(allowed_Z, Z)
    # fixing overflow errors when exponentiating hall by making a float
    # instead of an int
    hall = float(hall)

    delta_0 = [3.7703, 1.0465, 0.5814, 0.4106, 0.0961]
    delta_1 = [14.79, 10.80, 9.618, 9.055, 7.482]
    beta_1_prime = [5.101, 4.450, 4.233, 4.124, 3.798]
    beta_0_prime = [2.681, 0.9473, 0.5905, 0.4478, 0.1461]
    beta_1_doubleprime = [1.5, 1.5, 1.5, 1.5, 1.5]
    beta_0_doubleprime = [3.053, 1.784, 1.442, 1.285, 0.877]

    Delta = hall**4 + delta_1[Z_idx] * hall**2 + delta_0[Z_idx]
    beta_0 = beta_0_prime[Z_idx] / delta_0[Z_idx]
    #    beta_0 = 0.7110

    if field_orientation in ("parallel", "par"):
        beta_par = beta_0
        return beta_par

    if field_orientation in ("perpendicular", "perp"):
        beta_perp = (beta_1_prime[Z_idx] * hall**2 + beta_0_prime[Z_idx]) / Delta
        return beta_perp

    if field_orientation == "cross":
        beta_cross = (
            beta_1_doubleprime[Z_idx] * hall**3 + beta_0_doubleprime[Z_idx] * hall
        ) / Delta
        return beta_cross

    if field_orientation == "all":
        beta_par = beta_0

        beta_perp = (beta_1_prime[Z_idx] * hall**2 + beta_0_prime[Z_idx]) / Delta

        beta_cross = (
            beta_1_doubleprime[Z_idx] * hall**3 + beta_0_doubleprime[Z_idx] * hall
        ) / Delta
        return np.array((beta_par, beta_perp, beta_cross))


#
#               Abandon all hope, ye who enter here
#


def _nondim_tc_e_ji_held(hall, Z, field_orientation):
    """
    Dimensionless electron thermal conductivity — Ji-Held.

    Ji, Jeong-Young, and Eric D. Held. "Closure and transport theory for
    high-collisionality electron-ion plasmas." Physics of Plasmas 20.4 (2013):
    042114.
    """
    allowed_Z = [1, 2, "arbitrary"]
    Z_idx = _check_Z(allowed_Z, Z)
    # fixing overflow errors when exponentiating r by making a float
    # instead of an int
    r = float(np.abs(Z * hall))

    def f_kappa_par_e(Z):
        numerator = 13.5 * Z**2 + 54.4 * Z + 25.2
        denominator = Z**3 + 8.35 * Z**2 + 15.2 * Z + 4.51
        return numerator / denominator

    def f_kappa_0(Z):
        numerator = 9.91 * Z**3 + 75.3 * Z**2 + 518 * Z + 333
        denominator = 1000
        return numerator / denominator

    def f_kappa_1(Z):
        numerator = 0.211 * Z**3 + 12.7 * Z**2 + 48.4 * Z + 6.45
        denominator = Z + 57.1
        return numerator / denominator

    def f_kappa_2(Z):
        numerator = 0.932 * Z ** (7 / 3) + 0.135 * Z**2 + 12.3 * Z + 8.77
        denominator = Z + 4.84
        return numerator / denominator

    def f_kappa_3(Z):
        numerator = 0.246 * Z**3 + 2.65 * Z**2 - 92.8 * Z - 1.96
        denominator = Z**2 + 19.9 * Z + 35.3
        return numerator / denominator

    def f_kappa_4(Z):
        numerator = 2.76 * Z ** (5 / 3) - 0.836 * Z ** (2 / 3) - 0.0611
        denominator = Z - 0.214
        return numerator / denominator

    def f_k_0(Z):
        numerator = 0.0396 * Z**3 + 46.3 * Z + 176
        denominator = 1000
        return numerator / denominator

    def f_k_1(Z):
        numerator = 15.4 * Z**3 + 188 * Z**2 + 240 * Z + 35.3
        denominator = 1000 * Z + 397
        return numerator / denominator

    def f_k_2(Z):
        numerator = -0.159 * Z**2 - 12.5 * Z + 34.1
        denominator = Z ** (2 / 3) + 0.741 * Z ** (1 / 3) + 31.0
        return numerator / denominator

    def f_k_3(Z):
        numerator = 0.431 * Z**2 + 3.69 * Z + 0.0314
        denominator = Z + 3.62
        return numerator / denominator

    def f_k_4(Z):
        numerator = 0.0258 * Z**2 - 1.63 * Z + 0.711
        denominator = Z ** (4 / 3) + 4.36 * Z ** (2 / 3) + 2.75
        return numerator / denominator

    def f_k_5(Z):
        numerator = Z**3 + 11.9 * Z**2 + 28.8 * Z + 9.07
        denominator = 173 * Z + 133
        return numerator / denominator

    kappa_par_e = [3.204, 2.464, f_kappa_par_e(Z)]
    kappa_0 = [0.936, 1.749, f_kappa_0(Z)]
    kappa_1 = [1.166, 2.635, f_kappa_1(Z)]
    kappa_2 = [3.791, 5.644, f_kappa_2(Z)]
    kappa_3 = [-1.635, -2.212, f_kappa_3(Z)]
    kappa_4 = [2.370, 4.129, f_kappa_4(Z)]
    k_0 = [0.222, 0.269, f_k_0(Z)]
    k_1 = [0.343, 0.580, f_k_1(Z)]
    k_2 = [0.655, 0.252, f_k_2(Z)]
    k_3 = [0.899, 1.626, f_k_3(Z)]
    k_4 = [-0.110, -0.201, f_k_4(Z)]
    k_5 = [0.166, 0.255, f_k_5(Z)]

    kappa_par = kappa_par_e[Z_idx]
    if field_orientation in {"parallel", "par"}:
        return Z * kappa_par

    def f_kappa_perp(Z_idx):
        numerator = (13 / 4 * Z + np.sqrt(2)) * r + kappa_0[Z_idx] * kappa_par_e[Z_idx]
        denominator = (
            r**3
            + kappa_4[Z_idx] * r ** (7 / 3)
            + kappa_3[Z_idx] * r**2
            + kappa_2[Z_idx] * r ** (5 / 3)
            + kappa_1[Z_idx] * r
            + kappa_0[Z_idx]
        )
        return numerator / denominator

    kappa_perp = f_kappa_perp(Z_idx)
    if field_orientation in {"perpendicular", "perp"}:
        return Z * kappa_perp

    def f_kappa_cross(Z_idx):
        numerator = r * (5 / 2 * r + k_0[Z_idx] / k_5[Z_idx])
        denominator = (
            r**3
            + k_4[Z_idx] * r ** (7 / 3)
            + k_3[Z_idx] * r**2
            + k_2[Z_idx] * r ** (5 / 3)
            + k_1[Z_idx] * r
            + k_0[Z_idx]
        )
        return numerator / denominator

    kappa_cross = f_kappa_cross(Z_idx)
    if field_orientation == "cross":
        return Z * kappa_cross

    if field_orientation == "all":
        return np.array((Z * kappa_par, Z * kappa_perp, Z * kappa_cross))


def _nondim_resist_ji_held(hall, Z, field_orientation):
    """
    Dimensionless resistivity — Ji-Held.

    Ji, Jeong-Young, and Eric D. Held. "Closure and transport theory for
    high-collisionality electron-ion plasmas." Physics of Plasmas 20.4 (2013):
    042114.
    """
    allowed_Z = [1, 2, "arbitrary"]
    Z_idx = _check_Z(allowed_Z, Z)
    # fixing overflow errors when exponentiating r by making a float
    # instead of an int
    r = float(np.abs(Z * hall))

    def f_alpha_par_e(Z):
        numerator = Z ** (2 / 3)
        denominator = 1.46 * Z ** (2 / 3) - 0.330 * Z ** (1 / 3) + 0.888
        return 1 - numerator / denominator

    def f_alpha_0(Z):
        return 0.623 * Z ** (5 / 3) - 2.61 * Z ** (4 / 3) + 3.56 * Z + 0.557

    def f_alpha_1(Z):
        return 2.24 * Z ** (2 / 3) - 1.11 * Z ** (1 / 3) + 1.84

    def f_alpha_2(Z):
        return -0.0983 * Z ** (1 / 3) + 0.0176

    def f_a_0(Z):
        return 0.0759 * Z ** (8 / 3) + 0.897 * Z**2 + 2.06 * Z + 1.06

    def f_a_1(Z):
        return 2.18 * Z ** (5 / 3) + 5.31 * Z + 3.73

    def f_a_2(Z):
        return 7.41 * Z + 1.11 * Z ** (2 / 3) - 1.17

    def f_a_3(Z):
        return 3.89 * Z ** (2 / 3) - 4.51 * Z ** (1 / 3) + 6.76

    def f_a_4(Z):
        return 2.26 * Z ** (1 / 3) + 0.281

    def f_a_5(Z):
        return 1.18 * Z ** (5 / 3) - 1.03 * Z ** (4 / 3) + 3.60 * Z + 1.32

    alpha_par_e = [0.504, 0.431, f_alpha_par_e(Z)]
    alpha_0 = [2.130, 3.078, f_alpha_0(Z)]
    alpha_1 = [2.970, 3.997, f_alpha_1(Z)]
    alpha_2 = [-0.081, -0.106, f_alpha_2(Z)]
    a_0 = [4.093, 9.250, f_a_0(Z)]
    a_1 = [11.22, 21.27, f_a_1(Z)]
    a_2 = [7.350, 15.41, f_a_2(Z)]
    a_3 = [6.140, 7.253, f_a_3(Z)]
    a_4 = [2.541, 3.128, f_a_4(Z)]
    a_5 = [5.070, 9.671, f_a_5(Z)]

    alpha_par = alpha_par_e[Z_idx]
    if field_orientation in {"parallel", "par"}:
        return alpha_par

    def f_alpha_perp(Z_idx):
        numerator = 1.46 * Z ** (2 / 3) * r + alpha_0[Z_idx] * (1 - alpha_par_e[Z_idx])
        denominator = (
            r ** (5 / 3)
            + alpha_2[Z_idx] * r ** (4 / 3)
            + alpha_1[Z_idx] * r
            + alpha_0[Z_idx]
        )
        return 1 - numerator / denominator

    alpha_perp = f_alpha_perp(Z_idx)
    if field_orientation in {"perpendicular", "perp"}:
        return alpha_perp

    def f_alpha_cross(Z_idx):
        numerator = Z ** (2 / 3) * r * (2.53 * r + a_0[Z_idx] / a_5[Z_idx])
        denominator = (
            r ** (8 / 3)
            + a_4[Z_idx] * r ** (7 / 3)
            + a_3[Z_idx] * r**2
            + a_2[Z_idx] * r ** (5 / 3)
            + a_1[Z_idx] * r
            + a_0[Z_idx]
        )
        return numerator / denominator

    alpha_cross = f_alpha_cross(Z_idx)
    if field_orientation == "cross":
        return alpha_cross

    if field_orientation == "all":
        return np.array((alpha_par, alpha_perp, alpha_cross))


def _nondim_tec_ji_held(hall, Z, field_orientation):
    """
    Dimensionless thermoelectric conductivity — Ji-Held.

    Ji, Jeong-Young, and Eric D. Held. "Closure and transport theory for
    high-collisionality electron-ion plasmas." Physics of Plasmas 20.4 (2013):
    042114.
    """
    allowed_Z = [1, 2, "arbitrary"]
    Z_idx = _check_Z(allowed_Z, Z)
    # fixing overflow errors when exponentiating r by making a float
    # instead of an int
    r = float(np.abs(Z * hall))

    def f_beta_par_e(Z):
        numerator = Z ** (5 / 3)
        denominator = 0.693 * Z ** (5 / 3) - 0.279 * Z ** (4 / 3) + Z + 0.01
        return numerator / denominator

    def f_beta_0(Z):
        return 0.156 * Z ** (8 / 3) + 0.994 * Z**2 + 3.21 * Z - 0.84

    def f_beta_1(Z):
        return 3.69 * Z ** (5 / 3) + 3.77 * Z + 0.77

    def f_beta_2(Z):
        return 9.43 * Z + 4.22 * Z ** (2 / 3) - 12.9 * Z ** (1 / 3) + 4.56

    def f_beta_3(Z):
        return 2.70 * Z ** (2 / 3) + 1.46 * Z ** (1 / 3) - 0.17

    def f_beta_4(Z):
        return 2.58 * Z ** (1 / 3) + 0.17

    def f_b_0(Z):
        numerator = 6.87 * Z**3 + 78.2 * Z**2 + 623 * Z + 366
        denominator = 1000
        return numerator / denominator

    def f_b_1(Z):
        return 0.134 * Z**2 + 0.977 * Z + 0.17

    def f_b_2(Z):
        return 0.689 * Z ** (4 / 3) - 0.377 * Z ** (2 / 3) + 3.94 * Z ** (1 / 3) + 0.644

    def f_b_3(Z):
        return -0.109 * Z + 1.33 * Z ** (2 / 3) - 3.80 * Z ** (1 / 3) + 0.289

    def f_b_4(Z):
        return 2.46 * Z ** (2 / 3) + 0.522

    def f_b_5(Z):
        return 0.102 * Z**2 + 0.746 * Z + 0.072 * Z ** (1 / 3) + 0.211

    beta_par_e = [0.702, 0.905, f_beta_par_e(Z)]
    beta_0 = [3.520, 10.55, f_beta_0(Z)]
    beta_1 = [8.230, 20.03, f_beta_1(Z)]
    beta_2 = [5.310, 13.87, f_beta_2(Z)]
    beta_3 = [3.990, 5.955, f_beta_3(Z)]
    beta_4 = [2.750, 3.421, f_beta_4(Z)]
    b_0 = [1.074, 1.980, f_b_0(Z)]
    b_1 = [1.281, 2.660, f_b_1(Z)]
    b_2 = [4.896, 6.746, f_b_2(Z)]
    b_3 = [-2.290, -2.605, f_b_3(Z)]
    b_4 = [2.982, 4.427, f_b_4(Z)]
    b_5 = [1.131, 2.202, f_b_5(Z)]

    beta_par = beta_par_e[Z_idx]
    if field_orientation in {"parallel", "par"}:
        return beta_par

    def f_beta_perp(Z_idx):
        numerator = 6.33 * Z ** (5 / 3) * r + beta_0[Z_idx] * beta_par_e[Z_idx]
        denominator = (
            r ** (8 / 3)
            + beta_4[Z_idx] * r ** (7 / 3)
            + beta_3[Z_idx] * r**2
            + beta_2[Z_idx] * r ** (5 / 3)
            + beta_1[Z_idx] * r
            + beta_0[Z_idx]
        )
        return numerator / denominator

    beta_perp = f_beta_perp(Z_idx)
    if field_orientation in {"perpendicular", "perp"}:
        return beta_perp

    def f_beta_cross(Z_idx):
        numerator = Z * r * (3 / 2 * r + b_0[Z_idx] / b_5[Z_idx])
        denominator = (
            r**3
            + b_4[Z_idx] * r ** (7 / 3)
            + b_3[Z_idx] * r**2
            + b_2[Z_idx] * r ** (5 / 3)
            + b_1[Z_idx] * r
            + b_0[Z_idx]
        )
        return numerator / denominator

    beta_cross = f_beta_cross(Z_idx)
    if field_orientation == "cross":
        return beta_cross

    if field_orientation == "all":
        return np.array((beta_par, beta_perp, beta_cross))


def _nondim_visc_e_ji_held(hall, Z):
    """
    Dimensionless electron viscosity — Ji-Held.

    Ji, Jeong-Young, and Eric D. Held. "Closure and transport theory for
    high-collisionality electron-ion plasmas." Physics of Plasmas 20.4 (2013):
    042114.
    """
    allowed_Z = [1, 2, "arbitrary"]
    Z_idx = _check_Z(allowed_Z, Z)
    # fixing overflow errors when exponentiating r by making a float
    # instead of an int
    r = float(np.abs(Z * hall))

    def f_eta_0_e(Z):
        return 1 / (0.55 * Z + 0.083 * Z ** (1 / 3) + 0.732)

    def f_hprime_0(Z):
        return 0.0699 * Z**3 + 0.558 * Z**2 + 1.66 * Z + 1.06

    def f_hprime_1(Z):
        return 0.657 * Z**2 + 1.42 * Z + 0.416

    def f_hprime_2(Z):
        return -0.369 * Z ** (4 / 3) + 0.379 * Z + 0.339 * Z ** (1 / 3) + 2.17

    def f_hprime_3(Z):
        return 2.16 * Z - 0.657 * Z ** (1 / 3) + 0.0347

    def f_hprime_4(Z):
        return -0.0703 * Z ** (2 / 3) - 0.224 * Z ** (1 / 3) + 0.333

    def f_h_0(Z):
        return 0.0473 * Z**3 + 0.323 * Z**2 + 0.951 * Z + 0.407

    def f_h_1(Z):
        return 0.171 * Z**2 + 0.523 * Z + 0.336

    def f_h_2(Z):
        return 0.362 * Z ** (4 / 3) + 0.178 * Z + 1.06 * Z ** (1 / 3) + 1.26

    def f_h_3(Z):
        return 0.599 * Z + 0.106 * Z ** (2 / 3) - 0.444 * Z ** (1 / 3) - 0.161

    def f_h_4(Z):
        return -0.16 * Z ** (2 / 3) + 0.06 * Z ** (1 / 3) + 0.232

    def f_h_5(Z):
        return 0.183 * Z**2 + 0.714 * Z + 0.0375 * Z ** (1 / 3) + 0.47

    eta_0_e = [0.733, 0.516, f_eta_0_e(Z)]
    hprime_0 = [3.348, 7.171, f_hprime_0(Z)]
    hprime_1 = [2.493, 5.884, f_hprime_1(Z)]
    hprime_2 = [2.519, 2.425, f_hprime_2(Z)]
    hprime_3 = [1.538, 3.527, f_hprime_3(Z)]
    hprime_4 = [0.039, -0.061, f_hprime_4(Z)]
    h_0 = [1.728, 3.979, f_h_0(Z)]
    h_1 = [1.030, 2.066, f_h_1(Z)]
    h_2 = [2.860, 3.864, f_h_2(Z)]
    h_3 = [0.100, 0.646, f_h_3(Z)]
    h_4 = [0.132, 0.054, f_h_4(Z)]
    h_5 = [1.405, 2.677, f_h_5(Z)]

    eta_0 = eta_0_e[Z_idx]

    def f_eta_2(Z_idx, r):
        numerator = (6 / 5 * Z + 3 / 5 * np.sqrt(2)) * r + hprime_0[Z_idx] * eta_0_e[
            Z_idx
        ]
        denominator = (
            r**3
            + hprime_4[Z_idx] * r ** (7 / 3)
            + hprime_3[Z_idx] * r**2
            + hprime_2[Z_idx] * r ** (5 / 3)
            + hprime_1[Z_idx] * r
            + hprime_0[Z_idx]
        )
        return numerator / denominator

    eta_2 = f_eta_2(Z_idx, r)

    eta_1 = f_eta_2(Z_idx, 2 * r)

    def f_eta_4(Z_idx, r):
        numerator = r * (r + h_0[Z_idx] / h_5[Z_idx])
        denominator = (
            r**3
            + h_4[Z_idx] * r ** (7 / 3)
            + h_3[Z_idx] * r**2
            + h_2[Z_idx] * r ** (5 / 3)
            + h_1[Z_idx] * r
            + h_0[Z_idx]
        )
        return numerator / denominator

    eta_4 = f_eta_4(Z_idx, r)

    eta_3 = f_eta_4(Z_idx, 2 * r)

    return np.array((eta_0, eta_1, eta_2, eta_3, eta_4))


def _nondim_tc_i_ji_held(hall, Z, mu, theta, field_orientation, K=3):
    """
    Dimensionless ion thermal conductivity — Ji-Held.

    Ji, Jeong-Young, and Eric D. Held. "Closure and transport theory for
    high-collisionality electron-ion plasmas." Physics of Plasmas 20.4 (2013):
    042114.
    """
    #    mu = m_e / m_i
    #    theta = T_e / T_i
    zeta = 1 / Z * np.sqrt(mu / theta)
    r = np.abs(hall / np.sqrt(2))

    #    K = 2  # 2x2 moments, equivalent to original Braginskii
    #    K = 3  # 3x3 moments

    if K == 2:
        Delta_par_i1 = 1 + 13.50 * zeta + 36.46 * zeta**2
        kappa_par_i = (5.524 + 30.38 * zeta) / Delta_par_i1
    elif K == 3:
        Delta_par_i1 = 1 + 26.90 * zeta + 187.5 * zeta**2 + 346.9 * zeta**3
        kappa_par_i = (5.586 + 101.7 * zeta + 289.1 * zeta**2) / Delta_par_i1
    if field_orientation in ("parallel", "par"):
        return kappa_par_i / np.sqrt(2)

    if K == 3:
        Delta_perp_i1 = (
            r**6
            + (3.635 + 29.15 * zeta + 83 * zeta**2) * r**4
            + (
                1.395
                + 35.64 * zeta
                + 344.9 * zeta**2
                + 1345 * zeta**3
                + 1891 * zeta**4
            )
            * r**2
            + 0.09163 * Delta_par_i1**2
        )
        kappa_perp_i = (
            (np.sqrt(2) + 15 / 2 * zeta) * r**4
            + (3.841 + 57.59 * zeta + 297.8 * zeta**2 + 555 * zeta**3) * r**2
            + 0.09163 * kappa_par_i * Delta_par_i1**2
        ) / Delta_perp_i1
    elif K == 2:
        Delta_perp_i1 = (
            r**4
            + (1.352 + 12.49 * zeta + 34 * zeta**2) * r**2
            + 0.1693 * Delta_par_i1**2
        )
        kappa_perp_i = (
            (np.sqrt(2) + 15 / 2 * zeta) * r**2
            + 0.1693 * kappa_par_i * Delta_par_i1**2
        ) / Delta_perp_i1
    if field_orientation in ("perpendicular", "perp"):
        return kappa_perp_i / np.sqrt(2)

    if K == 2:
        kappa_cross_i = (
            r
            * (5 / 2 * r**2 + 2.323 + 22.73 * zeta + 62.5 * zeta**2)
            / Delta_perp_i1
        )
    elif K == 3:
        kappa_cross_i = (
            r
            * (
                5 / 2 * r**4
                + (7.963 + 64.40 * zeta + 185 * zeta**2) * r**2
                + 1.344
                + 44.54 * zeta
                + 511.9 * zeta**2
                + 2155 * zeta**3
                + 3063 * zeta**4
            )
            / Delta_perp_i1
        )
    if field_orientation == "cross":
        return kappa_cross_i / np.sqrt(2)

    if field_orientation == "all":
        return np.array(
            (
                kappa_par_i / np.sqrt(2),
                kappa_perp_i / np.sqrt(2),
                kappa_cross_i / np.sqrt(2),
            )
        )


def _nondim_visc_i_ji_held(hall, Z, mu, theta, K=3):
    """
    Dimensionless ion viscosity — Ji-Held.

    Ji, Jeong-Young, and Eric D. Held. "Closure and transport theory for
    high-collisionality electron-ion plasmas." Physics of Plasmas 20.4 (2013):
    042114.
    """
    zeta = 1 / Z * np.sqrt(mu / theta)
    r = np.abs(hall / np.sqrt(2))
    r13 = 2 * r

    #    K = 2  # 2x2 moments, equivalent to original Braginskii
    #    K = 3  # 3x3 moments

    if K == 3:
        Delta_par_i2 = 1 + 15.79 * zeta + 63.92 * zeta**2 + 71.69 * zeta**3
        eta_0_i = (1.365 + 16.75 * zeta + 35.84 * zeta**2) / Delta_par_i2

        def Delta_perp_i2(r, zeta, Delta_par_i2):
            return (
                r**6
                + (4.391 + 26.69 * zeta + 56 * zeta**2) * r**4
                + (
                    3.191
                    + 49.62 * zeta
                    + 306.4 * zeta**2
                    + 808.1 * zeta**3
                    + 784 * zeta**4
                )
                * r**2
                + 0.4483 * Delta_par_i2**2
            )

        Delta_perp_i2_24 = Delta_perp_i2(r, zeta, Delta_par_i2)
        Delta_perp_i2_13 = Delta_perp_i2(r13, zeta, Delta_par_i2)

        def f_eta_2(r, zeta, Delta_perp_i2):
            eta_2_i = (
                (3 / 5 * np.sqrt(2) + 2 * zeta) * r**4
                + (2.680 + 25.98 * zeta + 90.71 * zeta**2 + 104 * zeta**3) * r**2
                + 0.4483 * eta_0_i * Delta_par_i2**2
            ) / Delta_perp_i2
            return eta_2_i

        eta_2_i = f_eta_2(r, zeta, Delta_perp_i2_24)
        eta_1_i = f_eta_2(r13, zeta, Delta_perp_i2_13)

        def f_eta_4(r, zeta, Delta_perp_i2):
            eta_4_i = (
                r
                * (
                    r**4
                    + (3.535 + 23.30 * zeta + 52 * zeta**2) * r**2
                    + 0.9538
                    + 21.81 * zeta
                    + 174.2 * zeta**2
                    + 538.4 * zeta**3
                    + 576 * zeta**4
                )
                / Delta_perp_i2
            )
            return eta_4_i

        eta_4_i = f_eta_4(r, zeta, Delta_perp_i2_24)
        eta_3_i = f_eta_4(r13, zeta, Delta_perp_i2_13)

    elif K == 2:
        Delta_par_i2 = 1 + 7.164 * zeta + 10.49 * zeta**2
        eta_0_i = (1.357 + 5.243 * zeta) / Delta_par_i2

        def Delta_perp_i2(r, zeta, Delta_par_i2):
            Delta_perp_i2 = (
                r**4
                + (2.023 + 11.68 * zeta + 20 * zeta**2) * r**2
                + 0.5820 * Delta_par_i2**2
            )
            return Delta_perp_i2

        Delta_perp_i2_24 = Delta_perp_i2(r, zeta, Delta_par_i2)
        Delta_perp_i2_13 = Delta_perp_i2(r13, zeta, Delta_par_i2)

        def f_eta_2(r, zeta, Delta_perp_i2):
            eta_2_i = (
                (3 / 5 * np.sqrt(2) + 2 * zeta) * r**2
                + 0.5820 * eta_0_i * Delta_par_i2**2
            ) / Delta_perp_i2
            return eta_2_i

        eta_2_i = f_eta_2(r, zeta, Delta_perp_i2_24)
        eta_1_i = f_eta_2(r13, zeta, Delta_perp_i2_13)

        def f_eta_4(r, zeta, Delta_perp_i2):
            Delta_perp_i2 = (
                r**4
                + (2.023 + 11.68 * zeta + 20 * zeta**2) * r**2
                + 0.5820 * Delta_par_i2**2
            )
            eta_4_i = (
                r * (r**2 + 1.188 + 8.283 * zeta + 16 * zeta**2) / Delta_perp_i2
            )
            return eta_4_i

        eta_4_i = f_eta_4(r, zeta, Delta_perp_i2_24)
        eta_3_i = f_eta_4(r13, zeta, Delta_perp_i2_13)

    return np.array(
        (
            eta_0_i / np.sqrt(2),
            eta_1_i / np.sqrt(2),
            eta_2_i / np.sqrt(2),
            eta_3_i / np.sqrt(2),
            eta_4_i / np.sqrt(2),
        )
    )
