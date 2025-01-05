"""Lengths related to particle collisions."""

__all__ = ["impact_parameter_perp", "impact_parameter", "mean_free_path"]


import astropy.units as u
import numpy as np
from astropy.constants.si import eps0, hbar

from plasmapy.formulary.collisions import frequencies, misc
from plasmapy.formulary.lengths import Debye_length
from plasmapy.formulary.quantum import Wigner_Seitz_radius
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.particle_class import Particle
from plasmapy.utils.decorators import validate_quantities


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()}
)
@particle_input
def impact_parameter_perp(
    T: u.Quantity[u.K],
    species: (Particle, Particle),
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
) -> u.Quantity[u.m]:
    r"""
    Distance of the closest approach for a 90° Coulomb collision.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Temperature in units of temperature or energy per particle,
        which is assumed to be equal for both the test particle and the
        target particle

    species : `tuple`
        A tuple containing string representations of the test particle
        (listed first) and the target particle (listed second)

    V : `~astropy.units.Quantity`, optional
        The relative velocity between particles.  If not provided,
        thermal velocity is assumed: :math:`μ V^2 \sim 2 k_B T` where
        :math:`μ` is the reduced mass.

    Returns
    -------
    impact_parameter_perp : `float` or `numpy.ndarray`
        The distance of the closest approach for a 90° Coulomb collision.

    Raises
    ------
    `ValueError`
        If the mass or charge of either particle cannot be found, or
        any of the inputs contain incorrect values.

    `~astropy.units.UnitConversionError`
        If the units on any of the inputs are incorrect.

    `TypeError`
        If either of ``T`` or ``V`` is not a `~astropy.units.Quantity`.

    `~plasmapy.utils.exceptions.RelativityError`
        If the input velocity is same or greater than the speed of
        light.

    Warns
    -----
    : `~astropy.units.UnitsWarning`
        If units are not provided, SI units are assumed.

    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the input velocity is greater than 5% of the speed of light.

    Notes
    -----
    The distance of closest approach, ``impact_parameter_perp``, is
    given by (see Ch. 5 of :cite:t:`chen:2016`):

    .. math::

        b_⟂ = \frac{Z_1 Z_2}{4 π ε_0 m v^2}

    Examples
    --------
    >>> import astropy.units as u
    >>> T = 1e6 * u.K
    >>> species = ("e", "p")
    >>> impact_parameter_perp(T, species)
    <Quantity 8.3550...e-12 m>
    """
    # Note: This formulation corresponds to collisions that result in a
    #       deflection of 90°s, which is valid when classical effects
    #       dominate.
    # TODO: need to incorporate an average ionization parameter

    T, masses, charges, reduced_mass, V = misc._process_inputs(  # noqa: SLF001
        T=T, species=species, V=V
    )

    return charges[0] * charges[1] / (4 * np.pi * eps0 * reduced_mass * V**2)


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
    V={"none_shall_pass": True},
)
def impact_parameter(  # noqa: C901
    T: u.Quantity[u.K],
    n_e: u.Quantity[u.m**-3],
    species,
    z_mean: float = np.nan,
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    method: str = "classical",
):
    r"""
    Impact parameters for classical and quantum Coulomb collision.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Temperature in units of temperature or energy per particle,
        which is assumed to be equal for both the test particle and the
        target particle.

    n_e : `~astropy.units.Quantity`
        The electron number density in units convertible to per cubic
        meter.

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
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for
        more information about these methods.

    Returns
    -------
    bmin, bmax : `tuple` of floats
        The minimum and maximum impact parameters (distances) for a
        Coulomb collision.

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
        If units are not provided, SI units are assumed.

    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the input velocity is greater than 5% of the speed of light.

    Notes
    -----
    The minimum and maximum impact parameters may be calculated in a
    variety of ways. The maximum impact parameter is typically the Debye
    length.

    For quantum plasmas the maximum impact parameter can be the
    quadratic sum of the debye length and ion radius
    (``Wigner_Seitz_radius``) :cite:p:`gericke:2002`:

    .. math::

        b_{max} = \left(λ_{De}^2 + a_i^2\right)^{1/2}

    The minimum impact parameter is typically some combination of the
    thermal de Broglie wavelength and the distance of closest approach
    for a 90° Coulomb collision. A quadratic sum is used for all GMS
    methods, except for GMS-5, where ``b_min`` is simply set to the
    distance of closest approach :cite:p:`gericke:2002`.

    .. math::

        b_{min} = \left(Λ_{de Broglie}^2 + ρ_⟂^2\right)^{1/2}

    Examples
    --------
    >>> import astropy.units as u
    >>> n = 1e19 * u.m**-3
    >>> T = 1e6 * u.K
    >>> species = ("e", "p")
    >>> impact_parameter(T, n, species)
    (<Quantity 1.051...e-11 m>, <Quantity 2.182...e-05 m>)
    >>> impact_parameter(T, n, species, V=1e6 * u.m / u.s)
    (<Quantity 2.534...e-10 m>, <Quantity 2.182...e-05 m>)
    """
    T, masses, charges, reduced_mass, V = misc._process_inputs(  # noqa: SLF001
        T=T, species=species, V=V
    )
    # catching error where mean charge state is not given for non-classical
    # methods that require the ion density
    if method in {
        "ls_full_interp",
        "GMS-2",
        "hls_max_interp",
        "GMS-5",
        "hls_full_interp",
        "GMS-6",
    } and np.isnan(z_mean):
        raise ValueError(
            'Must provide a z_mean for "ls_full_interp", '
            '"hls_max_interp", and "hls_full_interp" methods.'
        )
    # Debye length
    lambdaDe = Debye_length(T, n_e)
    # de Broglie wavelength
    lambdaBroglie = hbar / (2 * reduced_mass * V)
    # distance of the closest approach in 90° Coulomb collision
    bPerp = impact_parameter_perp(T=T, species=species, V=V)

    # obtaining minimum and maximum impact parameters depending on which
    # method is requested
    if method in {"classical", "ls"}:
        bmax = lambdaDe
        # Coulomb-style collisions will not happen for impact parameters
        # shorter than either of these two impact parameters, so we choose
        # the larger of these two possibilities. That is, between the
        # de Broglie wavelength and the distance of the closest approach.
        # ARRAY NOTES
        # T and V should be guaranteed to be same size inputs from _process_inputs
        # therefore, lambdaBroglie and bPerp are either both scalar or both array
        # if np.isscalar(bPerp.value) and np.isscalar(lambdaBroglie.value):
        # both scalar
        try:  # assume both scalar
            bmin = max(bPerp, lambdaBroglie)
        except ValueError:  # both lambdaBroglie and bPerp are arrays
            bmin = lambdaBroglie
            bmin[bPerp > lambdaBroglie] = bPerp[bPerp > lambdaBroglie]
    elif method in {"ls_min_interp", "GMS-1"}:
        # 1st method listed in Table 1 of reference [1]
        # This is just another form of the classical Landau-Spitzer
        # approach, but bmin is interpolated between the de Broglie
        # wavelength and distance of the closest approach.
        bmax = lambdaDe
        bmin = (lambdaBroglie**2 + bPerp**2) ** (1 / 2)
    elif method in {"ls_full_interp", "GMS-2"}:
        # 2nd method listed in Table 1 of reference [1]
        # Another Landau-Spitzer like approach, but now bmax is also
        # being interpolated. The interpolation is between the Debye
        # length and the ion sphere radius, allowing for descriptions
        # of dilute plasmas.
        # Mean ion density.
        n_i = n_e / z_mean
        # mean ion sphere radius.
        ionRadius = Wigner_Seitz_radius(n_i)
        bmax = (lambdaDe**2 + ionRadius**2) ** (1 / 2)
        bmin = (lambdaBroglie**2 + bPerp**2) ** (1 / 2)
    elif method in {"ls_clamp_mininterp", "GMS-3"}:
        # 3rd method listed in Table 1 of reference [1]
        # same as GMS-1, but not Lambda has a clamp at Lambda_min = 2
        # where Lambda is the argument to the Coulomb logarithm.
        bmax = lambdaDe
        bmin = (lambdaBroglie**2 + bPerp**2) ** (1 / 2)
    elif method in {"hls_min_interp", "GMS-4"}:
        # 4th method listed in Table 1 of reference [1]
        bmax = lambdaDe
        bmin = (lambdaBroglie**2 + bPerp**2) ** (1 / 2)
    elif method in {"hls_max_interp", "GMS-5"}:
        # 5th method listed in Table 1 of reference [1]
        # Mean ion density.
        n_i = n_e / z_mean
        # mean ion sphere radius.
        ionRadius = Wigner_Seitz_radius(n_i)
        bmax = (lambdaDe**2 + ionRadius**2) ** (1 / 2)
        bmin = bPerp
    elif method in {"hls_full_interp", "GMS-6"}:
        # 6th method listed in Table 1 of reference [1]
        # Mean ion density.
        n_i = n_e / z_mean
        # mean ion sphere radius.
        ionRadius = Wigner_Seitz_radius(n_i)
        bmax = (lambdaDe**2 + ionRadius**2) ** (1 / 2)
        bmin = (lambdaBroglie**2 + bPerp**2) ** (1 / 2)
    else:
        raise ValueError(f"Method {method} not found!")

    # ARRAY NOTES
    # it could be that bmin and bmax have different sizes. If Te is a scalar,
    # T and V will be scalar from _process_inputs, so bmin will scalar. However
    # if n_e is an array, then bmax will be an array. if this is the case,
    # we want to extend the scalar bmin to match the dimensions of bmax.
    if bmin.size == 1 and bmax.size != 1:
        bmin = bmin * np.ones(bmax.shape)

    return bmin.to(u.m), bmax.to(u.m)


@validate_quantities(
    T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    n_e={"can_be_negative": False},
)
def mean_free_path(
    T: u.Quantity[u.K],
    n_e: u.Quantity[u.m**-3],
    species,
    z_mean: float = np.nan,
    V: u.Quantity[u.m / u.s] = np.nan * u.m / u.s,
    method: str = "classical",
) -> u.Quantity[u.m]:
    r"""
    Collisional mean free path (m).

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        Temperature in units of temperature or energy per particle,
        which is assumed to be equal for both the test particle and the
        target particle.

    n_e : `~astropy.units.Quantity`
        The electron number density in units convertible to per cubic
        meter.

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
        `~plasmapy.formulary.collisions.coulomb.Coulomb_logarithm` for
        more information about these methods.

    Returns
    -------
    mfp : `float` or `numpy.ndarray`
        The collisional mean free path for particles in a plasma.

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
        If units are not provided, SI units are assumed.

    : `~plasmapy.utils.exceptions.RelativityWarning`
        If the input velocity is greater than 5% of the speed of light.

    Notes
    -----
    The collisional mean free path (see :cite:t:`chen:2016`) is given by:

    .. math::

        λ_{mfp} = \frac{v}{ν}

    where :math:`v` is the inter-particle velocity (typically taken to
    be the thermal velocity) and :math:`ν` is the collision frequency.

    Examples
    --------
    >>> import astropy.units as u
    >>> n = 1e19 * u.m**-3
    >>> T = 1e6 * u.K
    >>> mean_free_path(T, n, ("e-", "p+"))
    <Quantity 7.839... m>
    >>> mean_free_path(T, n, ("e-", "p+"), V=1e6 * u.m / u.s)
    <Quantity 0.0109... m>
    """
    # collisional frequency
    freq = frequencies.collision_frequency(
        T=T, n=n_e, species=species, z_mean=z_mean, V=V, method=method
    )
    # boiler plate to fetch velocity
    # this has been moved to after collision_frequency to avoid use of
    # reduced mass thermal velocity in electron-ion collision case.
    # Should be fine since collision_frequency has its own _process_inputs
    # check, and we are only using this here to get the velocity.
    T, masses, charges, reduced_mass, V = misc._process_inputs(  # noqa: SLF001
        T=T, species=species, V=V
    )
    return V / freq
