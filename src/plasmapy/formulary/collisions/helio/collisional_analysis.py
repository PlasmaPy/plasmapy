"""Functionality related to collisional analysis of a plasma in transit."""

__all__ = ["temp_ratio"]

import logging
import numbers

import astropy.units as u
import numpy as np

from plasmapy.particles import ParticleLike, ParticleList
from plasmapy.utils.decorators import validate_quantities


@validate_quantities(
    T_1={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_2={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def temp_ratio(  # noqa: C901
    *,
    r_0: u.Quantity[u.au],
    r_n: u.Quantity[u.au],
    n_1: u.Quantity[u.cm**-3],
    n_2: u.Quantity[u.cm**-3],
    v_1: u.Quantity[u.km / u.s],
    T_1: u.Quantity[u.K],
    T_2: u.Quantity[u.K],
    ions: ParticleLike = ("p+", "He-4++"),
    n_step: int = 100,
    density_scale: float = -1.8,
    velocity_scale: float = -0.2,
    temperature_scale: float = -0.74,
    verbose: bool = False,
):
    r"""
    Calculate the thermalization ratio for a plasma in transit.

    This function allows the thermalization of a plasma to be modeled,
    predicting the temperature ratio for different ion species within a
    plasma at a different point in space. Taken from
    :cite:t:`maruca:2013` and :cite:t:`johnson:2023a`.

    Parameters
    ----------
    r_0 : `~astropy.units.Quantity`, |keyword-only|
        Starting position of the plasma in units convertible to
        astronomical units.

    r_n : `~astropy.units.Quantity`, |keyword-only|
        Final position of the plasma in units convertible to
        astronomical units.

    n_1 : `~astropy.units.Quantity`, |keyword-only|
        The primary ion number density in units convertible to
        m\ :sup:`-3`.

    n_2 : `~astropy.units.Quantity`, |keyword-only|
        The secondary ion number density in units convertible to
        m\ :sup:`-3`.

    v_1 : `~astropy.units.Quantity`, |keyword-only|
        The primary ion speed in units convertible to km s\ :sup:`-1`.

    T_1 : `~astropy.units.Quantity`, |keyword-only|
        Temperature of the primary ion in units convertible to kelvin.

    T_2 : `~astropy.units.Quantity`, |keyword-only|
        Temperature of the secondary ion in units convertible to
        kelvin.

    ions : |particle-list-like|, |keyword-only|, default: ``("p+, "He-4 2+")``
        Particle list containing two particles, with the primary ion
        first and the secondary ion second.

    n_step : `int`, |keyword-only|
        The number of intervals used in solving a differential equation
        via the Euler method. Must be positive.

    density_scale : `float`, |keyword-only|, default: ``-1.8``
        The value used as the scaling parameter for the primary ion
        density. The default value is taken from
        :cite:t:`hellinger:2011`.

    velocity_scale : `float`, |keyword-only|, default: ``-0.2``
        The value used as the scaling parameter for the primary ion
        velocity. The default value is taken from
        :cite:t:`hellinger:2011`.

    temperature_scale : `float`, |keyword-only|, default: ``-0.74``
        The value used as the scaling parameter for the primary ion
        temperature. The default value is taken from
        :cite:t:`hellinger:2011`.

    Returns
    -------
    theta : `float`
        The dimensionless ion temperature ratio prediction for the
        distance provided.

    Raises
    ------
    `TypeError`
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    Notes
    -----
    The processes by which Coulomb collisions bring ion temperatures
    into local thermal equilibrium (LTE) has received considerable
    attention :cite:p:`verscharen:2019`. The relative temperature
    between constituent plasma ion species is given as:

    .. math::

        θ_{21} = \frac{T_2}{T_1} \, ,

    where :math:`T_1` and :math:`T_2` are the scalar temperatures for
    the primary ion of interest and the secondary ion, respectively. The
    scalar temperature defined as:

    .. math::

        T_{\rm i} = \frac{2T_{{\rm i}, ⟂} + T_{{\rm i}, ∥}}{3} \, ,

    where :math:`T_{{\rm i}, ⟂}` and :math:`T_{{\rm i}, ∥}`
    are the temperature of the :math:`\mathrm{i}`-particles along the
    axes perpendicular and parallel to the ambient magnetic field.

    In order to determine how extensively an individual parcel of
    plasma has been processed by Coulomb collisions, :cite:t:`maruca:2013`
    introduced an approached called collisional analysis. This paper
    quantifies how collisions affect the plasma's departures from LTE.
    The equation for collisional thermalization :cite:p:`maruca:2013`
    is:

     .. math::

        \frac{d θ_{21}}{dr} = A \left (
        \frac{n_1}{v_1 T_1^{3/2}} \right ) \frac{\left( μ_1 μ_2
        \right )^{1/2} Z_1 Z_2 \left( 1 - θ_{21} \right )
        \left(1 + η_{21}θ_{21} \right )}{\left(
        \frac{μ_2}{μ_1} + θ_{21} \right )^{3/2}} λ_{21}

    and

    .. math::

         λ_{21} = 9 + \ln \left| B \left ( \frac{T^3_1}{n_1}
         \right )^{1/2} \left( \frac{Z_1Z_2(μ_1 + μ_2) }{θ_{21} +
         \frac{μ_2}{μ_1}} \right )
         \left( \frac{n_2 Z_2^2}{n_1 Z_1^2} + θ_{21}
         \right)^{1/2}\right |

    With :math:`η = \frac{n_2}{n_1}`, :math:`θ =
    \frac{T_2}{T_1}`, :math:`A = 2.60 × 10^{7} \, {\rm cm}^3
    \, {\rm km} \, {\rm K}^{3/2} \, {\rm s}^{-1} \, {\rm au}^{-1}`,
    and :math:`B = 1 \, {\rm cm}^{-3/2}{~\rm K}^{-3/2}`.

    The thermalization is from Coulomb collisions, which assumes
    "soft", small-angle deflections mediated by the electrostatic
    force :cite:p:`baumjohann:1997`. It is assumed that there is no
    relative drift between the ion species and that it is a mixed ion
    collision, and the Coulomb logarithm for a mixed ion collision is
    given by :cite:t:`nrlformulary`.

    The density, velocity and temperature of the primary ion can be
    radially scaled, as seen below. The values for the scaling can be
    altered, though the default values are taken from
    :cite:t:`hellinger:2011`.

    .. math::

        n(r) \propto r^{-1.8}\ , \hspace{1cm} v_{r}(r) \propto r^{-0.2}\ ,
        \hspace{0.5cm} {\rm and} \hspace{0.5cm} T(r) \propto r^{-0.74}

    Application is primarily for the solar wind.

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.formulary.collisions import helio
    >>> r_0 = [0.1, 0.1, 0.1] * u.au
    >>> r_n = [1.0, 1.0, 1.0] * u.au
    >>> n_1 = [300, 400, 500] * u.cm**-3
    >>> n_2 = [12, 18, 8] * u.cm**-3
    >>> v_1 = [450, 350, 400] * u.km / u.s
    >>> T_1 = [1.5 * 10**5, 2.1 * 10**5, 1.7 * 10**5] * u.K
    >>> T_2 = [2.5 * 10**6, 1.8 * 10**6, 2.8 * 10**6] * u.K
    >>> ions = ["p+", "He-4++"]
    >>> helio.temp_ratio(
    ...     r_0=r_0, r_n=r_n, n_1=n_1, n_2=n_2, v_1=v_1, T_1=T_1, T_2=T_2, ions=ions
    ... )
    [np.float64(2.78928645...), np.float64(1.04007...), np.float64(1.06914...)]
    """

    # Validate ions argument
    if not isinstance(ions, list | tuple | ParticleList):
        ions = [ions]
    ions = ParticleList(ions)

    # Validate number of ions
    if len(ions) != 2:
        raise ValueError(
            "Argument 'ions' can only take two (2) input values. "
            f"Instead received {len(ions)} input values."
        )

    if not ions.is_category("ion"):
        raise ValueError(
            f"Particle(s) in 'ions' must be ions, received {ions=} "
            "instead. Please renter the 'ions' input parameter."
        )

    # Validate n_step argument
    if not isinstance(n_step, numbers.Integral):
        raise TypeError(
            "Argument 'n_step' is of incorrect type, type of "
            f"{type(n_step)} received instead. While 'n_step' must be "
            "of type int."
        )

    # Validate scaling arguments
    for arg in (density_scale, velocity_scale, temperature_scale):
        if not isinstance(arg, numbers.Real):
            raise TypeError(
                "Scaling argument is of incorrect type, type of "
                f"{type(arg)} received instead. Scaling argument "
                "should be of type float or int."
            )

    # Define differential equation function
    def df_eq(
        r_0,
        r_n,
        n_1_0,
        n_2,
        v_1_0,
        T_1_0,
        T_2,
        ions,
        n_step,
        density,
        velocity,
        temperature: float,
    ):
        # Initialize the alpha-proton charge and mass ratios.
        z_1 = ions[0].charge_number
        mu_1 = ions[0].mass_number

        z_2 = ions[1].charge_number
        mu_2 = ions[1].mass_number

        # Initialise.
        d_r = (r_n - r_0) / n_step

        # Define constants
        A = 2.6 * 10**7 * (u.cm**3 * u.km * (u.K**1.5)) / (u.s * u.au)
        B = 1 / (u.cm * u.K) ** 1.5

        # Define Coulomb log for mixed ion collisions, see docstring
        def lambda_ba(
            theta: float,
            T_1,
            n_1,
            n_2,
            z_1,
            z_2,
            mu_1,
            mu_2,
        ):
            a = np.sqrt(T_1**3 / n_1)
            b = (z_1 * z_2 * (mu_1 + mu_2)) / (theta + mu_2 / mu_1)
            c = np.sqrt((n_2 * z_2**2 / n_1 * z_1**2) + theta)
            return 9 + np.log(B * a * b * c)

        theta = T_2 / T_1_0
        for i in range(n_step):
            r = r_0 + ((i + 1) * d_r)

            n_1 = n_1_0 * (r / r_n) ** density
            v_1 = v_1_0 * (r / r_n) ** velocity
            T_1 = T_1_0 * (r / r_n) ** temperature

            eta = n_2 / n_1

            alpha = n_1 / (v_1 * (T_1**1.5))

            beta = (
                np.sqrt(mu_1 * mu_2)
                * (z_1 * z_2) ** 2
                * (1 - theta)
                * (1 + eta * theta)
            ) / (np.sqrt((mu_2 / mu_1) + theta) ** 3)

            l_ba = lambda_ba(theta, T_1, n_1, n_2, z_1, z_2, mu_1, mu_2)

            d_theta = d_r * alpha * l_ba * A * beta

            theta = theta + d_theta

        return theta.value

    variables = [r_0, r_n, n_1, n_2, v_1, T_1, T_2]

    d_type = [bool(hasattr(var, "__len__")) for var in variables]

    var = all(d_type)

    if not var:
        return df_eq(
            r_0,
            r_n,
            n_1,
            n_2,
            v_1,
            T_1,
            T_2,
            ions,
            n_step,
            density_scale,
            velocity_scale,
            temperature_scale,
        )
    try:
        all(len(variables[0]) == len(z) for z in variables[1:])
        res = []
        for i in range(len(variables[0])):
            res.append(
                df_eq(
                    r_0[i],
                    r_n[i],
                    n_1[i],
                    n_2[i],
                    v_1[i],
                    T_1[i],
                    T_2[i],
                    ions,
                    n_step,
                    density_scale,
                    velocity_scale,
                    temperature_scale,
                )
            )
            if verbose:
                logging.info(f"\r {(i / len(variables[0])) * 100:.2f} %")  # noqa: G004

        return res  # noqa: TRY300

    except Exception as e:
        raise ValueError(
            "Argument(s) are of unequal lengths, the following "
            "arguments should be of equal length: 'r_0', 'r_n', "
            "'n_1', 'n_2', 'v_1', 'T_1' and 'T_2'."
        ) from e
