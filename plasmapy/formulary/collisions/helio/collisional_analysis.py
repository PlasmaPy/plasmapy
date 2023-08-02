"""
Module containing the Collisional Analysis formulation.
"""
__all__ = ["temp_ratio", "diff_flow"]

import astropy.units as u
import logging
import numbers
import numpy as np

from astropy import constants as const

from plasmapy.formulary.speeds import Alfven_speed
from plasmapy.particles import ParticleLike, ParticleList
from plasmapy.utils.decorators import validate_quantities

default_values = {
    "density": -1.8,
    "velocity": -0.2,
    "temperature": -0.74,
    "magnetic": -1.6,
}
m_u = const.u
mu_0 = const.mu0
e0 = const.eps0
k_B = const.k_B
q_e = const.e.si


# Define Coulomb log for mixed ion collisions, see docstring
def lambda_ba(
    theta,
    T_1,
    n_1,
    n_2,
    z_1,
    z_2,
    mu_1,
    mu_2,
):
    B = 1 / (u.cm * u.K) ** 1.5
    a = np.sqrt(T_1**3 / n_1)
    b = (z_1 * z_2 * (mu_1 + mu_2)) / (theta + mu_2 / mu_1)
    c = np.sqrt((n_2 * z_2**2 / n_1 * z_1**2) + theta)
    return 9 + np.log(B * a * b * c)


@validate_quantities(
    T_1={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_2={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def temp_ratio(  # noqa: C901
    *,
    r_0: u.au,
    r_n: u.au,
    n_1: u.cm**-3,
    n_2: u.cm**-3,
    v_1: u.km / u.s,
    T_1: u.K,
    T_2: u.K,
    ions: ParticleLike = ("p+", "He-4++"),
    n_step: int = 20,
    density_scale: float = default_values["density"],
    velocity_scale: float = default_values["velocity"],
    temperature_scale: float = default_values["temperature"],
    verbose=False,
):
    r"""
    Calculate the thermalization ratio for a plasma in transit, taken
    from :cite:t:`maruca:2013` and :cite:t:`johnson:2023a`. This
    function allows the thermalization of a plasma to be modeled,
    predicting the temperature ratio for different ion species
    within a plasma at a different point in space.

    Parameters
    ----------
    r_0 : `~astropy.units.Quantity`, |keyword-only|
        Starting position of the plasma in units convertible
        to astronomical units.

    r_n : `~astropy.units.Quantity`
        Final position of the plasma in units convertible
        to astronomical units.

    n_1 : `~astropy.units.Quantity`
        The primary ion number density in units convertible
        to m\ :sup:`-3`.

    n_2 : `~astropy.units.Quantity`
        The secondary ion number density in units convertible
        to m\ :sup:`-3`.

    v_1 : `~astropy.units.Quantity`
        The primary ion speed in units convertible to km s\ :sup:`-1`.

    T_1 : `~astropy.units.Quantity`
        Temperature of the primary ion in units convertible to
        temperature K.

    T_2 : `~astropy.units.Quantity`
        Temperature of the secondary ion in units convertible to
        temperature K.

    ions : |particle-list-like|, default: ``("p+, "He-4 2+")``
        Particle list containing two (2) particles, primary ion of
        interest is entered first, followed by the secondary ion.

    n_step : positive integer
        The number of intervals used in solving a differential
        equation via the Euler method.

    density_scale : real number, default: -1.8
        The value used as the scaling parameter for the primary ion
        density. The default value is taken from
        :cite:t:`hellinger:2011`.

    velocity_scale : `float`, default: -0.2
        The value used as the scaling parameter for the primary ion
        velocity. The default value is taken from
        :cite:t:`hellinger:2011`.

    temperature_scale : `float`, default: -0.74
        The value used as the scaling parameter for the primary ion
        temperature. The default value is taken from
        :cite:t:`hellinger:2011`.

    Returns
    -------
    theta : `float`
        The dimensionless ion temperature ratio prediction
        for the distance provided.

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

        \theta_{21} = \frac{T_{2}}{T_{1}} \, ,

    where :math:`T_{1}` and :math:`T_{2}` are the scalar temperatures
    for the primary ion of interest and the secondary ion, respectively.
    The scalar temperature defined as:

    .. math::

        T_{\rm i} = \frac{2T_{{\rm i}, \perp} + T_{{\rm i}, \parallel}}{3} \, ,

    where :math:`T_{{\rm i}, \perp}` and :math:`T_{{\rm i}, \parallel}`
    are the temperature of the :math:`{\rm i}`-particles along the
    axes perpendicular and parallel to the ambient magnetic field.

    In order to determine how extensively an individual parcel of
    plasma has been processed by Coulomb collisions :cite:p:`maruca:2013`
    introduced an approached called collisional analysis. This paper seeks
    to quantify how collisions affect the plasma's departures from
    LTE, the equation for collisional thermalization
    from :cite:t:`maruca:2013` is given below:

     .. math::

        \frac{d \theta_{21}}{dr} = A \left (
        \frac{n_1}{v_1 T_1^{3/2}} \right ) \frac{\left( \mu_{1} \mu_{2}
        \right )^{1/2} Z_{1} Z_{2} \left( 1 - \theta_{21} \right )
        \left(1 + \eta_{21}\theta_{21} \right )}{\left(
        \frac{\mu_{2}}{\mu_{1}} + \theta_{21} \right )^{3/2}} \lambda_{21}

    and

    .. math::

         \lambda_{21} = 9 + \ln \left| B \left ( \frac{T^{3}_{1}}{n_{1}}
         \right )^{1/2} \left( \frac{Z_{1}Z_{2}(\mu_{1} + \mu_{2}) }{\theta_{21} +
         \frac{\mu_{2}}{\mu_1}} \right )
         \left( \frac{n_{2}Z_{2}^{2}}{n_{1}Z_{1}^{2}} + \theta_{21}
         \right)^{1/2}\right |

    With :math:`\eta = \frac{n_{2}}{n_{1}}`, :math:`\theta =
    \frac{T_{2}}{T_{1}}`, :math:`A = 2.60 \times 10^{7} \, {\rm cm}^{3}
    \, {\rm km} \, {\rm K}^{3/2} \, {\rm s}^{-1} \, {\rm au}^{-1}`,
    and :math:`B = 1 \, {\rm cm}^{-3/2}{\rm K}^{-3/2}`.

    The thermalization is from Coulomb collisions, which assumes
    "soft", small-angle deflections mediated by the electrostatic
    force :cite:t:`baumjohann:1997`. It is assumed that there is no
    relative drift between the ion species and that it is a mixed ion
    collision, the Coulomb logarithm for a mixed ion collision is
    given by :cite:t:`nrlformulary:2019`.

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
    ...     )
    [2.8760..., 1.0271..., 1.0474...]
    """

    # Validate ions argument
    if not isinstance(ions, (list, tuple, ParticleList)):
        ions = [ions]
    ions = ParticleList(ions)

    # Validate number of ions
    if len(ions) != 2:
        raise ValueError(
            "Argument 'ions' can only take two (2) input values. "
            f"Instead received {len(ions)} input values."
        )

    if not all(ions.is_category("ion")):
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
        temperature,
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
                * (z_1**2 * z_2**2)
                * (1 - theta)
                * (1 + eta * theta)
            ) / (np.sqrt((mu_2 / mu_1) + theta) ** 3)
            l_ba = lambda_ba(theta, T_1, n_1, n_2, z_1, z_2, mu_1, mu_2)

            d_theta = d_r * alpha * l_ba * A * beta

            theta = theta + d_theta

        return theta.value

    variables = [r_0, r_n, n_1, n_2, v_1, T_1, T_2]

    d_type = [bool(hasattr(var, "__len__")) for var in variables]

    var = all(i for i in d_type)

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
    else:
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
                    logging.info(f"\r {(i / len(variables[0])) * 100:.2f} %")

            return res  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            raise ValueError(
                "Argument(s) are of unequal lengths, the following "
                "arguments should be of equal length: 'r_0', 'r_n', "
                "'n_1', 'n_2', 'v_1', 'T_1' and 'T_2'."
            ) from e


def diff_flow(  # noqa: C901, PLR0912, PLR0915
    *,
    r_0: u.au,
    r_n: u.au,
    n_1: u.cm**-3,
    n_2: u.cm**-3,
    v_1: u.km / u.s,
    v_2: u.km / u.s,
    T_1: u.K,
    T_2: u.K,
    B: u.T,
    ions: ParticleLike = ("p+", "He-4++"),
    density_scale: float = default_values["density"],
    velocity_scale: float = default_values["velocity"],
    temperature_scale: float = default_values["temperature"],
    magnetic_scale: float = default_values["magnetic"],
    alfven=False,
    n_step: int = 20,
    verbose=False,
):
    r"""
    Calculate the collisionality of the differential flow for a
    plasma in transit, taken from :cite:t:`johnson:2023b`. This
    function allows the thermalization of a plasma to be modeled,
    predicting the differential flow between two ion species within
    a plasma at a different point in space.

    Parameters
    ----------
    r_0 : `~astropy.units.Quantity`, |keyword-only|
        Starting position of the plasma in units convertible
        to astronomical units.

    r_n : `~astropy.units.Quantity`
        Final position of the plasma in units convertible
        to astronomical units.

    n_1 : `~astropy.units.Quantity`
        The primary ion number density in units convertible
        to m\ :sup:`-3`.

    n_2 : `~astropy.units.Quantity`
        The secondary ion number density in units convertible
        to m\ :sup:`-3`.

    v_1 : `~astropy.units.Quantity`
        The primary ion speed in units convertible to km s\ :sup:`-1`.

    v_2 : `~astropy.units.Quantity`
        The secondary ion speed in units convertible to km s\ :sup:`-1`.

    T_1 : `~astropy.units.Quantity`
        Temperature of the primary ion in units convertible to
        temperature K.

    T_2 : `~astropy.units.Quantity`
        Temperature of the secondary ion in units convertible to
        temperature K.

    B : `~astropy.units.Quantity`
        Magnetic field strength in units convertible to tesla T.

    ions : |particle-list-like|, default: ``("p+, "He-4 2+")``
        Particle list containing two (2) particles, primary ion of
        interest is entered first, followed by the secondary ion.

    density_scale : real number, default: -1.8
        The value used as the scaling parameter for the primary ion
        density. The default value is taken from
        :cite:t:`hellinger:2011`.

    velocity_scale : `float`, default: -0.2
        The value used as the scaling parameter for the primary ion
        velocity. The default value is taken from
        :cite:t:`hellinger:2011`.

    temperature_scale : `float`, default: -0.74
        The value used as the scaling parameter for the primary ion
        temperature. The default value is taken from
        :cite:t:`hellinger:2011`.

    magnetic_scale : `float`, default: -1.6
        The value used as the scaling parameter for the magnetic
        field. The default value is taken from
        :cite:t:`hellinger:2011`

    alfven : `bool`, default: False
        The analysis for differential flow can be performed on data
        and also be scaled by the Alfven speed. Setting this to true
        will scale the output, by default it is turned off.

    n_step : positive integer
        The number of intervals used in solving a differential
        equation via the Euler method.

    Returns
    -------
    Δ v_{12} : `float`
        The differential velocity prediction in units of
        km s\ :sup:`-1`, or dimensionless if Alfven scaling is
        enabled, for the distance provided.

    Raises
    ------
    `TypeError`
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    `ValueError`
        If applicable arguments are not of the same length or if an
        incorrect 'ions' argument is entered.

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    Notes
    -----
    Coulomb collisions act to bring a system into local thermal
    equilibirum (LTE) :cite:p:`verscharen:2019`, this affects how
    parameters, i.e. temperature, density and speed, evolve in
    transit. The collisional slowing time between two constituent
    plasma ion species is given as:

    .. math::

        \Delta \vec{v}_{ab} = \vec{v}_{a} - \vec{v}_{b} ,

    where :math:`\vec{v}_{a}` and :math:`\vec{v}_{b}` are the bulk
    velocities for the primary ion of interest and the secondary ion
    respectively. The rate of change in the differential flow due to collisions is
    taken from :cite:t:`nrlformulary:2019`:

    .. math::

        \frac{d \Delta \vec{v}_{ab}}{dt}  = \nu^{\rm s}_{(ab)} \Delta \vec{v}_{ab},

    where :math:`t` is time and  :math:`\nu^{\rm s}_{(ab)}` is the
    collision frequency for the slowing of the secondary (b) particles
    by primary (a) particles. The collision rate for the slowing down
    time is taken from :cite:t:`larroche:2021` and is shown below:

    .. math::

        \tau_{SD} = \frac{3m_{a}^{2}m_{b}^{2} \left( \frac{k_{B}T_{a}}{m_{a}} + \frac{k_{B}T_{b}}{m_{b}} \right)^{
        3/2}}{4 \sqrt{2\pi} e^{4} Z^{2}_{a}Z^{2}_{b}(m_{a} + m_{b})(n_{a}m_{a} + n_{b}m_{b}) \lambda_{ab} }

    where, for :math:`i`-particles, :math:`m_{i}` is mass,
    :math:`q_{i}` is charge, :math:`v_{i}` is bulk velocity,
    :math:`T_{i}` is scalar temperature, :math:`n_{i}` is number
    density and :math:`k_{\rm B}` is the Boltzmann constant. The
    Coulomb logarithm :math:`\lambda_{ab}` for  mixed ion-ion
    collisions is:

    .. math::

        \lambda_{ab} = 23 - \ln\!\left [ \frac{Z_{a}Z_{b}(\mu_{a} + \mu_{b})}{\mu_{a}T_{b} +
        \mu_{b}T_{a}} \left ( \frac{n_{a}Z_{a}^{2}}{T_{a}} + \frac{n_{b}Z_{b}^{2}}{T_{b}} \right )^{1/2} \right].

    The collisional timescale has a corresponding collisional
    frequency which can be used in the prior equation.

    .. math::

        \tau_{SD} = \frac{1}{\nu^{\rm s}_{(ab)}}

    Following the example in :cite:t:`maruca:2013`, the chain rule
    was applied to differential flow and the total derivative
    was converted into the convective derivative. This produces an
    equation in terms of :math:`r`, with  the velocity being the
    velocity of the background particle field.

    .. math::

        \frac{d \Delta \vec{v}_{ab}}{dr} = -\nu^{s}_{(ab)} \cdot \frac{\Delta \vec{v}_{ab}}{|\vec{v_{b}}|}

    Application is primarily concerned with the solar wind and the
    system is assumed to be in steady state. The density, velocity
    and temperature of the primary ion can be radially scaled,
    as seen below. The values for the scaling can be altered,
    though the default values are taken from :cite:t:`hellinger:2011`.

    .. math::
        n(r) \propto r^{-1.8}\ , \hspace{1cm} v_{r}(r) \propto r^{-0.2}\ , \hspace{1cm} T(r) \propto r^{-0.74}, \hspace{0.5cm} \, {\rm and} \hspace{0.5cm} B(r) \propto r^{-1.6}.


    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.formulary.collisions import helio
    >>> r_0 = [0.1, 0.1, 0.1] * u.au
    >>> r_n = [1.0, 1.0, 1.0] * u.au
    >>> n_1 = [100, 125, 150] * u.cm**-3
    >>> n_2 = [5, 15, 10] * u.cm**-3
    >>> v_1 = [300, 200, 250] * u.km/u.s
    >>> v_2 = [350, 400, 300] * u.km /u.s
    >>> T_1 = [1.5 * 10**5, 2.1 * 10**5, 1.7 * 10**5] * u.K
    >>> T_2 = [2.5 * 10**6, 1.8 * 10**6, 2.8 * 10**6] * u.K
    >>> B = [5, 5, 5 ] * u.T
    >>> ions = ["p+", "He-4++"]
    >>> helio.diff_flow(r_0=r_0, r_n=r_n, n_1=n_1, n_2=n_2, v_1=v_1, v_2=v_2, T_1=T_1, T_2=T_2, B=B, ions=ions)
     [<Quantity 43.071... km / s>, <Quantity 144.267... km / s>, <Quantity 39.766... km / s>]
    """

    # Validate ions argument
    if not isinstance(ions, (list, tuple, ParticleList)):
        ions = [ions]
    ions = ParticleList(ions)

    # Validate number of ions
    if len(ions) != 2:
        raise ValueError(
            "Argument 'ions' can only take two (2) input values. "
            f"Instead received {len(ions)} input values."
        )

    if not all(ions.is_category("ion")):
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
    for arg in (density_scale, velocity_scale, temperature_scale, magnetic_scale):
        if not isinstance(arg, numbers.Real):
            raise TypeError(
                "Scaling argument is of incorrect type, type of "
                f"{type(arg)} received instead. Scaling argument "
                "should be of type float or int."
            )

    # Validate booleans
    for arg in (alfven, verbose):
        if not isinstance(arg, bool):
            raise TypeError(
                f"Boolean input is of incorrect type {arg} is of "
                f"{type(arg)}, bool type is required."
            )

    # Define the differential equation
    def df_eq(
        r_0,
        r_n,
        n_1_0,
        n_2_0,
        T_1_0,
        T_2_0,
        v_1_0,
        v_2_0,
        B_0,
        ions,
        density_scale,
        velocity_scale,
        temperature_scale,
        magnetic_scale,
        alfven,
        n_step,
    ):
        # Initialize the alpha-proton charge and mass ratios.
        z_1 = ions[0].charge_number
        mu_1 = ions[0].mass_number

        z_2 = ions[1].charge_number
        mu_2 = ions[1].mass_number

        # Initialise.
        d_r = (r_n - r_0) / (1.0 * n_step)
        dv = abs(v_2_0 - v_1_0)

        for i in range(n_step):
            r = r_0 + ((i + 1) * d_r)

            n_1 = n_1_0 * (r / r_n) ** density_scale
            v_1 = v_1_0 * (r / r_n) ** velocity_scale
            T_1 = T_1_0 * (r / r_n) ** temperature_scale

            n_2 = n_2_0
            T_2 = T_2_0

            if alfven:
                B = B_0 * (r / r_n) ** magnetic_scale
                v_a = Alfven_speed(B, n_1 * mu_1 + n_2 * mu_2, "p") / 1000

            a = (3 * (mu_1 * mu_2) ** 2 * (m_u**4) * (8 * np.pi * e0) ** 2) / (
                4 * np.sqrt(2 * np.pi) * (q_e**4) * ((z_1 * z_2) ** 2)
            )
            b = (((k_B * T_1) / (mu_1 * m_u)) + ((k_B * T_2) / (mu_2 * m_u))) ** 1.5
            c = (m_u**2) * (mu_1 + mu_2) * (n_1 * mu_1 + n_2 * mu_2)
            d = lambda_ba(T_2 / T_1, T_1, n_1, n_2, z_1, z_2, mu_1, mu_2)

            vs = (c * d) / (a * b)

            d_dv = -(vs * (dv) / v_1) * d_r

            dv = dv + d_dv

        if alfven:
            return dv / v_a
        else:
            return dv

    variables = [r_0, r_n, n_1, n_2, v_1, T_1, T_2, B]

    d_type = [bool(hasattr(var, "__len__")) for var in variables]

    var = all(i for i in d_type)

    if not var:
        return df_eq(
            r_0,
            r_n,
            n_1,
            n_2,
            T_1,
            T_2,
            v_1,
            v_2,
            B,
            ions,
            density_scale,
            velocity_scale,
            temperature_scale,
            magnetic_scale,
            alfven,
            n_step,
        )
    else:
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
                        T_1[i],
                        T_2[i],
                        v_1[i],
                        v_2[i],
                        B[i],
                        ions,
                        density_scale,
                        velocity_scale,
                        temperature_scale,
                        magnetic_scale,
                        alfven,
                        n_step,
                    )
                )
                if verbose:
                    logging.info(f"\r {(i / len(variables[0])) * 100:.2f} %")

            return res  # noqa: TRY300

        except Exception as e:  # noqa: BLE001
            raise ValueError(
                "Argument(s) are of unequal lengths, the following "
                "arguments should be of equal length: 'r_0', 'r_n', "
                "'n_1', 'n_2', 'v_1', 'v_2', 'T_1', 'T_2'. and 'B'."
            ) from e
