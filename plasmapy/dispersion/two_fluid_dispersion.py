__all__ = ["two_fluid_dispersion_solution", "tfds_"]

import astropy.units as u
import numpy as np

from astropy.constants.si import c
from typing import Union
from warnings import warn

from plasmapy.formulary import parameters as pfp
from plasmapy.particles import Particle
from plasmapy.particles.exceptions import ChargeError
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsWarning


@validate_quantities(
    B={"can_be_negative": False},
    n_i={"can_be_negative": False},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def two_fluid_dispersion_solution(
    *,
    B: u.T,
    ion: Union[str, Particle],
    k: u.rad / u.m,
    n_i: u.m ** -3,
    T_e: u.K,
    T_i: u.K,
    gamma_e: Union[float, int] = 1,
    gamma_i: Union[float, int] = 3,
    theta: u.deg = 45 * u.deg,
    z_mean: Union[float, int] = None,
):
    r"""
    Using the solution provided by Bellan 2012, calculate the analytical
    solution to the two fluid, low-frequency (:math:`\omega/kc \ll 1`) dispersion
    relation presented by Stringer 1963.  This dispersion relation also
    assummes a uniform magnetic field :math:`\mathbf{B_o}`, no D.C. electric
    field :math:`\mathbf{E_o}=0`, and quasi-neutrality.  For more information
    see the **Notes** section below.

    **Aliases:** `tfds_`

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        The magnetic field magnitude in units convertible to :math:`T`.
    ion : `str` or `~plasmapy.particles.particle_class.Particle`
        Representation of the ion species (e.g., ``'p'`` for protons, ``'D+'``
        for deuterium, ``'He-4 +1'`` for singly ionized helium-4, etc.). If no
        charge state information is provided, then the ions are assumed to be
        singly ionized.
    k : `~astropy.units.Quantity`
        Wavenumber in units convertible to :math:`rad / m`.  Either single
        valued or 1-D array of length :math:`N`.
    n_i : `~astropy.units.Quantity`
        Ion number density in units convertible to :math:`m^{-3}`.
    T_e : `~astropy.units.Quantity`
        The electron temperature in units of :math:`K` or :math:`eV`.
    T_i : `~astropy.units.Quantity`
        The ion temperature in units of :math:`K` or :math:`eV`.
    gamma_e : `float` or `int`, optional
        The adiabatic index for electrons, which defaults to 1.  This
        value assumes that the electrons are able to equalize their
        temperature rapidly enough that the electrons are effectively
        isothermal.
    gamma_i : `float` or `int`, optional
        The adiabatic index for ions, which defaults to 3. This value
        assumes that ion motion has only one degree of freedom, namely
        along magnetic field lines.
    theta : `~astropy.units.Quantity`
        The angle of propagation of the wave with respect to the magnetic field,
        :math:`\cos^{-1}(k_z / k)`, in units must be convertible to :math:`deg`.
        Either single valued or 1-D array of size :math:`M`. (Default
        ``45 * u.deg``)
    z_mean : `float` or int, optional
        The average ionization state (arithmetic mean) of the ``ion`` composing
        the plasma.  Will override any charge state defined by argument ``ion``.

    Returns
    -------
    omega : Dict[str, `~astropy.units.Quantity`]
        A dictionary of computed wave frequencies in units :math:`rad/s`.  The
        dictionary contains three keys: ``'fast_mode'`` for the fast mode,
        ``'alfven_mode'`` for the Alfvén mode, and ``'acoustic_mode'`` for the
        ion-acoustic mode.  The value for each key will be a :math:`N x M` array.

    Raises
    ------
    TypeError
        If applicable arguments are not instances of `~astropy.units.Quantity` or
        cannot be converted into one.

    TypeError
        If ``ion`` is not of type or convertible to `~plasmapy.particles.Particle`.

    TypeError
        If ``gamma_e``, ``gamma_i``, or``z_mean`` are not of type `int` or `float`.

    ~astropy.units.UnitConversionError
        If applicable arguments do not have units convertible to the expected
        units.

    ValueError
        If the ``B``, ``k``, ``n_i``, ``T_e``, or ``T_i`` is negative.

    ValueError
        If ``ion`` is not of category ion or element.

    ValueError
        If ``B``, ``n_i``, ``T_e``, or ``T_I`` are not single valued
        `astropy.units.Quantity` (i.e. an array).

    ValueError
        If ``k`` or ``theta`` are not single valued or a 1-D array.

    Warns
    -----
    : `~plasmapy.utils.exceptions.PhysicsWarning`
        When the computed wave frequencies violate the low-frequency
        (:math:`\omega/kc \ll 1`) assumption of the dispersion relation.

    Notes
    -----

    The complete dispersion equation presented by Springer 1963 [2]_ (equation 1
    of Bellan 2012 [1]_) is:

    .. math::
        \left( \cos^2 \theta - Q \frac{\omega^2}{k^2 {v_A}^2} \right) &
        \left[
            \left( \cos^2 \theta - \frac{\omega^2}{k^2 {c_s}^2} \right)
            - Q \frac{\omega^2}{k^2 {v_A}^2} \left(
                1 - \frac{\omega^2}{k^2 {c_s}^2}
            \right)
        \right] \\
            &= \left(1 - \frac{\omega^2}{k^2 {c_s}^2} \right)
              \frac{\omega^2}{{\omega_{ci}}^2} \cos^2 \theta

    where

    .. math::
        Q &= 1 + k^2 c^2/{\omega_{pe}}^2 \\
        \cos \theta &= \frac{k_z}{k} \\
        \mathbf{B_o} &= B_{o} \mathbf{\hat{z}}

    :math:`\omega` is the wave frequency, :math:`k` is the wavenumber, :math:`v_A`
    is the Alfvén velocity, :math:`c_s` is the sound speed, :math:`\omega_{ci}` is
    the ion gyrofrequency, and :math:`\omega_{pe}` is the electron plasma frequency.
    This relation does additionally assumme low-frequency waves
    :math:`\omega/kc \ll 1`, no D.C. electric field :math:`\mathbf{E_o}=0` and
    quasi-neutrality.

    Following section 5 of Bellan 2012 [1]_ the exact roots of the above dispersion
    equation can be derived and expressed as one analytical solution (equation 38
    of Bellan 2012 [1]_):

    .. math::
        \frac{\omega}{\omega_{ci}} = \sqrt{
            2 \Lambda \sqrt{-\frac{P}{3}} \cos\left(
                \frac{1}{3} \cos^{-1}\left(
                    \frac{3q}{2p} \sqrt{-\frac{3}{p}}
                \right)
                - \frac{2 \pi}{3}j
            \right)
            + \frac{\Lambda A}{3}
        }

    where :math:`j = 0` represents the fast mode, :math:`j = 1` represents the
    Alfvén mode, and :math:`j = 2` represents the Acoustic mode.  Additionally,

    .. math::
        p &= \frac{3B-A^2}{3} \; , \; q = \frac{9AB-2A^3-27C}{27} \\
        A &= \frac{Q + Q^2 \beta + Q \alpha + \alpha \Lambda}{Q^2} \;
            , \; B = \alpha \frac{1 + 2 Q \beta + \Lambda \beta}{Q^2} \;
            , \; C = \frac{\alpha^2 \beta}{Q^2} \\
        \alpha &= \cos^2 \theta \;
            , \; \beta = \left( \frac{c_s}{v_A}\right)^2 \;
            , \; \Lambda = \left( \frac{k v_{A}}{\omega_{ci}}\right)^2

    References
    ----------
    .. [1] PM bellan, Improved basis set for low frequency plasma waves, 2012,
       JGR, 117, A12219, doi: `10.1029/2012JA017856
       <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2012JA017856>`_.

    .. [2] TE Stringer, Low-frequency waves in an unbounded plasma, 1963, JNE,
       Part C, doi: `10.1088/0368-3281/5/2/304
       <https://doi.org/10.1088/0368-3281/5/2/304>`_

    Examples
    --------
    >>> from astropy import units as u
    >>> from plasmapy.dispersion import two_fluid_dispersion_solution as tfds
    >>> k = 0.01 * u.m ** -1
    >>> theta = 30 * u.deg
    >>> B = 8.3E-9 * u.T
    >>> n = 5.e6 * u.m ** -3
    >>> T_e = 1.6e6 * u.K
    >>> T_i = 4.e5 * u.K
    >>> z = 1
    >>> omega = tfds(B=B, k=k, n=n, T_e=T_e, T_i=T_i, theta=theta, z=z)
    >>> omega
    {'fast_mode': <Quantity [[1520.5794506]] rad / s>,
     'alfven_mode': <Quantity [[1261.75471561]] rad / s>,
     'acoustic_mode': <Quantity [[0.6881521]] rad / s>}

    >>> k_arr = np.linspace(10**-7, 10**-2, 10000) * u.m ** -1
    >>> theta = np.linspace(5, 85, 100) * u.deg
    >>> n = 5.e6 * u.m ** -3
    >>> B = 8.3E-9 * u.T
    >>> T_e = 1.6e6 * u.K
    >>> T_i = 4.e5 * u.K
    >>> z = 1
    >>> c = 3.e8 * u.m/u.s
    >>> c_s = pfp.ion_sound_speed(T_e=T_e, T_i=T_i, n_e=z * n, ion='p+')
    >>> v_A = pfp.Alfven_speed( B, n, ion='p+')
    >>> omega_ci = pfp.gyrofrequency(B=B, particle='p+', signed=False, Z=z)
    >>> omega = tfds(n=n, B=B, T_e=T_e, T_i=T_i, theta=theta, z=z, k=k_arr)
    >>> omega['fast_mode'][:,40]
    <Quantity [1.61176312e-02, 1.77335334e-01, 3.38688590e-01, ...,
               1.52030361e+03, 1.52045553e+03, 1.52060745e+03] rad / s>
    """

    # validate argument ion
    if not isinstance(ion, Particle):
        try:
            ion = Particle(ion)
        except TypeError:
            raise TypeError(
                f"For argument 'ion' expected type {Particle} but got {type(ion)}."
            )
    if not (ion.is_ion or ion.is_category("element")):
        raise ValueError(f"The particle passed for 'ion' must be an ion or element.")

    # validate z_mean
    if z_mean is None:
        try:
            z_mean = abs(ion.integer_charge)
        except ChargeError:
            z_mean = 1
    else:
        if not isinstance(z_mean, (int, np.integer, float, np.floating)):
            raise TypeError(
                f"Expected int or float for argument 'z_mean', but got {type(z_mean)}."
            )
        z_mean = abs(z_mean)

    # validate arguments
    for arg_name in ("B", "n_i", "T_e", "T_i"):
        val = locals()[arg_name].squeeze()
        if val.shape != ():
            raise ValueError(
                f"Argument '{arg_name}' must a single value and not an array of "
                f"shape {val.shape}."
            )
        locals()[arg_name] = val

    # validate arguments
    for arg_name in ("gamma_e", "gamma_i"):
        if not isinstance(locals()[arg_name], (int, np.integer, float, np.floating)):
            raise TypeError(
                f"Expected int or float for argument '{arg_name}', but got "
                f"{type(locals()[arg_name])}.")

    # validate argument k
    k = k.squeeze()
    if not (k.ndim == 0 or k.ndim == 1):
        raise ValueError(
            f"Argument 'k' needs to be a single valued or 1D array astropy Quantity,"
            f" got array of shape {k.shape}."
        )
    if np.any(k < 0):
        raise ValueError(f"Argument 'k' can not be a or have negative values.")

    # validate argument theta
    theta = theta.squeeze()
    theta = theta.to(u.radian)
    if not (theta.ndim == 0 or theta.ndim == 1):
        raise ValueError(
            f"Argument 'theta' needs to be a single valued or 1D array astropy "
            f"Quantity, got array of shape {k.shape}."
        )

    # Required derived parameters
    # Compute the ion sound speed using the function from
    # plasmapy.formulary.parameters
    c_s = pfp.ion_sound_speed(
        T_e=T_e, T_i=T_i, n_e=z * n, gamma_e=gamma_e, gamma_i=gamma_i, ion=ion
    )

    # Compute the ion Alfven speed using the function from
    # plasmapy.formulary.parameters
    v_A = pfp.Alfven_speed(B, n, ion=ion)

    # Compute the ion gyro frequency using the function from
    # plasmapy.formulary.parameters
    omega_ci = pfp.gyrofrequency(B=B, particle="p+", signed=False, Z=z)

    # Compute the electron plasma frequency using the function from
    # plasmapy.formulary.parameters
    omega_pe = pfp.plasma_frequency(n=n, particle="e-", z_mean=z)

    # Compute the dimensionless parameters corresponding to equation 32 of
    # Bellan2012JGR
    alpha = (np.cos(theta.to("rad")) ** 2).value
    beta = (c_s ** 2 / v_A ** 2).value

    # Create a meshgrid of direction of propagation and the wavenumber
    alphav, kv = np.meshgrid(alpha, k)

    Lambda = (kv ** 2 * v_A ** 2 / omega_ci ** 2).value

    # Compute the dimensionless parameters corresponding to equation 2 of
    # Bellan2012JGR
    Q = 1 + (kv ** 2 * c ** 2 / omega_pe ** 2).value

    # Compute the dimensionless parameters corresponding to equation 35 of
    # Bellan2012JGR
    A = (Q + Q ** 2 * beta + Q * alphav + alphav * Lambda) / Q ** 2
    B = alphav * (1 + 2 * Q * beta + Lambda * beta) / Q ** 2
    C = alphav ** 2 * beta / Q ** 2

    # Compute the dimensionless parameters corresponding to equation 36 of
    # Bellan2012JGR
    p = (3 * B - A ** 2) / 3
    q = (9 * A * B - 2 * A ** 3 - 27 * C) / 27

    # These correspond to different parts of equation 38 of
    # Bellan2012JGR
    R = 2 * Lambda * np.lib.scimath.sqrt(-p / 3)
    S = 3 * q / (2 * p) * np.lib.scimath.sqrt(-3 / p)
    T = Lambda * A / 3

    # List out the three wave modes for which this function gives the
    # frequencies
    keys = ["fast_mode", "alfven_mode", "acoustic_mode"]

    # Create a dictionary with the wave mode names as its keys
    omega = dict.fromkeys(keys)

    # Compute the value of  omega for each key and for each value of wavenumber
    # and direction of propagation
    for (ind, key) in zip(range(3), keys):
        # The solution corresponding to equation 38
        omega[key] = omega_ci * np.lib.scimath.sqrt(
            R * np.cos(1 / 3 * np.lib.scimath.arccos(S) - 2 * np.pi / 3 * ind) + T
        )

    return omega


tfds_ = two_fluid_dispersion_solution
""" Alias to :func:`two_fluid_dispersion_solution`. """
