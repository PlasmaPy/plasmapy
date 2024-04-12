"""
This module contains functionality for calculating the timescales
for a range of configurations.
"""

__all__ = ["Hellinger", "Hellinger_2009", "Hellinger_2010", "Hellinger_2016"]

import astropy.units
import astropy.units as u
import numpy as np

from astropy.constants.si import eps0
from math import pi as pi
from mpmath import hyper2d
from scipy.stats import gausshyper as gh

from plasmapy.formulary.collisions import coulomb
from plasmapy.particles import Particle, ParticleList
from plasmapy.utils.decorators import validate_quantities


class validate:

    # Validate n_i argument
    def n_i(
        n_i: u.m**-3,
    ):
        if not isinstance(n_i, astropy.units.Quantity):
            raise TypeError(
                "Argument 'n_i' must be an astropy.units.Quantity, "
                f"instead got type of {type(n_i)}."
            )
        n_i = n_i.squeeze()
        if n_i.ndim != 0:
            raise ValueError(
                "Argument 'n_i' must be single value and not an array of"
                f" shape {n_i.shape}."
            )
        elif not isinstance(n_i.value, (int, float)):
            raise TypeError(
                "Argument 'n_i' must be an integer or float, received "
                f"{n_i} of type {type(n_i)}."
            )
        elif not n_i.value > 0:
            raise ValueError(
                f"Argument 'n_i' must be an positive argument, received "
                f"{n_i} of type {type(n_i)}."
            )

        return n_i

    # Validate ions argument
    def ions(
        ions: (Particle, Particle),
    ):
        if not isinstance(ions, (list, tuple, ParticleList)):
            ions = [ions]
        ions = ParticleList(ions)

        if not all(
            failed := [ion.is_ion and abs(ion.charge_number) > 0 for ion in ions]
        ):
            raise ValueError(
                "Particle(s) passed to 'ions' must be a charged"
                " ion. The following particle(s) is(are) not allowed "
                f"{[ion for ion, fail in zip(ions, failed) if not fail]}"
            )
        if len(ions) != 2:
            raise ValueError(
                f"Argument 'ions' can only take 2 inputs, received {ions}"
                f"with {len(ions)} inputs. Please try again."
            )
        return ions

    # Validate any speed argument
    def speeds(
        speeds: u.m / u.s,
    ):

        if len(speeds) != 2:
            raise ValueError(
                "Argument 'speeds' can only take 2 inputs, received "
                f"{speeds} with {speeds.ndim} inputs."
            )
            for speed in speeds:
                if not isinstance(speed, astropy.units.Quantity):
                    raise TypeError(
                        f"Argument in 'speeds' {speed} is of incorrect"
                        f" type, received {type(speed)} but expected "
                        f"type of `astropy.units.Quantity`."
                    )
        return speeds

    def Coulomb(
        Coulomb,
    ):
        valid_str = [
            "classical",
            "ls",
            "ls_min_interp",
            "GMS-1",
            "ls_full_interp",
            "GMS-2",
            "ls_clamp_mininterp",
            "GMS-3",
            "hls_min_interp",
            "GMS-4",
            "hls_max_interp",
            "GMS-5",
            "hls_full_interp",
            "GMS-6",
        ]

        if Coulomb is None:
            return "classical"
        elif isinstance(Coulomb, str):
            if Coulomb not in valid_str:
                raise ValueError(
                    "Argument 'Coulomb' was received as a str, but the"
                    " specific method type provided is not supported. "
                    f"Received {Coulomb} and valid entries are {valid_str}."
                )
        elif isinstance(Coulomb, (float, int)):
            return Coulomb
        else:
            raise TypeError(
                "Argument, 'Coulomb' can either be a numerical value, "
                "float or int, or a str specifying the desired method."
                " Currently supported methods are listed in notes"
            )

    # Validate any temperature argument
    @validate_quantities(
        T={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    )
    def temp(
        T: u.K,
    ):
        if not isinstance(T, astropy.units.Quantity):
            raise TypeError(
                "Argument 'T' must be an 'astropy.units.Quantity' with "
                f"a unit of K. Instead got type of {type(T)}."
            )
        elif T.shape != ():
            raise ValueError(
                "Argument 'T' must be single value and not an array of"
                f" shape {T.shape}."
            )
        elif not isinstance(T.value, (int, float)):
            raise TypeError(
                f"Argument 'T' must be an integer or float, received {T} "
                f"with type of {type(T)}."
            )
        elif not T.value > 0:
            raise ValueError(
                f"Argument 'T' must be a positive argument, received "
                f"{T} of type {type(T)}."
            )
        return T


def compute_coulomb(
    Coulomb,
    T,
    n_i,
    ions,
):
    Coulomb = validate.Coulomb(Coulomb)
    if isinstance(Coulomb, str):
        return coulomb.Coulomb_logarithm(T, n_i, ions, method=Coulomb)
    elif isinstance(Coulomb, (float, int)):
        return Coulomb
    else:
        raise ValueError(
            "Argument 'Coulomb' is of incorrect type, received "
            f"{Coulomb} of type {type(Coulomb)}."
        )


def Hellinger(
    inputs,
    method,
):
    r"""


    Parameters
    ----------
    method: int or float
        An integer or float representing the year of the desired method.
        Current options are ``'2009'``, ``'2010'`` and ``'2016'``.
        Please see the note's section below for additional details.

    kwargs: dict
        Arguments required for the specified method, each method
        requires differing inputs. See notes section below for
        additional details.

    Returns
    -------


    Raises
    ------


    Notes
    -----

    +----------+------------------+
    |  Method  |     Function     |
    +----------+------------------+
    |   2009   | `Hellinger_2009` |
    |   2010   | `Hellinger_2010` |
    |   2016   | `Hellinger_2016` |
    +----------+------------------+

    species s on species t.

    Example
    -------

    """

    functions = {2009: Hellinger_2009, 2010: Hellinger_2010, 2016: Hellinger_2016}

    if not isinstance(method, (float, int)):
        raise TypeError(
            "Argument 'method' must be of type float or integer, "
            f"instead got type of {type(method)}."
        )
    elif method not in functions.keys():
        raise ValueError(
            f"Argument 'method' is not a valid entry, received {method} "
            f" and valid methods are {functions.keys()}. Please try again."
        )

    return functions[method](**inputs)


def Hellinger_2009(
    T: u.K,
    n_i: u.m**-3,
    ions: (Particle, Particle),
    par_speeds: u.m / u.s,
    Coulomb=None,
):
    r"""
    Compute the collisional timescale as presented by :cite:t:`hellinger:2009`.
    For more details please see the notes section.

    Parameters
    ----------
    T : `~astropy.units.Quantity`
        The scalar temperature magnitude in units convertible to K.

    n_i : `~astropy.units.Quantity`
        Ion number density in units convertible to m\ :sup:`-3`.  Must
        be single value and should be the ion of prime interest.

    ions :  a `list` of length 2 containing :term:`particle-like` objects
        A list of length 2 with an instance of the
        :term:`particle-like` object representing the ion species in
        each entry. (e.g., ``"p"`` for protons, ``"D+"`` for deuterium,
         `["p", ``D+``]).

    par_speeds : a `list` of length 2 containing :term:`particle-like` objects
        A list of length 2 with an `~astropy.units.Quantity` representing
        the PARALLEL velocity with units of  in each entry. (e.g [
        500 * u.m / u.s, 745 * u.m / u.s]).

    Coulomb : `str`, `int` or `float`
        Can either be a string specifying the desired method for the
        Coulomb logarithm, for options please the notes section below.
        Or can be a numerical value specified for use instead, if no
        value is provided then option will default to classical.

    Returns
    -------
    :math:`\tau` : `~astropy.units.Quantity`
        The collisional timescale in units of seconds.

    Raises
    ------
    `TypeError`
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    `ValueError`
        Number of particles in ``ions`` is not 2 or the input values
        are not valid particles

    `ValueError`
        If ``n_i`` or ``T`` is negative or not a single value.

    `TypeError`
        If ``n_i`` or ``T`` is not of type integer or float.

    `ValueError`
        Number of parallel speeds in``par_speeds`` is not 2.

    `TypeError`
        If the parallel speeds in ``par_speeds`` is not of type
        integer or float

    `TypeError`
        If the value for ``method`` is not of type string

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    Notes
    -----
    Compute the collisional timescale as presented by :cite:t:`hellinger:2009`.
    In a weakly collisional plasma with the dominance of small angle
    deflections by Coulomb collisions, the collision frequency is
    calculated. The collision frequency is of species :math:`\alpha`
    on :math:`\beta` and is given by:

    .. math::
        \nu_{\alpha \beta} = \frac{q_{\alpha}^{2}q_{\beta}^{2}n_{\beta}}
        {12\pi^{3/2}\epsilon_{0}^{2}m_{\alpha}m_{\beta}
        v_{\alpha \beta \parallel}^{3}}\ln{\Lambda_{\alpha \beta}}

    where

    .. math::
        \ln{\Lambda_{\alpha \beta}} \equiv
        \int^{b_{\rm min, \alpha \beta}}_{b_{\rm max, \alpha \beta}}
        \frac{db}{b} = \ln{\left (\frac{b_{\rm max, \alpha \beta}}{b_{\rm min, \alpha \beta}} \right)}

    and

    .. math::
        v_{\alpha \beta \parallel} = \sqrt{\frac{v_{\alpha \parallel}^{2} +
        v_{\beta \parallel}^{2}}{2}}

    such that :math:`q` is the charge of the respective species,
    :math:`n` is the ion density of the species of interest,
    :math:`m` is the mass of the respective species and
    :math:`v_{\parallel}` is the parallel speed of the respective
    species.

    The following methods are supported by the Coulomb Logarithm
    1. ``"classical"`` or ``"ls"``
    2. ``"ls_min_interp"`` or ``"GMS-1"``
    3. ``"ls_full_interp"`` or ``"GMS-2"``
    4. ``"ls_clamp_mininterp"`` or ``"GMS-3"``
    5. ``"hls_min_interp"`` or ``"GMS-4"``
    6. ``"hls_max_interp"`` or ``"GMS-5"``
    7. ``"hls_full_interp"`` or ``"GMS-6"``


    Example
    _______

    """
    # Validate arguments argument
    T = validate.temp(T)
    n_i = validate.n_i(n_i)
    ions = validate.ions(ions)
    par_speeds = validate.speeds(par_speeds)

    Coulomb = compute_coulomb(Coulomb, T, n_i, ions)

    v_par = np.sqrt((par_speeds[0].value ** 2 + par_speeds[1].value ** 2) / 2)

    a = (ions[0].charge.value ** 2) * (ions[1].charge.value ** 2) * n_i.value

    b = (
        (12 * (pi**1.5))
        * (ions[0].mass.value * ions[1].mass.value)
        * (eps0**2)
        * (v_par**3)
    )

    return ((a / b.value) * Coulomb) / u.s


def Hellinger_2010(
    T_par: u.K,
    T_perp: u.K,
    n_i: u.m**-3,
    ions: (Particle, Particle),
    par_speeds: u.m / u.s,
    Coulomb=None,
):
    r"""
    Compute the collisional timescale as presented by :cite:t:`hellinger:2010`.
    For more details please see the notes section.

    Parameters
    ----------
    T_par : `~astropy.units.Quantity`
        The parallel temperature magnitude in units convertible to K.

    T_perp : `~astropy.units.Quantity`
        The perpendicular temperature magnitude in units convertible to K.

    n_i : `~astropy.units.Quantity`
        Ion number density in units convertible to m\ :sup:`-3`.  Must
        be single value and should be the ion of prime interest.

    ions :  a `list` of length 2 containing :term:`particle-like` objects
        A list of length 2 with an instance of the
        :term:`particle-like` object representing the ion species in
        each entry. (e.g., ``"p"`` for protons, ``"D+"`` for deuterium,
         `["p", ``D+``]).

    par_speeds : a `list` of length 2 containing :term:`particle-like` objects
        A list of length 2 with an `~astropy.units.Quantity` representing
        the PARALLEL velocity with units of  in each entry. (e.g [
        500 * u.m / u.s, 745 * u.m / u.s]).

    Coulomb : `str`, `int` or `float`
        Can either be a string specifying the desired method for the
        Coulomb logarithm, for options please the notes section below.
        Or can be a numerical value specified for use instead, if no
        value is provided then option will default and compute the
        Coulomb logarithm as classical.

    Returns
    -------
    :math:`\tau` : `~astropy.units.Quantity`
        The collisional timescale in units of seconds.

    Raises
    ------
    `TypeError`
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    `ValueError`
        Number of particles in ``ions`` is not 2 or the input values
        are not valid particles

    `ValueError`
        If ``n_i``, ``T_par`` or ``T_perp`` is negative or not a
        single value.

    `TypeError`
        If ``n_i``, ``T_par`` or ``T_perp`` is not of type
        integer or float.

    `ValueError`
        Number of parallel speeds in``par_speeds`` is not 2.

    `TypeError`
        If the parallel speeds in ``par_speeds`` is not of type
        integer or float

    `TypeError`
        If the value for ``method`` is not of type string

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    Notes
    -----
    Compute the collisional timescale as presented by :cite:t:`hellinger:2010`.
    In a weakly collisional plasma with the dominance of small angle
    deflections by Coulomb collisions, the collision frequency is
    calculated. The collision frequency is of species :math:`\alpha`
    on :math:`\beta` and is given by:

    .. math::
        \nu_{\alpha \beta} = \frac{q_{\alpha}^{2}q_{\beta}^{2}n_{\beta}}
        {20\pi^{3/2}\epsilon_{0}^{2}m_{\alpha}m_{\beta}
        v_{\alpha \beta \parallel}^{3}}\ln{\Lambda_{\alpha \beta}}
        \,_2F_1 \left (\begin{matrix}
        2, 3/2 \\
        7/2
        \end{matrix}, \, \begin{matrix}
        1 - \frac{T_{\perp}}{T_{\parallel}}
        \end{matrix} \right)

    where

    .. math::
        \ln{\Lambda_{\alpha \beta}} \equiv
        \int^{b_{\rm min, \alpha \beta}}_{b_{\rm max, \alpha \beta}}
        \frac{db}{b} = \ln{\left (\frac{b_{\rm max, \alpha \beta}}{b_{\rm min, \alpha \beta}} \right)}

    and

    .. math::
        v_{\alpha \beta \parallel} = \sqrt{\frac{v_{\alpha \parallel}^{2} +
        v_{\beta \parallel}^{2}}{2}}

    such that :math:`q` is the charge of the respective species,
    :math:`n` is the ion density of the species of interest,
    :math:`m` is the mass of the respective species and
    :math:`v_{\parallel}` is the parallel speed of the respective
    species. Note :math:`\,_2F_1` is the standard Gauss hyper geometric
    function.

    The following methods are supported by the Coulomb Logarithm
    1. ``"classical"`` or ``"ls"``
    2. ``"ls_min_interp"`` or ``"GMS-1"``
    3. ``"ls_full_interp"`` or ``"GMS-2"``
    4. ``"ls_clamp_mininterp"`` or ``"GMS-3"``
    5. ``"hls_min_interp"`` or ``"GMS-4"``
    6. ``"hls_max_interp"`` or ``"GMS-5"``
    7. ``"hls_full_interp"`` or ``"GMS-6"``

    Example
    _______

    """

    # Validate arguments
    T_par = validate.temp(T_par)
    T_perp = validate.temp(T_perp)
    n_i = validate.n_i(n_i)
    ions = validate.ions(ions)
    par_speeds = validate.speeds(par_speeds)

    if T_par == 0:
        raise ValueError(
            "Argument 'T_par' must be a non zero value, received a "
            f"value of {T_par}. Please try again."
        )
    else:
        T = (2 * T_perp + T_par) / 3
        return (
            Hellinger_2009(T, n_i, ions, par_speeds, Coulomb)
            * 3
            / 5
            * gh(a=2, b=1.5, c=7 / 2, x=(1 - (T_perp / T_par)))
        )


def Hellinger_2016(
    T_par: (u.K, u.K),
    T_perp: (u.K, u.K),
    n_i: u.m**-3,
    ions: (Particle, Particle),
    par_speeds: (u.m / u.s, u.m / u.s),
    perp_speeds: (u.m / u.s, u.m / u.s),
    Coulomb=None,
):
    r"""
    Compute the collisional timescale as presented by :cite:t:`hellinger:2016`.
    For more details please see the notes section.

    Parameters
    ----------
    T_par : a `list` of length 2 containing `~astropy.units.Quantity`
        The parallel temperature magnitude in units convertible to K.

    T_perp : a `list` of length 2 containing `~astropy.units.Quantity`
        The perpendicular temperature magnitude in units convertible to K.

    n_i : `~astropy.units.Quantity`
        Ion number density in units convertible to m\ :sup:`-3`.  Must
        be single value and should be the ion of prime interest.

    ions :  a `list` of length 2 containing :term:`particle-like` objects
        A list of length 2 with an instance of the
        :term:`particle-like` object representing the ion species in
        each entry. (e.g., ``"p"`` for protons, ``"D+"`` for deuterium,
         `["p", ``D+``]).

    par_speeds : a `list` of length 2 containing :term:`particle-like` objects
        A list of length 2 with an `~astropy.units.Quantity` representing
        the PARALLEL velocity with units of  in each entry. (e.g [
        500 * u.m / u.s, 745 * u.m / u.s]).

    perp_speeds : a `list` of length 2 containing :term:`particle-like` objects
        A list of length 2 with an `~astropy.units.Quantity` representing
        the PERPENDICULAR velocity with units of  in each entry. (e.g [
        500 * u.m / u.s, 745 * u.m / u.s]).

    Coulomb : `str`
        A string specifying the desired method for the Coulomb
        logarithm, for options please the notes section below.

    Returns
    -------
    :math:`\tau` : `~astropy.units.Quantity`
        The collisional timescale in units of seconds.

    Raises
    ------
    `TypeError`
        If applicable arguments are not instances of
        `~astropy.units.Quantity` or cannot be converted into one.

    `ValueError`
        Number of particles in ``ions`` is not 2 or the input values
        are not valid particles

    `ValueError`
        If ``n_i``, ``T_par`` or ``T_perp`` is negative or not a
        single value.

    `TypeError`
        If ``n_i``, ``T_par`` or ``T_perp`` is not of type
        integer or float.

    `ValueError`
        Number of parallel speeds in``par_speeds`` or ``perp_speeds``
        is not 2.

    `TypeError`
        If the parallel speeds in ``par_speeds`` or ``perp_speeds``
        is not of type integer or float

    `TypeError`
        If the value for ``method`` is not of type string

    ~astropy.units.UnitTypeError
        If applicable arguments do not have units convertible to the
        expected units.

    Notes
    -----
    Compute the collisional timescale as presented by :cite:t:`hellinger:2016`.
    Assuming a homogeneous plasma consisting of species with
    bi-Maxwellian velocity distribution and a weakly collisional plasma
    with the dominance of small angle deflections by Coulomb collisions.
    The collision frequency is calculated of species :math:`\alpha`
    on :math:`\beta` and is given by:

    .. math::
        \nu_{\alpha \beta} = \frac{q_{\alpha}^{2}q_{\beta}^{2}n_{\alpha \beta}}
        {24\pi^{3/2}\epsilon_{0}^{2}m_{\alpha}m_{\beta}
        v_{\alpha \beta \parallel}^{3}}\ln{\Lambda_{\alpha \beta}}
        \,_2F_1 \left (\begin{matrix}
        2, 3/2 \\
        7/2
        \end{matrix}, \, \begin{matrix}
        1 - \frac{T_{\perp}}{T_{\parallel}}
        \end{matrix} \right)

    where

    .. math::
        \ln{\Lambda_{\alpha \beta}} \equiv
        \int^{b_{\rm min, \alpha \beta}}_{b_{\rm max, \alpha \beta}}
        \frac{db}{b} = \ln{\left (\frac{b_{\rm max, \alpha \beta}}{b_{\rm min, \alpha \beta}} \right)}

    and

    .. math::
        v_{\alpha \beta \parallel} = \sqrt{\frac{v_{\alpha \parallel}^{2} +
        v_{\beta \parallel}^{2}}{2}}

    such that :math:`q` is the charge of the respective species,
    :math:`n` is the ion density, :math:`m` is the mass of the
    respective species and :math:`v_{\parallel}` is the parallel speed
    of the respective species.

    Note :math:`\,_2F_1` is the standard
    Gauss hyper geometric function.

    The following methods are supported by the Coulomb Logarithm
    1. ``"classical"`` or ``"ls"``
    2. ``"ls_min_interp"`` or ``"GMS-1"``
    3. ``"ls_full_interp"`` or ``"GMS-2"``
    4. ``"ls_clamp_mininterp"`` or ``"GMS-3"``
    5. ``"hls_min_interp"`` or ``"GMS-4"``
    6. ``"hls_max_interp"`` or ``"GMS-5"``
    7. ``"hls_full_interp"`` or ``"GMS-6"``

    Example
    _______
    """
    # Validate arguments
    T_par = validate.temp(T_par)
    T_perp = validate.temp(T_perp)
    n_i = validate.n_i(n_i)
    ions = validate.ions(ions)
    par_speeds = validate.speeds(par_speeds)
    perp_speeds = validate.speeds(perp_speeds)

    # Check for divide by zero error with t_par
    if T_par == 0:
        raise ValueError(
            "Argument 'T_par' must be a non zero value, received a "
            f"value of {T_par}. Please try again."
        )
    else:
        T = (2 * T_perp + T_par) / 3
        Coulomb = compute_coulomb(Coulomb, T, n_i, ions)
        vstpar = np.sqrt((par_speeds[0] ** 2 + par_speeds[1] ** 2) / 2)

        Ast = (ions.mass[0] * T_perp[1] + ions.mass[1] * T_perp[0]) / (
            ions.mass[0] * T_par[1] + ions.mass[1] * T_par[0]
        )

        vs = (par_speeds[0] ** 2 + 2 * perp_speeds[0] ** 2) / 3
        vt = (par_speeds[1] ** 2 + 2 * perp_speeds[1] ** 2) / 3

        vst = vs - vt

        return Hellinger_2009(T, n_i, ions, par_speeds, Coulomb) * hyper2d(
            1, 1.5, 2.5, 1 - Ast, Ast * (vst**2 / 4 * vstpar**2)
        )
