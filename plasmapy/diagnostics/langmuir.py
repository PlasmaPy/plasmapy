"""Defines the Langmuir analysis module as part of the diagnostics package."""
__all__ = [
    "Characteristic",
    "swept_probe_analysis",
    "get_plasma_potential",
    "get_floating_potential",
    "get_electron_saturation_current",
    "get_ion_saturation_current",
    "get_ion_density_LM",
    "get_electron_density_LM",
    "extract_exponential_section",
    "extract_ion_section",
    "get_electron_temperature",
    "extrapolate_electron_current",
    "reduce_bimaxwellian_temperature",
    "get_ion_density_OML",
    "extrapolate_ion_current_OML",
    "get_EEDF",
]

import astropy.units as u
import copy
import numpy as np

from astropy.constants import si as const
from astropy.visualization import quantity_support
from scipy.optimize import curve_fit
from warnings import warn

from plasmapy.particles import Particle
from plasmapy.utils.decorators import validate_quantities


def _langmuir_futurewarning() -> None:
    warn("The plasmapy.diagnostics.langmuir module will be deprecated in favor of "
         "the plasmapy.analysis.swept_langmuir sub-package.  The analysis package "
         " is in active development and is scheduled to be released with v0.5.0.",
         FutureWarning)


def _fit_func_lin(x, x0, y0, c0):
    r"""Linear fitting function."""

    return y0 + c0 * (x - x0)


def _fit_func_lin_inverse(x, x0, y0, T0):
    r"""Linear fitting function with inverse slope parameter for use in fitting
    of the electron current growth region.
    """

    return y0 + (x - x0) / T0


def _fit_func_double_lin_inverse(x, x0, y0, T0, Delta_T):
    r"""Piecewise linear fitting function with inverse slope parameters and
    an offset for use in fitting a bi-Maxwellian electron current growth
    region. (x0, y0) denotes the location of the knee of the transition,
    with T0 and T0 + Delta_T being the cold and hot temperatures, respectively.
    """

    hot_T_func = lambda x: y0 + (x - x0) / (T0 + Delta_T)
    cold_T_func = lambda x: y0 + (x - x0) / T0
    return np.piecewise(x, x < x0, [hot_T_func, cold_T_func])


class Characteristic:
    r"""Class representing a single I-V probe characteristic for convenient
    experimental data access and computation. Supports units.

    Attributes
    ----------
    bias : `astropy.units.Quantity`, ndarray
        Array of applied probe biases in units convertible to V.

    current : `astropy.units.Quantity`, ndarray
        Array of applied probe currents in units convertible to A.

    """

    @validate_quantities(bias={"can_be_inf": False}, current={"can_be_inf": False})
    def __init__(self, bias: u.V, current: u.A):
        _langmuir_futurewarning()

        self.bias = bias
        self.current = current
        self.get_unique_bias(True)
        self._check_validity()

    def __getitem__(self, key):
        r"""Allow array indexing operations directly on the Characteristic
        object.

        """

        b = Characteristic(self.bias[key], self.current[key])
        return b

    def __sub__(self, other):
        r"""Support current subtraction"""

        b = copy.deepcopy(self)
        b.current -= other.current
        return b

    def __add__(self, other):
        r"""Support current addition"""

        b = copy.deepcopy(self)
        b.current += other.current
        return b

    def sort(self):
        r"""Sort the characteristic by ascending bias."""

        _sort = self.bias.argsort()
        self.current = self.current[_sort]
        self.bias = self.bias[_sort]

    def get_unique_bias(self, inplace=False):
        r"""Remove any duplicate bias values through averaging."""

        if len(self.bias) != len(self.current):
            raise ValueError(
                f"Unequal array lengths of bias "
                f"({len(self.bias)}) and current "
                f"({len(self.current)})."
            )

        bias_unique = np.unique(self.bias)
        current_unique = []
        for bias in bias_unique:
            current_unique = np.append(
                current_unique, np.mean(self.current[self.bias == bias].to(u.A).value)
            )
        current_unique *= u.A

        if inplace:
            self.bias = bias_unique
            self.current = current_unique
        else:
            return Characteristic(bias_unique, current_unique)

    def _check_validity(self):
        r"""Check the unit and value validity of the characteristic."""

        if len(self.bias.shape) != 1:
            raise ValueError("Non-1D array for voltage!")

        if len(self.current.shape) != 1:
            raise ValueError("Non-1D array for current!")

        if len(self.bias) != len(self.current):
            raise ValueError(
                f"Unequal array lengths of bias "
                f"({len(self.bias)}) and current "
                f"({len(self.current)})."
            )

        if len(np.unique(self.bias)) != len(self.bias):
            raise ValueError(f"Bias array contains duplicate values.")

    def get_padded_limit(self, padding, log=False):  # coverage: ignore
        r"""Return the limits of the current range for plotting, taking into
        account padding. Matplotlib lacks this functionality.

        Parameters
        ----------
        padding : float
            The padding ratio as a float between 0.0 and 1.0.

        log : boolean, optional
            If True the calculation will be performed on a logarithmic scale.
            Default is False.

        """

        ymax = np.max(self.current).to(u.A).value

        if log:
            ymin = np.min(np.abs(self.current[self.current != 0])).to(u.A).value
            return [
                ymin * 10 ** (-padding * np.log10(ymax / ymin)),
                ymax * 10 ** (padding * np.log10(ymax / ymin)),
            ] * u.A
        else:
            ymin = np.min(self.current).to(u.A).value
            return [
                ymin - padding * (ymax - ymin),
                ymax + padding * (ymax - ymin),
            ] * u.A

    def plot(self):  # coverage: ignore
        r"""Plot the characteristic in matplotlib."""
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError) as e:
            from plasmapy.optional_deps import mpl_import_error

            raise mpl_import_error from e

        with quantity_support():
            plt.figure()
            plt.scatter(self.bias.to(u.V), self.current.to(u.mA), marker=".", color="k")
            plt.title("Probe characteristic")


@validate_quantities(
    probe_area={"can_be_negative": False, "can_be_inf": False, "can_be_nan": False}
)
def swept_probe_analysis(
    probe_characteristic,
    probe_area: u.m ** 2,
    gas_argument,
    bimaxwellian=False,
    visualize=False,
    plot_electron_fit=False,
    plot_EEDF=False,
):
    r"""Attempt to perform a basic swept probe analysis based on the provided
    characteristic and probe data. Suitable for single cylindrical probes in
    low-pressure DC plasmas, since OML is applied.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The swept probe characteristic that is to be analyzed.

    probe_area : ~astropy.units.Quantity
        The area of the probe exposed to plasma in units convertible to m^2.

    gas_argument : argument to instantiate the `Particle` class.
        `str`, `int`, or `~plasmapy.particles.Particle`
        A string representing a particle, element, isotope, or ion; an
        integer representing the atomic number of an element; or a
        `Particle` instance.

    visualize : bool, optional
        Can be used to plot the characteristic and the obtained parameters.
        Default is False.

    plot_electron_fit : bool, optional
        If True, the fit of the electron current in the exponential section is
        shown. Default is False.

    plot_EEDF : bool, optional
        If True, the EEDF is computed and shown. Default is False.

    Returns
    -------

    Results are returned as Dictionary

    "T_e" : `astropy.units.Quantity`
        Best estimate of the electron temperature in units of eV. Contains
        two values if bimaxwellian is True.

    "n_e" : `astropy.units.Quantity`
        Estimate of the electron density in units of m^-3. See the Notes on
        plasma densities.

    "n_i" : `astropy.units.Quantity`
        Estimate of the ion density in units of m^-3. See the Notes on
        plasma densities.

    "n_i_OML" : `astropy.units.Quantity`
        OML-theory estimate of the ion density in units of m^-3. See the Notes
        on plasma densities.

    "V_F" : `astropy.units.Quantity`
        Estimate of the floating potential in units of V.

    "V_P" : `astropy.units.Quantity`
        Estimate of the plasma potential in units of V.

    "I_es" : `astropy.units.Quantity`
        Estimate of the electron saturation current in units of Am^-2.

    "I_is" : `astropy.units.Quantity`
        Estimate of the ion saturation current in units of Am^-2.

    "hot_fraction" : float
        Estimate of the total hot (energetic) electron fraction.

    Notes
    -----
    This function combines the separate probe analysis functions into a single
    analysis. Results are returned as a Dictionary. On plasma densities: in an
    ideal quasi-neutral plasma all densities should be equal. However, in
    practice this will not be the case. The electron density is the poorest
    estimate due to the hard to obtain knee in the electron current. The
    density provided by OML theory is likely the best estimate as it is not
    dependent on the obtained electron temperature, given that the conditions
    for OML theory hold.

    """
    _langmuir_futurewarning()

    # Instantiate gas using the Particle class
    gas = Particle(argument=gas_argument)

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    # Obtain the plasma and floating potentials
    V_P = get_plasma_potential(probe_characteristic)
    V_F = get_floating_potential(probe_characteristic)

    # Obtain the electron and ion saturation currents
    I_es = get_electron_saturation_current(probe_characteristic)
    I_is = get_ion_saturation_current(probe_characteristic)

    # The OML method is used to obtain an ion density without knowing the
    # electron temperature. This can then be used to obtain the ion current
    # and subsequently a better electron current fit.
    n_i_OML, fit = get_ion_density_OML(
        probe_characteristic, probe_area, gas, return_fit=True
    )

    ion_current = extrapolate_ion_current_OML(probe_characteristic, fit)

    # First electron temperature iteration
    exponential_section = extract_exponential_section(
        probe_characteristic, ion_current=ion_current
    )
    T_e, hot_fraction = get_electron_temperature(
        exponential_section, bimaxwellian=bimaxwellian, return_hot_fraction=True
    )

    # Second electron temperature iteration, using an electron temperature-
    # adjusted exponential section
    exponential_section = extract_exponential_section(
        probe_characteristic, T_e=T_e, ion_current=ion_current
    )
    T_e, hot_fraction, fit = get_electron_temperature(
        exponential_section,
        bimaxwellian=bimaxwellian,
        visualize=plot_electron_fit,
        return_fit=True,
        return_hot_fraction=True,
    )

    # Extrapolate the fit of the exponential section to obtain the full
    # electron current. This has no use in the analysis except for
    # visualization.
    electron_current = extrapolate_electron_current(
        probe_characteristic, fit, bimaxwellian=bimaxwellian
    )

    # Using a good estimate of electron temperature, obtain the ion and
    # electron densities from the saturation currents.
    n_i = get_ion_density_LM(
        I_is, reduce_bimaxwellian_temperature(T_e, hot_fraction), probe_area, gas.mass
    )
    n_e = get_electron_density_LM(
        I_es, reduce_bimaxwellian_temperature(T_e, hot_fraction), probe_area
    )

    if visualize:  # coverage: ignore
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError) as e:
            from plasmapy.optional_deps import mpl_import_error

            raise mpl_import_error from e

        with quantity_support():
            fig, (ax1, ax2) = plt.subplots(2, 1)
            ax1.plot(
                probe_characteristic.bias,
                probe_characteristic.current,
                marker=".",
                color="k",
                linestyle="",
                label="Probe current",
            )
            ax1.set_title("Probe characteristic")
            ax2.set_ylim(probe_characteristic.get_padded_limit(0.1))

            ax2.plot(
                probe_characteristic.bias,
                np.abs(probe_characteristic.current),
                marker=".",
                color="k",
                linestyle="",
                label="Probe current",
            )
            ax2.set_title("Logarithmic")
            ax2.set_ylim(probe_characteristic.get_padded_limit(0.1, log=True))

            ax1.axvline(x=V_P.value, color="gray", linestyle="--")
            ax1.axhline(y=I_es.value, color="grey", linestyle="--")
            ax1.axvline(x=V_F.value, color="k", linestyle="--")
            ax1.axhline(y=I_is.value, color="r", linestyle="--")
            ax1.plot(ion_current.bias, ion_current.current, c="y", label="Ion current")
            ax1.plot(
                electron_current.bias,
                electron_current.current,
                c="c",
                label="Electron current",
            )
            tot_current = ion_current + electron_current
            ax1.plot(tot_current.bias, tot_current.current, c="g")

            ax2.axvline(x=V_P.value, color="gray", linestyle="--")
            ax2.axhline(y=I_es.value, color="grey", linestyle="--")
            ax2.axvline(x=V_F.value, color="k", linestyle="--")
            ax2.axhline(y=np.abs(I_is.value), color="r", linestyle="--")
            ax2.plot(
                ion_current.bias,
                np.abs(ion_current.current),
                label="Ion current",
                c="y",
            )
            ax2.plot(
                electron_current.bias,
                np.abs(electron_current.current),
                label="Electron current",
                c="c",
            )
            ax2.plot(tot_current.bias, np.abs(tot_current.current), c="g")
            ax2.set_yscale("log", nonposy="clip")
            ax1.legend(loc="best")
            ax2.legend(loc="best")

            fig.tight_layout()

    # Obtain and show the EEDF. This is only useful if the characteristic data
    # has been preprocessed to be sufficiently smooth and noiseless.
    if plot_EEDF:  # coverage: ignore
        get_EEDF(probe_characteristic, visualize=True)

    # Compile the results dictionary
    results = {
        "V_P": V_P,
        "V_F": V_F,
        "I_es": I_es,
        "I_is": I_is,
        "n_e": n_e,
        "n_i": n_i,
        "T_e": T_e,
        "n_i_OML": n_i_OML,
    }

    if bimaxwellian:
        results["hot_fraction"] = hot_fraction

    return results


def get_plasma_potential(probe_characteristic, return_arg=False):
    r"""Implement the simplest but crudest method for obtaining an estimate of
    the plasma potential from the probe characteristic.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    return_arg : bool, optional
        Controls whether or not the argument of the plasma potential within the
        characteristic array should be returned instead of the value of the
        voltage. Default is False.

    Returns
    -------
    V_P : ~astropy.units.Quantity
        Estimate of the plasma potential in units convertible to V.

    Notes
    -----
    The method used in the function takes the maximum gradient of the probe
    current as the 'knee' of the transition from exponential increase into the
    electron the saturation region.

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    # Sort the characteristic prior to differentiation
    probe_characteristic.sort()

    # Acquiring first derivative
    dIdV = np.gradient(
        probe_characteristic.current.to(u.A).value,
        probe_characteristic.bias.to(u.V).value,
    )

    arg_V_P = np.argmax(dIdV)

    if return_arg:
        return probe_characteristic.bias[arg_V_P], arg_V_P

    return probe_characteristic.bias[arg_V_P]


def get_floating_potential(probe_characteristic, return_arg=False):
    r"""Implement the simplest but crudest method for obtaining an estimate of
    the floating potential from the probe characteristic.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    return_arg : bool, optional
        Controls whether or not the argument of the floating potential within
        the characteristic array should be returned instead of the value of the
        voltage. Default is False.

    Returns
    -------
    V_F : ~astropy.units.Quantity
        Estimate of the floating potential in units convertible to V.

    Notes
    -----
    The method used in this function takes the probe current closest to zero
    Amperes as the floating potential.

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    arg_V_F = np.argmin(np.abs(probe_characteristic.current))

    if return_arg:
        return probe_characteristic.bias[arg_V_F], arg_V_F

    return probe_characteristic.bias[arg_V_F]


def get_electron_saturation_current(probe_characteristic):
    r"""Obtain an estimate of the electron saturation current corresponding
    to the obtained plasma potential.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    Returns
    -------
    I_es: ~astropy.units.Quantity
        Estimate of the electron saturation current in units convertible to A.

    Notes
    -----
    The function `get_plasma_potential` is used to obtain an estimate of the
    plasma potential. The corresponding electron saturation current is
    returned.

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    _, arg_V_P = get_plasma_potential(probe_characteristic, return_arg=True)

    return probe_characteristic.current[arg_V_P]


def get_ion_saturation_current(probe_characteristic):
    r"""Implement the simplest but crudest method for obtaining an estimate of
    the ion saturation current from the probe characteristic.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    Returns
    -------
    I_is : ~astropy.units.Quantity
        Estimate of the ion saturation current in units convertible to A.

    Notes
    -----
    The method implemented in this function assumes the ion saturation current
    to be the smallest probe current in the characteristic. This assumes the
    bias range in the ion region is sufficiently negative for the ion current to
    saturate.

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    return np.min(probe_characteristic.current)


@validate_quantities(
    ion_saturation_current={
        "can_be_negative": True,
        "can_be_inf": False,
        "can_be_nan": False,
    },
    T_e={
        "can_be_negative": False,
        "can_be_inf": False,
        "can_be_nan": False,
        "equivalencies": u.temperature_energy(),
    },
    probe_area={"can_be_negative": False, "can_be_inf": False, "can_be_nan": False},
    validations_on_return={"can_be_negative": False},
)
def get_ion_density_LM(
    ion_saturation_current: u.A, T_e: u.eV, probe_area: u.m ** 2, gas
) -> u.m ** -3:
    r"""Implement the Langmuir-Mottley (LM) method of obtaining the ion
    density.

    Parameters
    ----------
    ion_saturation_current : ~astropy.units.Quantity
        The ion saturation current in units convertible to A.

    T_e : ~astropy.units.Quantity
        The electron temperature in units convertible to eV.

    probe_area : ~astropy.units.Quantity
        The area of the probe exposed to plasma in units convertible to m^2.

    gas : ~astropy.units.Quantity
        The (mean) mass of the background gas in atomic mass units.

    Returns
    -------
    n_i : ~astropy.units.Quantity
        Estimate of the ion density in units convertible to m^-3.

    Notes
    -----
    The method implemented in this function obtains the ion density from the
    ion saturation current density assuming that the ion current loss to the
    probe is equal to the Bohm loss. The acoustic Bohm velocity is obtained
    from the electron temperature and the ion mass.

    The ion saturation current is given by

    .. math::
        I_{is} = 0.6 e A_p n_i \sqrt{\frac{T_e}{m_i}}.

    """

    _langmuir_futurewarning()

    # Calculate the acoustic (Bohm) velocity
    c_s = np.sqrt(T_e / gas)

    n_i = np.abs(ion_saturation_current) / (0.6 * const.e * probe_area * c_s)

    return n_i


@validate_quantities(
    electron_saturation_current={
        "can_be_negative": True,
        "can_be_inf": False,
        "can_be_nan": False,
    },
    T_e={
        "can_be_negative": False,
        "can_be_inf": False,
        "can_be_nan": False,
        "equivalencies": u.temperature_energy(),
    },
    probe_area={"can_be_negative": False, "can_be_inf": False, "can_be_nan": False},
    validations_on_return={"can_be_negative": False},
)
def get_electron_density_LM(
    electron_saturation_current: u.A, T_e: u.eV, probe_area: u.m ** 2
) -> u.m ** -3:
    r"""Implement the Langmuir-Mottley (LM) method of obtaining the electron
    density.

    Parameters
    ----------
    electron_saturation_current : ~astropy.units.Quantity
        The electron saturation current in units convertible to A.

    T_e : ~astropy.units.Quantity
        The electron temperature in units convertible to eV.

    probe_area : ~astropy.units.Quantity
        The area of the probe exposed to plasma in units convertible to m^2.

    Returns
    -------
    n_e : ~astropy.units.Quantity
        Estimate of the electron density in units convertible to m^-3.

    Notes
    -----
    The method implemented in this function obtains the electron density from
    the electron saturation current density, assuming a low plasma density. The
    electron saturation current is given by

    .. math::
        I_{es} = \frac{1}{4} e n_e A_p \sqrt{\frac{8 T_e}{\pi m_e}}.

    Please note that the electron saturation current density is a hard
    parameter to acquire and it is usually better to measure the ion density,
    which should be identical to the electron density in quasineutral plasmas.

    """

    _langmuir_futurewarning()

    # Calculate the thermal electron velocity
    v_th = np.sqrt(8 * T_e / (np.pi * const.m_e))

    n_e = 4 * electron_saturation_current / (probe_area * const.e * v_th)

    return n_e


def extract_exponential_section(probe_characteristic, T_e=None, ion_current=None):
    r"""Extract the section of exponential electron current growth from the
    probe characteristic.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    T_e : ~astropy.units.Quantity, optional
        If given, the electron temperature can improve the accuracy of the
        bounds of the exponential region.

    ion_current : ~plasmapy.diagnostics.langmuir.Characteristic, optional
        If given, the ion current will be subtracted from the probe
        characteristic to yield a better estimate of the electron current in
        the exponential region.

    Returns
    -------
    exponential_section : ~plasmapy.diagnostics.langmuir.Characteristic
        The exponential electron current growth section.

    Notes
    -----
    This function extracts the region of exponential electron growth from the
    probe characteristic under the assumption that this bias region is bounded
    by the floating and plasma potentials. Additionally, an improvement in
    accuracy can be made when the electron temperature is supplied.

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    V_F = get_floating_potential(probe_characteristic)

    V_P = get_plasma_potential(probe_characteristic)

    if T_e is not None:

        # If a bi-Maxwellian electron temperature is supplied grab the first
        # (cold) temperature
        if np.array(T_e).size > 1:
            T_e = np.min(T_e)

        _filter = (probe_characteristic.bias > V_F + 1.5 * T_e / const.e) & (
            probe_characteristic.bias < V_P - 0.2 * T_e / const.e
        )
    else:
        _filter = (probe_characteristic.bias > V_F) & (probe_characteristic.bias < V_P)

    exponential_section = probe_characteristic[_filter]

    if ion_current is not None:
        exponential_section = exponential_section - ion_current[_filter]

    return exponential_section


def extract_ion_section(probe_characteristic):
    r"""Extract the section dominated by ion collection from the probe
    characteristic.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    Returns
    -------
    ion_section : ~plasmapy.diagnostics.langmuir.Characteristic
        The exponential electron current growth section.

    Notes
    -----
    This function extracts the region dominated by ion collection from the
    probe characteristic under the assumption that this bias region is only
    bounded by the floating potential on the right hand side.

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    V_F = get_floating_potential(probe_characteristic)

    ion_section = probe_characteristic[probe_characteristic.bias < V_F]

    return ion_section


def get_electron_temperature(
    exponential_section,
    bimaxwellian=False,
    visualize=False,
    return_fit=False,
    return_hot_fraction=False,
):
    r"""Obtain the Maxwellian or bi-Maxwellian electron temperature using the
    exponential fit method.

    Parameters
    ----------
    exponential_section : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    bimaxwellian : bool, optional
        If True the exponential section will be fit assuming bi-Maxwellian
        electron populations, as opposed to Maxwellian. Default is False.

    visualize : bool, optional
        If True a plot of the exponential fit is shown. Default is False.

    return_fit: bool, optional
        If True the parameters of the fit will be returned in addition to the
        electron temperature. Default is False.

    return_hot_fraction: float, optional
        If True the total fraction of hot electrons will be returned if the
        population is bi-Maxwellian. Default is False.

    Returns
    -------
    T_e : ~astropy.units.Quantity, (ndarray)
        The estimated electron temperature in eV. In case of a bi-Maxwellian
        plasma an array containing two Quantities is returned.

    Notes
    -----
    In the electron growth region of the probe characteristic the electron
    current grows exponentially with bias voltage:

    .. math::
        I_e = I_{es} \textrm{exp} \left(
        -\frac{e\left(V_P - V \right)}{T_e} \right).

    In log space the current in this region should be a straight line if the
    plasma electrons are fully Maxwellian, or exhibit a knee in a
    bi-Maxwellian case. The slope is inversely proportional to the
    temperature of the respective electron population:

    .. math::
        \textrm{log} \left(I_e \right ) \propto \frac{1}{T_e}.

    """

    _langmuir_futurewarning()

    if not isinstance(exponential_section, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(exponential_section)}"
        )

    # Remove values in the section with a current equal to or smaller than
    # zero.
    exponential_section = exponential_section[
        exponential_section.current.to(u.A).value > 0
    ]

    initial_guess = None  # for fitting

    bounds = (-np.inf, np.inf)

    # Instantiate the correct fitting equation, initial values and bounds.
    if bimaxwellian:
        max_exp_bias = np.max(exponential_section.bias)
        min_exp_bias = np.min(exponential_section.bias)
        x0 = min_exp_bias + 2 / 3 * (max_exp_bias - max_exp_bias)

        initial_guess = [x0.to(u.V).value, 0.6, 2, 1]

        bounds = ([-np.inf, -np.inf, 0, 0], np.inf)

        fit_func = _fit_func_double_lin_inverse
    else:
        fit_func = _fit_func_lin_inverse

    # Perform the actual fit of the data
    fit, _ = curve_fit(
        fit_func,
        exponential_section.bias.to(u.V).value,
        np.log(exponential_section.current.to(u.A).value),
        p0=initial_guess,
        bounds=bounds,
    )

    hot_fraction = None

    # Obtain the plasma parameters from the fit
    if not bimaxwellian:
        slope = fit[2]

        T_e = slope ** -1 * u.eV
    else:
        x0, y0 = fit[0], fit[1]
        T0, Delta_T = [fit[2], fit[3]]

        # In order to obtain the energetic electron fraction the fits of the
        # cold and hot populations are extrapolated to the plasma potential
        # (ie. the maximum bias of the exponential section). The logarithmic
        # difference between these currents equates to the density difference.

        k1 = _fit_func_lin_inverse(
            np.max(exponential_section.bias.to(u.V).value), *[x0, y0, T0]
        )

        k2 = _fit_func_lin_inverse(
            np.max(exponential_section.bias.to(u.V).value), *[x0, y0, T0 + Delta_T]
        )

        # Compute the total hot (energetic) fraction
        hot_fraction = 1 / (1 + np.exp(k1 - k2))

        # If bi-Maxwellian, return main temperature first
        T_e = np.array([T0, T0 + Delta_T]) * u.eV

    if visualize:  # coverage: ignore
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError) as e:
            from plasmapy.optional_deps import mpl_import_error

            raise mpl_import_error from e

        with quantity_support():
            plt.figure()

            plt.scatter(
                exponential_section.bias.to(u.V),
                np.log(exponential_section.current.to(u.A).value),
                color="k",
                marker=".",
                label="Exponential section",
            )

            if bimaxwellian:
                plt.scatter(x0, y0, marker="o", c="g")
                plt.plot(
                    exponential_section.bias.to(u.V),
                    _fit_func_lin_inverse(
                        exponential_section.bias.to(u.V).value,
                        fit[0],
                        fit[1],
                        fit[2] + fit[3],
                    ),
                    c="g",
                    linestyle="--",
                    label="Bimaxwellian exponential section fit",
                )

            plt.plot(
                exponential_section.bias.to(u.V),
                fit_func(exponential_section.bias.to(u.V).value, *fit),
                label="Exponential fit",
                c="g",
            )

            plt.ylabel("Logarithmic current")
            plt.title("Exponential fit")
            plt.legend(loc="best")
            plt.tight_layout()

    k = [T_e]

    if return_hot_fraction:
        k.append(hot_fraction)

    if return_fit:
        k.append(fit)

    return k


def extrapolate_electron_current(
    probe_characteristic, fit, bimaxwellian=False, visualize=False
):
    r"""Extrapolate the electron current from the Maxwellian electron
    temperature obtained in the exponential growth region.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    fit : ndarray
        Polynomial fit coefficients returned by the electron temperature fit.

    bimaxwellian : bool, optional
        If True the electron current is extrapolated assuming bi-Maxwellian
        electron populations, as opposed to Maxwellian. Default is False.

    visualize : bool, optional
        If True a plot of the extracted electron current is shown. Default is
        False.

    Returns
    -------
    electron_current : ~plasmapy.diagnostics.langmuir.Characteristic
        The extrapolated electron current characteristic.

    Notes
    -----
    Assuming the electron population is fully Maxwellian the pure electron
    current is extrapolated from the fit of the exponential region for the
    entire bias range.

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    if bimaxwellian:
        fit_func = _fit_func_double_lin_inverse
    else:
        fit_func = _fit_func_lin_inverse

    electron_current = (
        np.exp(fit_func(probe_characteristic.bias.to(u.V).value, *fit)) * u.A
    )

    electron_current[electron_current > np.max(probe_characteristic.current)] = np.NaN

    electron_characteristic = Characteristic(
        probe_characteristic.bias, electron_current
    )

    if visualize:  # coverage: ignore
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError) as e:
            from plasmapy.optional_deps import mpl_import_error

            raise mpl_import_error from e

        with quantity_support():
            plt.figure()
            plt.scatter(
                probe_characteristic.bias,
                probe_characteristic.current.to(u.mA),
                marker=".",
                label="Probe characteristic",
                c="k",
            )
            plt.plot(
                electron_characteristic.bias,
                electron_characteristic.current.to(u.mA),
                label="Estimated electron characteristic",
            )
            plt.legend()

    return electron_characteristic


@validate_quantities(
    T_e={
        "can_be_negative": False,
        "can_be_inf": False,
        "can_be_nan": False,
        "equivalencies": u.temperature_energy(),
    },
    validations_on_return={"equivalencies": u.temperature_energy()},
)
def reduce_bimaxwellian_temperature(T_e: u.eV, hot_fraction: float) -> u.eV:
    r"""Reduce a bi-Maxwellian (dual) temperature to a single mean temperature
    for a given fraction.

    Parameters
    ----------
    T_e : ~astropy.units.Quantity, ndarray
        The bi-Maxwellian temperatures in eV. If a single temperature is
        provided, this is returned.

    hot_fraction : float
        Fraction of hot to total population. If this parameter is None the
        temperature is assumed to be singular Maxwellian.

    Returns
    -------
    T_e : ~astropy.units.Quantity
        The reduced (mean) temperature in units of eV.

    Notes
    -----
    This function aids methods that take a single electron temperature in
    situations where the electron population is bi-Maxwellian. The reduced
    temperature is obtained as the weighted mean:

    .. math::
        T_{e,red} = T_c \left( 1 - f_h \right) + T_h f_h

    """

    _langmuir_futurewarning()

    # Return the electron temperature itself if it is not bi-Maxwellian
    # in the first place.
    if hot_fraction is None or not np.array(T_e).size > 1:
        return T_e

    return T_e[0] * (1 - hot_fraction) + T_e[1] * hot_fraction


@validate_quantities(
    probe_area={"can_be_negative": False, "can_be_inf": False, "can_be_nan": False}
)
def get_ion_density_OML(
    probe_characteristic: Characteristic,
    probe_area: u.m ** 2,
    gas,
    visualize=False,
    return_fit=False,
):
    r"""Implement the Orbital Motion Limit (OML) method of obtaining an
    estimate of the ion density.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The swept probe characteristic that is to be analyzed.

    probe_area : ~astropy.units.Quantity
        The area of the probe exposed to plasma in units convertible to m^2.

    gas : ~astropy.units.Quantity
        The (mean) mass of the background gas in atomic mass units.

    visualize : bool, optional
        If True a plot of the OML fit is shown. Default is False.

    return_fit: bool, optional
        If True the parameters of the fit will be returned in addition to the
        ion density. Default is False.

    Returns
    -------
    n_i_OML : ~astropy.units.Quantity
        Estimated ion density in m^-3.

    Notes
    -----
    The method implemented in this function holds for cylindrical probes in a
    cold ion plasma, ie. :math:T_i=0` eV. With OML theory an expression is found
    for the ion current as function of probe bias independent of the electron
    temperature [mott-smith.langmuir-1926]_:

    .. math::
        I_i \xrightarrow[T_i = 0]{} A_p n_i e \frac{\sqrt{2}}{\pi}
        \sqrt{\frac{e \left( V_F - V \right)}{m_i}}

    References
    ----------
    .. [mott-smith.langmuir-1926] H. M. Mott-Smith, I. Langmuir,
        Phys. Rev. 28, 727-763 (Oct. 1926)

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    ion_section = extract_ion_section(probe_characteristic)

    fit = np.polyfit(
        ion_section.bias.to(u.V).value, ion_section.current.to(u.mA).value ** 2, 1
    )

    poly = np.poly1d(fit)

    slope = fit[0]

    ion = Particle(argument=gas)

    n_i_OML = np.sqrt(
        -slope
        * u.mA ** 2
        / u.V
        * np.pi ** 2
        * ion.mass
        / (probe_area ** 2 * const.e ** 3 * 2)
    )

    if visualize:  # coverage: ignore
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError) as e:
            from plasmapy.optional_deps import mpl_import_error

            raise mpl_import_error from e

        with quantity_support():
            plt.figure()
            plt.scatter(
                ion_section.bias.to(u.V),
                ion_section.current.to(u.mA) ** 2,
                color="k",
                marker=".",
            )
            plt.plot(
                ion_section.bias.to(u.V), poly(ion_section.bias.to(u.V).value), c="g"
            )
            plt.title("OML fit")
            plt.tight_layout()

    if return_fit:
        return n_i_OML.to(u.m ** -3), fit

    return n_i_OML.to(u.m ** -3)


def extrapolate_ion_current_OML(probe_characteristic, fit, visualize=False):
    r"""Extrapolate the ion current from the ion density obtained with the
    OML method.

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The probe characteristic that is being analyzed.

    fit : ndarray
        Fit coefficients returned by the OML method.

    visualize : bool, optional
        If True a plot of the extracted electron current is shown. Default is
        False.

    Returns
    -------
    ion_section : ~plasmapy.diagnostics.langmuir.Characteristic
        The exponential electron current growth section.

    Notes
    -----
    The exponential section of the probe characteristic should be a straight
    line if the plasma electrons are fully Maxwellian. The slope is then
    inversely proportional to the electron temperature.

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    slope = fit[0] * u.mA ** 2 / u.V
    offset = fit[1] * u.mA ** 2

    ion_current = -np.sqrt(
        np.clip(slope * probe_characteristic.bias + offset, 0.0, None)
    )

    ion_characteristic = Characteristic(probe_characteristic.bias, ion_current)

    if visualize:  # coverage: ignore
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError) as e:
            from plasmapy.optional_deps import mpl_import_error

            raise mpl_import_error from e

        with quantity_support():
            plt.figure()
            plt.scatter(
                probe_characteristic.bias,
                probe_characteristic.current.to(u.mA),
                marker=".",
                c="k",
            )
            plt.plot(
                probe_characteristic.bias, ion_characteristic.current.to(u.mA), c="y"
            )

    return ion_characteristic


def get_EEDF(probe_characteristic, visualize=False):
    r"""Implement the Druyvesteyn method of obtaining the normalized
    Electron Energy Distribution Function (EEDF).

    Parameters
    ----------
    probe_characteristic : ~plasmapy.diagnostics.langmuir.Characteristic
        The swept probe characteristic that is to be analyzed.

    visualize : bool, optional
        If True a plot of the extracted electron current is shown. Default is
        False.

    Returns
    -------
    energy : `astropy.units.Quantity`, ndarray
        Array of potentials in V.

    probability : float, ndarray
        Array of floats corresponding to the potentials representing the EEDF
        in normalized probabilities.

    Notes
    -----
    The Druyvesteyn method requires the second derivative of the probe
    I-V characteristic, which inherently amplifies noise and
    measurement errors. Therefore it is advisable to smooth the I-V
    prior to the use of this function.

    The Druyvesteyn analysis results in the following equation
    [druyvesteyn-1930]_:

    .. math::
        N_e \left( \epsilon \right) =
        \frac{2}{A_pe^2} \sqrt{\frac{2 m \epsilon}{e}}
        \frac{\textrm{d}^2 I}{\textrm{d} V^2}

    References
    ----------
    .. [druyvesteyn-1930] Druyvesteyn, M.J. Z. Physik (1930) 64: 781

    """

    _langmuir_futurewarning()

    if not isinstance(probe_characteristic, Characteristic):
        raise TypeError(
            f"For 'probe_characteristic' expected type "
            f"{Characteristic.__module__ + '.' + Characteristic.__qualname__} "
            f"and got {type(probe_characteristic)}"
        )

    probe_characteristic.sort()

    V_F = get_floating_potential(probe_characteristic)

    V_P = get_plasma_potential(probe_characteristic)

    probe_bias = probe_characteristic.bias
    probe_current = probe_characteristic.current

    # Obtain the correct EEDF energy range from the probe characteristic.
    _filter = (probe_bias > V_F) & (probe_bias < V_P)
    energy = const.e * (V_P - probe_bias[_filter])
    energy = energy.to(u.eV)

    # Obtain the second derivative of the I-V curve.
    dIdV = np.gradient(probe_current.to(u.A).value, probe_bias.to(u.V).value)
    dIdV2 = np.gradient(dIdV, probe_bias.to(u.V).value)

    # Division by the Druyvesteyn factor. Since the result will be normalized
    # all constant values are irrelevant.
    probability = dIdV2[_filter] * np.sqrt(energy.to(u.eV).value)

    # Integration of the EEDF for the purpose of normalization.
    integral = np.abs(np.trapz(probability, x=energy.to(u.eV).value))
    probability = probability / integral

    if visualize:  # coverage: ignore
        try:
            import matplotlib.pyplot as plt
        except (ImportError, ModuleNotFoundError) as e:
            from plasmapy.optional_deps import mpl_import_error

            raise mpl_import_error from e

        with quantity_support():
            plt.figure()
            plt.semilogy(energy, probability, c="k")
            plt.title("Electron Energy Distribution Function")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Probability")
            plt.grid()

    return energy, probability
