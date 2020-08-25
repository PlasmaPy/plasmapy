import astropy.constants as const
import astropy.units as u
import numpy as np

from scipy.special import exp1
from typing import List, Tuple, Union

from plasmapy.formulary.parameters import plasma_frequency
from plasmapy.particles import Particle
from plasmapy.utils.decorators import validate_quantities
from plasmapy.utils.exceptions import PhysicsError


@validate_quantities(
    frequencies={"can_be_negative": False},
    ne={"can_be_negative": False},
    ni={"can_be_negative": False},
    Te={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def thermal_bremsstrahlung(
    frequencies: u.Hz,
    ne: u.m ** -3,
    Te: u.K,
    ni: u.m ** -3 = None,
    ion_species: Union[str, Particle] = "H+",
    kmax: u.m = None,
) -> np.ndarray:
    r"""
    Calculate the Bremsstrahlung emission spectrum for a Maxwellian plasma
    in the Rayleigh-Jeans limit :math:`\hbar\omega \ll k_B*T_e`

    .. math::
       \frac{dP}{d\omega} = \frac{8 \sqrt{2}}{3\sqrt{\pi}}
       \bigg ( \frac{e^2}{4 \pi \epsilon_0} \bigg )^3
       \bigg ( m_e c^2 \bigg )^{-\frac{3}{2}}
       \bigg ( 1 - \frac{\omega_{pe}^2}{\omega^2} \bigg )^\frac{1}{2}
       \frac{Z_i^2 n_i n_e}{\sqrt(k_B T_e)}
       E_1(y)

    where E_1 is the exponential integral

    .. math::
        E_1 (y) = - \int_{-y}^\infty \frac{e^{-t}}{t}dt

    and y is the dimensionless argument

    .. math::
        y = \frac{1}{2} \frac{\omega^2 m_e}{k_{max}^2 k_B T_e}

    where   :math:`k_{max}` is a maximum wavenumber approximated here as
    :math:`k_{max} = 1/\lambda_B` where  :math:`\lambda_B` is the electron
    de Broglie wavelength.

    Parameters
    ----------

    frequencies : `~astropy.units.Quantity`
        Array of frequencies over which the bremsstrahlung spectrum will be
        calculated (convertable to Hz).

    ne : `~astropy.units.Quantity`
        Electron number density in the plasma (convertable to m^-3)

    Te : `~astropy.units.Quantity`
        Temperature of the electrons (in K or convertible to eV)

    ni : `~astropy.units.Quantity` (optional)
        Ion number density in the plasma (convertable to m^-3). Defaults
        to the quasi-neutral conditon ni=ne/Z.

    ion_species : str or `~plasmapy.particles.Particle`, optional
        An instance of `~plasmapy.particles.Particle`, or a string
        convertible to `~plasmapy.particles.Particle`. Default is `'H+'`
        corresponding to hydrogen ions.

    kmax :  `~astropy.units.Quantity`
        Cutoff wavenumber (convertable to u.rad/u.m). Defaults to the inverse
        of the electron de Broglie wavelength.

    Returns
    -------

    spectrum : `~astropy.units.Quantity`
        Computed bremsstrahlung spectrum over the frequencies provided.

    Notes
    -----

    For details, see "Radiation Processes in Plasmas" by
    Bekefi. `ISBN 978\\-0471063506`_.

    .. _`ISBN 978\\-0471063506`: https://ui.adsabs.harvard.edu/abs/1966rpp..book.....B/abstract
    """

    # Condition ion_species
    if isinstance(ion_species, str):
        ion_species = Particle(ion_species)

    # Default ni is ne/Z:
    if ni is None:
        ni = ne / ion_species.integer_charge

    # Default value of kmax is the electrom thermal de Broglie wavelength
    if kmax is None:
        kmax = (np.sqrt(const.m_e.si * const.k_B.si * Te) / const.hbar.si).to(1 / u.m)

    # Convert frequencies to angular frequencies
    w = (frequencies * 2 * np.pi * u.rad).to(u.rad / u.s)

    # Calculate the electron plasma frequency
    wpe = plasma_frequency(n=ne, particle="e-")

    # Check that all w < wpe (this formula is only valid in this limit)
    if np.min(w) < wpe:
        raise PhysicsError(
            "Lowest frequency must be larger than the electron"
            + "plasma frequency {:.1e}".format(wpe)
            + ", but min(w) = {:.1e}".format(np.min(w))
        )

    # Check that the parameters given fall within the Rayleigh-Jeans limit
    # hw << kTe
    rj_const = (np.max(w) * const.hbar.si / (2 * np.pi * u.rad * const.k_B.si * Te)).to(
        u.dimensionless_unscaled
    )
    if rj_const.value > 0.1:

        raise PhysicsError(
            "Rayleigh-Jeans limit not satisfied:"
            + "hbar*w/kTe = {:.2e} > 0.1".format(rj_const.value)
            + ". Try lower w or higher Te."
        )

    # Calculate the bremsstralung power spectral density in several steps
    c1 = (
        (8 / 3)
        * np.sqrt(2 / np.pi)
        * (const.e.si ** 2 / (4 * np.pi * const.eps0.si)) ** 3
        * 1
        / (const.m_e.si * const.c.si ** 2) ** 1.5
    )

    Zi = ion_species.integer_charge
    c2 = np.sqrt(1 - wpe ** 2 / w ** 2) * Zi ** 2 * ni * ne / np.sqrt(const.k_B.si * Te)

    # Dimensionless argument for exponential integral
    arg = 0.5 * w ** 2 * const.m_e.si / (kmax ** 2 * const.k_B.si * Te) / u.rad ** 2
    # Remove units, get ndarray of values
    arg = (arg.to(u.dimensionless_unscaled)).value

    return c1 * c2 * exp1(arg)
