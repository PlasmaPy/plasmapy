r"""This module implements helper functions from [1]_.

.. [1] Houlberg et al, Bootstrap current and neoclassical transport in tokamaks of arbitrary collisionality and aspect ratio, 1997,
   Physics of Plasmas 4, 3230 (1997); , JGR, 117, A12219, doi: `10.1063/1.872465
   <https://aip.scitation.org/doi/10.1063/1.872465>`_.

This work is referenced in docstrings as `Houlberg_1997`.
"""

from __future__ import annotations

import numpy as np

from astropy import constants
from astropy import units as u
from scipy.special import erf
from typing import Union, List, Iterable
from functools import cached_property
import warnings

from plasmapy.formulary import thermal_speed
from plasmapy.formulary.collisions import Coulomb_logarithm
from plasmapy.formulary.mathematics import Chandrasekhar_G
from plasmapy.particles import IonizationState, IonizationStateCollection
from plasmapy.plasma.fluxsurface import FluxSurface

try:
    from scipy.integrate import trapz as trapezoid
except ImportError:
    from scipy.integrate import trapezoid

__all__ = [
    "contributing_states",
    "mu_hat",
    "K",
    "K_ps_ai",
    "ν_T_ai",
    "ωm",
    "F_m",
    "_B17",
    "K_B_ai",
    "pitch_angle_diffusion_rate",
    "M_script",
    "N_script",
    "ξ",
    "effective_momentum_relaxation_rate",
    "N_matrix",
    "M_matrix",
    "xab_ratio",
    "ionizationstate_mass_densities",
]

from plasmapy.particles.particle_collections import ParticleList
from plasmapy.utils.decorators import (
    validate_quantities,
)

LaguerrePolynomials = [
    lambda x: np.ones_like(x),
    lambda x: 5 / 2 - x,
    lambda x: 35 / 8 - 7 / 2 * x + 1 / 2 * x ** 2,
]

class ExtendedParticleList(ParticleList):
    def __init__(self, particles: Iterable, 
                 T: u.eV,
                 n: u.m**-3,
                 dT: u.eV / u.m = None,
                 dn: u.m**-4 = None,
                 ):
        super().__init__(particles)
        T = u.Quantity(T).to(u.K, equivalencies = u.temperature_energy())
        self.n = u.Quantity(n)
        if T.size == 1:
            T = np.broadcast_to(T, self.n.shape) * T.unit
        self.T = T

        assert len(self.n) == len(particles)
        indices = np.argsort(self.basic_elements)
        is_sorted = (np.diff(indices) > 0).all()
        if not is_sorted:
            warnings.warn("Unsorted input array! Sorting.")
            self._data = np.array(self._data)[indices].tolist()
            self.n = self.n[indices]
            self.T = self.T[indices]

        if dT is not None:
            self.dT = u.Quantity(dT[indices])
            assert len(self.dT) == len(self.T)
        if dn is not None:
            self.dn = u.Quantity(dn[indices])
            assert len(self.dn) == len(self.n)
    @property
    def basic_elements(self):
        # TODO replace instances of element by isotope when that's fixed
        return [p.element if p.element is not None else p.symbol for p in self]

    @cached_property
    def mass_density(self):
        r"""Mass densities for each ionic level in an |IonizationState|.

        Parameters
        ----------
        a : IonizationState
            a

        Returns
        -------
        u.kg/u.m**3

        """
        return self.n * self.mass

    @cached_property
    def thermal_speed(self):
        return u.Quantity([thermal_speed(T, p) for T, p in zip(self.T, self)])

    @cached_property
    def ξ(self):
        # assert np.all(np.array(self.basic_elements) == np.sort(self.basic_elements))
        input_indices = np.unique(self.basic_elements,
                                  return_index=True,
                                  )[1]
        charge_numbers = np.split(self.charge_number, input_indices)
        ns = np.split(self.n, input_indices)
        outputs = []
        for charge_number, n in zip(charge_numbers, ns):
            array = charge_number**2 * n
            array /= array.sum()
            outputs.append(array)
        return np.concatenate(outputs)


    @cached_property
    def xab_ratio(self):
        return self.thermal_speed[:, np.newaxis] / self.thermal_speed[np.newaxis, :]

    @cached_property
    def mass_ratio(self) -> "(N, N)":
        return self.mass[:, ...] / self.mass[..., :]

    @cached_property
    def temperature_ratio(self) -> "(N, N)":
        return self.T[:, ...] / self.T[..., :] # TODO double check ordering of indices

    @cached_property
    def M_matrix(self) -> "(N, N, 3, 3)":
        xab = self.xab_ratio
        mass_ratio = self.mass_ratio
        M11 = -(1 + mass_ratio) / (1 + xab ** 2) ** (3 / 2)
        M12 = 3 / 2 * (1 + mass_ratio) / (1 + xab ** 2) ** (5 / 2)
        M21 = M12
        M22 = (13 / 4 + 4 * xab ** 2 + 15 / 2 * xab ** 4) / (1 + xab ** 2) ** (5 / 2)
        M13 = -15 / 8 * (1 + mass_ratio) / (1 + xab ** 2) ** (7 / 2)
        M31 = M13
        M23 = (69 / 16 + 6 * xab ** 2 + 63 / 4 * xab ** 4) / (1 + xab ** 2) ** (7 / 2)
        M32 = M23
        M33 = -(433 / 64 + 17 * xab ** 2 + 459 / 8 * xab ** 4 + 175 / 8 * xab ** 6) / (
            1 + xab ** 2
        ) ** (9 / 2)
        M = np.array([[M11, M12, M13], [M21, M22, M23], [M31, M32, M33]])
        return M

    @cached_property
    def N_matrix(self) -> "(N, N, 3, 3)":
        xab = self.xab_ratio
        temperature_ratio = self.temperature_ratio
        mass_ratio = self.mass_ratio
        N11 = (1 + mass_ratio) / (1 + xab ** 2) ** (3 / 2)
        N21 = -3 / 2 * (1 + mass_ratio) / (1 + xab ** 2) ** (5 / 2)
        N31 = 15 / 8 * (1 + mass_ratio) / (1 + xab ** 2) ** (7 / 2)
        M12 = 3 / 2 * (1 + mass_ratio) / (1 + xab ** 2) ** (5 / 2)
        N12 = -(xab ** 2) * M12
        N22 = 27 / 4 * (temperature_ratio) ** (1 / 2) * xab ** 2 / (1 + xab ** 2) ** (5 / 2)
        M13 = -15 / 8 * (1 + mass_ratio) / (1 + xab ** 2) ** (7 / 2)
        N13 = -(xab ** 4) * M13
        N23 = -225 / 16 * temperature_ratio * xab ** 4 / (1 + xab ** 2) ** (7 / 2)
        N32 = N23 / temperature_ratio
        N33 = (
            2625 / 64 * temperature_ratio ** (1 / 2) * xab ** 4 / (1 + xab ** 2) ** (9 / 2)
        )
        N = np.array([[N11, N12, N13], [N21, N22, N23], [N31, N32, N33]])
        return N

    @cached_property
    def effective_momentum_relaxation_rate(self):
        """Equations A3, A4 from |Houlberg_1997|

        """
        # TODO could be refactored using numpy
        collision_frequency = 1 / (3 * np.pi**(3/2)) *\
            (self.charge[:, np.newaxis] * self.charge[np.newaxis, :] / self.mass[:, np.newaxis] / constants.eps0)**2 *\
            (self.n[:, np.newaxis] * self.n[np.newaxis, :]) * self.mass[:, np.newaxis] / self.thermal_speed**3

        CL = lambda ai, bj, bn, bT: Coulomb_logarithm(
            bT,
            bn,
            (ai, bj),  # simplifying assumption after A4
        )
        CL_matrix = u.Quantity([[CL(ai, bj, bn, bT) for bj, bn, bT in zip(self, self.n, self.T)] for ai in self])
        # TODO double check ordering
        return (CL_matrix * collision_frequency).sum(axis=-1).si


    @cached_property
    def N_script(self):
        """Weighted field particle matrix - equation A2b from |Houlberg_1997|

        Parameters
        ----------
        a : IonizationState
        b : IonizationState
        """
        N = self.N_matrix
        # Equation A2b
        N_script = self.effective_momentum_relaxation_rate * N
        return N_script


    @cached_property
    def M_script(self):
        """Weighted test particle matrix - equation A2a from |Houlberg_1997|

        Parameters
        ----------
        a : IonizationState
        all_species : IonizationStateCollection
        """
        # Equation A2a
        integrand = (self.M_matrix * self.effective_momentum_relaxation_rate)
        return integrand.sum(axis=-1)

    def x_over_xab(self, x):
        xab = self.xab_ratio
        x_over_xab = (x[:, np.newaxis, np.newaxis] / xab[np.newaxis, ...]).value
        return x_over_xab

    def pitch_angle_diffusion_rate(
        self,
        x: np.ndarray,
    ):
        """The pitch angle diffusion rate, equation B4b from |Houlberg_1997|

        Parameters
        ----------
        x : np.ndarray
            distribution velocity relative to the thermal velocity.
        a : IonizationState
        all_species : IonizationStateCollection
        """
        # Houlberg_1997, equation B4b,
        xi = self.ξ
        denominator = x ** 3

        x_over_xab = self.x_over_xab(x)
        numerator = erf(x_over_xab) - Chandrasekhar_G(x_over_xab)
        fraction = numerator / denominator[:, np.newaxis, np.newaxis]
        sum_items = np.sum(fraction * self.effective_momentum_relaxation_rate, axis=-1)

        result = (
            xi
            / self.mass_density
            * (3 * np.sqrt(np.pi) / 4)
            * sum_items
        )
        return result


    def K_B_ai(
        self,
        x: np.ndarray,
        trapped_fraction: float,
        *,
        orbit_squeezing: bool = False,
    ):
        """Banana regime contribution to effective viscosity - eq. B1 from |Houlberg_1997|

        Parameters
        ----------
        x : np.ndarray
            x
        a_states : IonizationState
            a_states
        all_species : IonizationStateCollection
            all_species
        flux_surface : FluxSurface
            flux_surface
        orbit_squeezing : bool (default `False`)
            orbit_squeezing
        """
        # eq. B1-B4, Houlberg_1997
        f_t = trapped_fraction
        f_c = 1 - f_t
        if orbit_squeezing:
            raise NotImplementedError(
                "TODO allow for non-zero, changing radial electric fields (orbit squeezing)"
            )
        else:
            S_ai = 1  # Equation B2
        padr = self.pitch_angle_diffusion_rate(x)
        return padr * f_t / f_c / S_ai ** 1.5

    def ν_T_ai(self, x: np.ndarray):
        """Characteristic rate of anisotropy relaxation.

        Equation B12 from |Houlberg_1997|.

        Parameters
        ----------
        x : np.ndarray
            distribution velocity relative to the thermal velocity.
        a : IonizationState
            a
        all_species : IonizationStateCollection
            all_species
        """
        prefactor = 3 * np.pi ** 0.5 / 4 * self.ξ / self.mass_density

        # TODO check      if b is not a:  # TODO is not should work
        x_over_xab = self.x_over_xab(x)
        x = x.reshape(-1, 1, 1) 
        part1 = (erf(x_over_xab) - 3 * Chandrasekhar_G(x_over_xab)) / x ** 3
        part2 = 4 * (
            self.temperature_ratio + self.xab_ratio ** -2
        )
        part2full = part2 * Chandrasekhar_G(x_over_xab) / x
        summation = (part1 + part2full) * self.effective_momentum_relaxation_rate

        result = prefactor * summation.sum(axis=-1)
        return result

    def K_ps_ai(
        self,
        x: np.ndarray,
        flux_surface: FluxSurface,
        *,
        m_max=100,
    ):
        """Pfirsch-Schlüter regime contribution to effective viscosity - eq. B8 from |Houlberg_1997|

        Parameters
        ----------
        x : np.ndarray
            x
        a : IonizationState
            a
        all_species : IonizationStateCollection
            all_species
        flux_surface : FluxSurface
            flux_surface
        m_max :
            m_max
        """
        ν = self.ν_T_ai(x)[:, np.newaxis, :]

        m = np.arange(1, m_max + 1)
        F = F_m(m[:, np.newaxis], flux_surface)
        ω = ωm(x.reshape(-1, 1, 1) , m[:, np.newaxis], self.thermal_speed, flux_surface.gamma)[np.newaxis, ...]
        B10 = (
            1.5 * (ν / ω) ** 2
            - 9 / 2 * (ν / ω) ** 4
            + (1 / 4 + (3 / 2 + 9 / 4 * (ν / ω) ** 2) * (ν / ω) ** 2)
            * (2 * ν / ω)
            * np.arctan(ω / ν).si.value
        )
        onepart = F[:, np.newaxis] * B10
        full_sum = np.sum(onepart / ν, axis=1)

        return (
            3
            / 2
            * self.thermal_speed[np.newaxis, :]
            ** 2
            * x[:, np.newaxis] ** 2
            * full_sum
            / u.m ** 2   #TODO why the units here?
        )


    def K(
        x: np.ndarray,
        a: IonizationState,
        all_species: IonizationStateCollection,
        flux_surface: FluxSurface,
        *,
        m_max: int = 100,
        orbit_squeezing: bool = False,
    ):
        """Total effective velocity-dependent viscosity with contributions from -
        eq. 16 from |Houlberg_1997|

        Notes
        -----
        This expression originally comes from
        K. C. Shaing, C. T. Hsu, M. Yokoyama, and M. Wakatani, Phys. Plasmas
        2, 349 (1995); and K. C. Shaing, M. Yokoyama, M. Wakatani, and C. T. Hsu,
        ibid. 3, 965 (1996).

        Parameters
        ----------
        x : np.ndarray
            distribution velocity relative to the thermal velocity.
        a : IonizationState
        all_species : IonizationStateCollection
        flux_surface : FluxSurface
        m_max : int
        orbit_squeezing : bool
            orbit_squeezing
        """
        # Eq 16
        kb = K_B_ai(x, a, all_species, flux_surface, orbit_squeezing=orbit_squeezing)
        # print(f"got {kb=}")
        kps = K_ps_ai(x, a, all_species, flux_surface, m_max=m_max)
        # print(f"got {kps=}")
        return 1 / (1 / kb + 1 / kps)

    def mu_hat(
        a: IonizationState,
        all_species: IonizationStateCollection,
        flux_surface: FluxSurface,
        *,
        xmin: float = 0.0015,
        xmax: float = 10,
        N: int = 1000,
        **kwargs,
    ):
        """Viscosity coefficients - equation 15 from |Houlberg_1997|.

        Parameters
        ----------
        a : IonizationState
        all_species : IonizationStateCollection
        flux_surface : FluxSurface
        xmin :
            xmin
        xmax :
            xmax
        N :
            N
        kwargs :
            kwargs
        """
        if N is None:
            N = 1000
        orders = np.arange(1, 4)
        π = np.pi
        x = np.logspace(np.log10(xmin), np.log10(xmax), N)

        α = orders
        β = orders
        len_a = len(a)
        signs = (-1) * (α[:, None] + β[None, :])
        laguerres = np.vstack([LaguerrePolynomials[o - 1](x ** 2) for o in orders])
        kterm = K(x, a, all_species, flux_surface, **kwargs)
        kterm = kterm.reshape(len_a, N, 1, 1)
        xterm = (x ** 4 * np.exp(-(x ** 2))).reshape(1, N, 1, 1)
        y = laguerres.reshape(1, N, 3, 1) * laguerres.reshape(1, N, 1, 3) * kterm * xterm
        integral = trapezoid(y, x, axis=1)
        mu_hat_ai = integral * signs[None, ...]
        actual_units = (8 / 3 / np.sqrt(π)) * mu_hat_ai * a.mass_density[:, None, None]
        return actual_units


    def contributing_states(a: IonizationState):
        """Helper iterator over |IonizationState|s.

        Return a generator of tuples (`ξ(a)[i], a[i]`).

        Parameters
        ----------
        a : IonizationState

        """
        xi = ξ(a)
        for xii, ai in zip(xi, a):
            yield xii, ai




def _B17(flux_surface):
    """Equation B17 from |Houlberg_1997|. Likely bugged!

    Notes
    -----
    Eventually this should allow picking the right `m` in `K` below.

    Parameters
    ----------
    flux_surface :
        flux_surface
    """
    fs = flux_surface
    B20 = fs.Brvals * fs.Bprimervals + fs.Bzvals * fs.Bprimezvals
    under_average_B17 = (B20 / fs.Bmag) ** 2
    return fs.flux_surface_average(under_average_B17) / fs.flux_surface_average(fs.B2)


def F_m(m: Union[int, np.ndarray], fs: FluxSurface):
    """Mode weights for the Pfirsch-Schlüter contribution.

    Equation B9 from |Houlberg_1997|.

    Parameters
    ----------
    m : Union[int, np.ndarray]
        m
    flux_surface : FluxSurface
        flux_surface
    """
    B20 = fs.Brvals * fs.Bprimervals + fs.Bzvals * fs.Bprimezvals
    under_average_B16 = np.sin(m * fs.Theta) * B20
    under_average_B15 = under_average_B16 / fs.Bmag
    under_average_B16_cos = np.cos(m * fs.Theta) * B20
    under_average_B15_cos = under_average_B16_cos / fs.Bmag
    B15 = fs.flux_surface_average(under_average_B15)
    B16 = fs.gamma * fs.flux_surface_average(under_average_B16)
    B15_cos = fs.flux_surface_average(under_average_B15_cos)
    B16_cos = fs.gamma * fs.flux_surface_average(under_average_B16_cos)

    B2mean = fs.flux_surface_average(fs.B2)

    F_m = 2 / B2mean / fs.BDotNablaThetaFSA * (B15 * B16 + B15_cos * B16_cos)
    return F_m


def ωm(x: np.ndarray, m: Union[int, np.ndarray], thermal_speed: u.m/u.s, gamma):
    """Equation B11 from |Houlberg_1997|.

    Parameters
    ----------
    x : np.ndarray
        distribution velocity relative to the thermal velocity.
    m : Union[int, np.ndarray]
    a : IonizationState
    fs : FluxSurface
    """
    """Equation B11 of Houlberg_1997."""
    B11 = (
        x * thermal_speed * m * gamma / u.m
    )  # TODO why the u.m?
    return B11
