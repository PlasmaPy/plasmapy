from __future__ import annotations

__all__ = [
    "FlowCalculator",
]


import numpy as np
import typing
import xarray

from astropy import constants
from astropy import units as u
from collections import defaultdict, namedtuple

from plasmapy.particles import IonizationStateCollection, Particle
from plasmapy.plasma.fluxsurface import FluxSurface

from .Houlberg1997 import contributing_states, M_script, mu_hat, N_script, ξ, ExtendedParticleList

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property
particle_flux_unit = u.m ** -2 / u.s
heat_flux_unit = u.J * particle_flux_unit
from plasmapy.particles.exceptions import InvalidElementError
from plasmapy.utils.decorators import validate_quantities

Fluxes = namedtuple("Fluxes", ["particle_flux", "heat_flux"])


class FlowCalculator:
    r"""Interface to neoclassical transport calculations.

    Reimplements NCLASS from |Houlberg_1997|[1]_

    References
    ----------
    .. [1] Houlberg et al, Bootstrap current and neoclassical transport in tokamaks of arbitrary collisionality and aspect ratio, 1997,
       Physics of Plasmas 4, 3230 (1997); , JGR, 117, A12219, doi: `10.1063/1.872465
       <https://aip.scitation.org/doi/10.1063/1.872465>`_.

    Parameters
    ----------
    all_species : IonizationStateCollection
        all_species
    flux_surface : FluxSurface
        flux_surface
    density_gradient : dict
        density_gradient
    temperature_gradient : dict
        temperature_gradient
    mu_N : int
        mu_N
    dataset_input : xarray.Dataset
        dataset_input
    """

    @classmethod
    def from_xarray_surface(cls, dataset: xarray.Dataset, flux_surface: FluxSurface):
        """
        Alternate constructor from an `xarray.Dataset` and an associated `~FluxSurface`.
        """

        arrays = {}
        lists = []
        data_vars = ["n", "gradn", "T", "gradT"]
        dataset['basic_elements'] = 'particle', [Particle(i).element if Particle(i).element is not None else Particle(i).symbol for i in dataset.particle.values]
        # TODO does not handle deuterium because H 1+.isotope = None, why?
        for basic_element, a in dataset.groupby("basic_elements"):
            lists.append(
                ExtendedParticleList(
                    [Particle(i) for i in a.particle.values],
                    u.Quantity(a.T, dataset.attrs["T unit"]),
                    u.Quantity(a.n, dataset.attrs["n unit"]),
                    u.Quantity(a.dT, dataset.attrs["dT unit"]),
                    u.Quantity(a.dn, dataset.attrs["dn unit"]),
                )
            )

        return cls(
            lists,
            flux_surface,
            dataset_input=dataset,
        )

    # profile
    def __init__(
        self,
        all_species: IonizationStateCollection,
        flux_surface: FluxSurface,
        *,
        mu_N: int = None,
        dataset_input: xarray.Dataset = None,
    ):
        self.all_species = all_species
        self._all_species_map = {pl[0].symbol: pl for pl in all_species}
        self.flux_surface = fs = flux_surface
        self.density_gradient = density_gradient
        self.temperature_gradient = {
            particle: (u.m * temperature_gradient[particle]).to(
                u.K, equivalencies=u.temperature_energy()
            )
            / u.m
            for particle in temperature_gradient
        }
        self._dataset_input = dataset_input

        self.M_script_matrices = {}
        self.N_script_matrices = {}

        self.S_pt = {}
        self.μ = {}
        self.Aai = {}

        r"""$S_{\theta,\beta}^{ai}"""
        self.thermodynamic_forces = {}

        self.pressure_gradient = {}
        self.ξ = {}

        r_flows_list = []
        r_sources_list = []
        rbar_flows_list = []
        rbar_sources_list = []
        self.r_pt = {}
        S_pt_list = []
        for a in self.all_species:
            sym = a[0].symbol
            charges = a.charge_number * constants.e.si
            n_charge_states = len(charges)
            xi = ξ(a)
            T_i = a.T.to(u.K, equivalencies = u.temperature_energy())
            n_i = a.n
            density_gradient = self.density_gradient.get(
                sym, np.zeros(n_charge_states) * (u.m ** -4)
            )
            temperature_gradient = self.temperature_gradient.get(
                sym, np.zeros(n_charge_states) * (u.K / u.m)
            )
            pressure_gradient_over_n_i = constants.k_B * (
                T_i * density_gradient / n_i + temperature_gradient
            )
            # we divide by n_i, which can be zero, leadning to inf, so to correct that...
            pressure_gradient_over_n_i[np.isinf(pressure_gradient_over_n_i)] = 0
            μ = mu_hat(a, self.all_species, self.flux_surface, N=mu_N)

            Aai = xi[:, np.newaxis, np.newaxis] * self.M_script(a)[np.newaxis, ...] - μ
            # --- TD forces eq21
            S_pt_θ = thermodynamic_forces = (
                fs.Fhat
                / charges
                * u.Quantity(
                    [
                        pressure_gradient_over_n_i,
                        constants.k_B * temperature_gradient,
                        np.zeros(n_charge_states) * (u.J / u.m),
                    ]
                )
            ).T
            CHARGE_STATE_AXIS = 0
            BETA_AXIS = 2
            S_pt = (thermodynamic_forces[:, np.newaxis, :] * μ).sum(
                axis=BETA_AXIS
            )  # Equation 29
            S_pt_list.append(S_pt)
            r_pt = np.linalg.solve(Aai, S_pt)
            # TODO r_E itd
            r_sources = r_pt + 0
            r_sources_list.append(r_sources)
            rbar_sources = (xi[:, np.newaxis] * r_sources).nansum(
                axis=CHARGE_STATE_AXIS
            )
            rbar_sources_list.append(rbar_sources)
            S_flows = (xi[:, np.newaxis, np.newaxis] * np.eye(3)) * u.Unit("N T / m3")
            r_flows = np.linalg.solve(Aai, S_flows)
            r_flows_list.append(r_flows)
            rbar_flows = (xi[:, np.newaxis, np.newaxis] * r_flows).nansum(
                axis=CHARGE_STATE_AXIS
            )
            rbar_flows_list.append(rbar_flows)
            for i, ai in enumerate(a):
                sym = ai.ionic_symbol
                self.density_gradient[sym] = density_gradient[i]
                self.temperature_gradient[sym] = temperature_gradient[i]
                self.r_pt[sym] = r_pt[i]
                self.S_pt[sym] = S_pt[i]
                self.thermodynamic_forces[sym] = thermodynamic_forces[i]
                self.μ[sym] = μ[i]

        lhs = u.Quantity(np.eye(3 * len(all_species)), "J2 / (A m6)")
        for i, a in enumerate(all_species):
            rarray = rbar_flows_list[i]
            for j, b in enumerate(self.all_species):
                narray = self.N_script(a, b).sum(axis=0, keepdims=True)
                result = narray * rarray.T
                lhs[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] += result
        rhs = u.Quantity(rbar_sources_list)
        ubar = np.linalg.solve(lhs, rhs.ravel())

        self._charge_state_flows = {}
        for r_flows, r_sources, a in zip(
            r_flows_list, r_sources_list, self.all_species
        ):

            def gen():
                for j, b in enumerate(self.all_species):
                    ubar_b = ubar[3 * j : 3 * j + 3]
                    yield (self.N_script(a, b) * ubar_b.reshape(1, -1)).sum(axis=1)

            Λ = -sum(gen())

            self_consistent_u = np.sum(Λ[np.newaxis, :, np.newaxis] * r_flows, axis=2)
            u_velocity = self_consistent_u + r_sources

            for i, ai in enumerate(a):
                if np.isfinite(u_velocity[i]).all():
                    self._charge_state_flows[ai.ionic_symbol] = u_velocity[i]

    def all_contributing_states_symbols(self) -> typing.Iterator[str]:
        """Helper iterator over all charge levels of all isotopes in the calculation."""
        for a in self.all_species:
            for _, ai in contributing_states(a):
                yield ai.ionic_symbol

    def M_script(self, a: IonizationState) -> np.ndarray:
        """Thin, cached wrapper on top of `~plasmapy.transport.Houlberg1997.M_script`."""
        sym = a[0].symbol
        if sym not in self.M_script_matrices:
            self.M_script_matrices[sym] = M_script(a, self.all_species)
        return self.M_script_matrices[sym]

    def N_script(self, a: IonizationState, b: IonizationState) -> np.ndarray:
        """Thin, cached wrapper on top of `~plasmapy.transport.Houlberg1997.N_script`."""
        sym_tuple = a[0].symbol, b[0].symbol
        if sym_tuple not in self.N_script_matrices:
            self.N_script_matrices[sym_tuple] = N_script(a, b)
        return self.N_script_matrices[sym_tuple]

    # profile
    def _funnymatrix(self, a_symbol):
        a = self._all_species_map[a_symbol]
        M = self.M_script(a)
        outputs = {}
        for _, ai in contributing_states(a):
            sym = ai.ionic_symbol
            output = self.thermodynamic_forces[sym] * M
            for b in self.all_species:
                N = self.N_script(a, b)
                for xj, bj in contributing_states(b):
                    output += xj * N * self.thermodynamic_forces[bj.ionic_symbol]
            outputs[sym] = output.sum(axis=1)
        return outputs

    @cached_property
    def _fluxes_BP(self):
        Fhat = self.flux_surface.Fhat
        fs = self.flux_surface
        B2fsav = fs.flux_surface_average(fs.B2) * u.T ** 2  # flux surface averaged B^2
        results = {}
        for a in self.all_species:
            for (
                ai
            ) in (
                a
            ):  # this could be rfactored out by iterating over self._charge_state_flows, instead, given a way to access ionizationstate back from ioniclevel
                sym = ai.ionic_symbol
                if sym not in self._charge_state_flows:
                    continue

                u_θ = (
                    self._charge_state_flows[sym] + self.thermodynamic_forces[sym]
                ) / B2fsav
                μ = self.μ[sym]
                Γ_BP = -(Fhat / ai.charge * (μ[0, :] * u_θ).sum()).si
                q_BP = -(
                    fs.Fhat
                    * constants.k_B
                    * a.T
                    / ai.charge
                    * (μ[1, :] * u_θ).sum()
                ).si
                results[sym] = Fluxes(
                    Γ_BP.to(particle_flux_unit), q_BP.to(heat_flux_unit)
                )
        return results

    @cached_property
    def _fluxes_PS(self):
        fs = self.flux_surface
        B2fsav = fs.flux_surface_average(fs.B2) * u.T ** 2  # flux surface averaged B^2
        Binv2fsav = fs.flux_surface_average(1 / fs.B2) / u.T ** 2
        results = {}
        fs = self.flux_surface
        for a in self.all_species:
            silly = self._funnymatrix(a[0].symbol)
            for xi, ai in contributing_states(a):
                sym = ai.symbol
                prefactor = (
                    -fs.Fhat / ai.charge * xi / B2fsav * (1 - B2fsav * Binv2fsav)
                )
                Γ_PS = prefactor * silly[sym][0]  # overlarge by s/m5
                q_PS = (
                    prefactor * constants.k_B * a.T * silly[sym][1]
                )  # overlarge by μ.unit
                results[sym] = Fluxes(
                    Γ_PS.to(particle_flux_unit), q_PS.to(heat_flux_unit)
                )
        return results

    @cached_property
    def _fluxes_CL(self):
        fs = self.flux_surface
        FSA = fs.flux_surface_average(fs.GradRho2 / fs.B2) / u.T ** 2
        # TODO fs.rho is [m]; fs.GradRho2 is actually [fs.gradRho]^2, gradRho is [1]
        # TODO FSA does not drop units; B2 and the others are unitless
        Fhat = self.flux_surface.Fhat
        results = {}
        for a in self.all_species:
            silly = self._funnymatrix(a[0].symbol)
            for xi, ai in contributing_states(a):
                sym = ai.symbol
                prefactor = FSA / Fhat * xi / ai.charge
                Γ_CL = prefactor * silly[sym][0]
                q_CL = prefactor * constants.k_B * a.T * silly[sym][1]
                results[sym] = Fluxes(
                    Γ_CL.to(particle_flux_unit), q_CL.to(heat_flux_unit)
                )
        return results

    @cached_property
    def fluxes(self):
        results = {}
        for a in self.all_species:
            for _, ai in contributing_states(a):
                sym = ai.symbol
                Γ_BP, q_BP = self._fluxes_BP[sym]
                Γ_PS, q_PS = self._fluxes_PS[sym]
                Γ_CL, q_CL = self._fluxes_CL[sym]
                results[sym] = Fluxes(Γ_BP + Γ_PS + Γ_CL, q_BP + q_PS + q_CL)
        return results

    def to_dataset(self, *, with_input=True) -> xarray.Dataset:
        r"""
        Converts the outputs to `~xarray.Dataset`.

        Parameters
        ----------
        with_input: bool (default: True)
            if True and `self` was initialized with an xarray object through
            `~self.from_xarray_surface`, return a dataset merged with the input
            object. Otherwise, return just the calculation results.

        """
        result = xarray.Dataset(
            {
                "total_particle_flux": (
                    "particle",
                    u.Quantity([flux.particle_flux for flux in self.fluxes.values()]),
                ),
                "total_heat_flux": (
                    "particle",
                    u.Quantity([flux.heat_flux for flux in self.fluxes.values()]),
                ),
                "BP_particle_flux": (
                    "particle",
                    u.Quantity(
                        [flux.particle_flux for flux in self._fluxes_BP.values()]
                    ),
                ),
                "BP_heat_flux": (
                    "particle",
                    u.Quantity([flux.heat_flux for flux in self._fluxes_BP.values()]),
                ),
                "CL_particle_flux": (
                    "particle",
                    u.Quantity(
                        [flux.particle_flux for flux in self._fluxes_CL.values()]
                    ),
                ),
                "CL_heat_flux": (
                    "particle",
                    u.Quantity([flux.heat_flux for flux in self._fluxes_CL.values()]),
                ),
                "PS_particle_flux": (
                    "particle",
                    u.Quantity(
                        [flux.particle_flux for flux in self._fluxes_PS.values()]
                    ),
                ),
                "PS_heat_flux": (
                    "particle",
                    u.Quantity([flux.heat_flux for flux in self._fluxes_PS.values()]),
                ),
                "diffusion_coefficient": (
                    "particle",
                    u.Quantity(list(self.diffusion_coefficient.values())),
                ),
                "thermal_conductivity": (
                    "particle",
                    u.Quantity(list(self.thermal_conductivity.values())),
                ),
                "bootstrap_current": self.bootstrap_current,
                # TODO this probably won't fit in the common array because of spatial dependence
                # "local_heat_flux": (
                #     ("particle", "directions", "lp", ),
                #     u.Quantity(list(self.local_heat_flux_components.values())),
                # ),
                # "local_particle_velocities": (
                #     ("particle", "directions", "lp", ),
                #     u.Quantity(list(self.local_flow_velocities.values())),
                # ),
            },
            {
                "particle": list(self.all_contributing_states_symbols()),
                "psi": self.flux_surface.psi,
                # "directions": ["poloidal", "toroidal", "parallel", "perpendicular"],
                # "lp": self.flux_surface.lp,
            },
        )

        if with_input and self._dataset_input is not None:
            return xarray.merge([result, self._dataset_input])
        else:
            return result

    @cached_property
    def diffusion_coefficient(self):
        results = {}
        for a in self.all_species:
            for _, ai in contributing_states(a):
                sym = ai.ionic_symbol
                flux = self.fluxes[sym].particle_flux
                results[sym] = (
                    -flux / self.density_gradient[sym]
                )  # Eq48 TODO this is a partial adaptation
        return results

    @cached_property
    def thermal_conductivity(self):
        results = {}
        for a in self.all_species:
            for _, ai in contributing_states(a):
                sym = ai.ionic_symbol
                flux = self.fluxes[sym].heat_flux
                results[sym] = -flux / self.temperature_gradient[sym]
        return results

    @cached_property
    def bootstrap_current(self) -> u.J / u.m ** 2 / u.T:
        """
        Bootstrap current caused by the charge state flows.
        """

        def gen():
            for a in self.all_species:
                for (_, ai), n in zip(contributing_states(a), a.n):
                    sym = ai.symbol
                    u_velocity = self._charge_state_flows[sym][0]
                    yield ai.charge * n * u_velocity
                    # eq 37, second term

        return sum(gen())

    @cached_property
    def local_flow_velocities(self):
        fs = self.flux_surface
        B2fsav = fs.flux_surface_average(fs.B2) * u.T ** 2  # flux surface averaged B^2
        B_p = fs.Bp * u.T
        B_t = fs.Bphivals * u.T  # TODO needs renaming T_T
        B = fs.Bmag * u.T
        results = {}
        for a in self.all_species:
            for _, ai in contributing_states(a):
                sym = ai.ionic_symbol
                u_θ = (
                    self._charge_state_flows[sym] + self.thermodynamic_forces[sym]
                ) / B2fsav
                u_hat_theta_1_ai = u_θ[0]
                S_theta_1_ai = self.thermodynamic_forces[sym][0]
                u_p_ai = B_p * u_hat_theta_1_ai
                u_t_ai = B_t * u_hat_theta_1_ai - S_theta_1_ai / B_t
                u_parallel_ai = B * u_hat_theta_1_ai - S_theta_1_ai / B
                u_perp_ai = B_p / B / B_t * S_theta_1_ai
                results[sym] = u.Quantity([u_p_ai, u_t_ai, u_parallel_ai, u_perp_ai]).si
        return results

    @cached_property
    def local_heat_flux_components(self):
        fs = self.flux_surface
        B2fsav = fs.flux_surface_average(fs.B2) * u.T ** 2  # flux surface averaged B^2
        B_p = fs.Bp * u.T
        B_t = fs.Bphivals * u.T  # TODO needs renaming T_T
        B = fs.Bmag * u.T
        results = {}
        for a in self.all_species:
            n_i = a.n
            T_i = a.T
            p = n_i * T_i
            for _, ai in contributing_states(a):
                sym = ai.ionic_symbol
                u_θ = (
                    self._charge_state_flows[sym] + self.thermodynamic_forces[sym]
                ) / B2fsav
                u_hat_theta_2_ai = u_θ[1]
                S_theta_2_ai = self.thermodynamic_forces[sym][1]
                p_ai = p[ai.charge_number]
                q_p_ai = 5 / 2 * p_ai * B_p * u_hat_theta_2_ai
                q_t_ai = 5 / 2 * p_ai * (B_t * u_hat_theta_2_ai - S_theta_2_ai / B_t)
                q_parallel_ai = 5 / 2 * p_ai * (B * u_hat_theta_2_ai - S_theta_2_ai / B)
                q_perp_ai = 5 / 2 * p_ai * B_p / B / B_t * S_theta_2_ai
                results[sym] = u.Quantity([q_p_ai, q_t_ai, q_parallel_ai, q_perp_ai]).si
        return results
