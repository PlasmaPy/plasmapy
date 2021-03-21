import astropy.units as u
import numpy as np
import pytest

from plasmapy.formulary.neoclassical import (
    charge_weighting_factor,
    effective_momentum_relaxation_rate,
    K_B_ai,
    L_friction_coefficient,
    M_matrix,
    M_script,
    N_matrix,
    N_script,
    pitch_angle_diffusion_rate,
)
from plasmapy.particles import (
    IonizationState,
    IonizationStateCollection,
    Particle,
    proton,
)

all_species = IonizationStateCollection(
    {
        "H": [0, 1],
        #      "D": [0, 1],   raises ParticleError, why?
        "C": [0, 1 / 1.1, 0.1 / 1.1, 0, 0, 0, 0],
    },
    n0=1e20 * u.m ** -3,
    abundances={"H": 1, "C": 0.11},
    T_e=10 * u.eV,
)
hydrogen = all_species["H"]
carbon_states = all_species["C"]


@pytest.mark.parametrize(
    ["function", "shape"],
    [(N_matrix, (3, 3)), (M_matrix, (3, 3)), (N_script, (3, 3)),],
)
def test_matrix_between_elements(function, shape, num_regression):
    data = function(hydrogen, carbon_states)
    try:
        data = data.si.value
    except AttributeError:
        pass  # we're already a numpy array
    assert data.shape == shape, data.shape
    num_regression.check({function.__name__: data.flatten()})


@pytest.mark.parametrize(
    ["function", "shape"], [(M_script, (3, 3)),],
)
def test_matrix_between_element_and_all_species(function, shape, num_regression):
    data = function(hydrogen, all_species)
    assert data.shape == shape, data.shape
    num_regression.check({function.__name__: data.si.value.flatten()})


@pytest.mark.parametrize("function", [effective_momentum_relaxation_rate,])
def test_number_between_ionization_states(function, num_regression):
    data = function(hydrogen, carbon_states)
    num_regression.check({function.__name__: data.si.value})


def test_weighted_ionization_factor(num_regression):
    data = charge_weighting_factor(1, carbon_states)
    num_regression.check({"xi": data.si.value})
    assert data == 1 - charge_weighting_factor(2, carbon_states)


def test_L_friction_coefficient(num_regression):
    data = L_friction_coefficient(hydrogen, 1, carbon_states, 1, all_species)
    num_regression.check({"L": data.si.value.flatten()})


def test_pitch_angle_diffusion_rate_and_banana_vsicosity(num_regression, flux_surface):
    x = np.logspace(-6, 6, 5000)
    ν_D_ai = pitch_angle_diffusion_rate(x, 1, carbon_states, all_species)
    k = K_B_ai(x, 1, carbon_states, all_species, flux_surface)
    num_regression.check(
        {"x": x, "ν_D_ai": ν_D_ai.si.value, "K_B_ai": k.si.value},
        tolerances={"ν_D_ai": {"rtol": 1e-4}, "K_B_ai": {"rtol": 1e-4}},
    )
