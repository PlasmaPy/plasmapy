import pytest

from plasmapy.simulation.particletracker import ParticleTracker


@pytest.fixture(params=ParticleTracker.integrators.keys())
def integrator_name(request):
    return request.param


@pytest.fixture(params=ParticleTracker.integrators.values())
def integrator(request):
    return request.param
