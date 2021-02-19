"""Tests for particle collections."""

import astropy.units as u
import pytest

from plasmapy.particles import alpha, electron, neutron, proton
from plasmapy.particles.atomic import atomic_number
from plasmapy.particles.exceptions import InvalidParticleError
from plasmapy.particles.particle_class import (
    CustomParticle,
    DimensionlessParticle,
    Particle,
)
from plasmapy.particles.particle_collections import ParticleList

custom_particle = CustomParticle(mass=1e-25 * u.kg, charge=1e-18 * u.C)
dimensionless_particle = DimensionlessParticle(mass=1.25, charge=1.58)


attributes = [
    "charge",
    "half_life",
    "integer_charge",
    "mass",
    "mass_energy",
]


@pytest.fixture
def various_particles():
    """A sample `ParticleList` with several different valid particles."""
    return ParticleList(
        [
            "H",
            "He",
            "e-",
            "alpha",
            "tau neutrino",
            CustomParticle(mass=3 * u.kg, charge=5 * u.C),
            CustomParticle(),
            CustomParticle(mass=7 * u.kg),
            CustomParticle(charge=11 * u.C),
        ]
    )


def _everything_is_particle_or_custom_particle(iterable):
    """
    Test that every object in an iterable is either a `Particle` instance
    or a `CustomParticle` instance.
    """
    return all([isinstance(p, (Particle, CustomParticle)) for p in iterable])


@pytest.mark.parametrize(
    "args",
    [
        (),
        ([electron]),
        ([electron, proton]),
        ([electron, proton, alpha]),
        (["e-", "e+"]),
        ([electron, "e-"]),
        ([custom_particle]),
        ([custom_particle, electron, "e-"]),
    ],
)
def test_particle_list_membership(args):
    """
    Test that the particles in the `ParticleList` match the particles
    (or particle-like objects) that are passed to it.
    """
    particle_list = ParticleList(args)
    for arg, particle in zip(args, particle_list):
        assert particle == arg
    assert _everything_is_particle_or_custom_particle(particle_list)
    assert _everything_is_particle_or_custom_particle(particle_list.data)


@pytest.mark.parametrize("attribute", attributes)
def test_particle_list_attributes(attribute, various_particles):
    """
    Test that the attributes of ParticleList correspond to the
    attributes of the listed particles.

    This class does not test ParticleList instances that include
    CustomParticle instances inside of them.
    """
    particle_list_arguments = (electron, "e+", proton, neutron, alpha)
    particle_list = ParticleList(particle_list_arguments)
    expected_particles = [Particle(arg) for arg in particle_list_arguments]
    actual = getattr(particle_list, attribute)
    expected = [getattr(particle, attribute) for particle in expected_particles]
    assert u.allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("attribute", attributes)
def test_particle_list_no_redefining_attributes(various_particles, attribute):
    """
    Test that attributes of `ParticleList` cannot be manually redefined.

    This test may fail if `@cached_property` is used instead of `@property`
    because `@cached_property` allows reassignment while `@property` does
    not.
    """
    with pytest.raises(AttributeError):
        various_particles.__setattr__(attribute, 42)


def test_particle_list_len():
    """Test that using `len` on a `ParticleList` returns the expected number."""
    original_list = ["n", "p", "e-"]
    particle_list = ParticleList(original_list)
    assert len(particle_list) == len(original_list)


def test_particle_list_append(various_particles):
    """Test that a particle-like object can get appended to a `ParticleList`."""
    original_length = len(various_particles)
    various_particles.append("Li")
    appended_item = various_particles[-1]
    assert len(various_particles) == original_length + 1
    assert isinstance(appended_item, Particle)
    assert appended_item == Particle("Li")
    assert various_particles.data[-1] is appended_item


def test_particle_list_pop(various_particles):
    """Test that the last item in the `ParticleList` is removed when"""
    expected = various_particles[:-1]
    particle_that_should_be_removed = various_particles[-1]
    removed_particle = various_particles.pop()
    assert various_particles == expected
    assert removed_particle == particle_that_should_be_removed


def test_particle_list_extend(various_particles):
    """
    Test that a `ParticleList` can be extended when provided with an
    iterable that yields particle-like objects.
    """
    new_particles = ["Fe", Particle("e-"), CustomParticle()]
    various_particles.extend(new_particles)
    assert _everything_is_particle_or_custom_particle(various_particles)
    assert various_particles[-3:] == new_particles


invalid_particles = (0, "not a particle", DimensionlessParticle())


def test_particle_list_instantiate_with_invalid_particles():
    """
    Test that a `ParticleList` instance cannot be created when it is
    provided with invalid particles.
    """
    with pytest.raises(InvalidParticleError):
        ParticleList(invalid_particles)


@pytest.mark.parametrize("invalid_particle", invalid_particles)
def test_particle_list_append_invalid_particle(various_particles, invalid_particle):
    """
    Test that objects that are not particle-like cannot be appended to
    a `ParticleList` instance.
    """
    with pytest.raises((InvalidParticleError, TypeError)):
        various_particles.append(invalid_particle)


def test_particle_list_extend_with_invalid_particles(various_particles):
    """
    Test that a `ParticleList` instance cannot be extended with any
    objects that are not particle-like.
    """
    with pytest.raises(InvalidParticleError):
        various_particles.extend(invalid_particles)


@pytest.mark.parametrize("invalid_particle", invalid_particles)
def test_particle_list_insert_invalid_particle(various_particles, invalid_particle):
    """
    Test that objects that are not particle-like cannot be inserted into
    a `ParticleList` instance.
    """
    with pytest.raises((InvalidParticleError, TypeError)):
        various_particles.insert(1, invalid_particle)


def test_particle_list_sort_with_key_and_reverse():
    """
    Test that a `ParticleList` instance can be sorted if a key is
    provided, and that the ``reverse`` keyword argument works too.
    """
    elements = ["He", "H", "Fe", "U"]
    particle_list = ParticleList(elements)
    particle_list.sort(key=atomic_number, reverse=True)
    assert particle_list.symbols == ["U", "Fe", "He", "H"]


def test_particle_list_sort_without_key(various_particles):
    """Test that a `ParticleList` cannot be sorted if a key is not provided."""
    with pytest.raises(TypeError):
        various_particles.sort()
