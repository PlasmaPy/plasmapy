"""
The `plasmapy.particles` subpackage provides access to information about
atoms, isotopes, ions, and other particles.
"""
# __all__ will be auto populated below
__all__ = []

import inspect

from plasmapy.particles.atomic import (
    atomic_number,
    charge_number,
    common_isotopes,
    electric_charge,
    half_life,
    ionic_levels,
    is_stable,
    isotopic_abundance,
    known_isotopes,
    mass_number,
    particle_mass,
    reduced_mass,
    stable_isotopes,
    standard_atomic_weight,
)
from plasmapy.particles.decorators import particle_input
from plasmapy.particles.ionization_state import IonicLevel, IonizationState
from plasmapy.particles.ionization_state_collection import IonizationStateCollection
from plasmapy.particles.nuclear import nuclear_binding_energy, nuclear_reaction_energy
from plasmapy.particles.particle_class import (
    AbstractParticle,
    AbstractPhysicalParticle,
    CustomParticle,
    DimensionlessParticle,
    molecule,
    Particle,
    ParticleLike,
    Photon,
)
from plasmapy.particles.particle_collections import ParticleList, ParticleListLike
from plasmapy.particles.serialization import (
    json_load_particle,
    json_loads_particle,
    ParticleJSONDecoder,
)
from plasmapy.particles.symbols import (
    atomic_symbol,
    element_name,
    ionic_symbol,
    isotope_symbol,
    particle_symbol,
)

proton = Particle("p+")
"""A |Particle| instance representing a proton."""

electron = Particle("e-")
"""A |Particle| instance representing an electron."""

neutron = Particle("n")
"""A |Particle| instance representing a neutron."""

positron = Particle("e+")
"""A |Particle| instance representing a positron."""

deuteron = Particle("D 1+")
"""A |Particle| instance representing a positively charged deuterium ion."""

triton = Particle("T 1+")
"""A |Particle| instance representing a positively charged tritium ion."""

alpha = Particle("He-4 2+")
"""
A |Particle| instance representing an alpha particle (doubly charged
helium-4).
"""

# auto populate __all__
for name, obj in list(globals().items()):
    if inspect.ismodule(obj) or name.startswith("__") or name.endswith("__"):
        continue

    __all__.append(name)

__all__.sort()

del inspect, name, obj
