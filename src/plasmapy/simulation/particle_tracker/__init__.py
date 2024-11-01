"""
Functionality to track charged particles in electromagnetic fields.

This subpackage contains functionality related to the
|ParticleTracker| class. This functionality include the definition of
the particle tracker as well as the definitions of save routines and
termination conditions.
"""

__all__ = ["particle_tracker", "save_routines", "termination_conditions"]

from plasmapy.simulation.particle_tracker import (
    particle_tracker,
    save_routines,
    termination_conditions,
)
