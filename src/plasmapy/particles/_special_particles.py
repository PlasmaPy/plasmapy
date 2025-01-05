"""Functionality to store data and taxonomy information for special particles."""

__all__ = [
    "antiparticles",
    "create_particles_dict",
    "data_about_special_particles",
    "particle_zoo",
    "ParticleZoo",
    "special_ion_masses",
]

import astropy.constants as const
import astropy.units as u
import numpy as np

from plasmapy.particles import _elements


class ParticleZoo:
    """
    Create an object with taxonomy information for special particles.

    The `~plasmapy.particles._special_particles._ParticleZooClass._taxonomy_dict`
    attribute contains the name of each classification (e.g.,
    ``'lepton'``, ``'baryon'``, ``'matter'``, etc.) as the keys and a
    set of particle symbol strings of the particles belonging to that
    classification.

    The attributes of this class provide sets of strings representing
    particles in the corresponding category.

    Examples
    --------
    >>> particle_zoo_ = ParticleZoo()
    >>> "e-" in particle_zoo_.leptons
    True
    >>> "nu_e" in particle_zoo_.antineutrinos
    False
    >>> "mu+" in particle_zoo_.antiparticles
    True
    """

    def __init__(self) -> None:
        leptons = {"e-", "mu-", "tau-", "nu_e", "nu_mu", "nu_tau"}
        antileptons = {"e+", "mu+", "tau+", "anti_nu_e", "anti_nu_mu", "anti_nu_tau"}
        baryons = {"p+", "n"}
        antibaryons = {"p-", "antineutron"}

        particles = leptons | baryons
        antiparticles = antileptons | antibaryons
        fermions = leptons | antileptons | baryons | antibaryons

        bosons = set()

        neutrinos = {lepton for lepton in leptons if "nu" in lepton}

        antineutrinos = {antilepton for antilepton in antileptons if "nu" in antilepton}

        self._taxonomy_dict = {
            "lepton": leptons,
            "antilepton": antileptons,
            "baryon": baryons,
            "antibaryon": antibaryons,
            "fermion": fermions,
            "boson": bosons,
            "neutrino": neutrinos,
            "antineutrinos": antineutrinos,
            "matter": particles,
            "antimatter": antiparticles,
        }

    @property
    def leptons(self) -> set[str]:
        """Return all leptons, excluding antileptons."""
        return self._taxonomy_dict["lepton"]

    @property
    def antileptons(self) -> set[str]:
        """Return all antileptons."""
        return self._taxonomy_dict["antilepton"]

    @property
    def baryons(self) -> set[str]:
        """Return all baryons, excluding antibaryons."""
        return self._taxonomy_dict["baryon"]

    @property
    def antibaryons(self) -> set[str]:
        """Return all antibaryons."""
        return self._taxonomy_dict["antibaryon"]

    @property
    def fermions(self) -> set[str]:
        """Return all fermions."""
        return self._taxonomy_dict["fermion"]

    @property
    def bosons(self) -> set[str]:
        """Return all bosons."""
        return self._taxonomy_dict["boson"]

    @property
    def neutrinos(self) -> set[str]:
        """Return all neutrinos."""
        return self._taxonomy_dict["neutrino"]

    @property
    def antineutrinos(self) -> set[str]:
        """Return all antineutrinos."""
        return self._taxonomy_dict["antineutrinos"]

    @property
    def particles(self) -> set[str]:
        """Return all particles, excluding antiparticles."""
        return self._taxonomy_dict["matter"]

    @property
    def antiparticles(self) -> set[str]:
        """Return all antiparticles."""
        return self._taxonomy_dict["antimatter"]

    @property
    def everything(self) -> set[str]:
        """Return all particles and antiparticles."""
        return self._taxonomy_dict["matter"] | self._taxonomy_dict["antimatter"]


particle_zoo = ParticleZoo()
particle_zoo.__doc__ = """
Data container for representations of "special" particles, like leptons
and baryons.

An instance of `~plasmapy.particles._special_particles.ParticleZoo`.
"""


def create_particles_dict() -> dict[str, dict]:  # noqa: C901, PLR0912
    """
    Create a dictionary of dictionaries that contains physical
    information for particles and antiparticles that are not elements or
    ions.

    The keys of the top-level dictionary are the standard particle
    symbols. The values of the top-level dictionary are dictionaries for
    each particle or antiparticle with strings such as ``'name'``,
    ``'mass'``, and ``'spin'`` as the keys and the corresponding atomic
    properties as symbols.
    """

    symbols_and_names = [
        ("e-", "electron"),
        ("e+", "positron"),
        ("mu-", "muon"),
        ("mu+", "antimuon"),
        ("tau-", "tau"),
        ("tau+", "antitau"),
        ("nu_e", "electron neutrino"),
        ("anti_nu_e", "electron antineutrino"),
        ("nu_mu", "muon neutrino"),
        ("anti_nu_mu", "muon antineutrino"),
        ("nu_tau", "tau neutrino"),
        ("anti_nu_tau", "tau antineutrino"),
        ("p+", "proton"),
        ("p-", "antiproton"),
        ("n", "neutron"),
        ("antineutron", "antineutron"),
    ]

    particles = {particle: {} for particle in particle_zoo.everything}

    for symbol, name in symbols_and_names:
        particles[symbol]["name"] = name

    for fermion in particle_zoo.fermions:
        particles[fermion]["spin"] = 0.5

    for boson in particle_zoo.bosons:
        particles[boson]["spin"] = 0

    for lepton in particle_zoo.leptons:
        particles[lepton]["class"] = "lepton"
        particles[lepton]["lepton number"] = 1
        particles[lepton]["baryon number"] = 0
        if lepton not in particle_zoo.neutrinos:
            particles[lepton]["charge number"] = -1
        else:
            particles[lepton]["charge number"] = 0

    for antilepton in particle_zoo.antileptons:
        particles[antilepton]["class"] = "antilepton"
        particles[antilepton]["lepton number"] = -1
        particles[antilepton]["baryon number"] = 0
        if antilepton not in particle_zoo.antineutrinos:
            particles[antilepton]["charge number"] = 1
        else:
            particles[antilepton]["charge number"] = 0

    for baryon in particle_zoo.baryons:
        particles[baryon]["class"] = "baryon"
        particles[baryon]["lepton number"] = 0
        particles[baryon]["baryon number"] = 1

    for antibaryon in particle_zoo.antibaryons:
        particles[antibaryon]["class"] = "antibaryon"
        particles[antibaryon]["lepton number"] = 0
        particles[antibaryon]["baryon number"] = -1

    for particle in particle_zoo.leptons | particle_zoo.antileptons:
        if "e" in particle:
            particles[particle]["generation"] = 1
        elif "mu" in particle:
            particles[particle]["generation"] = 2
        elif "tau" in particle:
            particles[particle]["generation"] = 3

    for particle in particle_zoo.leptons | particle_zoo.antileptons:
        if "nu" not in particle:
            if "e" in particle:
                particles[particle]["mass"] = const.m_e
            elif "mu" in particle:
                particles[particle]["mass"] = 1.883_531_594e-28 * u.kg
                particles[particle]["half-life"] = 2.1969811e-6 * u.s
            elif "tau" in particle:
                particles[particle]["mass"] = 3.167_47e-27 * u.kg
                particles[particle]["half-life"] = 2.906e-13 * u.s

    # Setting the neutrino mass to None reminds us that, while neutrinos
    # are not massless, we only have upper limits on what the neutrino
    # mass actually is.

    for particle in particle_zoo.neutrinos | particle_zoo.antineutrinos:
        particles[particle]["mass"] = None

    special_attributes = {
        "p+": {
            "element": "H",
            "atomic number": 1,
            "mass number": 1,
            "element name": "hydrogen",
            "isotope": "H-1",
            "ion": "p+",
            "mass": const.m_p,
            "charge number": 1,
            "periodic table": _elements.PeriodicTable(
                group=1, period=1, block="s", category="nonmetal"
            ),
        },
        "p-": {"mass": const.m_p, "charge number": -1},
        "n": {"mass": const.m_n, "half-life": 881.5 * u.s, "charge number": 0},
        "antineutron": {
            "mass": const.m_n,
            "half-life": 881.5 * u.s,
            "charge number": 0,
        },
    }

    for particle, attributes in special_attributes.items():
        particles[particle] = {**attributes, **particles[particle]}

    for particle in particle_zoo.everything:
        if "half-life" not in particles[particle]:
            particles[particle]["half-life"] = np.inf * u.s

    for particle in particle_zoo.particles:
        particles[particle]["antimatter"] = False

    for antiparticle in particle_zoo.antiparticles:
        particles[antiparticle]["antimatter"] = True

    return particles


data_about_special_particles = create_particles_dict()

special_ion_masses = {
    "p+": const.m_p,
    "D 1+": 3.343583719e-27 * u.kg,
    "T 1+": 5.007356665e-27 * u.kg,
}

antiparticles = {
    "p+": "p-",
    "n": "antineutron",
    "e-": "e+",
    "mu-": "mu+",
    "tau-": "tau+",
    "nu_e": "anti_nu_e",
    "nu_mu": "anti_nu_mu",
    "nu_tau": "anti_nu_tau",
    "p-": "p+",
    "antineutron": "n",
    "e+": "e-",
    "mu+": "mu-",
    "tau+": "tau-",
    "anti_nu_e": "nu_e",
    "anti_nu_mu": "nu_mu",
    "anti_nu_tau": "nu_tau",
}
