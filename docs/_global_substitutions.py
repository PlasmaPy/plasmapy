"""Global substitutions that can be used throughout PlasmaPy's documentation."""

plasmapy_subs = {
    "Layer": "`~plasmapy.diagnostics.charged_particle_radiography.detector_stacks.Layer`",
    "ClassicalTransport": ":class:`~plasmapy.formulary.braginskii.ClassicalTransport`",
    "RelativisticBody": ":class:`~plasmapy.formulary.relativity.RelativisticBody`",
    "SingleParticleCollisionFrequencies": ":class:`~plasmapy.formulary.collisions.frequencies.SingleParticleCollisionFrequencies`",
    "CustomParticle": ":class:`~plasmapy.particles.particle_class.CustomParticle`",
    "DimensionlessParticle": ":class:`~plasmapy.particles.particle_class.DimensionlessParticle`",
    "IonicLevel": ":class:`~plasmapy.particles.ionization_state.IonicLevel`",
    "IonizationState": ":class:`~plasmapy.particles.ionization_state.IonizationState`",
    "IonizationStateCollection": ":class:`~plasmapy.particles.ionization_state_collection.IonizationStateCollection`",
    "Particle": ":class:`~plasmapy.particles.particle_class.Particle`",
    "particle_input": ":func:`~plasmapy.particles.decorators.particle_input`",
    "ParticleLike": ":obj:`~plasmapy.particles.particle_class.ParticleLike`",
    "ParticleList": ":class:`~plasmapy.particles.particle_collections.ParticleList`",
    "ParticleListLike": ":obj:`~plasmapy.particles.particle_collections.ParticleListLike`",
    "ChargeError": ":class:`~plasmapy.particles.exceptions.ChargeError`",
    "InvalidElementError": ":class:`~plasmapy.particles.exceptions.InvalidElementError`",
    "InvalidIonError": ":class:`~plasmapy.particles.exceptions.InvalidIonError`",
    "InvalidIsotopeError": ":class:`~plasmapy.particles.exceptions.InvalidIsotopeError`",
    "InvalidParticleError": ":class:`~plasmapy.particles.exceptions.InvalidParticleError`",
    "MissingParticleDataError": ":class:`~plasmapy.particles.exceptions.MissingParticleDataError`",
    "MissingParticleDataWarning": ":class:`~plasmapy.particles.exceptions.MissingParticleDataWarning`",
    "ParticleError": ":class:`~plasmapy.particles.exceptions.ParticleError`",
    "ParticleWarning": ":class:`~plasmapy.particles.exceptions.ParticleWarning`",
    "UnexpectedParticleError": ":class:`~plasmapy.particles.exceptions.UnexpectedParticleError`",
    "atomic_number": ":func:`~plasmapy.particles.atomic.atomic_number`",
    "atomic_symbol": ":func:`~plasmapy.particles.symbols.atomic_symbol`",
    "element_name": ":func:`~plasmapy.particles.symbols.element_name`",
    "half_life": ":func:`~plasmapy.particles.atomic.half_life`",
    "ionic_symbol": ":func:`~plasmapy.particles.symbols.ionic_symbol`",
    "is_stable": ":func:`~plasmapy.particles.atomic.is_stable`",
    "isotope_symbol": ":func:`~plasmapy.particles.symbols.isotope_symbol`",
    "isotopic_abundance": ":func:`~plasmapy.particles.atomic.isotopic_abundance`",
    "mass_number": ":func:`~plasmapy.particles.atomic.mass_number`",
    "charge_number": ":func:`~plasmapy.particles.atomic.charge_number`",
    "electric_charge": ":func:`~plasmapy.particles.atomic.electric_charge`",
    "standard_atomic_weight": ":func:`~plasmapy.particles.atomic.standard_atomic_weight`",
    "particle_mass": ":func:`~plasmapy.particles.atomic.particle_mass`",
    "particle_symbol": ":func:`~plasmapy.particles.symbols.particle_symbol`",
    "known_isotopes": ":func:`~plasmapy.particles.atomic.known_isotopes`",
    "common_isotopes": ":func:`~plasmapy.particles.atomic.common_isotopes`",
    "reduced_mass": ":func:`~plasmapy.particles.atomic.reduced_mass`",
    "stable_isotopes": ":func:`~plasmapy.particles.atomic.stable_isotopes`",
    "ParticleTracker": ":class:`~plasmapy.simulation.particletracker.ParticleTracker`",
    "validate_quantities": ":func:`~plasmapy.utils.decorators.validators.validate_quantities`",
}

# The backslash is needed for the substitution to work correctly when
# used just before a period.

doc_subs = {
    "bibliography": r":ref:`bibliography`\ ",
    "changelog guide": r":ref:`changelog guide`\ ",
    "coding guide": r":ref:`coding guide`\ ",
    "contributor guide": r":ref:`contributor guide`\ ",
    "documentation guide": r":ref:`documentation guide`\ ",
    "expect-api-changes": "This functionality is under development. Backward incompatible changes might occur in future releases.",
    "getting ready to contribute": r":ref:`getting ready to contribute`\ ",
    "glossary": r":ref:`glossary`\ ",
    "minpython": "3.9",
    "maxpython": "3.11",
    "plasma-calculator": r":ref:`plasmapy-calculator`\ ",
    "release guide": r":ref:`release guide`\ ",
    "testing guide": r":ref:`testing guide`\ ",
    "code contribution workflow": r":ref:`code contribution workflow <workflow>`\ ",
    "annotated": r":term:`annotated <annotation>`\ ",
    "annotation": r":term:`annotation`\ ",
    "argument": r":term:`argument`\ ",
    "arguments": r":term:`arguments <argument>`\ ",
    "atom-like": r":term:`atom-like`\ ",
    "charge number": r":term:`charge number`\ ",
    "decorated": r":term:`decorated <decorator>`\ ",
    "decorator": r":term:`decorator`\ ",
    "keyword-only": r":term:`keyword-only`\ ",
    "parameter": r":term:`parameter`\ ",
    "parameters": r":term:`parameters <parameter>`\ ",
    "particle-like": r":term:`particle-like`\ ",
    "particle-list-like": r":term:`particle-list-like`\ ",
}

numpy_subs = {
    "inf": "`~numpy.inf`",
    "nan": "`~numpy.nan`",
    "ndarray": ":class:`~numpy.ndarray`",
    "array_like": ":term:`numpy:array_like`",
    "DTypeLike": "`~numpy.typing.DTypeLike`",
}

astropy_subs = {
    "Quantity": ":class:`~astropy.units.Quantity`",
}

links = {
    "Astropy": "https://docs.astropy.org",
    "black": "https://black.readthedocs.io",
    "Citation File Format": "https://citation-file-format.github.io/",
    "DOI": "https://www.doi.org",
    "editable installation": "https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs",
    "git": "https://git-scm.com",
    "GitHub Actions": "https://docs.github.com/en/actions",
    "GitHub": "https://github.com",
    "h5py": "https://www.h5py.org",
    "intersphinx": "https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html",
    "isort": "https://pycqa.github.io/isort",
    "Jupyter": "https://jupyter.org",
    "lmfit": "https://lmfit.github.io/lmfit-py/",
    "matplotlib": "https://matplotlib.org",
    "Matrix chat room": "https://app.element.io/#/room/#plasmapy:openastronomy.org",
    "mpmath": "https://mpmath.org/doc/current",
    "nbsphinx": "https://nbsphinx.readthedocs.io",
    "Numba": "https://numba.readthedocs.io",
    "NumPy": "https://numpy.org",
    "office hours": "https://www.plasmapy.org/meetings/office_hours/",
    "pip": "https://pip.pypa.io",
    "Plasma Hack Week": "https://hack.plasmapy.org",
    "PlasmaPy": "https://www.plasmapy.org",
    "PlasmaPy's documentation": "https://docs.plasmapy.org/en/stable",
    "PlasmaPy's GitHub repository": "https://github.com/PlasmaPy/plasmapy",
    "PlasmaPy's data repository": "https://github.com/PlasmaPy/PlasmaPy-data",
    "PlasmaPy's Matrix chat room": "https://app.element.io/#/room/#plasmapy:openastronomy.org",
    "PlasmaPy's website": "https://www.plasmapy.org",
    "pre-commit": "https://pre-commit.com",
    "pydocstyle": "https://www.pydocstyle.org/en/stable",
    "pygments": "https://pygments.org",
    "PyPI": "https://pypi.org",
    "pytest": "https://docs.pytest.org",
    "Python": "https://www.python.org",
    "Python's documentation": "https://docs.python.org/3",
    "Read the Docs": "https://readthedocs.org",
    "reStructuredText": "https://docutils.sourceforge.io/rst.html",
    "ruff": "https://beta.ruff.rs/docs",
    "SciPy": "https://scipy.org",
    "Sphinx": "https://www.sphinx-doc.org",
    "towncrier": "https://github.com/twisted/towncrier",
    "tox": "https://tox.wiki/en/latest",
    "xarray": "https://docs.xarray.dev",
    "Zenodo": "https://zenodo.org",
}

processed_links = {key: f"`{key} <{value}>`_" for key, value in links.items()}

global_substitutions = (
    plasmapy_subs | doc_subs | numpy_subs | astropy_subs | processed_links
)
