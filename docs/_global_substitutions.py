"""
Global substitutions that can be used throughout PlasmaPy's
reStructuredText documentation, using the sphinxcontrib-globalsubs
extension to Sphinx.

The substitutions are defined in ``global_substitutions``, which is then
imported in :file:`docs/conf.py` so that sphinxcontrib-globalsubs can
use it to define substitutions.

The key to the dictionary is the name of the substitution. For example,
if the key is ``"particle-like"``, then it can be used as
``|particle-like|`` throughout the documentation.
"""

__all__ = ["global_substitutions", "make_global_substitutions_table"]

import collections
import pathlib

plasmapy_subs: dict[str, str] = {
    "atomic_number": ":func:`~plasmapy.particles.atomic.atomic_number`",
    "atomic_symbol": ":func:`~plasmapy.particles.symbols.atomic_symbol`",
    "charge_number": ":func:`~plasmapy.particles.atomic.charge_number`",
    "ChargeError": ":class:`~plasmapy.particles.exceptions.ChargeError`",
    "ClassicalTransport": ":class:`~plasmapy.formulary.braginskii.ClassicalTransport`",
    "common_isotopes": ":func:`~plasmapy.particles.atomic.common_isotopes`",
    "CustomParticle": ":class:`~plasmapy.particles.particle_class.CustomParticle`",
    "DimensionlessParticle": ":class:`~plasmapy.particles.particle_class.DimensionlessParticle`",
    "electric_charge": ":func:`~plasmapy.particles.atomic.electric_charge`",
    "element_name": ":func:`~plasmapy.particles.symbols.element_name`",
    "half_life": ":func:`~plasmapy.particles.atomic.half_life`",
    "InvalidElementError": ":class:`~plasmapy.particles.exceptions.InvalidElementError`",
    "InvalidIonError": ":class:`~plasmapy.particles.exceptions.InvalidIonError`",
    "InvalidIsotopeError": ":class:`~plasmapy.particles.exceptions.InvalidIsotopeError`",
    "InvalidParticleError": ":class:`~plasmapy.particles.exceptions.InvalidParticleError`",
    "ionic_symbol": ":func:`~plasmapy.particles.symbols.ionic_symbol`",
    "IonicLevel": ":class:`~plasmapy.particles.ionization_state.IonicLevel`",
    "IonizationState": ":class:`~plasmapy.particles.ionization_state.IonizationState`",
    "IonizationStateCollection": ":class:`~plasmapy.particles.ionization_state_collection.IonizationStateCollection`",
    "is_stable": ":func:`~plasmapy.particles.atomic.is_stable`",
    "isotope_symbol": ":func:`~plasmapy.particles.symbols.isotope_symbol`",
    "isotopic_abundance": ":func:`~plasmapy.particles.atomic.isotopic_abundance`",
    "known_isotopes": ":func:`~plasmapy.particles.atomic.known_isotopes`",
    "Layer": "`~plasmapy.diagnostics.charged_particle_radiography.detector_stacks.Layer`",
    "mass_number": ":func:`~plasmapy.particles.atomic.mass_number`",
    "MissingParticleDataError": ":class:`~plasmapy.particles.exceptions.MissingParticleDataError`",
    "MissingParticleDataWarning": ":class:`~plasmapy.particles.exceptions.MissingParticleDataWarning`",
    "Particle": ":class:`~plasmapy.particles.particle_class.Particle`",
    "particle_input": ":func:`~plasmapy.particles.decorators.particle_input`",
    "particle_mass": ":func:`~plasmapy.particles.atomic.particle_mass`",
    "particle_symbol": ":func:`~plasmapy.particles.symbols.particle_symbol`",
    "ParticleError": ":class:`~plasmapy.particles.exceptions.ParticleError`",
    "ParticleLike": ":obj:`~plasmapy.particles.particle_class.ParticleLike`",
    "ParticleList": ":class:`~plasmapy.particles.particle_collections.ParticleList`",
    "ParticleListLike": ":obj:`~plasmapy.particles.particle_collections.ParticleListLike`",
    "ParticleTracker": ":class:`~plasmapy.simulation.particle_tracker.particle_tracker.ParticleTracker`",
    "ParticleWarning": ":class:`~plasmapy.particles.exceptions.ParticleWarning`",
    "reduced_mass": ":func:`~plasmapy.particles.atomic.reduced_mass`",
    "RelativisticBody": ":class:`~plasmapy.formulary.relativity.RelativisticBody`",
    "SingleParticleCollisionFrequencies": ":class:`~plasmapy.formulary.collisions.frequencies.SingleParticleCollisionFrequencies`",
    "stable_isotopes": ":func:`~plasmapy.particles.atomic.stable_isotopes`",
    "standard_atomic_weight": ":func:`~plasmapy.particles.atomic.standard_atomic_weight`",
    "UnexpectedParticleError": ":class:`~plasmapy.particles.exceptions.UnexpectedParticleError`",
    "validate_quantities": ":func:`~plasmapy.utils.decorators.validators.validate_quantities`",
}

# The trailing backslash and space are needed for the substitution to
# work correctly when used just before a period.

doc_subs: dict[str, str] = {
    "annotated": r":term:`annotated <annotation>`\ ",
    "annotation": r":term:`annotation`\ ",
    "argument": r":term:`argument`\ ",
    "arguments": r":term:`arguments <argument>`\ ",
    "atom-like": r":term:`atom-like`\ ",
    "bibliography": r":ref:`bibliography`\ ",
    "changelog guide": r":ref:`changelog guide`\ ",
    "charge number": r":term:`charge number`\ ",
    "code contribution workflow": r":ref:`code contribution workflow <workflow>`\ ",
    "coding guide": r":ref:`coding guide`\ ",
    "contributor guide": r":ref:`contributor guide`\ ",
    "decorated": r":term:`decorated <decorator>`\ ",
    "decorator": r":term:`decorator`\ ",
    "documentation guide": r":ref:`documentation guide`\ ",
    "expect-api-changes": "This feature is under development. Breaking changes may occur in the future.",
    "getting ready to contribute": r":ref:`getting ready to contribute`\ ",
    "glossary": r":ref:`glossary`\ ",
    "IDE": r":abbr:`IDE (Integrated Development Environment)`\ ",
    "keyword-only": r":term:`keyword-only`\ ",
    "lite-function": r":term:`lite-function`\ ",
    "lite-functions": r":term:`lite-functions`\ ",
    "many ways to contribute": r":ref:`many ways`\ ",
    "maxpython": "3.13",
    "minpython": "3.11",
    "open a terminal": r":ref:`open a terminal <opening-a-terminal>`\ ",
    "Open a terminal": r":ref:`Open a terminal <opening-a-terminal>`\ ",
    "parameter": r":term:`parameter`\ ",
    "parameters": r":term:`parameters <parameter>`\ ",
    "particle-like": r":term:`particle-like`\ ",
    "particle-list-like": r":term:`particle-list-like`\ ",
    "plasma-calculator": r":ref:`plasmapy-calculator`\ ",
    "src/plasmapy": r":file:`src/plasmapy`\ ",
    "testing guide": r":ref:`testing guide`\ ",
    "tests": r":file:`tests`\ ",
}

numpy_subs: dict[str, str] = {
    "array_like": ":term:`numpy:array_like`",
    "DTypeLike": "`~numpy.typing.DTypeLike`",
    "inf": "`~numpy.inf`",
    "nan": "`~numpy.nan`",
    "ndarray": ":class:`~numpy.ndarray`",
}

astropy_subs: dict[str, str] = {
    "Quantity": ":class:`~astropy.units.Quantity`",
}

# Because sphinxcontrib-globalsubs does not work for regular reStructuredText
# links, we first define the links and then process them afterwards into
# the form of a reStructuredText external link.

links_to_become_subs: dict[str, str] = {
    "Astropy": "https://docs.astropy.org",
    "Astropy Equivalencies": "https://docs.astropy.org/en/stable/units/equivalencies.html",
    "Citation File Format": "https://citation-file-format.github.io/",
    "DOI": "https://www.doi.org",
    "editable installation": "https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs",
    "git": "https://git-scm.com",
    "GitHub Actions": "https://docs.github.com/en/actions",
    "GitHub": "https://github.com",
    "h5py": "https://www.h5py.org",
    "intersphinx": "https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html",
    "Jupyter": "https://jupyter.org",
    "lmfit": "https://lmfit.github.io/lmfit-py/",
    "matplotlib": "https://matplotlib.org",
    "Matrix chat room": "https://app.element.io/#/room/#plasmapy:openastronomy.org",
    "mpmath": "https://mpmath.org/doc/current",
    "mypy": "https://mypy.readthedocs.io",
    "nbsphinx": "https://nbsphinx.readthedocs.io",
    "Nox": "https://nox.thea.codes",
    "NumPy": "https://numpy.org",
    "office hours": "https://www.plasmapy.org/meetings/office_hours/",
    "OpenPMD": "https://www.openpmd.org",
    "pandas": "https://pandas.pydata.org",
    "pip": "https://pip.pypa.io",
    "Plasma Hack Week": "https://hack.plasmapy.org",
    "PlasmaPy": "https://www.plasmapy.org",
    "PlasmaPy's data repository": "https://github.com/PlasmaPy/PlasmaPy-data",
    "PlasmaPy's documentation": "https://docs.plasmapy.org/en/stable",
    "PlasmaPy's GitHub repository": "https://github.com/PlasmaPy/PlasmaPy",
    "PlasmaPy's Matrix chat room": "https://app.element.io/#/room/#plasmapy:openastronomy.org",
    "PlasmaPy's website": "https://www.plasmapy.org",
    "pre-commit": "https://pre-commit.com",
    "pygments": "https://pygments.org",
    "PyPI": "https://pypi.org",
    "pytest": "https://docs.pytest.org",
    "Python": "https://www.python.org",
    "Python's documentation": "https://docs.python.org/3",
    "Read the Docs": "https://about.readthedocs.com/",
    "reStructuredText": "https://docutils.sourceforge.io/rst.html",
    "ruff": "https://docs.astral.sh/ruff",
    "SciPy": "https://scipy.org",
    "SPEC 0": "https://scientific-python.org/specs/spec-0000",
    "Sphinx": "https://www.sphinx-doc.org",
    "static type checking": "https://realpython.com/lessons/python-type-checking-overview",
    "towncrier": "https://github.com/twisted/towncrier",
    "type hint annotations": "https://peps.python.org/pep-0484",
    "xarray": "https://docs.xarray.dev",
    "Zenodo": "https://zenodo.org",
    "uv": "https://github.com/astral-sh/uv",
}

link_subs = {key: f"`{key} <{value}>`_" for key, value in links_to_become_subs.items()}

global_substitutions = plasmapy_subs | doc_subs | numpy_subs | astropy_subs | link_subs


def make_global_substitutions_table(
    rst_file: str = "contributing/_global_substitutions_table.rst",
) -> None:
    """
    Create a file containing a table of global reStructuredText substitutions
    for inclusion in :file:`docs/contributing/doc_guide.rst`.
    """

    headers = ("substitution", "replaces", "example")
    Row = collections.namedtuple("Row", headers)

    rows = [
        Row(
            f"``|{substitution}|``",
            f"``{global_substitutions[substitution].rstrip()}``",
            f"|{substitution}|",
        )
        for substitution in sorted(global_substitutions, key=lambda x: x.lower())
    ]
    lines = [
        ".. list-table:: Global Substitutions",
        "   :header-rows: 1",
        "",
        f"   * - {headers[0].title()}",
        f"     - {headers[1].title()}",
        f"     - {headers[2].title()}",
    ]

    for row in rows:
        lines.extend(
            [
                f"   * - {row.substitution}",
                f"     - {row.replaces}",
                f"     - {row.example}",
            ]
        )

    content = "\n".join(lines)

    with pathlib.Path(rst_file).open("w", encoding="utf-8") as file:
        file.write(content)


if __name__ == "__main__":
    """
    To test generating the table of substitutions, run:

    .. code-block: bash

        python _global_substitutions.py
    """
    make_global_substitutions_table()
