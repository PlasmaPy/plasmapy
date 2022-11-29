.. These are ReST substitutions and links that can be used throughout the docs
   (and docstrings) because they are added to ``docs/conf.py::rst_epilog``.

.. ------------------
.. plasmapy.formulary
.. ------------------

.. |ClassicalTransport| replace:: :class:`~plasmapy.formulary.braginskii.ClassicalTransport`
.. |RelativisticBody| replace:: :class:`~plasmapy.formulary.relativity.RelativisticBody`

.. ------------------
.. plasmapy.particles
.. ------------------

.. |CustomParticle| replace:: :class:`~plasmapy.particles.particle_class.CustomParticle`
.. |DimensionlessParticle| replace:: :class:`~plasmapy.particles.particle_class.DimensionlessParticle`
.. |Photon| replace:: :class:`~plasmapy.particles.particle_class.Photon`
.. |IonicLevel| replace:: :class:`~plasmapy.particles.ionization_state.IonicLevel`
.. |IonizationState| replace:: :class:`~plasmapy.particles.ionization_state.IonizationState`
.. |IonizationStateCollection| replace:: :class:`~plasmapy.particles.ionization_state_collection.IonizationStateCollection`
.. |Particle| replace:: :class:`~plasmapy.particles.particle_class.Particle`
.. |particle_input| replace:: :func:`~plasmapy.particles.decorators.particle_input`
.. |ParticleLike| replace:: :obj:`~plasmapy.particles.particle_class.ParticleLike`
.. |ParticleList| replace:: :class:`~plasmapy.particles.particle_collections.ParticleList`
.. |ParticleListLike| replace:: :obj:`~plasmapy.particles.particle_collections.ParticleListLike`

.. |ChargeError| replace:: :class:`~plasmapy.particles.exceptions.ChargeError`
.. |InvalidElementError| replace:: :class:`~plasmapy.particles.exceptions.InvalidElementError`
.. |InvalidIonError| replace:: :class:`~plasmapy.particles.exceptions.InvalidIonError`
.. |InvalidIsotopeError| replace:: :class:`~plasmapy.particles.exceptions.InvalidIsotopeError`
.. |InvalidParticleError| replace:: :class:`~plasmapy.particles.exceptions.InvalidParticleError`
.. |MissingParticleDataError| replace:: :class:`~plasmapy.particles.exceptions.MissingParticleDataError`
.. |MissingParticleDataWarning| replace:: :class:`~plasmapy.particles.exceptions.MissingParticleDataWarning`
.. |ParticleError| replace:: :class:`~plasmapy.particles.exceptions.ParticleError`
.. |ParticleWarning| replace:: :class:`~plasmapy.particles.exceptions.ParticleWarning`
.. |UnexpectedParticleError| replace:: :class:`~plasmapy.particles.exceptions.UnexpectedParticleError`

.. |atomic_number| replace:: :func:`~plasmapy.particles.atomic.atomic_number`
.. |atomic_symbol| replace:: :func:`~plasmapy.particles.symbols.atomic_symbol`
.. |element_name| replace:: :func:`~plasmapy.particles.symbols.element_name`
.. |half_life| replace:: :func:`~plasmapy.particles.atomic.half_life`
.. |ionic_symbol| replace:: :func:`~plasmapy.particles.symbols.ionic_symbol`
.. |is_stable| replace:: :func:`~plasmapy.particles.atomic.is_stable`
.. |isotope_symbol| replace:: :func:`~plasmapy.particles.symbols.isotope_symbol`
.. |isotopic_abundance| replace:: :func:`~plasmapy.particles.atomic.isotopic_abundance`
.. |mass_number| replace:: :func:`~plasmapy.particles.atomic.mass_number`
.. |charge_number| replace:: :func:`~plasmapy.particles.atomic.charge_number`
.. |electric_charge| replace:: :func:`~plasmapy.particles.atomic.electric_charge`
.. |standard_atomic_weight| replace:: :func:`~plasmapy.particles.atomic.standard_atomic_weight`
.. |particle_mass| replace:: :func:`~plasmapy.particles.atomic.particle_mass`
.. |particle_symbol| replace:: :func:`~plasmapy.particles.symbols.particle_symbol`
.. |known_isotopes| replace:: :func:`~plasmapy.particles.atomic.known_isotopes`
.. |common_isotopes| replace:: :func:`~plasmapy.particles.atomic.common_isotopes`
.. |reduced_mass| replace:: :func:`~plasmapy.particles.atomic.reduced_mass`
.. |stable_isotopes| replace:: :func:`~plasmapy.particles.atomic.stable_isotopes`

.. -------------------
.. plasmapy.simulation
.. -------------------

.. |ParticleTracker| replace:: :class:`~plasmapy.simulation.particletracker.ParticleTracker`

.. --------------
.. plasmapy.utils
.. --------------

.. |validate_quantities| replace:: :func:`~plasmapy.utils.decorators.validators.validate_quantities`

.. ------------------
.. NumPy replacements
.. ------------------

.. |inf| replace:: `~numpy.inf`
.. |nan| replace:: `~numpy.nan`
.. |ndarray| replace:: :class:`~numpy.ndarray`
.. |array_like| replace:: :term:`numpy:array_like`
.. |ArrayLike| replace:: `~numpy.typing.ArrayLike`
.. |DTypeLike| replace:: `~numpy.typing.DTypeLike`

.. --------------------
.. Astropy replacements
.. --------------------

.. |Quantity| replace:: :class:`~astropy.units.Quantity`
.. |Time| replace:: :class:`~astropy.time.Time`
.. |TimeDelta| replace:: :class:`~astropy.time.TimeDelta`
.. |Unit| replace:: :class:`~astropy.units.UnitBase`

.. ----------------------
.. PlasmaPy documentation
.. ----------------------

.. The backslash is needed for the substitution to work correctly when
   used just before a period.

.. |bibliography| replace:: :ref:`bibliography`\
.. |changelog guide| replace:: :ref:`changelog guide`\
.. |coding guide| replace:: :ref:`coding guide`\
.. |contributor guide| replace:: :ref:`contributor guide`\
.. |documentation guide| replace:: :ref:`documentation guide`\
.. |glossary| replace:: :ref:`glossary`\
.. |minpython| replace:: 3.8
.. |plasma-calculator| replace:: :ref:`plasmapy-calculator`\
.. |release guide| replace:: :ref:`release guide`\
.. |testing guide| replace:: :ref:`testing guide`\

.. --------------
.. Glossary terms
.. --------------

.. |annotated| replace:: :term:`annotated <annotation>`\
.. |annotation| replace:: :term:`annotation`\
.. |argument| replace:: :term:`argument`\
.. |arguments| replace:: :term:`arguments <argument>`\
.. |atom-like| replace:: :term:`atom-like`\
.. |charge number| replace:: :term:`charge number`\
.. |decorated| replace:: :term:`decorated <decorator>`\
.. |decorator| replace:: :term:`decorator`\
.. |keyword-only| replace:: :term:`keyword-only`\
.. |parameter| replace:: :term:`parameter`\
.. |parameters| replace:: :term:`parameters <parameter>`\
.. |particle-like| replace:: :term:`particle-like`\
.. |particle-list-like| replace:: :term:`particle-list-like`\

.. --------
.. Websites
.. --------

.. _Astropy docs: https://docs.astropy.org
.. _Astropy: https://www.astropy.org
.. _BibTeX format: https://www.bibtex.com/g/bibtex-format
.. _BibTeX: http://www.bibtex.org
.. _black: https://black.readthedocs.io
.. _Conda: https://docs.conda.io
.. _Contributor Covenant: https://www.contributor-covenant.org
.. _Citation File Format: https://citation-file-format.github.io/
.. _create an issue: https://github.com/PlasmaPy/PlasmaPy/issues/new/choose
.. _CSS: https://www.w3schools.com:443/css
.. _Cython: https://cython.org/
.. _DOI: https://www.doi.org
.. _equivalencies: https://docs.astropy.org/en/stable/units/equivalencies.html
.. _flake8: https://flake8.pycqa.org/en/latest
.. _git: https://git-scm.com
.. _GitHub Actions: https://docs.github.com/en/actions
.. _GitHub Discussions page: https://github.com/PlasmaPy/PlasmaPy/discussions
.. _GitHub Flavored Markdown: https://github.github.com/gfm
.. _GitHub: https://github.com
.. _Gitter bridge: https://gitter.im/PlasmaPy/Lobby
.. _Graphviz: https://graphviz.org
.. _hypothesis: https://hypothesis.readthedocs.io
.. _intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _isort: https://pycqa.github.io/isort
.. _Jinja: https://jinja.palletsprojects.com
.. _Jupyter: https://jupyter.org
.. _LaTeX: https://www.latex-project.org
.. _mailing list: https://groups.google.com/forum/#!forum/plasmapy
.. _make: https://www.gnu.org/software/make
.. _Markdown: https://www.markdownguide.org
.. _MathJax: https://www.mathjax.org
.. _matplotlib: https://matplotlib.org
.. _Matrix chat room: https://app.element.io/#/room/#plasmapy:openastronomy.org
.. _nbqa: https://nbqa.readthedocs.io
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
.. _NumPy: https://numpy.org
.. _OpenPMD: https://www.openpmd.org/
.. _pandas: https://pandas.pydata.org
.. _pip: https://pip.pypa.io
.. _Plasma Hack Week: https://hack.plasmapy.org
.. _PlasmaPy: https://www.plasmapy.org
.. _PlasmaPy meetings: https://www.plasmapy.org/meetings
.. _PlasmaPy's documentation: https://docs.plasmapy.org/en/stable
.. _PlasmaPy's GitHub repository: https://github.com/PlasmaPy/plasmapy
.. _PlasmaPy's data repository: https://github.com/PlasmaPy/PlasmaPy-data
.. _PlasmaPy's Matrix chat room: https://app.element.io/#/room/#plasmapy:openastronomy.org
.. _`pre-commit.ci`: https://pre-commit.ci
.. _pydocstyle: https://www.pydocstyle.org/en/stable
.. _pygments: https://pygments.org
.. _PyPI: https://pypi.org
.. _pytest: https://docs.pytest.org
.. _Python: https://www.python.org
.. _Python's documentation: https://docs.python.org/3
.. _Read the Docs: https://readthedocs.org
.. _reST: https://docutils.sourceforge.io/rst.html
.. _reStructuredText (reST): https://docutils.sourceforge.io/rst.html
.. _SciPy: https://scipy.org
.. _sphinx_automodapi: https://sphinx-automodapi.readthedocs.io
.. _sphinx-build: https://www.sphinx-doc.org/en/master/man/sphinx-build.html
.. _Sphinx: https://www.sphinx-doc.org
.. _suggestion box: https://docs.google.com/forms/d/e/1FAIpQLSdT3O5iHZrLJRuavFyzoR23PGy0Prfzx2SQOcwJGWtvHyT2lw/viewform?usp=sf_link
.. _towncrier: https://github.com/twisted/towncrier
.. _tox: https://tox.wiki/en/latest
.. _virtualenv: https://pypi.org/project/virtualenv
.. _Wikipedia: https://www.wikipedia.org
.. _Zenodo: https://zenodo.org

.. ----------------------
.. Nested inline literals
.. ----------------------

.. A workaround for nested inline literals so that the filename will get
   formatted like a file but will be a link. In the text, these get used
   with the syntax for a substitution followed by an underscore to
   indicate that it's for a link: |docs/_static|_

.. For these workarounds, if the replacement is something in single back
   ticks (e.g., `xarray`), then it should also be added to
   nitpick_ignore_regex in docs/conf.py so that it doesn't get counted
   as an error in a nitpicky doc build (e.g., tox -e doc_build_nitpicky).

.. _`astropy.units`: https://docs.astropy.org/en/stable/units/index.html
.. |astropy.units| replace:: `astropy.units`

.. _`CITATION.cff`: https://github.com/PlasmaPy/PlasmaPy/blob/main/CITATION.cff
.. |CITATION.cff| replace:: :file:`CITATION.cff`

.. _`docs/_static`: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs/_static
.. |docs/_static| replace:: :file:`docs/_static`

.. _`docs/_static/css`: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs/_static/css
.. |docs/_static/css| replace:: :file:`docs/_static/css`

.. _`docs/api_static`: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs/api_static
.. |docs/api_static| replace:: :file:`docs/api_static`

.. _`docs/conf.py`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/conf.py
.. |docs/conf.py| replace:: :file:`docs/conf.py`

.. _`docs/glossary.rst`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/glossary.rst
.. |docs/glossary.rst| replace:: :file:`docs/glossary.rst`

.. _`docs/common_links.rst`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/common_links.rst
.. |docs/common_links.rst| replace:: :file:`docs/common_links.rst`

.. _`docs/bibliography.bib`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/bibliography.bib
.. |docs/bibliography.bib| replace:: :file:`docs/bibliography.bib`

.. _h5py: https://www.h5py.org/
.. |h5py| replace:: `h5py`

.. _`IPython.sphinxext.ipython_console_highlighting`: https://ipython.readthedocs.io/en/stable/sphinxext.html?highlight=IPython.sphinxext.ipython_console_highlighting#ipython-sphinx-directive-module
.. |IPython.sphinxext.ipython_console_highlighting| replace:: `IPython.sphinxext.ipython_console_highlighting`

.. _lmfit: https://lmfit.github.io/lmfit-py/
.. |lmfit| replace:: `lmfit`

.. _mpmath: https://mpmath.org/doc/current/
.. |mpmath| replace:: `mpmath`

.. _nbsphinx: https://nbsphinx.readthedocs.io
.. |nbsphinx| replace:: `nbsphinx`

.. _numba: https://numba.readthedocs.io
.. |numba| replace:: `numba`

.. _pre-commit: https://pre-commit.com
.. |pre-commit| replace:: ``pre-commit``

.. _`.pre-commit-config.yaml`: https://github.com/PlasmaPy/PlasmaPy/blob/main/.pre-commit-config.yaml
.. |.pre-commit-config.yaml| replace:: :file:`.pre-commit-config.yaml`

.. _`pyproject.toml`: https://github.com/PlasmaPy/PlasmaPy/blob/main/pyproject.toml
.. |pyproject.toml| replace:: :file:`pyproject.toml`

.. _`requirements`: https://github.com/PlasmaPy/PlasmaPy/tree/main/requirements
.. |requirements| replace:: :file:`requirements`

.. _`requirements/build.txt`: https://github.com/PlasmaPy/PlasmaPy/blob/main/requirements/build.txt
.. |requirements/build.txt| replace:: :file:`requirements/build.txt`

.. _`requirements/environment.yml`: https://github.com/PlasmaPy/PlasmaPy/blob/main/requirements/environment.yml
.. |requirements/environment.yml| replace:: :file:`requirements/environment.yml`

.. _`setup.cfg`: https://github.com/PlasmaPy/PlasmaPy/blob/main/setup.cfg
.. |setup.cfg| replace:: :file:`setup.cfg`

.. _`sphinxcontrib-bibtex`: https://sphinxcontrib-bibtex.readthedocs.io
.. |sphinxcontrib-bibtex| replace:: `sphinxcontrib-bibtex`

.. _`sphinx_copybutton`: https://sphinx-copybutton.readthedocs.io
.. |sphinx_copybutton| replace:: `sphinx_copybutton`

.. _`sphinx_gallery.load_style`: https://sphinx-gallery.github.io/stable/advanced.html?highlight=load_style#using-only-sphinx-gallery-styles
.. |sphinx_gallery.load_style| replace:: `sphinx_gallery.load_style`

.. _`sphinx_changelog`: https://sphinx-changelog.readthedocs.io
.. |sphinx_changelog| replace:: `sphinx_changelog`

.. _`sphinx-reredirects`: https://documatt.gitlab.io/sphinx-reredirects
.. |sphinx-reredirects| replace:: `sphinx-reredirects`

.. _`sphinx-hoverxref`: https://sphinx-hoverxref.readthedocs.io
.. |sphinx-hoverxref| replace:: `sphinx-hoverxref`

.. _`sphinx-issues`: https://github.com/sloria/sphinx-issues
.. |sphinx-issues| replace:: `sphinx-issues`

.. _`sphinx-notfound-page`: https://sphinx-notfound-page.readthedocs.io
.. |sphinx-notfound-page| replace:: `sphinx-notfound-page`

.. _`tox.ini`: https://github.com/PlasmaPy/PlasmaPy/blob/main/tox.ini
.. |tox.ini| replace:: :file:`tox.ini`

.. _xarray: https://docs.xarray.dev
.. |xarray| replace:: `xarray`
