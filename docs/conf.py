"""
The configuration file for building PlasmaPy's documentation.

For more information, please see the following links:

PlasmaPy documentation guide:
    https://docs.plasmapy.org/en/latest/contributing/doc_guide.html

Sphinx documentation:
    https://www.sphinx-doc.org

Sphinx configuration variables:
    https://www.sphinx-doc.org/en/master/usage/configuration.html

Sphinx extensions (built-in):
    https://www.sphinx-doc.org/en/master/usage/extensions/index.html
"""

#!/usr/bin/env python3

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import logging
import os
import sys
from datetime import UTC, datetime, timezone

from sphinx.application import Sphinx

from plasmapy import __version__ as version

# isort: off
sys.path.insert(0, os.path.abspath(".."))  # noqa: PTH100
sys.path.insert(0, os.path.abspath("."))  # noqa: PTH100
# isort: on

import _author_list_from_cff
import _changelog_index
import _global_substitutions

now = datetime.now(UTC)

# Project metadata

project = "PlasmaPy"
author = "PlasmaPy Community"
copyright = f"2015–{now.year}, {author}"  # noqa: A001
language = "en"

if "dev" in version:
    # We've had some problems with setuptools_scm providing an incorrect
    # version for non-releases, so base it on the date and git hash
    # instead.
    git_hash = version.split("dev")[-1].split("+")[-1].split(".")[0]
    version = f"{now.year}.{now.month}.0.dev+{git_hash}"
    version_info_message = f"Setting {version = !r}"
    logging.info(version_info_message)

if version.startswith("0"):
    version_warning_message = f"Incorrect {version = !r}"
    logging.warning(version_warning_message)

release = version

# Define global substitutions in docs/_global_substitutions.py

_global_substitutions.make_global_substitutions_table()
global_substitutions = _global_substitutions.global_substitutions

# Regenerate the changelog index file

_changelog_index.main()

# Generate author list from CITATION.cff

_author_list_from_cff.generate_rst_file()

# Sphinx configuration variables

extensions = [
    "hoverxref.extension",
    "IPython.sphinxext.ipython_console_highlighting",
    "nbsphinx",
    "notfound.extension",
    "plasmapy_sphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.duration",
    "sphinx.ext.extlinks",
    "sphinx.ext.graphviz",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_changelog",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_gallery.load_style",
    "sphinx_issues",
    "sphinx_reredirects",
    "sphinx_tabs.tabs",
    "sphinx_collapse",
    "sphinxcontrib.bibtex",
    "sphinxemoji.sphinxemoji",
    "sphinxcontrib.globalsubs",
]

exclude_patterns = [
    "**.ipynb_checkpoints",
    "**Untitled*",
    ".DS_Store",
    ".nox",
    ".tox",
    "_build",
    "notebooks/langmuir_samples",
    "plasmapy_sphinx",
    "Thumbs.db",
]

default_role = "py:obj"
html_extra_path = ["robots.txt"]
html_favicon = "./_static/icon.ico"
modindex_common_prefix = ["plasmapy."]
pygments_style = "default"  # code highlighting to meet WCAG AA contrast standard
root_doc = "index"
source_suffix = ".rst"
templates_path = ["_templates"]
maximum_signature_line_length = 90
sphinxemoji_style = "twemoji"

suppress_warnings = [
    "autosummary.import_cycle",
]

# Specify patterns to ignore when doing a nitpicky documentation build.
# These may include common expressions like "real number" as well as
# workarounds for nested inline literals as defined in docs/common_links.py

python_role = "py:.*"

nitpick_ignore_regex = [
    # Before adding patterns for type specifications in docstrings, note
    # that information on the *meaning* of a parameter should be
    # included in the parameter description instead.
    (python_role, "and"),
    (python_role, "array .*"),
    (python_role, "array_like"),
    (python_role, "callable"),
    # for defaults that are words, numbers, particle symbols like "p+"
    (python_role, r"default: [-\+]?\w+[-\+]?\.?\d*"),
    # for defaults that are lists, tuples, sets, dictionaries, and strings
    (python_role, r"default: ((\[.*\])|(\(.*\))|(\{.*\})|(\".*\")|(\'.*\'))"),
    # for defaults that are calls like Particle("p+") or items from a dictionary
    (python_role, r"default: \w+[\.\w]*[\(\[].*[\)\]]"),
    (python_role, "dictionary.*"),
    (python_role, "function"),
    (python_role, ".*integer.*"),
    (python_role, "iterable"),
    (python_role, "key"),
    (python_role, "keyword-only"),
    (python_role, ".* object"),
    (python_role, "optional"),
    (python_role, "or"),
    (python_role, "Real"),
    (python_role, ".*real number.*"),
    (python_role, ".*representation.*"),
    (python_role, "shape.*"),
    (python_role, r"u\..*"),
    (python_role, ".*Unit.*"),
    # pytest helpers
    (python_role, "_pytest.*"),
    (python_role, "Failed"),
    # charged_particle_radiography
    (python_role, "1"),
    (python_role, "2 ints"),
    (python_role, "a single int"),
    (python_role, "same"),
    (python_role, "Tuple of 1"),
    # thomson
    (python_role, "Ne"),
    (python_role, "Ni"),
    # utils
    (python_role, "docstring of"),
    (python_role, "validation specifications"),
    # for reStructuredText workarounds to allow nested inline literals
    (python_role, "git"),
    (python_role, "h5py"),
    (python_role, "IPython.sphinxext.ipython_console_highlighting"),
    (python_role, "lmfit"),
    (python_role, "mpmath"),
    (python_role, "nbsphinx"),
    (python_role, "xarray"),
    # plasmapy_sphinx
    (python_role, "automod.*"),
    (python_role, "Builder"),
    (python_role, "docutils.*"),
    (python_role, "Documenter"),
    (python_role, "Node"),
    (python_role, "level"),
    (python_role, ".*member.*"),
    (python_role, "OptionSpec"),
    (python_role, "py"),
    (python_role, "[Ss]phinx.*"),  # also for reStructuredText workarounds
    # The following patterns still need to be fixed.
    (python_role, "json.decoder.JSONDecoder"),
    (python_role, "plasmapy.analysis.swept_langmuir.find_floating_potential"),
    (python_role, "plasmapy.particles.particle_collections"),
    (python_role, "plasmapy.utils.decorators.lite_func"),
]

# The Sphinx configuration variables rst_prolog and rst_epilog contain
# text that gets prepended or appended to all reStructuredText sources.
# These variables can be used to make global definitions; however, long
# values of these variables can greatly slow down the documentation
# build, so use them in moderation!  Use docs/_global_substitutions.py
# to define substitutions.

rst_prolog = """
.. role:: py(code)
   :language: python
"""

# html output options

html_logo = "./_static/with-text-light-190px.png"
html_static_path = ["_static"]
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "includehidden": False,
}
htmlhelp_basename = "PlasmaPydoc"

# sphinx.ext.autodoc

autoclass_content = "both"
autodoc_typehints_format = "short"

# sphinxcontrib-bibtex

bibtex_bibfiles = ["bibliography.bib"]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"
bibtex_cite_id = "{key}"

# sphinx-codeautolink

codeautolink_concat_default = True
codeautolink_warn_on_failed_resolve = False  # turn on for debugging
codeautolink_warn_on_missing_inventory = False  # turn on for debugging

# intersphinx

intersphinx_mapping = {
    "astropy": ("https://docs.astropy.org/en/stable/", None),
    "lmfit": ("https://lmfit.github.io/lmfit-py/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytest": ("https://docs.pytest.org/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "readthedocs": ("https://docs.readthedocs.io/en/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "sphinx_automodapi": (
        "https://sphinx-automodapi.readthedocs.io/en/latest/",
        None,
    ),
}

# hoverxref

hoverxref_intersphinx = list(intersphinx_mapping.keys())

hoverxref_auto_ref = True
hoverxref_domains = ["py", "cite"]
hoverxref_mathjax = True
hoverxref_roles = ["confval", "term"]
hoverxref_sphinxtabs = True
hoverxref_tooltip_maxwidth = 600  # RTD main window is 696px

hoverxref_role_types = {
    # roles with cite domain
    "p": "tooltip",
    "t": "tooltip",
    # roles with py domain
    "attr": "tooltip",
    "class": "tooltip",
    "const": "tooltip",
    "data": "tooltip",
    "exc": "tooltip",
    "func": "tooltip",
    "meth": "tooltip",
    "mod": "tooltip",
    "obj": "tooltip",
    # roles with std domain
    "confval": "tooltip",
    "hoverxref": "tooltip",
    "ref": "tooltip",
    "term": "tooltip",
}

if building_on_readthedocs := os.environ.get("READTHEDOCS"):
    # Using the proxied API endpoint is a Read the Docs strategy to
    # avoid a cross-site request forgery block for docs using a custom
    # domain. See conf.py for sphinx-hoverxref.
    use_proxied_api_endpoint = os.environ.get("PROXIED_API_ENDPOINT")
    hoverxref_api_host = "/_" if use_proxied_api_endpoint else "https://readthedocs.org"

# sphinx-issues

issues_github_path = "PlasmaPy/PlasmaPy"

# sphinx.ext.todo

todo_include_todos = False

# sphinx_reredirects

redirects = {
    "contributing/install_dev": "../contributing/getting_ready.html",
    "development": "../contributing/",
    "development/changelog_guide": "../contributing/changelog_guide.html",
    "development/code_guide": "../contributing/code_guide.html",
    "development/doc_guide": "../contributing/doc_guide.html",
    "development/index": "../contributing/index.html",
    "development/install_dev": "../contributing/getting_ready.html",
    "development/release_guide": "../contributing/release_guide.html",
    "development/testing_guide": "../contributing/testing_guide.html",
    "whatsnew": "../changelog/",
    "whatsnew/0.1.0": "../changelog/0.1.0.html",
    "whatsnew/0.1.1": "../changelog/0.1.1.html",
    "whatsnew/0.2.0": "../changelog/0.2.0.html",
    "whatsnew/0.3.1": "../changelog/0.3.1.html",
    "whatsnew/0.4.0": "../changelog/0.4.0.html",
    "whatsnew/0.5.0": "../changelog/0.5.0.html",
    "whatsnew/0.6.0": "../changelog/0.6.0.html",
    "whatsnew/0.7.0": "../changelog/0.7.0.html",
    "whatsnew/0.8.1": "../changelog/0.8.1.html",
    "whatsnew/0.9.0": "../changelog/0.9.0.html",
    "whatsnew/0.9.1": "../changelog/0.9.1.html",
    "whatsnew/2023.1.0": "../changelog/2023.1.0.html",
    "whatsnew/index": "../changelog/index.html",
}

# Using sphinx.ext.extlinks lets us simplify the process of creating
# links to commonly used external sites. The key of the extlink becomes
# a new role, and the corresponding tuple contains the base url and the
# caption. For example, we can now do :orcid:`0000-0000-0000-0000` and
# have a link create to the corresponding ORCID page. New roles should
# be added to rst-roles in tox.ini to avoid being caught by
# flake8-rst-docstrings.

extlinks = {
    "orcid": ("https://orcid.org/%s", "%s"),
    "wikipedia": ("https://en.wikipedia.org/wiki/%s", "%s"),
}

# Settings for checking hyperlinks with `make linkcheck`. For a primer
# on regular expressions, see: https://docs.python.org/3/library/re.html

linkcheck_anchors = True
linkcheck_anchors_ignore = [
    "/room",
    r".+openastronomy.+",
    "L[0-9].+",
    "!forum/plasmapy",
]

# The default value of linkcheck_report_timeouts_as_broken will default
# to False for Sphinx 8.0, so the following line can be removed.
linkcheck_report_timeouts_as_broken = False

linkcheck_allowed_redirects = {
    r"https://doi\.org/.+": r"https://.+",  # DOI links are persistent
    r"https://.+\.org": r"https://.+\.org/en/.+",
    r"https://docs.+\.io": r"https://docs.+\.io/en/.+",
    r"https://docs.+\.com": r"https://docs.+\.com/en/.+",
    r"https://.+\.codes": r"https://.+\.codes/en/.+",
    r"https://docs.+\.dev": r"https://docs.+\.dev/en/.+",
    r"https://github\.com/sponsors/.+": r"https://github\.com/.+",
    # Allow :issue: role from sphinx-issues to point to GitHub discussions
    r"https://github\.com/.+/issues/.+": r"https://github\.com/.+/discussions/.+",
    r"https://en.wikipedia.org/wiki.+": "https://en.wikipedia.org/wiki.+",
    r"https://.+\.readthedocs\.io": r"https://.+\.readthedocs\.io/en/.+",
    r"https://www\.sphinx-doc\.org": r"https://www\.sphinx-doc\.org/en/.+",
    r"https://.+/github\.io": r"https://.+/github\.io/en/.+",
    r"https://.+": r".+(google|github).+[lL]ogin.+",  # some links require logins
    r"https://jinja\.palletsprojects\.com": r"https://jinja\.palletsprojects\.com/.+",
    r"https://pip\.pypa\.io": r"https://pip\.pypa\.io/en/.+",
    r"https://www\.python\.org/dev/peps/pep.+": r"https://peps\.python\.org/pep.+",
    r"https://matplotlib.org/stable/devel.*": r"https://matplotlib.org/devdocs/devel/.*",
}

# Hyperlinks for `make linkcheck` to ignore. This may include stable
# links (like DOIs), links prone to 403 errors, links requiring a login,
# or links that require you to verify that you are a human.

linkcheck_ignore = [
    r"https://agupubs\.onlinelibrary\.wiley\.com/doi/10\.1029/2012JA017856",
    r"https://doi\.org/10\.1007/978-3-319-22309-4",
    r"https://doi\.org/10\.1007/978-3-319-24121-0",
    r"https://doi\.org/10\.1007/978-3-319-67711-8.*",
    r"https://doi\.org/10\.1007/s11207-014-0526-6",
    r"https://doi\.org/10\.1007/s41116-019-0021-0",
    r"https://doi\.org/10\.1016/0032-0633(94)00197-Y",
    r"https://doi\.org/10\.1016/0032-0633\(94\)00197-Y",
    r"https://doi\.org/10\.1016/b978-0-12-374877-5\.00003-8",
    r"https://doi\.org/10\.1016/c2009-0-20048-1",
    r"https://doi\.org/10\.1016/c2013-0-12176-9",
    r"https://doi\.org/10\.1016/j\.physleta\.2004\.08\.021",
    r"https://doi\.org/10\.1029/95ja03712",
    r"https://doi\.org/10\.1029/1998ja900132",
    r"https://doi\.org/10\.1029/2011ja016674",
    r"https://doi\.org/10\.1029/2012ja017856",
    r"https://doi\.org/10\.1029/9503712",
    r"https://doi\.org/10\.1038/150405d0",
    r"https://doi\.org/10\.1063/1\.865901",
    r"https://doi\.org/10\.1063/1\.871810",
    r"https://doi\.org/10\.1063/1\.1706052",
    r"https://doi\.org/10\.1063/1\.2756751",
    r"https://doi\.org/10\.1063/1\.4775777",
    r"https://doi\.org/10\.1063/1\.4801022",
    r"https://doi\.org/10\.1086/523671",
    r"https://doi\.org/10\.1088/0004-637X/751/1/20",
    r"https://doi\.org/10\.1088/0368-3281/5/2/304",
    r"https://doi\.org/10\.1103/PhysRev\.89\.977",
    r"https://doi\.org/10\.1103/PhysRevE\.65\.036418",
    r"https://doi\.org/10\.1103/physrevlett\.111\.241101",
    r"https://doi\.org/10\.1140/epjd/s10053-021-00305-2",
    r"https://doi\.org/10\.1146/annurev-astro-082708-101726",
    r"https://doi\.org/10\.1201/9781315275048",
    r"https://doi\.org/10\.1371/journal\.pbio\.1001745",
    r"https://doi\.org/10\.1371/journal\.pcbi\.1005510",
    r"https://doi\.org/10\.2172/5259641",
    r"https://doi\.org/10\.3847/1538-4357/accc32",
    r"https://doi\.org/10\.5281/zenodo\.1436011",
    r"https://doi\.org/10\.5281/zenodo\.1460977",
    r"https://doi\.org/10\.5281/zenodo\.3406803",
    r"https://doi\.org/10\.5281/zenodo\.3766933",
    r"https://doi\.org/10\.5281/zenodo\.4602818",
    r"https://doi\.org/10\.5281/zenodo\.7734998",
    r"https://doi\.org/10\.5281/zenodo\.8015753",
    r"https://doi\.org/10\.18434/T4NC7P",
    r"https://github\.com/PlasmaPy/PlasmaPy/settings/secrets/actions",
    r"https://orcid\.org/0000-0001-5050-6606",
    r"https://orcid\.org/0000-0001-5270-7487",
    r"https://orcid\.org/0000-0001-5308-6870",
    r"https://orcid\.org/0000-0001-5394-9445",
    r"https://orcid\.org/0000-0001-6079-8307",
    r"https://orcid\.org/0000-0001-6291-8843",
    r"https://orcid\.org/0000-0001-6628-8033",
    r"https://orcid\.org/0000-0001-6849-3612",
    r"https://orcid\.org/0000-0001-7381-1996",
    r"https://orcid\.org/0000-0001-7959-8495",
    r"https://orcid\.org/0000-0001-8358-0482",
    r"https://orcid\.org/0000-0001-8745-204X",
    r"https://orcid\.org/0000-0002-0486-1292",
    r"https://orcid\.org/0000-0002-0762-3708",
    r"https://orcid\.org/0000-0002-1073-6383",
    r"https://orcid\.org/0000-0002-1192-2057",
    r"https://orcid\.org/0000-0002-1365-1908",
    r"https://orcid\.org/0000-0002-1444-9680",
    r"https://orcid\.org/0000-0002-1984-7303",
    r"https://orcid\.org/0000-0002-2105-0280",
    r"https://orcid\.org/0000-0002-2160-7288",
    r"https://orcid\.org/0000-0002-2373-8927",
    r"https://orcid\.org/0000-0002-3056-6334",
    r"https://orcid\.org/0000-0002-3713-6337",
    r"https://orcid\.org/0000-0002-4227-2544",
    r"https://orcid\.org/0000-0002-4237-2211",
    r"https://orcid\.org/0000-0002-4914-6612",
    r"https://orcid\.org/0000-0002-5598-046X",
    r"https://orcid\.org/0000-0002-5978-6840",
    r"https://orcid\.org/0000-0002-6468-5710",
    r"https://orcid\.org/0000-0002-7616-0946",
    r"https://orcid\.org/0000-0002-7757-5879",
    r"https://orcid\.org/0000-0002-7860-9567",
    r"https://orcid\.org/0000-0002-8078-214X",
    r"https://orcid\.org/0000-0002-8335-1441",
    r"https://orcid\.org/0000-0002-8475-8606",
    r"https://orcid\.org/0000-0002-8644-8118",
    r"https://orcid\.org/0000-0002-8676-1710",
    r"https://orcid\.org/0000-0002-9180-6565",
    r"https://orcid\.org/0000-0002-9258-4490",
    r"https://orcid\.org/0000-0003-0079-4114",
    r"https://orcid\.org/0000-0003-0223-7004",
    r"https://orcid\.org/0000-0003-0602-8381",
    r"https://orcid\.org/0000-0003-1439-4218",
    r"https://orcid\.org/0000-0003-2528-8752",
    r"https://orcid\.org/0000-0003-2892-6924",
    r"https://orcid\.org/0000-0003-2944-0424",
    r"https://orcid\.org/0000-0003-3309-3939",
    r"https://orcid\.org/0000-0003-3530-7910",
    r"https://orcid\.org/0000-0003-4217-4642",
    r"https://orcid\.org/0000-0003-4230-6916",
    r"https://orcid\.org/0000-0003-4397-027X",
    r"https://orcid\.org/0000-0003-4739-1152",
    r"https://orcid\.org/0009-0000-3029-8619",
    r"https://orcid\.org/0009-0002-5918-4652",
    r"https://orcid\.org/0009-0003-3159-0541",
    r"https://orcid\.org/0009-0004-6699-4869",
    r"https://orcid\.org/0009-0006-0863-0180",
    r"https://orcid\.org/0009-0007-0655-1347",
    r"https://orcid\.org/0009-0008-3588-0497",
    r"https://orcid\.org/0009-0008-5134-6171",
    r"https://orcid\.org/0009-0009-9490-5284",
    r"https://hdl\.handle\.net/10037/29416",
    r"https://www\.iter\.org/",
    r"https://www\.sciencedirect\.com/book/9780123748775/.*",
]

# nbsphinx

nbsphinx_thumbnails = {
    "notebooks/*": "_static/graphic-circular.png",
    "notebooks/*/*": "_static/graphic-circular.png",
    "notebooks/diagnostics/langmuir_analysis": (
        "_static/notebook_images/langmuir_analysis.png"
    ),
    "notebooks/formulary/magnetosphere": (
        "_static/notebook_images/mms.png"
    ),  # public domain
    "notebooks/getting_started/units": (
        "_static/notebook_images/astropy_logo_notext.png"
    ),  # CC BY-SA
    "notebooks/formulary/solar_plasma_beta": "_static/notebook_images/coronal_loops.png",
    "notebooks/plasma/grids_cartesian": (
        "_static/notebook_images/uniform_grid_thumbnail.png"
    ),
    "notebooks/plasma/grids_nonuniform": (
        "_static/notebook_images/nonuniform_grid_thumbnail.png"
    ),
}

# adapted from
# https://github.com/spatialaudio/nbsphinx/blob/58b8034dd9d7349c1b4ac3e7a7d6baa87ab2a6a9/doc/conf.py

# This is processed by Jinja2 and inserted before each notebook
nbsphinx_prolog = r"""
{% set docname = 'docs/' + env.doc2path(env.docname, base=None) %}
{% set nb_base = 'tree' if env.config.revision else 'blob' %}
{% set nb_where = env.config.revision if env.config.revision else 'main' %}

.. raw:: html

    <div class="admonition note">
      <p style="margin-bottom:0px">
        This page was generated by
        <a href="https://nbsphinx.readthedocs.io/">nbsphinx</a> from
        <a class="reference external" href="https://github.com/PlasmaPy/PlasmaPy/{{ nb_base|e }}/{{ nb_where|e }}/{{ docname|e }}">{{ docname|e }}</a>.
        <br>
        Interactive online version:
        <a href="https://mybinder.org/v2/gh/PlasmaPy/PlasmaPy/{{ nb_where|e }}/?filepath={{ docname|e }}"><img alt="Binder badge" src="https://mybinder.org/badge_logo.svg" style="vertical-align:text-bottom"></a>.
      </p>
    </div>

.. raw:: latex

    \nbsphinxstartnotebook{\scriptsize\noindent\strut
    \textcolor{gray}{The following section was generated from
    \sphinxcode{\sphinxupquote{\strut {{ docname | escape_latex }}}} \dotfill}}
"""

# plasmapy_sphinx settings

autosummary_generate = True
automodapi_custom_groups = {
    "aliases": {
        "title": "Aliases",
        "description": (
            """
            PlasmaPy provides :term:`aliases` of the most common plasma
            functionality for user convenience. Aliases in PlasmaPy are
            denoted with a trailing underscore (e.g., ``alias_``). For
            further details, please refer to the :ref:`contributor
            guide's section on aliases <aliases>`.
            """
        ),
        "dunder": "__aliases__",
    },
    "lite-functions": {
        "title": "Lite-Functions",
        "description": (
            """
            :term:`Lite-functions` are optimized versions of existing
            `plasmapy` functions that are intended for applications where
            computational efficiency matters most. Lite-functions accept
            numbers and NumPy arrays that are implicitly assumed to be
            in SI units, and do not accept |Quantity| objects as inputs.
            For further details, please refer to the :ref:`contributor
            guide's section on lite-functions <lite-functions>`.

            .. caution::

               Lite-functions do not include the safeguards that are
               included in most `plasmapy.formulary` functions. When
               using lite-functions, it is vital to double-check your
               implementation!
            """
        ),
        "dunder": "__lite_funcs__",
    },
}

automodapi_group_order = (
    "modules",
    "classes",
    "exceptions",
    "warnings",
    "functions",
    "aliases",
    "lite-functions",
    "variables",
)

# following https://about.readthedocs.com/blog/2024/07/addons-by-default/
# Define the canonical URL if you are using a custom domain on Read the Docs
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")

# Tell Jinja2 templates the build is running on Read the Docs
if os.environ.get("READTHEDOCS", "") == "True":
    if "html_context" not in globals():
        html_context = {}
    html_context["READTHEDOCS"] = True


def setup(app: Sphinx) -> None:
    app.add_config_value("revision", "", rebuild=True)
    app.add_css_file("css/admonition_color_contrast.css")
    app.add_css_file("css/plasmapy.css", priority=600)
