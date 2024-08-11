.. _documentation guide:

*******************
Documentation Guide
*******************

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

.. Define roles for in-line code formatting with pygments

.. role:: bash(code)
   :language: bash

.. role:: rest(code)
   :language: rst

.. Use :rest: to avoid conflicting with :rst:role: and :rst:dir:.
   Escape backticks with a \ in arguments provided to :rest: (like
   :rest:`:math:\`5+6\``)

Introduction
============

Documentation that is up-to-date and understandable is vital to the
health of a software project. |:books:| This page describes the
documentation requirements and guidelines to be followed during the
development of PlasmaPy and affiliated packages.

.. tip::

   Updating documentation is one of the best ways to make a first
   contribution to an open source software project. |:beginner:|

.. note::

   If you discover areas within |PlasmaPy's documentation| that are
   confusing or incomplete, please `raise an issue`_! This really helps
   PlasmaPy not only by helping us improve the documentation for all,
   but also by creating opportunities for new contributors to make their
   first contribution to the project.

|PlasmaPy's documentation| is hosted by |Read the Docs| and is
available at these locations:

* The documentation corresponding to the most recent release to |PyPI|
  is labeled ``stable`` and is found at https://docs.plasmapy.org or
  https://docs.plasmapy.org/en/stable.

* The documentation corresponding to the ongoing development on the
  ``main`` branch in |PlasmaPy's GitHub repository|, which is often
  ahead of the most recent release, is labeled ``latest`` and can be
  found at https://docs.plasmapy.org/en/latest.
  |:building_construction:|

.. tip::

  A preview of the documentation is generated every time a pull request
  is created or updated. You can access this preview by scrolling down
  to the checks at the bottom of a pull request, and clicking on
  :guilabel:`Details` next to :guilabel:`docs/readthedocs.org:plasmapy`.

.. image:: ../_static/contributor_guide/readthedocs_preview_link.png
   :width: 700
   :align: center
   :alt: Access to the preview of the documentation after a pull request

Markup Languages
================

ReStructuredText
----------------

PlasmaPy's documentation is written using the |reStructuredText| markup
language. |reStructuredText| is human readable when viewed within a
source code file or when printed out using `help`. |reStructuredText|
also contains markup that allows the text to be transformed into
|PlasmaPy's documentation|.  |reStructuredText| files use the file
extension :file:`.rst`. Documentation contained within :file:`.py` files
are in the form of :wikipedia:`docstrings <docstring>`, which are
written in |reStructuredText|.

ReStructuredText Examples
~~~~~~~~~~~~~~~~~~~~~~~~~

Here we show some examples of commonly used |reStructuredText| syntax in
PlasmaPy.  Please refer to the documentation for |Sphinx| and
|reStructuredText| for a list of available |roles| and |directives|.

This is an example of including headings for the document title,
sections, subsections, and so on. The lines surrounding each heading are
the same length as that heading.

.. code-block:: rst

   ==============
   Document title
   ==============

   Heading 1
   =========

   Heading 2
   ---------

   Heading 3
   ~~~~~~~~~

We can link to code objects by enclosing them in single backticks. This
linking will work for |Python| objects as well as certain packages like
|NumPy|, |SciPy|, |Astropy|, and |pandas|. This linking is described in the
section on :ref:`external-references`. In-line code samples are
typically enclosed in double backticks. To get inline code highlighting,
use the :rest:`:py:` role for Python code.

.. code-block:: rst

   Here `plasmapy.particles` provides a linked reference to the
   module's documentation.

   Adding a tilde at the beginning `~plasmapy.particles` still
   provides a linked reference to the associated documentation
   but shortens the display so only "particles" is displayed.

   Double backticks are used to show inline code that is not
   cross-referenced: ``plasmapy.particles``.

   The ``:py:`` role can be used for inline code highlighting:
   :py:`import astropy.units as u`.

This |reStructuredText| block renders as:

   Here `plasmapy.particles` provides a linked reference to the
   module's documentation.

   Adding a tilde at the beginning `~plasmapy.particles` still
   provides a linked reference to the associated documentation
   but shortens the display so only "particles" is displayed.

   Double backticks are used to show inline code that is not
   cross-referenced: ``plasmapy.particles``.

   The ``:py:`` role can be used for inline code highlighting:
   :py:`import astropy.units as u`.

|Sphinx| can format code blocks for |Python| and the |Python| console
using the :rst:dir:`code-block` :term:`directive`.

   .. code-block:: rst

      .. code-block:: python

         def sample_function():
             return 42

      .. code-block:: pycon

         >>> print(6 * 9)
         54

This |reStructuredText| block renders as:

   .. code-block:: python

      def sample_function():
          return 42

   .. code-block:: pycon

      >>> print(6 * 9)
      54

Here are some examples for linking to websites.

.. code-block:: rst

   `PlasmaPy Enhancement Proposals <https://github.com/PlasmaPy/PlasmaPy-PLEPs>`_
   are used to propose major changes to PlasmaPy.

   `Write the Docs`_ has a guide_ on writing software documentation.

   .. _`Write the Docs`: https://www.writethedocs.org
   .. _guide: https://www.writethedocs.org/

This |reStructuredText| block renders as:

   `PlasmaPy Enhancement Proposals <https://github.com/PlasmaPy/PlasmaPy-PLEPs>`_
   are used to propose major changes to PlasmaPy.

   `Write the Docs`_ has a guide_ on writing software documentation.

   .. _`Write the Docs`: https://www.writethedocs.org/
   .. _guide: https://www.writethedocs.org/

Displayed math may be created using the :rst:dir:`math`
:term:`directive` using LaTeX_ syntax.

.. code-block:: rst

   .. math::

      \alpha = \beta + \gamma

This |reStructuredText| block renders as:

   .. math::

      \alpha = \beta + \gamma

Math can be in-line using the :rst:role:`math` |role|.

.. code-block:: rst

   An example of in-line math is :math:`x`. Using Unicode characters
   like :math:`α + β + γ` makes math easier to read in the source code.

This |reStructuredText| block renders as:

   An example of in-line math is :math:`x`. Using Unicode characters
   like :math:`α + β + γ` makes math easier to read in the source code.

Markdown
--------

A few of PlasmaPy's files are written using Markdown_, such as README
files and licenses from other packages. Markdown_ is simpler but more
limited than |reStructuredText|. Markdown_ files use the file extension
:file:`.md`. Posts on GitHub are written in `GitHub Flavored
Markdown`_. The following code block contains a few common examples of
Markdown_ formatting.

.. code-block:: markdown

   # Header 1

   ## Header 2

   Here is a link to [PlasmaPy's documentation](https://docs.plasmapy.org).

   We can make text **bold** or *italic*.

   We can write in-line code like `x = 1` or create a Python code block:

   ```Python
   y = 2
   z = 3
   ```

.. _writing-documentation:

Writing Documentation
=====================

Docstrings
----------

A :wikipedia:`docstring` is a comment at the beginning of a function or
another object that provides information on how to use that function
(see :pep:`257`). Docstrings are designated by surrounding the content
with triple quotes :py:`"""This is my docstring."""`.

In order to improve readability and maintain consistency, PlasmaPy uses
the numpydoc_ standard for docstrings. Docstring conventions for
|Python| are more generally described in :pep:`257`.

.. tip::

   If a docstring contains math that utilizes LaTeX_ syntax, begin the
   docstring with :py:`r"""` instead of :py:`"""`.

   In a normal string, backslashes are used to begin escape sequences,
   and a single backslash needs to be represented with ``\\``. This
   complication is avoided by beginning the docstring with :py:`r"""`,
   which denotes the docstring as a `raw string`_. For example, the `raw
   string`_  ``r""":math:`\alpha`"""`` will render the same as the
   normal string ``""":math:`\\alpha`"""``.

.. _example docstring:

Example docstring
~~~~~~~~~~~~~~~~~

Here is an example docstring in the numpydoc_ format:

.. code-block:: python
   :caption: Example docstring

   import warnings

   import numpy as np


   def subtract(a, b, *, switch_order=False):
       r"""
       Compute the difference between two integers.

       Add ∼1–3 sentences here for an extended summary of what the function
       does. This extended summary is a good place to briefly define the
       quantity that is being returned.

       .. math::

           f(a, b) = a - b

       Parameters
       ----------
       a : `float`
           The number from which ``b`` will be subtracted.

       b : `float`
           The number being subtracted from ``a``.

       switch_order : `bool`, |keyword-only|, default: `True`
           If `True`, return :math:`a - b`. If `False`, then return
           :math:`b - a`.

       Returns
       -------
       float
           The difference between ``a`` and ``b``.

       Raises
       ------
       `ValueError`
           If ``a`` or ``b`` is `~numpy.inf`.

       Warns
       -----
       `UserWarning`
           If ``a`` or ``b`` is `~numpy.nan`.

       See Also
       --------
       add : Add two numbers.

       Notes
       -----
       The "Notes" section provides extra information that cannot fit in the
       extended summary near the beginning of the docstring. This section
       should include a discussion of the physics behind a particular concept
       that should be understandable to someone who is taking their first
       plasma physics class. This section can include a derivation of the
       quantity being calculated or a description of a particular algorithm.

       Examples
       --------
       Include a few example usages of the function here. Start with simple
       examples and then increase complexity when necessary.

       >>> from package.subpackage.module import subtract
       >>> subtract(9, 6)
       3

       Here is an example of a multi-line function call.

       >>> subtract(
       ...     9, 6, switch_order=True,
       ... )
       -3

       PlasmaPy's test suite will check that these commands provide the output
       that follows each function call.
       """
       if np.isinf(a) or np.isinf(b):
           raise ValueError("Cannot perform subtraction operations involving infinity.")

       warnings.warn("The `subtract` function encountered a nan value.", UserWarning)

       return b - a if switch_order else a - b

Template docstring
~~~~~~~~~~~~~~~~~~

This template docstring may be copied into new functions. Usually only
some of the sections will be necessary for a particular function, and
unnecessary sections should be deleted. Any sections that are included
should be in the order provided.

.. code-block:: python
  :caption: Docstring template

  def sample_function():
      r"""
      Compute ...

      Parameters
      ----------

      Returns
      -------

      Raises
      ------

      Warns
      -----

      See Also
      --------

      Notes
      -----

      References
      ----------

      Examples
      --------

      """

Doctests
~~~~~~~~

PlasmaPy's test suite runs code examples in docstrings to verify that
the expected output in the docstring matches the actual output from
running the code. These doctests_ verify that docstring examples
faithfully represent the behavior of the code.

.. code-block:: python

   def double(x):
       """
       >>> double(4)  # this line is tested that it matches the output below
       8
       """
       return 2 * x

An ellipsis (:py:`...`) denotes that the actual and expected outputs
should only be compared to the available precision. This capability is
needed for functions in `plasmapy.formulary` that depend on fundamental
constants that are occasionally revised.

.. code-block:: python

   def f():
       """
       >>> import numpy as np
       >>> np.pi
       3.14159...
       >>> np.pi ** 100
       5.187...e+49
       """

To skip the execution of a line of code in a docstring during tests, end
the line with :py:`# doctest: +SKIP`. This is appropriate for lines
where the output varies or an exception is raised.

.. code-block:: python

   def g():
       """
       >>> import random
       >>> random.random()  # doctest: +SKIP
       0.8905444
       >>> raise ValueError  # doctest: +SKIP
       """

.. _rst-roles:

Roles
~~~~~

The following are |roles| that are customized for PlasmaPy or come from
various |Sphinx| extensions.

.. rst:role:: commit

   Links to a GitHub commit.

.. rst:role:: issue

   Links to a GitHub issue.

.. rst:role:: orcid

   Links to an ORCID account.

.. rst:role:: pr

   Links to a GitHub pull request.

.. rst:role:: py

   Used for in-line Python code formatting. Defined using
   :confval:`rst_prolog` in |docs/conf.py|_.

.. rst:role:: sub

   For subscripts.

.. rst:role:: sup

   For superscripts.

.. rst:role:: user

   Links to a GitHub username.

.. rst:role:: wikipedia

   Links to a Wikipedia article.

.. _doc-guide-notebooks:

Example notebooks
-----------------

PlasmaPy's documentation includes an |example gallery| of |Jupyter|
notebooks. The notebooks are located in |docs/notebooks|_. The example
gallery is built using |nbsphinx|.

To add a notebook to the |example gallery|, include the :file:`.ipynb`
file in the appropriate subdirectory of |docs/notebooks|_. The notebook
must be included in an ``nbgallery`` directive in |docs/examples.rst|_
(though this contains wildcards for most existing directories, so no
additional step may be required).

Before adding the Jupyter notebook, open it and run
:menuselection:`Kernel --> Restart Kernel and Clear Outputs of All Cells`
and then :menuselection:`File --> Save Notebook` before committing the
and pushing the change. Doing this will signal |nbsphinx| that it should
execute the notebook during the documentation build. However, if the
notebook is computationally intensive, instead pre-execute it with
:menuselection:`Kernel --> Restart Kernel and Run All Cells` before
doing ``git commit``. Doing this will reduce documentation build times.

If a notebook makes use of a Python package that is not included in
PlasmaPy's requirements, included in the ``docs`` dependencies set as
defined in |pyproject.toml|_.

There are two methods for using markup in non-code cells in a Jupyter
notebook in the gallery:

- In "raw" cells, we can make use of |reStructuredText| formatting and
  thus enable linking to code objects.

- In Markdown_ cells, we can make use of Markdown formatting. However,
  to link to a code object, we must include a link to the actual
  documentation page. Be sure to link to the ``stable`` version of the
  documentation (as opposed to ``latest``), even if the link does not
  exist on ``stable`` yet. For example, a link to `plasmapy.formulary`
  can be accomplished with

  .. code-block:: markdown

     [`plasmapy.formulary`](https://docs.plasmapy.org/en/stable/formulary/index.html)

If a code cell in a notebook intentionally raises an exception, add a
metadata tag entitled "raises-exception" to that cell.

Glossary definitions
--------------------

Define important terms in PlasmaPy's :ref:`glossary`, which is located
at |docs/glossary.rst|_. Here is an example of a term defined within the
:rst:dir:`glossary` |directive|.

.. code-block:: rst

   .. glossary::

      kwargs
         An abbreviation for keyword arguments.

Using the :rst:role:`term` |role| allows us to link to the definitions
of terms. Using :rest:`:term:\`kwargs\`` will link to :term:`kwargs` in
the :ref:`glossary`. We can also refer to terms defined in the projects
connected via |intersphinx| if they have not already been defined in
PlasmaPy's :ref:`glossary`. Using :rest:`:term:\`role\`` will link to
|role| and :rest:`:term:\`directive\`` will link to |directive| in
`Sphinx's glossary`_.

Documentation guidelines
========================

This section contains guidelines and best practices for writing
documentation for PlasmaPy and affiliated packages.

* Write documentation to be understandable to students taking their
  first course or beginning their first research project in plasma
  science. Include highly technical information only when necessary.

* Use technical jargon sparingly. Define technical jargon when
  necessary.

* Use the :wikipedia:`active voice <active_voice>` in the present tense.

* Keep the documentation style consistent within a file or module, and
  preferably across all of |PlasmaPy's documentation|.

* Update code and corresponding documentation at the same time.

* Write sentences that are simple, concise, and direct rather than
  complicated, vague, or ambiguous. Prefer sentences with ≲ 20 words.

* Avoid idioms, metaphors, and references that are specific to a
  particular culture.

* Many words and software packages have more than one common spelling or
  acronym. Use the spelling that is used in the file you are modifying,
  which is preferably the spelling used throughout
  |PlasmaPy's documentation|.

  * More generally, it is preferable to use the spelling that is used in
    |Python's documentation| or the spelling that is used most commonly.

  * Represent names and acronyms for a software package or language as
    they are represented in the documentation for each project. Common
    examples include "Python", "Astropy", and "NumPy", and
    "reStructuredTest".

* When referencing PlasmaPy functionality, write the full namespace path
  to where the functionality is defined, not where it is conveniently
  accessed. For example, write
  ```~plasmapy.formulary.speeds.Alfven_speed``` rather than
  ```~plasmapy.formulary.Alfven_speed```.

  This does not necessarily need to be done when referencing external
  packages, since each package may have their own standard. For example,
  Astropy's |Quantity| class is defined in
  ```astropy.units.quantity.Quantity``` but is also indexed at
  ```~astropy.units.Quantity``` so either option will link to the same
  documentation.

* For readability, limit documentation line lengths to ≲ 72 characters.
  Longer line lengths may be used when necessary (e.g., for hyperlinks).

  .. note::

     Studies typically show that line lengths of 50–75 characters are
     optimal for readability. |:page_facing_up:|

* Use indentations of 3 spaces for |reStructuredText| blocks.

* Store images within the |docs/_static|_ directory, except for images
  that are generated during the |Sphinx| build. |:frame_photo:| The
  |docs/_static|_ directory contains files that are used for the online
  documentation but are not generated during the |Sphinx| build.

* Avoid linking to websites that might disappear due to :wikipedia:`link
  rot <link_rot>` such as documents hosted on personal websites.

  * When including references, use a link that includes a
    :wikipedia:`persistent identifier <persistent_identifier>` such as a
    digital object identifier (|DOI|) when one is available (e.g.,
    https://doi.org/10.5281/zenodo.4602818\ ). |:link:|

  * Wikipedia_ articles may be linked to when they contain a
    well-developed and accurate description of a concept.

* Include both the original references for a topic as well as accessible
  pedagogical references. Prefer references that are open access over
  references that require purchase of a subscription or are behind a
  :wikipedia:`paywall`.

.. note::

   Emphasize important points with admonitions_ like this one.

* Start the names of all physical units with a lower case letter, except
  at the beginning of a sentence and for "degree Celsius".
  |:thermometer:|

* Physical unit symbols should not be formatted as math. If units are
  needed inside a math block, use LaTeX_'s ``\text`` command as in the
  example below. The backslash followed by a space is needed to have a
  space between the number and the units.

  .. code-block:: rst

     The speed of light is approximately :math:`3 × 10^8` m/s or

     .. math::

        3 × 10^{10}\ \text{cm/s}

  This |reStructuredText| block renders as:

     The speed of light is approximately :math:`3 × 10^8` m/s or

     .. math::

        3 × 10^{10}\ \text{cm/s}

* The names of chemical elements are lower case, except at the beginning
  of a sentence.

* Particle and chemical symbols should be formatted as regular text. Use
  :rst:role:`sub` for subscripts and :rst:role:`sup` for
  superscripts.

  Because interpreted text must normally be surrounded by whitespace or
  punctuation, use a backslash followed by a space for the interpreted
  text to show up immediately next to the regular text. This is not
  necessary before a period or comma.

  .. code-block:: rst

     The symbol for helium is He.

     The symbol for an electron is e\ :sup:`-`.

     An alpha particle may be represented as :sup:`4`\ He\ :sup:`1+`.

  This |reStructuredText| block renders as:

     The symbol for helium is He.

     The symbol for an electron is e\ :sup:`-`.

     An alpha particle may be represented as :sup:`4`\ He\ :sup:`1+`.

* Begin each :file:`.py` file with a docstring that provides a
  high-level overview of what is contained in that module.

* Place the ``__all__`` dunder immediately after the docstring that
  begins a module and before the import statements (but after any
  :py:`from __future__` imports that must be at the beginning of a
  file). This dunder should be a `list` that contains the names of all
  objects in that module intended for use by users. Private objects
  (i.e., objects with names that begin with an underscore) should not be
  included in ``__all__``.  ``__all__`` is a leftover from the now
  dissuaded practice of star imports (e.g.,
  :samp:`from {package} import *`\ ), but is still used by |Sphinx| for
  selecting objects to document. Only objects contained within
  ``__all__`` will show up in the online documentation.

Docstring guidelines
--------------------

* All functions, classes, and objects that are part of the public
  :wikipedia:`API` must have a docstring that follows the numpydoc_
  standard. Refer to the numpydoc_ standard for how to write docstrings
  for classes, class attributes, and constants.

* The short summary statement at the beginning of a docstring should be
  one line long, but may be longer if necessary.

* The extended summary that immediately follows the short summary should
  be ≲ 4 sentences long. Any additional information should included in
  the "Notes" section.

* Put any necessary highly technical information in the "Notes" section
  of a docstring.

* The short summary should start on the line immediately following the
  triple quotes. There should not be any blank lines immediately before
  the closing triple quotes.

* The first line of the docstring for a function or method should begin
  with a word like "Calculate" or "Compute" and end with a period.

* The first line of an object that is not callable (for example, an
  attribute of a class decorated with `property`) should not begin with
  a verb and should end with a period.

* Keep the docstring indented at the same level as the :py:`r"""` or
  :py:`"""` that begins the docstring, except for |reStructuredText|
  constructs like lists, math, and code blocks. Use an indentation of
  four spaces more than the declaration of the object.

  .. code-block:: python

     def f():
         """This is indented four spaces relative to the `def` statement."""

* The first sentence of a docstring of a function should include a
  concise definition of the quantity being calculated, as in the
  following example.

  .. code-block:: python

     def beta(T, n, B):
         """Compute the ratio of thermal pressure to magnetic pressure."""

  When the definition of the quantity being calculated is unable to fit
  on ∼1–2 lines, include the definition in the extended summary instead.

  .. code-block:: python

     def beta(T, n, B):
         """
         Compute plasma beta.

         Plasma beta is the ratio of thermal pressure to magnetic pressure.
         """

* When a function calculates a formula, put the formula in the extended
  summary section when it can be included concisely. Put complicated
  formulae, derivations, and extensive discussions of physics or math in
  the "Notes" section.

* Private code objects (e.g., code objects that begin with a single
  underscore, like ``_private_object``) should have docstrings. A
  docstring for a private code object may be a single line, and
  otherwise should be in numpydoc_ format.

* Docstrings for private code objects do not get rendered in the online
  documentation, and should be intended for contributors.

Parameters
~~~~~~~~~~

Describe each :term:`parameter` in the "Parameters_" section of the
docstring using the following format:

.. code-block:: rst

   parameter_name : type specification
       Parameter description.

Some examples are:

.. code-block:: rst

   x : `float`
       Description of ``x``.

   y : `int`
       Description of ``y``.

   settings : `dict` of `str` to `int`
       Description of ``settings``.

Type specifications
^^^^^^^^^^^^^^^^^^^

The **type specification** may include:

* Size and/or shape information
* Type information
* Valid choices for the parameter
* Whether the parameter is |keyword-only|, optional, and/or
  positional-only
* Default values

The type specification should not include information about the
*meaning* of the parameter. Here are some example type specifications:

.. code-block:: rst

   |particle-like|
   `list` of `str`
   |array_like| of `int`, default: [-1, 1]
   |Quantity| [length], default: 10 m
   |Quantity| [temperature, energy], |keyword-only|, default: 0 K

* Use the substitution :rest:`|array_like|` to indicate that an
  |argument| must be |array_like| (i.e., convertible into an |ndarray|).

* Use :rest:`|particle-like|` to indicate that a |particle-like|
  argument should be convertible into a |Particle|, |CustomParticle|, or
  |ParticleList|.

* Use :rest:`|particle-list-like|` to indicate that a
  |particle-list-like| argument should be convertible into a
  |ParticleList|.

* Use :rest:`|atom-like|` to indicate that an argument must be
  |atom-like| (i.e., an element, isotope, and/or ion). ⚛️

* When the array must be :math:`n`\ -dimensional, precede the type by
  :samp:`{n}D` where :samp:`{n}` is replaced by the number of
  dimensions.

  .. code-block:: rst

     1D |array_like|
     3D |array_like|

* If the shapes and sizes of the parameters are interrelated, then
  include that information in parentheses immediately before the type
  information. Include a trailing comma inside the parentheses when the
  parameter is 1D. Use ``:`` for a single dimension of arbitrary size
  and ``...`` for an arbitrary number of dimensions of arbitrary size.

  .. code-block:: rst

     (M,) |array_like|
     (N,) |array_like|
     (M, N) |array_like|
     (N, :) |array_like|
     (M, N, ...) |array_like|

* If the parameter can only be specific values, enclose them in curly
  brackets. The options may be listed with the default value first,
  sorted alphanumerically, or ordered so as to maximize readability.

  .. code-block:: rst

     {"classical postmodernist", "retro-futuristic"}
     {"p+", "e-"}, default: "p+"
     {1, 2, 3, 4}, default: 3

  .. tip::

     Use `typing.Literal` in |type hint annotations| when a parameter
     should only be provided with specific values (e.g.,
     :py:`x: Literal{1, 2, 3, 4}`).

* If a default is given, it is not necessary to state that the parameter
  is optional. When the default is `None`, use ``optional`` instead of
  ``default: `None```.

.. tip::

   If a particular type specification is not covered above, look for
   conventions from the `numpydoc style guide
   <https://numpydoc.readthedocs.io/en/latest/format.html#parameters>`__,
   the `matplotlib documentation guide
   <https://matplotlib.org/stable/devel/document.html#parameter-type-descriptions>`__,
   or the `LSST docstring guide
   <https://developer.lsst.io/python/numpydoc.html>`__.

Parameter descriptions
^^^^^^^^^^^^^^^^^^^^^^

The **parameter description** should concisely describe the meaning of
the parameter, as well as any requirements or restrictions on allowed
values of the parameter (including those specified by
|validate_quantities| or |particle_input|. The parameter description
should not repeat information already in the type specification, but may
include type information when:

* The type specification does not fit with in the docstring line
  character limit;

* Different types have different meanings, requirements, or
  restrictions; or

* The docstring will be more understandable by doing so.

For functions that accept an arbitrary number of positional and/or
keyword arguments, include them in the "Parameters_" section with the
preceding asterisk(s). Order :py:`*args` and :py:`**kwargs` as they
appear in the signature.

.. code-block:: rst

   *args : tuple, optional
       Description of positional arguments.

   **kwargs : dict, optional
       Description of keyword arguments.

.. todo::

   Add a convention for multi-line type specifications.

Exceptions and warnings
~~~~~~~~~~~~~~~~~~~~~~~

* Docstrings may include a "Raises_" section that describes which
  exceptions get raised and under what conditions, and a "Warns_"
  section that describes which warnings will be issued and for what
  reasons.

  * The "Raises_" and "Warns_" sections should only include exceptions
    and warnings that are not obvious or have a high probability of
    occurring. For example, the "Raises_" section should usually not
    include a `TypeError` for when an :term:`argument` is not of the
    type that is listed in the "Parameters_" section of the docstring.

  * The "Raises_" section should include all exceptions that could
    reasonably be expected to require exception handling.

  * The "Raises_" section should be more complete for functionality that
    is frequently used (e.g., |Particle|).

  * The "Raises_" and "Warns_" sections should typically only include
    exceptions and warnings that are raised or issued by the function
    itself. Exceptions and warnings from commonly used decorators like
    |validate_quantities| and |particle_input| should usually not be
    included in these sections, but may be included if there is strong
    justification to do so.

Attributes
~~~~~~~~~~

* Dunder methods (e.g., code objects like ``__add__`` that begin and end
  with two underscores) only need docstrings if it is necessary to
  describe non-standard or potentially unexpected behavior. Custom
  behavior associated with dunder methods should be described in the
  class-level documentation.

  * Docstrings for most dunder methods are not rendered in the online
    documentation and should therefore be intended for contributors.

  * Docstrings for ``__init__``, ``__new__``, and ``__call__`` are
    rendered in the documentation, and should be written for users. The
    docstrings for ``__init__`` and ``__new__`` are included in the
    class-level docstring, while the docstring for ``__call__`` is
    included in the methods summary of a class.

* When an attribute in a class has both a getter (which is the method
  decorated with `property`) and a ``setter`` decoration, then the
  getter and ``setter`` functionality should be documented in the
  docstring of the attribute decorated with :py:`@property`.

  .. code-block:: python

     class Person:
         @property
         def age(self):
             """Document both getter and setter here."""
             return self._age

         @age.setter
         def age(self, n):
             self._age = n

Narrative documentation guidelines
----------------------------------

* Each top-level subpackage must have corresponding narrative
  documentation.

* Use narrative documentation to describe how different functionality
  works together.

* Narrative documentation should be used when the full scope of some
  functionality cannot be adequately described within only the
  docstrings of that functionality.

* Use title case for page titles (e.g., "This is Title Case") and
  sentence case for all other headings (e.g., "This is sentence case").

Sphinx
======

|Sphinx| is the software used to generate |PlasmaPy's documentation|
from |reStructuredText| files and |Python| docstrings. It was originally
created to write Python's documentation and has become the de facto
software for documenting |Python| packages. Most |Python| packages
utilize |Sphinx| to generate their documentation.

Configuration
-------------

The |docs/conf.py|_ file contains the configuration information needed
to customize |Sphinx| behavior. The documentation for |Sphinx| lists the
`configuration options`_ that can be set.

The |docs/_static/css|_ directory contains CSS_ files with `style
overrides`_ for the `Read the Docs Sphinx Theme`_ to customize the look
and feel of the online documentation.

Sphinx extensions
-----------------

|PlasmaPy's documentation| is built with the following |Sphinx|
extensions:

* `sphinx.ext.autodoc` for including documentation from docstrings.
* `sphinx.ext.extlinks` for shortening links to external sites (e.g.,
  :rst:role:`orcid` and :rst:role:`wikipedia`).
* `sphinx.ext.graphviz` to allow Graphviz_ graphs to be included.
* `sphinx.ext.intersphinx` for linking to other projects' documentation.
* `sphinx.ext.mathjax` for math rendering with MathJax_.
* `sphinx.ext.napoleon` for allowing NumPy style docstrings.
* `sphinx.ext.todo` to support ``todo`` |directives|.
* `sphinx.ext.viewcode` to generate links to pages showing source code.
* |nbsphinx| for including |Jupyter| notebooks.
* |sphinxcontrib-bibtex|_ to enable usage of a BibTeX_ file to create
  the :doc:`../bibliography`.
* |sphinx_copybutton|_ to add a "copy" button for code blocks.
* |sphinx_gallery.load_style|_ for using sphinx-gallery styles.
* |IPython.sphinxext.ipython_console_highlighting|_.
* |sphinx_changelog|_ for rendering |towncrier| changelogs.
* |sphinx-tabs|_ for creating tabbed content.
* |sphinx-hoverxref|_ for showing floating windows on cross references
  of the documentation.
* |sphinx-notfound-page|_ to add a :wikipedia:`404 <HTTP_404>` page for
  the documentation.
* |sphinx-issues|_ to add roles for linking to GitHub (:rst:role:`commit`,
  :rst:role:`issue`, :rst:role:`pr`, and :rst:role:`user`).
* |sphinx-reredirects|_ to enable hyperlink redirects.
* |sphinx-toolbox|_ for handy tools for |Sphinx| documentation
* |sphinxemoji|_ for emoji substitutions
* ``plasmapy_sphinx`` for customizations created for use in PlasmaPy and
  affiliated packages. Note that ``plasmapy_sphinx`` is expected to be
  broken out into its own package in the future.

These extensions are specified in :confval:`extensions` configuration
value in |docs/conf.py|_.

.. _external-references:

Cross-referencing external packages
-----------------------------------

Intersphinx_ allows the automatic generation of links to the
documentation of objects in other projects. |:link:| This cross-package
linking is made possible with the `sphinx.ext.intersphinx` extension and
proper package indexing by the external package using
`sphinx.ext.autodoc`.

When we include ```astropy.units.Quantity``` in the documentation, it
will show up as `astropy.units.Quantity` with a link to the appropriate
page in Astropy documentation. Similarly, ```~astropy.units.Quantity```
will show up as `~astropy.units.Quantity`.

The external packages that we can cross-reference via the magic of
`~sphinx.ext.intersphinx` are defined in :confval:`intersphinx_mapping`
in |docs/conf.py|_. Intersphinx has already been set up in PlasmaPy to
include the central |Python| documentation, as well as frequently used
packages such as |Astropy|, |lmfit|, |matplotlib|, |NumPy|, |pandas|,
|SciPy|, and |Sphinx|.

.. tip::

   When adding new packages to :confval:`intersphinx_mapping`, please
   double check that the configuration has been set up correctly.

   If a cross-link is not working as expected this is usually due to one
   of the following reasons:

   * A typo;
   * The package not being defined in :confval:`intersphinx_mapping`, or
   * The referenced source package not properly or fully indexing their
     own code, which is common in |Python| packages.

   For some packages, the name of the package itself does not link
   correctly.

.. _substitutions:

Substitutions
-------------

Some functions and classes are referred to repeatedly throughout the
documentation. |reStructuredText| allows us to `define substitutions`_

.. code-block:: rst

   .. |Particle| replace:: `~plasmapy.particles.particle_class.Particle`

Here whenever :rest:`|Particle|` is used |Sphinx| will replace it with
```~plasmapy.particles.particle_class.Particle``` during build time.

PlasmaPy contains pre-defined global substitutions that can be used
elsewhere in the documentation. For example, we can write :rest:`|Quantity|`
instead of ```~astropy.units.Quantity```, and :rest:`|Particle|` instead of
```~plasmapy.particles.particle_class.Particle```. These global
substitutions are defined in |docs/_global_substitutions.py|_, and are
summarized in the following table.

.. collapse:: Click here to expand/collapse table of global substitutions

   .. include:: _global_substitutions_table.rst

.. note::

   Since substitutions are executed by |Sphinx| when the documentation
   is built, any substitution used in docstrings will not show up when
   using `help`. For example, when :rest:`|Particle|` is used in a
   docstring, `help` will show it as :rest:`|Particle|` rather than
   ```~plasmapy.particles.particle_class.Particle```. Consequently,
   substitutions should not be used in docstrings when it is important
   that users have quick access to the full path of the `object` (such
   as in the ``See Also`` section).

Emoji
-----

Emojis in software documentation help enhance readability, convey
emotions, and make the content friendlier and less intimidating. Emojis
improve comprehension by providing visual cues, such as |:atom:| for
`plasmapy.particles`, |:ocean:| for `plasmapy.dispersion`,
|:stethoscope:| for `plasmapy.diagnostics`, and |:abacus:| for
`plasmapy.formulary`.

Not all text editors, terminals, or :abbr:`IDEs (integrated development
environments)` have the ability to display emojis properly. PlasmaPy's
documentation makes use of the |sphinxemoji|_ extension which adds
substitutions for emojis. For example, we can denote good first issues
with :rest:`|:beginner:|` for |:beginner:|. Click here for the `full list
of emoji codes`_ supported by |sphinxemoji|_.

.. tip::

   Add emojis to research articles to strike up conversations with
   referees and editors! |:sweat_smile:|

.. _citation-instructions:

Bibliography
------------

PlasmaPy uses |sphinxcontrib-bibtex|_ to manage references for its
documentation. This |Sphinx| extension allows us to store references in
a BibTeX_ file which is then used to generate the
:doc:`../bibliography`.  References in the :doc:`../bibliography` are
then citeable from anywhere in the documentation.

To add a new reference to the :doc:`../bibliography`, open
|docs/bibliography.bib|_ and add the reference in `BibTeX format`_. The
citekey should generally be the surname of the first author (all lower
case) followed by a colon and the year. A letter should be added after
the year when needed to disambiguate multiple references. Include the
|DOI| if the reference has one. If the reference does not have a |DOI|,
then include the URL. The ISBN or ISSN number should be included for
books. The ``misc`` field type should be used when citing data sets and
software. Please follow the existing style in |docs/bibliography.bib|_
and alphabetize references by the surname of the first author. To
preserve capitalization, enclose words or phrases within curly brackets
(e.g., ``{NumPy}``).

Use :rest:`:cite:p:\`citekey\`` to create a parenthetical citation and
:rest:`:cite:t:\`citekey\`` to create a textual citation, where
``citekey`` is replaced with the BibTeX_ citekey. Multiple citekeys can
also be used when separated by commas, like
:rest:`:cite:p:\`citekey1, citekey2\``. For example,
:rest:`:cite:p:`wilson:2014\`` will show up as
:cite:p:`wilson:2014`, :rest:`:cite:t:\`wilson:2014\`` will show up as
:cite:t:`wilson:2014`, and :rest:`:cite:p:\`wilson:2014, wilson:2017\``
will show up as :cite:p:`wilson:2014, wilson:2017`.

.. _api-static:

Creating a documentation stub file for a new module
---------------------------------------------------

When the narrative documentation does not index a subpackage (a
directory) or module (a :file:`.py` file) with ``automodule``,
``automodapi``, or the like, then a stub file must be created for that
particular subpackage or module in |docs/api_static|_. For example, the
stub file for `plasmapy.particles.atomic` is placed at
:file:`docs/api_static/plasmapy.particles.atomic.rst` and its contents
look like:

  .. code-block:: rst

     :orphan:

     `plasmapy.particles.atomic`
     ===========================

     .. currentmodule:: plasmapy.particles.atomic

     .. automodapi::  plasmapy.particles.atomic

A missing stub file may lead to either a ``reference target not found``
error or the absence of the module in the documentation build.

.. note::

   If a pull request adds a new subpackage *and* a new module, then a
   stub file must be created for both of them.

   For example, suppose a pull request creates the :py:`plasmapy.io`
   subpackage in the :file:`src/plasmapy/io/` directory and the
   :py:`plasmapy.io.readers` module via :file:`src/plasmapy/io/readers.py`. It
   will then be necessary to create stub files at both
   :file:`docs/api_static/plasmapy.io.rst` and
   :file:`docs/api_static/plasmapy.io.readers.rst`.

Templating
----------

|Sphinx| uses the Jinja_ templating engine to generate HTML code. Jinja_
may be used within the documentation when templating is necessary. For
more details, please refer to `Sphinx's templating page`_.

.. _sphinx-rabbit-dragon-holes:

.. danger::

   There are certain tasks that one would expect to be straightforward
   with |reStructuredText| and |Sphinx| but are only possible by doing a
   horrible workaround that can take hours to figure out. This has given
   rise to the saying:

      *Sphinx rabbit holes often have dragons in them.* |:rabbit2:| |:hole:| |:dragon:|

   Remember: your happiness and well-being are more important than
   `nested inline markup`_!

.. _building-docs:

Building documentation
======================

.. tip::

   Because a documentation preview is generated automatically by |Read
   the Docs| for every pull request, it is not necessary to build the
   documentation locally on your own computer. New contributors can
   safely skip this section.

There are two methods for building the documentation: make_ and |Nox|.

* Using make_ will build the documentation based off of what is in the
  current directory structure. make_ is quicker for local builds than
  |Nox| but requires you to install and set up all dependencies.

* Using |Nox| does not require setting up all dependencies ahead of
  time, but is more computationally intensive since it creates a virtual
  environment and builds the package before building the documentation.
  Consequently, PlasmaPy uses |Nox| for building the documentation on
  continuous integration testing platforms.

.. _doc-build-prereqs:

Prerequisites
-------------

Prior to building the documentation, please follow the
:ref:`instructions on getting ready to contribute <getting ready to
contribute>`. Alternatively, the dependencies for building docs can be
installed by entering the top-level directory of the repository and
running:

.. code-block:: bash

    pip install -e .[docs,tests]

It may also be necessary to install the following software:

* `graphviz <https://graphviz.org/download>`__
* `pandoc <https://pandoc.org/installing.html>`__
* make_ (not necessary for building the documentation with |Nox| or
  sphinx_build)

Building documentation
----------------------

PlasmaPy's documentation can be built using |Nox|, make_, or
sphinx-build_. We recommend starting with |Nox|.

.. tabs::

   .. group-tab:: nox

      We can use |Nox| to build the documentation locally by running:

      .. code-block:: bash

         nox -s docs

      To pass any `options to sphinx-build`_, put them after ``--``. For
      example, use :bash:`nox -s docs -- -v` to increase output
      verbosity.

      Building with |Nox| is well-suited for reproducible documentation
      builds in an isolated Python environment, which is why it is used
      in continuous integration tests.

   .. group-tab:: make

      Building documentation with make_ is useful for interactively
      building and rebuilding the documentation.

      If make_ is installed, we can build the documentation by entering
      the |docs|_ directory and running:

      .. code-block:: bash

         make html

      The documentation build can then be removed with

      .. code-block:: bash

         make clean

   .. group-tab:: sphinx-build

      Using sphinx-build_ allows us to choose different `options to
      sphinx-build`_ than the defaults used by |Nox| and make_.

      PlasmaPy's documentation can be build by going to the top-level
      directory of the repository and running:

      .. code-block:: bash

         sphinx-build docs docs/_build/_html -b html

The documentation landing page can be opened with a web browser at
:file:`docs/_build/html/index.html`.

To check hyperlinks locally, run:

.. tabs::

   .. group-tab:: nox

      .. code-block:: bash

         nox -s linkcheck

   .. group-tab:: make

      .. code-block:: bash

         make linkcheck

   .. group-tab:: sphinx-build

      .. code-block:: bash

         sphinx-build docs docs/_build/_html -b linkcheck

.. tip::

   When writing documentation, please fix any new warnings that arise.
   To enforce this, the ``docs`` |Nox| environment fails if there are
   any warnings.

Troubleshooting
===============

This section describes how to fix common documentation errors and
warnings. |:tools:|

.. _missing-target:

Reference target not found
--------------------------

Warnings like ``py:obj reference target not found`` occur when |Sphinx|
attempts to interpret text as a Python object, but is unable to do so.
For example, if a docstring includes ```y```, Sphinx will attempt to
link to an object named ``y``. If there is no object named ``y``, then
Sphinx will issue this warning, which gets treated like an error.

If the text is meant to be an inline code example, surround it with
double backticks instead of single backticks.

When the text is meant to represent a code object, this warning usually
indicates a typo or a namespace error. For example, the warning
resulting from ```plasmapy.paritcles``` could be resolved by fixing the
typo and changing it to ```plasmapy.particles```.

.. important::

   For PlasmaPy objects, use the full namespace of the object (i.e., use
   ```plasmapy.particles.particle_class.Particle``` instead of
   ```plasmapy.particles.Particle```) or a :ref:`reStructuredText
   substitution <substitutions>` like :rest:`|Particle|` as defined in
   |docs/_global_substitutions.py|_.

This warning may occur when a new module or subpackage is created
without :ref:`creating a stub file <api-static>` for it.

.. _in-parameter-description:

This warning sometimes occurs in the type specification of a |parameter|
in a docstring. Sphinx attempts to link words in type specifications to
code objects. Type lines are intended to provide concise information
about allowed types, sizes, shapes, physical types, and default values
of a parameter. To resolve this warning, first move information about
the *meaning* of a parameter from the type specification into the
parameter description that begins on the following line. To expand the
list of allowed words or patterns in type specifications, add a regular
expression to ``nitpick_ignore_regex`` in |docs/conf.py|_.

This warning may also occur when there is an extra space between a
Sphinx |role| and the argument it is intended to act on. For example,
this warning would be fixed by changing ``:math: `y``` to ``:math:`y```.

Missing documentation pages for new modules
-------------------------------------------

When a new module or subpackage is created, it is usually necessary to
:ref:`create a stub file <api-static>` for it in |docs/api_static|_. A
missing stub file can lead to either a ``reference target not found``
error or missing documentation pages.

Missing attribute errors
------------------------

An `AttributeError` may occur when an :py:`import` statement is missing
in a :file:`__init__.py` file. For example, the error

.. ChatGPT suggested the `diff` pygments language for error/warning messages

.. code-block:: diff

   AttributeError: module 'plasmapy.subpackage' has no attribute 'module'

will occur when :file:`src/plasmapy/subpackage/__init__.py` is missing
:py:`from plasmapy.subpackage import module`. Make sure that
:py:`__all__` contains :py:`"module"` as well.

List ends without a blank line
------------------------------

Warnings like the following:

.. code-block:: diff

   WARNING: :40: (WARNING/2) Bullet list ends without a blank line; unexpected unindent.
   WARNING: :47: (WARNING/2) Definition list ends without a blank line; unexpected unindent.

may show up when Sphinx attempts to interpret text as a list, but is
unable to do so. This warning might not show the file that it occurs in.

If this documentation contains a list, make sure that it is followed by
a blank line and follows the formatting described in `Sphinx's
documentation on lists`_.

This warning may occur in other places due to an indentation or other
formatting problem. Try checking out the formatting in the :ref:`example
docstring` above.

This warning can occur when a changelog entry contains lines that start
with a backtick. Try editing each changelog entry so that it is on a
single really long line, rewording the changelog entry, or using
:ref:`substitutions`.

.. _Sphinx's documentation on lists:
  https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html#lists-and-quote-like-blocks

Could not match a code example to HTML
--------------------------------------

This warning occurs when `sphinx-codeautolink`_ cannot match a code
object to its corresponding documentation. Double check that the code is
correct, and consider adding any missing :py:`import` statements. The
documentation for this extension contains `examples
<https://sphinx-codeautolink.readthedocs.io/en/latest/examples.html>`__
on how to skip blocks with :rest:`.. autolink-skip::` and how to do
invisible imports with :rest:`.. autolink-preface::`.

If this warning occurs in the "Examples" section of a docstring, put
:rest:`.. autolink-skip: section` at the beginning of that section (see
:pr:`2554`). These warnings sometimes only show up when rebuilding
the documentation.

A related warning is ``Could not match transformation of _ on source
lines``.

Errors that are unrelated to a pull request
-------------------------------------------

Occasionally, documentation builds will start failing for reasons that
have nothing to do with the changes made in a pull request. Such errors
generally result from a new release of a package that is required for
PlasmaPy's documentation build.

.. tip::

   If you are a new contributor and have encountered a strange
   documentation build failure, first check recent issues_ to see if one
   has already been created about it. If an issue has not already been
   created, please `raise an issue about the documentation build
   failure`_.

To figure out if a new release caused the error, search |PyPI| for
recently released packages, including `packages related to Sphinx`_ and
any that came up in the error message. You can also check if the same
documentation build failure happened in the `weekly tests`_ on the
``main`` branch. After identifying the package that caused the error, a
pull request can be submitted that sets a temporary maximum allowed
version of the package that can be revisited later.

.. tip::

   When dealing with this kind of error, procrastination often pays off!
   |:balloon:| These errors usually get resolved after the upstream
   package makes a bugfix release, so it is typically better to wait a
   week before spending a large amount of time trying to fix it.
   |:stopwatch:|

Document isn't included in any toctree
--------------------------------------

In general, each source file in the documentation must be included in a
table of contents (toctree_). Otherwise, |Sphinx| will issue a warning
like:

.. code-block:: diff

   WARNING: document isn't included in any toctree

This warning may occur when adding a new :file:`.rst` file or example
Jupyter notebook without adding it to a toctree.

This warning can be resolved by:

* Adding the file to the appropriate toctree, or

* Adding the ``orphan`` `metadata field`_ at the top of the file (not
  recommended in most situations).

In the |docs|_ directory, the tables of contents are generally located
in :file:`index.rst` in the same directory as the source files. For
example Jupyter notebooks, the tables of contents are in
|docs/examples.rst|_.

.. |role| replace:: :term:`role`
.. |roles| replace:: :term:`roles <role>`
.. |directive| replace:: :term:`directive`
.. |directives| replace:: :term:`directives <directive>`

.. _admonitions: https://docutils.sourceforge.io/docs/ref/rst/directives.html#admonitions
.. _BibTeX: https://www.bibtex.org
.. _BibTeX format: https://www.bibtex.com/g/bibtex-format
.. _configuration options: https://www.sphinx-doc.org/en/master/usage/configuration.html
.. _CSS: https://www.w3schools.com:443/css
.. _define substitutions: https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#substitution-definitions
.. _doctests: https://docs.pytest.org/en/6.2.x/doctest.html
.. _full list of emoji codes: https://sphinxemojicodes.readthedocs.io/en/stable/#supported-codes
.. _GitHub Flavored Markdown: https://github.github.com/gfm
.. _Graphviz: https://graphviz.org
.. _intersphinx: https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
.. _issues: https://github.com/PlasmaPy/PlasmaPy/issues
.. _metadata field: https://www.sphinx-doc.org/en/master/usage/restructuredtext/field-lists.html#special-metadata-fields
.. _jinja: https://jinja.palletsprojects.com
.. _LaTeX: https://www.latex-project.org
.. _make: https://www.gnu.org/software/make
.. _markdown: https://www.markdownguide.org
.. _MathJax: https://www.mathjax.org
.. _nested inline markup: https://docutils.sphinx-users.jp/docutils/docs/dev/rst/alternatives.html#nested-inline-markup
.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
.. _options to sphinx-build: https://www.sphinx-doc.org/en/master/man/sphinx-build.html#options
.. _packages related to Sphinx: https://pypi.org/search/?q=sphinx+or+nbsphinx&o=-created&c=Framework+%3A%3A+Sphinx
.. _parameters: https://numpydoc.readthedocs.io/en/latest/format.html#parameters
.. _raise an issue: https://github.com/PlasmaPy/PlasmaPy/issues/new?title=Improve+documentation+for...&labels=Documentation
.. _raise an issue about the documentation build failure: https://github.com/PlasmaPy/PlasmaPy/issues/new?title=Documentation+build+failure&labels=Documentation
.. _raises: https://numpydoc.readthedocs.io/en/latest/format.html#raises
.. _raw string: https://docs.python.org/3/reference/lexical_analysis.html#literals
.. _Read the Docs Sphinx Theme: https://sphinx-rtd-theme.readthedocs.io
.. _SciPy: https://scipy.org
.. _sphinx-build: https://www.sphinx-doc.org/en/master/man/sphinx-build.html
.. _sphinx-codeautolink: https://sphinx-codeautolink.readthedocs.io
.. _Sphinx's glossary: https://www.sphinx-doc.org/en/master/glossary.html
.. _Sphinx's templating page: https://www.sphinx-doc.org/en/master/development/templating.html
.. _style overrides: https://docs.readthedocs.io/en/stable/guides/adding-custom-css.html
.. _toctree: https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#table-of-contents
.. _warns: https://numpydoc.readthedocs.io/en/latest/format.html#warns
.. _weekly tests: https://github.com/PlasmaPy/PlasmaPy/actions/workflows/weekly.yml
.. _Wikipedia: https://www.wikipedia.org

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
   as an error.

.. |example gallery| replace:: :ref:`example gallery <examples>`

.. _`docs`: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs
.. |docs| replace:: :file:`docs`

.. _`docs/_static`: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs/_static
.. |docs/_static| replace:: :file:`docs/_static`

.. _`docs/_static/css`: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs/_static/css
.. |docs/_static/css| replace:: :file:`docs/_static/css`

.. _`docs/about/credits.rst`: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs/about/credits.rst
.. |docs/about/credits.rst| replace:: :file:`docs/about/credits.rst`

.. _`docs/api_static`: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs/api_static
.. |docs/api_static| replace:: :file:`docs/api_static`

.. _`docs/conf.py`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/conf.py
.. |docs/conf.py| replace:: :file:`docs/conf.py`

.. _`docs/examples.rst`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/examples.rst
.. |docs/examples.rst| replace:: :file:`docs/examples.rst`

.. _`docs/glossary.rst`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/glossary.rst
.. |docs/glossary.rst| replace:: :file:`docs/glossary.rst`

.. _`docs/bibliography.bib`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/bibliography.bib
.. |docs/bibliography.bib| replace:: :file:`docs/bibliography.bib`

.. _`docs/_global_substitutions.py`: https://github.com/PlasmaPy/PlasmaPy/blob/main/docs/_global_substitutions.py
.. |docs/_global_substitutions.py| replace:: :file:`docs/_global_subtitutions.py`

.. _docs/notebooks: https://github.com/PlasmaPy/PlasmaPy/tree/main/docs/notebooks
.. |docs/notebooks| replace:: :file:`docs/notebooks`

.. _`IPython.sphinxext.ipython_console_highlighting`: https://ipython.readthedocs.io/en/stable/sphinxext.html?highlight=IPython.sphinxext.ipython_console_highlighting#ipython-sphinx-directive-module
.. |IPython.sphinxext.ipython_console_highlighting| replace:: `IPython.sphinxext.ipython_console_highlighting`

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

.. _`sphinx-tabs`: https://sphinx-tabs.readthedocs.io/
.. |sphinx-tabs| replace:: `sphinx-tabs`

.. _`sphinx-toolbox`: https://sphinx-toolbox.readthedocs.io
.. |sphinx-toolbox| replace:: `sphinx-toolbox`

.. _`sphinxemoji`: https://sphinxemojicodes.readthedocs.io
.. |sphinxemoji| replace:: `sphinxemoji`

.. _`pyproject.toml`: https://github.com/PlasmaPy/PlasmaPy/blob/main/pyproject.toml
.. |pyproject.toml| replace:: :file:`pyproject.toml`
