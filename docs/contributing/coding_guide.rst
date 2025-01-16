.. _coding guide:

***************
Coding Guide 👾
***************

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

.. Define roles for in-line code formatting with pygments

.. role:: bash(code)
   :language: bash

.. role:: toml(code)
   :language: TOML

Introduction
============

This guide describes common conventions, guidelines, and strategies for
contributing code to PlasmaPy. The purpose of this guide is not to
provide a set of rigid guidelines that must be adhered to, but rather to
provide a common framework that helps us develop PlasmaPy together as a
community.

Having a shared coding style makes it easier to understand code written
by multiple contributors. The particulars of the coding style are not as
important as consistency, readability, and maintainability.

This guide can (and should!) be regularly refined by the PlasmaPy
community as we collectively learn new practices and our shared coding
style changes. Please feel free to propose revisions to this guide by
:ref:`submitting a pull request <workflow>` or by bringing up an idea at
a community meeting.

PlasmaPy generally follows the :pep:`8` style guide for Python code,
while using tools like |pre-commit| and |ruff| to perform
autoformatting, code quality checks, and automatic fixes.

Coding guidelines
=================

Writing clean code
------------------

* Write short functions that do exactly one thing with no side effects.

* Use |NumPy| array options instead of :py:`for` loops to make code more
  compact, readable, and performant.

* Instead of defining variables like :py:`a0`, :py:`a1`, & :py:`a2`,
  define these values in a collection such as an |ndarray| or a `list`.

* Use the `property` :term:`decorator` instead of getters and setters.

* Some plasma parameters depend on more than one |Quantity| of the same
  physical type. For example, when reading the following line of code,
  we cannot immediately tell which is the electron temperature and which
  is the ion temperature. |:thermometer:|

  .. code-block:: python

     f(1e6 * u.K, 2e6 * u.K)

  Spell out the :term:`parameter` names to improve readability and
  reduce the likelihood of errors.

  .. code-block:: python

     f(T_i=1e6 * u.K, T_e=2e6 * u.K)

  Similarly, when a function has parameters named :py:`T_e` and
  :py:`T_i`, these parameters should be made |keyword-only| to avoid
  ambiguity and reduce the chance of errors.

  .. code-block:: python

     def f(*, T_i, T_e):
         ...

* The :py:`__eq__` and :py:`__ne__` methods of a class should not raise
  exceptions. If the comparison for equality is being made between
  objects of different types, these methods should return `False`
  instead. This behavior is for consistency with operations like
  :py:`1 == "1"` which will return `False`.

* Limit usage of :py:`lambda` functions to one-liners, such as when
  defining the default factory of a `~collections.defaultdict`). For
  anything longer than one line, use :py:`def` instead.

* List and dictionary comprehensions can be used for simple :py:`for`
  loops, like:

  .. code-block:: pycon

     >>> [x**2 for x in range(17) if x % 2 == 0]
     [0, 4, 16, 36, 64, 100, 144, 196, 256]

* Avoid putting any significant implementation code in
  :file:`__init__.py` files. Implementation details should be contained
  in a different file, and then imported into :file:`__init__.py`.

* Avoid defining global variables when possible.

* Use :py:`assert` statements only in tests.

* Use formatted string literals (f-strings) instead of legacy formatting
  for strings.

  .. code-block:: pycon

     >>> package_name = "PlasmaPy"
     >>> print(f"The name of the package is {package_name}.")
     The name of the package is PlasmaPy.
     >>> print(f"{package_name=}")
     package_name='PlasmaPy'
     >>> print(f"{package_name!r}")  # shortcut for f"{repr(package_name)}"
     'PlasmaPy'

* Functions that accept |array_like| or |Quantity| inputs should accept
  and return |nan| (`not a number`_) values. This guideline applies when
  |nan| is the input as well as when |nan| values are included in an
  array.

  .. tip::

     Normally, :py:`numpy.nan == numpy.nan` evaluates to `False`, which
     complicates testing |nan| behavior. The :py:`equal_nan` keyword of
     functions like `numpy.allclose` and `numpy.testing.assert_allclose`
     makes it so that |nan| is considered equal to itself.

* Do not use :term:`mutable` objects as default values in the function
  or method declaration. This can lead to unexpected behavior.

  .. code:: pycon

     >>> def function(l=[]):
     ...     l.append("x")
     ...     print(l)
     ...
     >>> function()
     ['x']
     >>> function()
     ['x', 'x']

* Use `pathlib` when working with paths to data files.

Names
-----

Names are our most fundamental means of communicating the intent and
purpose of code. Wisely chosen names can greatly improve the
understandability of code, while inadequate names can obfuscate what the
code is supposed to be doing.

* PlasmaPy generally uses the :pep:`8` conventions for variable names.

  - Use lowercase words separated by underscores for function and
    variable names (e.g., :py:`function_name` and :py:`variable_name`).

  - Use capitalized words without separators when naming a class (e.g.,
    :py:`ClassName`), but keep acronyms capitalized (e.g.,
    :py:`MHDEquations`).

  - Use capital letters words separated by underscores when naming
    constants (e.g., :py:`CONSTANT` or :py:`CONSTANT_NAME`).

  There are some situations in PlasmaPy which justify a departure from
  the :pep:`8` conventions.

  - Functions based on plasma parameters that are named after people may
    be capitalized (e.g., :py:`Alfven_speed`).

  - Capital letters may be used for a variable when it matches the
    standard usage in plasma science (e.g., :py:`B` for magnetic field
    and :py:`T` for temperature).

* Choose names that are pronounceable to make them more memorable and
  compatible with text-to-speech technology.

* Choose names will produce more relevant results when searching the
  internet.

* Avoid unnecessary abbreviations, as these make code harder to read.
  Prefer clarity over brevity, except for code that is used frequently
  and interactively (e.g., :command:`cd` or :command:`ls`).

  .. tip::

     Measure the length of a variable not by the number of characters,
     but rather by the time needed to understand its meaning.

     By this measure, :py:`cggglm` is significantly longer than
     :py:`solve_gauss_markov_linear_model`.

* Avoid ambiguity. Does :py:`temp` mean "temperature", "temporary", or
  "template"?

* Append :py:`_e` to a variable name to indicate that it refers to
  electrons, :py:`_i` for ions, and :py:`_p` for protons (e.g.,
  :py:`T_e`, :py:`T_i`, and :py:`T_p`).

* Only ASCII_ characters should be used in code that is part of the
  public :wikipedia:`API`.

* Python allows alphanumeric Unicode characters to be used in object
  names (e.g., :py:`πλάσμα` or :py:`φυσική`). These characters may be
  used for *internal* code when doing so improves readability (i.e.,
  to match a commonly used symbol) and in |Jupyter| notebooks.

* If a plasma parameter has multiple names, then use the name that
  provides the most physical insight. For example, :py:`gyrofrequency`
  indicates gyration but :py:`Larmor_frequency` does not. |:dizzy:|

* It is *usually* preferable to name a variable after its name rather
  than its symbol. An object named :py:`Debye_length` is more broadly
  understandable and searchable than :py:`lambda_D`. However, there are
  some exceptions to this guideline.

  * Symbols used widely across plasma science can be used with low risk
    of confusion, such as :math:`T` for temperature or :math:`β` for
    plasma `~plasmapy.formulary.dimensionless.beta`.

  * Symbols that are defined in docstrings can be used with decreased
    likelihood of confusion.

  * Sometimes code that represents an equation will be more readable if
    the Unicode characters for the symbols are used, especially for
    complex equations. For someone who is familiar with the symbols,
    :py:`λ = c / ν` will be more readable than :py:`lambda = c / nu` or
    :py:`wavelength = speed_of_light / frequency`.

  * If an implementation is based on a journal article, then variable
    names may be based on the symbols used in that article. The article
    should be :ref:`cited <citation-instructions>` in the appropriate
    docstring so that it appears in the |bibliography|.

* To mark that an object is not part of PlasmaPy's public
  :wikipedia:`API`, begin its name with a leading underscore (e.g.,
  :py:`_private_variable`). Private variables should not be included in
  :py:`__all__`.

* Avoid single character variable names except for standard plasma
  physics symbols (e.g., :py:`B`) or as indices in :py:`for` loops.

* Avoid encoding type information in a variable name.

* Intermediate variable names can provide additional context and
  meaning. For example, suppose we have a conditional operating on a
  complicated expression:

  .. code-block:: python

     if u[0] < x < u[1] and v[0] < y < v[1] and w[0] < z < w[1]:
         ...

  Defining an intermediate variable allows us to communicate the meaning
  and intent of the expression.

  .. code-block:: python

     point_is_in_grid_cell = u[0] < x < u[1] and v[0] < y < v[1] and w[0] < z < w[1]

     if point_is_in_grid_cell:
         ...

  In :py:`for` loops, this may take the form of assignment expressions
  with the walrus operator (:py:`:=`).

.. tip::

   It is common for an |IDE| to have a built-in tool for simultaneously
   renaming a variable throughout a project. For example, a `rename
   refactoring in PyCharm
   <https://www.jetbrains.com/help/pycharm/rename-refactorings.html>`__
   can be done with :kbd:`Shift+F6` on Windows or Linux, and :kbd:`⇧F6`
   or :kbd:`⌥⌘R` on macOS.

Comments
--------

A well-placed and well-written comment can prevent future frustrations.
However, comments are not inherently good. As code evolves, an
unmaintained comment may become outdated, or get separated from the
section of code that it was meant to describe. Cryptic and obsolete
comments may end up confusing contributors. In the worst case, an
unmaintained comment may contain inaccurate or misleading information
(hence the saying that "a comment is a lie waiting to happen").

.. important::

   The code we write should read like a book. The full meaning of code's
   functionality should be attainable by reading the code. Comments
   should only be used when the code itself cannot communicate its full
   meaning.

* Refactor code to make it more readable, rather than explaining how it
  works :cite:p:`wilson:2014`.

* Instead of using a comment to define a variable, rename the variable
  to encode its meaning and intent. For example, code like:

  .. code-block:: python

     # collision frequency
     nu = 1e6 * u.s**-1

  could be achieved with no comment by doing:

  .. code-block:: python

     collision_frequency = 1e6 * u.s**-1

* Use comments to communicate information that you wish you knew before
  starting to work on a particular section of code, including
  information that took some time to learn.

* Use comments to communicate information that the code cannot,
  such as why an alternative approach was *not* taken.

* Use comments to include references to books or articles that describe
  the equation, algorithm, or software design pattern that is being
  implemented. Even better, include these references in docstrings.

* Provide enough contextual information in the comment for a new user
  to be able to understand it.

* Remove commented out code before merging a pull request.

* When updating code, be sure to review and update, if necessary,
  associated comments too!

* When a comment is used as the header for a section of code, consider
  extracting that section of code into its own function. For example, we
  might start out with a function that includes multiple lines of code
  for each step.

  .. code-block:: python

     def analyze_experiment(data):
         # Step 1: calibrate the data
         ...
         # Step 2: normalize the data
         ...

  We can apply the `extract function refactoring pattern`_ by creating a
  separate function for each of these steps. The name of each function
  can often be extracted directly from the comment.

  .. code-block:: python

     def calibrate_data(data):
         ...
         return calibrated_data


     def normalize_data(data):
         ...
         return normalized_data


     def analyze_experiment(data):
         calibrated_data = calibrate_data(data)
         normalized_data = normalize_data(calibrated_data)

  This refactoring pattern is appropriate for long functions where the
  different steps can be cleanly separated from each other. This pattern
  leads to functions that are shorter, more reusable, and easier to
  test. The original function contains fewer low-level implementation
  details and thus gives a higher level view of what the function is
  doing. This pattern reduces `cognitive complexity`_.

  The `extract function refactoring pattern`_ should be used
  judiciously, as taking it to an extreme and applying it at too fine of
  a scale can reduce readability and maintainability by producing overly
  fragmented code.

  .. hint::

     The `extract function refactoring pattern`_ might not be
     appropriate if the different sections of code are intertwined with
     each other (e.g., if both sections require the same intermediate
     variables). An alternative in such cases would be to create a class
     instead.

Error messages
--------------

Error messages are a vital but underappreciated form of documentation. A
good error message can help someone pinpoint the source of a problem in
seconds, while a cryptic or missing error message can lead to hours of
frustration.

* Use error messages to indicate the source of the problem while
  providing enough information for the user to troubleshoot it. When
  possible, make it clear what the user should do next.

* Include diagnostic information when appropriate. For example, if an
  error occurred at a single index in an array operation, then including
  the index where the error happened can help the user better understand
  the cause of the error.

* Write error messages that are concise when possible, as users often
  skim or skip long error messages.

* Avoid including information that is irrelevant to the source of the
  problem.

* Write error messages in language that is plain enough to be
  understandable to someone who is undertaking their first research
  project.

  - If necessary, technical information may be placed after a plain
    language summary statement.

  - Alternatively, an error message may reference a docstring or a page
    in the narrative documentation.

* Write error messages that are friendly, supportive, and helpful. Error
  message should never be condescending or blame the user.

Type hint annotations
=====================

PlasmaPy uses |type hint annotations| and |mypy| to perform
|static type checking|. Type hints improve readability and
maintainability by clarifying the types that a function accepts and
returns. Type hints also help Jupyter notebooks and IDEs provide better
tooltips and perform auto-completion.

Type hint annotations specify the expected types of arguments and return
values. A function that accepts a `float` or `str` and returns a `str`
may be written as:

.. code-block:: python

   def f(x: float | str) -> str:
       return str(x)

The :py:`|` operator is used to represent unions between types. To learn
more, check out the `type hints cheat sheet`_.

.. note::

   Type hint annotations are by default not enforced at runtime, and
   instead are used to _indicate_ the types that a function or method
   accepts and returns. However, there are some situations where type
   hints do play a role at runtime, such as in functions decorated by
   |particle_input| and/or |validate_quantities|.

Automatically adding type hint annotations
------------------------------------------

PlasmaPy has defined multiple |Nox| sessions in |noxfile.py|_ that can
automatically add type hints using autotyping_ and MonkeyType_.

The ``autotyping(safe)`` session uses autotyping_ to automatically add
type hints for common patterns, while producing very few incorrect
annotations:

.. code-block:: shell

   nox -s 'autotyping(safe)'

The ``autotyping(aggressive)`` session uses autotyping_ to automatically
add even more type hints than ``autotyping(safe)``. Because it is less
reliable, the newly added type hints should be carefully reviewed:

.. code-block:: shell

   nox -s 'autotyping(aggressive)'

The ``monkeytype`` session automatically adds type hint annotations to a
module based on the types of variables that were observed when running
`pytest`. Like ``autotyping(aggressive)``, it can add incorrect or
incomplete type hints, so newly added type hints should be carefully
reviewed. It is run for a single module at a time:

.. code-block:: shell

   nox -s monkeytype -- plasmapy.particles.atomic

.. tip::

   Run :bash:`nox -s 'autotyping(safe)'` and commit the changes before
   executing the ``autotyping(aggressive)`` or ``monkeytype`` sessions.

Static type checking
--------------------

PlasmaPy uses |mypy| to perform |static type checking| to detect
incorrect or inconsistent |type hint annotations|. Static type checking
helps us find type related errors during the development process, and
thus improve code quality.

We can perform static type checking by running:

.. code-block:: shell

   nox -s mypy

The configuration for |mypy| is in |mypy.ini|_.

Using |mypy| helps us identify errors and fix problems. For example,
suppose we run |mypy| on the following function:

.. code-block:: python

   def return_object(x: int | str) -> int:  # should be: -> int | str
       return x

We will then get the following error:

.. code-block:: diff

   Incompatible return value type (got "int | str", expected "int")  [return-value]

.. tip::

   To learn more about a particular |mypy| error code, search for it in
   its documentation pages on `error codes enabled by default`_ and
   `error codes for optional checks`_.

Ignoring mypy errors
~~~~~~~~~~~~~~~~~~~~

Static type checkers like |mypy| are unable to follow the behavior of
functions that dynamically change the types of objects, which occurs in
functions decorated by |particle_input|. In situations like this, we can
use a :py:`# type: ignore` comment to indicate that |mypy| should ignore
a particular error.

.. code-block:: python

   from plasmapy.particles import particle_input, ParticleLike

   @particle_input
   def f(particle: ParticleLike) -> Particle | CustomParticle | ParticleList:
       return particle  # type: ignore[return-value]

.. important::

   Because type hints are easier to add while writing code, please use
   :py:`# type ignore` comments sparingly!

.. note::

   PlasmaPy only recently added |mypy| to its continuous integration
   suite. If you run into |mypy| errors that frequently need to be
   ignored, please bring them up in :issue:`2589`.

Quantity type hints
-------------------

When a function accepts a |Quantity|, the annotation should additionally
include the corresponding unit in brackets. When the function is
|decorated| with |validate_quantities|, then the |Quantity| provided to
and/or returned by the function will be converted to that unit.

.. code-block:: python

   import astropy.units as u
   from plasmapy.utils.decorators import validate_quantities


   @validate_quantities
   def speed(distance: u.Quantity[u.m], time: u.Quantity[u.s]) -> u.Quantity[u.m / u.s]:
       return distance / time

Particle type hints
-------------------

Functions that accept particles or particle collections should annotate
the corresponding function with |ParticleLike| or |ParticleListLike|.
When the function is decorated with |particle_input|, then it will
convert the function to the corresponding |Particle|, |CustomParticle|,
or |ParticleList|.

.. code-block:: python

   from plasmapy.particles.decorators import particle_input
   from plasmapy.particles.particle_class import CustomParticle, Particle, ParticleLike


   @particle_input
   def get_particle(particle: ParticleLike) -> Particle | CustomParticle:
       return particle  # type: ignore[return-value]

The :py:`# type: ignore[return-value]` comment for |mypy| is needed
because |particle_input| dynamically (rather than statically) changes
the type of ``particle``.

Imports
-------

* Use standard abbreviations for imported packages:

  .. code-block:: python

     import astropy.constants as const
     import astropy.units as u
     import matplotlib.pyplot as plt
     import numpy as np
     import pandas as pd

* PlasmaPy uses |ruff| to organize import statements via a |pre-commit|
  hook.

* For most objects, import the package, subpackage, or module rather
  than the individual code object. Including more of the namespace
  provides contextual information that can make code easier to read. For
  example, :py:`json.loads` is more readable than using only
  :py:`loads`.

* For the most frequently used PlasmaPy objects (e.g., |Particle|) and
  |type hint annotations| (e.g., `~typing.Optional`), import the object
  directly instead of importing the package, subpackage, or module.

* Use absolute imports (e.g., :py:`from plasmapy.particles import
  Particle`) rather than relative imports (e.g., :py:`from ..particles
  import Particle`).

* Do not use star imports (e.g., :py:`from package.subpackage import *`),
  except in very limited situations.

Project infrastructure
======================

* Package requirements are specified in |pyproject.toml|_.

For general information about Python packaging, check out the
`Python Packaging User Guide`_.

Configuration
-------------

PlasmaPy's main configuration file is |pyproject.toml|_ (which is
written in the TOML_ format). The Python Packaging User Guide contains
a page on `writing your pyproject.toml file`_. The :toml:`project`
table defines overall project metadata, while tables like
:toml:`[tool.ruff]` include the configuration for tools like |ruff|.

Dependencies and requirements
-----------------------------

* PlasmaPy's dependencies and requirements are specified in
  |pyproject.toml|_ under :toml:`[project.dependencies]` (i.e., in the
  :toml:`dependencies` array in the :toml:`[project]` table).

* PlasmaPy releases should follow the recommendations in |SPEC 0|,
  including that:

  - Support for Python versions be dropped **3 years** after their
    initial release.
  - Support for core package dependencies be dropped **2 years** after
    their initial release.

* The |uv.lock|_ file contains pinned requirements files
  for use in continuous integration tests.

  - These files are updated periodically via pull requests created by a
    GitHub workflow to `update pinned requirements`_.

  - When updating requirements in |pyproject.toml|_, run
    :bash:`nox -s requirements` to update the pinned requirements files.

  - Validate requirements with :bash:`nox -s validate_requirements`.

* Even if a dependency is unlikely to be shared with packages installed
  alongside PlasmaPy, that dependency may have strict requirements that
  do cause conflicts. For example, requiring the newest version of
  voila_ once caused dependency conflicts with other packages in the
  heliopythoniverse because voila_ had strict dependencies on packages
  in the Jupyter ecosystem.

* Only set maximum or exact requirements (e.g., ``numpy <= 2.0.0`` or
  ``scipy == 1.13.1``) when absolutely necessary. After setting a
  maximum or exact requirement, create a GitHub issue to loosen that
  requirement.

  .. important::

     Maximum requirements can lead to version conflicts when installed
     alongside other packages. It is preferable to update PlasmaPy to
     become compatible with the latest versions of its dependencies than
     to set a maximum requirement.

* It sometimes takes a few months for packages like Numba to become
  compatible with the newest minor version of |Python|.

* The ``tests`` and ``docs`` dependency sets are required for running
  tests and building documentation, but are not required for package
  installation. Consequently, we can require much newer versions of the
  packages in these dependency sets.

.. tip::

   Packages that depend on PlasmaPy should periodically run their tests
   against the ``main`` branch of PlasmaPy. Similarly, PlasmaPy has
   |Nox| sessions used in GitHub workflows that run its test suite
   against the development versions of important dependencies such as
   NumPy and Astropy. Such tests can help find problems before
   they are included in an official release.

Special function categories
===========================

.. _aliases:

Aliases
-------

An :term:`alias` is an abbreviated version of a commonly used function.
For example, `~plasmapy.formulary.speeds.va_` is an alias to
`~plasmapy.formulary.speeds.Alfven_speed`.

:term:`Aliases` are intended to give users the option for shortening
their code while maintaining some readability and explicit meaning. As
such, :term:`aliases` are given to functionality that already has a
widely-used symbol in plasma literature.

Here is a minimal example of an alias :py:`f_` to :py:`function` as
would be defined in :file:`src/plasmapy/subpackage/module.py`.

.. code-block:: python

   __all__ = ["function"]
   __aliases__ = ["f_"]

   __all__ += __aliases__


   def function():
       ...


   f_ = function
   """Alias to `~plasmapy.subpackage.module.function`."""

* Aliases should only be defined for frequently used plasma parameters
  which already have a symbol that is widely used in the community's
  literature. This is to ensure that the abbreviated function name is
  still reasonably understandable. For example,
  `~plasmapy.formulary.lengths.cwp_` is a shortcut for :math:`c/ω_p`\ .

* The name of an alias should end with a trailing underscore.

* An alias should be defined immediately after the original function.

* Each alias should have a one-line docstring that refers users to the
  original function.

* The name of the original function should be included in :py:`__all__`
  near the top of each module, and the name of the alias should be
  included in :py:`__aliases__`, which will then get appended to
  :py:`__all__`. This is done so both the :term:`alias` and the original
  function get properly documented.

* Aliases are intended for end users, and should not be used in PlasmaPy
  or other collaborative software development efforts because of
  reduced readability and searchability for someone new to plasma
  science.

.. _lite-functions:

Lite Functions
--------------

Most functions in `plasmapy.formulary` accept |Quantity| instances as
arguments and use |validate_quantities| to verify that |Quantity|
arguments are valid. The use of |Quantity| operations and validations do
not noticeably impact performance during typical interactive use, but
the performance penalty can become significant for numerically intensive
applications.

A :term:`lite-function` is an optimized version of another `plasmapy`
function that accepts numbers and |NumPy| arrays in assumed SI units.
:term:`Lite-functions` skip all validations and instead prioritize
performance. Most :term:`lite-functions` are defined in
`plasmapy.formulary`.

.. caution::

   Unlike most `~plasmapy.formulary` functions, no validations are
   performed on the arguments provided to a :term:`lite-function` for
   the sake of computational efficiency. When using
   :term:`lite-functions`, it is vital to double-check your
   implementation!

Here is a minimal example of a :term:`lite-function` :py:`function_lite`
that corresponds to :py:`function` as would be defined in
:file:`src/plasmapy/subpackage/module.py`.

.. code-block:: python

   __all__ = ["function"]
   __lite_funcs__ = ["function_lite"]

   from numbers import Real

   from plasmapy.utils.decorators import bind_lite_func

   __all__ += __lite_funcs__


   def function_lite(v: float) -> float:
       """
       The lite-function which accepts and returns real numbers in
       assumed SI units.
       """
       ...


   @bind_lite_func(function_lite)
   def function(v):
       """A function that accepts and returns Quantity arguments."""
       ...

* The name of each :term:`lite-function` should be the name of the
  original function with :py:`_lite` appended at the end. For example,
  `~plasmapy.formulary.speeds.thermal_speed_lite` is the
  :term:`lite-function` associated with
  `~plasmapy.formulary.speeds.thermal_speed`.

* :term:`Lite-functions` assume SI units for all arguments that
  represent physical quantities.

* :term:`Lite-functions` should be defined immediately before the normal
  version of the function.

* :term:`Lite-functions` should be used by their associate non-lite
  counterpart, except for well reasoned exceptions. This is done to
  reduce code duplication.

* :term:`Lite-functions` are bound to their normal version as the
  :py:`lite` attribute using the
  `~plasmapy.utils.decorators.lite_func.bind_lite_func` decorator. This
  allows the :term:`lite-function` to also be accessed like
  :py:`thermal_speed.lite()`.

* A :term:`lite-function` should not include any "extra" code beyond the
  raw calculation. In the future

* The name of the original function should be included in :py:`__all__`
  near the top of each module, and the name of the :term:`lite-function`
  should be included in :py:`__lite_funcs__`, which will then get
  appended to :py:`__all__`. This is done so both the
  :term:`lite-function` and the original function get properly
  documented.

Physics
=======

Units
-----

PlasmaPy uses |astropy.units|_ to assign physical units to values in the
form of a |Quantity|.

.. code-block:: pycon

   >>> import astropy.units as u
   >>> 5 * u.m / u.s
   <Quantity 5. m / s>

Using |astropy.units|_ improves compatibility with Python packages in
adjacent fields such as astronomy and heliophysics. To get started with
|astropy.units|_, check out this `example notebook on units`_.

  .. caution::

     Some `scipy` functions silently drop units when used on |Quantity|
     instances.

* Only SI units should be used within PlasmaPy, unless there is a strong
  justification to do otherwise. Example notebooks may occasionally use
  other unit systems to show the flexibility of |astropy.units|_.

* Use operations between |Quantity| instances except when needed for
  performance. To improve performance in |Quantity| operations, check
  out `performance tips
  <https://docs.astropy.org/en/stable/units/index.html#performance-tips>`__
  for |astropy.units|_.

* Use unit annotations with the |validate_quantities| decorator to
  validate |Quantity| arguments and return values
  (see :ref:`validating_quantities`).

  .. caution::

     Recent versions of |Astropy| allow unit-aware |Quantity|
     annotations such as :py:`u.Quantity[u.m]`. However, these
     annotations are not yet compatible with |validate_quantities|.

* Avoid using electron-volts as a unit of temperature within PlasmaPy
  because it is defined as a unit of energy. However, functions in
  `plasmapy.formulary` and elsewhere should accept temperatures in units
  of electron-volts, which can be done using |validate_quantities|.

* Non-standard unit conversions can be made using equivalencies_ such
  as `~astropy.units.temperature_energy`.

  .. code-block:: pycon

     >>> (1 * u.eV).to(u.K, equivalencies=u.temperature_energy())
     11604.518...

* The names of SI units should not be capitalized except at the
  beginning of a sentence, including when they are named after a person.
  The sole exception is "degree Celsius".

.. _validating_quantities:

Validating quantities
~~~~~~~~~~~~~~~~~~~~~

Use |validate_quantities| to enforce |Quantity| type hints:

.. code-block:: python

   @validate_quantities
   def magnetic_pressure(B: u.Quantity[u.T]) -> u.Quantity[u.Pa]:
       return B**2 / (2 * const.mu0)

Use |validate_quantities| to verify function arguments and impose
relevant restrictions:

.. code-block:: python

   from plasmapy.utils.decorators.validators import validate_quantities

   @validate_quantities(
       n={"can_be_negative": False},
       validations_on_return={"equivalencies": u.dimensionless_angles()},
   )
   def inertial_length(n: u.Quantity[u.m**-3], particle) -> u.Quantity[u.m]:
       ...

Particles
---------

The |Particle| class provides an object-oriented interface for accessing
basic particle data. |Particle| accepts |particle-like| inputs.

.. code-block:: pycon

   >>> from plasmapy.particles import Particle
   >>> alpha = Particle("He-4 2+")
   >>> alpha.mass
   <Quantity 6.6446...e-27 kg>
   >>> alpha.charge
   <Quantity 3.20435...e-19 C>

To get started with `plasmapy.particles`, check out this `example
notebook on particles`_.

.. caution::

   When an element is provided to a PlasmaPy function without isotope
   information, it is assumed that the mass is given by the standard
   atomic weight. While :py:`Particle("p+")` represents a proton,
   :py:`Particle("H+")` includes some deuterons.

.. tip::

   Avoid using implicit default particle assumptions for function
   arguments (see :issue:`453`).

.. _particle-like-arguments:

Transforming particle-like arguments
------------------------------------

* The |particle_input| decorator can automatically transform a
  |particle-like| |argument| into a |Particle|, |CustomParticle|, or
  |ParticleList| instance when the corresponding |parameter| is
  annotated with |ParticleLike|.

* For |particle-list-like| parameters, use |ParticleListLike| as the
  annotation so that the corresponding argument is transformed into a
  |ParticleList|.

  .. code-block:: python

     from plasmapy.particles import Particle, ParticleLike, particle_input


     @particle_input
     def get_particle(particle: ParticleLike) -> Particle:
         return particle

  If we use :py:`get_particle` on something |particle-like|, it will
  return the corresponding particle object.

  .. code-block:: pycon

     >>> return_particle("p+")
     Particle("p+")

  The documentation for |particle_input| describes ways to ensure that
  the particle meets certain categorization criteria.

Equations and Physical Formulae
-------------------------------

* Physical formulae should be inputted without first evaluating all of
  the physical constants. For example, the following line of code
  obscures information about the physics being represented:

  .. autolink-skip:: section

  .. code-block:: python

     omega_ce = 1.76e7*(B/u.G)*u.rad/u.s  # doctest: +SKIP

  In contrast, the following line of code shows the exact formula
  which makes the code much more readable.

  .. code-block:: python

     omega_ce = (e * B) / (m_e * c)  # doctest: +SKIP

  The origins of numerical coefficients in formulae should be
  documented.

* Docstrings should describe the physics associated with these
  quantities in ways that are understandable to students who are
  taking their first course in plasma physics while still being useful
  to experienced plasma physicists.

Angular Frequencies
-------------------

Unit conversions involving angles must be treated with care. Angles are
dimensionless but do have units. Angular velocity is often given in
units of radians per second, though dimensionally this is equivalent to
inverse seconds. Astropy will treat radians dimensionlessly when using
the :py:`dimensionless_angles` equivalency, but
:py:`dimensionless_angles` does not account for the multiplicative
factor of :math:`2π` that is used when converting between frequency
(1/s) and angular frequency (rad/s). An explicit way to do this
conversion is to set up an equivalency between cycles/s and Hz:

.. code-block:: python

   import astropy.units as u
   f_ce = omega_ce.to(u.Hz, equivalencies=[(u.cy/u.s, u.Hz)])  # doctest: +SKIP

However, :py:`dimensionless_angles` does work when dividing a velocity by
an angular frequency to get a length scale:

.. code-block:: python

   d_i = (c/omega_pi).to(u.m, equivalencies=u.dimensionless_angles())  # doctest: +SKIP

.. _performing releases:

Performing releases
===================

Before beginning the release process, first run the workflow to `create
a release issue`_ (i.e., :issue:`2723`). The resulting issue will
include the `release checklist`_ which describes the release process in
detail.

The overall process of performing a release is:

* `Create a release issue`_.
* Make code quality and documentation updates.
* Run the workflows for CI and weekly tests.
* Reserve a DOI on Zenodo.
* Run the workflow to `mint a release`_ to build the changelog, create a
  release branch, and tag the version for the release.
* Create a release on GitHub based on that tag. This step will trigger
  the workflow to publish to the Python Package Index.
* Create a pull request to merge the release back into ``main`` (but do
  not squash merge it).
* Download the :file:`.tar.gz` file for the tagged version on GitHub and
  upload it to Zenodo (while updating metadata).
* Make sure that the automated pull request to the conda-forge feedstock
  is merged successfully
* Activate the release on Read the Docs.
* Test the release.
* Update the `release checklist`_.

.. _ASCII: https://en.wikipedia.org/wiki/ASCII
.. _autotyping: https://github.com/JelleZijlstra/autotyping
.. _cognitive complexity: https://docs.codeclimate.com/docs/cognitive-complexity
.. _create a release issue: https://github.com/PlasmaPy/PlasmaPy/actions/workflows/create-release-issue.yml
.. _Cython: https://cython.org
.. _equivalencies: https://docs.astropy.org/en/stable/units/equivalencies.html
.. _error codes enabled by default: https://mypy.readthedocs.io/en/stable/error_code_list.html
.. _error codes for optional checks: https://mypy.readthedocs.io/en/stable/error_code_list2.html
.. _example notebook on particles: ../notebooks/getting_started/particles.ipynb
.. _example notebook on units: ../notebooks/getting_started/units.ipynb
.. _extract function refactoring pattern: https://refactoring.guru/extract-method
.. _mint a release:
.. _MonkeyType: https://monkeytype.readthedocs.io
.. _NEP 29: https://numpy.org/neps/nep-0029-deprecation_policy.html
.. _not a number: https://en.wikipedia.org/wiki/NaN
.. _NumPy Enhancement Proposal 29: https://numpy.org/neps/nep-0029-deprecation_policy.html
.. _Python Packaging User Guide: https://packaging.python.org
.. _pyupgrade: https://github.com/asottile/pyupgrade
.. _release checklist: https://github.com/PlasmaPy/PlasmaPy/blob/main/.github/content/release-checklist.md
.. _rename refactoring in PyCharm: https://www.jetbrains.com/help/pycharm/rename-refactorings.html
.. _TOML: https://toml.io/en/v1.0.0
.. _type hints cheat sheet: https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html
.. _update pinned requirements: https://github.com/PlasmaPy/PlasmaPy/actions/workflows/update-pinned-reqs.yml
.. _voila: https://voila.readthedocs.io
.. _writing your pyproject.toml file: https://packaging.python.org/en/latest/guides/writing-pyproject-toml/

.. _`astropy.units`: https://docs.astropy.org/en/stable/units/index.html
.. |astropy.units| replace:: `astropy.units`

.. _`uv.lock`: https://github.com/PlasmaPy/PlasmaPy/blob/main/uv.lock
.. |uv.lock| replace:: :file:`uv.lock`

.. _`mypy.ini`: https://github.com/PlasmaPy/PlasmaPy/blob/main/mypy.ini
.. |mypy.ini| replace:: :file:`mypy.ini`

.. _`noxfile.py`: https://github.com/PlasmaPy/PlasmaPy/blob/main/noxfile.py
.. |noxfile.py| replace:: :file:`noxfile.py`

.. _`pyproject.toml`: https://github.com/PlasmaPy/PlasmaPy/blob/main/pyproject.toml
.. |pyproject.toml| replace:: :file:`pyproject.toml`
