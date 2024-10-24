.. currentmodule:: plasmapy

.. _glossary:

********
Glossary
********

.. glossary::
   :sorted:

   alias
   aliases
      An abbreviated version of a commonly used function.  For
      example, `~plasmapy.formulary.speeds.va_` is an alias for
      `~plasmapy.formulary.speeds.Alfven_speed`.  Aliases are
      named with a trailing underscore.

      For further details, please refer to the :ref:`contributor guide's
      section on aliases <aliases>`.

   args
      An abbreviation for `positional arguments`_.

   atom-like
      A |particle-like| `object` is **atom-like** if it is or could be
      cast into:

      * A |Particle| representing an element, isotope, or ionic level; or
      * A |ParticleList| including only elements, isotopes, or ionic levels.

      For example, ``"p+"``, ``"He-4"``, ``"deuterium"``, ``"O 0+"``,
      ``Particle("Fe-56 16+)"``, ``["He-4 1+", "He-4 2+"]``,  and
      integers representing atomic numbers are all atom-like.

      Examples of objects that are |particle-like| but *not* atom-like
      include ``"neutron"``, ``"e-"``, and ``["e-", "e+"]``.
      Additionally, ``["He-4", "e-"]`` is not atom-like because this
      `list` contains an item that is not atom-like.

      Please refer to the glossary entry for |particle-like| for a full
      description of valid representations of elements, isotopes, and
      ions.

   -like
      Used to indicate an `object` of that type or that can instantiate
      that type.  For example, ``"He 2+"`` is |particle-like| because it
      can be used to instantiate |Particle|.

   charge number
      The electrical charge of a particle in units of the elementary
      charge. The charge number of an ion or neutral particle is usually
      denoted as ``Z``.

   fit-function
   fit-functions
      Any instance of a subclass of
      `~plasmapy.analysis.fit_functions.AbstractFitFunction`.  Also see
      module `~plasmapy.analysis.fit_functions`.


   force-free
      In plasma physics, a **force-free magnetic field** is a magnetic
      field where the Lorentz force is zero, meaning the magnetic
      pressure greatly exceeds the plasma pressure, allowing
      non-magnetic forces to be neglected. This condition is often
      approximated in the Sun's corona.

      For more details, visit the `Wikipedia page <https://en.wikipedia.org/wiki/Force-free_magnetic_field>`_.

   integration test
      An **integration test** verifies that multiple software
      components work together as intended.

      Compared to a :term:`unit test`, an integration test is typically
      harder to write, slower to run, more difficult to maintain, and
      less useful at pinpointing the specific cause of a problem.
      However, integration tests are able to find problems that unit
      tests cannot. In particular, integration tests are able to find
      problems at the interfaces between different modules. On average,
      each integration test covers more lines of code than each related
      :term:`unit test`. Because unit tests and integration tests
      complement each other, both are important constituents of a test
      suite.

   keyword-only
      An |argument| or |parameter| is *keyword-only* when the |argument|
      must be provided with the name of the corresponding |parameter|.

      If ``z`` is a keyword-only |parameter| to ``f(z)``, then the
      |argument| ``2`` can be provided as ``f(z=2)`` but not ``f(2)``.

   kwargs
      An abbreviation for `keyword arguments`_.

   lite-function
   lite-functions
      An optimized version of an existing `plasmapy` function intended
      for applications where computational efficiency is most important.
      While most `~plasmapy.formulary` functions accept |Quantity|
      objects created using `astropy.units`, lite-functions accept
      numbers and |array_like| inputs that are implicitly assumed to be
      in SI units. The name of a lite-function ends with ``_lite``. A
      lite-function can be accessed as the ``lite`` attribute of the
      corresponding regular function.

      .. caution::

         Unlike most `~plasmapy.formulary` functions, no validations are
         performed on the arguments provided to a lite-function for the
         sake of computational efficiency. When using lite-functions, it
         is vital to double-check your implementation!

      For further details, please refer to the :ref:`contributor guide's
      section on lite-functions <lite-functions>`.

   normalization constant
      A numerical factor that is used to transform a physical quantity
      into a dimensionless number and vice versa. For example, using the
      sound speed as a normalization for velocity will provide us
      with the dimensionless :wikipedia:`Mach number`.

      Normalizations are often employed to provide a convenient
      reference value for the interpretation of data, to avoid
      calculations between very large and very small numbers in
      numerical simulations, and/or to simplify systems of equations for
      analytical theory or numerical solution.

      In PlasmaPy's documentation, a normalization constant is denoted
      with a subscript :math:`⭑` and dimensionless quantities are
      denoted by a tilde :wikipedia:`diacritic` over the symbol. For
      example, in this equation for the velocity :math:`V = V_⭑ Ṽ`\ ,
      :math:`V_⭑` is the normalization for velocity and :math:`Ṽ` is the
      dimensionless velocity.

   particle-like
      An `object` is *particle-like* if it is a |Particle| or
      |CustomParticle|, or can be cast into one.

      An **element** may be represented by a string containing the
      atomic symbol (case-sensitive), the name of the element, or an
      integer representing the atomic number. The element iron can be
      represented as ``"Fe"``, ``"iron"``, or ``26``.

      An **isotope** may be represented by a string that contains an
      atomic symbol or element name, followed by a hyphen and the mass
      number (with no spaces in between). The isotope :sup:`56`\ Fe can
      be represented as ``"Fe-56"``, or ``"iron-56"``. :sup:`2`\ H can
      be represented by ``"D"`` or ``"deuterium"``, and :sup:`3`\ H can
      be represented by ``"T"`` or ``"tritium"``.

      An **ion** or **neutral atom** may be represented by a string that
      contains a representation of an element or isotope, followed by
      charge information which is typically an integer representing the
      charge number and a plus or minus sign to indicate the electrical
      charge. For example, a deuteron may be represented as ``"D 1+"``
      and :sup:`56`\ Fe\ :sup:`1+` may be represented as ``"Fe-56 1+"``.

      A **special particle** may be represented by a string that
      contains the name of the particle (case insensitive) or a
      standard symbol for it (case insensitive). A neutron can be
      represented as ``"n"`` or ``"neutron"``; a proton can be
      represented as ``"p+"``, ``"p"``, or ``"proton"``; and an electron
      can be represented by ``"e-"``, ``"e"``, or ``"electron"``.

      |DimensionlessParticle| instances are not particle-like because,
      without normalization information, they do not uniquely identify a
      physical particle.

      For more complete details, refer to |ParticleLike|.

   particle-list-like
      An `object` is *particle-list-like* if it is a |ParticleList|, or
      can be cast into one.

      For more complete details, refer to |ParticleListLike|.

   real number
      Any numeric type that represents a real number. This could include
      a `float`, `int`, a dimensionless |Quantity|, or any of the
      `numpy.number` types. Note that if a PlasmaPy function expects a
      dimensional |Quantity| and a real number is provided, then the
      real number is often assumed to have the appropriate SI units.

   temperature
      Most functions in PlasmaPy accept temperature, :math:`T`, as a
      `~astropy.units.Quantity` with units of temperature (e.g., kelvin)
      or energy (e.g., electron-volts). A value for energy that is
      provided will be divided by the Boltzmann constant, :math:`k_B`,
      to be converted into units of temperature.

   unit test
      A **unit test** verifies a single unit of behavior, does it
      quickly, and does it in isolation from other tests
      :cite:p:`khorikov:2020`.

      Unit tests are intended to provide fast feedback that help pinpoint
      the locations of errors. Unit tests often abide by the following
      pattern :cite:p:`osherove:2013`:

      1. *Arrange*: gather inputs and get the system to the state in which
         the test is expected to run.

      2. *Act*: make the system under test undertake the operation that is
         being tested.

      3. *Assert*: verify that the actual outcome of the *act* phase matches
         the expected outcome.

      In a unit test for a function, the *arrange* phase involves
      collecting or constructing the inputs for the function. The *act*
      phase occurs when the function is called with those inputs. The
      *assert* phase is when the value returned by the function is
      compared to the expected result.

.. _`keyword arguments`: https://docs.python.org/3/glossary.html#term-argument
.. _`positional arguments`: https://docs.python.org/3/glossary.html#term-argument
