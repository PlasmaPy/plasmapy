.. _formulary:

********************************
Formulary (`plasmapy.formulary`)
********************************

`plasmapy.formulary` provides theoretical formulas for calculation of
physical quantities helpful for plasma physics.

.. table::
   :widths: 5 16

   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Classical Transport <braginskii>          | `plasmapy.formulary.braginskii`         |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Collisions <collisions>                   | `plasmapy.formulary.collisions`         |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Dielectrics <dielectric>                  | `plasmapy.formulary.dielectric`         |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Dimensionless <dimensionless>             | `plasmapy.formulary.dimensionless`      |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Distribution Functions <distribution>     | `plasmapy.formulary.distribution`       |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Drifts <drifts>                           | `plasmapy.formulary.drifts`             |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Ionization <ionization>                   | `plasmapy.formulary.ionization`         |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Magnetostatics <magnetostatics>           | `plasmapy.formulary.magnetostatics`     |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Mathematics <mathematics>                 | `plasmapy.formulary.mathematics`        |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Plasma Parameters <parameters>            | `plasmapy.formulary.parameters`         |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Quantum Relations <quantum>               | `plasmapy.formulary.quantum`            |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Radiation <radiation>                     | `plasmapy.formulary.radiation`          |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Relativistic Relations <relativity>       | `plasmapy.formulary.relativity`         |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+
   | .. toctree:: Shocks <shocks>                           | `plasmapy.formulary.shocks`             |
   |    :maxdepth: 1                                        |                                         |
   +--------------------------------------------------------+-----------------------------------------+

The subpackage makes heavy use of `astropy.units.Quantity` for handling
conversions between different unit systems. This is especially important
for electron-volts, commonly used in plasma physics to denote
temperature, although it is technically a unit of energy.

Most functions expect `astropy.units.Quantity` as input, however some
will use the `~plasmapy.utils.decorators.validate_quantities` decorator
to automatically cast arguments to Quantities with appropriate units. If
that happens, you will be notified via an `astropy.units.UnitsWarning`.

Please note that well maintained physical constant data with units and
uncertainties can be found in `astropy.constants`.

For a general overview of how unit-based input works, take a look at the
following example:


Examples
========

.. nbgallery::

    /notebooks/physics

Notes for developers
====================

Values should be returned as an Astropy Quantity in SI units.

If a quantity has several names, then the function name should be the
one that provides the most physical insight into what the quantity
represents.  For example, 'gyrofrequency' indicates gyration, while
Larmor frequency indicates that this frequency is somehow related to a
human (or perhaps a cat?) named Larmor.  Similarly, using omega_ce as
a function name for this quantity will make the code less readable to
people who are unfamiliar with the notation or use a different symbol.

The docstrings for plasma parameter methods should describe the
physics associated with these quantities in ways that are
understandable to students who are taking their first course in plasma
physics while still being useful to experienced plasma physicists.

API
===

.. automodapi:: plasmapy.formulary
