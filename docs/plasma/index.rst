.. _plasma:

===============
PlasmaPy Plasma
===============

.. module:: plasmapy.plasma
.. currentmodule:: plasmapy.plasma

.. attention::

   |expect-api-changes|

Overview
--------

One of the core classes in PlasmaPy is
`~plasmapy.plasma.plasma_factory.Plasma`. In order to make it easy to work with
different plasma data in PlasmaPy, the
`~plasmapy.plasma.plasma_factory.Plasma` object provides a number of methods for
commonly existing plasmas in nature.

All Plasma objects are created using the Plasma factory
`~plasmapy.plasma.plasma_factory.Plasma`.

A number of plasma data structures are supported by subclassing this
base object. See :ref:`plasma-sources` to see a list of all of them.

Creating Plasma Objects
-----------------------

Plasma objects are constructed using the special factory class
`~plasmapy.plasma.plasma_factory.Plasma`: ::

    >>> x = plasmapy.plasma.Plasma(T_e=T_e,
    ...                            n_e=n_e,
    ...                            Z=Z,
    ...                            particle=particle)  # doctest: +SKIP

The result of a call to `~plasmapy.plasma.plasma_factory.Plasma` will be
either a `~plasmapy.plasma.plasma_base.GenericPlasma` object, or a
subclass of
`~plasmapy.plasma.plasma_base.GenericPlasma` which deals with a specific
type of data, e.g. `~plasmapy.plasma.sources.plasmablob.PlasmaBlob` or
`~plasmapy.plasma.sources.plasma3d.Plasma3D` (see :ref:`plasma-sources`
to see a list of all of them).

.. autoclass:: plasmapy.plasma.plasma_factory.PlasmaFactory
   :noindex:

Using Plasma Objects
--------------------

Once a Plasma object has been created using
`~plasmapy.plasma.plasma_factory.Plasma` it will be a instance or a
subclass of the `~plasmapy.plasma.plasma_base.GenericPlasma` class. The
documentation of
`~plasmapy.plasma.plasma_base.GenericPlasma` lists the attributes and
methods that are available on all Plasma objects.

.. _plasma-plasma:

Plasma Classes
--------------

Defined in `plasmapy.plasma.sources` are a set of
`~plasmapy.plasma.plasma_base.GenericPlasma` subclasses which convert the
keyword arguments data to the standard
`~plasmapy.plasma.plasma_base.GenericPlasma` interface. These subclasses also
provide a method, which describes to the
`Plasma <plasmapy.plasma.plasma_factory.PlasmaFactory>` factory
which the data match its plasma data structure.

.. automodapi:: plasmapy.plasma
   :noindex:
   :no-main-docstr:

.. _plasma-sources:

Plasma Subclasses
-----------------

The :class:`~plasmapy.plasma.sources.plasma3d.Plasma3D` class is a basic
structure to contain spatial information about a plasma. To initialize
a Plasma3D system, first create an instance of the
:class:`~plasmapy.plasma.sources.plasma3d.Plasma3D` class and then set
the :attr:`~plasmapy.plasma.sources.plasma3d.Plasma3D.density`,
:attr:`~plasmapy.plasma.sources.plasma3d.Plasma3D.momentum`,
:attr:`~plasmapy.plasma.sources.plasma3d.Plasma3D.pressure` and
:attr:`~plasmapy.plasma.sources.plasma3d.Plasma3D.magnetic_field`.

.. note::

   This feature is currently under development.

The :class:`~plasmapy.plasma.sources.plasmablob.PlasmaBlob` class is a
basic structure to contain just plasma parameter information about a
plasma with no associated spatial or temporal scales. To initialize a
:class:`~plasmapy.plasma.sources.plasmablob.PlasmaBlob` system, call
it with arguments: electron temperature (``T_e``) and electron density
(``n_e``). You may also optionally define the ionization (``Z``), and
relevant plasma particle (``particle``).

.. note::

   This feature is currently under development.

.. automodapi:: plasmapy.plasma.sources
   :noindex:
   :skip: HDF5Reader

Writing a new Plasma subclass
-----------------------------

Any subclass of `~plasmapy.plasma.plasma_base.GenericPlasma` which
defines a method named ``is_datasource_for`` will automatically be
registered with the
`Plasma <plasmapy.plasma.plasma_factory.PlasmaFactory>` factory. The
``is_datasource_for`` method describes the form of the data for which
the `~plasmapy.plasma.plasma_base.GenericPlasma` subclass is valid. For
example, it might check the number and types of keyword arguments. This
makes it straightforward to define your own
`~plasmapy.plasma.plasma_base.GenericPlasma` subclass for a new data
structure or a custom data source like simulated data. These classes
only have to be imported for this to work, as demonstrated by the
following example.

.. code-block:: python

    import astropy.units as u
    import plasmapy.plasma


    class FuturePlasma(plasmapy.plasma.GenericPlasma):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        # Specify a classmethod that determines if the input data matches
        # this new subclass
        @classmethod
        def is_datasource_for(cls, **kwargs):
            """
            Determines if any of keyword arguments have a dimensionless value.
            """
            for _, value in kwargs.items():
                try:
                    if value.unit == u.dimensionless_unscaled:
                        return True
                except AttributeError:
                    pass

            return False


This class will now be available through the
`Plasma <plasmapy.plasma.plasma_factory.PlasmaFactory>` factory as long
as this class has been defined, i.e. imported into the current session.

If you do not want to create a method named ``is_datasource_for`` you
can manually register your class and matching method using the following
method.

.. code-block:: python

    import plasmapy.plasma

    plasmapy.plasma.Plasma.register(FuturePlasma, FuturePlasma.some_matching_method)


API
---

.. automodapi:: plasmapy.plasma
   :noindex:
   :no-main-docstr:
