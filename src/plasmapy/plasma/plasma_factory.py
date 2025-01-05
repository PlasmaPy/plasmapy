"""
Defines the framework around the plasma factory.

.. attention::

   |expect-api-changes|
"""

__all__ = ["PlasmaFactory", "Plasma"]

from plasmapy.plasma.plasma_base import GenericPlasma
from plasmapy.utils.datatype_factory_base import BasicRegistrationFactory


class PlasmaFactory(BasicRegistrationFactory):
    """
    Plasma factory class. Used to create a variety of Plasma objects.
    Valid plasma structures are specified by registering them with the
    factory.

    .. attention::

       |expect-api-changes|
    """


Plasma = PlasmaFactory(
    default_widget_type=GenericPlasma,
    registry=GenericPlasma._registry,  # noqa: SLF001
    additional_validation_functions=["is_datasource_for"],
)
