"""
Module for defining the base framework of the plasma classes.
"""
__all__ = ["BasePlasma", "GenericPlasma"]

from abc import ABC, abstractmethod


class BasePlasma(ABC):
    """
    Registration class for `~plasmapy.plasma.GenericPlasma` and declares
    some abstract methods for data common in different kinds of plasmas.

    This class checks for the existence of a method named ``is_datasource_for``
    when a subclass of `GenericPlasma` is defined. If it exists it will add that
    class to the registry.
    """

    # GenericPlasma subclass registry
    _registry = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "is_datasource_for"):
            cls._registry[cls] = cls.is_datasource_for

    # This class is supposed to declare abstract methods (@abstractmethod or
    # @abstractproperty as appropriate) that are common in most plasmas
    # (like `electronTemperature`, `ionTemperature`,
    # `electronDensitry`, `ionDensity`, `averageIonization`, etc.)
    # where as `GenericPlasma` class will hold the definitions for these
    # abstract methods.

    # For reference, see
    # https://github.com/sunpy/ndcube/blob/main/ndcube/ndcube.py#L26

    @property
    @abstractmethod
    def electron_temperature(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def ion_temperature(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def electron_density(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def ion_density(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def average_ionization(self):
        raise NotImplementedError


class GenericPlasma(BasePlasma):
    """
    A Generic Plasma class. This class contains definitions for abstract
    methods declared in the `BasePlasma`.
    """

    def __init__(self, **kwargs):
        pass

    # The definitions for the abstract methods declared in `BasePlasma`
    # goes here.

    def electron_temperature(self):
        raise NotImplementedError

    def ion_temperature(self):
        raise NotImplementedError

    def electron_density(self):
        raise NotImplementedError

    def ion_density(self):
        raise NotImplementedError

    def average_ionization(self):
        raise NotImplementedError
