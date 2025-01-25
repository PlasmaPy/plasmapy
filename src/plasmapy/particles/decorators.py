"""Decorators for `plasmapy.particles`."""

__all__ = ["particle_input"]


import functools
import inspect
import warnings
from collections.abc import Callable, Iterable, MutableMapping
from inspect import BoundArguments
from numbers import Integral, Real
from typing import Any, TypeAlias, TypedDict, get_type_hints

import numpy as np
import wrapt

from plasmapy.particles._factory import _physical_particle_factory
from plasmapy.particles.exceptions import (
    ChargeError,
    InvalidElementError,
    InvalidIonError,
    InvalidIsotopeError,
    InvalidParticleError,
    ParticleError,
)
from plasmapy.particles.particle_class import CustomParticle, Particle, ParticleLike
from plasmapy.particles.particle_collections import ParticleList, ParticleListLike
from plasmapy.utils.exceptions import PlasmaPyDeprecationWarning


class _CallableDataDict(TypedDict, total=False):
    allow_custom_particles: bool
    allow_particle_lists: bool
    annotations: dict[str, Any]
    any_of: str | Iterable[str] | None
    callable_: Callable[..., Any]
    exclude: str | Iterable[str] | None
    parameters_to_process: list[str]
    require: str | Iterable[str] | None
    signature: inspect.Signature


_basic_particle_input_annotations: tuple[type | TypeAlias, ...] = (
    Particle,  # deprecated
    ParticleLike,
    ParticleListLike,
    ParticleLike | ParticleListLike,
    (Particle, Particle),  # deprecated
)
_optional_particle_input_annotations = tuple(
    annotation | None
    # remove [:-1] index in following line when dropping (Particle, Particle)
    # as a valid annotation
    for annotation in _basic_particle_input_annotations[:-1]
    if annotation != (Particle, Particle)  # temporary hack
)
_particle_input_annotations = (
    _basic_particle_input_annotations + _optional_particle_input_annotations
)


def _make_into_set_or_none(obj: Any) -> Iterable[str] | None:
    """
    Return `None` if ``obj`` is `None`, and otherwise convert ``obj``
    into a `set`.

    If ``obj`` is a string, then a `set` containing only ``obj`` will
    be returned (i.e., ``obj`` will not be treated as iterable).
    """
    if obj is None:
        return obj
    return {obj} if isinstance(obj, str) else set(obj)


def _bind_arguments(
    wrapped_signature: inspect.Signature,
    callable_: Callable[..., Any],
    args: Iterable[Any],
    kwargs: MutableMapping[str, Any],
    instance: Any = None,
) -> inspect.BoundArguments:
    """
    Bind the arguments provided by ``args`` and ``kwargs`` to
    the corresponding parameters in the signature of the callable_
    or method being decorated.

    Parameters
    ----------
    wrapped_signature : `inspect.Signature`
        The signature of the function or method to which to bind
        ``args`` and ``kwargs``.

    callable_ : callable
        The function or method to which to bind ``args`` and ``kwargs``.
        This argument is only needed for a deprecation warning message.

    args : tuple, optional
        Positional arguments.

    kwargs : `dict` of `str` to `object`, optional
        Keyword arguments.

    instance
        If ``callable_`` is a class instance method, then ``instance``
        should be the instance to which ``callable_`` belongs.

    Returns
    -------
    dict
        A dictionary with the parameters of ``callable_`` as keys and
        the corresponding arguments as values, but removing ``self`` and
        ``cls``.
    """

    # We should keep the warning about "z_mean" for perhaps ∼2
    # releases following the last pull request that removes a "z_mean"
    # parameter from a callable decorated with @particle_input. After
    # that, we should change this warning to an exception for ∼2 more
    # releases before deleting it.

    if "z_mean" in kwargs and "Z" not in kwargs and "Z" in wrapped_signature.parameters:
        function_name = getattr(callable_, "__name__", None)
        name_clause = f"to '{function_name}' " if function_name else ""

        warnings.warn(
            f"The 'z_mean' parameter {name_clause}has been deprecated "
            "and will be removed in a subsequent release. Define the "
            "(mean) charge number with 'Z' instead.",
            category=PlasmaPyDeprecationWarning,
        )

        kwargs["Z"] = kwargs.pop("z_mean")

    # When decorating a callable_ or staticmethod, instance will
    # be None. When decorating a class instance method, instance
    # will be the class instance, and will need to be bound to
    # the "self" parameter but later removed. For a class method,
    # it will be bound to the "cls" parameter instead.

    if instance is None:
        bound_arguments = wrapped_signature.bind(*args, **kwargs)
    else:
        bound_arguments = wrapped_signature.bind(instance, *args, **kwargs)

    bound_arguments.apply_defaults()

    bound_arguments.arguments.pop("self", None)
    bound_arguments.arguments.pop("cls", None)

    return bound_arguments


class _ParticleInput:
    """
    Processes arguments for |particle_input|.

    Parameters
    ----------
    callable_ : callable
        The callable_ or method to be decorated.

    require : `str`, `set`, `list`, or `tuple`, |keyword-only|, optional
        Categories that a particle must be in.  If a particle is not in
        all of these categories, then a |ParticleError| will be raised.

    any_of : `str`, `set`, `list`, or `tuple`, |keyword-only|, optional
        Categories that a particle may be in.  If a particle is not in
        any of these categories, then a |ParticleError| will be raised.

    exclude : `str`, `set`, `list`, or `tuple`, |keyword-only|, optional
        Categories that a particle cannot be in.  If a particle is in
        any of these categories, then a |ParticleError| will be raised.

    allow_custom_particles : bool, |keyword-only|, default: `True`
        If `True`, allow |CustomParticle| instances to be passed through.

    allow_particle_lists : bool, |keyword-only|, default: `True`
        If `True`, allow |ParticleList| instances to be passed through.
    """

    def __init__(
        self,
        callable_: Callable[..., Any],
        *,
        require: str | Iterable[str] | None = None,
        any_of: str | Iterable[str] | None = None,
        exclude: str | Iterable[str] | None = None,
        allow_custom_particles: bool = True,
        allow_particle_lists: bool = True,
    ) -> None:
        self._data: _CallableDataDict = {}
        self.callable_: Callable[..., Any] = callable_
        self.require = require
        self.any_of = any_of
        self.exclude = exclude
        self.allow_custom_particles = allow_custom_particles
        self.allow_particle_lists = allow_particle_lists

    @property
    def callable_(self) -> Callable[..., Any]:
        """
        The callable that is being decorated.

        Returns
        -------
        callable
        """
        return self._data["callable_"]

    @callable_.setter
    def callable_(self, callable_: Callable[..., Any]) -> None:
        self._data["callable_"] = callable_
        self._data["annotations"] = get_type_hints(callable_)
        self._data["parameters_to_process"] = self.find_parameters_to_process()
        self._data["signature"] = inspect.signature(callable_)

    @property
    def signature(self) -> inspect.Signature:
        """The signature of the wrapped callable."""
        return self._data["signature"]

    def find_parameters_to_process(self) -> list[str]:
        """
        Identify the parameters that have annotations to indicate that
        they should be processed.

        Returns
        -------
        `list` of `str`
        """
        return [
            parameter
            for parameter, annotation in self.annotations.items()
            if annotation in _particle_input_annotations and parameter != "return"
        ]

    @property
    def annotations(self) -> dict[str, Any]:
        """
        The annotations of the decorated callable_.

        Returns
        -------
        `dict` of `str` to `object`
        """
        return self._data.get("annotations")  # type: ignore[return-value]

    @property
    def require(self) -> Iterable[str] | None:
        """
        Categories that the particle must belong to.

        Returns
        -------
        `set` of `str`, or `None`
        """
        return self._data["require"]

    @require.setter
    def require(self, require_: str | Iterable[str] | None) -> None:
        self._data["require"] = _make_into_set_or_none(require_)

    @property
    def any_of(self) -> Iterable[str] | None:
        """
        Categories of which the particle must belong to at least one.

        Returns
        -------
        `set` of `str`, or `None`
        """
        return self._data["any_of"]

    @any_of.setter
    def any_of(self, any_of_: str | Iterable[str] | None) -> None:
        self._data["any_of"] = _make_into_set_or_none(any_of_)

    @property
    def exclude(self) -> Iterable[str] | None:
        """
        Categories that the particle cannot belong to.

        Returns
        -------
        `set` of `str`, or `None`
        """
        return self._data["exclude"]

    @exclude.setter
    def exclude(self, exclude_: str | Iterable[str] | None) -> None:
        self._data["exclude"] = _make_into_set_or_none(exclude_)

    @property
    def allow_custom_particles(self) -> bool:
        """
        If `True`, then the decorated argument may be or include
        |CustomParticle| instances.

        Returns
        -------
        bool
        """
        return self._data["allow_custom_particles"]

    @allow_custom_particles.setter
    def allow_custom_particles(self, allow_custom_particles_: bool) -> None:
        self._data["allow_custom_particles"] = allow_custom_particles_

    @property
    def allow_particle_lists(self) -> bool:
        """
        If `True`, then the decorated argument may be a |ParticleList|.

        Returns
        -------
        bool
        """
        return self._data["allow_particle_lists"]

    @allow_particle_lists.setter
    def allow_particle_lists(self, allow_particle_lists_: bool) -> None:
        self._data["allow_particle_lists"] = allow_particle_lists_

    @property
    def parameters_to_process(self) -> list[str]:
        """
        The parameters of
        `~plasmapy.particles.decorators._ParticleInput.callable_` that have
        annotations to be processed by |particle_input|.

        Returns
        -------
        `list` of `str`
        """
        return self._data["parameters_to_process"]

    def verify_charge_categorization(
        self, particle: Particle | CustomParticle | ParticleList
    ) -> None:
        """
        Raise an exception if the particle does not meet charge
        categorization criteria.

        Raises
        ------
        |ChargeError|
            If the particle is required to have charge information and
            does not, or if the particle is required to be charged and
            is either uncharged or lacks charge information.
        """
        must_be_charged = self.require is not None and "charged" in self.require
        must_have_charge_info = self.any_of == {"charged", "uncharged"}

        if isinstance(particle, ParticleList):
            uncharged = particle.is_category("uncharged", particlewise=True)
            lacks_charge_info = particle.is_category(
                exclude={"charged", "uncharged"}, particlewise=True
            )
        else:
            uncharged = particle.is_category("uncharged")
            lacks_charge_info = particle.is_category(exclude={"charged", "uncharged"})

        if isinstance(uncharged, Iterable):
            uncharged = any(uncharged)
            lacks_charge_info = any(lacks_charge_info)

        if must_be_charged and (uncharged or must_have_charge_info):
            raise ChargeError(f"{self.callable_} can only accept charged particles.")

        if must_have_charge_info and lacks_charge_info:
            raise ChargeError(
                f"{self.callable_} can only accept particles which have "
                f"explicit charge information."
            )

    @staticmethod
    def category_errmsg(
        particle: Particle | CustomParticle | ParticleList,
        require: str | Iterable[str] | None,
        exclude: str | Iterable[str] | None,
        any_of: str | Iterable[str] | None,
        callable_name: str,
    ) -> str:
        """
        Return an error message for when a particle does not meet
        categorization criteria.

        Returns
        -------
        str
        """
        category_errmsg = (
            f"The particle {particle} does not meet the required "
            f"classification criteria to be a valid input to {callable_name}. "
        )

        errmsg_table = [
            (require, "must belong to all"),
            (any_of, "must belong to any"),
            (exclude, "cannot belong to any"),
        ]

        for condition, phrase in errmsg_table:
            if condition:
                category_errmsg += (
                    f"The particle {phrase} of the following categories: {condition}. "
                )

        return category_errmsg

    def verify_particle_categorization(
        self, particle: Particle | CustomParticle | ParticleList
    ) -> None:
        """
        Verify that the particle meets the categorization criteria.

        Parameters
        ----------
        particle : Particle | CustomParticle

        Raises
        ------
        |ParticleError|
            If the particle does not meet the categorization criteria.

        Notes
        -----
        This method does not yet work with |ParticleList| objects.

        See Also
        --------
        ~plasmapy.particles.particle_class.Particle.is_category
        """
        particle_in_category = particle.is_category(
            require=self.require,
            any_of=self.any_of,
            exclude=self.exclude,
        )
        if not particle_in_category:
            errmsg = self.category_errmsg(
                particle,
                self.require,
                self.exclude,
                self.any_of,
                self.callable_.__name__,
            )
            raise ParticleError(errmsg)

    def verify_particle_name_criteria(
        self, parameter: str, particle: Particle | CustomParticle | ParticleList
    ) -> None:
        """
        Check that parameters with special names meet the expected
        categorization criteria.
        """

        if (
            parameter == "ion"
            and isinstance(particle, CustomParticle)
            and not np.isnan(particle.charge)
            and particle.mass.value > 0
        ):
            return None

        name_categorization_exception: list[
            tuple[str, dict[str, str | Iterable[str] | None], type]
        ] = [
            ("element", {"require": "element"}, InvalidElementError),
            ("isotope", {"require": "isotope"}, InvalidIsotopeError),
            (
                "ion",
                {"require": "element", "any_of": {"charged", "uncharged"}},
                InvalidIonError,
            ),
        ]

        for name, categorization, exception in name_categorization_exception:
            if parameter != name or particle is None:
                continue

            if isinstance(particle, ParticleList):
                meets_name_criteria = particle.is_category(
                    **categorization, particlewise=True
                )
            else:
                meets_name_criteria = particle.is_category(**categorization)

            if isinstance(particle, Iterable) and not isinstance(particle, str):
                meets_name_criteria = all(meets_name_criteria)  # type: ignore[arg-type]

            if not meets_name_criteria:
                raise exception(
                    f"The argument {parameter} = {particle!r} to "
                    f"{self.callable_.__name__} does not correspond to a "
                    f"valid {parameter}."
                )

    def verify_allowed_types(
        self, particle: Particle | CustomParticle | ParticleList
    ) -> None:
        """
        Verify that the particle object contains only the allowed types
        of particles.
        """
        if not self.allow_custom_particles and isinstance(particle, CustomParticle):
            raise InvalidParticleError(
                f"{self.callable_.__name__} does not accept CustomParticle "
                f"or CustomParticle-like inputs."
            )

        if not self.allow_particle_lists and isinstance(particle, ParticleList):
            raise InvalidParticleError(
                f"{self.callable_.__name__} does not accept ParticleList "
                "or particle-list-like inputs."
            )

        if (
            not self.allow_custom_particles
            and isinstance(particle, ParticleList)
            and any(particle.is_category("custom", particlewise=True))  # type: ignore[arg-type]
        ):
            raise InvalidParticleError(
                f"{self.callable_.__name__} does not accept CustomParticle "
                f"or CustomParticle-like inputs."
            )

    def process_argument(
        self,
        parameter: str,
        argument: Any,
        Z: float | None,
        mass_numb: int | None,
    ) -> Any:
        """
        Process an argument that has an appropriate annotation.

        If the annotation is not one covered by |particle_input|, then
        this method will return ``argument`` without alteration.

        Otherwise, if the annotation is ``Optional[...]`` and
        ``argument`` is `None`, then this method will return `None`.

        Otherwise, this method will use ``argument`` (and possibly ``Z``
        and/or ``mass_numb``) to create a |Particle|, |CustomParticle|,
        or |ParticleList|. If the resulting particle object does not
        meet the specified criteria (if provided), then this method will
        raise an exception. Otherwise, the resulting particle object
        will be returned. See the docstring for |particle_input| for
        more details.

        Parameters
        ----------
        parameter : str
            The name of the |parameter| that was decorated.

        argument : object
            The value of the |argument| associated with ``parameter``.

        Z : integer, optional
            The |charge number| of an ion or neutral particle.

        mass_numb : integer, optional
            The mass number of an isotope.

        Returns
        -------
        object
            This method will return a |Particle|, |CustomParticle|,
            |ParticleList|, or `None` if the parameter has an annotation
            as described in the docstring for |particle_input|. For all
            other annotations, this method will return ``argument``
            without alteration.
        """
        annotation = self.annotations.get(parameter)

        if annotation not in _particle_input_annotations:
            return argument

        if annotation in _optional_particle_input_annotations and argument is None:
            return argument

        # This does not yet include cases like Optional[ParticleList],
        # Union[ParticleList, ParticleLike], etc. and thus needs updating.

        if annotation == (Particle, Particle):  # deprecated
            if not hasattr(argument, "__len__") or len(argument) != 2:
                raise ValueError(f"The length of {argument} must be 2.")
            return Particle(argument[0]), Particle(argument[1])
        elif annotation in [ParticleList, ParticleListLike]:
            # Apply casting to arguments annotated with ParticleList and ParticleListLike
            # If the argument is already an iterable, it will be cast to a ParticleList
            # otherwise, cast it to a list
            if (
                not isinstance(argument, ParticleList)
                and (not isinstance(argument, Iterable) or isinstance(argument, str))
            ):  # TODO: figure out the python hierarchy for arrays like this so we can find the difference Iterable \ str
                argument = [argument]

        if annotation in _basic_particle_input_annotations and argument is None:
            raise TypeError(f"{parameter} may not be None.")

        particle = _physical_particle_factory(argument, Z=Z, mass_numb=mass_numb)

        self.verify_charge_categorization(particle)
        self.verify_particle_categorization(particle)
        self.verify_particle_name_criteria(parameter, particle)
        self.verify_allowed_types(particle)

        return particle

    parameters_to_skip = ("Z", "mass_numb")

    def perform_pre_validations(self, Z: float | None, mass_numb: int | None) -> None:
        """
        Perform a variety of pre-checks on the arguments.

        Check that there are annotated parameters. Check that ``Z`` is
        a real number if not `None`. Check that ``mass_numb`` is an
        integer if not `None`. Verify that ``Z`` and ``mass_numb`` are
        not included if there are multiple annotated parameters.
        """

        if not self.parameters_to_process:
            raise ParticleError(
                "No parameters have an annotation that will invoke particle_input."
            )

        Z_or_mass_numb = Z is not None or mass_numb is not None
        multiple_annotated_parameters = len(self.parameters_to_process) > 1

        if Z is not None and not isinstance(Z, Real):
            raise TypeError("Z must be a real number.")

        if mass_numb is not None and not isinstance(mass_numb, Integral):
            raise TypeError("mass_numb must be an integer.")

        if Z_or_mass_numb and multiple_annotated_parameters:
            raise ParticleError(
                "The arguments Z and mass_numb are not allowed when more "
                "than one argument or keyword is annotated with ParticleLike "
                "in callables decorated with @particle_input."
            )

    def process_arguments(
        self,
        args: Iterable[Any],
        kwargs: MutableMapping[str, Any],
        instance: Any = None,
    ) -> BoundArguments:
        """
        Process the arguments passed to the callable_ callable.

        Parameters
        ----------
        args : tuple
            Positional arguments passed to the callable_ callable.

        kwargs : `dict` of `str` to `object`
            Keyword arguments provided to the callable_ callable.

        instance : `object`, optional
            If the callable_ callable is a class instance method, then
            ``instance`` should be the class instance to which ``func``
            belongs.

        Notes
        -----
        This method does not work when there are positional arguments
        before variadic positional arguments.  See :issue:`2150`.
        """

        bound_arguments = _bind_arguments(
            self.signature, self.callable_, args, kwargs, instance
        )

        Z = bound_arguments.arguments.pop("Z", None)
        mass_numb = bound_arguments.arguments.pop("mass_numb", None)

        self.perform_pre_validations(Z, mass_numb)

        processed_kwargs = {
            parameter: self.process_argument(parameter, argument, Z, mass_numb)
            for parameter, argument in bound_arguments.arguments.items()
        }

        for parameter, processed_kwarg in processed_kwargs.items():
            bound_arguments.arguments[parameter] = processed_kwarg

        return bound_arguments


def particle_input(
    callable_: Callable[..., Any] | None = None,
    *,
    require: str | Iterable[str] | None = None,
    any_of: str | Iterable[str] | None = None,
    exclude: str | Iterable[str] | None = None,
    allow_custom_particles: bool = True,
    allow_particle_lists: bool = True,
) -> Callable[..., Any]:
    r"""Convert |particle-like| |arguments| into particle objects.

    When a callable is |decorated| with |particle_input|,
    |particle-like| arguments that are appropriately |annotated| (i.e.,
    with |ParticleLike| or |ParticleListLike|) will be converted into a
    |Particle|, |CustomParticle|, or |ParticleList|.

    The parameters ``Z`` and ``mass_numb`` may be used to specify the
    |charge number| of an ion and mass number of an isotope,
    respectively, as long as ``Z`` and/or ``mass_numb`` are
    |parameters| of the callable and only one parameter is
    annotated with |ParticleLike| or |ParticleListLike|.

    To indicate that `None` can be passed to a parameter, annotate it
    with :py:`ParticleLike | None` or :py:`ParticleListLike | None`.

    If the particle representation does not satisfy any categorization
    criteria that have been provided, then |particle_input| will raise
    an exception.

    If the annotated parameter is named ``element``, ``isotope``, or
    ``ion``, then |particle_input| will raise an exception if the
    argument provided to the callable is not consistent with
    parameter.

    .. note::

       An annotated parameter named ``ion`` will accept neutral atoms
       and |CustomParticle|\ -like objects as long as the
       |charge number| is explicitly defined. To enforce that the
       particle be charged, provide :py:`require={"charged"}` to
       |particle_input|.

    .. note::

       When both |particle_input| and |validate_quantities| are used to
       decorate a :term:`function`, they may be used in either order.
       When using both |particle_input| and |validate_quantities| to
       decorate an instance :term:`method`, |particle_input| should be
       the outer decorator and |validate_quantities| should be the inner
       decorator (see :issue:`2035`).

       .. code-block:: python

          import astropy.units as u
          from plasmapy.particles import particle_input, ParticleLike
          from plasmapy.utils.decorators.validators import validate_quantities


          class SomeClass:
              @particle_input
              @validate_quantities
              def instance_method(self, particle: ParticleLike, B: u.Quantity[u.T]): ...

    .. note::

       When decorating a class method with |particle_input|,
       `classmethod` should be the outer decorator and |particle_input|
       should be the inner decorator, and the first argument
       (representing the class) must be named ``cls``.

    Parameters
    ----------
    callable_ : callable, optional
        The function or method to be decorated.

    require : `str` | `set` | `list` | `tuple`, |keyword-only|, optional
        Categories that each particle are required to be in.

    any_of : `str` | `set` | `list` | `tuple`, |keyword-only|, optional
        Categories of which each particle must belong to at least one.

    exclude : `str` | `set` | `list` | `tuple`, |keyword-only|, optional
        Categories that each particle cannot be in.

    allow_custom_particles : bool, |keyword-only|, default: `True`
        If `True`, allow |CustomParticle| instances to be passed through.

    allow_particle_lists : bool, |keyword-only|, default: `True`
        If `True`, allow |ParticleList| instances to be passed through.

    Returns
    -------
    callable

    Raises
    ------
    `TypeError`
        If the annotated argument is not |particle-like|; or if ``Z`` or
        ``mass_numb`` is not an integer.

    |InvalidParticleError|
        If the annotated argument does not correspond to a valid
        particle, ``allow_custom_particles`` is `False` and the argument
        corresponds to a |CustomParticle|, or ``allow_particle_lists``
        is `False` and the argument corresponds to a |ParticleList|.

    |InvalidParticleError|
        If the decorated argument has charge and/or mass number
        information, and ``Z`` and/or ``mass_numb`` contain
        contradictory information.

    |InvalidElementError|
        If an annotated argument is named ``element``, and the input
        does not correspond to an element, isotope, or ion.

    |InvalidIsotopeError|
        If an annotated argument is named ``isotope``, and the input
        does not correspond to an isotope or an ion of an isotope.

    |InvalidIonError|
        If an annotated argument is named ``ion``, and the input does
        not correspond to an ion.

    |ChargeError|
        If ``"charged"`` is in the ``require`` argument and the particle
        is not explicitly charged, or if
        :py:`any_of = {"charged", "uncharged"}` and the particle does
        not have charge information associated with it.

    |ParticleError|
        If the returned particle(s) do not meet the categorization
        criteria specified through ``require``, ``any_of``, or
        ``exclude``; or if none of the parameters of ``callable_`` have
        been appropriately annotated.

    `~astropy.units.UnitConversionError`
        If the annotated argument is a |Quantity|, but does not have a
        physical type of mass or charge.

    Warns
    -----
    : `~plasmapy.particles.exceptions.ParticleWarning`
        If decorated argument has charge and/or mass number information,
        and ``Z`` and/or ``mass_numb`` contain redundant information.

    See Also
    --------
    ~plasmapy.particles.particle_class.CustomParticle
    ~plasmapy.particles.particle_class.Particle
    ~plasmapy.particles.particle_collections.ParticleList
    ~plasmapy.utils.decorators.validators.validate_quantities

    Notes
    -----
    There are some known limitations to |particle_input|.

    - Particle categorization criteria are not yet applied to
      arguments that get converted into a |ParticleList| (see
      :issue:`2048`).

    - This decorator is not compatible with setters (see
      :issue:`2507`).

    - |particle_input| has limited compatibility with positional-only,
      variadic positional, and variadic keyword arguments (see
      :issue:`2150`).

    - When |particle_input| and |validate_quantities| are both
      used to decorate an instance method on a class, |particle_input|
      must be the outside decorator (see :issue:`2035`).

    - Because it dynamically changes arguments, functions decorated with
      |particle_input| often do not work well with static type checkers
      like mypy. These errors may be silenced by commenting
      :py:`# type: ignore[union-attr]` on a line of code, where
      ``union-attr`` is the name of the mypy error code.

    Examples
    --------
    The |particle_input| decorator takes appropriately annotated
    |particle-like| or |particle-list-like| arguments that are provided
    to ``callable_``, and converts them into a |Particle|,
    |CustomParticle|, or |ParticleList| object.

    The following decorated function accepts a |particle-like| input and
    returns the corresponding particle object.

    >>> from plasmapy.particles import particle_input, ParticleLike
    >>> import astropy.units as u

    >>> @particle_input
    ... def get_particle(particle: ParticleLike):
    ...     return particle

    >>> get_particle("p+")
    Particle("p+")
    >>> get_particle(["p+", "e-"])
    ParticleList(['p+', 'e-'])
    >>> get_particle(1e-26 * u.kg)
    CustomParticle(mass=1e-26 kg, charge=nan C)

    To allow `None` to pass, use :py:`ParticleLike | None` as the
    annotation.

    >>> from typing import Optional
    >>> @particle_input
    ... def get_particle_or_none(particle: ParticleLike | None):
    ...     return particle
    >>> get_particle_or_none("p+")
    Particle("p+")
    >>> print(get_particle_or_none(None))
    None

    If the decorated callable has parameters named ``Z`` and/or
    ``mass_numb`` and exactly one annotated parameter, then ``Z`` may
    be used to provide the |charge number| and ``mass_numb`` may be used
    to provide the mass number. Making ``Z`` and ``mass_numb``
    |keyword-only| reduces the chances of confusion or mistakes.

    >>> @particle_input
    ... def make_particle(particle: ParticleLike, *, Z=None, mass_numb=None):
    ...     return particle
    >>> make_particle("H", Z=0, mass_numb=3)
    Particle("T 0+")

    Instance methods can also be decorated with |particle_input| as long
    as the first argument (representing the instance itself) is named
    ``self`` following standard convention.

    >>> class SampleClass:
    ...     @particle_input
    ...     def __init__(self, particle: ParticleLike):
    ...         self.particle = particle
    ...
    ...     @particle_input
    ...     def return_particle(self, new_particle: ParticleLike):
    ...         return new_particle

    >>> instance = SampleClass("α")
    >>> instance.particle
    Particle("He-4 2+")
    >>> instance.return_particle("T+")
    Particle("T 1+")

    The ``allow_custom_particles`` and ``allow_particle_lists`` keyword
    arguments indicate whether ``callable_`` should accept
    |CustomParticle| and |ParticleList| objects, respectively.

    >>> @particle_input(allow_custom_particles=False, allow_particle_lists=False)
    ... def get_atomic_number(particle: ParticleLike):
    ...     return particle.atomic_number
    >>> get_atomic_number("Fe")
    26

    The ``require``, ``any_of``, and ``exclude`` keyword arguments may
    be used to specify categorization criteria that a particle must be
    consistent with. For more details, please refer to the docstring for
    `~plasmapy.particles.particle_class.Particle.is_category`.

    >>> @particle_input(require="element", any_of={"charged", "uncharged"})
    ... def return_ionic_level(particle: ParticleLike):
    ...     return particle
    >>> return_ionic_level("Fe-56 0+")
    Particle("Fe-56 0+")

    When the parameter is named ``element``, ``isotope``, or ``ion``,
    then the corresponding argument must be consistent with the name.
    When the parameter is named ``ion``, then the particle(s) may also
    be a neutral atom as long as the |charge number| is explicitly
    defined.

    >>> @particle_input
    ... def mass_number(isotope: ParticleLike):
    ...     return isotope.mass_number
    >>> mass_number("D")
    2
    """

    # The following pattern comes from the docs for wrapt, and requires
    # that the arguments to the decorator are keyword-only.

    if callable_ is None:
        return functools.partial(
            particle_input,
            require=require,
            any_of=any_of,
            exclude=exclude,
            allow_custom_particles=allow_custom_particles,
            allow_particle_lists=allow_particle_lists,
        )

    particle_validator = _ParticleInput(
        callable_=callable_,
        require=require,
        any_of=any_of,
        exclude=exclude,
        allow_custom_particles=allow_custom_particles,
        allow_particle_lists=allow_particle_lists,
    )

    @wrapt.decorator
    def wrapper(
        callable__: Callable[..., Any],
        instance: Any,
        args: Iterable[Any],
        kwargs: MutableMapping[str, Any],
    ) -> Callable[..., Any]:
        bound_arguments = particle_validator.process_arguments(args, kwargs, instance)
        return callable__(  # type: ignore[no-any-return]
            *bound_arguments.args,
            **bound_arguments.kwargs,
        )

    return wrapper(callable_, instance=None, args=(), kwargs={})
