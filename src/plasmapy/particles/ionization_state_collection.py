"""
Objects for storing ionization state data for multiple elements or
isotopes.
"""

from collections.abc import Iterator

__all__ = ["IonizationStateCollection"]

from numbers import Integral, Real

import astropy.units as u
import numpy as np

from plasmapy.particles.atomic import atomic_number
from plasmapy.particles.exceptions import (
    ChargeError,
    InvalidParticleError,
    ParticleError,
)
from plasmapy.particles.ionization_state import IonicLevel, IonizationState
from plasmapy.particles.particle_class import CustomParticle, Particle, ParticleLike
from plasmapy.particles.particle_collections import ParticleList
from plasmapy.particles.symbols import particle_symbol
from plasmapy.utils.decorators import validate_quantities


def _atomic_number_and_mass_number(p: ParticleLike):
    return p.atomic_number, p.mass_number if p.isotope else 0


class IonizationStateCollection:
    """
    Describe the ionization state distributions of multiple elements
    or isotopes.

    Parameters
    ----------
    inputs : `list`, `tuple`, or `dict`
        A `list` or `tuple` of elements or isotopes (if ``T_e`` is
        provided); a `list` of `~plasmapy.particles.ionization_state.IonizationState`
        instances; a `dict` with elements or isotopes as keys and
        a `~numpy.ndarray` of ionic fractions as the values; or a `dict`
        with elements or isotopes as keys and `~astropy.units.Quantity`
        instances with units of number density.

    abundances : `dict`, |keyword-only|, optional
        A `dict` with `~plasmapy.particles.particle_class.ParticleLike`
        objects used as the keys and the corresponding relative abundance as the
        values.  The values must be positive real numbers.

    log_abundances : `dict`, |keyword-only|, optional
        A `dict` with `~plasmapy.particles.particle_class.ParticleLike`
        objects used as the keys and the corresponding base 10 logarithms of their
        relative abundances as the values.  The values must be real numbers.

    n0 : `~astropy.units.Quantity`, |keyword-only|, optional
        The number density normalization factor corresponding to the
        abundances.  The number density of each element is the product
        of its abundance and ``n0``.

    T_e : `~astropy.units.Quantity`, |keyword-only|, optional
        The electron temperature in units of temperature or thermal
        energy per particle.

    kappa : `float`, |keyword-only|, optional
        The value of kappa for a kappa distribution function.

    tol : `float` or `integer`, |keyword-only|, default: ``1e-15``
        The absolute tolerance used by `~numpy.isclose` when testing
        normalizations and making comparisons.

    Raises
    ------
    `~plasmapy.particles.exceptions.ParticleError`
        If this class cannot be instantiated.

    See Also
    --------
    ~plasmapy.particles.ionization_state.IonicLevel
    ~plasmapy.particles.ionization_state.IonizationState

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.particles import IonizationStateCollection
    >>> states = IonizationStateCollection(
    ...     {"H": [0.5, 0.5], "He": [0.95, 0.05, 0]},
    ...     T_e=1.2e4 * u.K,
    ...     n0=1e15 * u.m**-3,
    ...     abundances={"H": 1, "He": 0.08},
    ... )
    >>> states.ionic_fractions
    {'H': array([0.5, 0.5]), 'He': array([0.95, 0.05, 0.  ])}

    The number densities are given by the ionic fractions multiplied by
    the abundance and the number density scaling factor ``n0``.

    >>> states.number_densities["H"]
    <Quantity [5.e+14, 5.e+14] 1 / m3>
    >>> states["He"] = [0.4, 0.59, 0.01]

    To change the ionic fractions for a single element, use item
    assignment.

    >>> states = IonizationStateCollection(["H", "He"])
    >>> states["H"] = [0.1, 0.9]

    Item assignment will also work if you supply number densities.

    >>> states["He"] = [0.4, 0.6, 0.0] * u.m**-3
    >>> states.ionic_fractions["He"]
    array([0.4, 0.6, 0. ])
    >>> states.number_densities["He"]
    <Quantity [0.4, 0.6, 0. ] 1 / m3>

    Notes
    -----
    No more than one of ``abundances`` and ``log_abundances`` may be
    specified.

    If the value provided during item assignment is a
    `~astropy.units.Quantity` with units of number density that retains
    the total element density, then the ionic fractions will be set
    proportionately.

    When making comparisons between
    `~plasmapy.particles.ionization_state_collection.IonizationStateCollection`
    instances, `~numpy.nan` values are treated as equal.  Equality tests
    are performed to within a tolerance of ``tol``.
    """

    # TODO: Improve explanation of dunder methods in docstring

    # TODO: Add functionality to equilibrate initial ionization states

    @validate_quantities(T_e={"equivalencies": u.temperature_energy()})
    def __init__(
        self,
        inputs: dict[str, np.ndarray] | list | tuple,
        *,
        T_e: u.Quantity[u.K] = np.nan * u.K,
        abundances: dict[str, float] | None = None,
        log_abundances: dict[str, float] | None = None,
        n0: u.Quantity[u.m**-3] = np.nan * u.m**-3,
        tol: float = 1e-15,
        kappa: float = np.inf,
    ) -> None:
        set_abundances = True
        if isinstance(inputs, dict) and np.all(
            [isinstance(fracs, u.Quantity) for fracs in inputs.values()]
        ):
            right_units = np.all(
                [fracs[0].si.unit == u.m**-3 for fracs in inputs.values()]
            )
            if not right_units:
                raise ParticleError(
                    "Units must be inverse volume for number densities."
                )
            abundances_provided = abundances is not None or log_abundances is not None

            if abundances_provided:
                raise ParticleError(
                    "Abundances cannot be provided if inputs "
                    "provides number density information."
                )
            set_abundances = False

        try:
            self._pars = {}
            self.T_e = T_e
            self.n0 = n0
            self.tol = tol
            self.ionic_fractions = inputs
            if set_abundances:
                self.abundances = abundances
                self.log_abundances = log_abundances
            self.kappa = kappa
        except (ValueError, TypeError) as exc:
            raise ParticleError(
                "Unable to create IonizationStateCollection object."
            ) from exc

    def __len__(self) -> int:
        return len(self._base_particles)

    def __str__(self) -> str:
        return f"<IonizationStateCollection for: {', '.join(self.base_particles)}>"

    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, *values) -> IonizationState | IonicLevel:
        errmsg = f"Invalid indexing for IonizationStateCollection instance: {values[0]}"

        one_input = not isinstance(values[0], tuple)
        two_inputs = len(values[0]) == 2

        if not one_input and not two_inputs:
            raise IndexError(errmsg)

        try:
            arg1 = values[0] if one_input else values[0][0]
            int_charge = None if one_input else values[0][1]
            particle = arg1 if arg1 in self.base_particles else particle_symbol(arg1)

            if int_charge is None:
                return IonizationState(
                    particle=particle,
                    ionic_fractions=self.ionic_fractions[particle],
                    T_e=self._pars["T_e"],
                    n_elem=np.sum(self.number_densities[particle]),
                    tol=self.tol,
                )

            if not isinstance(int_charge, Integral):
                raise TypeError(f"{int_charge} is not a valid charge for {particle}.")
            elif not 0 <= int_charge <= atomic_number(particle):
                raise ChargeError(f"{int_charge} is not a valid charge for {particle}.")

        except (ChargeError, KeyError, TypeError) as exc:
            raise IndexError(errmsg) from exc
        else:
            return IonicLevel(
                ion=particle_symbol(particle, Z=int_charge),
                ionic_fraction=self.ionic_fractions[particle][int_charge],
                number_density=self.number_densities[particle][int_charge],
            )

    def __setitem__(self, key, value) -> None:  # noqa: C901, PLR0912
        errmsg = (
            f"Cannot set item for this IonizationStateCollection instance for "
            f"key = {key!r} and value = {value!r}"
        )

        try:
            particle = particle_symbol(key)
            self.ionic_fractions[key]
        except (ParticleError, TypeError):
            raise KeyError(
                f"{errmsg} because {key!r} is an invalid particle."
            ) from None
        except KeyError:
            raise KeyError(
                f"{errmsg} because {key!r} is not one of the base "
                f"particles whose ionization state is being kept track "
                f"of."
            ) from None

        if isinstance(value, u.Quantity) and value.unit != u.dimensionless_unscaled:
            try:
                new_number_densities = value.to(u.m**-3)
            except u.UnitConversionError:
                raise ValueError(
                    f"{errmsg} because the units of value do not "
                    f"correspond to a number density."
                ) from None

            old_n_elem = np.sum(self.number_densities[particle])
            new_n_elem = np.sum(new_number_densities)

            density_was_nan = np.all(np.isnan(self.number_densities[particle]))
            same_density = u.quantity.allclose(old_n_elem, new_n_elem, rtol=self.tol)

            if not same_density and not density_was_nan:
                raise ValueError(
                    f"{errmsg} because the old element number density "
                    f"of {old_n_elem} is not approximately equal to "
                    f"the new element number density of {new_n_elem}."
                )

            value = (new_number_densities / new_n_elem).to(u.dimensionless_unscaled)

            # If the abundance of this particle has not been defined,
            # then set the abundance if there is enough (but not too
            # much) information to do so.

            abundance_is_undefined = np.isnan(self.abundances[particle])
            isnan_of_abundance_values = np.isnan(list(self.abundances.values()))
            all_abundances_are_nan = np.all(isnan_of_abundance_values)
            n_is_defined = not np.isnan(self.n0)

            if abundance_is_undefined:
                if n_is_defined:
                    self._pars["abundances"][particle] = new_n_elem / self.n0
                elif all_abundances_are_nan:
                    self.n0 = new_n_elem
                    self._pars["abundances"][particle] = 1
                else:
                    raise ParticleError(
                        f"Cannot set number density of {particle} to "
                        f"{value * new_n_elem} when the number density "
                        f"scaling factor is undefined, the abundance "
                        f"of {particle} is undefined, and some of the "
                        f"abundances of other elements/isotopes is "
                        f"defined."
                    )

        try:
            new_fractions = np.array(value, dtype=float)
        except TypeError as exc:
            raise TypeError(
                f"{errmsg} because value cannot be converted into an "
                f"array that represents ionic fractions."
            ) from exc

        # TODO: Create a separate function that makes sure ionic
        # TODO: fractions are valid to reduce code repetition.  This
        # TODO: would probably best go as a private function in
        # TODO: ionization_state.py.

        required_nstates = atomic_number(particle) + 1
        new_nstates = len(new_fractions)
        if new_nstates != required_nstates:
            raise ValueError(
                f"{errmsg} because value must have {required_nstates} "
                f"ionization levels but instead corresponds to "
                f"{new_nstates} levels."
            )

        all_nans = np.all(np.isnan(new_fractions))
        if not all_nans and (new_fractions.min() < 0 or new_fractions.max() > 1):
            raise ValueError(
                f"{errmsg} because the new ionic fractions are not "
                f"all between 0 and 1."
            )

        normalized = np.isclose(np.sum(new_fractions), 1, rtol=self.tol)
        if not normalized and not all_nans:
            raise ValueError(
                f"{errmsg} because the ionic fractions are not normalized to one."
            )

        self._ionic_fractions[particle][:] = new_fractions.copy()

    def __iter__(self) -> Iterator:
        yield from [self[key] for key in self.ionic_fractions]

    def __eq__(self, other) -> bool:
        if not isinstance(other, IonizationStateCollection):
            return False

        if self.base_particles != other.base_particles:
            return False

        min_tol = np.min([self.tol, other.tol])

        # Check any of a whole bunch of equality measures, recalling
        # that np.nan == np.nan is False.

        for attribute in ("T_e", "n_e", "kappa"):
            this = getattr(self, attribute)
            that = getattr(other, attribute)

            this_equals_that = np.any(
                [
                    this == that,
                    this is that,
                    np.isnan(this) and np.isnan(that),
                    np.isinf(this) and np.isinf(that),
                    u.quantity.allclose(this, that, rtol=min_tol),
                ]
            )

            if not this_equals_that:
                return False

        for attribute in ("ionic_fractions", "number_densities"):
            this_dict = getattr(self, attribute)
            that_dict = getattr(other, attribute)

            for particle in self.base_particles:
                this = this_dict[particle]
                that = that_dict[particle]

                this_equals_that = np.any(
                    [
                        this is that,
                        np.all(np.isnan(this)) and np.all(np.isnan(that)),
                        u.quantity.allclose(this, that, rtol=min_tol),
                    ]
                )

                if not this_equals_that:
                    return False

        return True

    @property
    def ionic_fractions(self) -> dict[str, np.array]:
        """
        A `dict` containing the ionic fractions for each element and
        isotope.

        The keys of this `dict` are the symbols for each element or
        isotope.  The values will be `~numpy.ndarray` objects containing
        the ionic fractions for each ionization level corresponding to
        each element or isotope.
        """
        return self._ionic_fractions

    @ionic_fractions.setter
    def ionic_fractions(  # noqa: C901, PLR0912, PLR0915
        self,
        inputs: dict | list | tuple,
    ):
        """
        Set the ionic fractions.

        Notes
        -----
        The ionic fractions are initialized during instantiation of
        `~plasmapy.particles.ionization_state_collection.IonizationStateCollection`.
        After this, the only way to reset the ionic fractions via the
        ``ionic_fractions`` attribute is via a `dict` with elements or
        isotopes that are a superset of the previous elements or
        isotopes.  However, you may use item assignment of the
        `~plasmapy.particles.ionization_state.IonizationState`
        instance to assign new ionic fractions one element or isotope
        at a time.

        Raises
        ------
        `~plasmapy.particles.exceptions.ParticleError`
            If the ionic fractions cannot be set.
        """

        # A potential problem is that using item assignment on the
        # ionic_fractions attribute could cause the original attributes
        # to be overwritten without checks being performed.  We might
        # eventually want to create a new class or subclass of UserDict
        # that goes through these checks.  In the meantime, we should
        # make it clear to users to set ionic_fractions by using item
        # assignment on the IonizationStateCollection instance as a whole.  An
        # example of the problem is `s = IonizationStateCollection(["He"])` being
        # followed by `s.ionic_fractions["He"] = 0.3`.

        if hasattr(self, "_ionic_fractions"):
            if not isinstance(inputs, dict):
                raise TypeError(
                    "Can only reset ionic_fractions with a dict if "
                    "ionic_fractions has been set already."
                )
            old_particles = set(self.base_particles)
            new_particles = {particle_symbol(key) for key in inputs}
            missing_particles = old_particles - new_particles
            if missing_particles:
                raise ParticleError(
                    "Can only reset ionic fractions with a dict if "
                    "the new base particles are a superset of the "
                    "prior base particles.  To change ionic fractions "
                    "for one base particle, use item assignment on the "
                    "IonizationStateCollection instance instead."
                )

        if isinstance(inputs, dict):
            original_keys = inputs.keys()
            ionfrac_types = {type(inputs[key]) for key in original_keys}
            inputs_have_quantities = u.Quantity in ionfrac_types

            if inputs_have_quantities and len(ionfrac_types) != 1:
                raise TypeError(
                    "Ionic fraction information may only be inputted "
                    "as a Quantity object if all ionic fractions are "
                    "Quantity arrays with units of inverse volume."
                )

            try:
                particles = {key: Particle(key) for key in original_keys}
            except (InvalidParticleError, TypeError) as exc:
                raise ParticleError(
                    "Unable to create IonizationStateCollection instance "
                    "because not all particles are valid."
                ) from exc

            # The particles whose ionization states are to be recorded
            # should be elements or isotopes but not ions or neutrals.

            for particle in particles:  # noqa: PLC0206
                is_element = particles[particle].is_category("element")
                has_charge_info = particles[particle].is_category(
                    any_of=["charged", "uncharged"]
                )

                if not is_element or has_charge_info:
                    raise ParticleError(
                        f"{particle} is not an element or isotope without "
                        f"charge information."
                    )

            # We are sorting the elements/isotopes by atomic number and
            # mass number since we will often want to plot and analyze
            # things and this is the most sensible order.

            def _sort_entries_by_atomic_and_mass_numbers(k):
                return (
                    particles[k].atomic_number,
                    particles[k].mass_number if particles[k].isotope else 0,
                )

            sorted_keys = sorted(
                original_keys, key=_sort_entries_by_atomic_and_mass_numbers
            )

            _elements_and_isotopes = []
            _particle_instances = []
            new_ionic_fractions = {}

            if inputs_have_quantities:
                n_elems = {}

            for particle in sorted_keys:
                new_key = particles[particle].symbol
                _particle_instances.append(particles[particle])
                if new_key in _elements_and_isotopes:
                    raise ParticleError(
                        "Repeated particles in IonizationStateCollection."
                    )

                nstates_input = len(inputs[particle])
                nstates = particles[particle].atomic_number + 1
                if nstates != nstates_input:
                    raise ParticleError(
                        f"The ionic fractions array for {particle} must "
                        f"have a length of {nstates}."
                    )

                _elements_and_isotopes.append(new_key)
                if inputs_have_quantities:
                    try:
                        number_densities = inputs[particle].to(u.m**-3)
                        n_elem = np.sum(number_densities)
                        new_ionic_fractions[new_key] = np.array(
                            number_densities / n_elem
                        )
                        n_elems[particle] = n_elem
                    except u.UnitConversionError as exc:
                        raise ParticleError("Units are not inverse volume.") from exc
                elif (
                    isinstance(inputs[particle], np.ndarray)
                    and inputs[particle].dtype.kind == "f"
                ):
                    new_ionic_fractions[particles[particle].symbol] = inputs[particle]
                else:
                    try:
                        new_ionic_fractions[particles[particle].symbol] = np.array(
                            inputs[particle], dtype=float
                        )
                    except ValueError as exc:
                        raise ParticleError(
                            f"Inappropriate ionic fractions for {particle}."
                        ) from exc

            for particle in _elements_and_isotopes:
                fractions = new_ionic_fractions[particle]
                if not np.all(np.isnan(fractions)):
                    if np.min(fractions) < 0 or np.max(fractions) > 1:
                        raise ParticleError(
                            f"Ionic fractions for {particle} are not between 0 and 1."
                        )
                    if not np.isclose(np.sum(fractions), 1, atol=self.tol, rtol=0):
                        raise ParticleError(
                            f"Ionic fractions for {particle} are not normalized to 1."
                        )

            # When the inputs provide the densities, the abundances must
            # not have been provided because that would be redundant
            # or contradictory information.  The number density scaling
            # factor might or might not have been provided.  Have the
            # number density scaling factor default to the total number
            # of neutrals and ions across all elements and isotopes, if
            # it was not provided.  Then go ahead and calculate the
            # abundances based on that.  However, we need to be careful
            # that the abundances are not overwritten during the
            # instantiation of the class.

            if inputs_have_quantities:
                if np.isnan(self.n0):
                    new_n = 0 * u.m**-3
                    for particle in _elements_and_isotopes:
                        new_n += n_elems[particle]
                    self.n0 = new_n

                new_abundances = {}
                for particle in _elements_and_isotopes:
                    new_abundances[particle] = float(n_elems[particle] / self.n0)

                self._pars["abundances"] = new_abundances

        elif isinstance(inputs, list | tuple):
            try:
                _particle_instances = [Particle(particle) for particle in inputs]
            except (InvalidParticleError, TypeError) as exc:
                raise ParticleError(
                    "Invalid inputs to IonizationStateCollection."
                ) from exc

            _particle_instances.sort(key=_atomic_number_and_mass_number)

            _elements_and_isotopes = [
                particle.symbol for particle in _particle_instances
            ]
            new_ionic_fractions = {
                particle.symbol: np.full(
                    particle.atomic_number + 1, fill_value=np.nan, dtype=float
                )
                for particle in _particle_instances
            }
        else:
            raise TypeError("Incorrect inputs to set ionic_fractions.")

        for i in range(1, len(_particle_instances)):
            if (
                _particle_instances[i - 1].element == _particle_instances[i].element
            ) and (
                not _particle_instances[i - 1].isotope
                and _particle_instances[i].isotope
            ):
                raise ParticleError(
                    "Cannot have an element and isotopes of that element."
                )

        self._particle_instances = _particle_instances
        self._base_particles = _elements_and_isotopes
        self._ionic_fractions = new_ionic_fractions

    def normalize(self) -> None:
        """
        Normalize the ionic fractions so that the sum for each element
        equals one.
        """
        for particle in self.base_particles:
            tot = np.sum(self.ionic_fractions[particle])
            self.ionic_fractions[particle] = self.ionic_fractions[particle] / tot

    @property
    @validate_quantities
    def n_e(self) -> u.Quantity[u.m**-3]:
        """The electron number density under the assumption of quasineutrality."""
        number_densities = self.number_densities
        n_e = 0.0 * u.m**-3
        for elem in self.base_particles:
            atomic_numb = atomic_number(elem)
            number_of_ionization_states = atomic_numb + 1
            charge_numbers = np.linspace(0, atomic_numb, number_of_ionization_states)
            n_e += np.sum(number_densities[elem] * charge_numbers)
        return n_e

    @property
    @validate_quantities
    def n0(self) -> u.Quantity[u.m**-3]:
        """The number density scaling factor."""
        return self._pars["n"]

    @n0.setter
    @validate_quantities
    def n0(self, n: u.Quantity[u.m**-3]):
        """Set the number density scaling factor."""
        try:
            n = n.to(u.m**-3)
        except u.UnitConversionError as exc:
            raise ParticleError("Units cannot be converted to u.m ** -3.") from exc
        except ParticleError as exc:
            raise ParticleError(f"{n} is not a valid number density.") from exc
        if n < 0 * u.m**-3:
            raise ParticleError("Number density cannot be negative.")
        self._pars["n"] = n.to(u.m**-3)

    @property
    def number_densities(self) -> dict[str, u.Quantity]:
        """
        A `dict` containing the number densities for the elements and/or
        isotopes composing the collection.
        """
        return {
            elem: self.n0 * self.abundances[elem] * self.ionic_fractions[elem]
            for elem in self.base_particles
        }

    @property
    def abundances(self) -> dict[ParticleLike, float] | None:
        """The elemental abundances."""
        return self._pars["abundances"]

    @abundances.setter
    def abundances(self, abundances_dict: dict[ParticleLike, float] | None):
        """
        Set the elemental (or isotopic) abundances.  The elements and
        isotopes must be the same as or a superset of the elements whose
        ionization states are being tracked.
        """
        if abundances_dict is None:
            self._pars["abundances"] = {elem: np.nan for elem in self.base_particles}
        elif not isinstance(abundances_dict, dict):
            raise TypeError(
                "The abundances attribute must be a dict with "
                "elements or isotopes as keys and real numbers "
                "representing relative abundances as values."
            )
        else:
            old_keys = abundances_dict.keys()
            try:
                new_keys_dict = {}
                for old_key in old_keys:
                    new_keys_dict[particle_symbol(old_key)] = old_key
            except ParticleError as ex:
                raise ParticleError(
                    f"The key {old_key!r} in the abundances "
                    f"dictionary is not a valid element or isotope."
                ) from ex

            new_elements = new_keys_dict.keys()

            old_elements_set = set(self.base_particles)
            new_elements_set = set(new_elements)

            if old_elements_set - new_elements_set:
                raise ParticleError(
                    f"The abundances of the following particles are "
                    f"missing: {old_elements_set - new_elements_set}"
                )

            new_abundances_dict = {}

            for element in new_elements:
                inputted_abundance = abundances_dict[new_keys_dict[element]]
                try:
                    inputted_abundance = float(inputted_abundance)
                except TypeError:
                    raise TypeError(
                        f"The abundance for {element} was provided as"
                        f"{inputted_abundance}, which cannot be "
                        f"converted to a real number."
                    ) from None

                if inputted_abundance < 0:
                    raise ParticleError(f"The abundance of {element} is negative.")
                new_abundances_dict[element] = inputted_abundance

            self._pars["abundances"] = new_abundances_dict

    @property
    def log_abundances(self) -> dict[str, float]:
        """
        A `dict` with atomic or isotope symbols as keys and the base 10
        logarithms of the relative abundances as the corresponding values.
        """
        return {
            atom: np.log10(abundance) for atom, abundance in self.abundances.items()
        }

    @log_abundances.setter
    def log_abundances(self, value: dict[str, float] | None):
        """Set the base 10 logarithm of the relative abundances."""
        if value is not None:
            try:
                new_abundances_input = {
                    atom: 10**log_abundance for atom, log_abundance in value.items()
                }
                self.abundances = new_abundances_input
            except ParticleError:
                raise ParticleError("Invalid log_abundances.") from None

    @property
    def T_e(self) -> u.Quantity[u.K]:
        """The electron temperature."""
        return self._pars["T_e"]

    @T_e.setter
    @validate_quantities(electron_temperature={"equivalencies": u.temperature_energy()})
    def T_e(self, electron_temperature: u.Quantity[u.K]):
        """Set the electron temperature."""
        try:
            temperature = electron_temperature.to(
                u.K, equivalencies=u.temperature_energy()
            )
        except (AttributeError, u.UnitsError):
            raise ParticleError(
                f"{electron_temperature} is not a valid temperature."
            ) from None
        if temperature < 0 * u.K:
            raise ParticleError("The electron temperature cannot be negative.")
        self._pars["T_e"] = temperature

    @property
    def kappa(self) -> np.real:
        """
        The κ parameter for a kappa distribution function for electrons.

        The value of ``kappa`` must be greater than ``1.5`` in order to
        have a valid distribution function.  If ``kappa`` equals
        `~numpy.inf`, then the distribution function reduces to a
        Maxwellian.

        """
        return self._pars["kappa"]

    @kappa.setter
    def kappa(self, value: float):
        """
        Set the kappa parameter for a kappa distribution function for
        electrons.  The value must be between ``1.5`` and `~numpy.inf`.
        """
        kappa_errmsg = "kappa must be a real number greater than 1.5"
        if not isinstance(value, Real):
            raise TypeError(kappa_errmsg)
        if value <= 1.5:
            raise ValueError(kappa_errmsg)
        self._pars["kappa"] = np.real(value)

    @property
    def base_particles(self) -> list[str]:
        """
        A `list` of the elements and isotopes whose ionization states
        are being kept track of.
        """
        return self._base_particles

    @property
    def tol(self) -> np.real:
        """The absolute tolerance for comparisons."""
        return self._tol

    @tol.setter
    def tol(self, atol: float):
        """Set the absolute tolerance for comparisons."""
        if not isinstance(atol, Real):
            raise TypeError("The attribute tol must be a real number.")
        if 0 <= atol <= 1.0:
            self._tol = np.real(atol)
        else:
            raise ValueError("Need 0 <= tol <= 1.")

    def average_ion(
        self,
        *,
        include_neutrals: bool = True,
        use_rms_charge: bool = False,
        use_rms_mass: bool = False,
    ) -> CustomParticle:
        """
        Return a |CustomParticle| representing the mean particle
        included across all ionization states.

        By default, this method will use the weighted mean to calculate
        the properties of the |CustomParticle|, where the weights for
        each ionic level is given by its ionic fraction multiplied by
        the abundance of the base element or isotope. If
        ``use_rms_charge`` or ``use_rms_mass`` is `True`, then this
        method will return the root mean square of the charge or mass,
        respectively.

        Parameters
        ----------
        include_neutrals : `bool`, |keyword-only|, default: `True`
            If `True`, include neutrals when calculating the mean values
            of the different particles.  If `False`, exclude neutrals.

        use_rms_charge : `bool`, |keyword-only|, default: `False`
            If `True`, use the root-mean-square charge instead of the
            mean charge.

        use_rms_mass : `bool`, |keyword-only|, default: `False`
            If `True`, use the root-mean-square mass instead of the mean
            mass.

        Raises
        ------
        `~plasmapy.particles.exceptions.ParticleError`
            If the abundance of any of the elements or isotopes is not
            defined and the |IonizationStateCollection| instance includes
            more than one element or isotope.

        Returns
        -------
        ~plasmapy.particles.particle_class.CustomParticle

        Examples
        --------
        >>> states = IonizationStateCollection(
        ...     {"H": [0.1, 0.9], "He": [0, 0.1, 0.9]}, abundances={"H": 1, "He": 0.1}
        ... )
        >>> states.average_ion()
        CustomParticle(mass=2.12498...e-27 kg, charge=1.5876...e-19 C)
        >>> states.average_ion(
        ...     include_neutrals=False,
        ...     use_rms_charge=True,
        ...     use_rms_mass=True,
        ... )
        CustomParticle(mass=2.633...e-27 kg, charge=1.805...e-19 C)
        """
        min_charge = 0 if include_neutrals else 1

        all_particles = ParticleList()
        all_abundances = []

        for base_particle in self.base_particles:
            ionization_state = self[base_particle]
            ionic_levels = ionization_state.to_list()[min_charge:]
            all_particles.extend(ionic_levels)

            base_particle_abundance = self.abundances[base_particle]

            if np.isnan(base_particle_abundance):
                if len(self) == 1:
                    base_particle_abundance = 1
                else:
                    raise ParticleError(
                        "Unable to provide an average particle without abundances."
                    )

            ionic_fractions = ionization_state.ionic_fractions[min_charge:]
            ionic_abundances = base_particle_abundance * ionic_fractions
            all_abundances.extend(ionic_abundances)

        return all_particles.average_particle(
            use_rms_charge=use_rms_charge,
            use_rms_mass=use_rms_mass,
            abundances=all_abundances,
        )

    def summarize(
        self, minimum_ionic_fraction: float = 0.01
    ) -> None:  # coverage: ignore
        """
        Print quicklook information.

        Parameters
        ----------
        minimum_ionic_fraction : `float`, default: ``0.01``
            If the ionic fraction for a particular ionization state is
            below this level, then information for it will not be
            printed.

        Examples
        --------
        >>> import astropy.units as u
        >>> states = IonizationStateCollection(
        ...     {"H": [0.1, 0.9], "He": [0.95, 0.05, 0.0]},
        ...     T_e=12000 * u.K,
        ...     n0=3e9 * u.cm**-3,
        ...     abundances={"H": 1.0, "He": 0.1},
        ...     kappa=3.4,
        ... )
        >>> states.summarize()
        IonizationStateCollection instance for: H, He
        ----------------------------------------------------------------
        H  0+: 0.100    n_i = 3.00e+14 m**-3    T_i = 1.20e+04 K
        H  1+: 0.900    n_i = 2.70e+15 m**-3    T_i = 1.20e+04 K
        ----------------------------------------------------------------
        He  0+: 0.950    n_i = 2.85e+14 m**-3    T_i = 1.20e+04 K
        He  1+: 0.050    n_i = 1.50e+13 m**-3    T_i = 1.20e+04 K
        ----------------------------------------------------------------
        n_e = 2.71e+15 m**-3
        T_e = 1.20e+04 K
        kappa = 3.40
        ----------------------------------------------------------------
        """
        separator_line = 64 * "-"

        output = [
            f"IonizationStateCollection instance for: {', '.join(self.base_particles)}"
        ]

        # Get the ionic symbol with the corresponding ionic fraction and
        # number density (if available), but only for the most abundant
        # ionization levels for each element.

        for ionization_state in self:
            states_info = ionization_state._get_states_info(  # noqa: SLF001
                minimum_ionic_fraction,
            )
            if len(states_info) > 0:
                output += states_info
                output[-1] += "\n" + separator_line

        attributes = []
        if np.isfinite(self.n_e):
            attributes.append(f"n_e = {self.n_e.value:.2e} m**-3")
        if np.isfinite(self.T_e):
            attributes.append(f"T_e = {self.T_e.value:.2e} K")
        if np.isfinite(self.kappa):
            attributes.append(f"kappa = {self.kappa:.2f}")

        if attributes:
            attributes.append(separator_line)

        output.append("\n".join(attributes))

        if len(output) > 1:
            output[0] += "\n" + separator_line
            output_string = "\n".join(output)
        else:
            output_string = output[0]

        print(output_string.strip("\n"))  # noqa: T201
