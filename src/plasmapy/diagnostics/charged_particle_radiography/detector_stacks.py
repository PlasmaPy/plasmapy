"""
Objects representing stacks of film and/or filter layers for charged
particle detectors.
"""

__all__ = [
    "Stack",
    "Layer",
]


import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d


class Layer:
    r"""
    A layer in a detector film stack.

    The layer could either be an active layer (the actual film medium) or
    an inum_active layer (a filter or inum_active part of the film, such as
    a substrate.)

    Tabulated stopping powers for protons and electrons can be found in the
    `NIST PSTAR database
    <https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html>`_
    and the
    `NIST ESTAR database
    <https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html>`_.

    Parameters
    ----------
    thickness : `~astropy.units.Quantity`
        The thickness of the layer, in units convertible to meters.

    energy_axis : `~astropy.units.Quantity`
        The energies corresponding to the stopping power array.

    stopping_power : `~astropy.units.Quantity`
        The stopping power in the material. Either the linear stopping
        power (units of J/m) or the mass stopping power
        (units convertible to J m\ :sup:`2` / kg) can be provided. If the
        mass stopping power is provided, the material_density keyword
        is required.

    mass_density : `~astropy.units.Quantity`, optional
        The material mass density in units convertible to kg/m\ :sup:`3`.
        This keyword is required if the provided stopping power is the
        mass stopping power.

    active : `bool`, default: `True`
        If `True`, this layer is marked as an active layer.

    name : `str`, optional
        An optional name for the layer.
    """

    def __init__(
        self,
        thickness: u.Quantity[u.m],
        energy_axis: u.Quantity[u.J],
        stopping_power: u.Quantity[u.J / u.m, u.J * u.m**2 / u.kg],
        mass_density: u.Quantity[u.kg / u.m**3] | None = None,
        active: bool = True,
        name: str = "",
    ) -> None:
        self.thickness = thickness
        self.energy_axis = energy_axis
        self.active = active
        self.name = name

        # Handle stopping power provided as either linear or
        # mass stopping power
        if stopping_power.unit.is_equivalent(u.J / u.m):
            self.linear_stopping_power = stopping_power.to(u.J / u.m)

        elif stopping_power.unit.is_equivalent(u.J * u.m**2 / u.kg):
            if mass_density is None:
                raise ValueError(
                    "mass_density keyword is required if "
                    "stopping power is not provided in units "
                    "convertible to J/m"
                )

            # Ensure the mass density has the right units
            try:
                mass_density = mass_density.to(u.kg / u.m**3)
            except u.UnitConversionError as e:
                raise ValueError(
                    "mass_density keyword must have units convertible to kg/m^3."
                ) from e

            self.linear_stopping_power = (stopping_power * mass_density).to(u.J / u.m)

        else:
            raise ValueError(
                "Units of stopping_power keyword not recognized:"
                f"{stopping_power.unit}"
            )


class Stack:
    r"""
    An ordered list of |Layer| objects.

    Parameters
    ----------
    layers : list of |Layer|
        The objects that make up the film stack.

    """

    def __init__(self, layers: list[Layer]) -> None:
        self._layers = layers
        self._energy_bands = None

    @property
    def num_layers(self):
        r"""
        The number of layers in the stack.
        """

        return len(self._layers)

    @property
    def num_active(self):
        r"""
        The number of layers in the stack marked ``active``.
        """

        return len([layer for layer in self._layers if layer.active])

    @property
    def thickness(self):
        r"""
        The total thickness of the stack.
        """
        thickness = np.array([layer.thickness.to(u.m).value for layer in self._layers])
        return np.sum(thickness) * u.m

    def deposition_curves(
        self, energies: u.Quantity[u.J], dx=1 * u.um, return_only_active: bool = True
    ):
        """
        Calculate the deposition of an ensemble of particles over a range of
        energies in a stack of films and filters.

        Parameters
        ----------
        energies : (``nenergies``,) `~astropy.units.Quantity` array
            Energies axis over which to calculate the deposition. Units
            convertible to J.

        dx : `~astropy.units.Quantity`, optional
            The spatial resolution of the numerical integration of the
            stopping power. Defaults to 1 μm.

        return_only_active : `bool`, default: `True`
            If `True`, only the energy bands of layers in which the
            active property is `True` will be returned. This is usually
            desirable, since particles captured in other layers will not
            be measured. If `False`, energy bands in all layers of the
            stack are returned. The default is `True`.

        Returns
        -------
        deposited : (``nlayers``, ``nenergies``) `~numpy.ndarray`
            The fraction of particles at each energy that will be deposited in
            each layer of the film. The array is normalized such that the sum
            along the first dimension (all of the layers) for each population
            is unity.
        """

        energies = energies.to(u.J).value

        deposited_energy = np.zeros([len(self._layers), energies.size])

        for i, layer in enumerate(self._layers):
            # Interpolate stopping power for each energy
            # stopping power here is in MeV/cm
            sp_fcn = interp1d(
                layer.energy_axis.to(u.J).value,
                layer.linear_stopping_power.to(u.J / u.m).value,
                fill_value=(0, np.inf),
                bounds_error=False,
            )

            # Slice the layer into sublayer dx thick
            nsublayers = int(np.floor(layer.thickness.to(u.m).value / dx.to(u.m).value))
            sublayers = np.ones(nsublayers) * dx.to(u.m)
            # Include any remainder in the last sublayer
            sublayers[-1] += layer.thickness.to(u.m) % dx.to(u.m)

            # Calculate the energy deposited in each sublayer
            # This is essentially numerically integrating the stopping power
            for ds in sublayers:
                # Interpolate the stopping power at the current energies
                interpolated_stopping_power = sp_fcn(energies)
                # dE is in MeV
                dE = interpolated_stopping_power * ds.to(u.m).value

                # If dE > E for a given energy, set dE=E (stop the particle)
                dE = np.where(dE > energies, energies, dE)

                energies += -dE
                deposited_energy[i, :] += dE

        # Normalize the deposited energy array so that each number represents
        # the fraction of a population of particles of that energy stopped
        # in that layer.
        deposited_energy /= np.sum(deposited_energy, axis=0)

        # If this flag is set, return only the layers that correspond to active
        # medium, ignoring the filter and substrate layers
        if return_only_active:
            active_ind = [i for i in range(len(self._layers)) if self._layers[i].active]
            deposited_energy = deposited_energy[active_ind, :]

        return deposited_energy

    def energy_bands(
        self,
        energy_range: u.Quantity[u.J],
        dE: u.Quantity[u.J],
        dx=1e-6 * u.m,  # noqa: ARG002
        return_only_active: bool = True,
    ):
        """
        Calculate the energy bands in each of the active layers of a film
        stack.

        Parameters
        ----------
        energy_range : (2,) `~astropy.units.Quantity` array
            A range of energies to include in the calculation. Units
            convertible to eV.

        dE :  `~astropy.units.Quantity`
            Spacing between energy bins in the calculation. Units convertible
            to J.

        dx : `~astropy.units.Quantity`, default: 1 μm
            The spatial resolution of the numerical integration of the stopping
            power. Passed directly to the `~deposition_curves` method.

        return_only_active : `bool`, default: `True`
            If `True`, only the energy bands of layers in which the active
            property is `True` will be returned. This is usually desirable,
            since particles captured in other layers will not be measured.
            If `False`, energy bands in all layers of the stack are returned.

        Returns
        -------
        energy_bands : (``nlayers``, 2) `~astropy.units.Quantity`
            The full-width-half-max energy range of the Bragg peak in each
            active layer of the film stack, in J.
        """

        energies = (
            np.arange(
                *energy_range.to(u.J).value,
                dE.to(u.J).value,
            )
            * u.J
        )

        deposited = self.deposition_curves(
            energies, return_only_active=return_only_active
        )

        energy_bands = np.zeros([deposited.shape[0], 2]) * u.J

        for i in range(deposited.shape[0]):
            bragg_curve = deposited[i, :]

            # Find the indices corresponding to half the maximum value
            # on either side of the peak
            halfmax = np.max(bragg_curve) / 2

            inds = np.argwhere(bragg_curve > halfmax)
            # Store those energies

            energy_bands[i, 0] = energies[inds[0][0]]
            energy_bands[i, 1] = energies[inds[-1][0]]

        self._energy_bands = energy_bands

        return energy_bands
