class CrossSectionEnergyRangeError(Exception):
    """Exception raised for out of range cross section CM energies.

    Attributes:
        reaction -- reaction in question
        energy -- out of range energy in keV
        allowed_range -- allowed range of CM energies in keV, list of [min_range, max_range]
    """

    def __init__(self, reaction, energy, allowed_range):
        self.reaction = reaction
        self.energy = energy
        self.allowed_range = allowed_range
        message = f"Supported CM energy range for {self.reaction} cross section is {self.allowed_range} keV. {self.energy} keV is out of range."
        super().__init__(message)


class ReactivityTemperatureTooLowError(Exception):
    """Exception raised for out of range reactivity temperatures.

    Attributes:
        value -- out of range values
        allowed_range -- allowed range, list of [min_range, max_range]
    """

    def __init__(self, reaction, temperature, allowed_range):
        self.reaction = reaction
        self.temperature = temperature
        self.allowed_range = allowed_range
        message = f"Supported temperature range for {self.reaction} reactivity is {self.allowed_range} keV. {self.temperature} keV is too low and out of range."
        super().__init__(message)


class ReactivityTemperatureTooHighError(Exception):
    """Exception raised for out of range reactivity temperatures.

    Attributes:
        value -- out of range values
        allowed_range -- allowed range, list of [min_range, max_range]
    """

    def __init__(self, reaction, temperature, allowed_range):
        self.reaction = reaction
        self.temperature = temperature
        self.allowed_range = allowed_range
        message = f"Supported temperature range for {self.reaction} reactivity is {self.allowed_range} keV. {self.temperature} keV is too high and out of range."
        super().__init__(message)
