import astropy.units as u
import numpy as np
import pytest

import plasmapy.transport.classical.base as base

from plasmapy.particles import Particle

chi_e = 10 ** (np.linspace(-2, 2, num=50))
chi_i = 10 ** (np.linspace(-3, 3, num=50))

test_input = [
    (chi_e, chi_i, 10, None),  # Nominal input
    (chi_e, chi_i, None, ValueError),  # Z cannot be None
    (None, None, 10, ValueError),  # chi_e and chi_i cannot both be None
    (chi_e, None, 10, None),  # Either chi_e or chi_i can be None
    (None, chi_i, 10, None),  # Either chi_e or chi_i can be None
]


@pytest.mark.parametrize("chi_e,chi_i,Z,error", test_input)
def test_constructor_dimensionless(chi_e, chi_i, Z, error):
    """
    Tests the dimensionless constructor
    """

    if error is None:
        base.AbstractClassicalTransportCoefficients(chi_e=chi_e, chi_i=chi_i, Z=Z)

    else:
        with pytest.raises(error):
            base.AbstractClassicalTransportCoefficients(chi_e=chi_e, chi_i=chi_i, Z=Z)


particle = Particle("H+")
B = np.random.random() * u.T
ne = np.random.random() * 1e20 * u.cm**-3
ni = np.random.random() * 1e20 * u.cm**-3
Te = np.random.random() * 1e4 * u.K
Ti = np.random.random() * 1e4 * u.K
efreq = np.random.random() / u.ns
ifreq = np.random.random() / u.ns

test_input = [
    # Nominal
    (particle, B, ne, ni, Te, Ti, efreq, ifreq, None),
    # No efreq or ifreq -> they get calculated automatically
    (particle, B, ne, ni, Te, Ti, None, None, None),
    # Particle as string
    ("H+", B, ne, ni, Te, Ti, efreq, ifreq, None),
    # B and particle are both required
    (None, B, ne, ni, Te, Ti, efreq, ifreq, ValueError),
    ("H+", None, ne, ni, Te, Ti, efreq, ifreq, ValueError),
    # Incorrect units
    (particle, 1 * u.kg, ne, ni, Te, Ti, efreq, ifreq, u.core.UnitTypeError),
    # Inconsistent shape
    (particle, B, ne, ni, np.array([1e4, 2e4]) * u.K, Ti, efreq, ifreq, ValueError),
    # Consistent shapes (no error)
    (
        particle,
        np.array([10, 20]) * u.T,
        np.array([1e20, 2e20]) * u.cm**-3,
        np.array([1e20, 2e20]) * u.cm**-3,
        np.array([1e4, 2e4]) * u.K,
        np.array([1e4, 2e4]) * u.K,
        None,
        None,
        None,
    ),
]


@pytest.mark.parametrize("particle,B,ne,ni,Te,Ti,efreq,ifreq,error", test_input)
def test_constructor_dimensional(particle, B, ne, ni, Te, Ti, efreq, ifreq, error):
    """
    Tests the dimensional constructor

    """

    if error is None:
        obj = base.AbstractClassicalTransportCoefficients(
            particle=particle,
            B=B,
            ne=ne,
            ni=ni,
            Te=Te,
            Ti=Ti,
            e_collision_freq=efreq,
            i_collision_freq=ifreq,
        )

        assert isinstance(obj.e_collision_freq, u.Quantity)
        assert isinstance(obj.i_collision_freq, u.Quantity)

    else:
        with pytest.raises(error):
            base.AbstractClassicalTransportCoefficients(
                particle=particle,
                B=B,
                ne=ne,
                ni=ni,
                Te=Te,
                Ti=Ti,
                e_collision_freq=efreq,
                i_collision_freq=ifreq,
            )


test_input = [
    # Nominal
    (particle, B, ne, ni, Te, Ti, efreq, ifreq, None),
    # No electron info given
    (particle, B, None, ni, None, Ti, None, ifreq, None),
    # No ion info given
    (particle, B, ne, None, Te, None, efreq, None, None),
]


@pytest.mark.parametrize("particle,B,ne,ni,Te,Ti,efreq,ifreq,error", test_input)
def test_constructor_just_ions_or_electrons(
    particle, B, ne, ni, Te, Ti, efreq, ifreq, error
):
    """
    Tests the dimensional constructor behavior when ion or electron terms
    are missing

    """

    obj = base.AbstractClassicalTransportCoefficients(
        particle=particle,
        B=B,
        ne=ne,
        ni=ni,
        Te=Te,
        Ti=Ti,
        e_collision_freq=efreq,
        i_collision_freq=ifreq,
    )

    if all(v is None for v in [ne, Te, efreq]):
        assert obj.chi_e is None

    elif all(v is None for v in [ni, Ti, ifreq]):
        assert obj.chi_i is None

    # class TestClassClassicalTransportCoefficients(AbstractClassicalTransportCoefficients):
    """
    Dummy class which replaces all of the normalized transport coefficents with
    unity in order to test the normalizations.
    """
