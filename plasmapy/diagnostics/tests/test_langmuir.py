"""Tests for Langmuir probe analysis functions."""

import astropy.constants.si as const
import numpy as np
import pytest

from astropy import units as u

from plasmapy.diagnostics import langmuir

np.random.seed(42)
N = 30  # array length of dummy probe characteristic

current_arr = np.random.rand(N) * u.A
bias_arr = np.random.rand(N) * u.V


class Test__fitting_functions:
    "Tests the different available fit functions"

    x = 20
    x0 = 10
    y0 = 5
    c0 = 2
    T0 = 3
    Delta_T = 15

    def test_fit_func_lin(self):
        """Test linear fitting function"""

        value = langmuir._fit_func_lin(self.x, self.x0, self.y0, self.c0)
        expect_value = 25

        errStr = (
            "Linear fitting function did not return the expected value {expect_value} and instead returned {value}"
        )
        assert value == expect_value, errStr

    def test_fit_func_lin_inverse(self):
        """Test linear fitting function with inverse slope"""

        value = langmuir._fit_func_lin_inverse(self.x, self.x0, self.y0, self.T0)
        expect_value = 8.33333

        errStr = (
            f"Linear fitting function with inverse slope did not return the expected value {expect_value} and instead returned {value}"
        )
        assert np.allclose(value, expect_value), errStr

    def test_fit_func_double_lin_inverse(self):
        """Test double linear fitting function with inverse slope and an offset
        for use in fitting a bi-Maxwellian electron current growth region"""

        value = langmuir._fit_func_double_lin_inverse(
            self.x, self.x0, self.y0, self.T0, self.Delta_T
        )
        expect_value = 8

        errStr = (
            f"Linear fitting function with inverse slope and an offset for use in fitting a bi-Maxwellian electron current growth region did not return the expected value {expect_value} and instead returned {value}"
        )
        assert value == expect_value, errStr


class Test__characteristic_errors:
    """Test the Characteristic class constructor in langmuir.py"""

    bias_2darr = np.array((np.random.rand(N), np.random.rand(N))) * u.A
    current_2darr = np.array((np.random.rand(N), np.random.rand(N))) * u.A

    bias_infarr = np.append(np.random.rand(N - 1), np.inf) * u.A * u.V
    current_infarr = np.append(np.random.rand(N - 1), np.inf) * u.A

    bias_longarr = np.random.rand(N + 1) * u.A * u.V
    current_longarr = np.random.rand(N + 1) * u.A

    current_arr2 = np.random.rand(N) * u.A *u.V

    def test_invalid_dimensions(self):
        """Test 2D arrays work with Characteristic class"""

        # `Characteristic.get_unique_bias` runs during initialization
        # which prevents an exception from being raised..
        with pytest.warns(FutureWarning):
            langmuir.Characteristic(self.bias_2darr, self.current_2darr)

    def test_infinite_I(self):
        """Test error upon an infinite current value"""

        with pytest.raises(ValueError):
            langmuir.Characteristic(bias_arr, self.current_infarr)

    def test_infinite_U(self):
        """Test error upon an infinite bias value"""

        with pytest.raises(ValueError):
            langmuir.Characteristic(self.bias_infarr, current_arr)

    def test_unequal_arrays(self):
        """Test errors upon unequal array lengths"""

        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                langmuir.Characteristic(self.bias_longarr, current_arr)

        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                langmuir.Characteristic(bias_arr, self.current_longarr)

    def test_addition(self):
        """Test addition of characteristic objects"""

        with pytest.warns(FutureWarning):
            a = langmuir.Characteristic(bias_arr, current_arr)
        with pytest.warns(FutureWarning):
            b = langmuir.Characteristic(bias_arr, self.current_arr2)

        ab_sum = a + b

        errStr = "Addition of characteristic objects is not behaving as it should."
        assert (a.current + b.current == ab_sum.current).all(), errStr

    def test_subtraction(self):
        """Test addition of characteristic objects"""

        with pytest.warns(FutureWarning):
            a = langmuir.Characteristic(bias_arr, current_arr)
        with pytest.warns(FutureWarning):
            b = langmuir.Characteristic(bias_arr, self.current_arr2)

        ab_sub = a - b

        errStr = "Subtraction of characteristic objects is not behaving as it should."
        assert (a.current - b.current == ab_sub.current).all(), errStr


@pytest.fixture
def characteristic():
    """Create a dummy characteristic with random values"""

    with pytest.warns(FutureWarning):
        return langmuir.Characteristic(bias_arr, current_arr)


@pytest.fixture
def characteristic_simulated():
    """Create a simulated probe characteristic (provisional)"""

    T_e_sim = 1 * u.eV
    n_e_sim = 1e18 * u.m**-3
    probe_area_sim = 1 * u.cm**2
    I_es_sim = (
        n_e_sim * probe_area_sim * const.e * np.sqrt(T_e_sim / (2 * np.pi * const.m_e))
    )

    # Create bias array
    bias_simarr = np.arange(-20, 15, 0.1) * u.V

    # Calculate electron current and limit to electron saturation current
    current_simarr = np.exp(const.e * bias_simarr / T_e_sim) * u.A
    current_simarr[current_simarr > I_es_sim] = I_es_sim * u.A

    # Add simulated linear sheath expansion current
    current_simarr[current_simarr == I_es_sim] += (
        bias_simarr[current_simarr == I_es_sim] * 5e-4 * u.A / u.V
    )

    # Add simulated linear ion collection current
    current_simarr[current_simarr < I_es_sim] += (
        bias_simarr[current_simarr < I_es_sim] * 1e-4 * u.A / u.V
    )
    with pytest.warns(FutureWarning):
        return langmuir.Characteristic(bias_simarr, current_simarr)


def shuffle_characteristic(characteristic):
    """Shuffle a given characteristic"""

    _shuffle = sorted(
        np.arange(len(characteristic.bias)), key=lambda k: np.random.random()
    )
    U_shuffled = characteristic.bias[_shuffle]
    I_shuffled = characteristic.current[_shuffle]
    return langmuir.Characteristic(U_shuffled, I_shuffled)


class DryCharacteristic(langmuir.Characteristic):
    """
    Overrides the constructor in Characteristic class such that `bias` is not filtered for
    unique values.
    """

    def __init__(self, bias, current):
        super().__init__(bias, current)
        self.bias = bias
        self.current = current
        self._check_validity()


class Test__Characteristic_inherited_methods:
    """Test methods on DryCharacteristic class."""

    bias_2darr = np.array((np.random.rand(N), np.random.rand(N))) * u.V
    current_2darr = np.array((np.random.rand(N), np.random.rand(N))) * u.A

    bias_4length_arr = np.array(np.random.rand(N - 1)) * u.V
    current_5length_arr = np.array(np.random.rand(N)) * u.V

    bias_duplicates_arr = np.array((1, 2) * int(N / 2)) * u.V

    def test_invalid_bias_dimensions(self):
        """Test error on non-1D bias array"""
        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                DryCharacteristic(self.bias_2darr, current_arr)

    def test_invalid_current_dimensions(self):
        """Test error on non-1d current array"""
        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                DryCharacteristic(bias_arr, self.current_2darr)

    def test_bias_and_current_length_mismatch(self):
        """Test error on non-1d bias and current arrays"""

        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                DryCharacteristic(self.bias_4length_arr, self.current_5length_arr)

    def test_duplicate_bias_values(self):
        """Test error on bias array containing duplicate values"""

        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                DryCharacteristic(self.bias_duplicates_arr, current_arr)

    @staticmethod
    def test_inplace_unique_bias():
        """Test new Characteristic instance being returned"""

        with pytest.warns(FutureWarning):
            char = DryCharacteristic(bias_arr, current_arr)
        with pytest.warns(FutureWarning):
            new_char = char.get_unique_bias(inplace=False)
        assert char != new_char
        assert isinstance(new_char, langmuir.Characteristic)

    @staticmethod
    def test_getpadded_limit(characteristic):
        """Test padding limit on Characteristic instance"""

        char = characteristic
        limits = char.get_padded_limit(0.1)
        log_limits = char.get_padded_limit(0.1, log=True)
        assert np.allclose(limits.to(u.A).value, np.array((-0.07434804, 1.06484239)))
        assert np.allclose(log_limits.to(u.A).value, np.array((0.014003, 1.42577333)))


@pytest.mark.slow
class Test__swept_probe_analysis:
    """Test the swept_probe_analysis function in langmuir.py"""

    @staticmethod
    def test_nan_area():
        """Test error upon NaN area"""

        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                langmuir.swept_probe_analysis(
                    characteristic, np.nan * u.cm**2, "Ar-40 1+"
                )

    @staticmethod
    def test_unit_conversion_error():
        """Test error upon incorrect probe area unit"""

        with pytest.raises(u.UnitTypeError):
            with pytest.warns(FutureWarning):
                langmuir.swept_probe_analysis(characteristic, 1 * u.cm, "Ar-40 1+")

    @staticmethod
    def test_negative_area():
        """Test error upon negative probe area"""

        with pytest.raises(ValueError):
            with pytest.warns(FutureWarning):
                langmuir.swept_probe_analysis(
                    characteristic, -1 * u.cm**2, "Ar-40 1+"
                )

    @staticmethod
    @pytest.mark.parametrize("bimaxwellian", [True, False])
    def test_ordering_invariance(bimaxwellian, characteristic_simulated):
        """Test invariance to ordering of the bias and current values"""

        with pytest.warns(FutureWarning):
            sim_result = langmuir.swept_probe_analysis(
                characteristic_simulated,
                1 * u.cm**2,
                "Ar-40 1+",
                bimaxwellian=bimaxwellian,
            )

        with pytest.warns(FutureWarning):
            sim_result_shuffled = langmuir.swept_probe_analysis(
                shuffle_characteristic(characteristic_simulated),
                1 * u.cm**2,
                "Ar-40 1+",
                bimaxwellian=bimaxwellian,
            )

        errStr = "Analysis should be invariant to the ordering of the input data."
        for key in sim_result:
            assert (sim_result[key] == sim_result_shuffled[key]).all(), errStr


def test_get_floating_potential_with_return_arg(characteristic):
    """Test floating potential and the return argument"""

    with pytest.warns(FutureWarning):
        potential, arg = langmuir.get_floating_potential(
            characteristic, return_arg=True
        )
    assert np.allclose((potential.to(u.V).value, arg), (0.12203823, 5))


def test_get_ion_density_OML_without_return_fit(characteristic):
    """Test ion density without returning the fit value"""

    with pytest.warns(FutureWarning):
        density = langmuir.get_ion_density_OML(
            characteristic, 5000000 * u.m**2, "p+", return_fit=False
        )
    assert np.isclose(density.value, 385344135.12064785)


def test_get_EEDF():
    """Test the obtained EEDF"""

    with pytest.warns(FutureWarning):
        char = langmuir.Characteristic(bias_arr[:17], current_arr[:17])
    with pytest.warns(FutureWarning):
        energy, probability = langmuir.get_EEDF(char, visualize=False)

    expect_energy = (0.14118696, 0.05293109, 0.00709731)
    expect_probability = (8.64838751, 8.05295503, 3.42348436)

    assert np.allclose(energy.to(u.eV).value, np.array(expect_energy))
    assert np.allclose(probability, np.array(expect_probability))
