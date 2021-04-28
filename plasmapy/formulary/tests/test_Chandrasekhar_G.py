import hypothesis
import numpy as np

from hypothesis import given, settings
from hypothesis import strategies as st

from plasmapy.formulary.mathematics import Chandrasekhar_G


@given(
    x=st.floats(
        max_value=1e90,
        allow_nan=False,
        allow_infinity=False,
    )
)
@settings(max_examples=500)
def test_Chandrasekhar_with_hypothesis(x):
    result = Chandrasekhar_G(x)
    assert x > 0
    assert np.isfinite(x)


def test_Chandrasekhar_regression(num_regression):
    x = np.linspace(-5, 5, 100)
    y = Chandrasekhar_G(x)
    num_regression.check({"x": x, "y": y})


def test_limiting_behavior_small_x():
    x = np.logspace(-7, -4, 100)
    y = Chandrasekhar_G(x)
    limiting_behavior = 2 * x / (3 * np.sqrt(np.pi))
    # useful debug check
    # import matplotlib.pyplot as plt
    # plt.semilogx(x, y - limiting_behavior)
    # plt.show()
    np.testing.assert_allclose(y, limiting_behavior, rtol=0, atol=1e-8)


def test_limiting_behavior_high_x():
    x = np.logspace(6, 9, 100)
    y = Chandrasekhar_G(x)
    limiting_behavior = x ** -2 / 2
    np.testing.assert_allclose(y, limiting_behavior)
