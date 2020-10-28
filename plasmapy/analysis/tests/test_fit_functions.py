"""
Tests for the fitting function classes defined in `plasmapy.analysis.fit_functions`.
"""
import numpy as np
import pytest

from abc import ABC, abstractmethod
from contextlib import ExitStack as does_not_raise
# ExitStack can be replaced with nullcontext when we require >= python 3.7

import plasmapy.analysis.fit_functions as ffuncs


class TestAbstractFitFunction:
    """
    Tests for fit function class `plasmapy.analysis.fit_functions.AbstractFitFunction`.

    Notes
    -----
    Since `AbstractFitFunction` can not be instantiated, its complete
    functionality can not be directly tested.  To resolve this, `BaseFFTests`,
    which is the base test class for the fit function classes, will test all
    the functionality within `AbstractFitFunction`.
    """

    ff_class = ffuncs.AbstractFitFunction

    def test_is_abs(self):
        """Test `AbstractFitFunction` is an abstract base class."""
        assert issubclass(self.ff_class, ABC)

    @pytest.mark.parametrize(
        "name, isproperty",
        [
            ("__call__", False),
            ("curve_fit", False),
            ("curve_fit_results", True),
            ("func", False),
            ("func_err", False),
            ("latex_str", True),
            ("param_errors", True),
            ("param_names", True),
            ("params", True),
            ("rsq", True),
            ("root_solve", False),
        ],
    )
    def test_methods(self, name, isproperty):
        """Test for required methods and properties."""
        assert hasattr(self.ff_class, name)

        if isproperty:
            assert isinstance(getattr(self.ff_class, name), property)

    @pytest.mark.parametrize(
        "name", ["__str__", "func", "func_err", "latex_str"],
    )
    def test_abstractmethods(self, name):
        """Test for required abstract methods."""
        assert name in self.ff_class.__abstractmethods__


class BaseFFTests(ABC):
    abc = ffuncs.AbstractFitFunction
    _test_params = NotImplemented  # type: tuple
    _test_param_errors = NotImplemented  # type: tuple
    _test_param_names = NotImplemented  # type: tuple
    _test_latex_str = NotImplemented  # type: str
    _test__str__ = NotImplemented  # type: str

    @property
    @abstractmethod
    def ff_class(self):
        """Fit function class to be tested."""
        ...

    @staticmethod
    @abstractmethod
    def func(x, *args):
        """
        Formula/Function that the fit function class is suppose to be modeling.
        This is used to test the fit function `func` method.
        """
        ...

    @abstractmethod
    def func_err(self, x, params, param_errors, x_err=None):
        """
        Function representing the propagation of error associated with fit function
        model. This is used to test the fit function `func_err` method.
        """
        ...

    def test_inheritance(self):
        """Test inheritance from `AbstractFitFunction`."""
        assert issubclass(self.ff_class, self.abc)

    def test_iscallable(self):
        """Test instantiated fit function is callable."""
        assert callable(self.ff_class())

    def test_repr(self):
        """Test __repr__."""
        foo = self.ff_class()
        assert foo.__repr__() == f"{foo.__str__()} {foo.__class__}"

    @pytest.mark.parametrize(
        "name, isproperty",
        [
            ("__call__", False),
            ("_param_names", False),
            ("curve_fit", False),
            ("curve_fit_results", True),
            ("func", False),
            ("func_err", False),
            ("latex_str", True),
            ("param_errors", True),
            ("param_names", True),
            ("params", True),
            ("rsq", True),
            ("root_solve", False),
        ],
    )
    def test_methods(self, name, isproperty):
        """Test attribute/method/property existence."""
        assert hasattr(self.ff_class, name)

        if isproperty:
            assert isinstance(getattr(self.ff_class, name), property)

        if name == "_param_names":
            if self.ff_class._param_names == NotImplemented:
                pytest.fail(
                    f"{self.ff_class} class attribute '_param_names' needs to "
                    f" be defined as a tuple of strings representing the names of "
                    f"the fit parameters."
                )

    @pytest.mark.parametrize(
        "name, value_ref_name",
        [
            ("param_names", "_test_param_names"),
            ("latex_str", "_test_latex_str"),
            ("__str__", "_test__str__"),
        ],
    )
    def test_abstractmethod_values(self, name, value_ref_name):
        """Test value of all abstract methods, except `func` and `func_err`."""
        ff_obj = self.ff_class()

        value = getattr(ff_obj, name)
        if callable(value):
            value = value()

        exp_value = getattr(self, value_ref_name)
        if exp_value == NotImplemented:
            pytest.fail(
                f"The expected value for abstract method {name} is not "
                f"implemented/defined in the test class attribute {value_ref_name}."
            )

        assert value == exp_value

    def test_instantiation(self):
        """Test behavior of fit function class instantiation."""
        # default
        foo = self.ff_class()

        assert isinstance(foo.param_names, tuple)
        assert len(foo.param_names) != 0
        assert all(isinstance(val, str) for val in foo.param_names)

        assert hasattr(foo, "FitParamTuple")
        assert issubclass(foo.FitParamTuple, tuple)
        for name in foo.param_names:
            assert hasattr(foo.FitParamTuple, name)

        assert foo.curve_fit_results is None
        assert foo.params is None
        assert foo.param_errors is None
        assert foo.rsq is None

        assert isinstance(foo.latex_str, str)

        # assign at instantiation
        params = [1] * len(foo.param_names)
        foo = self.ff_class(params=params, param_errors=params)
        assert foo.params == foo.FitParamTuple(*params)
        assert foo.param_errors == foo.FitParamTuple(*params)

        with pytest.raises(ValueError):
            self.ff_class(params=5)
        with pytest.raises(ValueError):
            self.ff_class(param_errors=5)

        params = [2] * len(foo.param_names)
        params[0] = "let me in"
        with pytest.raises(ValueError):
            self.ff_class(params=params)
        with pytest.raises(ValueError):
            self.ff_class(param_errors=params)

        params = [2] * len(foo.param_names)
        params += [5]
        with pytest.raises(ValueError):
            self.ff_class(params=params)
        with pytest.raises(ValueError):
            self.ff_class(param_errors=params)

    @pytest.mark.parametrize(
        "params, extra, with_condition",
        [
            ([2], None, does_not_raise()),
            (5, None, pytest.raises(ValueError)),
            (["wrong"], None, pytest.raises(ValueError)),
            ([3], 10, pytest.raises(ValueError))
        ],
    )
    def test_params_setting(self, params, extra, with_condition):
        """Tests for property setting of attribute `params`."""
        ff_obj = self.ff_class()

        if isinstance(params, list) and len(params) == 1:
            params = params * len(ff_obj.param_names)
        if extra is not None:
            params.append(extra)

        with with_condition:
            ff_obj.params = params
            assert ff_obj.params == ff_obj.FitParamTuple(*params)

    @pytest.mark.parametrize(
        "param_errors, extra, with_condition",
        [
            ([2], None, does_not_raise()),
            (5, None, pytest.raises(ValueError)),
            (["wrong"], None, pytest.raises(ValueError)),
            ([3], 10, pytest.raises(ValueError))
        ],
    )
    def test_param_errors_setting(self, param_errors, extra, with_condition):
        """Tests for property setting of attribute `param_errors`."""
        ff_obj = self.ff_class()

        if isinstance(param_errors, list) and len(param_errors) == 1:
            param_errors = param_errors * len(ff_obj.param_names)
        if extra is not None:
            param_errors.append(extra)

        with with_condition:
            ff_obj.param_errors = param_errors
            assert ff_obj.param_errors == ff_obj.FitParamTuple(*param_errors)

    @pytest.mark.parametrize(
        "x, replace_a_param, with_condition",
        [
            (0, None, does_not_raise()),
            (1.0, None, does_not_raise()),
            (np.linspace(10, 30, num=20), None, does_not_raise()),
            ([4, 5, 6], None, does_not_raise()),
            ("hello", None, pytest.raises(TypeError)),
            (5, "hello", pytest.raises(TypeError)),
        ],
    )
    def test_func(self, x, replace_a_param, with_condition):
        """Test the `func` method."""
        ff_obj = self.ff_class()

        params = self._test_params
        if replace_a_param is not None:
            params = list(params)
            params[0] = replace_a_param

        with with_condition:
            y = ff_obj.func(x, *params)

            if isinstance(x, list):
                x = np.array(x)
            y_expected = self.func(x, *params)

            assert np.allclose(y, y_expected)

    def test_func_err(self):
        """Test the `func_err` method."""
        foo = self.ff_class(
            params=self._test_params, param_errors=self._test_param_errors
        )

        for x in (0, 1.0, np.linspace(10, 30, num=20)):
            assert np.allclose(
                foo.func_err(x),
                self.func_err(x, self._test_params, self._test_param_errors),
            )

        x = [4, 5, 6]
        results = foo.func_err(x, x_err=0.1, rety=True)
        assert np.allclose(
            results[0],
            self.func_err(
                np.array(x), self._test_params, self._test_param_errors, x_err=0.1
            ),
        )
        assert np.allclose(results[1], self.func(np.array(x), *self._test_params))

        with pytest.raises(TypeError):
            foo.func_err("hello")

        with pytest.raises(TypeError):
            foo.func_err(5, x_err="goodbye")

        with pytest.raises(ValueError):
            foo.func_err(5, x_err=[0.1, 0.1])

    def test_call(self):
        """Test __call__ behavior."""
        foo = self.ff_class()
        foo.params = self._test_params
        foo.param_errors = self._test_param_errors

        for x in (0, 1.0, np.linspace(10, 30, num=20)):
            assert np.allclose(foo(x), self.func(x, *self._test_params))

            # also return error
            y, y_err = foo(x, reterr=True)
            assert np.allclose(y, self.func(x, *self._test_params))
            assert np.allclose(
                y_err, self.func_err(x, self._test_params, self._test_param_errors),
            )

        x = [4, 5, 6]
        x_err = 0.05
        assert np.allclose(foo(x), self.func(np.array(x), *self._test_params))
        y, y_err = foo(x, x_err=x_err, reterr=True)
        assert np.allclose(y, self.func(np.array(x), *self._test_params))
        assert np.allclose(
            y_err,
            self.func_err(
                np.array(x), self._test_params, self._test_param_errors, x_err=x_err
            ),
        )

        with pytest.raises(TypeError):
            foo("hello")

        with pytest.raises(ValueError):
            foo(5, x_err=[1, 2], reterr=True)

    @abstractmethod
    def test_root_solve(self):
        ...

    def test_curve_fit(self):
        """Test the `curve_fit` method."""
        foo = self.ff_class()

        xdata = np.linspace(-10, 10)
        ydata = self.func(xdata, *self._test_params)

        assert foo.params is None
        assert foo.param_errors is None
        assert foo.rsq is None
        assert foo.curve_fit_results is None

        foo.curve_fit(xdata, ydata)

        assert foo.curve_fit_results is not None
        assert np.isclose(foo.rsq, 1.0)
        assert np.allclose(
            foo.param_errors, tuple([0] * len(foo.param_names)), atol=1.5e-8,
        )
        assert np.allclose(foo.params, self._test_params)


class TestFFExponential(BaseFFTests):
    """
    Tests for fit function class `plasmapy.analysis.fit_functions.Exponential`.
    """

    ff_class = ffuncs.Exponential
    _test_params = (5.0, 1.0)
    _test_param_errors = (0.1, 0.1)
    _test_param_names = ("a", "alpha")
    _test_latex_str = fr"a \, \exp(\alpha x)"
    _test__str__ = f"f(x) = a exp(alpha x)"

    @staticmethod
    def func(x, a, alpha):
        return a * np.exp(alpha * x)

    def func_err(self, x, params, param_errors, x_err=None):
        a, alpha = params
        a_err, alpha_err = param_errors
        y = self.func(x, *params)

        a_term = (a_err / a) ** 2
        alpha_term = (x * alpha_err) ** 2

        err = a_term + alpha_term

        if x_err is not None:
            x_term = (alpha * x_err) ** 2
            err += x_term

        err = np.abs(y) * np.sqrt(err)

        return err

    def test_root_solve(self):
        foo = self.ff_class(params=(1, 1), param_errors=(0, 0))
        root, err = foo.root_solve()
        assert np.isnan(root)
        assert np.isnan(err)


class TestFFExponentialPlusLinear(BaseFFTests):
    """
    Tests for fit function class
    `plasmapy.analysis.fit_functions.ExponentialPlusLinear`.
    """

    ff_class = ffuncs.ExponentialPlusLinear
    _test_params = (2.0, 1.0, 5.0, -10.0)
    _test_param_errors = (0.1, 0.1, 0.1, 0.1)
    _test_param_names = ("a", "alpha", "m", "b")
    _test_latex_str = fr"a \, \exp(\alpha x) + m x + b"
    _test__str__ = f"f(x) = a exp(alpha x) + m x + b"

    @staticmethod
    def func(x, a, alpha, m, b):
        return a * np.exp(alpha * x) + m * x + b

    def func_err(self, x, params, param_errors, x_err=None):
        a, alpha, m, b = params
        a_err, alpha_err, m_err, b_err = param_errors

        exp_y = a * np.exp(alpha * x)

        a_term = (exp_y * a_err / a) ** 2
        alpha_term = (exp_y * x * alpha_err) ** 2
        m_term = (m_err * x) ** 2
        b_term = b_err ** 2

        err = a_term + alpha_term + m_term + b_term

        if x_err is not None:
            x_term = (exp_y * alpha * x_err) ** 2
            x_term += (m * x_err) ** 2
            x_term += 2 * a * alpha * m * np.exp(alpha * x) * (x_err ** 2)

            err += x_term

        err = np.sqrt(err)

        return err

    def test_root_solve(self):
        foo = self.ff_class(params=(5.0, 0.5, 1.0, 5.0), param_errors=(0, 0, 0, 0))
        root, err = foo.root_solve(-5)
        assert np.isclose(root, -5.345338)
        assert np.isnan(err)


class TestFFExponentialPlusOffset(BaseFFTests):
    """
    Tests for fit function class
    `plasmapy.analysis.fit_functions.ExponentialPlusOffset`.
    """

    ff_class = ffuncs.ExponentialPlusOffset
    _test_params = (2.0, 1.0, -10.0)
    _test_param_errors = (0.1, 0.1, 0.1)
    _test_param_names = ("a", "alpha", "b")
    _test_latex_str = fr"a \, \exp(\alpha x) + b"
    _test__str__ = f"f(x) = a exp(alpha x) + b"

    @staticmethod
    def func(x, a, alpha, b):
        return a * np.exp(alpha * x) + b

    def func_err(self, x, params, param_errors, x_err=None):
        a, alpha, b = params
        a_err, alpha_err, b_err = param_errors

        exp_y = a * np.exp(alpha * x)

        a_term = (exp_y * a_err / a) ** 2
        alpha_term = (exp_y * x * alpha_err) ** 2
        b_term = b_err ** 2

        err = a_term + alpha_term + b_term

        if x_err is not None:
            x_term = (exp_y * alpha * x_err) ** 2
            err += x_term

        err = np.sqrt(err)

        return err

    def test_root_solve(self):
        foo = self.ff_class(params=(3.0, 0.5, -5.0), param_errors=(0, 0, 0))
        root, err = foo.root_solve()
        assert root == np.log(5.0 / 3.0) / 0.5
        assert err == 0

        foo.params = (3.0, 0.5, 5.0)
        root, err = foo.root_solve()
        assert np.isnan(root)
        assert np.isnan(err)


class TestFFLinear(BaseFFTests):
    """
    Tests for fit function class `plasmapy.analysis.fit_functions.Linear`.
    """

    ff_class = ffuncs.Linear
    _test_params = (5.0, 4.0)
    _test_param_errors = (0.1, 0.1)
    _test_param_names = ("m", "b")
    _test_latex_str = fr"m x + b"
    _test__str__ = f"f(x) = m x + b"

    @staticmethod
    def func(x, m, b):
        return m * x + b

    def func_err(self, x, params, param_errors, x_err=None):
        m, b = params
        m_err, b_err = param_errors

        m_term = (m_err * x) ** 2
        b_term = b_err ** 2
        err = m_term + b_term

        if x_err is not None:
            x_term = (m * x_err) ** 2
            err += x_term
        err = np.sqrt(err)

        return err

    def test_root_solve(self):
        foo = self.ff_class(params=(1, 1), param_errors=(0, 0))
        assert foo.root_solve() == (-1, 0)

        foo.params = (5.0, 1.3)
        foo.param_errors = (0.1, 0.1)
        root, err = foo.root_solve()
        assert root == -1.3 / 5.0
        assert err == np.abs(root) * np.sqrt((0.1 / 5.0) ** 2 + (0.1 / 1.3) ** 2)
