"""
Curve fitting functionality.

This functionality assists in curve fitting of swept Langmuir traces.
"""

__all__ = [
    "AbstractFitFunction",
    "Exponential",
    "ExponentialPlusLinear",
    "ExponentialPlusOffset",
    "Linear",
]

import numbers
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple

import numpy as np
from scipy.optimize import curve_fit, fsolve
from scipy.stats import linregress

from plasmapy.utils.decorators import modify_docstring

#: Named tuple for :meth:`AbstractFitFunction.root_solve`.
_RootResults = namedtuple("RootResults", ("root", "err"))


class AbstractFitFunction(ABC):
    """
    Abstract class for defining fit functions :math:`f(x)` and the tools for
    fitting the function to a set of data.
    """

    _param_names = NotImplemented  # type: tuple[str, ...]

    def __init__(
        self,
        params: tuple[float, ...] | None = None,
        param_errors: tuple[float, ...] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        params : tuple[float, ...], optional
            Tuple of values for the function parameters. Equal in size to
            :attr:`param_names`.

        param_errors : tuple[float, ...], optional
            Tuple of values for the errors associated with the function
            parameters.  Equal in size to :attr:`param_names`.

        """

        self._FitParamTuple = namedtuple("FitParamTuple", self._param_names)

        if params is None:
            self._params = None
        else:
            self.params = params

        if param_errors is None:
            self._param_errors = None
        else:
            self.param_errors = param_errors

        self._curve_fit_results = None
        self._rsq = None

    def __call__(self, x, x_err=None, reterr: bool = False):  # noqa: ANN204
        """
        Direct call of the fit function :math:`f(x)`.

        Parameters
        ----------
        x : |array_like|
            Dependent variables.

        x_err : |array_like|, optional
            Errors associated with the independent variables ``x``.  Must be of
            size one or equal to the size of ``x``.

        reterr : bool, optional
            (Default: `False`) If `True`, return an array of uncertainties
            associated with the calculated independent variables

        Returns
        -------
        y : `numpy.ndarray`
            Corresponding dependent variables :math:`y=f(x)` of the independent
            variables ``x``.

        y_err : `numpy.ndarray`
            Uncertainties associated with the calculated dependent variables
            :math:`\\delta y`
        """
        if reterr:
            y_err, y = self.func_err(x, x_err=x_err, rety=True)

            return y, y_err

        return self.func(x, *self.params)

    def __repr__(self) -> str:
        return f"{self.__str__()} {self.__class__}"

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def func(self, x, *args):
        """
        The fit function.  This signature of the function must first take the
        independent variable followed by the parameters to be fitted as
        separate arguments.

        Parameters
        ----------
        x : |array_like|
            Independent variables to be passed to the fit function.

        *args : tuple[Union[float, int],...]
            The parameters that will be adjusted to make the fit.

        Returns
        -------
        `numpy.ndarray`:
            The calculated dependent variables of the independent variables ``x``.

        Notes
        -----
        * When sub-classing the definition should look something like::

            def func(self, x, a, b, c):
                x = self._check_x(x)
                self._check_params(a, b, c)

                return a * x**2 + b * x + c
        """
        ...

    @abstractmethod
    @modify_docstring(
        prepend="""
        Calculate dependent variable uncertainties :math:`\\delta y` for
        dependent variables :math:`y=f(x)`.
        """,
        append="""
        * When sub-classing the definition should look something like::

            @modify_docstring(append=AbstractFitFunction.func_err.__original_doc__)
            def func_err(self, x, x_err=None, rety=False):
                '''
                A simple docstring giving the equation for error propagation, but
                excluding the parameter descriptions.  The @modify_docstring
                decorator will append the docstring from the parent class.
                '''
                x, x_err = self._check_func_err_params(x, x_err)

                a, b, c = self.params
                a_err, b_err, c_err = self.param_errors

                # calculate error

                if rety:
                    y = self.func(x, a, b, c)
                    return err, y

                return err
        """,
    )
    def func_err(self, x, x_err=None, rety: bool = False):
        """
        Parameters
        ----------
        x : |array_like|
            Independent variables to be passed to the fit function.

        x_err : |array_like|, optional
            Errors associated with the independent variables ``x``.  Must be of
            size one or equal to the size of ``x``.

        rety : bool
            Set to `True` to also return the associated dependent variables
            :math:`y = f(x)`.

        Returns
        -------
        err : `numpy.ndarray`
            The calculated uncertainties :math:`\\delta y` of the dependent
            variables (:math:`y = f(x)`) of the independent variables ``x``.

        y : `numpy.ndarray`, optional
            (if ``rety == True``) The associated dependent variables
            :math:`y = f(x)`.

        Notes
        -----
        * A good reference for formulating propagation of uncertainty expressions is:

            J. R. Taylor.  *An Introduction to Error Analysis: The Study of
            Uncertainties in Physical Measurements.* University Science Books,
            second edition, August 1996 (ISBN: 093570275X)

        """
        ...

    @property
    def curve_fit_results(self):
        """
        The results returned by the curve fitting routine used by
        :attr:`curve_fit`.  This is typically from `scipy.stats.linregress` or
        `scipy.optimize.curve_fit`.
        """
        return self._curve_fit_results

    @property
    def FitParamTuple(self):
        """
        A `~collections.namedtuple` used for attributes :attr:`params` and
        :attr:`param_errors`.  The attribute :attr:`param_names` defines
        the tuple field names.
        """
        return self._FitParamTuple

    @property
    def params(self) -> tuple | None:
        """The fitted parameters for the fit function."""
        if self._params is None:
            return self._params

        return self.FitParamTuple(*self._params)

    @params.setter
    def params(self, val) -> None:
        if isinstance(val, self.FitParamTuple) or (
            isinstance(val, tuple | list)
            and len(val) == len(self.param_names)
            and all(isinstance(vv, numbers.Real) for vv in val)
        ):
            self._params = tuple(val)
        else:
            raise ValueError(
                f"Got {val} for 'val', expecting tuple of ints and "
                f"floats of length {len(self.param_names)}."
            )

    @property
    def param_errors(self) -> tuple | None:
        """The associated errors of the fitted :attr:`params`."""
        if self._param_errors is None:
            return self._param_errors

        return self.FitParamTuple(*self._param_errors)

    @param_errors.setter
    def param_errors(self, val) -> None:
        if isinstance(val, self.FitParamTuple) or (
            isinstance(val, tuple | list)
            and len(val) == len(self.param_names)
            and all(isinstance(vv, numbers.Real) for vv in val)
        ):
            self._param_errors = tuple(val)
        else:
            raise ValueError(
                f"Got {val} for 'val', expecting tuple of ints and "
                f"floats of length {len(self.param_names)}."
            )

    @property
    def param_names(self) -> tuple[str, ...]:
        """Names of the fitted parameters."""
        return self._param_names

    @property
    @abstractmethod
    def latex_str(self) -> str:
        """LaTeX friendly representation of the fit function."""
        ...

    def _check_func_err_params(self, x, x_err):
        """Check the ``x`` and ``x_err`` parameters for :meth:`func_err`."""
        x = self._check_x(x)
        if x_err is not None:
            x_err = self._check_x(x_err)

            if x_err.shape == ():
                pass
            elif x_err.shape != x.shape:
                raise ValueError(
                    f"x_err shape {x_err.shape} must be equal the shape of "
                    f"x {x.shape}."
                )
        return x, x_err

    @staticmethod
    def _check_params(*args) -> None:
        """
        Check fitting parameters so that they are an expected type for the
        class functionality.
        """
        for arg in args:
            if not isinstance(arg, numbers.Real):
                raise TypeError(
                    f"Expected int or float for parameter argument, got "
                    f"{type(arg)}."
                )

    @staticmethod
    def _check_x(x):
        """
        Check the independent variable ``x`` so that it is an expected
        type for the class functionality.
        """
        if isinstance(x, numbers.Real):
            x = np.array(x)
        else:
            if not isinstance(x, np.ndarray):
                x = np.array(x)

            if not (
                np.issubdtype(x.dtype, np.integer)
                or np.issubdtype(x.dtype, np.floating)
            ):
                raise TypeError(
                    "Argument x needs to be an array_like object of integers "
                    "or floats."
                )

            x = x.squeeze()
            if x.shape == ():
                # force x to be a scalar
                x = x[()]

        return x

    def root_solve(self, x0):
        """
        Solve for the root of the fit function (i.e. :math:`f(x_r) = 0`).  This
        method used `scipy.optimize.fsolve` to find the function roots.

        Parameters
        ----------
        x0 : `~numpy.ndarray`
            The starting estimate for the roots of :math:`f(x_r) = 0`.

        Returns
        -------
        x : `~numpy.ndarray`
            The solution (or the result of the last iteration for an
            unsuccessful call).

        x_err : `~numpy.ndarray`
            The uncertainty associated with the root calculation.  **Currently
            this returns an array of** `numpy.nan` **values equal in shape to**
            ``x`` **, since there is no determined way to calculate the
            uncertainties.**

        Notes
        -----
        If the full output of `scipy.optimize.fsolve` is desired then one can do:

        >>> func = Linear()
        >>> func.params = (1.0, 5.0)
        >>> func.param_errors = (0.0, 0.0)
        >>> roots = fsolve(func, -4.0, full_output=True)
        >>> roots
        (array([-5.]), {'nfev': 6, 'fjac': array([[-1.]]), 'r': array([-1.]), 'qtf': array([2.1813662e-12]), 'fvec': np.float64(0.0)}, 1, 'The solution converged.')
        """
        results = fsolve(self.func, x0, args=self.params)

        root = np.squeeze(results[0])
        err = np.tile(np.nan, root.shape)
        if root.shape == ():
            # force x to be a scalar
            root = root[()]
            err = np.nan

        return _RootResults(root, err)

    @property
    def rsq(self):
        """
        Coefficient of determination (r-squared) value of the fit.

        .. math::

            r^2 &= 1 - \\frac{SS_{res}}{SS_{tot}}

            SS_{res} &= \\sum\\limits_{i} (y_i - f(x_i))^2

            SS_{tot} &= \\sum\\limits_{i} (y_i - \\bar{y})^2

        where :math:`(x_i, y_i)` are the sample data pairs, :math:`f(x_i)` is
        the fitted dependent variable corresponding to :math:`x_i`, and
        :math:`\\bar{y}` is the average of the :math:`y_i` values.

        The :math:`r^2` value is an indicator of how close the points
        :math:`(x_i, y_i)` lie to the model :math:`f(x)`.  :math:`r^2` values
        range between 0 and 1.  Values close to 0 indicate that the points
        are uncorrelated and have little tendency to lie close to the model,
        whereas, values close to 1 indicate a high correlation to the model.

        """
        return self._rsq

    def curve_fit(self, xdata, ydata, **kwargs) -> None:
        """
        Use a non-linear least squares method to fit the fit function to
        (``xdata``, ``ydata``), using `scipy.optimize.curve_fit`.  This will set
        the attributes :attr:`params`, :attr:`param_errors`, and
        :attr:`rsq`.

        The results of `scipy.optimize.curve_fit` can be obtained via
        :attr:`curve_fit_results`.

        Parameters
        ----------
        xdata : |array_like|
            The independent variable where data is measured.  Should be 1D of
            length M.

        ydata : |array_like|
            The dependent data associated with ``xdata``.

        **kwargs
            Any keywords accepted by `scipy.optimize.curve_fit`.

        Raises
        ------
        ValueError
            If either ``ydata`` or ``xdata`` contain `numpy.nan`'s, or if
            incompatible options are used.

        RuntimeError
            If the least-squares minimization fails.

        ~scipy.optimize.OptimizeWarning
            If covariance of the parameters can not be estimated.

        """
        popt, pcov = curve_fit(self.func, xdata, ydata, **kwargs)
        self._curve_fit_results = (popt, pcov)
        self.params = tuple(popt.tolist())
        self.param_errors = tuple(np.sqrt(np.diag(pcov)).tolist())

        # calc rsq
        # rsq = 1 - (ss_res / ss_tot)
        residuals = ydata - self.func(xdata, *self.params)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        self._rsq = 1 - (ss_res / ss_tot)


class Linear(AbstractFitFunction):
    """
    A sub-class of `AbstractFitFunction` to represent a linear function.

    .. math::

        y &= f(x) = m \\, x + b

        (\\delta y)^2 &= (x \\, \\delta m)^2 + (m \\, \\delta x)^2 + (\\delta b)^2

    where :math:`m` and :math:`b` are real constants to be fitted and :math:`x` is
    the independent variable.  :math:`\\delta m`, :math:`\\delta b`, and
    :math:`\\delta x` are the respective uncertainties for :math:`m`, :math:`b`,
    and :math:`x`.
    """

    _param_names = ("m", "b")

    def __str__(self) -> str:
        return "f(x) = m x + b"

    @property
    def latex_str(self) -> str:
        """LaTeX friendly representation of the fit function."""
        return r"m x + b"

    def func(self, x, m, b):
        """
        The fit function, a linear function.

        .. math::

            f(x) = m \\, x + b

        where :math:`m` and :math:`b` are real constants representing the
        slope and intercept, respectively, and :math:`x` is the independent
        variable.

        Parameters
        ----------
        x : |array_like|
            Independent variable.

        m : float
            Value for slope :math:`m`

        b : float
            Value for intercept :math:`b`

        Returns
        -------
        y : |array_like|
            Dependent variables corresponding to :math:`x`

        """
        x = self._check_x(x)
        self._check_params(m, b)

        return m * x + b

    @modify_docstring(append=AbstractFitFunction.func_err.__original_doc__)
    def func_err(self, x, x_err=None, rety: bool = False):
        """
        Calculate dependent variable uncertainties :math:`\\delta y` for
        dependent variables :math:`y=f(x)`.

        .. math::

            (\\delta y)^2 = (x \\, \\delta m)^2 + (m \\, \\delta x)^2 + (\\delta b)^2

        """
        x, x_err = self._check_func_err_params(x, x_err)

        m, b = self.params
        m_err, b_err = self.param_errors

        m_term = (m_err * x) ** 2
        b_term = b_err**2
        err = m_term + b_term

        if x_err is not None:
            x_term = (m * x_err) ** 2
            err += x_term
        err = np.sqrt(err)

        if rety:
            y = self.func(x, m, b)
            return err, y

        return err

    @property
    def rsq(self):
        """
        Coefficient of determination (r-squared) value of the fit.  Calculated
        by `scipy.stats.linregress` from the fit.
        """
        return self._rsq

    def root_solve(self, *args, **kwargs):
        """
        The root :math:`f(x_r) = 0` for the fit function.

        .. math::

            x_r &= \\frac{-b}{m}

            \\delta x_r &= |x_r| \\sqrt{
                \\left( \\frac{\\delta m}{m} \\right)^2
                + \\left( \\frac{\\delta b}{b} \\right)^2
            }

        Parameters
        ----------
        *args
            Not needed.  This is to ensure signature comparability with
            `AbstractFitFunction`.

        **kwargs
            Not needed.  This is to ensure signature comparability with
            `AbstractFitFunction`.

        Returns
        -------
        root : float
            The root value for the given fit :attr:`params`.

        err : float
            The uncertainty in the calculated root for the given fit
            :attr:`params` and :attr:`param_errors`.
        """
        m, b = self.params

        if m == 0.0:
            warnings.warn(
                "Slope of Linear fit function is zero so no finite root exists. ",
                RuntimeWarning,
            )
            return _RootResults(np.nan, np.nan)

        root = -b / m

        m_err, b_err = self.param_errors
        m_term = (root * m_err / m) ** 2
        b_term = (b_err / m) ** 2
        err = np.sqrt(m_term + b_term)

        return _RootResults(root, err)

    def curve_fit(self, xdata, ydata, **kwargs) -> None:
        """
        Calculate a linear least-squares regression of (``xdata``, ``ydata``)
        using `scipy.stats.linregress`.  This will set the attributes
        :attr:`params`, :attr:`param_errors`, and :attr:`rsq`.

        The results of `scipy.stats.linregress` can be obtained via
        :attr:`curve_fit_results`.

        Parameters
        ----------
        xdata : |array_like|
            The independent variable where data is measured.  Should be 1D of
            length M.

        ydata : |array_like|
            The dependent data associated with ``xdata``.

        **kwargs
            Any keywords accepted by `scipy.stats.linregress`.

        """
        results = linregress(xdata, ydata, **kwargs)
        self._curve_fit_results = results

        m = results[0]
        b = results[1]
        self.params = (m, b)

        m_err = results[4]
        b_err = np.sum(xdata**2) - ((np.sum(xdata) ** 2) / xdata.size)
        b_err = m_err * np.sqrt(1.0 / b_err)
        self.param_errors = (m_err, b_err)

        self._rsq = results[2] ** 2


class Exponential(AbstractFitFunction):
    """
    A sub-class of `AbstractFitFunction` to represent an exponential with an
    offset.

    .. math::

        y &= f(x) = a \\, e^{\\alpha \\, x}

        \\left( \\frac{\\delta y}{|y|} \\right)^2 &=
            \\left( \\frac{\\delta a}{a} \\right)^2
            + (x \\, \\delta \\alpha)^2
            + (\\alpha \\, \\delta x)^2

    where :math:`a` and :math:`\\alpha` are the real constants to be fitted and
    :math:`x` is the independent variable.  :math:`\\delta a`,
    :math:`\\delta \\alpha`, and :math:`\\delta x` are the respective
    uncertainties for :math:`a`, :math:`\\alpha`, and :math:`x`.
    """

    _param_names = ("a", "alpha")

    def __str__(self) -> str:
        return "f(x) = a exp(alpha x)"

    @property
    def latex_str(self) -> str:
        """LaTeX friendly representation of the fit function."""
        return r"a \, \exp(\alpha x)"

    def func(self, x: float, a: float, alpha: float):
        """
        The fit function, a exponential function.

        .. math::

            f(x) = a \\, e^{\\alpha \\, x}

        where :math:`a` and :math:`\\alpha` are real constants and :math:`x`
        is the independent variable.

        Parameters
        ----------
        x : |array_like|
            Independent variable.

        a : float
            Value for the exponential "normalization" constant, :math:`a`

        alpha : float
            Value for the growth constant, :math:`\\alpha`

        Returns
        -------
        y : |array_like|
            dependent variables corresponding to ``x``
        """
        x = self._check_x(x)
        self._check_params(a, alpha)

        return a * np.exp(alpha * x)

    @modify_docstring(append=AbstractFitFunction.func_err.__original_doc__)
    def func_err(self, x, x_err=None, rety: bool = False):
        """
        Calculate dependent variable uncertainties :math:`\\delta y` for
        dependent variables :math:`y=f(x)`.

        .. math::

            \\left( \\frac{\\delta y}{|y|} \\right)^2 =
                \\left( \\frac{\\delta a}{a} \\right)^2
                + (x \\, \\delta \\alpha)^2
                + (\\alpha \\, \\delta x)^2

        """
        x, x_err = self._check_func_err_params(x, x_err)

        a, alpha = self.params
        a_err, alpha_err = self.param_errors
        y = self.func(x, a, alpha)

        a_term = (a_err / a) ** 2
        alpha_term = (x * alpha_err) ** 2

        err = a_term + alpha_term

        if x_err is not None:
            x_term = (alpha * x_err) ** 2
            err += x_term

        err = np.abs(y) * np.sqrt(err)

        return (err, y) if rety else err

    def root_solve(self, *args, **kwargs):
        """
        The root :math:`f(x_r) = 0` for the fit function. **An exponential has no
        real roots**.

        Parameters
        ----------
        *args
            Not needed.  This is to ensure signature compatibility with
            `AbstractFitFunction`.

        **kwargs
            Not needed.  This is to ensure signature compatibility with
            `AbstractFitFunction`.

        Returns
        -------
        root : float
            The root value for the given fit :attr:`params`.

        err : float
            The uncertainty in the calculated root for the given fit
            :attr:`params` and :attr:`param_errors`.
        """

        return _RootResults(np.nan, np.nan)


class ExponentialPlusLinear(AbstractFitFunction):
    """
    A sub-class of `AbstractFitFunction` to represent an exponential with an
    linear offset.

    .. math::

        y =& f(x) = a \\, e^{\\alpha \\, x} + m \\, x + b\\\\
        (\\delta y)^2 =&
            \\left( a e^{\\alpha x}\\right)^2 \\left[
                \\left( \\frac{\\delta a}{a} \\right)^2
                + (x \\, \\delta \\alpha)^2
                + (\\alpha \\, \\delta x)^2
            \\right]\\\\
            & + \\left(2 \\, a \\, \\alpha \\, m \\, e^{\\alpha x}\\right)
                (\\delta x)^2\\\\
            & + \\left[(x \\, \\delta m)^2 + (\\delta b)^2 +(m \\, \\delta x)^2
            \\right]

    where :math:`a`, :math:`\\alpha`, :math:`m`, and :math:`b` are the
    real constants to be fitted and :math:`x` is the independent
    variable. :math:`\\delta a`, :math:`\\delta \\alpha`,
    :math:`\\delta m`, :math:`\\delta b`, and :math:`\\delta x` are the
    respective uncertainties for :math:`a`, :math:`\\alpha`, :math:`m`,
    and :math:`b`, and :math:`x`.
    """

    _param_names = ("a", "alpha", "m", "b")

    def __init__(
        self,
        params: tuple[float, ...] | None = None,
        param_errors: tuple[float, ...] | None = None,
    ) -> None:
        self._exponential = Exponential()
        self._linear = Linear()
        super().__init__(params=params, param_errors=param_errors)

    def __str__(self) -> str:
        exp_str = self._exponential.__str__().replace("f(x) = ", "")
        lin_str = self._linear.__str__().replace("f(x) = ", "")
        return f"f(x) = {exp_str} + {lin_str}"

    @property
    def latex_str(self) -> str:
        """LaTeX friendly representation of the fit function."""
        exp_str = self._exponential.latex_str
        lin_str = self._linear.latex_str
        return rf"{exp_str} + {lin_str}"

    @AbstractFitFunction.params.setter
    def params(self, val) -> None:  # noqa: D102
        AbstractFitFunction.params.fset(self, val)
        self._exponential.params = (self.params.a, self.params.alpha)
        self._linear.params = (self.params.m, self.params.b)

    @AbstractFitFunction.param_errors.setter
    def param_errors(self, val) -> None:  # noqa: D102
        AbstractFitFunction.param_errors.fset(self, val)
        self._exponential.param_errors = (
            self.param_errors.a,
            self.param_errors.alpha,
        )
        self._linear.param_errors = (self.param_errors.m, self.param_errors.b)

    def func(self, x: float, a: float, alpha: float, m: float, b: float):
        """
        The fit function, an exponential with a linear offset.

        .. math::

            f(x) = a \\, e^{\\alpha \\, x} + m \\, x + b\\\\

        where :math:`a`, :math:`\\alpha`, :math:`m`, and :math:`b` are the real
        constants and :math:`x` is the independent variable.

        Parameters
        ----------
        x : |array_like|
            Independent variable.

        a : float
            Value for constant :math:`a`

        alpha : float
            Value for constant :math:`\\alpha`

        m : float
            Value for slope :math:`m`

        b : float
            Value for intercept :math:`b`

        Returns
        -------
        y : |array_like|
            dependent variables corresponding to ``x``

        """
        exp_term = self._exponential.func(x, a, alpha)
        lin_term = self._linear.func(x, m, b)
        return exp_term + lin_term

    @modify_docstring(append=AbstractFitFunction.func_err.__original_doc__)
    def func_err(self, x, x_err=None, rety: bool = False):
        """
        Calculate dependent variable uncertainties :math:`\\delta y` for
        dependent variables :math:`y=f(x)`.

        .. math::

            (\\delta y)^2 =&
                \\left( a e^{\\alpha x}\\right)^2 \\left[
                    \\left( \\frac{\\delta a}{a} \\right)^2
                    + (x \\, \\delta \\alpha)^2
                    + (\\alpha \\, \\delta x)^2
                \\right]\\\\
                & + \\left(2 \\, a \\, \\alpha \\, m \\, e^{\\alpha x}\\right)
                    (\\delta x)^2\\\\
                & + \\left[(
                        x \\, \\delta m)^2 + (\\delta b)^2 +(m \\, \\delta x)^2
                    \\right]

        """
        x, x_err = self._check_func_err_params(x, x_err)

        a, alpha, m, b = self.params

        exp_y, exp_err = self._exponential(x, x_err=x_err, reterr=True)
        lin_y, lin_err = self._linear(x, x_err=x_err, reterr=True)
        err = exp_err**2 + lin_err**2

        if x_err is not None:
            blend_err = 2 * a * alpha * m * np.exp(alpha * x) * (x_err**2)
            err += blend_err
        err = np.sqrt(err)

        return (err, exp_y + lin_y) if rety else err


class ExponentialPlusOffset(AbstractFitFunction):
    """
    A sub-class of `AbstractFitFunction` to represent an exponential with a
    constant offset.

    .. math::

        y =& f(x) = a \\, e^{\\alpha \\, x} + m \\, x + b\\\\
        (\\delta y)^2 =&
            \\left( a e^{\\alpha x}\\right)^2 \\left[
                \\left( \\frac{\\delta a}{a} \\right)^2
                + (x \\, \\delta \\alpha)^2
                + (\\alpha \\, \\delta x)^2
            \\right]
            + (\\delta b)^2


    where :math:`a`, :math:`\\alpha`, and :math:`b` are the real constants to
    be fitted and :math:`x` is the independent variable.  :math:`\\delta a`,
    :math:`\\delta \\alpha`, :math:`\\delta b`, and :math:`\\delta x` are the
    respective uncertainties for :math:`a`, :math:`\\alpha`, and :math:`b`, and
    :math:`x`.

    """

    _param_names = ("a", "alpha", "b")

    def __init__(
        self,
        params: tuple[float, ...] | None = None,
        param_errors: tuple[float, ...] | None = None,
    ) -> None:
        self._explin = ExponentialPlusLinear()
        super().__init__(params=params, param_errors=param_errors)

    def __str__(self) -> str:
        return "f(x) = a exp(alpha x) + b"

    @property
    def latex_str(self) -> str:
        """LaTeX friendly representation of the fit function."""
        return r"a \, \exp(\alpha x) + b"

    @AbstractFitFunction.params.setter
    def params(self, val) -> None:  # noqa: D102
        AbstractFitFunction.params.fset(self, val)
        self._explin.params = (
            self.params.a,
            self.params.alpha,
            0.0,
            self.params.b,
        )

    @AbstractFitFunction.param_errors.setter
    def param_errors(self, val) -> None:  # noqa: D102
        AbstractFitFunction.param_errors.fset(self, val)
        self._explin.param_errors = (
            self.param_errors.a,
            self.param_errors.alpha,
            0.0,
            self.param_errors.b,
        )

    def func(self, x: float, a: float, alpha: float, b: float):
        """
        The fit function, an exponential with a constant offset.

        .. math::

            f(x) = a \\, e^{\\alpha \\, x} + b\\\\

        where :math:`a`, :math:`\\alpha`, and :math:`b` are the real constants
        and :math:`x` is the independent variable.

        Parameters
        ----------
        x : |array_like|
            Independent variable.

        a : float
            Value for constant :math:`a`

        alpha : float
            Value for constant :math:`\\alpha`

        b : float
            Value for DC offset :math:`b`

        Returns
        -------
        y : |array_like|
            Dependent variables corresponding to ``x``

        """
        return self._explin.func(x, a, alpha, 0.0, b)

    @modify_docstring(append=AbstractFitFunction.func_err.__original_doc__)
    def func_err(self, x, x_err=None, rety: bool = False):
        """
        Calculate dependent variable uncertainties :math:`\\delta y` for
        dependent variables :math:`y=f(x)`.

        .. math::

            (\\delta y)^2 =
                \\left( a e^{\\alpha x}\\right)^2 \\left[
                    \\left( \\frac{\\delta a}{a} \\right)^2
                    + (x \\, \\delta \\alpha)^2
                    + (\\alpha \\, \\delta x)^2
                \\right]
                + (\\delta b)^2

        """
        return self._explin.func_err(x, x_err=x_err, rety=rety)

    def root_solve(self, *args, **kwargs):
        """
        The root :math:`f(x_r) = 0` for the fit function.

        .. math::

            x_r &= \\frac{1}{\\alpha} \\ln \\left( \\frac{-b}{a} \\right)

            \\delta x_r &= \\sqrt{
                \\left( \\frac{1}{\\alpha} \\frac{\\delta a}{a} \\right)^2
                + \\left( x_r \\frac{\\delta \\alpha}{\\alpha} \\right)^2
                + \\left( \\frac{1}{\\alpha} \\frac{\\delta b}{b} \\right)^2
            }

        Parameters
        ----------
        *args
            Not needed.  This is to ensure signature compatibility with
            `AbstractFitFunction`.

        **kwargs
            Not needed.  This is to ensure signature compatibility with
            `AbstractFitFunction`.

        Returns
        -------
        root : float
            The root value for the given fit :attr:`params`.

        err : float
            The uncertainty in the calculated root for the given fit
            :attr:`params` and :attr:`param_errors`.
        """
        a, alpha, b = self.params
        a_err, b_err, c_err = self.param_errors

        root = np.log(-b / a) / alpha

        a_term = a_err / (a * alpha)
        b_term = b_err * root / alpha
        c_term = c_err / (alpha * b)
        err = np.sqrt(a_term**2 + b_term**2 + c_term**2)

        return _RootResults(root, err)
