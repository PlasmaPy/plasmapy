"""
`FitFunction` classes designed to assist in curve fitting of swept Langmuir
traces.
"""
__all__ = [
    "AbstractFitFunction",
    "ExponentialOffsetFitFunction",
    "LinearFitFunction",
]

import numpy as np

from abc import ABC, abstractmethod
from collections import namedtuple
from scipy.stats import linregress
from scipy.optimize import curve_fit, fsolve
from typing import Any, NamedTuple, Tuple, Union


class AbstractFitFunction(ABC):
    """
    Abstract class for defining fit functions :math:`f(x)` and the tools for
    fitting the function to a set of data.  These were originally designed for
    assisting in fitting curves to swept Langmuir data.
    """

    _parameter_names = ()  # type: Tuple[str, ...]
    _parameters = None  # type: Union[None, Tuple[Any, ...]]
    _parameters_err = None  # type: Union[None, Tuple[Any, ...]]
    _covariance_matrix = None
    _rsq = None
    _curve_fit_results = None

    def __init__(self):
        self._ParamTuple = namedtuple("_ParamTuple", self._parameter_names)

    def __call__(self, x):
        """
        Direct call of the fit function :math:`f(x)``.

        Parameters
        ----------
        x: array_like
            Dependent variables.

        Returns
        -------
        array_like
            Corresponding independent variables of dependent variables
            :math:`x`.
        """
        return self._func(x, *self.parameters)

    def __repr__(self):
        return f"{self.__str__()} {self.__class__}"

    def __str__(self):
        return f"Unspecified f(x)"

    @abstractmethod
    def _func(self, x, *args):
        """
        The fit function.

        Parameters
        ----------
        x: array_like
            Independent variables to be passed to the fit function.
        *args
            The parameters that will be adjusted to make the fit.

        Returns
        -------
        array:
            The calculated dependent variables of the independent variables `x`.
        """
        raise NotImplementedError

    @property
    def curve_fit_results(self):
        """
        The results returned by the curve fitting routine used by
        :attr:`curve_fit`.  This is typically from `scipy.stats.linregress` or
        `scipy.optimize.curve_fit`.
        """
        return self._curve_fit_results

    @property
    def parameters(self) -> Union[None, NamedTuple]:
        """The fitted parameters for the fit function."""
        if self._parameters is None:
            return self._parameters
        else:
            return self._ParamTuple(*self._parameters)

    @property
    def parameters_err(self) -> Union[None, NamedTuple]:
        """The associated errors of the fit `parameters`."""
        if self._parameters_err is None:
            return self._parameters_err
        else:
            return self._ParamTuple(*self._parameters_err)

    @property
    @abstractmethod
    def latex_str(self) -> str:
        """Latex friendly representation of the fit function."""
        raise NotImplementedError

    def root_solve(self, x0, **kwargs):
        """
        Solve for the root of the fit function (i.e. :math:`f(x_r) = 0`).

        Parameters
        ----------
        x0: `~numpy.ndarray`
            The starting estimate for the roots of :math:`f(x_r) = 0`.

        **kwargs
            Any keyword accepted by `scipy.optimize.fsolve`, except for `args`.

        Returns
        -------
        x : `~numpy.ndarray`
            The solution (or the result of the last iteration for an
            unsuccessful call).
        infodict : `dict`
            A dictionary of optional outputs with the keys:

            ``nfev``
                number of function calls
            ``njev``
                number of Jacobian calls
            ``fvec``
                function evaluated at the output
            ``fjac``
                the orthogonal matrix, q, produced by the QR
                factorization of the final approximate Jacobian
                matrix, stored column wise
            ``r``
                upper triangular matrix produced by QR factorization
                of the same matrix
            ``qtf``
                the vector ``(transpose(q) * fvec)``

        ier : `int`
            An integer flag.  Set to 1 if a solution was found, otherwise refer
            to `mesg` for more information.
        mesg : `str`
            If no solution is found, `mesg` details the cause of failure.
        """
        kwargs["args"] = self.parameters
        return fsolve(self._func, x0, **kwargs)

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

        """
        return self._rsq

    def curve_fit(self, xdata, ydata, **kwargs) -> None:
        """
        Use a non-linear least squares method to fit the fit function to
        (`xdata`, `ydata`), using `scipy.optimize.curve_fit`.  This will set
        the attributes :attr:`parameters`, :attr:`parameters_err`, and
        :attr:`rsq`.

        The results of `scipy.optimize.curve_fit` can be obtained via
        :attr:`curve_fit_results`.

        Parameters
        ----------
        xdata: array_like
            The independent variable where data is measured.  Should be 1D of
            length M.

        ydata: array_like
            The dependent data associated with `xdata`.

        **kwargs
            Any keywords accepted by `scipy.optimize.curve_fit`.

        Raises
        ------
        ValueError
            if either `ydata` or `xdata` contain NaNs, or if incompatible options
            are used.

        RuntimeError
            if the least-squares minimization fails.

        ~scipy.optimize.OptimizeWarning
            if covariance of the parameters can not be estimated.

        """
        popt, pcov = curve_fit(self._func, xdata, ydata, **kwargs)
        self._curve_fit_results = (popt, pcov)
        self._parameters = tuple(popt.tolist())
        self._parameters_err = tuple(np.sqrt(np.diag(pcov)).tolist())

        # calc rsq
        # rsq = 1 - (ss_res / ss_tot)
        residuals = ydata - self._func(xdata, *self.parameters)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        self._rsq = 1 - (ss_res / ss_tot)


class ExponentialOffsetFitFunction(AbstractFitFunction):
    """
    A sub-class of `AbstractFitFunction` to represent an exponential with an
    offset.

    .. math::

        f(x) = A \\, \\exp(B \\, x) + C

    where :math:`A`, :math:`B`, and :math:`C` are positive real constants
    and :math:`x` is the independent variable.

    """
    _parameter_names = ("a", "b", "c")

    def __str__(self):
        return f"f(x) = A exp(B x) + C"

    def _func(self, x, a, b, c):
        """
        The fit function, an exponential with an offset.

        .. math::

            f(x) = A \\, \\exp(B \\, x) + C

        where :math:`A`, :math:`B`, and :math:`C` are positive real constants
        and :math:`x` is the independent variable.

        Parameters
        ----------
        x: array_like
            Independent variable.

        a: float
            value for constant :math:`A`

        b: float
            value for constant :math:`B`

        c: float
            value for constant :math:`C`

        Returns
        -------
        y: array_like
            dependent variables corresponding to :math:`x`

        """
        return a * np.exp(b * x) + c

    @property
    def latex_str(self) -> str:
        return fr"A \, \exp(B \, x) + C"

    def root_solve(self, *args, **kwargs):
        """
        The root :math:`f(x_r) = 0` for the fit function.

        .. math::

            x_r &= \\frac{1}{B} \\ln \\left( \\frac{-C}{A} \\right)

            \\delta x_r &= \\sqrt{
                \\left( \\frac{\\delta A}{A B} \\right)^2
                + \\left( x_r \\frac{\\delta B}{B} \\right)^2
                + \\left( \\frac{\\delta C}{B C} \\right)^2
            }

        Parameters
        ----------
        *args
            Not needed.  This is to ensure signature comparability with
            `AbstractFitFunction`.

        *kwargs
            Not needed.  This is to ensure signature comparability with
            `AbstractFitFunction`.

        Returns
        -------
        root: float
            The root value for the given fit :attr:`parameters`.

        err: float
            The error in the calculated root for the given fit
            :attr:`parameters` and :attr:`parameters_err`.
        """
        a, b, c = self.parameters
        root = np.log(-c / a) / b

        a_err, b_err, c_err = self.parameters_err
        a_term = a_err / (a * b)
        b_term = b_err * root / b
        c_term = c_err / (b * c)
        err = np.sqrt(a_term ** 2 + b_term ** 2 + c_term ** 2)

        return root, err


class LinearFitFunction(AbstractFitFunction):
    """
    A sub-class of `AbstractFitFunction` to represent a linear function.

    .. math::

        f(x) = m \\, x + b

    where :math:`m` and :math:`b` are positive real constants representing the
    slope and intercept, respectively, and :math:`x` is the independent
    variable.

    """

    _parameter_names = ("m", "b")

    def __str__(self):
        return f"f(x) = m x + b"

    def _func(self, x, m, b):
        """
        The fit function, a linear function.

        .. math::

            f(x) = m \\, x + b

        where :math:`m` and :math:`b` are positive real constants representing the
        slope and intercept, respectively, and :math:`x` is the independent
        variable.

        Parameters
        ----------
        x: array_like
            Independent variable.
        m: float
            value for slope :math:`m`

        b: float
            value for intercept :math:`b`

        Returns
        -------
        y: array_like
            dependent variables corresponding to :math:`x`

        """
        return m * x + b

    @property
    def latex_str(self) -> str:
        return fr"m \, x + b"

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

        *kwargs
            Not needed.  This is to ensure signature comparability with
            `AbstractFitFunction`.

        Returns
        -------
        root: float
            The root value for the given fit :attr:`parameters`.

        err: float
            The error in the calculated root for the given fit
            :attr:`parameters` and :attr:`parameters_err`.
        """
        m, b = self.parameters
        root = -b / m

        m_err, b_err = self.parameters_err
        err = np.abs(root) * np.sqrt((m_err / m) ** 2 + (b_err / b) ** 2)

        return root, err

    def curve_fit(self, xdata, ydata, **kwargs) -> None:
        """
        Calculate a linear least-squares regression of (`xdata`, `ydata`) using
        `scipy.stats.linregress`.  This will set the attributes
        :attr:`parameters`, :attr:`parameters_err`, and :attr:`rsq`.

        The results of `scipy.stats.linregress` can be obtained via
        :attr:`curve_fit_results`.

        Parameters
        ----------
        xdata: array_like
            The independent variable where data is measured.  Should be 1D of
            length M.

        ydata: array_like
            The dependent data associated with `xdata`.

        **kwargs
            Any keywords accepted by `scipy.stats.linregress.curve_fit`.

        """
        results = linregress(xdata, ydata)
        self._curve_fit_results = results

        m = results[0]
        b = results[1]
        self._parameters = (m, b)

        m_err = results[4]
        b_err = np.sum(xdata ** 2) - ((np.sum(xdata) ** 2) / xdata.size)
        b_err = m_err * np.sqrt(1.0 / b_err)
        self._parameters_err = (m_err, b_err)

        self._rsq = results[2] ** 2
