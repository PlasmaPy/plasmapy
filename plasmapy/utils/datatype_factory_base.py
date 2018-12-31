# SunPy is released under a BSD-style open source licence:

# Copyright (c) 2013-2019 The SunPy developers
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:

# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Project     : https://github.com/sunpy/sunpy
# File        : sunpy/util/datatype_factory_base.py
# Commit hash : f6330eea602ea796b5b004dee283b8877b24da23


import inspect


class BasicRegistrationFactory:
    """
    Generalized registerable factory type.

    Widgets (classes) can be registered with an instance of this class.
    Arguments to the factory's `__call__` method are then passed to a function
    specified by the registered factory, which validates the input and returns
    a instance of the class that best matches the inputs.

    Attributes
    ----------

    registry : dict
        Dictionary mapping classes (key) to function (value) which validates
        input.

    default_widget_type : type
        Class of the default widget.  Defaults to None.

    validation_functions : list of strings
        List of function names that are valid validation functions.

    Parameters
    ----------

    default_widget_type : type, optional

    additional_validation_functions : list of strings, optional
        List of strings corresponding to additional validation function names.

    Notes
    -----

    * A valid validation function must be a classmethod of the registered widget
      and it must return True or False.

    """

    def __init__(self, default_widget_type=None,
                 additional_validation_functions=[], registry=None):

        if registry is None:
            self.registry = dict()
        else:
            self.registry = registry

        self.default_widget_type = default_widget_type

        self.validation_functions = (['_factory_validation_function'] +
                                     additional_validation_functions)

    def __call__(self, *args, **kwargs):
        """ Method for running the factory.

        Arguments args and kwargs are passed through to the validation
        function and to the constructor for the final type.
        """

        # Any preprocessing and massaging of inputs can happen here

        return self._check_registered_widget(*args, **kwargs)

    def _check_registered_widget(self, *args, **kwargs):

        candidate_widget_types = list()

        for key in self.registry:

            # Call the registered validation function for each registered class
            if self.registry[key](*args, **kwargs):
                candidate_widget_types.append(key)

        n_matches = len(candidate_widget_types)

        if n_matches == 0:
            if self.default_widget_type is None:
                raise NoMatchError("No types match specified arguments and no default is set.")
            else:
                candidate_widget_types = [self.default_widget_type]
        elif n_matches > 1:
            print(candidate_widget_types)
            raise MultipleMatchError("Too many candidate types identified ({0})."
                                     "Specify enough keywords to guarantee unique type "
                                     "identification.".format(n_matches))

        # Only one is found
        WidgetType = candidate_widget_types[0]

        return WidgetType(*args, **kwargs)

    def register(self, WidgetType, validation_function=None, is_default=False):
        """ Register a widget with the factory.

        If `validation_function` is not specified, tests `WidgetType` for
        existence of any function in in the list `self.validation_functions`,
        which is a list of strings which must be callable class attribute

        Parameters
        ----------

        WidgetType : type
            Widget to register.

        validation_function : function, optional
            Function to validate against.  Defaults to None, which indicates
            that a classmethod in validation_functions is used.

        is_default : bool, optional
            Sets WidgetType to be the default widget.

        """
        if is_default:
            self.default_widget_type = WidgetType

        elif validation_function is not None:
            if not callable(validation_function):
                raise AttributeError("Keyword argument 'validation_function' must be callable.")

            self.registry[WidgetType] = validation_function

        else:
            found = False
            for vfunc_str in self.validation_functions:
                if hasattr(WidgetType, vfunc_str):
                    vfunc = getattr(WidgetType, vfunc_str)

                    # check if classmethod: stackoverflow #19227724
                    _classmethod = inspect.ismethod(vfunc) and vfunc.__self__ is WidgetType

                    if _classmethod:
                        self.registry[WidgetType] = vfunc
                        found = True
                        break
                    else:
                        raise ValidationFunctionError("{0}.{1} must be a classmethod."
                                                      .format(WidgetType.__name__, vfunc_str))

            if not found:
                raise ValidationFunctionError("No proper validation function for class {0} "
                                              "found.".format(WidgetType.__name__))

    def unregister(self, WidgetType):
        """ Remove a widget from the factory's registry."""
        self.registry.pop(WidgetType)


class NoMatchError(Exception):
    """Exception for when no candidate class is found."""


class MultipleMatchError(Exception):
    """Exception for when too many candidate classes are found."""


class ValidationFunctionError(AttributeError):
    """Exception for when no candidate class is found."""
