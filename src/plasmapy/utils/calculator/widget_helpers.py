"""
Functions that create widgets and process properties for the plasma
calculator.
"""

__all__: list[str] = []

import abc
import importlib
from inspect import signature

import ipywidgets as widgets

from plasmapy.particles import Particle

BLACK = (0, 0, 0)
"""RGB constant for black."""

DARK_RED = (255, 0, 0)
"""RGB Constant for dark red."""

LIGHT_GREEN = (0, 128, 0)
"""RGB Constant for light green."""

ERROR_STYLE = "2px solid red"
"""Constant for error style."""

EQUAL_SPACING_CONFIG = "10px 10px 10px 10px"
"""Constant for equal spacing config among widgets."""

values_container = {}
"""stores the values of widget with corresponding ``property_name``."""

_process_queue = []
"""
Stores the functions to be processed. This data is gathered from
``properties_metadata.json``.
"""


class _GenericWidget(abc.ABC):
    """
    Generic widget class.

    Parameters
    ----------
    property_name: `str`
        Name of the property the widget is associated with.
        This value is key for the values_container.

    property_alias: `str`
        Alias of the property. Useful to display in validation error messages.

    values_cont: `dict`
        Reference to global dictionary to store the values of the widgets.

    Raises
    ------
    `NotImplementedError`
        If the method `create_widget` is not implemented.
    """

    def __init__(
        self, property_name: str, property_alias: str = "", values_cont=values_container
    ) -> None:
        self.property_name = property_name
        self.property_alias = property_alias or property_name
        self.widget = None
        self.values_cont = values_cont
        self.unit = None
        self.units_dropdown = None

    def set_unit(self, unit) -> None:
        """
        Set unit for the value of widget, defaults to `None`.

        Parameters
        ----------
        unit: `astropy.units.Unit`
            Unit to be set for the value of widget
        """
        self.unit = unit

    def get_widget(self):
        """
        Get current widget object reference.

        Returns
        -------
        `ipywidgets.Widget`
            Current widget object reference
        """
        return self.widget

    def get_dropdown_widget(self):
        """
        Get dropdown widget associated with current widget, defaults to `None`.

        Returns
        -------
        `ipywidgets.Dropdown`
            Dropdown widget associated with current widget
        """
        return self.units_dropdown

    def set_place_holder(self, text: str) -> None:
        """
        Set place holder text of the widget, defaults to empty string.

        Parameters
        ----------
        text: `str`
            Place holder text to be set
        """
        self.widget.placeholder = text

    @abc.abstractmethod
    def create_widget(self):
        """
        Virtual method to create widget.
        """

    def post_creation(self) -> None:
        """
        Default method that is called after widget creation.
        Attaches change listener to the widget.
        """
        self.set_place_holder("")
        self.widget.observe(self.handle_change, names="value")

    def edge_case(self, value) -> None:  # noqa: B027
        """
        Edge case handling for the widget. This is called within handle_change.

        Parameters
        ----------
        value: `any`
            Value of the widget
        """

    def edge_case_condition(self, value) -> bool:  # noqa: ARG002
        """
        Edge case condition for the widget.

        Parameters
        ----------
        value: `any`
            Value of the widget

        Returns
        -------
        `bool`
            `True` if the value is an edge case, `False` otherwise
        """
        return False

    def try_change_value(self, value) -> None:
        """
        Set property_name in values_container to value.

        Parameters
        ----------
        value: `any`
            Value to be set
        """
        self.values_cont[self.property_name] = value

    def display_error(self, value) -> None:  # noqa: ARG002
        """
        Handle invalid input provide realtime validation.

        Parameters
        ----------
        value: `any`
            Value of the widget
        """
        if self.widget:
            self.widget.layout.border = ERROR_STYLE
            self.widget.description = f"Invalid {self.property_alias}"
            self.values_cont[self.property_name] = None

    def convert_to_unit(self, change):
        """
        Convert the value of the widget to the unit specified by the dropdown.

        Parameters
        ----------
        change: `any`
            New value of the widget

        Returns
        -------
        `any`
            Value of the widget in the unit specified by the dropdown
        """
        return change.new * self.unit if self.unit else change.new

    def attach_units_dropdown(self, options) -> None:
        """
        Special method that attaches dropdown widget to the input widget,
        and handles the change event of the dropdown widget.

        Parameters
        ----------
        options: `list`
            List of units to be displayed in the dropdown widget
        """
        self.units_dropdown = widgets.Dropdown(
            options=options, value=options[0], layout=widgets.Layout(width="100px")
        )
        self.units_dropdown.observe(self.handle_dropdown_change, names="value")

    def handle_dropdown_change(self, change) -> None:  # noqa: ARG002
        """
        Handle change event of the dropdown widget.

        Parameters
        ----------
        change: `any`
            New value of the dropdown widget
        """
        self.set_unit(self.units_dropdown.value)
        if self.property_name in self.values_cont:
            self.values_cont[self.property_name] = (
                self.values_cont[self.property_name].value * self.unit
            )

    def handle_change(self, change) -> None:
        """
        Handle change event of the widget, follows same process
        for all widgets.

        Gets the new value with units, checks for invalid input,
        edge case and updates values_container accordingly.

        Parameters
        ----------
        change: `any`
            New value of the widget
        """
        value = self.convert_to_unit(change)
        if self.edge_case_condition(value):
            self.edge_case(value)
        else:
            try:
                self.try_change_value(value)
            except ValueError:
                self.display_error(value)


class _FloatBox(_GenericWidget):
    """
    Derived from _GenericWidget, a FloatBox input widget
    with incremental options.

    Parameters
    ----------
    property_name: `str`
        Name of the property the widget is associated with.

    min: `float`
        Minimum value the widget can take

    max: `float`
        Maximum value the widget can take
    """

    def __init__(
        self,
        property_name: str,
        min: float = -1e50,  # noqa: A002
        max: float = 1e50,  # noqa: A002
    ) -> None:
        super().__init__(property_name)
        self.min = min
        self.max = max

    def create_widget(self, style=None) -> None:
        """
        Implements create_widget. description_width is set to initial
        to make the widget as wide as possible.
        """
        if style is None:
            style = {"description_width": "initial"}
        self.widget = widgets.BoundedFloatText(
            name=self.property_name,
            min=self.min,
            max=self.max,
            value=0,
            step=0.1,
            style=style,
        )
        self.post_creation()


class _CheckBox(_GenericWidget):
    """
    Derived from _GenericWidget, a CheckBox input widget.

    Parameters
    ----------
    property_name: `str`
        Name of the property the widget is associated with.
    """

    def __init__(self, property_name: str) -> None:
        super().__init__(property_name)

    def create_widget(self) -> None:
        """
        Implements create_widget.
        """
        self.widget = widgets.Checkbox(value=False)
        self.post_creation()


class _ParticleBox(_GenericWidget):
    """
    Derived from _GenericWidget, input widget specific for particle
    name.

    Parameters
    ----------
    property_name: `str`
        Name of the property the widget is associated with.

    property_alias: `str`
        Alias of the property the widget is associated with.
        (particle_type in this case)
    """

    def __init__(self, property_name: str, property_alias=None) -> None:
        super().__init__(property_name, property_alias=property_alias)

    def edge_case_condition(self, value):
        """
        Edge case for particle box, checks if value is empty.

        Parameters
        ----------
        value: `str`
            Value of the widget

        Returns
        -------
        `bool`
            `True` if the value is empty, `False` otherwise
        """
        return value is None or value == ""

    def edge_case(self, value) -> None:  # noqa: ARG002
        """
        Edge case to handle empty value of particle box
        resets the container value to `None`, and resets the error status.
        """
        self.values_cont[self.property_name] = None
        self.widget.description = ""
        self.widget.layout.border = ""

    def try_change_value(self, value) -> None:
        """
        Set property_name in values_container to value,
        and resets the error status.

        Parameters
        ----------
        value: `str`
            Value to be set

        Raises
        ------
        `~plasmapy.particles.exceptions.InvalidParticleError`
            Raised when the particle input does not correspond to a valid
            particle or is contradictory.
        """
        particle = Particle(value)
        self.values_cont[self.property_name] = particle
        self.widget.layout.border = ""
        self.widget.description = ""

    def create_widget(self, style=None) -> None:
        """
        Implements create_widget. description_width is set to initial
        to make the widget as wide as possible.
        """
        if style is None:
            style = {"description_width": "initial"}
        self.widget = widgets.Text(style=style)
        self.post_creation()


class _IonBox(_ParticleBox):
    """
    Derived from _ParticleBox, input widget specific for ion.

    Parameters
    ----------
    property_name: `str`
        Name of the property the widget is associated with.

    property_alias: `str`
        Alias of the property the widget is associated with.
    """

    def __init__(self, property_name: str, property_alias=None) -> None:
        super().__init__(property_name, property_alias=property_alias)

    def try_change_value(self, value):
        """
        Set property_name in values_container to value on validating input.

        Parameters
        ----------
        value: `str`
            Value to be set

        Raises
        ------
        `~plasmapy.particles.exceptions.InvalidParticleError`
            Raised when the particle input does not correspond to a valid
            particle or is contradictory.
        `ValueError`
            Raised when the input is not a valid ion
        """
        ion = Particle(value)
        if not ion.is_ion:
            raise ValueError(f"{ion} is not an ion")

        self.values_cont[self.property_name] = ion
        self.widget.layout.border = ""
        self.widget.description = ""


class _FunctionInfo:
    """
    Class to store information about a function. Gets the function's parameters,
    and uses to process input based on function signature.

    Parameters
    ----------
    module_name: `str`
        Name of the module the function is in

    function_name: `str`
        Name of the function

    values_container: `dict`
        Reference to global dictionary of values to be passed to the function
    """

    def __init__(
        self, module_name: str, function_name: str, values_cont=values_container
    ) -> None:
        self.module = module_name
        self.fname = function_name
        self.fattr = getattr(importlib.import_module(module_name), function_name)
        self.values_cont = values_cont
        self.spec_combo = None
        self.sig = list(signature(self.fattr).parameters.keys())
        self.output_widget = widgets.Output()
        self.output_widget.layout.margin = EQUAL_SPACING_CONFIG
        self.output_widget.layout.padding = EQUAL_SPACING_CONFIG

    def add_combo(self, spec_combo) -> None:
        """
        Specify selective combination of parameters to be used in the function,
        This is the case for few functions where having all parameters doesn't yield
        output.

        Parameters
        ----------
        spec_combo: `list`
            List of parameters to be used in the function

        Examples
        --------
        For `plasmapy.formulary.gyroradius` the specific combos are as follows:
        :py:`["B","particle","Vperp"]` and :py:`["B","particle","T"]`
        """
        if not self.spec_combo:
            self.spec_combo = []
        self.spec_combo.append(spec_combo)

    def get_output_widget(self):
        """
        Returns the output widget of the function.

        Returns
        -------
        `~ipywidgets.widgets.Output`
            Output widget of the function
        """
        return self.output_widget

    def produce_arg(self, spec):
        """
        Prepares a dictionary of arguments that is present in both values_container,
        and in spec.

        Parameters
        ----------
        spec: `list`
            List of parameters to be used in the function

        Returns
        -------
        `dict`
            Dictionary of arguments that is available
        """
        return {
            arg: self.values_cont[arg]
            for arg in spec
            if arg in self.values_cont and self.values_cont[arg] is not None
        }

    def error_message(self, spec) -> None:
        """
        Generates an error message for the function when parameters are missing.

        Parameters
        ----------
        spec: `list`
            List of parameters to be used in the function
        """
        # We'll need to switch from print() to using logging library

        print(_colored_text(BLACK, "["), end="")  # noqa: T201

        for arg in spec:
            if arg in self.values_cont and self.values_cont[arg] is not None:
                print(  # noqa: T201
                    _colored_text(LIGHT_GREEN, f"{arg}:present,"),
                    end="",
                )
            else:
                print(  # noqa: T201
                    _colored_text(DARK_RED, f"{arg}:missing,"),
                    end="",
                )

        print(_colored_text(BLACK, "]"))  # noqa: T201

    def process(self) -> None:
        """
        Processes the function based on signatures and spec_combo provided.
        Spec_combo is prioritized over the function signature.
        """
        self.output_widget.clear_output()
        args_dict = {}
        if self.spec_combo:
            for spec in self.spec_combo:
                args_dict = self.produce_arg(spec)

                if len(args_dict) == len(spec):
                    break
        else:
            args_dict = self.produce_arg(self.sig)
        with self.output_widget:
            try:
                self.output_widget.layout.border = "0px"

                # We'll need to switch from print() to using logging library

                print(f" : {self.fattr(**args_dict)}")  # noqa: T201

            except Exception as e:  # noqa: BLE001
                self.output_widget.layout.border = ERROR_STYLE
                print(e)  # noqa: T201
                print(  # noqa: T201
                    " : could not be computed one or more parameter is missing - check below for missing parameters"
                )
                if self.spec_combo:
                    for spec in self.spec_combo:
                        self.error_message(spec)
                else:
                    self.error_message(self.sig)


def _create_label(label: str, color: str = "black"):
    """
    Creates a label widget with the given text and color.

    Parameters
    ----------
    label: `str`
        Text of the label

    color: `str`
        Color of the label, defaults to black
    """
    # This is done so voila switches colors according to theme
    color_param = f"color:{color}" if color != "black" else ""
    return widgets.HTML(f"<h3 style='margin:0px;{color_param}'>{label}<h3>")


def _handle_button_click(event) -> None:
    """
    Handles the click event of the calculate properties button.
    """
    for fn in _process_queue:
        fn.process()


def _handle_clear_click(event) -> None:
    """
    Handles the click event of the clear properties button.
    Clears output of all functions.
    """
    for fn in _process_queue:
        fn.output_widget.clear_output()


def _colored_text(color, text: str) -> str:
    """
    Prepares an inline string with the given color.

    Parameters
    ----------
    color: `list`
        RGB color of the text

    text: `str`
        Text to be colored

    Returns
    -------
    `str`
        Colored text
    """
    return f"\033[38;2;{color[0]};{color[1]};{color[2]}m{text} \033[38;2;255;255;255m"


_calculate_button = widgets.Button(
    description="Calculate Properties", button_style="info"
)
_calculate_button.on_click(_handle_button_click)

_clear_button = widgets.Button(description="Clear Output", button_style="danger")
_clear_button.on_click(_handle_clear_click)


def _create_widget(widget_type, **kwargs):
    """
    Creates a widget of the given type with the given parameters.
    Widget can be with/without units dropdown.

    Parameters
    ----------
    widget_type: `any`
        Type of the widget to be created.

    **kwargs: `dict`
        Parameters specific to the widget.

    Returns
    -------
    |Widget| or [|Widget|, |Widget|]
        widget or [widget, units_dropdown]

    .. |Widget| replace:: `~ipywidgets.widgets.Widget`
    """
    unit = None
    placeholder = None
    opts = None
    if "unit" in kwargs:
        unit = kwargs["unit"]
        kwargs.pop("unit")
    if "placeholder" in kwargs:
        placeholder = kwargs["placeholder"]
        kwargs.pop("placeholder")
    if "opts" in kwargs:
        opts = kwargs["opts"]
        kwargs.pop("opts")
    widget_element = widget_type(**kwargs)
    widget_element.create_widget()
    if unit:
        widget_element.set_unit(unit)
    if placeholder:
        widget_element.set_place_holder(placeholder)

    if opts:
        widget_element.attach_units_dropdown(opts)
        return [widget_element.get_widget(), widget_element.get_dropdown_widget()]

    return widget_element.get_widget()
