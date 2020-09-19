__all__ = [
    "AbstractProbe",
    "XAbstractDiagnostic",
    "XDiagnosticEnabler",
]

import abc
import importlib
import xarray as xr

from numbers import Number
from typing import Any, Hashable, List, Mapping, Union
from warnings import warn
from xarray.core.utils import either_dict_or_kwargs


class AbstractProbe(abc.ABC):
    """Abstract class for defining probe characteristics/parameters."""


class XAbstractDiagnostic(abc.ABC):
    """
    Abstract class for `xarray` flavored diagnostic classes.
    """
    def __init__(self, dataset: xr.Dataset):
        self._ds = dataset
        self._probe_class = AbstractProbe

    @property
    def is_probe_defined(self) -> bool:
        return 'probe_parameters' in self._ds.attrs.keys()

    @property
    def is_probe_correct(self) -> bool:
        return isinstance(self.probe_parameters, self._probe_class)

    @property
    def probe_parameters(self):
        try:
            return self._ds.attrs['probe_parameters']
        except KeyError:
            raise AttributeError(f"Dataset attribute 'probe_parameters' is not "
                                 f"defined.")

    @probe_parameters.setter
    def probe_parameters(self, value):
        if isinstance(value, self._probe_class):
            self._ds.attrs['probe_parameters'] = value
        else:
            raise ValueError(f"Expected instance of {self._probe_class} "
                             f"and got {value.__class__}.")

    def sel_indexers(self,
                     indexers: Mapping[Hashable, Any] = None,
                     **indexers_kwargs: Any) -> Mapping[Hashable, Any]:
        # this member is currently a static method but subclasses may override the
        # member and the override method is not necessarily static
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs,
                                         "swept_langmuir.sel_indexers")
        return indexers

    def isel_indexers(self,
                      indexers: Mapping[Hashable, Any] = None,
                      **indexers_kwargs: Any) -> Mapping[Hashable, Any]:
        # this member is currently a static method but subclasses may override the
        # member and the override method is not necessarily static
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs,
                                         "swept_langmuir.isel_indexers")
        return indexers

    def sel(self, indexers: Mapping[Hashable, Any] = None,
            method: str = None, tolerance: Number = None,
            drop: bool = False, **indexers_kwargs: Any) -> xr.Dataset:
        # this member is currently a static method but subclasses may override the
        # member and the override method is not necessarily static
        indexers = self.sel_indexers(indexers=indexers, **indexers_kwargs)
        return self._ds.sel(indexers=indexers, method=method,
                            tolerance=tolerance, drop=drop)

    def isel(self,
             indexers: Mapping[Hashable, Any] = None,
             drop: bool = False,
             missing_dims: str = "raise",
             **indexers_kwargs: Any) -> xr.Dataset:
        # this member is currently a static method but subclasses may override the
        # member and the override method is not necessarily static
        indexers = self.isel_indexers(indexers=indexers, **indexers_kwargs)
        return self._ds.isel(indexers=indexers, drop=drop, missing_dims=missing_dims)


# @xr.register_dataset_accessor('plasmapy')
# class PlasmaPyAccessor:
class XDiagnosticEnabler:
    __available_diagnostics = {
        'swept_langmuir': {
            "path": ".swept_langmuir.xdiagnostic",
            "cls": "XSweptLangmuirDiagnostic",
            "renamed": None,
        },
    }

    # def __init__(self, xr_obj: xr.Dataset):
    #     self._ds = xr_obj

    def __repr__(self) -> str:
        summary = [
            super().__repr__(),
            "",
            "Enabled   Available Diagnostic   Bound As",
        ]

        for name, value in self.__available_diagnostics.items():
            bound_as = name if value["renamed"] is None else value["renamed"]
            bound = hasattr(xr.Dataset, bound_as)
            if not bound:
                bound_as = ""

            sum_str = "    ["
            sum_str += "x" if bound else " "
            sum_str += f"]   {name}"

            pad = 23 - len(name)
            sum_str += (" " * pad) + f"{bound_as}"

            summary.append(sum_str)

        return '\n'.join(summary)

    @property
    def available(self) -> List[str]:
        return list(self.__available_diagnostics)

    def enable(self, *args, rename: Union[str, List[str]] = None) -> None:
        if rename is None:
            rename = (None, ) * len(args)
        else:
            if isinstance(rename, str):
                rename = (rename, )

            rename = tuple(rename)
            if len(args) != len(rename):
                raise ValueError(
                    f"The number of renames {len(rename)} does not match the number "
                    f"of diagnostics to enable {len(args)}."
                )
            elif not all([isinstance(rn, str) for rn in rename]):
                raise TypeError(f"Expected all renames to be strings.")

        for arg, rn in zip(args, rename):
            if arg in self.__available_diagnostics:
                possible_names = (arg, self.__available_diagnostics[arg]["renamed"])
                registered = False
                for name in possible_names:
                    if name is not None and hasattr(xr.Dataset, name):
                        # Do NOT register accessor to xarray.Dataset if already there
                        registered = True
                if registered:
                    continue

                diag_name = arg
                if rn is not None:
                    diag_name = rn
                    self.__available_diagnostics[arg]["renamed"] = rn
                mod_name = self.__available_diagnostics[arg]["path"]
                cls_name = self.__available_diagnostics[arg]["cls"]
                cls = getattr(
                    importlib.import_module(mod_name, package="plasmapy.diagnostics"),
                    cls_name,
                )
                xr.register_dataset_accessor(diag_name)(cls)
            else:
                warn(f"Requested diagnostic '{arg}' is not among available "
                     f"diagnostics.")
