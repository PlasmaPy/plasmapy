__all__ = ["XSweptLangmuirDiagnostic"]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from typing import Any, Hashable, Mapping
from warnings import warn

from plasmapy.analysis.swept_langmuir.floating_potential import find_floating_potential
from plasmapy.diagnostics import XAbstractDiagnostic
from plasmapy.diagnostics.swept_langmuir import SweptLangmuirProbe
from plasmapy.utils.exceptions import PlasmaPyWarning


class XSweptLangmuirDiagnostic(XAbstractDiagnostic):
    _sig_index = None
    _time_index = None
    _id_index = "id"
    _ds_structured_changed = False

    def __init__(self, dataset: xr.Dataset):
        super().__init__(dataset)
        self._probe_class = SweptLangmuirProbe
        self._check_dataset()

    @property
    def _id_wstart(self) -> str:
        return self._id_index + "_wstart"

    @property
    def _id_wstop(self) -> str:
        return self._id_index + "_wstop"

    @property
    def _id_signum(self) -> str:
        return self._id_index + "_signum"

    @property
    def _id_sweepnum(self) -> str:
        return self._id_index + "_swpnum"

    def _check_dataset(self) -> None:
        """
        Examine xarray `~xarray.Dataset` for expected data.
        """
        # check for XArray Variables
        for dname in ('voltage', 'current'):
            if dname not in self._ds.data_vars.keys():
                raise ValueError(f"Dataset does not have the '{dname}' Variable.")

        # check DataArrays are matched
        if self._ds.voltage.dims != self._ds.current.dims:
            raise ValueError(
                f"The 'voltage' DataArray dims {self._ds.voltage.dims} does not "
                f"match the 'current' DataAray dims {self._ds.current.dims}."
            )
        if self._ds.voltage.shape != self._ds.current.shape:
            raise ValueError(
                f"The 'voltage' DataArray shape {self._ds.voltage.dims} does "
                f"not match the 'current' DataArray shape {self._ds.current.dims}."
            )

        # Examine dimensions and get index names
        if len(self._ds.current.dims) == 1:
            self._set_time_index(self._ds.current.dims[0])
        elif len(self._ds.current.dims) == 2:
            self._set_shot_index(self._ds.current.dims[0])
            self._set_time_index(self._ds.current.dims[1])
        else:
            raise ValueError(
                f"Expected DataArrays with 1 or 2 dimensions and got "
                f"{len(self._ds.current.dims)}."
            )
        if self._ds_structured_changed:
            self._set_trace_id()

    def _set_time_index(self, new_index: str) -> None:
        if self._time_index == new_index:
            pass
        elif self._time_index is None:
            self._time_index = new_index
        else:
            warn(f"The expected time index '{self._time_index}' has changed"
                 f"'{new_index}'.", PlasmaPyWarning)
            self._time_index = new_index

    def _set_shot_index(self, new_index: str) -> None:
        if self._sig_index == new_index:
            pass
        elif self._sig_index is None:
            self._sig_index = new_index
            self._ds_structured_changed = True
        else:
            warn(f"The expected shot index '{self._sig_index}' has changed"
                 f"'{new_index}'.", PlasmaPyWarning)
            self._sig_index = new_index
            self._ds_structured_changed = True

    def _set_trace_id(self) -> None:
        if self._id_index in self._ds.coords:
            coords = (self._id_index, self._id_signum, self._id_sweepnum,
                      self._id_wstart, self._id_wstop)
            warn(f"Overriding the dataset coords '{coords}'.")
            for coord in coords:
                del self._ds[coord]

        if len(self._ds.current.dims) == 1:
            self._ds.coords[self._id_index] = (self._id_index, 0)
            self._ds.coords[self._id_wstart] = (self._id_index,
                                                self._ds[self._time_index].data[0])
            self._ds.coords[self._id_wstop] = (self._id_index,
                                               self._ds[self._time_index].data[-1])
        else:
            # this should only be reached if len(dims) == 2
            _shot_coord = self._ds[self._sig_index]
            _time_coord = self._ds[self._time_index]

            # initialize "id" coordinate
            self._ds.coords[self._id_index] = (
                self._id_index,
                pd.MultiIndex.from_product([_shot_coord, [0]],
                                           names=[self._id_signum, self._id_sweepnum]),
            )

            # initialize "sid" coordinate
            # self._ds.coords[self._id_shot] = (
            #     self._id_index,
            #     self._ds[self._sig_index]
            # )

            # initialize "id_wstart" coordinate
            self._ds.coords[self._id_wstart] = (
                self._id_index,
                np.repeat(self._ds[self._time_index].data[0],
                          self._ds[self._id_index].size)
            )
            self._ds.coords[self._id_wstop] = (
                self._id_index,
                np.repeat(self._ds[self._time_index].data[-1],
                          self._ds.current.shape[0])
            )

        self._ds_structured_changed = False

    def rename_trace_id(self, new_is) -> None:
        raise NotImplementedError

    # @property
    # def floating_potential(self):
    #     try:
    #         return self._ds.floating_potential[0].data[()]
    #     except AttributeError:
    #         warn('Floating potential has not be calculated.')
    #         return np.nan

    # def analyze(self):
    #     self.find_floating_potential()

    def find_floating_potential(self, id=slice(None), **kwargs) -> None:
        self._check_dataset()

        # find floating potential
        vf, vf_err, fit = find_floating_potential(self._ds.voltage.data,
                                                  self._ds.current.data,
                                                  **kwargs)
        # update Dataset
        self._ds['floating_potential'] = xr.DataArray(
            [vf, vf_err],
            dims=('vf',),
            coords={'vf': ['value', 'err']}
        )
        self._ds['floating_potential_fit'] = xr.DataArray(
            [[fit['slope'], fit['slope_err']],
             [fit['intercept'], fit['intercept']],
             [fit['indices'], np.nan]],
            dims=('vf_fit_params', 'vf_fit_values'),
            coords={'vf_fit_params': ['slope', 'intercept', 'indices'],
                    'vf_fit_values': ['value', 'error']}
        )

        return vf, vf_err, fit

    #     def plot(what='all'):
    #         plot_methods = {
    #             'all': None,
    #             'floating_potential': self.plot_floating_potential,
    #         }
    #         try:
    #             plot_methods[what]()
    #         except KeyError:
    #             pass

    def sel_indexers(self,
                     indexers: Mapping[Hashable, Any] = None,
                     **indexers_kwargs: Any) -> Mapping[Hashable, Any]:
        indexers = super().sel_indexers(indexers=indexers, **indexers_kwargs)
        indexers = dict(indexers)

        _shot_index = self._sig_index
        _time_index = self._time_index
        _id_index = self._id_index
        _id_shot = self._id_signum
        _id_trace = self._id_sweepnum
        _id_wstart = self._id_wstart
        _id_wstop = self._id_wstop

        if _id_index in indexers and _shot_index in indexers:
            print(f"{_id_index} and {_shot_index}")

            raise ValueError(
                f"I'm not smart enough to resolve indexing using both {_id_index} and "
                f"{_shot_index}, use just one."
            )
        elif _id_index in indexers:
            # generate sel for shot index
            sn = self._ds[_id_shot].sel({_id_index: indexers[_id_index]}).data
            indexers[_shot_index] = sn

            # generate sel for time index
            if _time_index in indexers:
                # let the id time interval be overruled by _time_index
                pass
            else:
                wstart = self._ds[_id_wstart].sel({_id_index: indexers[_id_index]}).data
                wstop = self._ds[_id_wstop].sel({_id_index: indexers[_id_index]}).data
                indexers[_time_index] = slice(np.min(wstart), np.max(wstop))
        elif _shot_index in indexers and _time_index in indexers:
            # need to filter for both sid's and trid's

            # get shot index values
            sn = indexers[_shot_index]

            # get trace index values
            # pick id's where any time value falls between wstart and wstop
            mask = xr.where((self._ds[_id_wstart] >= np.min(indexers[_time_index]))
                            & (self._ds[_id_wstop] <= np.max(indexers[_time_index])),
                            True, False).data
            warn(f"I'm not smart enough to select trace id's '{_id_trace}' based on "
                 f"time index '{_time_index}' values.  Only filtering for shot index "
                 f"values",
                 PlasmaPyWarning)

        elif _shot_index in indexers:
            # need to filter for sid only
            pass
        elif _time_index in indexers:
            # need to filter for trid only
            pass

        return indexers

    def sweep_count(self, signum: int) -> int:
        if not isinstance(signum, int):
            raise TypeError(f"Expected in for 'signum' and got type "
                            f"{type(signum).__name__}.")

        indexer = {self._id_index: (signum, )}
        try:
            count = self._ds[self._id_signum].sel(**indexer).size
        except KeyError:
            raise KeyError(f"Signal number {signum} does not exist in the Dataset.")

        return count

    def plot_floating_potential_fit(self):
        self._check_dataset()
        fig, ax = plt.subplots()

        # get data subset
        sub = self._ds.floating_potential_fit.sel(vf_fit_params='indices',
                                                  vf_fit_values='value')
        Vsub = self._ds.voltage[sub].data
        Isub = self._ds.current[sub].data

        # generate fit curve
        a = self._ds.floating_potential_fit.sel(vf_fit_params='slope',
                                                vf_fit_values='value').data[()]
        b = self._ds.floating_potential_fit.sel(vf_fit_params='intercept',
                                                vf_fit_values='value').data[()]
        Isub_fit = a * Vsub + b

        #
        vf = self._ds.floating_potential.sel(vf='value').data[()]
        vf_err = self._ds.floating_potential.sel(vf='err').data[()]

        # calc plot limits
        xpad = 0.4 * np.abs(Vsub.max() - Vsub.min())
        ypad = 0.4 * np.abs(Isub.max() - Isub.min())
        xlim = [Vsub.min() - xpad, Vsub.max() + xpad]
        ylim = [Isub.min() - ypad, Isub.max() + ypad]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        ax.set_xlabel('Probe Bias (V)')
        ax.set_ylabel('Current (A)')
        ax.set_title('Floating Potential V$_f$ Fit')

        # add plot elemnts
        ax.plot(ax.get_xlim(), [0.0, 0.0], 'r--', label='I = 0')
        ystop = ax.transData.inverted().transform(ax.transAxes.transform([0.0, 0.85]))[
            1]
        ax.plot([vf, vf],
                [ylim[0], ystop],
                'r',
                label='vf')
        ax.fill_between([vf - vf_err, vf + vf_err],
                        ylim[0],
                        ylim[1],
                        color='r', alpha=0.1)
        ax.plot(Vsub, Isub_fit, label='fit', color='orange')
        ax.scatter(self._ds['voltage'], self._ds['current'], label='data')
        ax.scatter(Vsub, Isub,
                   s=12 ** 2,
                   label='fit data',
                   facecolors='none',
                   edgecolors='orange')

        # labels
        txt = f"V$_f$ = {vf.data[()]:.2f} V"
        txt_loc = [vf, ylim[0]]
        txt_loc = ax.transAxes.inverted().transform(ax.transData.transform(txt_loc))
        txt_loc[1] = 0
        txt_loc += 0.03
        ax.text(txt_loc[0], txt_loc[1], txt, color='r', fontsize='large',
                transform=ax.transAxes)

        vals = [a, b]
        imax = np.abs(np.log10(vals)).argmax()
        power = np.log10(vals[imax])
        power = int(np.sign(power)) * int(np.ceil(np.abs(power)))
        txt = (f"I = ( {a * (10 ** -power):.1f} x 10$^{{{power}}}$ ) * V "
               f"+ ( {b * (10 ** -power):.1f} x 10$^{{{power}}}$ )")
        txt_loc = [0.0 + 0.03, 1.0 - 0.1]
        ax.text(txt_loc[0], txt_loc[1], txt, color='black', fontsize='large',
                transform=ax.transAxes)

        plt.show()

        return fig, ax
