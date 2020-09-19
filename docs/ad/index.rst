.. include:: ad_attention.inc

.. _ad:

======================
Analyses & Diagnostics
======================

Analyses and diagnostics go hand-in-hand, but have subtle differences.  Thus,
PlasmaPy gives each their own sub-packages, `plasmapy.analysis` and
`plasmapy.diagnostics` respectively.


Think of the `plasmapy.analysis` as your toolbox.  It has all the tools
(functionality) you need to analyze your data.  Functionality is built around
`numpy` arrays and each function has a well-defined, focused task.  For example,
:func:`numpy.fft.fft` does one specific task, compute the one-dimensional
discrete Fourier Transform.  Similarly,
:func:`plasmapy.analysis.swept_langmuir.find_floating_potential` only finds the
floating potential for a given Langmuir trace.  It does not have smoothing.
It does not do any filtering.  It does not do any signal conditioning.  Its
sole task is to find the floating potential of a single Langmuir trace.

Diagnostics on the other-hand have a much broader scope and leverage the tools
defined in `plasmapy.analysis` to give a more integrated user experience when
analyzing data.  Diagnostics try to enhance the analysis workflow by focusing
on some of following key areas...

#. A more human-friendly way of managing data by building an interface around
   `xarray` arrays and datasets via custom diagnostic accessors.

   - `xarray` provides labelled multi-dimensional arrays and datasets.
   - Diagnostics self-manage the computed analysis data within a `xarray`
     dataset while maintaining the computed data's relation to the original
     data.

#. Quick viewing of analyzed data with default plotting routines.
#. Fully defining the physical parameters of a diagnostic with purposely
   designed Probe classes that are integrated into the analysis workflow.
#. Adding GUI's to the analysis workflow via notebook widgets.

----

.. toctree::
   :maxdepth: 2

   Analyses <analysis/index>
   Diagnostics <diagnostics/index>
