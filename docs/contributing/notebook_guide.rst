
.. _notebook guide:

**************
Notebook Guide
**************

PlasmaPy's online documentation includes :ref:`example notebooks
<examples>`. Jupyter notebooks may be added to PlasmaPy's
`example gallery <example-gallery>`_ by putting them in
:file:`docs/notebooks` or one of its subdirectories. The directory or
notebook must be included in :file:`docs/examples.rst` to show up in the
gallery.

These notebooks are and built using the |nbsphinx|_ extension to Sphinx_
and thus show up in the online documentation. The `Jupyter Project
Documentation`_ has more information on how to use Jupyter notebooks.

.. _Jupyter Project Documentation: https://docs.jupyter.org/en/latest/

.. _example Markdown cells: https://nbsphinx.readthedocs.io/en/latest/markdown-cells.html

*Need to discuss* `example Markdown cells`_...

.. code-block:: markdown

   [example notebooks]: https://docs.plasmapy.org/en/latest/examples.html
   [astropy.units]: https://docs.astropy.org/en/stable/units/index.html

   PlasmaPy's documentation includes [example notebooks]. We can refer
   to code objects by using backticks, such as for [astropy.units].


Markdown cells
==============

*Add text here*

Links
-----

Note that links must be redefined in each Markdown_ cell.

External links
~~~~~~~~~~~~~~

*Add text here*

.. code-block:: markdown

   [astropy.units]: https://docs.astropy.org/en/stable/units/index.html

   This is an example link to [astropy.units].

.. [cosmic latte]: https://en.wikipedia.org/wiki/Cosmic_latte
   The average color of the universe is [cosmic latte].

Internal links
~~~~~~~~~~~~~~

Linking to an object in PlasmaPy's API requires including a relative
link to the corresponding file that gets created in :file:`docs/api`
when the documentation is built.

If a notebook is included in :file:`docs/notebooks/getting_started`,
then a link to |Particle| can be created as

*Finish this section*

.. code-block:: markdown

   [Particle]: ../../api/plasmapy.particles.particle_class.Particle.rst

   This is a link to the [Particle] class.

The

Glossary terms
~~~~~~~~~~~~~~

.. code-block:: markdown

   [particle-like]: ../../glossary.rst#term-particle-like

   An object is [particle-like] if it can be transformed into a
   `Particle`.

Table of contents
~~~~~~~~~~~~~~~~~

Longer notebooks should include a table of contents to improve ease of
navigation.

.. code-block:: markdown

   ## Contents

   1. [Introduction](#Introduction)
   2. [Second section](#Second section)

   ## Introduction

   ## Second section

Math
----

Math can be included in a Markdown_ with some common LaTeX_ commands.

.. code-block:: markdown

   In-line math is enclosed in dollar signs, like $\gamma = \frac{5}{3}$.

   It is possible to include displayed math in a Markdown cell too.

   \begin{equation}
   \mathbf{E} + \mathbf{V} \times \mathbf{B} = \eta \mathbf{J}
   \end{equation}

Expected exceptions
-------------------

If a cell is expected to raise an exception, label it with
``raises-exception``.

*Add a link on how to label a cell in a notebook and/or instructions on
how to use* ``raises-exception``.

reStructuredText cells
======================


Pre-executing notebooks
=======================

The most computationally intensive notebooks in the `example gallery`_
have been pre-executed to save time during continuous integration.

These notebooks should be re-executed prior to releases
