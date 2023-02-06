.. _plasmapy-install:

*******************
Installing PlasmaPy
*******************

.. note::

   If you would like to contribute to PlasmaPy, please refer to the
   instructions on :ref:`installing PlasmaPy for development
   <install-plasmapy-dev>`.

.. contents:: Contents
   :local:

Installing Python
=================

PlasmaPy requires Python_ |minpython| or newer. If you do not have Python_
installed already, here are the instructions to `download Python`_ and
install it.

.. tip::

   New versions of Python_ are released annually in October, and it can
   take a few months for the scientific Python ecosystem to catch up. If
   you have trouble installing `plasmapy` on the most recent Python_
   version between October and ∼December, then try installing it on the
   second most recent version.

.. _install-pip:

Installing PlasmaPy with pip
============================

To install the most recent release of `plasmapy` on PyPI_ with pip_ into
an existing Python_ |minpython|\ + environment, open a terminal and run:

.. tab-set::

    .. tab-item:: Linux/MacOS
       :sync: LinuxMacOs

       .. code-block:: bash

          python -m pip install plasmapy

       On some systems, it might be necessary to specify the Python_ version
       number by using ``python3``, ``python3.9``, or
       ``python3.10`` instead of ``python``.

    .. tab-item:: Windows
       :sync: Windows

       .. code-block:: bash

          py -3.10 -m pip install plasmapy

       The version of Python_ may be changed from ``3.10`` to another supported
       Python |minpython|\ + release that has been installed on your computer.

For more detailed information, please refer to this tutorial on
`installing packages`_.

.. _install-conda:

Installing PlasmaPy with Conda
==============================

Conda_ is a package management system and environment manager that is
commonly used in the scientific Python_ ecosystem. Conda_ lets us create
and switch between Python_ environments that are isolated from each
other and the system installation. Conda_ can also be used for packages
written in languages other than Python_.

After `installing Conda`_ or miniconda_, `plasmapy` can be installed
into an activated Conda_ environment by opening a terminal and running:

.. code-block:: bash

   conda install -c conda-forge plasmapy

Here ``-c conda-forge`` indicates that `plasmapy` should be installed
from the conda-forge_ channel.

To install `plasmapy` into another existing Conda_ environment, append
:samp:`-n {env_name}` to the previous command, where :samp:`{env_name}`
is replaced with the name of the environment.

To create a new environment with `plasmapy` installed in it, run:

.. code-block:: bash

    conda create -n env_name -c conda-forge plasmapy

where :samp:`{env_name}` is replaced by the name of the environment.
This step may take several minutes. To activate this environment, run:

.. code-block:: bash

   conda activate env_name

To update `plasmapy` to the most recent version within a currently
activated Conda_ environment, run:

.. code-block:: bash

   conda update plasmapy

.. tip::

   Creating a Conda_ environment can sometimes take a few minutes. If it
   takes longer than that, try updating to the newest version of Conda_
   with ``conda update conda`` or checking out these tips for
   `improving Conda performance`_.

Installing PlasmaPy with Anaconda Navigator
===========================================

.. note::

   This section contains instructions on how to install PlasmaPy with
   `Anaconda Navigator`_ at the time of writing. For the most up-to-date
   information, please go to the official documentation on `installing
   Anaconda Navigator`_ and `managing packages`_.

`Anaconda Navigator`_ is a graphical user interface (GUI) for Conda_
that can be used to install Python packages. It is installed
automatically with newer versions of Conda_. If you are using Miniconda_
or a different Conda_ environment, you can install it with
``conda install anaconda-navigator``. After that it can be opened by
entering ``anaconda-navigator`` in the terminal.

First, go to the :guilabel:`Environments` tab and select
:guilabel:`Channels`. If ``conda-forge`` is not listed, then go to
:guilabel:`Add`, enter ``https://conda.anaconda.org/conda-forge``, and
click on :guilabel:`Update channels` and then :guilabel:`Update index`.

.. Is conda-forge now being added by default?
.. Is clicking on `Update index` actually necessary?

Next, while on the :guilabel:`Environments` tab, select the environment
that you would like to install `plasmapy` in. The default is generally
``base (root)``. Optionally, you may select :guilabel:`Create` to start
a new environment. In the search bar, enter ``plasmapy``. Click on the
checkbox next to ``plasmapy``, and select :guilabel:`Apply` to begin the
installation process. It may take several minutes for Anaconda Navigator
to solve package specifications.

To test the installation, click on the :guilabel:`▶` icon that should be
present next to the activated environment, and select
:guilabel:`Open terminal`. Enter ``python`` in the terminal, and then
``import plasmapy`` to make sure it works.

Installing PlasmaPy from source code
====================================

Obtaining official releases
---------------------------

A ZIP_ file containing the source code for official releases of
`plasmapy` can be obtained `from PyPI`_ or `from Zenodo`_.

Alternatively, official releases since 0.7.0 can be downloaded from the
releases_ page on `PlasmaPy's GitHub repository`_.

Obtaining source code from GitHub
---------------------------------

If you have git_ installed on your computer, you may clone `PlasmaPy's
GitHub repository`_ and access the source code from the most recent
development version by running:

.. code:: bash

   git clone https://github.com/PlasmaPy/PlasmaPy.git

The repository will be cloned inside a new subdirectory called
:file:`PlasmaPy`.

If you do not have git_ installed on your computer, then you may download
the most recent source code from `PlasmaPy's GitHub repository`_ by
going to :guilabel:`Code` and selecting :guilabel:`Download ZIP`.
`Unzipping <https://www.wikihow.com/Unzip-a-File>`__ the file will
create a subdirectory called :file:`PlasmaPy` that contains the source
code.

Building and installing
-----------------------

To install the downloaded version of `plasmapy`, enter the
:file:`PlasmaPy` directory and run:

.. code:: bash

   pip install .

If you expect to occasionally edit the source code, instead run:

.. code:: bash

   pip install -e ".[tests,docs]"

The ``-e`` flag makes the installation editable and ``[tests,docs]``
specifies that all of the additional dependencies used while testing
the package should also be installed.

.. note::

   If you noticed any places where the installation instructions could
   be improved or have become out of date, please `create an issue`_ on
   `PlasmaPy's GitHub repository`_. It would really help!

.. tip::

   The :ref:`contributing-to-plasmapy` guide has instructions on how to
   fork a repository and create branches so that you may make
   contributions via pull requests.

.. _Anaconda Navigator: https://docs.anaconda.com/navigator/install
.. _clone a repository using SSH: https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-ssh-urls
.. _conda-forge: https://conda-forge.org
.. _download Python: https://www.python.org/downloads
.. _from PyPI: https://pypi.org/project/plasmapy
.. _from Zenodo: https://doi.org/10.5281/zenodo.1436011
.. _improving Conda performance: https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/conda-performance.html#improving-conda-performance
.. _installing Anaconda Navigator: https://docs.anaconda.com/navigator/install
.. _installing Conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _installing packages: https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-vcs
.. _managing packages: https://docs.anaconda.com/navigator/tutorials/manage-packages/#installing-a-package
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _releases: https://github.com/PlasmaPy/PlasmaPy/releases
.. _ZIP: https://en.wikipedia.org/wiki/ZIP_(file_format)
