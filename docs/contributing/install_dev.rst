.. _install-plasmapy-dev:

***********************************
Installing PlasmaPy for Development
***********************************

.. contents:: Table of Contents
   :depth: 2
   :local:
   :backlinks: none

Obtaining PlasmaPy source code
==============================

After creating your GitHub account, go to `PlasmaPy's GitHub repository`_
and **fork a copy of PlasmaPy to your account**.

To access Git commands on Windows, try `Git Bash
<https://git-scm.com/downloads>`_.

Next you must **clone your fork to your computer**.  Go to the
directory that will host your PlasmaPy directory, and run one of the
following commands (after changing *your-username* to your username).
If you would like to use HTTPS (which is the default and easier to set
up), then run:

.. code-block:: bash

   git clone https://github.com/your-username/PlasmaPy.git

SSH is a more secure option, but requires you to `set up an SSH key
<https://docs.github.com/en/get-started/getting-started-with-git/about-remote-repositories#cloning-with-ssh-urls>`_
beforehand.  The equivalent SSH command is:

.. code-block:: bash

   git clone git@github.com:your-username/PlasmaPy.git

After cloning, we must tell git where the development version of
PlasmaPy is by running:

.. code-block:: bash

   cd PlasmaPy
   git remote add upstream git://github.com/PlasmaPy/PlasmaPy.git

To check on which remotes exist, run ``git remote -v``.  You should get
something like this:

.. code-block:: bash

   origin	git@github.com:namurphy/PlasmaPy.git (fetch)
   origin	git@github.com:namurphy/PlasmaPy.git (push)
   upstream	git@github.com:PlasmaPy/PlasmaPy.git (fetch)
   upstream	git@github.com:PlasmaPy/PlasmaPy.git (push)

Setting up an environment for development
=========================================

Setup procedures for the two most popular virtual environments, conda
and virtualenv, are listed below.

Conda
-----

To set up a development environment for PlasmaPy, we strongly recommend
the `Anaconda distribution <https://www.anaconda.com/products/distribution>`_.

Activate Anaconda
~~~~~~~~~~~~~~~~~

After installing Anaconda, launch any conda environment. By default,
conda installs a ``root`` environment, which you should be able to
activate via

.. code-block:: bash

   source /home/user/anaconda3/bin/activate root

where ``/home/user/anaconda3/`` can be swapped to wherever your anaconda
installation resides.

On newer versions of Anaconda the recommended activation process has
changed to:

.. code-block:: bash

   . /home/user/anaconda3/etc/profile.d/conda.sh
   conda activate

.. note::

   On Windows, the way to do this is via running ``Anaconda Prompt``
   from the Start Menu. ``Git Bash`` may also work if you have added
   Anaconda to ``PATH``.

Create your environment
~~~~~~~~~~~~~~~~~~~~~~~

Having activated Anaconda, enter PlasmaPy's repository root directory
and create an environment with our suggested packages by executing the
following:

.. code-block:: bash

   conda create -n plasmapy python=3.10

You may now enter the environment via

.. tab-set::

   .. tab-item:: Linux/MacOS
      :sync: LinuxMacOs

      .. code-block:: bash

         source activate plasmapy

   .. tab-item:: Windows
      :sync: Windows

      .. code-block:: bash

         activate plasmapy

In newer Conda versions, the command to run is

.. code-block:: bash

   conda activate plasmapy

Virtualenv
----------

Create a directory for holding the PlasmaPy repository, move into it
and create the virtual environment

.. code-block:: bash

   virtualenv -p python3 venv

You may need to make sure that this directory's path doesn't contain
any spaces, otherwise virtualenv may throw an error.

Your virtual environment should now be created. If you run ``ls venv`` you
will notice that virtualenv has created a number of subdirectories:
``bin/``, ``lib/``, and ``include/``. This is why we're not creating the
virtualenv within the repository itself - so as to not pollute it. To
activate the virtualenv you will run:

.. code-block:: bash

   source venv/bin/activate

You should now see that your shell session is prepended with
(plasmapy), like so:

.. code-block:: bash

   (plasmapy) user@name:~/programming/plasmapy$

This indicates that the virtualenv is running. Congratulations!  When
your're done working on PlasmaPy, you can deactivate the virtualenv by
running

.. code-block:: bash

   source deactivate

Now that you have plasmapy on your local computer and you have a
virtual environment, you will want to "install" this development
version of PlasmaPy along with its dependencies. Start by activating
your virtual environment.

Then, set up the development version of PlasmaPy which you just cloned
by moving into the root directory of the cloned repo and running the
following there:


.. code-block:: bash

   (plasmapy) user@name:~/programming/plasmapy$ pip install --editable ".[tests,docs]"

You should now be all set to run development versions of PlasmaPy
modules via ``import plasmapy`` in your test scripts!

The ``--editable`` flag will create a soft link to your cloned
repository.  Any changes in Python code you make there will be there
when you ``import plasmapy`` from an interactive session.

.. tip::

   If you are not working within a virtual environment, this may end in
   a permission error — this can be avoided via also adding the
   ``--user`` flag. But seriously, use a virtual environment and spare
   yourself the trouble.

Running anaconda with virtualenv
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are running the Anaconda suite and want to use virtualenv to
setup your virtual environment, you will have to let the system know
where the Python interpreter can be found. On Linux this is done with
(for example, assuming having installed Anaconda into ``~/anaconda3``):

.. code-block:: bash

   export LD_LIBRARY_PATH="$HOME/anaconda3/lib/"

Exporting the library path to the dynamic linker will only last for
the duration of the current shell session.

You will have to add the python library directory to LD_LIBRARY_PATH,
as described in a previous step, prior to activating the virtualenv
for every new shell session.


Installing pre-commit
=====================

PlasmaPy uses the |pre-commit|_ framework to perform validations and
automatically apply a consistent style to code contributions. Using
|pre-commit|_ helps us find errors and shortens code reviews. PlasmaPy's
pre-commit suite includes hooks such as:

* ``check-ast`` to verify that the Python code is valid.
* ``trailing-whitespace`` to remove trailing whitespace.
* black_ to format code.
* isort_ to sort imports.
* nbqa_ to format notebooks.

Most of the changes required by |pre-commit|_ can be applied
automatically. To apply these changes in a pull request, add a comment
that says ``pre-commit.ci autofix``. After doing this, be sure to `pull
the changes`_ from GitHub to your computer with ``git pull``.

To enable |pre-commit|_ locally, open a terminal, enter the directory of
the PlasmaPy repository, and run:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

Now suppose we added some trailing whitespace to :file:`some_file.py`
and attempted to commit it. If |pre-commit|_ has been installed, then
the ``trailing-whitespace`` hook will cause |pre-commit|_ to fail while
modifying :file:`some_file.py` to remove the trailing whitespace.

.. code-block:: console

   $ git add some_file.py
   $ git commit -m "Add trailing whitespace"
   Trim Trailing Whitespace.................................................Failed
   - hook id: trailing-whitespace
   - exit code: 1
   - files were modified by this hook

At this point it will be necessary to run these two commands again to
commit the changes. The changes made by |pre-commit|_ will be unstaged and
thus could be seen by running ``git diff``. Sometimes |pre-commit|_ will
not be able to automatically fix the files, such as when there are
syntax errors in Python code. In these cases, the files will need to be
changed manually before running the ``git add`` and ``git commit``
commands again. Alternatively, the |pre-commit|_ hooks can be skipped
using ``git commit --no-verify`` instead.

The |pre-commit|_ configuration is given in |.pre-commit-config.yaml|_.

After adding or updating |pre-commit|_ hooks, run the following command to
apply the changes to all files.

.. code-block:: bash

   pre-commit run --all-files

.. _nbqa: https://nbqa.readthedocs.io
.. _pull the changes: https://docs.github.com/en/get-started/using-git/getting-changes-from-a-remote-repository#pulling-changes-from-a-remote-repository
