=============
Build GOTCHA
=============

This section describes how to build GOTCHA, and what
:ref:`configure time options <configure-options-label>` are available.

There are two build options:

* build GOTCHA with Spack, and 
* build GOTCHA with cmake

----------

-----------------------------------------
Build GOTCHA with Spack
-----------------------------------------


One may install GOTCHA with Spack_.
If you already have Spack, make sure you have the latest release.
If you use a clone of the Spack develop branch, be sure to pull the latest changes.

.. _build-label:

Install Spack
*************
.. code-block:: Bash

    $ git clone https://github.com/spack/spack
    $ # create a packages.yaml specific to your machine
    $ . spack/share/spack/setup-env.sh

Use `Spack's shell support`_ to add Spack to your ``PATH`` and enable use of the
``spack`` command.

Build and Install GOTCHA
*************************

.. code-block:: Bash

    $ spack install gotcha
    $ spack load gotcha

If the most recent changes on the development branch ('dev') of GOTCHA are
desired, then do ``spack install gotcha@develop``.

.. attention::

    The initial install could take a while as Spack will install build
    dependencies (autoconf, automake, m4, libtool, and pkg-config) as well as
    any dependencies of dependencies (cmake, perl, etc.) if you don't already
    have these dependencies installed through Spack or haven't told Spack where
    they are locally installed on your system (i.e., through a custom
    packages.yaml_).
    Run ``spack spec -I gotcha`` before installing to see what Spack is going
    to do.

----------

-------------------------
Build GOTCHA with CMake
-------------------------

Download the latest GOTCHA release from the Releases_ page or clone the develop
branch ('develop') from the GOTCHA repository
`https://github.com/LLNL/GOTCHA <https://github.com/LLNL/GOTCHA>`_.


.. code-block:: Bash
    
    cmake . -B build -DCMAKE_INSTALL_PREFIX=<where you want to install GOTCHA>
    cmake --build build
    cmake --install build

-----------

.. explicit external hyperlink targets

.. _Releases: https://github.com/LLNL/GOTCHA/releases
.. _Spack: https://github.com/spack/spack
.. _Spack's shell support: https://spack.readthedocs.io/en/latest/getting_started.html#add-spack-to-the-shell
.. _packages.yaml: https://spack.readthedocs.io/en/latest/build_settings.html#external-packages
