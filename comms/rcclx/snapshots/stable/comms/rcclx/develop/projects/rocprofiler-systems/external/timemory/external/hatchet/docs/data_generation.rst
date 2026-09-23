.. Copyright 2021-2023 University of Maryland and other Hatchet Project
   Developers. See the top-level LICENSE file for details.

   SPDX-License-Identifier: MIT

*****************************
Generating Profiling Datasets
*****************************

HPCToolkit
==========
HPCToolkit can be installed using `Spack <https://spack.io>`_ or manually.
Instructions to build HPCToolkit manually can be found at
http://hpctoolkit.org/software-instructions.html.

You can see a basic example of how to use HPCToolkit and generate performance
data below.

.. code-block:: console

   $ mpirun -np <num_ranks> hpcrun <hpcrun_args> ./program.exe <program_args>

This command generates a "measurements" directory. Hatchet cannot read
this natively and requires another step to generate a "database" directory
using ``hpcprof-mpi`` as described below.

.. code-block:: console

   $ hpcstruct ./program.exe
   $ mpirun -np 1 hpcprof-mpi --metric-db=yes -S ./program.exe.struct -I <path_to_src> <measurements-directory>

The first command generates a struct file for the executable ``program.exe``.
This is provided as one of the arguments in the second command along with
pointers to the source code and the generated measurements directory.  You must
add the ``--metric-db=yes`` option to ``hpcprof-mpi`` to generate the database
directory in the format recognizable by hatchet.

You can specify the events you want to record as arguments to ``hpcrun``. For
example: ``-e CPUTIME@5000`` or ``-e PAPI_TOT_CYC@5000000 -e PAPI_TOT_INS -e
PAPI_L2_TCM -e PAPI_BR_INS``.

If you want to record data only for the main thread (0) and not for other
helper threads, you can set this environment variable: ``export
HPCRUN_IGNORE_THREAD=1,2,..``.

More information information about HPCToolkit can be found at HPCToolkit's
`documentation page <http://hpctoolkit.org/documentation.html>`_.


Caliper
=======
Caliper can be installed using `Spack <https://spack.io>`_ or manually from its
`GitHub repository <https://github.com/LLNL/Caliper>`_. Instructions to build
Caliper manually can be found in its `documentation
<https://software.llnl.gov/Caliper/build.html>`_.

To record performance profiles using Caliper, you need to include ``cali.h``
and call the ``cali_init()`` function in your source code.  You also need to
link the Caliper library in your executable or load it using ``LD_PRELOAD``.
Information about basic Caliper usage can be found in the `Caliper
documentation <https://software.llnl.gov/Caliper/CaliperBasics.html>`_.

To generate profiling data, you can use Caliper's `built-in profiling
configurations <https://software.llnl.gov/Caliper/BuiltinConfigurations.htm>`_ customized for Hatchet: ``hatchet-region-profile`` or
``hatchet-sample-profile``. The former generates a profile based on user
annotations in the code while the latter generates a call path profile (similar
to HPCToolkit's output).  If you want to use one of the built-in
configurations, you should set the ``CALI_CONFIG`` environment variable (e.g.
``CALI_CONFIG=hatchet-sample-profile``).

Alternatively, you can use a custom Caliper .config file (default:
caliper.config).  If you create your own .config file, you can set the
CALI_CONFIG_FILE environment variable to point to it.  Two sample
caliper.config files are presented below.  Other example configuration files
can be found in the Caliper `GitHub repository
<https://github.com/LLNL/Caliper/tree/master/examples/configs>`_.

.. code-block:: console

   CALI_SERVICES_ENABLE=aggregate,event,mpi,mpireport,timestamp
   CALI_EVENT_TRIGGER=annotation,function,loop,mpi.function
   CALI_TIMER_SNAPSHOT_DURATION=true
   CALI_AGGREGATE_KEY=prop:nested,mpi.rank
   CALI_MPI_WHITELIST=MPI_Send,MPI_Recv,MPI_Isend,MPI_Irecv,MPI_Wait,MPI_Waitall,MPI_Bcast,MPI_Reduce,MPI_Allreduce,MPI_Barrier
   CALI_MPIREPORT_CONFIG="SELECT annotation,function,loop,mpi.function,mpi.rank,sum(sum#time.duration),inclusive_sum(sum#time.duration) group by mpi.rank,prop:nested format json-split"
   CALI_MPIREPORT_FILENAME="lulesh-annotation-profile.json"

.. code-block:: console

   CALI_SERVICES_ENABLE=aggregate,callpath,mpi,mpireport,sampler,symbollookup,timestamp
   CALI_SYMBOLLOOKUP_LOOKUP_MODULE=true
   CALI_TIMER_SNAPSHOT_DURATION=true
   CALI_CALIPER_FLUSH_ON_EXIT=false
   CALI_SAMPLER_FREQUENCY=200
   CALI_CALLPATH_SKIP_FRAMES=4
   CALI_AGGREGATE_KEY=callpath.address,cali.sampler.pc,mpi.rank
   CALI_MPIREPORT_CONFIG="select source.function#callpath.address,sourceloc#cali.sampler.pc,mpi.rank,sum(sum#time.duration),sum(count),module#cali.sampler.pc group by source.function#callpath.address,sourceloc#cali.sampler.pc,mpi.rank,module#cali.sampler.pc format json-split"
   CALI_MPIREPORT_FILENAME="cpi-sample-callpathprofile.json"

You can read more about Caliper services in the `Caliper documentation
<https://software.llnl.gov/Caliper/services.html>`_. Hatchet can read two Caliper outputs: the native .cali files and the split-JSON format (.json files).


TAU
===
TAU can be installed using `Spack <https://spack.io>`_ or manually via
instructions in its `install guide
<https://www.cs.uoregon.edu/research/tau/tau-installguide.pdf>`_.

You can instrument and/or sample your program using TAU. To instrument your
program, you can compile it with ``tau_cc.sh`` or ``tau_cxx.sh`` like any other
compiler. To sample your program, you can run it with ``tau_exec``. 

Below, you can find the required environment variables to sample your program
and get call path data using TAU. You can both instrument and sample your
program using 
these environment variables and ``tau_exec`` after compiling your program with ``tau_cc/cxx.sh``. 

.. code-block:: console

   TAU_PROFILE=1
   TAU_CALLPATH=1
   TAU_SAMPLING=1
   TAU_CALLPATH_DEPTH=100
   TAU_EBS_UNWIND=1
   (optional) TAU_METRICS=<TAU/PAPI_metrics>
   (optional) PROFILEDIR=<directore_name_for_profile_data>

After setting these environment variables, you can run your program as:

.. code-block:: console

   $ mpirun -np <num_ranks> tau_exec -T mpi,openmp -ebs ./program.exe <program_args>

More information about using TAU can be found in its `user guide
<https://www.cs.uoregon.edu/research/tau/tau-usersguide.pdf>`_.


timemory
========
Timemory can be installed using `Spack <https://spack.io>`_ or manually as
suggested in its `documentation
<https://timemory.readthedocs.io/en/develop/installation.html>`_.

Timemory can perform both runtime instrumentation and binary rewriting, but
recommends using binary rewriting for distributed memory parallelism.  To use
binary rewriting, you need to first generate an instrumented executable and
then run that instrumented executable as below.

.. code-block:: console

   $ timemory-run <timemory-run_options> -o <instrumented_executable> --mpi -- <executable>
   $ mpirun -np <num_ranks> ./<instrumented_executable>

More information about how to use timemory can be found at https://timemory.readthedocs.io/en/develop/index.html.

..
   Callgrind
   =========

..
   cProfile
   ========


pyinstrument
============
Hatchet can read `pyinstrument <https://github.com/joerick/pyinstrument>`_ JSON
files which can be generated 

by using its Python API or using the command line:

**Command line**

.. code-block:: console

   $ pyinstrument -r json -o <output.json> ./program.py

**Python API**

.. code-block:: python

   from pyinstrument import Profiler
   from pyinstrument.renderers import JSONRenderer
   profiler = Profiler()

   profiler.start()
   # do some work
   profiler.stop()

   print(JSONRenderer().render(profiler.last_session))

