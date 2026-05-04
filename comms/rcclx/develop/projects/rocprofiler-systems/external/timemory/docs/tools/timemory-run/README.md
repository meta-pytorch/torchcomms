# timemory-run

Dynamic instrumentation and binary re-writing command-line tool.

## Build Requirements

timemory-run requires [DynInst](https://github.com/dyninst/dyninst), which must be externally installed. Dyninst has several 3rd-party library dependencies so it is highly recommended to use a package manager such as [spack](https://github.com/spack/spack) to install it. 

### Dyninst Installation via Spack

Quick start to installing [DynInst](https://github.com/dyninst/dyninst) via [spack](https://github.com/spack/spack):

```console
git clone https://github.com/spack/spack.git
source ${PWD}/spack/share/spack/setup-env.sh
spack compiler find
spack external find
spack install dyninst
spack load -r dyninst
```

### CMake Option

Once [DynInst](https://github.com/dyninst/dyninst) is installed, enable `-DTIMEMORY_BUILD_TOOLS=ON -DTIMEMORY_USE_DYNINST=ON` in CMake.

## Dynamic Instrumentation Modes

There are two execution modes: (1) runtime-instrumentation and (2) binary rewriting. Runtime instrumentation
will temporarily patch an executable with timemory instrumentation and can be launched on an existing process
or `timemory-run` can launch the executable as a subprocess.
Binary rewriting generates a _new_ executable from an existing executable and cannot be applied to an existing
process.

### Runtime Instrumentation vs. Binary Rewriting

Runtime-instrumentation generates more profiling info because the entire executable along with the linked
libraries are fully loaded into memory, thus instrumentation can be generated for the function calls which
exist in linked libraries. However, in binary rewriting mode, only the executable itself is loaded and
the functions which exist in a linked library cannot be instrumented directly: the executable only has
a _reference_ to the functions and cannot modify the symbol. Thus, one should choose runtime instrumentation
when detailed profilers are desired and binary rewriting should be chosen for targeted analysis of a
specific executable and/or library.

In general, binary rewriting is an excellent choice if only interested in the profiling the function calls
in your executable and/or library. Runtime instrumentation is an excellent choice for detailed profiling
for one single process -- runtime instrumentation is generally __*not*__ the ideal choice for [distributed memory
parallelism](#distributed-memory-parallelism), e.g. MPI, UPC, UPC++.

> Development Note: Currently, the binary rewriting is slightly more stable than runtime-instrumentation

## General Syntax

### Runtime Instrumentation

```console
# general form to run exe as a subprocess
timemory-run <OPTIONS> -- <EXECUTABLE> <ARGS>
# example running exe "foo"
timemory-run -- ./foo
```

```console
# general from to attach to running executable
timemory-run <OPTIONS> -p <PID> -- <EXECUTABLE>
# example attaching to exe with PID of 3252
timemory-run -p 3252
```

### Binary Rewriting

In order to use the binary rewriting mode, specify an output file via the `-o` short option
or the `--output` long option followed by the name of the instrumented file to be generated.
The target executable/library for instrumentation will be the first argument after the `--`.

```console
timemory-run <OPTIONS> -o <OUTPUT_EXECUTABLE> -- <EXECUTABLE>
```

#### Examples

The example below creates a new instrumented executable (`foo.inst`)
from an existing executable `foo`.

```console
timemory-run -o foo.inst -- ./foo
```

The example below creates a new instrumented library (`libomp.so`) in the current
working directory from the system `/usr/lib/libomp.so`.

```console
timemory-run -o libomp.so -- /usr/lib/libomp.so
```

## Component Selection

> Default components: `wall_clock`

The `timemory-run` executable has a `-d/--default-components` option for specifying which components
to use for analysis. The available components can be viewed via the `timemory-avail` command line tool
and the `-s` option to this tool will display the valid string identifiers for these components.
This command line option is overridden by the environment variables:
`TIMEMORY_TRACE_COMPONENTS` in trace mode and `TIMEMORY_COMPONENTS` in region mode
(See [Region vs. Trace](#region-vs-trace)). This command line option can also be left blank or set to `none`
and the environment variable `TIMEMORY_GLOBAL_COMPONENTS` can be used to control the components in
trace and region mode. However, `TIMEMORY_GLOBAL_COMPONENTS` is a fallback environment environment
variable and will be superceded by nearly any other component environment variable. When using
the `--mpip` and/or `--ompt` command line options, these tools check for `TIMEMORY_MPIP_COMPONENTS`
and `TIMEMORY_OMPT_COMPONENTS` respectively, and in the absence of this environment variable,
use `TIMEMORY_GLOBAL_COMPONENTS`. In other words, the modularity of timemory allows for specific tools
to collect their own sets of metrics so each tool generally checks an environment variable unique
to the tool and then search a series of generic environment variables.

### Examples

```console
# binary rewrite w/ cpu_clock
timemory-run -d cpu_clock -o foo.inst -- ./foo
./foo.inst
```

```console
# set the PAPI hardware counters
export TIMEMORY_PAPI_EVENTS="PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_LST_INS"

# binary rewrite w/ wall_clock, peak_rss, and PAPI hardware counters
timemory-run -d wall_clock peak_rss papi_vector -o foo.inst -- ./foo
./foo.inst

# override default components
export TIMEMORY_TRACE_COMPONENTS="wall_clock, cpu_clock"
./foo.inst
```

```console
# runtime instrumentation with a trip counter
timemory-run -d trip_count -- ./foo
```

```console
# runtime instrumentation using TIMEMORY_GLOBAL_COMPONENTS environment variable
export TIMEMORY_GLOBAL_COMPONENTS="wall_clock, thread_cpu_clock"
timemory-run -d -- ./foo
```

## Distributed Memory Parallelism

### Runtime Instrumentation with MPI

The toolkit used for dynamic instrumentation acts as a supervisor for the process when the toolkit launches an executable
as a subprocess. The toolkit is not designed to forward the communicators and thus no communication would occur if
`timemory-run` was launched via `mpirun`, e.g. `mpirun -np 2 timemory-run -- ./foo` because `timemory-run` launches
`foo` via a fork/join operation and all communicator info is lost.
Similarly, `timemory-run -- mpirun -np 2 ./foo` does not work because `mpirun` would be instrumented instead
of `foo` and even if `foo` could be identified as the executable to modify, the instrumentation would be quite
complicated for numerous reasons. If _runtime instrumentation_ is desired for an MPI process, the only current
solution is to launch the MPI jobs, e.g. `mpirun -np 2 ./foo` and attach to one of the processes.

### Runtime Instrumentation with MPI Example

```console
# launch the MPI executable
mpirun -np 2 ./foo &
# get the process ID of one of the MPI ranks
PID=$(pgrep foo | head -n 1)
# attach timemory-run to this process ID
timemory-run -p ${PID}
```

> NOTE: In above, we attach to the first PID of two PIDs generated by `mpirun`.
> The rank which `timemory-run` attached to (rank 0) will be stopped, instrumented, and then
> will resume execution once instrumentation is complete.
> The second process (rank 1) will continue to execute until a synchronization is required with
> the instrumented rank. Thus, communication wait times measured between the instrumented rank
> and the non-instrumented rank(s) will be misleading.

### Binary Rewriting with MPI

Binary rewriting is the preferred method for instrumenting an executable or library which will be utilizing
distributed memory parallelism. In order to use binary rewriting with an MPI process, use the command line
option `--mpi` on any executable or library targeted for instrumentation. This option enables a GOTCHA
wrapper around `MPI_Init` or `MPI_Init_thread` in order to delay the initialization of the timemory
library until after one of these functions have been invoked in the application. If the executable/library
dynamic links to the MPI library, binary rewriting will not instrument the MPI functions. In order to
instrument dynamically linked function calls, one must either create a locally instrumented copy
of `libmpich.so` or `libopenmpi.so` (as demonstrated above with `libomp.so`) or use the `--mpip` command-line option. The
`--mpip` command line option uses a pre-compiled set of MPI GOTCHA wrappers which are activated when
the executable is launched.

### Binary Rewriting with MPI Example

```console
timemory-run -o foo.inst --mpi --mpip -- ./foo
mpirun -np 2 ./foo.inst
```

```console
timemory-run -o foo.inst --mpi -- ./foo
timemory-run -o libmpich.so --mpi -- /usr/lib/libmpich.so
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
mpirun -np 2 ./foo.inst
```

## Command Line Options

`timemory-run --help` will provide a help menu with regard to all the possible options.
There are a few key concepts to understand, however.

### Modules vs. Functions

Identifying a function for instrumentation has two components: a module name and a function name.
The function name as an identifier is self-explanatory. The module is either the name of file which
contained the definition of the function when it was compiled or the name of the library which
contains the symbol for the function.

In general:

- If function `foo` was compiled in `foo.c` and linked into `libfoo.so` (dynamic library)
    - Module name: `libfoo.so`
- If function `foo` was compiled in `foo.c` and linked into `libfoo.a` (static library)
    - Module name: `foo.c`
- If function `foo` was compiled in `foo.c` and linked directly into the executable
    - Module name: `foo.c`

Thus, `timemory-run` provides command line options which use regular expressions (regex)
to permit explicit selection of which modules/functions to include, exclude, or the union of
an exclude and include based on module names and/or function names. When an include option
is present for either category (function or module), `timemory-run` defaults to excluding any
functions or modules which do not match the include expression. In general, the exclude option
should be used remove instrumentation from unwanted functions/modules and the include option should
be used for selecting specific functions/modules.

```console
# include only function names which start with 'foo' or end with 'bar'
timemory-run -I '(^foo|bar$)' -- ./foo

# include only functions defined in modules 'libfoo.so' and 'bar.cpp'
timemory-run -MI libfoo.so bar.cpp -- ./foo

# exclude any functions starting with 'ompt_'
timemory-run -E '^ompt_' -- ./foo

# exclude any functions in libomptarget.so
timemory-run -ME 'libomptarget.so' -- ./foo
```

### Collections

`timemory-run` can accept "collection" files which are an explicit list of the
function names to be instrumented. Several pre-defined collection sets for popular libraries
(e.g. BLAS, CUDA, FFTW, GMP, HDF5, HIP, LAPACK, MPI, OMP, OPENCL, PETSc, UPC)
and category sets of library functions (e.g. memory contains memcmp, memcpy, etc.).

### Region vs. Trace

`timemory-run` has a command line option (`-M/--mode`) which designates whether to synchronize the
instrumentation with the timemory library API.
When `--mode=trace` (default), the components used by the dynamic instrumentation will be independent
of any changes made to the components via the library API. Thus, the library interface can enable/disable
components freely without affecting the dynamic instrumentation and the dynamic instrumentation can be
use for detailed analysis of certain components.
When `--mode=region`, the dynamic instrumentation uses the `timemory_push_region` and `timemory_pop_region`
function calls exposed by the library API.
Any changes to the measurement components via `timemory_set_default`, `timemory_push_components`,
and `timemory_pop_components` will also modify the components used by the dynamic instrumentation.

### Supplemental Libraries

`timemory-run` provides options to enable OpenMP tools (`--ompt`) and MPI (`--mpip`) instrumentation in binary rewrite mode.
These options are useful if generic OpenMP and/or MPI performance info is desired instead of the detailed instrumentation
that would arise from creating an instrumented version of these libraries.

A generic set of options are provided to add instrumentation from custom instrumentation libraries: `--load` takes a
list of libraries, e.g. `--load libfoo libbar` and the default behavior provided by `timemory-run` is search
for two symbols: `void timemory_register_<NAME>()` and `void timemory_deregister_<NAME>()` where `<NAME>` is
the name of the package, i.e. `foo` and `bar` for `libfoo` and `libbar`, respectively. However, one can also
specify a list of initialization and finalization functions via `--init-functions` and `--fini-functions`.
These symbols get inserted into before and after `main` in an executable and within `_init` and `_fini`
in a library. Although it might seem more intuitive for these libraries to be injected into the instrumentation
around the function calls, new components can be added easily:

```cpp
#include "timemory/library.h"
#include "timemory/timemory.hpp"

extern "C" void
timemory_register_ex_custom_dynamic_instr()
{
    using namespace tim::component;
    // insert monotonic clock component into structure
    // used by timemory-run in --mode=trace
    user_trace_bundle::global_init(nullptr);
    user_trace_bundle::configure<monotonic_clock>();

    // insert monotonic clock component into structure
    // used by timemory-run in --mode=region
    timemory_add_components("monotonic_clock");
}

extern "C" void
timemory_deregister_ex_custom_dynamic_instr()
{}
```

> `timemory-run -o foo.inst --load libex_custom_dynamic_instr.so -- ./foo`
