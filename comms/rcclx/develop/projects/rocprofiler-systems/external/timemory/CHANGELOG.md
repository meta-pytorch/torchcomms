# timemory

## Version 4.0.1rc0

> Date: Tue Dec 3, 2025

### Bug Fixes

- Fixed hard-coded thread limit in `operation::set_storage`
  - Changed from `max_threads = 4096` to use `TIMEMORY_MAX_THREADS`
  - Prevents segfaults when Linux thread IDs exceed 4096

## Version 3.2.4

> Date: Mon Jul 19 17:22:28 2021 -0500

- Fixes issues with Windows CUDA builds
- Removes requirement for NVTX_LIBRARY for NVTX_FOUND

## Version 3.2.3

> Date: Sun Jul 18 05:53:34 2021 -0500

- Fixes: Fatal Python error: PyMUTEX_LOCK(gil->mutex) failed on macOS

## Version 3.2.2

> Date: Wed Jul 14 20:42:29 2021 -0500

- Python gotcha fixes
  - Fixed issues with mallocp segfaulting from Python
  - Fixed storage merge() segfaulting
- New Python tools submodule (timemory.tools)
  - tools.function_wrappers combines {start,stop}_{mpip,ompt,ncclp,mallocp}
    into one configurable handle and provides decorator + context-manager features
- New Python functions which are used within tools.function_wrappers
  - timemory.start_function_wrappers
  - timemory.stop_function_wrappers
- Fixed timemory-python-line-profiler script calling timemory.profiler
- API change in ring_buffer template
  - read/write member functions return pointer to object read/written to
    instead of bytes
- API change in storage and tsettings
  - Classes are declared as final to optimize any vtable calls
- Removed runtime_configurable restriction for do_enumerator_generate
  - This enables user_bundles to be used again in Python
- Added operation::python_class_name
- Updated examples:
  - ex_python_bindings (and libex_python_bindings)
- Fix to get_hash_identifier
- Removed concurrency comparison when generating a diff b/t two runs
- Fixed issues with popen.cpp guarding with TIMEMORY_WINDOWS but never defined

## Version 3.2.1

> Date: Fri Jul 9 16:55:33 2021 -0500

- pytimem fix
  - fix missing import of component_bundle and component_tuple
- added additional python tests
- Ability to build with static libraries: python bindings, mpip library, mallocp library, ompt library, ncclp library, KokkosP libraries
- Setting TIMEMORY_BUILD_PYTHON to OFF now results in searching for external pybind11 install
- Renamed some CMake files in cmake/Modules
- Updated caliper and gotcha submodules to support {CALIPER,GOTCHA}_INSTALL_{CONFIG,HEADER} options
- Added TIMEMORY_INSTALL_PYTHON option
- Fixed BUILD_STATIC_LIBS=ON + CMAKE_POSITION_INDEPENDENT_CODE=ON
- Fixed TIMEMORY_USE_CUDA=ON + TIMEMORY_REQUIRE_PACKAGES=ON to fail
- If TIMEMORY_REQUIRED_PACKAGES=OFF, search for packages first before adding submodule
- Extended setup.py to support more options and support non-development install (no headers or cmake config)
- Removed TIMEMORY_EMBED_PYTHON option
- Disable timemory-jump when no shared libraries are built since dlopen isn't possible
- Replaced allocator member functions construct, destroy, allocate, deallocate with calls to static functions of allocator traits
- added support for CMAKE_ARGS env variable in setup.py
- remove absolute rpath when SKBUILD/SPACK_BUILD (since these have staging directories)
- timemory-{c,cxx,fortran} alias libraries in build tree
- toggled python function profiler to not include line number by default
  - This can cause strange results when generators are used

## Version 3.2.0

> Date: Sun Jun 27 21:10:57 2021 -0500

- Numerous stability fixes
- Fortran module
- Compiler instrumentation
- NCCL support
- timemory-mallocp
- timemory-ncclp
- timemory-nvml
- Python line-by-line tracing
- I/O {read,write}_{char,bytes}
- Network stats components
- libunwind support
- CMake minimum upgraded to 3.15
- Type-traits for tree/flat/timeline
- Hierarchical serialization ([hatchet](https://github.com/hatchet/hatchet) support)
- Concepts
- Improved settings
- Python tracer (line-by-line)
- CTestNotes support
- Command-line options for settings
- Migrated cereal to internal (i.e. `cereal::` -> `tim::cereal::`)
- Dramatically improved Windows support
- Improved kokkos support
  - Command-line options
  - Print help
- XML serialization support
- Shared caches for components
- Support for C++17 `string_view`
- Python bindings to storage classes
- Windows support for different CPU timers
- CUDA Cupti PCSampling support (CUDA v11+)
- User metadata
- Sampling support in opaque (i.e. within user-bundles)
- Static polymorphic base for bundlers
- Namespace re-organization
- CUDA compilation with Clang compiler
- Piecewise installation
- timem support md5sum hashing of command-line
- `papi_threading` setting
- `is_invalid` in base_state
- New operations
  - `stack_push`
  - `stack_pop`
  - `insert`
  - `set_depth_change`
  - `set_is_flat`
  - `set_is_on_stack`
  - `set_is_invalid`
  - `set_iterator`
  - `get_is_flat`
  - `get_is_invalid`
  - `get_is_on_stack`
  - `get_depth`
  - `get_storage`
  - `get_iterator`


## Version 3.0.0

> Date: Wed, 30 Oct 2019 01:23:50 -0700

- Introducing the variadic interface
- Refer to new documentation at https://timemory.readthedocs.io

## Version 2.3.0

> Date: Wed Oct 10 03:11:33 2018 -0700

- Fixed issue with cxx11_abi between compilers
- TIMEMORY_USE_MPI=OFF by default
- timem updates

## Version 2.2.2

> Date: Wed Jun 6 03:19:39 2018 -0700

- Minor fix to avoid very rare FPE when serializing

## Version 2.2.1

> Date: Tue Jun 6 01:32:45 2018 -0700

- fix to TiMemoryConfig.cmake when installed via sudo

## Version 2.2.0

> Date: Tue Jun 5 00:28:10 2018 -0700

- self-cost available in manager + plotting safeguards
- Improved singleton deletion
- alternative colors for when len(_types) == 1 in plotting
- plotting label fix

## Version 2.1.0

> Date: Wed May 16 11:38:28 2018 -0700

- Significant performance improvement (~2x)
- new C interface for TiMemory
    - requires variable assignment and freeing
      - void* atimer = TIMEMORY\_AUTO\_TIMER("")
      - FREE\_TIMEMORY\_AUTO\_TIMER(atimer)
- command-line tools: timem (UNIX-only) and pytimem
- Environment control
    - TIMEMORY\_VERBOSE
    - TIMEMORY\_DISABLE\_TIMER\_MEMORY
    - TIMEMORY\_NUM\_THREADS\_ENV
    - TIMEMORY\_NUM\_THREADS
    - TIMEMORY\_ENABLE
    - TIMEMORY\_TIMING\_FORMAT
    - TIMEMORY\_TIMING\_PRECISION
    - TIMEMORY\_TIMING\_WIDTH
    - TIMEMORY\_TIMING\_UNITS
    - TIMEMORY\_TIMING\_SCIENTIFIC
    - TIMEMORY\_MEMORY\_FORMAT
    - TIMEMORY\_MEMORY\_PRECISION
    - TIMEMORY\_MEMORY\_WIDTH
    - TIMEMORY\_MEMORY\_UNITS
    - TIMEMORY\_MEMORY\_SCIENTIFIC
    - TIMEMORY\_TIMING\_MEMORY\_FORMAT
    - TIMEMORY\_TIMING\_MEMORY\_PRECISION
    - TIMEMORY\_TIMING\_MEMORY\_WIDTH
    - TIMEMORY\_TIMING\_MEMORY\_UNITS
    - TIMEMORY\_TIMING\_MEMORY\_SCIENTIFIC
- Ability of push/pop default formatting
- improved thread-local singleton using C++ shared\_ptrs
    - automatic merge and deletion of manager instances at sub-thread exit
- Hard-code python exe into timemory python scripts
- Various fixes (plotting, argparse, etc.)

## Version 2.0.0

> Date: Wed Apr 25 12:59:06 2018 -0700

- Large re-write of formatting
- Python format module with classes timemory.format.rss and
  timemory.format.timer
- Python units module
- format names variables prefix/suffix instead of begin/close
- timemory.rss\_usage has more initialization options
- Intel -xHOST and -axMIC-AVX512 flags enabled for Intel compilers
- Added units.hpp
- Added formatters.{hpp,cpp}
- Some minor serialization updates

## Version 1.3.1

> Date: Thu Apr 12 02:02:20 2018 -0700

- Fixes to Windows

## Version 1.3.0

> Date: Tue Apr 10 07:40:01 2018 -0700

- Custom TiMemory namespace was removed, now just tim
- Large rewrite of plotting utilities resulting in a significant
  improvement
- Replaced timing\_manager with manager but typedef in C++ and Python
  to allow backwards-compatibility
- Added new features to auto\_timer
- Removed clone from timer
- Added rss\_{tot,self}\_min
- Updated pybind11 to v2.2.2
- Updated docs and README.rst
- Shared library linking + plotting fixes
- All cmake options are not prefixed with TIMEMORY\_
- Improved Windows DLL support
- setup.py will install CMake config properly
- platform-default settings on whether to use dynamic linking
  (Windows=OFF, else=ON)

## Version 1.2.2

> Date: Wed Feb 28 15:31:53 2018 -0800

- Improved testing + memory unit improvements
- Memory units are now always in multiples of 1024
- Added some thread-safety
- Updated README to deprecate is\_class in decorator

## Version 1.2.1

> Date: Wed Feb 28 02:49:51 2018 -0800

- added auto-detection of is\_class in decorators
- Fixed build flags
  - Removed -march=native (GNU) and -xHOST (Intel) from non-debug
    builds as these flags create illegal instructions in Docker --
    specifically NERSC's Edison

## Version 1.2.0

> Date: Tue Feb 6 05:12:56 2018 -0800

- Large restructuring to fix submodule nesting issue
  - Python \3.1 now allows: "from timemory.util import rss\_usage"
  - requires importlib.util
  - not available in older versions
- Better C++ auto\_timer tagging and second option
  - TIMEMORY\_AUTO\_TIMER (<func@'file'>:line)
  - TIMEMORY\_AUTO\_TIMER\_SIMPLE (func)
  - TIMEMORY\_AUTO\_TIMER\_SIMPLE was the old TIMEMORY\_AUTO\_TIMER
- Squashed bugs + I/O and test improvements
- Excluded non-displayed timers (i.e. falling below minimum) from
  setting the output width
- Improved MPI detection
- Included tests in installation --\timemory.tests.run(pattern="")
- timemory.plotting routines have improved handling of bar graphs to
  help eliminate hidden graphs in the overlay
- added context managers
- moved report\_fname field in options to report\_filename
- moved serial\_fname field in options to serial\_filename

## Version 1.1.7

> Date: Wed Jan 31 14:28:19 2018 -0800

- I/O fix for RSS to report negative values (i.e. deallocation)

## Version 1.1.5

> Date: Mon Jan 29 18:46:09 2018 -0800

- Backported CMake to support older version of CMake (previous min:
3.1.3, new min: 2.8.12)

## Version 1.1.3

> Date: Mon Jan 29 18:46:09 2018 -0800

- added timemory.set\_exit\_action(...) capability for defining a
function to handle the exit of the application due to a signal
being raised (e.g. SIGHUP, SIGINT, SIGABRT)

## Version 1.1.2

> Date: Mon Jan 29 16:20:06 2018 -0800

- removed Python 'cmake' requirement in 'setup.py

## Version 1.1.1

> Date: Mon Jan 29 15:00:12 2018 -0800

- Added 'report\_at\_exit' parameter to auto\_timer decorator
- Added added\_args flag for auto\_timer decorator
- Fixed I/O output bug
- Added setup.cfg
- Fixed auto\_timer decorator issue with self.key, self.is\_class,
and self.add\_args

## Version 1.1b0

> Date: Fri Jan 26 17:24:42 2018 -0800

- Updated documentation for TiMemory 1.1b0
- added rss\_usage decorator
- made a base class for the decorators
- update the setup.py to 1.1b0
- +=, -=, +, -, current, and peak methods to RSS in Python
- updated timemory\_test.py
- restructured submodules: originally all submodules were under
util, now only the decorators live there
- new submodules are: options, mpi\_support, plotting, util, and
signals
- timemory.options: I/O options, formerly timemory.util.options
- timemory.plotting: plotting utilities, formerly timemory.util.plot
- timemory.util: decorators, formerly all-encompassing submodule
- timemory.signals: signal enumeration, new submodule
- timemory.mpi\_support: report MPI information, new submodule
- added new RSS capability (+=, -= usage)
- added Python RSS interface
- added signals interface
