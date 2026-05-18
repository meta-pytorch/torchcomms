# LIBMSR Component

The LIBMSR component supports measuring and capping power usage on recent Intel architectures using the RAPL interface exposed through MSRs (model-specific registers).

* [Enabling the LIBMSR Component](#enabling-the-libmsr-component)
* [Environment Variables](#environment-variables)
* [FAQ](#faq)

***
## Enabling the LIBMSR Component

To enable reading LIBMSR events the user needs to link against a PAPI library that was configured with the LIBMSR component enabled. As an example the following command: `./configure --with-components="libmsr"` is sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in `papi/src/utils/papi_components_avail`) will display the components available to the user, and whether they are disabled, and when they are disabled why.

## Environment Variables

For LIBMSR, PAPI requires one environment variable: **PAPI\_LIBMSR\_ROOT**.

This is required for both compiling, and at runtime. 

Example:

    export PAPI_LIBMSR_ROOT=/sw/libmsr/0.1.17

Within PAPI\_LIBMSR\_ROOT, we expect the following standard directories:

    PAPI_LIBMSR_ROOT/include or PAPI_LIBMSR_ROOT/include/msr
    PAPI_LIBMSR_ROOT/lib

***
## FAQ

1. [Background](#background)
2. [Enable Access to the MSRs](#enable-access-to-the-msrs-model-specific-registers)
3. [Compile the LIBMSR Library to Access the MSRs](#compile-the-libmsr-library-to-access-the-msrs)
4. [Testing PAPI with LIBMSR Enabled](#testing-papi-with-libmsr-enabled)
5. [List LIBMSR Supported Events](#list-libmsr-supported-events)
6. [Use the PAPI LIBMSR Component](#use-the-papi-libmsr-component)

## Background
This libmsr component is an initial version, and has been tested
with libmsr (v0.1.17 11/2015) and the msr_safe kernel module (19/2015
version).

* https://github.com/scalability-llnl/libmsr
* https://github.com/scalability-llnl/msr-safe


Lawrence Livermore National Laboratory has released a library (libmsr)
designed to provide a simple, safe, consistent interface to several of
the model-specific registers (MSRs) in Intel processors.  The problem
is that permitting open access to the MSRs on a machine can be a
safety hazard, so access to MSRs is usually limited.  In order to
encourage system administrators to give wider access to the MSRs on a
machine, LLNL has released a Linux kernel module (msr_safe) which
provides safer, white-listed access to the MSRs.

PAPI has created a libmsr component that can provide read and write
access to the information and controls exposed via the libmsr library.

This PAPI component introduces a new ability for PAPI; it is the first
case where PAPI is writing information to a counter as well as reading
the data from the counter.

## Enable Access to the MSRs (Model Specific Registers)

https://github.com/scalability-llnl/msr-safe

To use this component, the system will need to provide access to Model
Specific Registers (MSRs) from user space.  The actions described
below will generally require superuser ability.  Note, these actions
may have security and performance consequences, so please make sure
you know what you are doing.

### OPTION 1: Enable MSR access using msr-safe
Install the msr-safe module from LLNL. 
       
    lsmod | grep msr        (should show msr_safe)

Use chmod to set site-appropriate access permissions (e.g. 766) for 
       
/dev/cpu/*/msr_safe /dev/cpu/msr_batch /dev/cpu/msr_whitelist

Load a whitelist appropriate for your machine, e.g. for SandyBridge: 
         
    cat msr-safe/whitelists/wl_062D > /dev/cpu/msr_whitelist
    
### OPTION 2: Enable MSR access via the filesystem and elevated permissions
Or, enable access to the standard MSRs filesystem
    
For Linux kernel version < 3.7, using only file system checks
         
    chmod 666 /dev/cpu/*/msr
    
For Linux kernel version >= 3.7, using capabilities
         
    chmod 666 /dev/cpu/*/msr

The final executable needs `CAP_SYS_RWIO` to open MSR device files [1]
         
    setcap cap_sys_rawio=ep <user_executable>
         
The final executable cannot be on a shared network partition.
    
The dynamic linker on most operating systems will remove variables
that control dynamic linking from the environment of executables
with extended rights, such as setuid executables or executables
with raised capabilities. One such variable is
`LD_LIBRARY_PATH`. Therefore, executables that have the RAWIO
capability can only load shared libraries from default system
directories.
    
One can work around this restriction by either installing the
shared libraries in system directories, linking statically against
those libraries, or using the -rpath linker option to specify the
full path to the shared libraries during the linking step.


## Compile the LIBMSR Library to Access the MSRs

https://github.com/scalability-llnl/libmsr

Get the library and follow the instructions to build using CMake.
This library contains a subdirectory, test, which will exercise the
functionality.

## Testing PAPI with LIBMSR Enabled

From papi/src:

    utils/papi_component_avail

For the LMSENSORS component to be operational, it must find the dynamic
library `libmsr.so`.

If it is not found (or is not functional) then the component will be
listed as "disabled" with a reason explaining the problem. If library
was not found, then it is not in the expected place.  The component
can be configured to look for the library in a specific place, and
using an alternate name if desired. Detailed instructions are
contained in the `Rules.libmsr` file.  They are technical, users may wish
to enlist the help of a sysadmin.

## List LIBMSR Supported Events
From papi/src:

    utils/papi_native_avail | grep -i libmsr

## Use the PAPI LIBMSR Component 

See the components/libmsr/utils/README file for instructions.  This
test demonstrates how to write power constraints, and gives an
estimate of the overheads for reading and writing information to the
RAPL MSRs.


[1] http://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/commit/?id=c903f0456bc69176912dee6dd25c6a66ee1aed00


