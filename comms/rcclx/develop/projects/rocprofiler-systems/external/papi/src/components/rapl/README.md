# RAPL Component

The RAPL component enables PAPI to access Linux RAPL energy measurements.

* [Enabling the RAPL Component](#enabling-the-rapl-component)
* [Known Limitations](#known-limitations)

***
## Enabling the RAPL Component

To enable reading of RAPL counters the user needs to link against a
PAPI library that was configured with the RAPL component enabled. As an
example the following command: `./configure --with-components="rapl"` is
sufficient to enable the component.

Typically, the utility `papi_components_avail` (available in
`papi/src/utils/papi_components_avail`) will display the components available
to the user, and whether they are disabled, and when they are disabled why.

## Known Limitations

RAPL \_ENERGY\_ values 2019-11-08: The MSRs for energy return a uint64; but only
the bottom 32 bits are meaningful; the upper 32 bits are "reserved" by Intel,
and not guaranteed to be zeros. Before using these values arithmetically, the
upper bits need to be masked to zeros. This is now done. These same MSR can
wraparound; but the energy is a monotonically increasing amount and this is
what we should report.  To prevent PAPI from reporting a wrap-around, at each
read we compute the difference between what we read and what we previously
read (including at PAPI\_start), handling any overflow, and add this to a 64
bit accumulator which is what we report. We always zero the accumulator at any
PAPI\_start.

RAPL uses the MSR kernel module to read model specific registers (MSRs) from
user space. To enable the msr module interface the admin needs to 'chmod 666
/dev/cpu/*/msr'.  For kernels older than 3.7, this is all that is required to
use the PAPI RAPL component.

Historically, the Linux MSR driver only relied upon file system checks. This
means that anything as root with any capability set could read and write to
MSRs.

Changes in the mainline Linux kernel since around 3.7 now require an
executable to have capability CAP\_SYS\_RAWIO to open the MSR device file [1].
This change impacts user programs that use PAPI APIs that rely on the MSR
device driver. Besides loading the MSR kernel module and setting the
appropriate file permissions on the msr device file, one must grant the
CAP\_SYS\_RAWIO capability to any user executable that needs access to the MSR
driver, using the command below:

	setcap cap_sys_rawio=ep <user_executable>

Note that one needs superuser privileges to grant the RAWIO capability to an
executable, and that the executable cannot be located on a shared network file
system partition.

The dynamic linker on most operating systems will remove variables that
control dynamic linking from the environment of executables with extended
rights, such as setuid executables or executables with raised capabilities.
One such variable is LD\_LIBRARY\_PATH. Therefore, executables that have the
RAWIO capability can only load shared libraries from default system
directories.  One can work around this restriction by either installing the
shared libraries in system directories, linking statically against those
libraries, or using the -rpath linker option to specify the full path to the
shared libraries during the linking step.

[1] http://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/commit/?id=c903f0456bc69176912dee6dd25c6a66ee1aed00

