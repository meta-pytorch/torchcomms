# timem

Command-line tool which provides the same capabilities of the UNIX command-line tool `time` but extends it to several additional metrics:

- Percent CPU utilization
- Memory usage
- Page faults
- Context switches
- Read/write bytes
- Read/write bytes per second
- Hardware counters (when built with PAPI support)
- Aggregation of data for multiple MPI processes (when using `timem-mpi`)

## Installation

### General

`timem` is automatically built whenever `TIMEMORY_BUILD_TIMEM=ON`. This option defaults to the value of `TIMEMORY_BUILD_TOOLS=ON` on non-Windows systems.

```console
cmake -B /path/to/build -DTIMEMORY_BUILD_TIMEM=ON /path/to/source
```

### Standalone

If you want to build and install `timem` exclusively without building or installing anything else from timemory,
this is possible by setting `TIMEMORY_INSTALL_ALL=OFF`, which removes the CMake target dependency on `all` for the `install` target (i.e. `make install` does not run `make all` first). In this situation, since you will have
an "incomplete" build, it is advisable to also suppress installation of the CMake configuration files
(`TIMEMORY_INSTALL_CONFIG=OFF`) and the installation of the timemory headers (`TIMEMORY_INSTALL_HEADERS=OFF`).

```console
$ cmake -B /path/to/build \
    -DCMAKE_INSTALL_PREFIX=/path/to/install
    -DTIMEMORY_INSTALL_HEADERS=OFF \
    -DTIMEMORY_INSTALL_CONFIG=OFF \
    -DTIMEMORY_INSTALL_ALL=OFF \
    -DTIMEMORY_BUILD_TIMEM=ON \
    /path/to/source
$ cmake --build /path/to/build --target timem
$ cmake --build /path/to/build --target install
```

## Output Examples

```console
$ timem ls /
bin  boot  dev	etc  home  lib	lib32  lib64  libx32  media  mnt  opt  proc  root  run	sbin  srv  sys	tmp  usr  var

[ls]> Measurement totals:
            0.001380 sec wall
            0.000000 sec user
            0.000000 sec sys
            0.000000 sec cpu
            0.000000 % cpu_util
            2.152000 MB peak_rss
            0.004096 MB page_rss
            0.004096 MB virtual_memory
                   0 major_page_flts
                 122 minor_page_flts
                   0 prio_cxt_swch
                   1 vol_cxt_swch
            0.001116 MB char read
            0.912436 MB/sec char read
            0.000000 MB bytes read
            0.000000 MB/sec bytes read
            0.000000 MB char written
            0.000000 MB/sec char written
            0.000000 MB bytes written
            0.000000 MB/sec bytes written
```

### JSON Output

```console
$ timem -o sleep -- sleep 2

[sleep]> Outputting 'sleep.json'...
[sleep]> Measurement totals:
            1.997019 sec wall
            0.000000 sec user
            0.000000 sec sys
            0.000000 sec cpu
            0.000000 % cpu_util
            1.912000 MB peak_rss
            0.798720 MB page_rss
            4.648960 MB virtual_memory
                   0 major_page_flts
                  87 minor_page_flts
                   0 prio_cxt_swch
                   2 vol_cxt_swch
            0.003896 MB char read
            0.021460 MB/sec char read
            0.000000 MB bytes read
            0.000000 MB/sec bytes read
            0.000000 MB char written
            0.000000 MB/sec char written
            0.000000 MB bytes written
            0.000000 MB/sec bytes written
```

### Aggregated MPI Output

```console
mpirun -n 2 timem-mpi -- sleep 2

[sleep]> Measurement totals (# ranks = 2):
    0|>         3.998239 sec wall
    0|>         0.000000 sec user
    0|>         0.000000 sec sys
    0|>         0.000000 sec cpu
    0|>         0.000000 % cpu_util
    0|>        13.040000 MB peak_rss
    0|>         9.879552 MB page_rss
    0|>       117.964800 MB virtual_memory
    0|>                0 major_page_flts
    0|>              187 minor_page_flts
    0|>                2 prio_cxt_swch
    0|>                4 vol_cxt_swch
    0|>         0.007792 MB char read
    0|>         0.021438 MB/sec char read
    0|>         0.000000 MB bytes read
    0|>         0.000000 MB/sec bytes read
    0|>         0.000000 MB char written
    0|>         0.000000 MB/sec char written
    0|>         0.000000 MB bytes written
    0|>         0.000000 MB/sec bytes written
```

### Individual MPI Output

```console
$ mpirun -n 2 timem-mpi -i -- sleep 2

[sleep]> Measurement totals (# ranks = 2):
    0|>         1.997152 sec wall
    0|>         0.000000 sec user
    0|>         0.000000 sec sys
    0|>         0.000000 sec cpu
    0|>         0.000000 % cpu_util
    0|>         6.488000 MB peak_rss
    0|>         4.947968 MB page_rss
    0|>        58.994688 MB virtual_memory
    0|>                0 major_page_flts
    0|>               90 minor_page_flts
    0|>                1 prio_cxt_swch
    0|>                2 vol_cxt_swch
    0|>         0.003896 MB char read
    0|>         0.021459 MB/sec char read
    0|>         0.000000 MB bytes read
    0|>         0.000000 MB/sec bytes read
    0|>         0.000000 MB char written
    0|>         0.000000 MB/sec char written
    0|>         0.000000 MB bytes written
    0|>         0.000000 MB/sec bytes written
[sleep]> Measurement totals (# ranks = 2):
    1|>         1.996507 sec wall
    1|>         0.000000 sec user
    1|>         0.000000 sec sys
    1|>         0.000000 sec cpu
    1|>         0.000000 % cpu_util
    1|>         6.484000 MB peak_rss
    1|>         0.856064 MB page_rss
    1|>         4.648960 MB virtual_memory
    1|>                0 major_page_flts
    1|>               91 minor_page_flts
    1|>                0 prio_cxt_swch
    1|>                2 vol_cxt_swch
    1|>         0.003896 MB char read
    1|>         0.021467 MB/sec char read
    1|>         0.000000 MB bytes read
    1|>         0.000000 MB/sec bytes read
    1|>         0.000000 MB char written
    1|>         0.000000 MB/sec char written
    1|>         0.000000 MB bytes written
    1|>         0.000000 MB/sec bytes written
```

## Options

### Command Line

#### Standard Executable

The `timem` and `timem-mpi` executation wrappers support simple prefixing of the executable and any arguments
are assumed to belong to the file being executed unless `"--"` (two dashes) surrounded by spaces is detected.
If the two stand-alone dashes are detected, these executables assume every argument preceding the
`"--"` is an argument to `timem`/`timem-mpi` and every argument following the two dashes is the command
to execute, e.g. `timem <timem-arguments> -- <exe> <exe-arguments>`.

```console
$ timem --help
Usage: ./timem [tim::argparse::argument_parser arguments...] -- <CMD> <ARGS>

Options:
    -h, -?, --help                 Shows this page
    --debug                        Debug output
    -v, --verbose                  Verbose output
    -q, --quiet                    Suppress as much reporting as possible
    -d, --sample-delay             Set the delay before the sampler starts (seconds)
    -f, --sample-freq              Set the frequency of the sampler (number of interrupts per second)
    --disable-sample               Disable UNIX signal-based sampling.
                                   Sampling is the most common culprit for timem hanging (i.e. failing to exit after the child process exits)
    -e, --events, --papi-events    Set the hardware counter events to record (ref: `timemory-avail -H | grep PAPI`)
    --disable-papi                 Disable hardware counters
    -o, --output                   Write results to JSON output file.
                                   Use:
                                   - '%p' to encode the process ID
                                   - '%j' to encode the slurm job ID
                                   - '%r' to encode the MPI comm rank
                                   - '%s' to encode the MPI comm size
                                   E.g. '-o timem-output-%p'.
                                   If verbosity >= 2 or debugging is enabled, will also write sampling data to log file.
    -s, --shell                    Enable launching command via a shell command (if no arguments, $SHELL is used)
    --shell-flags                  Set the shell flags to use (pass as single string as leading dashes can confuse parser) [default: -i]
```

#### MPI-Compatible Executable

```console
$ ./timem-mpi --help
Usage: ./timem-mpi [tim::argparse::argument_parser arguments...] -- <CMD> <ARGS>

Options:
    -h, -?, --help                 Shows this page
    --debug                        Debug output
    -v, --verbose                  Verbose output
    -q, --quiet                    Suppress as much reporting as possible
    -d, --sample-delay             Set the delay before the sampler starts (seconds)
    -f, --sample-freq              Set the frequency of the sampler (number of interrupts per second)
    --disable-sample               Disable UNIX signal-based sampling.
                                   Sampling is the most common culprit for timem hanging (i.e. failing to exit after the child process exits)
    -e, --events, --papi-events    Set the hardware counter events to record (ref: `timemory-avail -H | grep PAPI`)
    --disable-papi                 Disable hardware counters
    -o, --output                   Write results to JSON output file.
                                   Use:
                                   - '%p' to encode the process ID
                                   - '%j' to encode the slurm job ID
                                   - '%r' to encode the MPI comm rank
                                   - '%s' to encode the MPI comm size
                                   E.g. '-o timem-output-%p'.
                                   If verbosity >= 2 or debugging is enabled, will also write sampling data to log file.
    -s, --shell                    Enable launching command via a shell command (if no arguments, $SHELL is used)
    --shell-flags                  Set the shell flags to use (pass as single string as leading dashes can confuse parser) [default: -i]
    --mpi                          Launch processes via MPI_Comm_spawn_multiple (reduced functionality)
    --disable-mpi                  Disable MPI_Finalize
    -i, --indiv                    Output individual results for each process (i.e. rank) instead of reporting the aggregation
```

The `--mpi` option is only recommended if forking MPI processes causes issues.
The processes spawn through this method are not children of the `timem-mpi` process and
thus `timem-mpi` is more restricted with respect to the data that it can collect for the
spawned processes. The `--disable-mpi` option is generally not recommended unless `timem-mpi`
is hanging and using `--disable-sample` does not resolve the issue.

### Environment

All the standard environment variables provided by `timemory-avail -S` are supported (when applicable).
However, these environment variables use `TIMEM_` as the prefix to the environment variable instead
of `TIMEMORY_`, e.g. instead of using `TIMEMORY_PRECISION=4` to set the output precision to 4, use `TIMEM_PRECISION=4`.

In addition to the standard environment variables, many of the command-line options are configurable via the
following environment variables:

| Environment Variable | Default value           | Equivalent Command-line Argument         |
| -------------------- | ----------------------- | ---------------------------------------- |
| `TIMEM_OUTPUT`       | `""`                    | `-o`, `--output` (with argument)         |
| `TIMEM_USE_SHELL`    | `"OFF"`                 | `-s`, `--shell` (with no argument)       |
| `TIMEM_SHELL_FLAGS`  | `""`                    | `--shell-flags`                          |
| `TIMEM_SHELL`        | `$SHELL`                | `-s`, `--shell` (with argument)          |
| `TIMEM_SAMPLE`       | `"ON"`                  | `--disable-sample` (if value is `"OFF"`) |
| `TIMEM_SAMPLE_FREQ`  | `5.0`                   | `-f`, `--sample-freq`                    |
| `TIMEM_SAMPLE_DELAY` | `1.0e-6`                | `-d`, `--sample-delay`                   |
| `TIMEM_USE_MPI`      | `false`                 | `--mpi`                                  |
| `TIMEM_USE_PAPI`     | `true` (when available) | `--disable-papi` (if value is `"OFF"`)   |

> Command-line arguments override environment variables,
> e.g. `TIMEM_OUTPUT=foo timem -o bar -- <CMD>` will output `bar.json`, not `foo.json`.

## Customization Demonstration

The ability to customize the behavior of several components without altering the components themselves in demonstrated in
`timem.hpp`. In this tool, additional type-traits are added, the printing output of a bundle of components is customized,
and new capabilities are introduced:

- Introduction of `custom_print` and `timem_sample` operations
  - `custom_print` replaces comma delimiters with new line delimiters
  - `timem_sample` invokes a
- Specialization of the `trait::custom_label_printing` type-trait for `papi_array_t`
  - This type-trait communicates to timemory that data labels for the components should not be included when invoking `operator<<` on this component
- Specialization of the `sample`, `start`, and `stop` operations for `papi_array_t` component
  - The default behavior of `operation::sample<T>` pushes a measurement to an array in the `storage<T>`
  - However, in this tool, we modify the operation to just update the value of the current object
- Specialization of the `operation::base_printer<T>` for `read_bytes` and `written_bytes` components
  - The default behavior of `read_bytes` and `written_bytes` is to print the bytes read/written + the bytes read/written
  per second on the same line. This is customized to print on separate lines
- Introduction of `timem_tuple` -- an extension to `component_tuple` which:
  - Reimplements the `component_tuple<T...>::sample()` member function to include the `timem_sample` operation
  - Replaces the `operator<<` to utilize the `custom_print` operation
