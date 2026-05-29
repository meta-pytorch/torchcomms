# KokkosP Profiling Tools

Kokkos tools support is built into the main library (`libtimemory.so` or `libtimemory.dylib`) and can be
enabled in any Kokkos application by settings the environment variable `KOKKOS_PROFILE_LIBRARY` to the path
to this library or via the command-line `--kokkos-tools-library=/path/to/libtimemory.<ext>` if the application
passes the arguments from `main(...)` to Kokkos during initialization, i.e. `Kokkos::Initialize(argc, argv);`.
Selecting which components are collected can be done via the kokkos-tools specific environment variable
`TIMEMORY_KOKKOS_COMPONENTS` or the generic environment variable `TIMEMORY_GLOBAL_COMPONENTS`. If timemory
is not directly used anywhere in your application, it does not matter which environment variable is used.
If timemory is used elsewhere and you would like the kokkos-tools component selection to not be applied
globally, use the former.

If you would prefer to avoid setting several environment variables, it maybe useful to create a timemory
config file in `~/.timemory.cfg` or `~/.config/timemory.cfg` or specify the location of the config
file via the environment variable `TIMEMORY_CONFIG=/path/to/file`. The format is simple: `setting = value [values...]`.
Additionally, timemory provides several zero-config libraries when cmake is configured with `TIMEMORY_BUILD_KOKKOS_TOOLS=ON`
and `TIMEMORY_BUILD_KOKKOS_CONFIG=ON`. The exact set of libraries built are subject to which components are
available and will be installed in `${CMAKE_PREFIX_PATH}/${CMAKE_INSTALL_LIBDIR}/timemory/kokkos-tools/`.
These libraries have a default output path to `timemory-@LIB_NAME@-output/@TIMESTAMP@`, where `@LIB_NAME@` is the
basename of the library and `@TIMESTAMP@` is the time-stamp from when timemory was initialized.
Use `timemory-avail -S` to see how to disable time-stamping and customize other settings.

Configuring cmake with `TIMEMORY_BUILD_KOKKOS_TOOLS=ON` and `TIMEMORY_BUILD_KOKKOS_CONFIG=OFF` will install just
two libraries in `${CMAKE_PREFIX_PATH}/${CMAKE_INSTALL_LIBDIR}/timemory/kokkos-tools/`: `kp_timemory.<ext>`
and `kp_timemory_filter.<ext>`. In most cases, using `kp_timemory.<ext>` is not necessary, it is essentially
the same as using `libtimemory.<ext>` except it has hidden symbols so that separate data can be collected
when timemory is used in the main application. `kp_timemory_filter.<ext>` ensures that all external
profiler controller components (currently: vtune_profiler, cuda_profiler, craypat_record, and allinea_map)
are disabled outside of the Kokkos regions matching the regex values of the `KOKKOS_PROFILE_REGEX` environment
variable. The default value of `KOKKOS_PROFILE_REGEX` is `"^[A-Za-z]"` (i.e. any Kokkos label starting with a letter).

## Build

```bash
git clone https://github.com/NERSC/timemory.git timemory
cmake -B build-timemory -DTIMEMORY_BUILD_KOKKOS_TOOLS=ON -DTIMEMORY_BUILD_KOKKOS_CONFIG=ON timemory
cmake --build build-timemory --target install --parallel 8
```

> Specify `-DCMAKE_INSTALL_PREFIX=/path/to/install/dir` and add any additional options, e.g. `-DTIMEMORY_USE_GOTCHA=ON`

## Running Kokkos application with timemory enabled

Before executing the Kokkos application you have to set the environment variable `KOKKOS_PROFILE_LIBRARY`
to point to the path to the installed `libtimemory.<ext>` or pre-built component configuration library.
Here are several examples:

```console
export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/libtimemory.so
export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/timemory/kokkos-tools/kp_timemory.so
export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/timemory/kokkos-tools/kp_timemory_context_switch.so
./myexe --kokkos-tools-library=/opt/timemory/lib/libtimemory.so
```

## Selecting Different Components

Use the `timemory-avail` command-line tool provided by timemory to find the alias for the tool desired. Use `-d` to get a
description of the tool or `-h` to see all options. Once the desired components have been identified, place
the components in a comma-delimited list in the environment variable `TIMEMORY_KOKKOS_COMPONENTS`, e.g.

```console
export TIMEMORY_KOKKOS_COMPONENTS="wall_clock, peak_rss, cpu_roofline_dp_flops"
```

## Filtering Kokkos Labels

The following will filter out any Kokkos profiling label which was not specifically given a name
(and thus was labeled with the demangled function name):

```console
export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/timemory/kokkos-tools/kp_timemory_filter.so
export KOKKOS_PROFILE_REGEX="(?!Kokkos::)"
```

## Kernel Logging

The main timemory library (`libtimemory.<ext>`) supports real-time printing to stderr from all the kokkosp functions
for debugging purposes. This behavior can be activated via `TIMEMORY_KOKKOS_KERNEL_LOGGER=ON` in the environment
or passing `--kokkos-tools-args="--timemory-kokkos-kernel-logger"` to a Kokkos application which passes the
command-line arguments to `Kokkos::initialize`.

## Sample Output

```console
$ export TIMEMORY_KOKKOS_COMPONENTS="wall_clock, peak_rss"
$ export TIMEMORY_TIMING_PRECISION=6
$ export TIMEMORY_MEMORY_UNITS=kb
$ export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/timemory/kokkos-tools/kp_timemory.so
$ ./sample

  Total size S = 4194304 N = 4096 M = 1024
#---------------------------------------------------------------------------#
# KokkosP: timemory Connector (sequence is 0, version: 20210106)
#---------------------------------------------------------------------------#

  Computed result for 4096 x 1024 is 4194304.000000
  N( 4096 ) M( 1024 ) nrepeat ( 100 ) problem( 33.5954 MB ) time( 0.147967 s ) bandwidth( 22.7047 GB/s )

#---------------------------------------------------------------------------#
KokkosP: Finalization of timemory Connector. Complete.
#---------------------------------------------------------------------------#

[kokkos_memory]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/kokkos_memory.json'...
[kokkos_memory]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/kokkos_memory.tree.json'...
[kokkos_memory]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/kokkos_memory.txt'...

[wall]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/wall.json'...
[wall]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/wall.tree.json'...
[wall]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/wall.txt'...

[peak_rss]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/peak_rss.tree.json'...
[peak_rss]|0> Outputting 'timemory-sample-output/2021-02-09_05.27_PM/peak_rss.txt'...

$ cat timemory-sample-output/2021-02-09_05.27_PM/peak_rss.txt
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                             MEASURES CHANGES IN THE HIGH-WATER MARK FOR THE AMOUNT OF MEMORY ALLOCATED IN RAM. MAY FLUCTUATE IF SWAP IS ENABLED                           |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                              LABEL                                | COUNT  | DEPTH  | METRIC   | UNITS  |   SUM     |   MEAN    |   MIN     |   MAX     | STDDEV | % SELF |
|-------------------------------------------------------------------|--------|--------|----------|--------|-----------|-----------|-----------|-----------|--------|--------|
| >>> kokkos/dev0/Kokkos::View::initialization [unsigned long long] |      1 |      0 | peak_rss | kb     |    32.768 |    32.768 |    32.768 |    32.768 |  0.000 |  100.0 |
| >>> kokkos/dev0/Kokkos::View::initialization [long long]          |      1 |      0 | peak_rss | kb     |     8.192 |     8.192 |     8.192 |     8.192 |  0.000 |  100.0 |
| >>> kokkos/dev0/Kokkos::View::initialization [A]                  |      1 |      0 | peak_rss | kb     | 33554.432 | 33554.432 | 33554.432 | 33554.432 |  0.000 |  100.0 |
| >>> kokkos/dev16777216/yAx                                        |    100 |      0 | peak_rss | kb     |    12.288 |     0.123 |    12.288 |    12.288 |  1.229 |  100.0 |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

$ cat timemory-sample-output/2021-02-09_05.27_PM/wall.txt
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                              LABEL                                | COUNT  | DEPTH  | METRIC | UNITS  |   SUM    |   MEAN   |   MIN    |   MAX    | STDDEV   | % SELF |
|-------------------------------------------------------------------|--------|--------|--------|--------|----------|----------|----------|----------|----------|--------|
| >>> kokkos/dev0/Kokkos::View::initialization [unsigned long long] |      1 |      0 | wall   | sec    | 0.000090 | 0.000090 | 0.000090 | 0.000090 | 0.000000 |  100.0 |
| >>> kokkos/dev0/Kokkos::View::initialization [long long]          |      1 |      0 | wall   | sec    | 0.000009 | 0.000009 | 0.000009 | 0.000009 | 0.000000 |  100.0 |
| >>> kokkos/dev0/Kokkos::View::initialization [A]                  |      1 |      0 | wall   | sec    | 0.009131 | 0.009131 | 0.009131 | 0.009131 | 0.000000 |  100.0 |
| >>> kokkos/deep_copy/Host=y/Host=y                                |      1 |      0 | wall   | sec    | 0.000001 | 0.000001 | 0.000001 | 0.000001 | 0.000000 |  100.0 |
| >>> kokkos/deep_copy/Host=x/Host=x                                |      1 |      0 | wall   | sec    | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
| >>> kokkos/deep_copy/Host=A/Host=A                                |      1 |      0 | wall   | sec    | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |  100.0 |
| >>> kokkos/dev16777216/yAx                                        |    100 |      0 | wall   | sec    | 0.146778 | 0.001468 | 0.001307 | 0.001307 | 0.000160 |  100.0 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|

$ cat timemory-sample-output/2021-02-09_05.27_PM/kokkos_memory.txt
|----------------------------------------------------------------------------------------------------------|
|                                           KOKKOS MEMORY TRACKER                                          |
|----------------------------------------------------------------------------------------------------------|
|               LABEL                | COUNT  | DEPTH  |    METRIC     | UNITS  |  SUM   | MEAN   | % SELF |
|------------------------------------|--------|--------|---------------|--------|--------|--------|--------|
| >>> kokkos/allocate/Host/y         |      1 |      0 | kokkos_memory | kb     |     32 |     32 |    100 |
| >>> kokkos/allocate/Host/x         |      1 |      0 | kokkos_memory | kb     |      8 |      8 |    100 |
| >>> kokkos/allocate/Host/A         |      1 |      0 | kokkos_memory | kb     |  33554 |  33554 |    100 |
| >>> kokkos/deep_copy/Host=y/Host=y |      1 |      0 | kokkos_memory | kb     |     32 |     32 |    100 |
| >>> kokkos/deep_copy/Host=x/Host=x |      1 |      0 | kokkos_memory | kb     |      8 |      8 |    100 |
| >>> kokkos/deep_copy/Host=A/Host=A |      1 |      0 | kokkos_memory | kb     |  33554 |  33554 |    100 |
| >>> kokkos/deallocate/Host/A       |      1 |      0 | kokkos_memory | kb     |  33554 |  33554 |    100 |
| >>> kokkos/deallocate/Host/x       |      1 |      0 | kokkos_memory | kb     |      8 |      8 |    100 |
| >>> kokkos/deallocate/Host/y       |      1 |      0 | kokkos_memory | kb     |     32 |     32 |    100 |
|----------------------------------------------------------------------------------------------------------|
```

## Kernel Logging Sample Output

```console
$ export TIMEMORY_KOKKOS_COMPONENTS="wall_clock, peak_rss"
$ export TIMEMORY_TIMING_PRECISION=6
$ export TIMEMORY_MEMORY_UNITS=kb
$ export KOKKOS_PROFILE_LIBRARY=/opt/timemory/lib/libtimemory.so
$ ./sample --kokkos-tools-args="--timemory-kokkos-kernel-logger"

  Total size S = 4194304 N = 4096 M = 1024
#---------------------------------------------------------------------------#
# KokkosP: timemory Connector (sequence is 0, version: 20210106)
#---------------------------------------------------------------------------#

[kokkos_kernel_logger]> kokkosp_allocate_data/Host/y/[0x7fc770008040]/32768
[kokkos_kernel_logger]> kokkosp_begin_parallel_for/Kokkos::View::initialization [y]/0
[kokkos_kernel_logger]> kokkosp_end_parallel_for/0
[kokkos_kernel_logger]> kokkosp_allocate_data/Host/x/[0x7fc77e017a40]/8192
[kokkos_kernel_logger]> kokkosp_begin_parallel_for/Kokkos::View::initialization [x]/1
[kokkos_kernel_logger]> kokkosp_end_parallel_for/1
[kokkos_kernel_logger]> kokkosp_allocate_data/Host/A/[0x7fc76dc00040]/33554432
[kokkos_kernel_logger]> kokkosp_begin_parallel_for/Kokkos::View::initialization [A]/2
[kokkos_kernel_logger]> kokkosp_end_parallel_for/2
[kokkos_kernel_logger]> kokkosp_begin_deep_copy/Host/y/[0x7fc7700080c0]/Host/y/[0x7fc7700080c0]/32768
[kokkos_kernel_logger]> kokkosp_end_deep_copy
[kokkos_kernel_logger]> kokkosp_begin_deep_copy/Host/x/[0x7fc77e017ac0]/Host/x/[0x7fc77e017ac0]/8192
[kokkos_kernel_logger]> kokkosp_end_deep_copy
[kokkos_kernel_logger]> kokkosp_begin_deep_copy/Host/A/[0x7fc76dc000c0]/Host/A/[0x7fc76dc000c0]/33554432
[kokkos_kernel_logger]> kokkosp_end_deep_copy
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/3
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/3
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/4
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/4
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/5
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/5
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/6
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/6
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/7
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/7
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/8
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/8
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/9
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/9
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/10
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/10
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/11
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/11
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/12
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/12
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/13
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/13
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/14
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/14
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/15
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/15
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/16
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/16
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/17
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/17
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/18
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/18
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/19
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/19
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/20
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/20
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/21
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/21
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/22
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/22
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/23
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/23
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/24
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/24
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/25
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/25
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/26
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/26
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/27
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/27
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/28
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/28
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/29
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/29
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/30
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/30
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/31
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/31
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/32
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/32
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/33
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/33
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/34
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/34
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/35
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/35
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/36
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/36
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/37
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/37
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/38
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/38
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/39
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/39
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/40
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/40
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/41
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/41
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/42
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/42
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/43
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/43
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/44
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/44
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/45
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/45
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/46
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/46
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/47
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/47
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/48
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/48
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/49
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/49
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/50
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/50
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/51
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/51
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/52
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/52
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/53
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/53
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/54
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/54
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/55
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/55
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/56
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/56
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/57
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/57
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/58
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/58
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/59
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/59
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/60
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/60
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/61
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/61
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/62
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/62
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/63
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/63
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/64
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/64
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/65
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/65
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/66
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/66
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/67
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/67
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/68
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/68
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/69
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/69
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/70
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/70
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/71
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/71
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/72
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/72
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/73
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/73
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/74
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/74
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/75
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/75
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/76
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/76
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/77
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/77
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/78
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/78
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/79
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/79
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/80
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/80
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/81
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/81
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/82
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/82
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/83
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/83
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/84
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/84
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/85
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/85
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/86
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/86
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/87
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/87
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/88
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/88
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/89
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/89
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/90
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/90
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/91
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/91
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/92
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/92
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/93
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/93
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/94
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/94
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/95
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/95
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/96
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/96
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/97
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/97
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/98
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/98
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/99
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/99
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/100
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/100
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/101
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/101
[kokkos_kernel_logger]> kokkosp_begin_parallel_reduce/yAx/102
[kokkos_kernel_logger]> kokkosp_end_parallel_reduce/102
  Computed result for 4096 x 1024 is 4194304.000000
  N( 4096 ) M( 1024 ) nrepeat ( 100 ) problem( 33.5954 MB ) time( 0.147473 s ) bandwidth( 22.7807 GB/s )
[kokkos_kernel_logger]> kokkosp_deallocate_data/Host/A/[0x7fc76dc00040]/33554432
[kokkos_kernel_logger]> kokkosp_deallocate_data/Host/x/[0x7fc77e017a40]/8192
[kokkos_kernel_logger]> kokkosp_deallocate_data/Host/y/[0x7fc770008040]/32768

#---------------------------------------------------------------------------#
KokkosP: Finalization of timemory Connector. Complete.
#---------------------------------------------------------------------------#

[kokkos_memory]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/kokkos_memory.json'...
[kokkos_memory]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/kokkos_memory.tree.json'...
[kokkos_memory]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/kokkos_memory.txt'...

[wall]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/wall.json'...
[wall]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/wall.tree.json'...
[wall]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/wall.txt'...

[peak_rss]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/peak_rss.json'...
[peak_rss]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/peak_rss.tree.json'...
[peak_rss]|0> Outputting 'timemory-sample-output/2021-02-09_05.30_PM/peak_rss.txt'...
```
