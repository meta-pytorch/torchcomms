# timemory-jump library

The timemory jump library implements the **jump** instrumentation mode for `timemory-run` tool. Additionally, this library can
be linked to in lieu of the traditional timemory library and provide instrumentation via setting the environment variable
`TIMEMORY_JUMP_LIBRARY` on libraries which provide `dlsym` and `dlopen`.

## Description

The **jump** mode is used to insert dynamic instrumentation code in an application binary using **function pointers**. The inserted function pointers dereference to the instrumentation code at runtime alleviating the risk of self instrumentation loop in case a library that is being used by instrumentation code is instrumented. For example, dynamic instrumentation of `libm` may lead to a self instrumentation loop. The **jump** instrumentation mode can be used by using the `--jump` option with `timemory-run` tool.

## About timemory-run tool

Please refer to [timemory-run documentation](../timemory-run/README.md) for infomation about this tool.

## Usage

**NOTE:** Make sure the libtimemory-jump.so is in the `LD_LIBRARY_PATH` environment variable before running `timemory-run`.

```bash
$ timemory-run --jump [OPTIONS] -o <INSTRUMENTED_BINARY> -- <BINARY>
```

## Examples

```bash
$ timemory-run --jump -o lscpu.inst -- /usr/bin/lscpu

 [command]: /usr/bin/lscpu

instrumentation target: /usr/bin/lscpu
loading library: 'libtimemory-jump.so'...
timemory-run: Unable to find function exit
timemory-run: Unable to find function MPI_Init
timemory-run: Unable to find function MPI_Finalize
Instrumenting with 'timemory_push_trace' and 'timemory_pop_trace'...
Parsing module: lscpu
Dumping 'available_module_functions.txt'...
Dumping 'instrumented_module_functions.txt'...

The instrumented executable image is stored in '/home/mhaseeb/lscpu.inst'
[timemory-run]> Getting linked libraries for /usr/bin/lscpu...
[timemory-run]> Consider instrumenting the relevant libraries...

        /lib/x86_64-linux-gnu/libsmartcols.so.1
        /lib/x86_64-linux-gnu/libc.so.6
        /lib64/ld-linux-x86-64.so.2
```

### Testing the instrumented binary
```bash
$ ./lscpu.inst
#------------------------- tim::manager initialized [id=0][pid=12885] -------------------------#

[pid=12885][tid=0][timemory_trace_init@'../source/trace.cpp':636]> rank = 0, pid = 12885, thread = 0, args = wall_clock...
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              12
On-line CPU(s) list: 0-11
Thread(s) per core:  2
Core(s) per socket:  6
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               79
Model name:          Intel(R) Core(TM) i7-6800K CPU @ 3.40GHz
Stepping:            1
CPU MHz:             1204.187
CPU max MHz:         4000.0000
CPU min MHz:         1200.0000
BogoMIPS:            6799.28
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            15360K
NUMA node0 CPU(s):   0-11
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 cdp_l3 invpcid_single pti intel_ppin ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid fsgsbase tsc_adjust bmi1 hle avx2 smep bmi2 erms invpcid rtm cqm rdt_a rdseed adx smap intel_pt xsaveopt cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local dtherm ida arat pln pts md_clear flush_l1d
[wall]|0> Outputting 'timemory-lscpu.inst-output/wall.flamegraph.json'...
[wall]|0> Outputting 'timemory-lscpu.inst-output/wall.json'...
[wall]|0> Outputting 'timemory-lscpu.inst-output/wall.txt'...
/home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/bin/python: Error while finding module specification for 'timemory.plotting' (ModuleNotFoundError: No module named 'timemory')
[timemory]> Command: '/home/mhaseeb/repos/spack/opt/spack/linux-ubuntu18.04-broadwell/gcc-8.4.0/python-3.7.7-2dybrjceqs3qc4k7ci56t56bvzb4csxc/bin/python -m timemory.plotting -f timemory-lscpu.inst-output/wall.json -t "wall " -o timemory-lscpu.inst-output' returned a non-zero exit code: 256... plot/definition.hpp:77 plot generation failed

|------------------------------------------------------------------------------------------------------------------------|
|                                            REAL-CLOCK TIMER (I.E. WALL-CLOCK TIMER)                                    |
| ---------------------------------------------------------------------------------------------------------------------- |
|   LABEL  | COUNT    | DEPTH    | METRIC   | UNITS    | SUM      | MEAN     | MIN      | MAX      | STDDEV   | % SELF   |
| ---------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| >>> main | 1        | 0        | wall     | sec      | 0.026    | 0.026    | 0.026    | 0.026    | 0.000    | 100.0    |
|------------------------------------------------------------------------------------------------------------------------|

[metadata::manager::finalize]> Outputting 'timemory-lscpu.inst-output/metadata.json'...
```
