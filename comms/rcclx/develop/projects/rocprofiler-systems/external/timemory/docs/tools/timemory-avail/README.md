# timemory-avail

Command-line tool for queries about component availability, settings, and hardware counters.

## Available Components

`timemory-avail -C` provides the list of available components. Using the `-s` option provides the list of available string identifiers
which can be used in the environment variable settings.

> Sample output from `timemory-avail -C --description --brief`

| COMPONENT                                  | DESCRIPTION                                                                                                                        |
| ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `allinea_map`                              | Controls the AllineaMAP sampler                                                                                                    |
| `caliper`                                  | Forwards markers to Caliper instrumentation                                                                                        |
| `cpu_clock`                                | Total CPU time spent in both user- and kernel-mode                                                                                 |
| `cpu_roofline<double>`                     | Model used to provide performance relative to the peak possible performance on a CPU architecture.                                 |
| `cpu_roofline<float, double>`              | Model used to provide performance relative to the peak possible performance on a CPU architecture.                                 |
| `cpu_roofline<float>`                      | Model used to provide performance relative to the peak possible performance on a CPU architecture.                                 |
| `cpu_util`                                 | Percentage of CPU-clock time divided by wall-clock time                                                                            |
| `craypat_counters`                         | Names and value of any counter events that have been set to count on the hardware category                                         |
| `craypat_flush_buffer`                     | Writes all the recorded contents in the data buffer. Returns the number of bytes flushed                                           |
| `craypat_heap_stats`                       | Undocumented by 'pat_api.h'                                                                                                        |
| `craypat_record`                           | Toggles CrayPAT recording on calling thread                                                                                        |
| `craypat_region`                           | Adds region labels to CrayPAT output                                                                                               |
| `cuda_event`                               | Records the time interval between two points in a CUDA stream. Less accurate than 'cupti_activity' for kernel timing               |
| `cuda_profiler`                            | Control switch for a CUDA profiler running on the application                                                                      |
| `cupti_activity`                           | Wall-clock execution timing for the CUDA API                                                                                       |
| `cupti_counters`                           | Hardware counters for the CUDA API                                                                                                 |
| `current_peak_rss`                         | Absolute value of high-water mark of memory allocation in RAM                                                                      |
| `data_rss`                                 | Integral unshared data size                                                                                                        |
| `gperftools_cpu_profiler`                  | Control switch for gperftools CPU profiler                                                                                         |
| `gperftools_heap_profiler`                 | Control switch for the gperftools heap profiler                                                                                    |
| `gpu_roofline<double>`                     | Model used to provide performance relative to the peak possible performance on a GPU architecture.                                 |
| `gpu_roofline<cuda::half2, float, double>` | Model used to provide performance relative to the peak possible performance on a GPU architecture.                                 |
| `gpu_roofline<cuda::half2>`                | Model used to provide performance relative to the peak possible performance on a GPU architecture.                                 |
| `gpu_roofline<float>`                      | Model used to provide performance relative to the peak possible performance on a GPU architecture.                                 |
| `kernel_mode_time`                         | CPU time spent executing in kernel mode (via rusage)                                                                               |
| `likwid_marker`                            | LIKWID perfmon (CPU) marker forwarding                                                                                             |
| `likwid_nvmarker`                          | LIKWID nvmon (GPU) marker forwarding                                                                                               |
| `malloc_gotcha`                            | GOTCHA wrapper for memory allocation functions                                                                                     |
| `monotonic_clock`                          | Wall-clock timer which will continue to increment even while the system is asleep                                                  |
| `monotonic_raw_clock`                      | Wall-clock timer unaffected by frequency or time adjustments in system time-of-day clock                                           |
| `num_io_in`                                | Number of times the filesystem had to perform input                                                                                |
| `num_io_out`                               | Number of times the filesystem had to perform output                                                                               |
| `num_major_page_faults`                    | Number of page faults serviced that required I/O activity                                                                          |
| `num_minor_page_faults`                    | Number of page faults serviced without any I/O activity via 'reclaiming' a page frame from the list of pages awaiting reallocation |
| `num_msg_recv`                             | Number of IPC messages received                                                                                                    |
| `num_msg_sent`                             | Number of IPC messages sent                                                                                                        |
| `num_signals`                              | Number of signals delivered                                                                                                        |
| `num_swap`                                 | Number of swaps out of main memory                                                                                                 |
| `nvtx_marker`                              | Generates high-level region markers for CUDA profilers                                                                             |
| `ompt_handle<api::native_tag>`             | Control switch for enabling/disabling OpenMP tools defined by the api::native_tag tag                                              |
| `page_rss`                                 | Amount of memory allocated in pages of memory. Unlike peak_rss, value will fluctuate as memory is freed/allocated                  |
| `papi_array<8ul>`                          | Fixed-size array of PAPI HW counters                                                                                               |
| `papi_vector`                              | Dynamically allocated array of PAPI HW counters                                                                                    |
| `peak_rss`                                 | Measures changes in the high-water mark for the amount of memory allocated in RAM. May fluctuate if swap is enabled                |
| `priority_context_switch`                  | Number of context switch due to higher priority process becoming runnable or because the current process exceeded its time slice)  |
| `process_cpu_clock`                        | CPU-clock timer for the calling process (all threads)                                                                              |
| `process_cpu_util`                         | Percentage of CPU-clock time divided by wall-clock time for calling process (all threads)                                          |
| `read_bytes`                               | Physical I/O reads                                                                                                                 |
| `stack_rss`                                | Integral unshared stack size                                                                                                       |
| `system_clock`                             | CPU time spent in kernel-mode                                                                                                      |
| `tau_marker`                               | Forwards markers to TAU instrumentation (via Tau_start and Tau_stop)                                                               |
| `thread_cpu_clock`                         | CPU-clock timer for the calling thread                                                                                             |
| `thread_cpu_util`                          | Percentage of CPU-clock time divided by wall-clock time for calling thread                                                         |
| `trip_count`                               | Counts number of invocations                                                                                                       |
| `user_clock`                               | CPU time spent in user-mode                                                                                                        |
| `user_bundle<10000ul, api::native_tag>`    | Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion        |
| `user_bundle<11100ul, api::native_tag>`    | Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion        |
| `user_mode_time`                           | CPU time spent executing in user mode (via rusage)                                                                                 |
| `user_bundle<11111ul, api::native_tag>`    | Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion        |
| `user_bundle<11110ul, api::native_tag>`    | Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion        |
| `user_bundle<11000ul, api::native_tag>`    | Generic bundle of components designed for runtime configuration by a user via environment variables and/or direct insertion        |
| `virtual_memory`                           | Records the change in virtual memory                                                                                               |
| `voluntary_context_switch`                 | Number of context switches due to a process voluntarily giving up the processor before its time slice was completed                |
| `vtune_event`                              | Creates events for Intel profiler running on the application                                                                       |
| `vtune_frame`                              | Creates frames for Intel profiler running on the application                                                                       |
| `vtune_profiler`                           | Control switch for Intel profiler running on the application                                                                       |
| `wall_clock`                               | Real-clock timer (i.e. wall-clock timer)                                                                                           |
| `written_bytes`                            | Physical I/O writes                                                                                                                |

## Available Settings

`timemory-avail -S` provides the list of available environment variable settings, descriptions, and C++ static accessors.

> Sample output from `timemory-avail -S --description --brief --value`

| ENVIRONMENT VARIABLE              | DATA TYPE      | DESCRIPTION                                                                                                                   |
| --------------------------------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| TIMEMORY_SUPPRESS_PARSING         | bool           | Disable parsing environment                                                                                                   |
| TIMEMORY_ENABLED                  | bool           | Activation state of timemory                                                                                                  |
| TIMEMORY_AUTO_OUTPUT              | bool           | Generate output at application termination                                                                                    |
| TIMEMORY_COUT_OUTPUT              | bool           | Write output to stdout                                                                                                        |
| TIMEMORY_FILE_OUTPUT              | bool           | Write output to files                                                                                                         |
| TIMEMORY_TEXT_OUTPUT              | bool           | Write text output files                                                                                                       |
| TIMEMORY_JSON_OUTPUT              | bool           | Write json output files                                                                                                       |
| TIMEMORY_DART_OUTPUT              | bool           | Write dart measurements for CDash                                                                                             |
| TIMEMORY_TIME_OUTPUT              | bool           | Output data to subfolder w/ a timestamp (see also: TIMEMORY_TIME_FORMAT)                                                      |
| TIMEMORY_PLOT_OUTPUT              | bool           | Generate plot outputs from json outputs                                                                                       |
| TIMEMORY_DIFF_OUTPUT              | bool           | Generate a difference output vs. a pre-existing output (see also: TIMEMORY_INPUT_PATH and TIMEMORY_INPUT_PREFIX)              |
| TIMEMORY_FLAMEGRAPH_OUTPUT        | bool           | Write a json output for flamegraph visualization (use chrome://tracing)                                                       |
| TIMEMORY_VERBOSE                  | int            | Verbosity level                                                                                                               |
| TIMEMORY_DEBUG                    | bool           | Enable debug output                                                                                                           |
| TIMEMORY_BANNER                   | bool           | Notify about manager creation and destruction                                                                                 |
| TIMEMORY_FLAT_PROFILE             | bool           | Set the label hierarchy mode to default to flat                                                                               |
| TIMEMORY_TIMELINE_PROFILE         | bool           | Set the label hierarchy mode to default to timeline                                                                           |
| TIMEMORY_COLLAPSE_THREADS         | bool           | Enable/disable combining thread-specific data                                                                                 |
| TIMEMORY_MAX_DEPTH                | unsigned short | Set the maximum depth of label hierarchy reporting                                                                            |
| TIMEMORY_TIME_FORMAT              | string         | Customize the folder generation when TIMEMORY_TIME_OUTPUT is enabled (see also: strftime)                                     |
| TIMEMORY_PRECISION                | short          | Set the global output precision for components                                                                                |
| TIMEMORY_WIDTH                    | short          | Set the global output width for components                                                                                    |
| TIMEMORY_MAX_WIDTH                | int            | Set the maximum width for component label outputs                                                                             |
| TIMEMORY_SCIENTIFIC               | bool           | Set the global numerical reporting to scientific format                                                                       |
| TIMEMORY_TIMING_PRECISION         | short          | Set the precision for components with 'is_timing_category' type-trait                                                         |
| TIMEMORY_TIMING_WIDTH             | short          | Set the output width for components with 'is_timing_category' type-trait                                                      |
| TIMEMORY_TIMING_UNITS             | string         | Set the units for components with 'uses_timing_units' type-trait                                                              |
| TIMEMORY_TIMING_SCIENTIFIC        | bool           | Set the numerical reporting format for components with 'is_timing_category' type-trait                                        |
| TIMEMORY_MEMORY_PRECISION         | short          | Set the precision for components with 'is_memory_category' type-trait                                                         |
| TIMEMORY_MEMORY_WIDTH             | short          | Set the output width for components with 'is_memory_category' type-trait                                                      |
| TIMEMORY_MEMORY_UNITS             | string         | Set the units for components with 'uses_memory_units' type-trait                                                              |
| TIMEMORY_MEMORY_SCIENTIFIC        | bool           | Set the numerical reporting format for components with 'is_memory_category' type-trait                                        |
| TIMEMORY_MPI_INIT                 | bool           | Enable/disable timemory calling MPI_Init / MPI_Init_thread during certain timemory_init(...) invocations                      |
| TIMEMORY_MPI_FINALIZE             | bool           | Enable/disable timemory calling MPI_Finalize during timemory_finalize(...) invocations                                        |
| TIMEMORY_MPI_THREAD               | bool           | Call MPI_Init_thread instead of MPI_Init (see also: TIMEMORY_MPI_INIT)                                                        |
| TIMEMORY_MPI_THREAD_TYPE          | string         | MPI_Init_thread mode: 'single', 'serialized', 'funneled', or 'multiple' (see also: TIMEMORY_MPI_INIT and TIMEMORY_MPI_THREAD) |
| TIMEMORY_MPI_OUTPUT_PER_RANK      | bool           | Generate MPI output per-rank (skip aggregation)                                                                               |
| TIMEMORY_MPI_OUTPUT_PER_NODE      | bool           | Aggregate MPI output per-node                                                                                                 |
| TIMEMORY_OUTPUT_PATH              | string         | Explicitly specify the output folder for results                                                                              |
| TIMEMORY_OUTPUT_PREFIX            | string         | Explicitly specify a prefix for all output files                                                                              |
| TIMEMORY_INPUT_PATH               | string         | Explicitly specify the input folder for difference comparisons (see also: TIMEMORY_DIFF_OUTPUT)                               |
| TIMEMORY_INPUT_PREFIX             | string         | Explicitly specify the prefix for input files used in difference comparisons (see also: TIMEMORY_DIFF_OUTPUT)                 |
| TIMEMORY_INPUT_EXTENSIONS         | string         | File extensions used when searching for input files used in difference comparisons (see also: TIMEMORY_DIFF_OUTPUT)           |
| TIMEMORY_DART_TYPE                | string         | Only echo this measurement type (see also: TIMEMORY_DART_OUTPUT)                                                              |
| TIMEMORY_DART_COUNT               | unsigned long  | Only echo this number of dart tags (see also: TIMEMORY_DART_OUTPUT)                                                           |
| TIMEMORY_DART_LABEL               | bool           | Echo the category instead of the label (see also: TIMEMORY_DART_OUTPUT)                                                       |
| TIMEMORY_CPU_AFFINITY             | bool           | Enable pinning threads to CPUs (Linux-only)                                                                                   |
| TIMEMORY_TARGET_PID               | int            | Process ID for the components which require this                                                                              |
| TIMEMORY_STACK_CLEARING           | bool           | Enable/disable stopping any markers still running during finalization                                                         |
| TIMEMORY_ADD_SECONDARY            | bool           | Enable/disable components adding secondary (child) entries                                                                    |
| TIMEMORY_THROTTLE_COUNT           | unsigned long  | Minimum number of laps before throttling                                                                                      |
| TIMEMORY_THROTTLE_VALUE           | unsigned long  | Average call time in nanoseconds when # laps > throttle_count that triggers throttling                                        |
| TIMEMORY_PAPI_MULTIPLEXING        | bool           | Enable multiplexing when using PAPI                                                                                           |
| TIMEMORY_PAPI_FAIL_ON_ERROR       | bool           | Configure PAPI errors to trigger a runtime error                                                                              |
| TIMEMORY_PAPI_QUIET               | bool           | Configure suppression of reporting PAPI errors/warnings                                                                       |
| TIMEMORY_PAPI_EVENTS              | string         | PAPI presets and events to collect (see also: papi_avail)                                                                     |
| TIMEMORY_PAPI_ATTACH              | bool           | Configure PAPI to attach to another process (see also: TIMEMORY_TARGET_PID)                                                   |
| TIMEMORY_PAPI_OVERFLOW            | int            | Value at which PAPI hw counters trigger an overflow callback                                                                  |
| TIMEMORY_CUDA_EVENT_BATCH_SIZE    | unsigned long  | Batch size for create cudaEvent_t in cuda_event components                                                                    |
| TIMEMORY_NVTX_MARKER_DEVICE_SYNC  | bool           | Use cudaDeviceSync when stopping NVTX marker (vs. cudaStreamSychronize)                                                       |
| TIMEMORY_CUPTI_ACTIVITY_LEVEL     | int            | Default group of kinds tracked via CUpti Activity API                                                                         |
| TIMEMORY_CUPTI_ACTIVITY_KINDS     | string         | Specific cupti activity kinds to track                                                                                        |
| TIMEMORY_CUPTI_EVENTS             | string         | Hardware counter event types to collect on NVIDIA GPUs                                                                        |
| TIMEMORY_CUPTI_METRICS            | string         | Hardware counter metric types to collect on NVIDIA GPUs                                                                       |
| TIMEMORY_CUPTI_DEVICE             | int            | Target device for CUPTI hw counter collection                                                                                 |
| TIMEMORY_ROOFLINE_MODE            | string         | Configure the roofline collection mode. Options: 'op' 'ai'.                                                                   |
| TIMEMORY_ROOFLINE_MODE_CPU        | string         | Configure the roofline collection mode for CPU specifically. Options: 'op' 'ai'                                               |
| TIMEMORY_ROOFLINE_MODE_GPU        | string         | Configure the roofline collection mode for GPU specifically. Options: 'op' 'ai'.                                              |
| TIMEMORY_ROOFLINE_EVENTS_CPU      | string         | Configure custom hw counters to add to the cpu roofline                                                                       |
| TIMEMORY_ROOFLINE_EVENTS_GPU      | string         | Configure custom hw counters to add to the gpu roofline                                                                       |
| TIMEMORY_ROOFLINE_TYPE_LABELS     | bool           | Configure roofline labels/descriptions/output-files encode the list of data types                                             |
| TIMEMORY_ROOFLINE_TYPE_LABELS_CPU | bool           | Configure labels, etc. for the roofline components for CPU (see also: TIMEMORY_ROOFLINE_TYPE_LABELS)                          |
| TIMEMORY_ROOFLINE_TYPE_LABELS_GPU | bool           | Configure labels, etc. for the roofline components for GPU (see also: TIMEMORY_ROOFLINE_TYPE_LABELS)                          |
| TIMEMORY_INSTRUCTION_ROOFLINE     | bool           | Configure the roofline to include the hw counters required for generating an instruction roofline                             |
| TIMEMORY_ERT_NUM_THREADS          | unsigned long  | Number of threads to use when running ERT                                                                                     |
| TIMEMORY_ERT_NUM_THREADS_CPU      | unsigned long  | Number of threads to use when running ERT on CPU                                                                              |
| TIMEMORY_ERT_NUM_THREADS_GPU      | unsigned long  | Number of threads which launch kernels when running ERT on the GPU                                                            |
| TIMEMORY_ERT_NUM_STREAMS          | unsigned long  | Number of streams to use when launching kernels in ERT on the GPU                                                             |
| TIMEMORY_ERT_GRID_SIZE            | unsigned long  | Configure the grid size (number of blocks) for ERT on GPU (0 == auto-compute)                                                 |
| TIMEMORY_ERT_BLOCK_SIZE           | unsigned long  | Configure the block size (number of threads per block) for ERT on GPU                                                         |
| TIMEMORY_ERT_ALIGNMENT            | unsigned long  | Configure the alignment (in bits) when running ERT on CPU (0 == 8 * sizeof(T))                                                |
| TIMEMORY_ERT_MIN_WORKING_SIZE     | unsigned long  | Configure the minimum working size when running ERT (0 == device specific)                                                    |
| TIMEMORY_ERT_MIN_WORKING_SIZE_CPU | unsigned long  | Configure the minimum working size when running ERT on CPU                                                                    |
| TIMEMORY_ERT_MIN_WORKING_SIZE_GPU | unsigned long  | Configure the minimum working size when running ERT on GPU                                                                    |
| TIMEMORY_ERT_MAX_DATA_SIZE        | unsigned long  | Configure the max data size when running ERT on CPU                                                                           |
| TIMEMORY_ERT_MAX_DATA_SIZE_CPU    | unsigned long  | Configure the max data size when running ERT on CPU                                                                           |
| TIMEMORY_ERT_MAX_DATA_SIZE_GPU    | unsigned long  | Configure the max data size when running ERT on GPU                                                                           |
| TIMEMORY_ERT_SKIP_OPS             | string         | Skip these number of ops (i.e. ERT_FLOPS) when were set at compile time                                                       |
| TIMEMORY_ALLOW_SIGNAL_HANDLER     | bool           | Allow signal handling to be activated                                                                                         |
| TIMEMORY_ENABLE_SIGNAL_HANDLER    | bool           | Enable signals in timemory_init                                                                                               |
| TIMEMORY_ENABLE_ALL_SIGNALS       | bool           | Enable catching all signals                                                                                                   |
| TIMEMORY_DISABLE_ALL_SIGNALS      | bool           | Disable catching any signals                                                                                                  |
| TIMEMORY_NODE_COUNT               | int            | Total number of nodes used in application                                                                                     |
| TIMEMORY_DESTRUCTOR_REPORT        | bool           | Configure default setting for auto_{list,tuple,hybrid} to write to stdout during destruction of the bundle                    |
| TIMEMORY_PYTHON_EXE               | string         | Configure the python executable to use                                                                                        |
| TIMEMORY_UPCXX_INIT               | bool           | Enable/disable timemory calling upcxx::init() during certain timemory_init(...) invocations                                   |
| TIMEMORY_UPCXX_FINALIZE           | bool           | Enable/disable timemory calling upcxx::finalize() during timemory_finalize()                                                  |

## Available Hardware Counters

`timemory-avail -H` provides the list of available hardware counters for the CPU and, potentially, the GPU.

> Sample output from `timemory-avail -H --description --brief`

| HARDWARE COUNTER | DESCRIPTION                                                                             |
| ---------------- | --------------------------------------------------------------------------------------- |
| PAPI_L1_DCM      | Level 1 data cache misses                                                               |
| PAPI_L1_ICM      | Level 1 instruction cache misses                                                        |
| PAPI_L2_DCM      | Level 2 data cache misses                                                               |
| PAPI_L2_ICM      | Level 2 instruction cache misses                                                        |
| PAPI_L3_DCM      | Level 3 data cache misses                                                               |
| PAPI_L3_ICM      | Level 3 instruction cache misses                                                        |
| PAPI_L1_TCM      | Level 1 cache misses                                                                    |
| PAPI_L2_TCM      | Level 2 cache misses                                                                    |
| PAPI_L3_TCM      | Level 3 cache misses                                                                    |
| PAPI_CA_SNP      | Requests for a snoop                                                                    |
| PAPI_CA_SHR      | Requests for exclusive access to shared cache line                                      |
| PAPI_CA_CLN      | Requests for exclusive access to clean cache line                                       |
| PAPI_CA_INV      | Requests for cache line invalidation                                                    |
| PAPI_CA_ITV      | Requests for cache line intervention                                                    |
| PAPI_L3_LDM      | Level 3 load misses                                                                     |
| PAPI_L3_STM      | Level 3 store misses                                                                    |
| PAPI_BRU_IDL     | Cycles branch units are idle                                                            |
| PAPI_FXU_IDL     | Cycles integer units are idle                                                           |
| PAPI_FPU_IDL     | Cycles floating point units are idle                                                    |
| PAPI_LSU_IDL     | Cycles load/store units are idle                                                        |
| PAPI_TLB_DM      | Data translation lookaside buffer misses                                                |
| PAPI_TLB_IM      | Instruction translation lookaside buffer misses                                         |
| PAPI_TLB_TL      | Total translation lookaside buffer misses                                               |
| PAPI_L1_LDM      | Level 1 load misses                                                                     |
| PAPI_L1_STM      | Level 1 store misses                                                                    |
| PAPI_L2_LDM      | Level 2 load misses                                                                     |
| PAPI_L2_STM      | Level 2 store misses                                                                    |
| PAPI_BTAC_M      | Branch target address cache misses                                                      |
| PAPI_PRF_DM      | Data prefetch cache misses                                                              |
| PAPI_L3_DCH      | Level 3 data cache hits                                                                 |
| PAPI_TLB_SD      | Translation lookaside buffer shootdowns                                                 |
| PAPI_CSR_FAL     | Failed store conditional instructions                                                   |
| PAPI_CSR_SUC     | Successful store conditional instructions                                               |
| PAPI_CSR_TOT     | Total store conditional instructions                                                    |
| PAPI_MEM_SCY     | Cycles Stalled Waiting for memory accesses                                              |
| PAPI_MEM_RCY     | Cycles Stalled Waiting for memory Reads                                                 |
| PAPI_MEM_WCY     | Cycles Stalled Waiting for memory writes                                                |
| PAPI_STL_ICY     | Cycles with no instruction issue                                                        |
| PAPI_FUL_ICY     | Cycles with maximum instruction issue                                                   |
| PAPI_STL_CCY     | Cycles with no instructions completed                                                   |
| PAPI_FUL_CCY     | Cycles with maximum instructions completed                                              |
| PAPI_HW_INT      | Hardware interrupts                                                                     |
| PAPI_BR_UCN      | Unconditional branch instructions                                                       |
| PAPI_BR_CN       | Conditional branch instructions                                                         |
| PAPI_BR_TKN      | Conditional branch instructions taken                                                   |
| PAPI_BR_NTK      | Conditional branch instructions not taken                                               |
| PAPI_BR_MSP      | Conditional branch instructions mispredicted                                            |
| PAPI_BR_PRC      | Conditional branch instructions correctly predicted                                     |
| PAPI_FMA_INS     | FMA instructions completed                                                              |
| PAPI_TOT_IIS     | Instructions issued                                                                     |
| PAPI_TOT_INS     | Instructions completed                                                                  |
| PAPI_INT_INS     | Integer instructions                                                                    |
| PAPI_FP_INS      | Floating point instructions                                                             |
| PAPI_LD_INS      | Load instructions                                                                       |
| PAPI_SR_INS      | Store instructions                                                                      |
| PAPI_BR_INS      | Branch instructions                                                                     |
| PAPI_VEC_INS     | Vector/SIMD instructions (could include integer)                                        |
| PAPI_RES_STL     | Cycles stalled on any resource                                                          |
| PAPI_FP_STAL     | Cycles the FP unit(s) are stalled                                                       |
| PAPI_TOT_CYC     | Total cycles                                                                            |
| PAPI_LST_INS     | Load/store instructions completed                                                       |
| PAPI_SYC_INS     | Synchronization instructions completed                                                  |
| PAPI_L1_DCH      | Level 1 data cache hits                                                                 |
| PAPI_L2_DCH      | Level 2 data cache hits                                                                 |
| PAPI_L1_DCA      | Level 1 data cache accesses                                                             |
| PAPI_L2_DCA      | Level 2 data cache accesses                                                             |
| PAPI_L3_DCA      | Level 3 data cache accesses                                                             |
| PAPI_L1_DCR      | Level 1 data cache reads                                                                |
| PAPI_L2_DCR      | Level 2 data cache reads                                                                |
| PAPI_L3_DCR      | Level 3 data cache reads                                                                |
| PAPI_L1_DCW      | Level 1 data cache writes                                                               |
| PAPI_L2_DCW      | Level 2 data cache writes                                                               |
| PAPI_L3_DCW      | Level 3 data cache writes                                                               |
| PAPI_L1_ICH      | Level 1 instruction cache hits                                                          |
| PAPI_L2_ICH      | Level 2 instruction cache hits                                                          |
| PAPI_L3_ICH      | Level 3 instruction cache hits                                                          |
| PAPI_L1_ICA      | Level 1 instruction cache accesses                                                      |
| PAPI_L2_ICA      | Level 2 instruction cache accesses                                                      |
| PAPI_L3_ICA      | Level 3 instruction cache accesses                                                      |
| PAPI_L1_ICR      | Level 1 instruction cache reads                                                         |
| PAPI_L2_ICR      | Level 2 instruction cache reads                                                         |
| PAPI_L3_ICR      | Level 3 instruction cache reads                                                         |
| PAPI_L1_ICW      | Level 1 instruction cache writes                                                        |
| PAPI_L2_ICW      | Level 2 instruction cache writes                                                        |
| PAPI_L3_ICW      | Level 3 instruction cache writes                                                        |
| PAPI_L1_TCH      | Level 1 total cache hits                                                                |
| PAPI_L2_TCH      | Level 2 total cache hits                                                                |
| PAPI_L3_TCH      | Level 3 total cache hits                                                                |
| PAPI_L1_TCA      | Level 1 total cache accesses                                                            |
| PAPI_L2_TCA      | Level 2 total cache accesses                                                            |
| PAPI_L3_TCA      | Level 3 total cache accesses                                                            |
| PAPI_L1_TCR      | Level 1 total cache reads                                                               |
| PAPI_L2_TCR      | Level 2 total cache reads                                                               |
| PAPI_L3_TCR      | Level 3 total cache reads                                                               |
| PAPI_L1_TCW      | Level 1 total cache writes                                                              |
| PAPI_L2_TCW      | Level 2 total cache writes                                                              |
| PAPI_L3_TCW      | Level 3 total cache writes                                                              |
| PAPI_FML_INS     | Floating point multiply instructions                                                    |
| PAPI_FAD_INS     | Floating point add instructions                                                         |
| PAPI_FDV_INS     | Floating point divide instructions                                                      |
| PAPI_FSQ_INS     | Floating point square root instructions                                                 |
| PAPI_FNV_INS     | Floating point inverse instructions                                                     |
| PAPI_FP_OPS      | Floating point operations                                                               |
| PAPI_SP_OPS      | Floating point operations; optimized to count scaled single precision vector operations |
| PAPI_DP_OPS      | Floating point operations; optimized to count scaled double precision vector operations |
| PAPI_VEC_SP      | Single precision vector/SIMD instructions                                               |
| PAPI_VEC_DP      | Double precision vector/SIMD instructions                                               |
| PAPI_REF_CYC     | Reference clock cycles                                                                  |
