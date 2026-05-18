# Roofline Components

> Namespace: `tim::component`

Roofline is a visually intuitive performance model used to bound the performance of various numerical methods and operations running on multicore, manycore, or accelerator processor architectures.
Rather than simply using percent-of-peak estimates, the model can be used to assess the quality of attained performance by combining locality, bandwidth, and different parallelization paradigms
into a single performance figure.
One can examine the resultant Roofline figure in order to determine both the implementation and inherent performance limitations.

More information on roofline can be found [here](https://docs.nersc.gov/programming/performance-debugging-tools/roofline/).

| Component Name     | Category | Template Specification   | Dependencies | Description                                                     |
| ------------------ | -------- | ------------------------ | ------------ | --------------------------------------------------------------- |
| **`cpu_roofline`** | CPU      | `cpu_roofline<Types...>` | PAPI         | Records the rate at which the hardware counters are accumulated |
| **`gpu_roofline`** | CPU      | `gpu_roofline<Types...>` | CUDA, CUPTI  | Records the rate at which the hardware counters are accumulated |

The roofline components provided by timemory execute a workflow during application termination that calculates the theoretical peak for the roofline.
A pre-defined set of algorithms for the theoretical peak are provided but these can be customized.
An example can be found in `timemory/examples/ex-cpu-roofline/test_cpu_roofline.cpp` and `timemory/examples/ex-gpu-roofline/test_gpu_roofline.cpp`.

## Pre-defined Types

> Namespace: `tim::component`

| Component Name              | Underlying Template Specification     | Description                                        |
| --------------------------- | ------------------------------------- | -------------------------------------------------- |
| **`cpu_roofline_flops`**    | `cpu_roofline<float, double>`         | Rate of single- and double-precision FLOP/s        |
| **`cpu_roofline_dp_flops`** | `cpu_roofline<double>`                | Rate of double-precision FLOP/s                    |
| **`cpu_roofline_sp_flops`** | `cpu_roofline<float>`                 | Rate of single-precision FLOP/s                    |
| **`gpu_roofline_flops`**    | `gpu_roofline<fp16_t, float, double>` | Rate of half-, single- and double-precision FLOP/s |
| **`gpu_roofline_dp_flops`** | `gpu_roofline<double>`                | Rate of double-precision FLOP/s                    |
| **`gpu_roofline_sp_flops`** | `gpu_roofline<float>`                 | Rate of single-precision FLOP/s                    |
| **`gpu_roofline_hp_flops`** | `gpu_roofline<fp16_t>`                | Rate of half-precision FLOP/s                      |
