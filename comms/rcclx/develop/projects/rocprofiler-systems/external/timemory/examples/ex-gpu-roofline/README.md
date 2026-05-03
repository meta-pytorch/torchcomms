# ex-gpu-roofline

This example demonstrates the use of timemory 's roofline component to execute a amypx kernel wiht fp16, fp32 and fp64 data types on GPU and measure the theoretical peak and then plotting a customized roofline plot.

## Roofline and Roofline Component 

See [Roofline Components](../../docs/components/roofline.md)

## Build

See [examples](../README.md##Build). Further requires `-DTIMEMORY_USE_CUDA=ON` and `-DTIMEMORY_USE_CUPTI=ON` flags enabled in cmake as well as CUDA and CUPTI installed on the system for this example to build.