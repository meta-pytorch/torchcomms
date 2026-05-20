# Intel GPU Component: (intel_gpu)

PAPI "intel_gpu" component is for accessing Intel Graphics Processing Unit (GPU) hardware performance metrics through Intel oneAPI Level Zero Interface.

## Enable intel_gpu component

To enable intel_gpu component, the PAPI library is built with configure option as 

	./configure --with-components="intel_gpu"

It is requred to build with Intel oneAPI Level Zero header files. The directory of Level0 header files can be defined using INTEL_L0_HEADERS. Default installation location is /usr/include/level_zero/.


## Prerequisites 

* [oneAPI Level Zero loader (libze_loader.so)](https://github.com/oneapi-src/level-zero)
* [Intel(R) Metrics Discovery Application Programming Interface](https://github.com/intel/metrics-discovery)
* /proc/sys/dev/i915/perf_stream_paranoid is set to "0"
* User needs to be added into Linux render/video groups

## Runtime environment:

*  ```sh
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-libze_loader>:<path-to-libmd>
   ```
* To enable metrics
	```sh
	ZET_ENABLE_METRICS=1
	```
* To change the sampling period from default 400000, 
	```sh
    METRICS_SAMPLING_PERIOD=value
	```

* To enable metrics query on an kernel
	```sh 
    ZET_ENABLE_API_TRACING_EXP=1
	```

## Metric collection mode:

Two metrics collection modes are supported.

* Time based sampling. In this mode, data collection and app can run in separate processes. 
* Metrics query on a kernel. In this mode,  the PAPI_start() and PAPI_stop must be called before kernel launch and after kernel execution completes. When setting ZET_ENABLE_API_TRACING_EXP=1,  the collection will switch to metrics query mode.

## Metrics:

Use "test/gpu_metrc_list" or "papi_native_avail" to find metrics name for Intel GPU.
* Metrics are named as: intel_gpu:::<group_name>.<metric_name>
* Metrics can be added with qulifier for a select device or subdevice

       intel_gpu:::<group_name>.<metric_name>:device=<n>:tile=<m>
       n: device id, start from 0.
	   m: subdevice id, start from 0.

See test/readme.txt for how to use it.
