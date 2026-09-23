# Intel GPU Component (intel_gpu):

PAPI `intel_gpu` component is for accessing Intel Graphics Processing Unit (GPU) hardware performance metrics through Intel oneAPI Level Zero Interface.

## Enable intel_gpu Component

To enable intel_gpu component, the PAPI library is built with configure option as

```sh
./configure --with-components="intel_gpu"
```

## Prerequisites 

* [oneAPI Level Zero loader (libze_loader.so)](https://github.com/oneapi-src/level-zero)
* [Intel(R) Metrics Discovery Application Programming Interface (libigdmd.so)](https://github.com/intel/metrics-discovery)
* [Intel(R) Metrics Library Application Programming Interface (libigdml.so)](https://github.com/intel/metrics-library)
* /proc/sys/dev/i915/perf_stream_paranoid is set to `0`
* User needs to be added into Linux render/video groups

## Environment Variables

PAPI requires the location of the Level Zero install directory. This can be
specified by one environment variable: `PAPI_INTEL_GPU_ROOT`.

Default installation location is /usr.
    
Access to this directory is required at both compile (for include files) and
at runtime (for libraries).

## Runtime Environment:

* To monitor performance events, libze_loader.so, libigdmd.so, and libigdml.so must be found.
  If these shared object files are not located in one of the Linux default directories listed
  under /etc/ld.so.conf.d/ (usually /usr/lib64, /lib64, /usr/lib and /lib), then the user must
  specify where they are located via the following command.
    ```sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-libze_loader>:<path-to-libigdmd>:<path-to-libigdml>
    ```
* To enable metrics
    ```sh
    ZET_ENABLE_METRICS=1
    ```
  Note that the component sets this automatically.
* To change the sampling period from default 400000
    ```sh
    METRICS_SAMPLING_PERIOD=value
    ```

## Counter Collection Modes

Two metrics collection modes are supported.
* Time-based sampling. In this mode, data collection and app can run in separate processes. To enable time-based event monitoring:
    ```sh
    ZE_ENABLE_TRACING_LAYER=0
    ```
* Query-based sampling. In this mode, PAPI_start() must be called before kernel launch and PAPI_stop() must be called after kernel execution completes. To enable query-based event monitoring:
    ```sh
    ZE_ENABLE_TRACING_LAYER=1
    ```

## Metrics:

Use `test/gpu_metrc_list` or `papi_native_avail` to find metrics name for Intel GPU.
* Metrics are named as: intel_gpu:::<group_name>.<metric_name>
* Metrics can be added with qualifier for a select device or subdevice

       intel_gpu:::<group_name>.<metric_name>:device=<n>:tile=<m>
       n: device id, start from 0.
       m: subdevice id, start from 0.

See tests/readme.txt for how to use it.
