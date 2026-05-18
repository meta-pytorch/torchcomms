// Copyright 2021 NVIDIA Corporation. All rights reserved
//
// This sample demonstrates two ways to use the CUPTI Profiler API with concurrent kernels.
// By taking the ratio of runtimes for a consecutive series of kernels, compared
// to a series of concurrent kernels, one can difinitively demonstrate that concurrent
// kernels were running while metrics were gathered and the User Replay mechanism was in use.
//
// Example:
// 4 kernel launches, with 1x, 2x, 3x, and 4x amounts of work, each sized to one SM (one warp
// of threads, one thread block).
// When run synchronously, this comes to 10x amount of work.
// When run concurrently, the longest (4x) kernel should be the only measured time (it hides the others).
// Thus w/ 4 kernels, the concurrent : consecutive time ratio should be 4:10.
// On test hardware this does simplify to 3.998:10.  As the test is affected by memory layout, this may not
// hold for certain architectures where, for example, cache sizes may optimize certain kernel calls.
//
// After demonstrating concurrency using multpile streams, this then demonstrates using multiple devices.
// In this 3rd configuration, the same concurrent workload with streams is then duplicated and run
// on each device concurrently using streams.
// In this case, the wallclock time to launch, run, and join the threads should be roughly the same as the
// wallclock time to run the single device case.  If concurrency was not working, the wallcock time
// would be (num devices) times the single device concurrent case.
//
//  * If the multiple devices have different performance, the runtime may be significantly different between
//    devices, but this does not mean concurrent profiling is not happening.

// This code has been adapted to PAPI from 
// `<CUDA-TOOLKIT-11.4>/extras/CUPTI/samples/concurrent_profiling/cpncurrent_profiling.cu`

#ifdef PAPI
extern "C" {
    #include <papi.h>
    #include "papi_test.h"
}
#endif

// Standard CUDA, CUPTI, Profiler, NVPW headers
#include "cuda.h"

// Standard STL headers
#include <chrono>
#include <cstdint>
#include <cstdio>

#include <string>
using ::std::string;

#include <thread>
using ::std::thread;

#include <vector>
using ::std::vector;

#define PRINT(quiet, format, args...) {if (!quiet) {fprintf(stderr, format, ## args);}}
int quiet;

#ifdef PAPI
#define PAPI_CALL(apiFuncCall)                                          \
do {                                                                           \
    int _status = apiFuncCall;                                         \
    if (_status != PAPI_OK) {                                              \
        fprintf(stderr, "error: function %s failed.", #apiFuncCall);  \
        test_fail(__FILE__, __LINE__, "", _status);  \
    }                                                                          \
} while (0)
#endif

// Helpful error handlers for standard CUPTI and CUDA runtime calls
#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define MEMORY_ALLOCATION_CALL(var)                                             \
do {                                                                            \
    if (var == NULL) {                                                          \
        fprintf(stderr, "%s:%d: Error: Memory Allocation Failed \n",            \
                __FILE__, __LINE__);                                            \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

typedef struct
{
    int device;                //!< compute device number
} profilingConfig;

// Per-device configuration, buffers, stream and device information, and device pointers
typedef struct
{
    int deviceID;
    profilingConfig config;                 // Each device (or each context) needs its own CUPTI profiling config
    vector<cudaStream_t> streams;           // Each device needs its own streams
    vector<double *> d_x;                   // And device memory allocation
    vector<double *> d_y;                   // ..
    long long values[100];                  // Capture PAPI measured values for each device
} perDeviceData;

#define DAXPY_REPEAT 32768
// Loop over array of elements performing daxpy multiple times
// To be launched with only one block (artificially increasing serial time to better demonstrate overlapping replay)
__global__ void daxpyKernel(int elements, double a, double * x, double * y)
{
    for (int i = threadIdx.x; i < elements; i += blockDim.x)
        // Artificially increase kernel runtime to emphasize concurrency
        for (int j = 0; j < DAXPY_REPEAT; j++)
            y[i] = a * x[i] + y[i]; // daxpy
}

// Initialize kernel values
double a = 2.5;

// Normally you would want multiple warps, but to emphasize concurrency with streams and multiple devices
// we run the kernels on a single warp.
int threadsPerBlock = 32;
int threadBlocks = 1;

// Configurable number of kernels (streams, when running concurrently)
int const numKernels = 4;
int const numStreams = numKernels;
vector<size_t> elements(numKernels);

// Each kernel call allocates and computes (call number) * (blockSize) elements
// For 4 calls, this is 4k elements * 2 arrays * (1 + 2 + 3 + 4 stream mul) * 8B/elem =~ 640KB
int const blockSize = 4 * 1024;

// Wrapper which will launch numKernel kernel calls on a single device
// The device streams vector is used to control which stream each call is made on
// If 'serial' is non-zero, the device streams are ignored and instead the default stream is used
void profileKernels(perDeviceData &d,
                    vector<string> const &metricNames,
                    char const * const rangeName, bool serial)
{
    RUNTIME_API_CALL(cudaSetDevice(d.config.device));  // Orig code has mistake here
#ifdef PAPI
    int eventset = PAPI_NULL, i, papi_errno;
    PAPI_CALL(PAPI_create_eventset(&eventset));
    // Switch to desired device
    string evt_name;
    for (i = 0; i < metricNames.size(); i++) {
        evt_name = metricNames[i] + std::to_string(d.config.device);
        PRINT(quiet, "Adding event name: %s\n", evt_name.c_str());
        papi_errno = PAPI_add_named_event(eventset, evt_name.c_str());
        if (papi_errno != PAPI_OK) {
            fprintf(stderr, "Failed to add event %s\n", evt_name.c_str());
            test_skip(__FILE__, __LINE__, "", 0);
        }
    }
    PAPI_CALL(PAPI_start(eventset));
#endif
    for (unsigned int stream = 0; stream < d.streams.size(); stream++)
    {
        cudaStream_t streamId = (serial ? 0 : d.streams[stream]);
        daxpyKernel <<<threadBlocks, threadsPerBlock, 0, streamId>>> (elements[stream], a, d.d_x[stream], d.d_y[stream]);
    }

    // After launching all work, synchronize all streams
    if (serial == false)
    {
        for (unsigned int stream = 0; stream < d.streams.size(); stream++)
        {
            RUNTIME_API_CALL(cudaStreamSynchronize(d.streams[stream]));
        }
    }
    else
    {
        RUNTIME_API_CALL(cudaStreamSynchronize(0));
    }
#ifdef PAPI
    PAPI_CALL(PAPI_stop(eventset, d.values));
    PAPI_CALL(PAPI_cleanup_eventset(eventset));
    PAPI_CALL(PAPI_destroy_eventset(&eventset));
#endif
}

void print_measured_values(perDeviceData &d, vector<string> const &metricNames)
{
    string evt_name;
    PRINT(quiet, "PAPI event name\t\t\t\t\t\t\tMeasured value\n");
    PRINT(quiet, "%s\n", std::string(80, '-').c_str());
    for (int i=0; i < metricNames.size(); i++) {
        evt_name = metricNames[i] + std::to_string(d.config.device);
        PRINT(quiet, "%s\t\t\t%ld\n", evt_name.c_str(), d.values[i]);
    }
}

int main(int argc, char **argv)
{
    quiet = 0;
    int i;
#ifdef PAPI
    char *test_quiet = getenv("PAPI_CUDA_TEST_QUIET");
    if (test_quiet)
        quiet = (int) strtol(test_quiet, (char**) NULL, 10);

    int event_count = argc - 1;
    /* if no events passed at command line, just report test skipped. */
    if (event_count == 0) {
        fprintf(stderr, "No eventnames specified at command line.\n");
        test_skip(__FILE__, __LINE__, "", 0);
    }

    vector<string> metricNames;
    for (i=0; i < event_count; i++) {
        metricNames.push_back(argv[i+1]);
    }

    // Initialize the PAPI library
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init failed.", 0);
    }
#else
    vector<string> metricNames = {""};
#endif

    int numDevices;
    RUNTIME_API_CALL(cudaGetDeviceCount(&numDevices));

    // Per-device information
    vector<int> device_ids;

    // Find all devices capable of running CUPTI Profiling (Compute Capability >= 7.0)
    for (i = 0; i < numDevices; i++)
    {
        // Get device properties
        int major;
        RUNTIME_API_CALL(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i));
        if (major >= 7)
        {
            // Record device number
            device_ids.push_back(i);
        }
    }

    numDevices = device_ids.size();
    PRINT(quiet, "Found %d compatible devices\n", numDevices);

    // Ensure we found at least one device
    if (numDevices == 0)
    {
        fprintf(stderr, "No devices detected compatible with CUPTI Profiling (Compute Capability >= 7.0)\n");
#ifdef PAPI
        test_skip(__FILE__, __LINE__, "", 0);
#endif
    }

    // Initialize kernel input to some known numbers
    vector<double> h_x(blockSize * numKernels);
    vector<double> h_y(blockSize * numKernels);
    for (size_t i = 0; i < blockSize * numKernels; i++)
    {
        h_x[i] = 1.5 * i;
        h_y[i] = 2.0 * (i - 3000);
    }

    // Initialize a vector of 'default stream' values to demonstrate serialized kernels
    vector<cudaStream_t> defaultStreams(numStreams);
    for (int stream = 0; stream < numStreams; stream++)
    {
        defaultStreams[stream] = 0;
    }

    // Scale per-kernel work by stream number
    for (int stream = 0; stream < numStreams; stream++)
    {
        elements[stream] = blockSize * (stream + 1);
    }

    // For each device, configure profiling, set up buffers, copy kernel data
    vector<perDeviceData> deviceData(numDevices);

    for (int device = 0; device < numDevices; device++)
    {
        int device_id = device_ids[device];
        RUNTIME_API_CALL(cudaSetDevice(device_id));
        PRINT(quiet, "Configuring device %d\n", device_id);
        deviceData[device].deviceID = device_id;

        // Required CUPTI Profiling configuration & initialization
        // Can be done ahead of time or immediately before startSession() call
        // Initialization & configuration images can be generated separately, then passed to later calls
        // For simplicity's sake, in this sample, a single config struct is created per device and passed to each CUPTI Profiler API call
        // For more complex cases, each combination of CUPTI Profiler Session and Config requires additional initialization
        profilingConfig config;
        config.device = device_id;         // Device ID, used to get device name for metrics enumeration
        // config.maxLaunchesPerPass = 1;     // Must be >= maxRangesPerPass.  Set this to the largest count of kernel launches which may be encountered in any Pass in this Session

        // // Device 0 has max of 3 passes; other devices only run one pass in this sample code
        deviceData[device].config = config;// Save this device config

        // Initialize CUPTI Profiling structures
        // Per-stream initialization & memory allocation - copy from constant host array to each device array
        deviceData[device].streams.resize(numStreams);
        deviceData[device].d_x.resize(numStreams);
        deviceData[device].d_y.resize(numStreams);
        for (int stream = 0; stream < numStreams; stream++)
        {
            RUNTIME_API_CALL(cudaStreamCreate(&(deviceData[device].streams[stream])));

            // Each kernel does (stream #) * blockSize work on doubles
            size_t size = elements[stream] * sizeof(double);

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_x[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_x[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_x[stream], h_x.data(), size, cudaMemcpyHostToDevice));

            RUNTIME_API_CALL(cudaMalloc(&(deviceData[device].d_y[stream]), size));
            MEMORY_ALLOCATION_CALL(deviceData[device].d_y[stream]); // Validate pointer
            RUNTIME_API_CALL(cudaMemcpy(deviceData[device].d_y[stream], h_x.data(), size, cudaMemcpyHostToDevice));
        }
    }

    //
    // First version - single device, kernel calls serialized on default stream
    //

    // Use wallclock time to measure performance
    auto begin_time = ::std::chrono::high_resolution_clock::now();

    // Run on first device and use default streams, which run serially
    profileKernels(deviceData[0], metricNames, "single_device_serial", true);

    auto end_time = ::std::chrono::high_resolution_clock::now();
    int elapsed_serial_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time).count();
    int numBlocks = 0;
    for (int i = 1; i <= numKernels; i++)
    {
        numBlocks += i;
    }
    PRINT(quiet, "It took %d ms on the host to profile %d  kernels in serial.", elapsed_serial_ms, numKernels);

    //
    // Second version - same kernel calls as before on the same device, but now using separate streams for concurrency
    // (Should be limited by the longest running kernel)
    //

    begin_time = ::std::chrono::high_resolution_clock::now();

    // Still only use first device, but this time use its allocated streams for parallelism
    profileKernels(deviceData[0], metricNames, "single_device_async", false);

    end_time = ::std::chrono::high_resolution_clock::now();
    int elapsed_single_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time).count();
    PRINT(quiet, "It took %d ms on the host to profile %d  kernels on a single device on separate streams.", elapsed_single_device_ms, numKernels);
    PRINT(quiet, "--> If the separate stream wallclock time is less than the serial version, the streams were profiling concurrently.\n");

    //
    // Third version - same as the second case, but duplicates the concurrent work across devices to show cross-device concurrency
    // This is done using devices so no serialization is needed between devices
    // (Should have roughly the same wallclock time as second case if the devices have similar performance)
    //

    if (numDevices == 1)
    {
        PRINT(quiet, "Only one compatible device found; skipping the multi-threaded test.\n");
    }
    else
    {
#ifdef PAPI
        int papi_errno = PAPI_thread_init((unsigned long (*)(void)) std::this_thread::get_id);
        if ( papi_errno != PAPI_OK ) {
            test_fail(__FILE__, __LINE__, "Error setting thread id function.\n", papi_errno);
        }
#endif
        PRINT(quiet, "Running on %d devices, one thread per device.\n", numDevices);

        // Time creation of the same multiple streams (on multiple devices, if possible)
        vector<::std::thread> threads;
        begin_time = ::std::chrono::high_resolution_clock::now();

        // Now launch parallel thread work, duplicated on one thread per device
        for (int thread = 0; thread < numDevices; thread++)
        {
            threads.push_back(::std::thread(profileKernels, ::std::ref(deviceData[thread]), metricNames, "multi_device_async", false));
        }

        // Wait for all threads to finish
        for (auto &t: threads)
        {
            t.join();
        }

        // Record time used when launching on multiple devices
        end_time = ::std::chrono::high_resolution_clock::now();
        int elapsed_multiple_device_ms = ::std::chrono::duration_cast<::std::chrono::milliseconds>(end_time - begin_time).count();
        double ratio = elapsed_multiple_device_ms / (double) elapsed_single_device_ms;
        PRINT(quiet, "It took %d ms on the host to profile the same %d kernels on each of the %d devices in parallel\n", elapsed_multiple_device_ms, numKernels, numDevices);
        PRINT(quiet, "--> Wallclock ratio of parallel device launch to single device launch is %f\n", ratio);
        PRINT(quiet, "--> If the ratio is close to 1, that means there was little overhead to profile in parallel on multiple devices compared to profiling on a single device.\n");
        PRINT(quiet, "--> If the devices have different performance, the ratio may not be close to one, and this should be limited by the slowest device.\n");
    }

    // Free stream memory for each device
    for (int i = 0; i < numDevices; i++)
    {
        for (int j = 0; j < numKernels; j++)
        {
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_x[j]));
            RUNTIME_API_CALL(cudaFree(deviceData[i].d_y[j]));
        }
    }

#ifdef PAPI
    // Display metric values
    PRINT(quiet, "\nMetrics for device #0:\n");
    PRINT(quiet, "Look at the sm__cycles_elapsed.max values for each test.\n");
    PRINT(quiet, "This value represents the time spent on device to run the kernels in each case, and should be longest for the serial range, and roughly equal for the single and multi device concurrent ranges.\n");
    print_measured_values(deviceData[0], metricNames);

    // Only display next device info if needed
    if (numDevices > 1)
    {
        PRINT(quiet, "\nMetrics for the remaining devices only display the multi device async case and should all be similar to the first device's values if the device has similar performance characteristics.\n");
        PRINT(quiet, "If devices have different performance characteristics, the runtime cycles calculation may vary by device.\n");
    }
    for (int i = 1; i < numDevices; i++)
    {
        PRINT(quiet, "\nMetrics for device #%d:\n", i);
        print_measured_values(deviceData[i], metricNames);
    }
    PAPI_shutdown();
    test_pass(__FILE__);
#endif
    return 0;
}
