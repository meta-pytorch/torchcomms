/**
 * @file    amdsmi_gemm.c
 * @author  Dong Jun Woun
 *          djwoun@gmail.com
 * @brief   Launches a large HIP DGEMM workload (on device 1) while sampling a
 *          small set of AMD SMI counters via PAPI.
 *
 * The monitor thread polls the PAPI EventSet ~3 times per second while the kernel runs.
 * This is intended for simple integration/soak testing rather than performance tuning.
 *
 * If no events are specified, we enumerate the AMD-SMI component and add the
 * first five usable native events (skipping ones that cannot be added).
 */

#include "test_harness.h"

#include "papi.h"
#include "hip/hip_runtime.h"
#include <rocblas.h>
#include <errno.h>
#include <limits.h>
#include <stdbool.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

/* ----------------------------- Configuration ----------------------------- */

#define DEFAULT_M_DIM 7296
#define DEFAULT_K_DIM 14592
#define DEFAULT_N_DIM 7296

#define NUM_STREAMS 1
#define ITERATIONS_PER_STREAM 1
#define DEFAULT_EVENT_COUNT 5
#define MAX_EVENT_SLOTS 8

#if MAX_EVENT_SLOTS < 7
#error "MAX_EVENT_SLOTS must accommodate all test-mode events."
#endif

static const char *const kTestEventTemplates[] = {
    "amd_smi:::temp_current_sensor=1:device=%d",
    "amd_smi:::temp_current_sensor=2:device=%d",
    "amd_smi:::temp_current_sensor=7:device=%d",
    "amd_smi:::gfx_activity:device=%d",
    "amd_smi:::umc_activity:device=%d",
    "amd_smi:::power_current:device=%d",
    "amd_smi:::power_average:device=%d",
    "amd_smi:::process_cu_occupancy_proc=0:device=%d",
};

static const int kTestEventCount =
    (int)(sizeof(kTestEventTemplates) / sizeof(kTestEventTemplates[0]));

struct run_config {
    long m_dim;
    long k_dim;
    long n_dim;
    int event_count;
    int iterations;
    useconds_t iteration_delay_us;
    bool test_mode;
    bool use_rocblas;
    int device_index;
    char csv_path[PATH_MAX];
};

static struct run_config g_run_config = {
    .m_dim = DEFAULT_M_DIM,
    .k_dim = DEFAULT_K_DIM,
    .n_dim = DEFAULT_N_DIM,
    .event_count = DEFAULT_EVENT_COUNT,
    .iterations = ITERATIONS_PER_STREAM,
    .iteration_delay_us = 3000000, /* default ~3s between iterations */
    .test_mode = false,
    .use_rocblas = false,
    .device_index = 0,
    .csv_path = {0},
};

static int parse_test_override(int *argc, char **argv);

struct sample_entry {
    double elapsed_sec;
    long long values[MAX_EVENT_SLOTS];
};

struct sample_buffer {
    struct sample_entry *entries;
    size_t count;
    size_t capacity;
    int push_failed;
    pthread_mutex_t lock;
};

static int sample_buffer_init(struct sample_buffer *buffer) {
    if (!buffer) return -1;
    buffer->entries = NULL;
    buffer->count = 0;
    buffer->capacity = 0;
    buffer->push_failed = 0;
    return pthread_mutex_init(&buffer->lock, NULL);
}

static void sample_buffer_destroy(struct sample_buffer *buffer) {
    if (!buffer) return;
    free(buffer->entries);
    buffer->entries = NULL;
    buffer->count = 0;
    buffer->capacity = 0;
    pthread_mutex_destroy(&buffer->lock);
}

static int sample_buffer_push(struct sample_buffer *buffer,
                              double elapsed_sec,
                              const long long *values,
                              int event_count) {
    if (!buffer || !values || event_count <= 0 || event_count > MAX_EVENT_SLOTS) {
        return -1;
    }

    if (pthread_mutex_lock(&buffer->lock) != 0) {
        return -1;
    }

    if (buffer->count == buffer->capacity) {
        size_t new_capacity = buffer->capacity ? buffer->capacity * 2 : 64;
        struct sample_entry *new_entries = (struct sample_entry *)realloc(
            buffer->entries, new_capacity * sizeof(struct sample_entry));
        if (!new_entries) {
            buffer->push_failed = 1;
            pthread_mutex_unlock(&buffer->lock);
            return -1;
        }
        buffer->entries = new_entries;
        buffer->capacity = new_capacity;
    }

    struct sample_entry *entry = &buffer->entries[buffer->count++];
    entry->elapsed_sec = elapsed_sec;
    memset(entry->values, 0, sizeof(entry->values));
    memcpy(entry->values, values, event_count * sizeof(long long));

    pthread_mutex_unlock(&buffer->lock);
    return 0;
}

static int write_results_csv(const char *path,
                             const char names[][PAPI_MAX_STR_LEN],
                             int event_count,
                             const struct sample_buffer *samples) {
    if (!path || !names || event_count <= 0 || !samples) {
        return -1;
    }

    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open %s for writing.\n", path);
        return -1;
    }

    fprintf(fp, "timestamp");
    for (int i = 0; i < event_count; ++i) {
        fprintf(fp, ",\"%s\"", names[i]);
    }
    fprintf(fp, "\n");

    for (size_t i = 0; i < samples->count; ++i) {
        const struct sample_entry *entry = &samples->entries[i];
        fprintf(fp, "%.6f", entry->elapsed_sec);
        for (int j = 0; j < event_count; ++j) {
            fprintf(fp, ",%lld", entry->values[j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 0;
}

static int parse_test_override(int *argc, char **argv) {
    if (!argc || !argv) {
        return -1;
    }

    for (int i = 1; i < *argc; ++i) {
        bool request_rocblas = false;
        if (strcmp(argv[i], "--test") == 0) {
            request_rocblas = false;
        } else if (strcmp(argv[i], "--testrocblas") == 0) {
            request_rocblas = true;
        } else {
            continue;
        }

        if (g_run_config.test_mode) {
            fprintf(stderr, "Duplicate --test/--testrocblas option is not supported.\n");
            return -1;
        }

        if (i + 3 >= *argc) {
            fprintf(stderr, "%s requires three integer arguments for M, K, and N.\n",
                    request_rocblas ? "--testrocblas" : "--test");
            return -1;
        }

        long dims[3] = {0};
        for (int d = 0; d < 3; ++d) {
            const char *arg = argv[i + 1 + d];
            errno = 0;
            char *endptr = NULL;
            long value = strtol(arg, &endptr, 10);
            if (endptr == arg || !endptr || *endptr != '\0') {
                fprintf(stderr, "Invalid integer value for %s argument: %s\n",
                        request_rocblas ? "--testrocblas" : "--test", arg);
                return -1;
            }
            if (errno == ERANGE) {
                fprintf(stderr, "Dimension for %s is out of range: %s\n",
                        request_rocblas ? "--testrocblas" : "--test", arg);
                return -1;
            }
            if (value <= 0) {
                fprintf(stderr, "Dimension for %s must be a positive integer: %s\n",
                        request_rocblas ? "--testrocblas" : "--test", arg);
                return -1;
            }
            dims[d] = value;
        }

        if (request_rocblas) {
            for (int d = 0; d < 3; ++d) {
                if (dims[d] > INT_MAX) {
                    fprintf(stderr, "rocBLAS test requires dimensions to be <= %d: %s\n",
                            INT_MAX, argv[i + 1 + d]);
                    return -1;
                }
            }
        }

        long m = dims[0];
        long k = dims[1];
        long n = dims[2];
        if ((m > LONG_MAX / k) || (n > LONG_MAX / k) || (m > LONG_MAX / n)) {
            fprintf(stderr, "Input dimensions are too large; ensure M*K, N*K, and M*N are <= %ld.\n",
                    LONG_MAX);
            return -1;
        }

        g_run_config.m_dim = m;
        g_run_config.k_dim = k;
        g_run_config.n_dim = n;
        g_run_config.event_count = kTestEventCount;
        g_run_config.iterations = ITERATIONS_PER_STREAM;
        g_run_config.iteration_delay_us = 5000000; /* 5 second pause between iterations */
        g_run_config.test_mode = true;
        g_run_config.use_rocblas = request_rocblas;
        g_run_config.device_index = 0;
        snprintf(g_run_config.csv_path, sizeof(g_run_config.csv_path),
                 "amdsmi_gemm_%ld_%ld_%ld%s.csv",
                 g_run_config.m_dim,
                 g_run_config.k_dim,
                 g_run_config.n_dim,
                 g_run_config.use_rocblas ? "_rocblas" : "");

        int args_to_remove = 4;
        if (i + 4 < *argc) {
            char *iter_end = NULL;
            errno = 0;
            long iter_val = strtol(argv[i + 4], &iter_end, 10);
            if (errno == 0 && iter_end && *iter_end == '\0') {
                if (iter_val <= 0 || iter_val > INT_MAX) {
                    fprintf(stderr, "Iteration count for %s must be between 1 and %d: %s\n",
                            request_rocblas ? "--testrocblas" : "--test", INT_MAX, argv[i + 4]);
                    return -1;
                }
                g_run_config.iterations = (int)iter_val;
                args_to_remove = 5;
            }
        }

        /* Remove the --test arguments from argv so the harness does not see them. */
        for (int j = i + args_to_remove; j < *argc; ++j) {
            argv[j - args_to_remove] = argv[j];
        }
        *argc -= args_to_remove;
        argv[*argc] = NULL;
        break;
    }

    return 0;
}

/* --------------------------- HIP error helpers --------------------------- */

#define HIP_CHECK(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "Failed: HIP error %s:%d '%s' (code: %d)\n", \
                __FILE__, __LINE__, hipGetErrorString(e), e); \
        return 1; \
    } \
} while(0)

#define HIP_CHECK_CLEANUP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        fprintf(stderr, "Warning: HIP cleanup error %s:%d '%s' (code: %d)\n", \
                __FILE__, __LINE__, hipGetErrorString(e), e); \
    } \
} while(0)

/* --------------------------- Monitoring thread --------------------------- */

/**
 * @brief Background poller for PAPI EventSet values.
 *
 * If params->print is 1, it writes one line per sample to stdout with a timestamp.
 */
static volatile int stop_monitor = 0;

struct monitor_params {
    int EventSet;
    struct timeval start_time;
    int print; // 0/1: whether to print readings (controls stdout chatter)
    int event_count;
    struct sample_buffer *samples;
};

static void *monitor_events(void *args) {
    struct monitor_params *params = (struct monitor_params *)args;
    int statusFlag;
    long long values[MAX_EVENT_SLOTS] = {0};

    while (!stop_monitor) {
        statusFlag = PAPI_read(params->EventSet, values);
        if (statusFlag != PAPI_OK) {
            fprintf(stderr, "PAPI read failed in monitor: %s\n", PAPI_strerror(statusFlag));
            break;
        }

        struct timeval current_time;
        gettimeofday(&current_time, NULL);
        double elapsed = (current_time.tv_sec - params->start_time.tv_sec) +
                         (current_time.tv_usec - params->start_time.tv_usec) / 1e6;

        if (params->print) {
            fprintf(stdout, "Time: %.6f sec", elapsed);
            for (int i = 0; i < params->event_count; ++i) {
                fprintf(stdout, "%s e%d: %lld", (i == 0 ? " ->" : ","), i + 1, values[i]);
            }
            fprintf(stdout, "\n");
            fflush(stdout);
        }

        if (params->samples) {
            if (sample_buffer_push(params->samples, elapsed, values, params->event_count) != 0) {
                fprintf(stderr, "Failed to record monitoring sample.\n");
                break;
            }
        }

        usleep(100000); // ~10 Hz sampling cadence
    }
    return NULL;
}

/* ------------------------------- Workload -------------------------------- */

/**
 * @brief Naive DGEMM: C = alpha * A * B + beta * C
 *        A: MxK, B: KxN, C: MxN (row-major)
 */
__global__ void dgemm_kernel(const double *A, const double *B, double *C,
                             long M, long N, long K, double alpha, double beta) {
    long row = (long)blockIdx.y * blockDim.y + threadIdx.y;
    long col = (long)blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        long row_offset = row * K;
        for (long k = 0; k < K; ++k) {
            sum += A[row_offset + k] * B[k * N + col];
        }
        long c_index = row * N + col;
        C[c_index] = alpha * sum + beta * C[c_index];
    }
}

static int rocblas_dgemm_row_major(rocblas_handle handle,
                                   const double *dA,
                                   const double *dB,
                                   double *dC,
                                   long M,
                                   long N,
                                   long K,
                                   double alpha,
                                   double beta) {
    if (!handle) {
        return -1;
    }

    if (M > INT_MAX || N > INT_MAX || K > INT_MAX) {
        fprintf(stderr, "rocBLAS dgemm requires dimensions <= %d.\n", INT_MAX);
        return -1;
    }

    const int m_prime = (int)N;
    const int n_prime = (int)M;
    const int k_prime = (int)K;

    const int lda = m_prime;
    const int ldb = k_prime;
    const int ldc = m_prime;

    rocblas_status rs = rocblas_dgemm(handle,
                                      rocblas_operation_none,
                                      rocblas_operation_none,
                                      m_prime,
                                      n_prime,
                                      k_prime,
                                      &alpha,
                                      dB,
                                      lda,
                                      dA,
                                      ldb,
                                      &beta,
                                      dC,
                                      ldc);
    if (rs != rocblas_status_success) {
        fprintf(stderr, "rocBLAS dgemm failed (%d)\n", (int)rs);
        return -1;
    }
    return 0;
}

static void destroy_rocblas_handles(rocblas_handle *handles, int count) {
    if (!handles || count <= 0) {
        return;
    }
    for (int i = 0; i < count; ++i) {
        if (handles[i]) {
            rocblas_destroy_handle(handles[i]);
            handles[i] = NULL;
        }
    }
}

/* ------------------------------- Test body -------------------------------- */

static int real_main(const HarnessOpts *opts) {
    struct sample_buffer sample_buf = {0};
    bool sample_buf_initialized = false;
    /* Gracefully skip if the PAPI AMD SMI component isn't available. */
    const char* root = getenv("PAPI_AMDSMI_ROOT");
    if (!root || !*root) {
        SKIP("PAPI_AMDSMI_ROOT not set");
    }

    /* Initialize PAPI */
    int statusFlag = PAPI_library_init(PAPI_VER_CURRENT);
    if (statusFlag != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI shared library version error: %s\n", PAPI_strerror(statusFlag));
        if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
        return 1;
    }

    /* Create EventSet */
    int EventSet = PAPI_NULL;
    statusFlag = PAPI_create_eventset(&EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI create eventset: %s\n", PAPI_strerror(statusFlag));
        if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
        return 1;
    }

    /* ------------------ Enumerate first five usable AMD-SMI events ------------------ */
    int cid = -1;
    const int ncomps = PAPI_num_components();
    for (int i = 0; i < ncomps && cid < 0; ++i) {
        const PAPI_component_info_t *cinfo = PAPI_get_component_info(i);
        if (cinfo && strcmp(cinfo->name, "amd_smi") == 0) cid = i;
    }
    if (cid < 0) {
        SKIP("Unable to locate the amd_smi component (PAPI built without ROCm?)");
    }

    const long m_dim = g_run_config.m_dim;
    const long k_dim = g_run_config.k_dim;
    const long n_dim = g_run_config.n_dim;
    const int event_count = g_run_config.event_count;
    const int iterations = g_run_config.iterations;
    const useconds_t pause_us = g_run_config.iteration_delay_us;

    char chosen_names[MAX_EVENT_SLOTS][PAPI_MAX_STR_LEN] = {{0}};
    int added = 0;

    if (g_run_config.test_mode) {
        if (sample_buffer_init(&sample_buf) != 0) {
            fprintf(stderr, "Failed to initialize sample buffer.\n");
            return 1;
        }
        sample_buf_initialized = true;
    }

    if (g_run_config.test_mode) {
        for (int i = 0; i < event_count; ++i) {
            char formatted[PAPI_MAX_STR_LEN] = {0};
            int formatted_len = snprintf(formatted,
                                         sizeof(formatted),
                                         kTestEventTemplates[i],
                                         g_run_config.device_index);
            if (formatted_len < 0 || (size_t)formatted_len >= sizeof(formatted)) {
                fprintf(stderr, "Failed to format test event template index %d\n", i);
                if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
                return 1;
            }
            char canonical[PAPI_MAX_STR_LEN] = {0};
            statusFlag = harness_canonicalize_event_name(formatted,
                                                         canonical,
                                                         sizeof(canonical));
            if (statusFlag != PAPI_OK) {
                fprintf(stderr, "Failed to resolve event %s: %s\n",
                        formatted, PAPI_strerror(statusFlag));
                if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
                return 1;
            }

            int event_code = 0;
            statusFlag = PAPI_event_name_to_code(canonical, &event_code);
            if (statusFlag != PAPI_OK) {
                fprintf(stderr, "Failed to translate event %s: %s\n",
                        canonical, PAPI_strerror(statusFlag));
                if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
                return 1;
            }

            statusFlag = PAPI_add_event(EventSet, event_code);
            if (statusFlag != PAPI_OK) {
                EXIT_WARNING_ON_ADD(statusFlag, canonical);
                fprintf(stderr, "PAPI_add_event failed for %s: %s\n",
                        canonical, PAPI_strerror(statusFlag));
                if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
                return 1;
            }

            strncpy(chosen_names[added], canonical, PAPI_MAX_STR_LEN - 1);
            ++added;
        }
    } else {
        int code = PAPI_NATIVE_MASK;
        if (PAPI_enum_cmp_event(&code, PAPI_ENUM_FIRST, cid) != PAPI_OK) {
            SKIP("No native events found for AMD-SMI component");
        }

        do {
            char base_name[PAPI_MAX_STR_LEN] = {0};
            if (PAPI_event_code_to_name(code, base_name) != PAPI_OK || base_name[0] == '\0') {
                continue;
            }

            int qualified_code = 0;
            if (PAPI_event_name_to_code(base_name, &qualified_code) != PAPI_OK) {
                continue;
            }

            char name[PAPI_MAX_STR_LEN] = {0};
            if (PAPI_event_code_to_name(qualified_code, name) != PAPI_OK || name[0] == '\0') {
                continue; /* couldn't resolve name; try next */
            }

            /* Skip process* events as in amdsmi_basics (not testable in this harness). */
            if (strncmp(name, "amd_smi:::process", 17) == 0 || strncmp(name, "process", 7) == 0) {
                continue;
            }

            statusFlag = PAPI_add_event(EventSet, qualified_code);
            if (statusFlag == PAPI_OK) {
                strncpy(chosen_names[added], name, PAPI_MAX_STR_LEN - 1);
                ++added;
            } else if (statusFlag == PAPI_ENOEVNT || statusFlag == PAPI_ECNFLCT || statusFlag == PAPI_EPERM) {
                /* Not usable (missing or HW/resource-limited) — keep enumerating. */
                continue;
            } else {
                /* Unexpected error; keep going to try to fill request. */
                continue;
            }
        } while (added < event_count && PAPI_enum_cmp_event(&code, PAPI_ENUM_EVENTS, cid) == PAPI_OK);

        if (added < event_count) {
            SKIP("Fewer than the requested number of usable AMD-SMI events available");
        }
    }

    if (opts->print) {
        printf("Monitoring events:\n");
        for (int i = 0; i < event_count; ++i) {
            printf("  %d) %s\n", i + 1, chosen_names[i]);
        }
    }

    /* HIP runtime preflight so HIP_CHECK won't hard-exit. */
    int device_count = 0;
    hipError_t device_query_status = hipGetDeviceCount(&device_count);
    if (device_query_status != hipSuccess ||
        device_count <= g_run_config.device_index) {
        if (opts->print) {
            fprintf(stdout,
                    "Requested HIP device %d not available (status=%d, count=%d)\n",
                    g_run_config.device_index,
                    (int)device_query_status,
                    device_count);
        }
        SKIP("Requested HIP device not available");
    }

    /* Use configured device and (optionally) print basic properties. */
    HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t deviceProp;
    HIP_CHECK(hipGetDeviceProperties(&deviceProp, g_run_config.device_index));
    if (opts->print) {
        printf("Using HIP device %d\n", g_run_config.device_index);
        printf("Device Name: %s\n", deviceProp.name);
        printf("Compute Units: %d\n", deviceProp.multiProcessorCount);
        printf("Max Threads Per Block: %d\n", deviceProp.maxThreadsPerBlock);
    }

    /* Host buffers (pinned) */
    size_t elements_A = (size_t)m_dim * (size_t)k_dim;
    size_t elements_B = (size_t)k_dim * (size_t)n_dim;
    size_t elements_C = (size_t)m_dim * (size_t)n_dim;

    size_t size_A = elements_A * sizeof(double);
    size_t size_B = elements_B * sizeof(double);
    size_t size_C = elements_C * sizeof(double);

    double *h_A = NULL, *h_B = NULL, *h_C = NULL;
    HIP_CHECK(hipHostMalloc(&h_A, size_A, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_B, size_B, hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&h_C, size_C, hipHostMallocDefault));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed.\n");
        if (h_A) HIP_CHECK_CLEANUP(hipHostFree(h_A));
        if (h_B) HIP_CHECK_CLEANUP(hipHostFree(h_B));
        if (h_C) HIP_CHECK_CLEANUP(hipHostFree(h_C));
        if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
        return 1;
    }

    for (size_t i = 0; i < elements_A; ++i) h_A[i] = (double)(i % 100);
    for (size_t i = 0; i < elements_B; ++i) h_B[i] = (double)(i % 100);
    for (size_t i = 0; i < elements_C; ++i) h_C[i] = 0.0;

    /* Device buffers per stream */
    double *d_A[NUM_STREAMS], *d_B[NUM_STREAMS], *d_C[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipMalloc((void**)&d_A[s], size_A));
        HIP_CHECK(hipMalloc((void**)&d_B[s], size_B));
        HIP_CHECK(hipMalloc((void**)&d_C[s], size_C));
    }

    hipStream_t streams[NUM_STREAMS];
    hipEvent_t events[NUM_STREAMS];
    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipStreamCreateWithFlags(&streams[s], hipStreamNonBlocking));
        HIP_CHECK(hipEventCreate(&events[s]));
    }

    rocblas_handle handles[NUM_STREAMS] = {0};
    if (g_run_config.use_rocblas) {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            rocblas_status rb_status = rocblas_create_handle(&handles[s]);
            if (rb_status != rocblas_status_success) {
                fprintf(stderr, "rocBLAS create handle failed for stream %d (%d)\n", s, (int)rb_status);
                destroy_rocblas_handles(handles, NUM_STREAMS);
                if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
                return 1;
            }
            rb_status = rocblas_set_pointer_mode(handles[s], rocblas_pointer_mode_host);
            if (rb_status != rocblas_status_success) {
                fprintf(stderr, "rocBLAS set_pointer_mode failed for stream %d (%d)\n", s, (int)rb_status);
                destroy_rocblas_handles(handles, NUM_STREAMS);
                if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
                return 1;
            }
            rb_status = rocblas_set_stream(handles[s], streams[s]);
            if (rb_status != rocblas_status_success) {
                fprintf(stderr, "rocBLAS set_stream failed for stream %d (%d)\n", s, (int)rb_status);
                destroy_rocblas_handles(handles, NUM_STREAMS);
                if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
                return 1;
            }
        }
    }

    /* H2D copies */
    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK(hipMemcpyAsync(d_A[s], h_A, size_A, hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_B[s], h_B, size_B, hipMemcpyHostToDevice, streams[s]));
        HIP_CHECK(hipMemcpyAsync(d_C[s], h_C, size_C, hipMemcpyHostToDevice, streams[s]));
    }

    /* Start counters */
    statusFlag = PAPI_start(EventSet);
    if (statusFlag == PAPI_ECNFLCT || statusFlag == PAPI_EPERM) {
        SKIP("Cannot start counters due to HW/resource limits");
    } else if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI_start: %s\n", PAPI_strerror(statusFlag));
        if (g_run_config.use_rocblas) {
            destroy_rocblas_handles(handles, NUM_STREAMS);
        }
        if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
        return 1;
    }

    /* Launch monitor thread (prints unless suppressed) */
    pthread_t monitor_thread;
    struct monitor_params params;
    params.EventSet = EventSet;
    params.print    = opts->print ? 1 : 0;
    params.event_count = event_count;
    params.samples = sample_buf_initialized ? &sample_buf : NULL;
    gettimeofday(&params.start_time, NULL);
    stop_monitor = 0;
    statusFlag = pthread_create(&monitor_thread, NULL, monitor_events, &params);
    if (statusFlag != 0) {
        fprintf(stderr, "pthread_create failed\n");
        if (g_run_config.use_rocblas) {
            destroy_rocblas_handles(handles, NUM_STREAMS);
        }
        if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
        return 1;
    }

    if (pause_us > 0) {
        usleep(pause_us); // Allow monitor to gather baseline samples before first kernel launch
    }

    /* Ensure copies are done */
    for (int s = 0; s < NUM_STREAMS; s++) HIP_CHECK(hipStreamSynchronize(streams[s]));

    double alpha = 0.75;
    double beta  = 0.5;

    dim3 blockDim(32, 32);
    dim3 gridDim((unsigned int)((n_dim + blockDim.x - 1) / blockDim.x),
                 (unsigned int)((m_dim + blockDim.y - 1) / blockDim.y));

    for (int iter = 0; iter < iterations; ++iter) {
        for (int s = 0; s < NUM_STREAMS; s++) {
            if (g_run_config.use_rocblas) {
                if (rocblas_dgemm_row_major(handles[s],
                                            d_A[s],
                                            d_B[s],
                                            d_C[s],
                                            m_dim,
                                            n_dim,
                                            k_dim,
                                            alpha,
                                            beta) != 0) {
                    destroy_rocblas_handles(handles, NUM_STREAMS);
                    stop_monitor = 1;
                    pthread_join(monitor_thread, NULL);
                    if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
                    return 1;
                }
            } else {
                hipLaunchKernelGGL(dgemm_kernel, gridDim, blockDim, 0, streams[s],
                                   d_A[s], d_B[s], d_C[s],
                                   m_dim, n_dim, k_dim, alpha, beta);
            }
            HIP_CHECK(hipEventRecord(events[s], streams[s]));
            HIP_CHECK(hipStreamSynchronize(streams[s]));
            if (pause_us > 0) {
                usleep(pause_us); // Allow the monitor to capture samples between launches
            }
        }
    }

    /* Stop the monitor and clean up */
    stop_monitor = 1;
    pthread_join(monitor_thread, NULL);

    if (sample_buf_initialized && sample_buf.push_failed) {
        fprintf(stderr, "Failed to store monitoring samples (out of memory).\n");
        if (g_run_config.use_rocblas) {
            destroy_rocblas_handles(handles, NUM_STREAMS);
        }
        sample_buffer_destroy(&sample_buf);
        return 1;
    }

    if (g_run_config.use_rocblas) {
        destroy_rocblas_handles(handles, NUM_STREAMS);
    }

    for (int s = 0; s < NUM_STREAMS; s++) {
        HIP_CHECK_CLEANUP(hipEventDestroy(events[s]));
        HIP_CHECK_CLEANUP(hipStreamDestroy(streams[s]));
        HIP_CHECK_CLEANUP(hipFree(d_A[s]));
        HIP_CHECK_CLEANUP(hipFree(d_B[s]));
        HIP_CHECK_CLEANUP(hipFree(d_C[s]));
    }
    HIP_CHECK_CLEANUP(hipHostFree(h_A));
    HIP_CHECK_CLEANUP(hipHostFree(h_B));
    HIP_CHECK_CLEANUP(hipHostFree(h_C));

    long long stop_values[MAX_EVENT_SLOTS] = {0};  // event_count were added
    statusFlag = PAPI_stop(EventSet, stop_values);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI_stop: %s\n", PAPI_strerror(statusFlag));
        if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
        return 1;
    }

    if (sample_buf_initialized) {
        struct timeval end_time;
        gettimeofday(&end_time, NULL);
        double elapsed_stop = (end_time.tv_sec - params.start_time.tv_sec) +
                              (end_time.tv_usec - params.start_time.tv_usec) / 1e6;
        if (sample_buffer_push(&sample_buf, elapsed_stop, stop_values, event_count) != 0) {
            fprintf(stderr, "Failed to record final sample.\n");
            sample_buffer_destroy(&sample_buf);
            return 1;
        }
    }

    statusFlag = PAPI_cleanup_eventset(EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI_cleanup_eventset: %s\n", PAPI_strerror(statusFlag));
        if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
        return 1;
    }
    statusFlag = PAPI_destroy_eventset(&EventSet);
    if (statusFlag != PAPI_OK) {
        fprintf(stderr, "PAPI_destroy_eventset: %s\n", PAPI_strerror(statusFlag));
        if (sample_buf_initialized) sample_buffer_destroy(&sample_buf);
        return 1;
    }

    HIP_CHECK_CLEANUP(hipDeviceReset());   // Optional; reduces "still reachable" reports from HIP in leak checkers
    PAPI_shutdown();                       // Triggers component cleanup and AMD SMI shutdown

    if (sample_buf_initialized) {
        if (write_results_csv(g_run_config.csv_path, chosen_names, event_count, &sample_buf) != 0) {
            sample_buffer_destroy(&sample_buf);
            return 1;
        }
        sample_buffer_destroy(&sample_buf);
    }
    return 0;
}

/* --------------------------- Test harness glue --------------------------- */

int main(int argc, char *argv[]) {
    harness_accept_tests_quiet(&argc, argv);
    if (parse_test_override(&argc, argv) != 0) {
        return 1;
    }
    HarnessOpts opts = parse_harness_cli(argc, argv);
    int papi_errno = real_main(&opts);
    return eval_result(opts, papi_errno);
}
