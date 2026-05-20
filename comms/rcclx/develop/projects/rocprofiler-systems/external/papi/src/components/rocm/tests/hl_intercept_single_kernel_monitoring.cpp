/**
 * @file   hl_intercepte_single_kernel_monitoring.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 */
#include "common.h"

int quiet;

int main(int argc, char *argv[])
{
    int papi_errno = PAPI_OK;
    hipError_t hip_errno = hipSuccess;
    quiet = tests_quiet(argc, argv);

    setenv("ROCP_HSA_INTERCEPT", "1", 1);
    setenv("PAPI_EVENTS", "rocm:::SQ_INSTS_VALU:device=0,rocm:::SQ_INSTS_SALU:device=0,rocm:::SQ_WAVES:device=0", 1);

    papi_errno = PAPI_hl_region_begin("matmul_intercept");
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_hl_region_begin", papi_errno);
    }

    hipStream_t stream;
    hip_errno = hipSetDevice(0);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipSetDevice", hip_errno);
    }

    hip_errno = hipStreamCreate(&stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamCreate", hip_errno);
    }

    void *handle;
    hip_do_matmul_init(&handle);
    hip_do_matmul_work(handle, stream);
    hip_errno = hipStreamSynchronize(stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipStreamSynchronize", hip_errno);
    }

    papi_errno = PAPI_hl_region_end("matmul_intercept");
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_hl_region_end", papi_errno);
    }
    hip_do_matmul_cleanup(&handle);

    PAPI_hl_stop();
    test_hl_pass(__FILE__);
    return 0;
}
