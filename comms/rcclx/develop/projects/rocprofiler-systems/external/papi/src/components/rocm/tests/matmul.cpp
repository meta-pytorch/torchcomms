/**
 * @file   matmul.cpp
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 * Simple matrix to matrix multiplication kernel
 * written in hip and used by other tests in this
 * directory
 */
#include <stdio.h>
#include "common.h"

extern int quiet;
extern void hip_test_fail(const char *, int, const char *, hipError_t);

__global__ void
matmul(float *A, float *B, float *C, int N)
{
    int i = (hipBlockIdx_y * hipBlockDim_y) + hipThreadIdx_y;
    int j = (hipBlockIdx_x * hipBlockDim_x) + hipThreadIdx_x;

    if (i < N and j < N) {
        float sum = 0.0;
        for (int k = 0; k < N; ++k) {
            sum += A[(i * N) + k] * B[(k * N) + j];
        }
        C[(i * N) + j] = sum;
    }
}

struct memory {
    float *h_A;
    float *h_B;
    float *h_C;
    float *d_A;
    float *d_B;
    float *d_C;
};

void
hip_do_matmul_init(void **handle)
{
    hipError_t hip_errno;

    struct memory *handle_p = (struct memory *) malloc(sizeof(struct memory));
    if (handle_p == NULL) {
        test_fail(__FILE__, __LINE__, "malloc", PAPI_ESYS);
    }
    hip_errno = hipHostMalloc(&handle_p->h_A, sizeof(float) * ROWS * COLS);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipHostMalloc", hip_errno);
    }
    hip_errno = hipHostMalloc(&handle_p->h_B, sizeof(float) * ROWS * COLS);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipHostMalloc", hip_errno);
    }
    hip_errno = hipHostMalloc(&handle_p->h_C, sizeof(float) * ROWS * COLS);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipHostMalloc", hip_errno);
    }

    hip_errno = hipMalloc(&handle_p->d_A, sizeof(float) * ROWS * COLS);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipMalloc", hip_errno);
    }
    hip_errno = hipMalloc(&handle_p->d_B, sizeof(float) * ROWS * COLS);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipMalloc", hip_errno);
    }
    hip_errno = hipMalloc(&handle_p->d_C, sizeof(float) * ROWS * COLS);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipMalloc", hip_errno);
    }

    for (int i = 0; i < ROWS * COLS; ++i) {
        handle_p->h_A[i] = handle_p->h_B[i] = (float) (rand() % 1000);
        handle_p->h_C[i] = 0.0;
    }

    *handle = handle_p;
}

void
hip_do_matmul_work(void *handle, hipStream_t stream)
{
    hipError_t hip_errno;
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    struct memory *handle_p = (struct memory *) handle;
    h_A = handle_p->h_A;
    h_B = handle_p->h_B;
    h_C = handle_p->h_C;
    d_A = handle_p->d_A;
    d_B = handle_p->d_B;
    d_C = handle_p->d_C;

    hip_errno = hipMemcpyAsync(d_A, h_A, sizeof(float) * ROWS * COLS,
                               hipMemcpyHostToDevice, stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipMemcpyAsync", hip_errno);
    }
    hip_errno = hipMemcpyAsync(d_B, h_B, sizeof(float) * ROWS * COLS,
                               hipMemcpyHostToDevice, stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipMemcpyAsync", hip_errno);
    }

    dim3 grid_dim  = dim3(ROWS / BLOCK_DIM_X, COLS / BLOCK_DIM_Y);
    dim3 block_dim = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);

    hipLaunchKernelGGL(matmul, grid_dim, block_dim, 0, stream, d_A, d_B, d_C,
                       ROWS);
    hip_errno = hipGetLastError();
    if (hip_errno != hipSuccess) {
        if (!quiet) {
            fprintf(stderr, "Error! Failed launching kernel -> '%s'\n",
                    hipGetErrorString(hip_errno));
        }
        hip_test_fail(__FILE__, __LINE__, "hipLaunchKernelGGL", hip_errno);
    }

    hip_errno = hipMemcpyAsync(h_C, d_C, sizeof(float) * ROWS * COLS,
                               hipMemcpyDeviceToHost, stream);
    if (hip_errno != hipSuccess) {
        hip_test_fail(__FILE__, __LINE__, "hipMemcpyAsync", hip_errno);
    }
}

void
hip_do_matmul_cleanup(void **handle)
{
    struct memory *handle_p = (struct memory *) (*handle);
    (void)hipFree(handle_p->h_A);
    (void)hipFree(handle_p->h_B);
    (void)hipFree(handle_p->h_C);
    (void)hipFree(handle_p->d_A);
    (void)hipFree(handle_p->d_B);
    (void)hipFree(handle_p->d_C);
    free(handle_p);
    *handle = NULL;
}
