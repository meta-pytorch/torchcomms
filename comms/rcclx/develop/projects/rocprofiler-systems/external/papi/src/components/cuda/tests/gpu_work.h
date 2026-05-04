#include <stdio.h>
#include <cuda.h>

#define _GW_CALL(call)  \
do {  \
    cudaError_t _status = (call);  \
    if (_status != cudaSuccess) {  \
        fprintf(stderr, "%s: %d: " #call "\n", __FILE__, __LINE__);  \
    }  \
} while (0);

// Device code
__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

// Device code
__global__ void VecSub(const int* A, const int* B, int* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] - B[i];
}

static void initVec(int *vec, int n)
{
    for (int i=0; i< n; i++)
        vec[i] = i;
}

static void cleanUp(int *h_A, int *h_B, int *h_C, int *h_D, int *d_A, int *d_B, int *d_C, int *d_D)
{
    if (d_A)
        _GW_CALL(cudaFree(d_A));
    if (d_B)
        _GW_CALL(cudaFree(d_B));
    if (d_C)
        _GW_CALL(cudaFree(d_C));
    if (d_D)
        _GW_CALL(cudaFree(d_D));

    // Free host memory
    if (h_A)
        free(h_A);
    if (h_B)
        free(h_B);
    if (h_C)
        free(h_C);
    if (h_D)
        free(h_D);
}

static void VectorAddSubtract(int N, int quiet)
{
    if (N==0) N = 50000;
    size_t size = N * sizeof(int);
    int threadsPerBlock = 0;
    int blocksPerGrid = 0;
    int *h_A, *h_B, *h_C, *h_D;
    int *d_A, *d_B, *d_C, *d_D;
    int i, sum, diff;
    int device;
    cudaGetDevice(&device);
    // Allocate input vectors h_A and h_B in host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);
    h_D = (int*)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL || h_D == NULL) {
        fprintf(stderr, "Allocating input vectors failed.\n");
    }

    // Initialize input vectors
    initVec(h_A, N);
    initVec(h_B, N);
    memset(h_C, 0, size);
    memset(h_D, 0, size);

    // Allocate vectors in device memory
    _GW_CALL(cudaMalloc((void**)&d_A, size));
    _GW_CALL(cudaMalloc((void**)&d_B, size));
    _GW_CALL(cudaMalloc((void**)&d_C, size));
    _GW_CALL(cudaMalloc((void**)&d_D, size));

    // Copy vectors from host memory to device memory
    _GW_CALL(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    _GW_CALL(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Invoke kernel
    threadsPerBlock = 256;
    blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    if (!quiet) fprintf(stderr, "Launching kernel on device %d: blocks %d, thread/block %d\n",
    device, blocksPerGrid, threadsPerBlock);

    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    VecSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_D, N);

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    _GW_CALL(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    _GW_CALL(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));
    if (!quiet) fprintf(stderr, "Kernel launch complete and mem copied back from device %d\n", device);
    // Verify result
    for (i = 0; i < N; ++i) {
        sum = h_A[i] + h_B[i];
        diff = h_A[i] - h_B[i];
        if (h_C[i] != sum || h_D[i] != diff) {
            fprintf(stderr, "error: result verification failed\n");
            exit(-1);
        }
    }

    cleanUp(h_A, h_B, h_C, h_D, d_A, d_B, d_C, d_D);
}
