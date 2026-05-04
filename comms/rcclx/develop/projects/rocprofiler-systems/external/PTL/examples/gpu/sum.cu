//
// MIT License
// Copyright (c) 2018 Jonathan R. Madsen
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
// "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
// LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
// PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ---------------------------------------------------------------
//
//   PTL CUDA implementation
//
//

//============================================================================//

#include "sum.hh"

#define PRINT_HERE(extra)                                                                \
    printf("> %s@'%s':%i %s\n", __FUNCTION__, __FILE__, __LINE__, extra)

//============================================================================//

//  gridDim:    This variable contains the dimensions of the grid.
//  blockIdx:   This variable contains the block index within the grid.
//  blockDim:   This variable and contains the dimensions of the block.
//  threadIdx:  This variable contains the thread index within the block.

//============================================================================//
//
//  efficient reduction
//  https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//
//============================================================================//

template <unsigned int blockSize, typename _Tp>
__device__ void
warpReduce(volatile _Tp* _data, unsigned int tid)
{
    if(blockSize >= 64)
        _data[tid] += _data[tid + 32];
    if(blockSize >= 32)
        _data[tid] += _data[tid + 16];
    if(blockSize >= 16)
        _data[tid] += _data[tid + 8];
    if(blockSize >= 8)
        _data[tid] += _data[tid + 4];
    if(blockSize >= 4)
        _data[tid] += _data[tid + 2];
    if(blockSize >= 2)
        _data[tid] += _data[tid + 1];
}

//----------------------------------------------------------------------------//

template <unsigned int blockSize, typename _Tp>
__global__ void
reduce(_Tp* _idata, _Tp* _odata, unsigned int n)
{
    extern __shared__ _Tp _data[];
    unsigned int          tid      = threadIdx.x;
    unsigned int          i        = (2 * blockSize) * blockIdx.x + tid;
    unsigned int          gridSize = 2 * blockSize * gridDim.x;
    _data[tid]                     = 0;

    while(i < n)
    {
        _data[tid] += _idata[i] + _idata[i + blockSize];
        i += gridSize;
    }

    __syncthreads();

    if(blockSize >= 512)
    {
        if(tid < 256)
        {
            _data[tid] += _data[tid + 256];
        }
        __syncthreads();
    }

    if(blockSize >= 256)
    {
        if(tid < 128)
        {
            _data[tid] += _data[tid + 128];
        }
        __syncthreads();
    }

    if(blockSize >= 128)
    {
        if(tid < 64)
        {
            _data[tid] += _data[tid + 64];
        }
        __syncthreads();
    }

    if(tid < 32)
        warpReduce<blockSize, _Tp>(_data, tid);

    if(tid == 0)
        _odata[blockIdx.x] = _data[0];
}

//----------------------------------------------------------------------------//

template <typename _Tp>
void
compute_reduction(int threads, _Tp* _idata, _Tp* _odata, int dimGrid, int dimBlock,
                  int smemSize, cudaStream_t stream)
{
    cudaStreamSynchronize(stream);
    CUDA_CHECK_LAST_ERROR();

    switch(threads)
    {
        case 512:
            reduce<512, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 256:
            reduce<256, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 128:
            reduce<128, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 64:
            reduce<64, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 32:
            reduce<32, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 16:
            reduce<16, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 8:
            reduce<8, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 4:
            reduce<4, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 2:
            reduce<2, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
        case 1:
            reduce<1, _Tp>
                <<<dimGrid, dimBlock, smemSize, stream>>>(_idata, _odata, threads);
            break;
    }
    CUDA_CHECK_LAST_ERROR();

    cudaStreamSynchronize(stream);
    CUDA_CHECK_LAST_ERROR();
}

//============================================================================//

template <typename _Tp>
void
call_compute_reduction(int64_t& _i, uint64_t& _offset, int nthreads, _Tp* _idata,
                       _Tp* _odata, int dimGrid, int dimBlock, int smemSize,
                       cudaStream_t stream)
{
    // assumes nthreads < cuda_max_threads_per_block()
    compute_reduction(nthreads, _idata + _offset, _odata + _offset, dimGrid, dimBlock,
                      smemSize, stream);
    _i -= nthreads;
    _offset += nthreads;
}

//============================================================================//

float
compute_sum_host(aligned_ptr<float>& data, cudaStream_t stream, bool with_thrust,
                 float* buffer)
{
    float _sum;

    if(with_thrust)
    {
        cudaStreamSynchronize(stream);
        CUDA_CHECK_LAST_ERROR();

        _sum = thrust::reduce(thrust::system::cuda::par.on(stream), data.ptr,
                              data.ptr + data.size, 0.0f, thrust::plus<float>());

        CUDA_CHECK_LAST_ERROR();

        cudaStreamSynchronize(stream);
        CUDA_CHECK_LAST_ERROR();
    }
    else
    {
        // PRINT_HERE("");
        // PRINT_HERE(std::string(std::string("size    : ") +
        // std::to_string(data.size)).c_str());
        // PRINT_HERE(std::string(std::string("padding : ") +
        // std::to_string(data.padding)).c_str());
        // PRINT_HERE(std::string(std::string("storage : ") +
        // std::to_string(data.storage_size)).c_str());

        if(data.size < 1 || data.storage_size < 1)
            return 0.0f;

        int64_t  remain = data.size;
        uint64_t offset = 0;

        int smemSize = cuda_shared_memory_per_block();
        int dimGrid  = cuda_multi_processor_count();
        int dimBlock = cuda_max_threads_per_block();

        float* _idata = data.ptr;
        float* _odata = buffer;
        async_gpu_memset<float>(_odata, data.storage_size, stream);

        CUDA_CHECK_LAST_ERROR();

        while(remain > 0)
        {
            for(const auto& itr : { 512, 256, 128, 64, 32, 16, 8, 4, 2, 1 })
            {
                if(remain >= itr)
                {
                    call_compute_reduction(remain, offset, itr, _idata, _odata, dimGrid,
                                           dimBlock, smemSize, stream);
                    break;
                }
            }
        }

        cudaMemcpyAsync(&_sum, _odata, 1 * sizeof(float), cudaMemcpyDeviceToHost, stream);
        CUDA_CHECK_LAST_ERROR();
        cudaDeviceSynchronize();
        CUDA_CHECK_LAST_ERROR();
    }

    return _sum;
}

//============================================================================//
