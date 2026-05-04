// MIT License
//
// Copyright (c) 2019 Jonathan R. Madsen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#pragma once

#include "common.hh"
#include "constants.hh"
#include "macros.hh"
#include "rotate_utils.hh"
#include "typedefs.hh"

//======================================================================================//

class CpuData
{
public:
    typedef std::shared_ptr<CpuData>                       data_ptr_t;
    typedef std::vector<data_ptr_t>                        data_array_t;
    typedef std::tuple<data_array_t, float*, const float*> init_data_t;

public:
    CpuData(unsigned id, int dy, int dt, int dx, int nx, int ny, const float* data,
            float* recon, float* update, Mutex* upd_mutex, Mutex* sum_mutex)
    : m_id(id)
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_rot(farray_t(scast<uintmax_t>(m_nx * m_ny), 0.0f))
    , m_tmp(farray_t(scast<uintmax_t>(m_nx * m_ny), 0.0f))
    , m_update(update)
    , m_recon(recon)
    , m_data(data)
    , m_upd_mutex(upd_mutex)
    , m_sum_mutex(sum_mutex)
    {
        // we don't want null pointers here
        assert(m_upd_mutex && m_sum_mutex);
    }

    ~CpuData() {}

public:
    farray_t&       rot() { return m_rot; }
    farray_t&       tmp() { return m_tmp; }
    const farray_t& rot() const { return m_rot; }
    const farray_t& tmp() const { return m_tmp; }

    float*       update() const { return m_update; }
    float*       recon() { return m_recon; }
    const float* recon() const { return m_recon; }
    const float* data() const { return m_data; }

    Mutex* upd_mutex() const { return m_upd_mutex; }
    Mutex* sum_mutex() const { return m_sum_mutex; }

    void reset()
    {
        // reset temporaries to zero (NECESSARY!)
        // -- note: the OpenCV effectively ensures that we overwrite all values
        //          because we use cv::Mat::zeros and copy that to destination
        memset(m_rot.data(), 0, scast<uintmax_t>(m_nx * m_ny) * sizeof(float));
        memset(m_tmp.data(), 0, scast<uintmax_t>(m_nx * m_ny) * sizeof(float));
    }

public:
    // static functions
    static init_data_t initialize(unsigned nthreads, int dy, int dt, int dx, int ngridx,
                                  int ngridy, float* recon, const float* data,
                                  float* update, Mutex* upd_mtx, Mutex* sum_mtx)
    {
        data_array_t cpu_data(nthreads);
        for(unsigned ii = 0; ii < nthreads; ++ii)
        {
            cpu_data[ii] = data_ptr_t(new CpuData(ii, dy, dt, dx, ngridx, ngridy, data,
                                                  recon, update, upd_mtx, sum_mtx));
        }
        return init_data_t(cpu_data, recon, data);
    }

    static void reset(data_array_t& data)
    {
        // reset "update" to zero
        for(auto& itr : data)
            itr->reset();
    }

protected:
    unsigned     m_id;
    int          m_dy;
    int          m_dt;
    int          m_dx;
    int          m_nx;
    int          m_ny;
    farray_t     m_rot;
    farray_t     m_tmp;
    float*       m_update;
    float*       m_recon;
    const float* m_data;
    Mutex*       m_upd_mutex;
    Mutex*       m_sum_mutex;
};

//======================================================================================//

#if defined(__NVCC__) && defined(PTL_USE_CUDA)

//======================================================================================//

class GpuData
{
public:
    // typedefs
    typedef GpuData                                  this_type;
    typedef std::shared_ptr<GpuData>                 data_ptr_t;
    typedef std::vector<data_ptr_t>                  data_array_t;
    typedef std::tuple<data_array_t, float*, float*> init_data_t;

public:
    // ctors, dtors, assignment
    GpuData(int device, int id, int dy, int dt, int dx, int nx, int ny, const float* data,
            float* recon, float* update)
    : m_device(device)
    , m_id(id)
    , m_grid(GetGridSize())
    , m_block(GetBlockSize())
    , m_dy(dy)
    , m_dt(dt)
    , m_dx(dx)
    , m_nx(nx)
    , m_ny(ny)
    , m_rot(nullptr)
    , m_tmp(nullptr)
    , m_update(update)
    , m_recon(recon)
    , m_data(data)
    {
        cuda_set_device(m_device);
        m_streams = create_streams(m_num_streams, cudaStreamNonBlocking);
        m_rot     = gpu_malloc<float>(m_dy * m_nx * m_ny);
        m_tmp     = gpu_malloc<float>(m_dy * m_nx * m_ny);
    }

    ~GpuData()
    {
        cudaFree(m_rot);
        cudaFree(m_tmp);
        destroy_streams(m_streams, m_num_streams);
    }

    GpuData(const this_type&) = delete;
    GpuData(this_type&&)      = default;

    this_type& operator=(const this_type&) = delete;
    this_type& operator=(this_type&&) = default;

public:
    // access functions
    int          device() const { return m_device; }
    int          grid() const { return compute_grid(m_dx); }
    int          block() const { return m_block; }
    float*       rot() const { return m_rot; }
    float*       tmp() const { return m_tmp; }
    float*       update() const { return m_update; }
    float*       recon() { return m_recon; }
    const float* recon() const { return m_recon; }
    const float* data() const { return m_data; }
    cudaStream_t stream(int n = 0) { return m_streams[n % m_num_streams]; }

public:
    // assistant functions
    int compute_grid(int size) const
    {
        return (m_grid < 1) ? ((size + m_block - 1) / m_block) : m_grid;
    }

    void sync(int stream_id = -1)
    {
        auto _sync = [&](cudaStream_t _stream) { stream_sync(_stream); };

        if(stream_id < 0)
            for(int i = 0; i < m_num_streams; ++i)
                _sync(m_streams[i]);
        else
            _sync(m_streams[stream_id % m_num_streams]);
    }

    void reset()
    {
        // reset destination arrays (NECESSARY!)
        gpu_memset<float>(m_rot, 0, m_dy * m_nx * m_ny, *m_streams);
        gpu_memset<float>(m_tmp, 0, m_dy * m_nx * m_ny, *m_streams);
    }

public:
    // static functions
    static init_data_t initialize(int device, int nthreads, int dy, int dt, int dx,
                                  int ngridx, int ngridy, float* cpu_recon,
                                  const float* cpu_data, float* update)
    {
        uintmax_t nstreams = 2;
        auto      streams  = create_streams(nstreams, cudaStreamNonBlocking);
        float*    recon =
            gpu_malloc_and_memcpy<float>(cpu_recon, dy * ngridx * ngridy, streams[0]);
        float* data = gpu_malloc_and_memcpy<float>(cpu_data, dy * dt * dx, streams[1]);
        data_array_t gpu_data(nthreads);
        for(int ii = 0; ii < nthreads; ++ii)
        {
            gpu_data[ii] = data_ptr_t(
                new GpuData(device, ii, dy, dt, dx, ngridx, ngridy, data, recon, update));
        }

        // synchronize and destroy
        destroy_streams(streams, nstreams);

        return init_data_t(gpu_data, recon, data);
    }

    static void reset(data_array_t& data)
    {
        // reset "update" to zero
        for(auto& itr : data)
            itr->reset();
    }

    static void sync(data_array_t& data)
    {
        // sync all the streams
        for(auto& itr : data)
            itr->sync();
    }

protected:
    // data
    int           m_device;
    int           m_id;
    int           m_grid;
    int           m_block;
    int           m_dy;
    int           m_dt;
    int           m_dx;
    int           m_nx;
    int           m_ny;
    float*        m_rot;
    float*        m_tmp;
    float*        m_update;
    float*        m_recon;
    const float*  m_data;
    int           m_num_streams = 1;
    cudaStream_t* m_streams     = nullptr;
};

#endif  // NVCC and PTL_USE_CUDA

//======================================================================================//
