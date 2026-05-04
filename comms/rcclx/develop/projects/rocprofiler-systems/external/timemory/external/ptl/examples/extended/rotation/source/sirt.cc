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

#include "common.hh"
#include "constants.hh"
#include "data.hh"
#include "rotate_utils.hh"

//======================================================================================//

typedef CpuData::init_data_t  init_data_t;
typedef CpuData::data_array_t data_array_t;

//======================================================================================//

void
sirt_cpu(const float* data, int dy, int dt, int dx, const float* /*center*/,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter);

//======================================================================================//

int
cxx_sirt(const float* data, int dy, int dt, int dx, const float* center,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    auto tid = get_this_thread_id();
    ConsumeParameters(tid);
    static std::atomic<int> active;
    int                     count = active++;

    START_TIMER(cxx_timer);
    TIMEMORY_AUTO_TIMER("");

    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           get_this_thread_id(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    {
        TIMEMORY_AUTO_TIMER("");
        // run_algorithm(sirt_cpu, sirt_cuda, data, dy, dt, dx, center, theta, recon,
        // ngridx,
        //              ngridy, num_iter);
        sirt_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
    }

    auto tcount = GetEnv("PTL_PYTHON_THREADS", HW_CONCURRENCY);
    auto remain = --active;
    REPORT_TIMER(cxx_timer, __FUNCTION__, count, tcount);
    if(remain == 0)
    {
        std::stringstream ss;
        PrintEnv(ss);
        printf("[%lu] Reporting environment...\n\n%s\n", get_this_thread_id(),
               ss.str().c_str());
#if defined(PTL_USE_CUDA)
        for(int i = 0; i < cuda_device_count(); ++i)
        {
            // set the device
            cudaSetDevice(i);
            // sync the device
            cudaDeviceSynchronize();
            // reset the device
            cudaDeviceReset();
        }
#endif
    }
    else
    {
        printf("[%lu] Threads remaining: %i...\n", get_this_thread_id(), remain);
    }

    return scast<int>(true);
}

//======================================================================================//

void
sirt_cpu_compute_projection(data_array_t& cpu_data, int p, int dy, int dt, int dx, int nx,
                            int ny, const float* theta)
{
    ConsumeParameters(dy);
    auto cache = cpu_data[get_this_thread_id() % cpu_data.size()];

    // calculate some values
    float    theta_p = fmodf(theta[p] + constants::halfpi, constants::twopi);
    farray_t tmp_update(dy * nx * ny, 0.0);

    for(int s = 0; s < dy; ++s)
    {
        const float* data  = cache->data() + s * dt * dx;
        const float* recon = cache->recon() + s * nx * ny;
        auto&        rot   = cache->rot();
        auto&        tmp   = cache->tmp();

        // reset intermediate data
        cache->reset();

        // forward-rotate
        cxx_rotate_ip<float>(rot, recon, -theta_p, nx, ny);

        // compute simdata
        for(int d = 0; d < dx; ++d)
        {
            float sum = 0.0f;
            for(int i = 0; i < nx; ++i)
                sum += rot[d * nx + i];
            float upd = (data[p * dx + d] - sum);
            for(int i = 0; i < nx; ++i)
                rot[d * nx + i] += upd;
        }

        // back-rotate object
        cxx_rotate_ip<float>(tmp, rot.data(), theta_p, nx, ny);

        // update local update array
        for(uintmax_t i = 0; i < scast<uintmax_t>(nx * ny); ++i)
            tmp_update[(s * nx * ny) + i] += tmp[i];
    }

    cache->upd_mutex()->lock();
    for(int s = 0; s < dy; ++s)
    {
        // update shared update array
        float* update = cache->update() + s * nx * ny;
        float* tmp    = tmp_update.data() + s * nx * ny;
        for(uintmax_t i = 0; i < scast<uintmax_t>(nx * ny); ++i)
            update[i] += tmp[i];
    }
    cache->upd_mutex()->unlock();
}

//======================================================================================//

void
sirt_cpu(const float* data, int dy, int dt, int dx, const float* /*center*/,
         const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    typedef decltype(HW_CONCURRENCY) nthread_type;

    printf("[%lu]> %s : nitr = %i, dy = %i, dt = %i, dx = %i, nx = %i, ny = %i\n",
           get_this_thread_id(), __FUNCTION__, num_iter, dy, dt, dx, ngridx, ngridy);

    // explicitly set OpenMP number of threads to 1 so OpenCV doesn't try to
    // create (HW_CONCURRENCY * PYTHON_NUM_THREADS * PTL_NUM_THREADS) threads
    setenv("OMP_NUM_THREADS", "1", 1);

    // compute some properties (expected python threads, max threads, device assignment)
    auto min_threads = nthread_type(1);
    auto pythreads   = GetEnv("PTL_PYTHON_THREADS", HW_CONCURRENCY);
    auto max_threads = HW_CONCURRENCY / std::max(pythreads, min_threads);
    auto nthreads    = std::max(GetEnv("PTL_NUM_THREADS", max_threads), min_threads);

    typedef TaskManager manager_t;
    TaskRunManager*     run_man = cpu_run_manager();
    init_run_manager(run_man, nthreads);
    TaskManager* task_man = run_man->GetTaskManager();

    TIMEMORY_AUTO_TIMER("");

    Mutex       upd_mutex;
    Mutex       sum_mutex;
    uintmax_t   recon_pixels = scast<uintmax_t>(dy * ngridx * ngridy);
    farray_t    update(recon_pixels, 0.0f);
    init_data_t init_data =
        CpuData::initialize(nthreads, dy, dt, dx, ngridx, ngridy, recon, data,
                            update.data(), &upd_mutex, &sum_mutex);
    data_array_t cpu_data = std::get<0>(init_data);
    iarray_t     sum_dist = cxx_compute_sum_dist(dy, dt, dx, ngridx, ngridy, theta);

    //----------------------------------------------------------------------------------//
    for(int i = 0; i < num_iter; i++)
    {
        START_TIMER(t_start);
        TIMEMORY_AUTO_TIMER();

        // reset global update
        memset(update.data(), 0, recon_pixels * sizeof(float));

        // sync and reset
        CpuData::reset(cpu_data);

        // execute the loop over slices and projection angles
        execute<manager_t, data_array_t>(task_man, dt, std::ref(cpu_data),
                                         sirt_cpu_compute_projection, dy, dt, dx, ngridx,
                                         ngridy, theta);

        // update the global recon with global update and sum_dist
        for(uintmax_t ii = 0; ii < recon_pixels; ++ii)
        {
            if(sum_dist[ii] != 0.0f && dx != 0 && std::isfinite(update[ii]))
            {
                recon[ii] += update[ii] / sum_dist[ii] / scast<float>(dx);
            }
            else if(!std::isfinite(update[ii]))
            {
                std::cout << "update[" << ii << "] is not finite : " << update[ii]
                          << std::endl;
            }
        }
        REPORT_TIMER(t_start, "iteration", i, num_iter);
    }

    printf("\n");
}

//======================================================================================//
#if !defined(PTL_USE_CUDA)
void
sirt_cuda(const float* data, int dy, int dt, int dx, const float* center,
          const float* theta, float* recon, int ngridx, int ngridy, int num_iter)
{
    sirt_cpu(data, dy, dt, dx, center, theta, recon, ngridx, ngridy, num_iter);
}
#endif
//======================================================================================//
