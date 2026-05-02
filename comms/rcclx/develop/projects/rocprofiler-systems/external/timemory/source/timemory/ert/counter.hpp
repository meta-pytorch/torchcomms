// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/** \file timemory/ert/counter.hpp
 * \headerfile timemory/ert/counter.hpp "timemory/ert/counter.hpp"
 * Provides counter (i.e. timer, hw counters) for when executing ERT
 *
 */

#pragma once

#include "timemory/backends/device.hpp"
#include "timemory/backends/dmp.hpp"
#include "timemory/backends/hip.hpp"
#include "timemory/components/cuda/backends.hpp"
#include "timemory/components/timing/ert_timer.hpp"
#include "timemory/defines.h"
#include "timemory/ert/aligned_allocator.hpp"
#include "timemory/ert/barrier.hpp"
#include "timemory/ert/cache_size.hpp"
#include "timemory/ert/data.hpp"
#include "timemory/ert/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/policy.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/utility/filepath.hpp"
#include "timemory/utility/macros.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace tim
{
namespace ert
{
using namespace memory;

//--------------------------------------------------------------------------------------//
//  measure floating-point or integer operations
//
template <typename DeviceT, typename Tp, typename Counter>
class counter
{
public:
    using mutex_t       = std::recursive_mutex;
    using lock_t        = std::unique_lock<mutex_t>;
    using counter_type  = Counter;
    using ert_data_t    = exec_data<Counter>;
    using this_type     = counter<DeviceT, Tp, Counter>;
    using callback_type = std::function<void(uint64_t, this_type&)>;
    using data_type     = typename ert_data_t::value_type;
    using data_ptr_t    = std::shared_ptr<ert_data_t>;
    using ull           = unsigned long long;
    using skip_ops_t    = std::unordered_set<size_t>;

public:
    //----------------------------------------------------------------------------------//
    //  default construction
    //
    counter()                   = default;
    ~counter()                  = default;
    counter(const counter&)     = default;
    counter(counter&&) noexcept = default;
    counter& operator=(const counter&) = default;
    counter& operator=(counter&&) noexcept = default;

    //----------------------------------------------------------------------------------//
    // standard creation
    //
    explicit counter(const exec_params& _params, data_ptr_t _exec_data,
                     uint64_t _align = 8 * sizeof(Tp))
    : params(_params)
    , align(_align)
    , data(std::move(_exec_data))
    {
        compute_internal();
    }

    //----------------------------------------------------------------------------------//
    // overload how to create the counter with a callback function
    //
    counter(const exec_params& _params, callback_type _func, data_ptr_t _exec_data,
            uint64_t _align = 8 * sizeof(Tp))
    : params(_params)
    , align(_align)
    , data(std::move(_exec_data))
    , configure_callback(std::move(_func))
    {
        compute_internal();
    }

public:
    //----------------------------------------------------------------------------------//
    ///  allocate a buffer for the ERT calculation
    ///     uses this function if device is CPU or device is GPU and type is not half2
    ///
    template <typename Up = Tp, typename Dev = DeviceT,
              typename std::enable_if<(std::is_same<Dev, device::cpu>::value ||
                                       (std::is_same<Dev, device::gpu>::value &&
                                        !std::is_same<Up, gpu::fp16_t>::value)),
                                      int>::type = 0>
    Up* get_buffer()
    {
        // check alignment and
        align = std::max<uint64_t>(align, 8 * sizeof(Up));
        compute_internal();

        if(settings::debug())
            printf("[%s]> nsize = %llu\n", __FUNCTION__, (ull) nsize);
        Up* buffer = allocate_aligned<Up, DeviceT>(nsize, align);
        if(settings::debug())
            printf("[%s]> buffer = %p\n", __FUNCTION__, (void*) buffer);
        device::params<DeviceT> _params(0, 512, 0, 0);
        device::launch(nsize, _params, initialize_buffer<DeviceT, Up, uint64_t>, buffer,
                       Up{ 1 }, nsize);
        return buffer;
    }

    //----------------------------------------------------------------------------------//
    ///  allocate a buffer for the ERT calculation
    ///     uses this function if device is GPU and type is half2
    ///
    template <typename Up = Tp, typename Dev = DeviceT,
              typename std::enable_if<(std::is_same<Up, gpu::fp16_t>::value &&
                                       std::is_same<Dev, device::gpu>::value),
                                      int>::type = 0>
    Up* get_buffer()
    {
        // check alignment and
        align = std::max<uint64_t>(align, 8 * sizeof(Up));
        compute_internal();

        if(settings::debug())
            printf("[%s]> nsize = %llu\n", __FUNCTION__, (ull) nsize);
        Up* buffer = allocate_aligned<Up, DeviceT>(nsize, align);
        if(settings::debug())
            printf("[%s]> buffer = %p\n", __FUNCTION__, (void*) buffer);
        device::params<DeviceT> _params(0, 512, 0, 0);
        device::launch(nsize, _params, initialize_buffer<DeviceT, Up, uint32_t>, buffer,
                       Up{ 1, 1 }, nsize);
        return buffer;
    }

    //----------------------------------------------------------------------------------//
    //  destroy associated buffer
    //
    void destroy_buffer(Tp* buffer) { free_aligned<Tp, DeviceT>(buffer); }

    //----------------------------------------------------------------------------------//
    // execute the callback that may customize the thread before returning the object
    // that provides the measurement
    //
    void configure(uint64_t tid) { configure_callback(tid, *this); }

    //----------------------------------------------------------------------------------//
    // execute the callback that may customize the thread before returning the object
    // that provides the measurement
    //
    counter_type get_counter() const { return counter_type(); }

    //----------------------------------------------------------------------------------//
    // record the data from a thread/process. Extra exec_params (_itrp) should contain
    // the computed grid size for serialization
    //
    void record(counter_type& _counter, int n, int trials, uint64_t nops,
                const exec_params& _itrp)
    {
        uint64_t working_set_size = n * params.nthreads * params.nproc;
        uint64_t working_set      = working_set_size * bytes_per_element;
        uint64_t total_bytes      = trials * working_set * memory_accesses_per_element;
        uint64_t total_ops        = trials * working_set_size * nops;

        std::stringstream ss;
        ss << label;
        if(label.length() == 0)
        {
            if(nops > 1)
            {
                ss << "vector_op";
            }
            else
            {
                ss << "scalar_op";
            }
        }

        auto      _label = tim::demangle<Tp>();
        data_type _data(ss.str(), working_set, trials, total_bytes, total_ops, nops,
                        _counter, DeviceT::name(), _label, _itrp);

#if !defined(TIMEMORY_WINDOWS)
        // using namespace tim::stl::ostream;
        // if(settings::verbose() > 1 || settings::debug())
        //    std::cout << "[RECORD]> " << _data << std::endl;
#endif

        static std::mutex _mutex;
        // std::unique_lock<std::mutex> _lock(_mutex);
        _mutex.lock();
        *data += _data;
        _mutex.unlock();
    }

    //----------------------------------------------------------------------------------//
    //
    template <typename FuncT>
    void set_callback(FuncT&& _f)
    {
        configure_callback = std::forward<FuncT>(_f);
    }

    //----------------------------------------------------------------------------------//
    //      provide ability to write to JSON/XML
    //
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int)
    {
        if(!data.get())  // for input
            data = std::make_shared<ert_data_t>();
        ar(cereal::make_nvp("params", params), cereal::make_nvp("data", *data));
    }

    //----------------------------------------------------------------------------------//
    //      write to stream
    //
    friend std::ostream& operator<<(std::ostream& os, const counter& obj)
    {
        std::stringstream ss;
        ss << obj.params << ", "
           << "bytes_per_element = " << obj.bytes_per_element << ", "
           << "memory_accesses_per_element = " << obj.memory_accesses_per_element << ", "
           << "alignment = " << obj.align << ", "
           << "nsize = " << obj.nsize << ", "
           << "label = " << obj.label << ", "
           << "data entries = " << ((obj.data.get()) ? obj.data->size() : 0);
        os << ss.str();
        return os;
    }

    //----------------------------------------------------------------------------------//
    //  Get the data pointer
    //
    data_ptr_t&       get_data() { return data; }
    const data_ptr_t& get_data() const { return data; }

    //----------------------------------------------------------------------------------//
    //  Skip the flop counts
    //
    void add_skip_ops(size_t _Nops) { skip_ops.insert(_Nops); }

    void add_skip_ops(std::initializer_list<size_t> _args)
    {
        for(const auto& itr : _args)
            skip_ops.insert(itr);
    }

    bool skip(size_t _Nops) { return (skip_ops.count(_Nops) > 0); }

public:
    //----------------------------------------------------------------------------------//
    //  public data members, modify as needed
    //
    exec_params params                      = exec_params();                   // NOLINT
    int         bytes_per_element           = 0;                               // NOLINT
    int         memory_accesses_per_element = 0;                               // NOLINT
    uint64_t    align                       = sizeof(Tp);                      // NOLINT
    uint64_t    nsize                       = 0;                               // NOLINT
    data_ptr_t  data                        = std::make_shared<ert_data_t>();  // NOLINT
    std::string label                       = "";                              // NOLINT
    skip_ops_t  skip_ops                    = skip_ops_t();                    // NOLINT

private:
    callback_type configure_callback = [](uint64_t, this_type&) {};

private:
    //----------------------------------------------------------------------------------//
    //  compute the data size
    //
    void compute_internal()
    {
        if(device::is_cpu<DeviceT>::value)
            params.nstreams = 1;
        nsize = params.memory_max / params.nproc / params.nthreads;
        nsize = nsize & (~(align - 1));
        nsize = nsize / sizeof(Tp);
        nsize = std::max<uint64_t>(nsize, 1);
    }
};

//--------------------------------------------------------------------------------------//

template <typename Counter>
inline void
serialize(std::string fname, exec_data<Counter>& obj)
{
    using exec_data_vec_t = std::vector<exec_data<Counter>>;

    int dmp_rank = dmp::rank();
    int dmp_size = dmp::size();

    exec_data_vec_t results(dmp_size);
    if(dmp::is_initialized())
    {
        dmp::barrier();

#if defined(TIMEMORY_USE_MPI) || defined(TIMEMORY_USE_UPCXX)
        auto space = cereal::JSONOutputArchive::Options::IndentChar::space;

        //------------------------------------------------------------------------------//
        //  Used to convert a result to a serialization
        //
        auto send_serialize = [&](const exec_data<Counter>& src) {
            std::stringstream ss;
            {
                cereal::JSONOutputArchive::Options opt{ 16, space, 0 };
                cereal::JSONOutputArchive          oa{ ss, opt };
                oa(cereal::make_nvp("data", src));
            }
            return ss.str();
        };

        //------------------------------------------------------------------------------//
        //  Used to convert the serialization to a result
        //
        auto recv_serialize = [&](const std::string& src) {
            exec_data<Counter> ret{};
            std::stringstream  ss{};
            ss << src;
            {
                cereal::JSONInputArchive ia{ ss };
                try
                {
                    ia(cereal::make_nvp("data", ret));
                } catch(cereal::Exception& e)
                {
                    std::string _msg = ss.str();
                    // truncate
                    constexpr size_t max_msg_len = 60;
                    if(_msg.length() > max_msg_len)
                        _msg = TIMEMORY_JOIN("...", _msg.substr(0, max_msg_len - 13),
                                             _msg.substr(_msg.length() - 10));
                    TIMEMORY_PRINT_HERE(
                        "Warning! Exception in ert::serialize::recv_serialize: "
                        "%s\n\t%s",
                        e.what(), _msg.c_str());
                }
            }
            return ret;
        };
#endif

#if defined(TIMEMORY_USE_MPI)

        auto str_ret = send_serialize(obj);

        if(dmp_rank == 0)
        {
            for(int i = 1; i < dmp_size; ++i)
            {
                std::string str;
                mpi::recv(str, i, 0, mpi::comm_world_v);
                results[i] = recv_serialize(str);
            }
            results[dmp_rank] = std::move(obj);
        }
        else
        {
            mpi::send(str_ret, 0, 0, mpi::comm_world_v);
        }

#elif defined(TIMEMORY_USE_UPCXX)

        //------------------------------------------------------------------------------//
        //  Function executed on remote node
        //
        auto remote_serialize = [=]() { return send_serialize(obj); };

        //------------------------------------------------------------------------------//
        //  Combine on master rank
        //
        if(dmp_rank == 0)
        {
            for(int i = 1; i < dmp_size; ++i)
            {
                upcxx::future<std::string> fut = upcxx::rpc(i, remote_serialize);
                while(!fut.ready())
                    upcxx::progress();
                fut.wait();
                results[i] = recv_serialize(fut.result());
            }
            results[dmp_rank] = std::move(obj);
        }

#endif
    }
    else
    {
        results.clear();
        results.resize(1);
        results.at(0) = std::move(obj);
    }

    if(dmp_rank == 0)
    {
        fname = settings::compose_output_filename(fname, ".json");
        if(settings::verbose() >= 0)
            TIMEMORY_PRINTF_SOURCE(stderr, "[%s][ert]|%i> Outputting '%s'...\n",
                                   TIMEMORY_PROJECT_NAME, dmp_rank, fname.c_str());
        std::ofstream ofs{};
        if(filepath::open(ofs, fname))
        {
            // ensure json write final block during destruction before the file is closed
            using policy_type = policy::output_archive_t<Counter>;
            auto oa           = policy_type::get(ofs);
            oa->setNextName("timemory");
            oa->startNode();
            oa->setNextName("ranks");
            oa->startNode();
            oa->makeArray();
            for(uint64_t i = 0; i < results.size(); ++i)
            {
                oa->startNode();
                (*oa)(cereal::make_nvp("rank", i),
                      cereal::make_nvp("roofline", results.at(i)));
                oa->finishNode();
            }
            oa->finishNode();
            oa->finishNode();
        }
        if(ofs)
            ofs << std::endl;
        ofs.close();
    }
}

//--------------------------------------------------------------------------------------//

}  // namespace ert
}  // namespace tim
