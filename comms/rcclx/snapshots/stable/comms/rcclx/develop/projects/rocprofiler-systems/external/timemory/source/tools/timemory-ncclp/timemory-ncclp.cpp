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

#include "timemory/components/gotcha/ncclp.hpp"
#include "timemory/library.h"
#include "timemory/timemory.hpp"

#include <dlfcn.h>
#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;

TIMEMORY_DECLARE_COMPONENT(nccl_comm_data)
//
//--------------------------------------------------------------------------------------//
//
struct ncclp_tag
{};
//
struct nccl_data_tag
{};
//
using nccl_data_tracker_t = data_tracker<float, nccl_data_tag>;
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(nccl_data_tracker_t, float)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, nccl_data_tracker_t, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, nccl_data_tracker_t, true_type)
//
//--------------------------------------------------------------------------------------//
//
using api_t = ncclp_tag;
using nccl_toolset_t =
    tim::component_tuple<user_ncclp_bundle, nccl_data_tracker_t, nccl_comm_data>;
using ncclp_handle_t           = ncclp_handle<nccl_toolset_t, api_t>;
static uint64_t global_id      = 0;
static void*    libnccl_handle = nullptr;
//
//--------------------------------------------------------------------------------------//
//
extern "C"
{
    void timemory_ncclp_library_ctor()
    {
        nccl_data_tracker_t::label()       = "nccl_comm_data";
        nccl_data_tracker_t::description() = "Tracks NCCL communication data";
    }

    uint64_t timemory_start_ncclp()
    {
        // provide environment variable for enabling/disabling
        if(tim::get_env<bool>("TIMEMORY_ENABLE_NCCLP", true))
        {
            // make sure the symbols are loaded to be wrapped
            auto libpath =
                tim::get_env<std::string>("TIMEMORY_NCCL_LIBRARY", "libnccl.so");
            libnccl_handle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if(!libnccl_handle)
                fprintf(stderr, "%s\n", dlerror());
            dlerror();  // Clear any existing error

            configure_ncclp<nccl_toolset_t, api_t>();
            tim::operation::init<user_ncclp_bundle>(
                tim::operation::mode_constant<tim::operation::init_mode::global>{});
            auto ret = activate_ncclp<nccl_toolset_t, api_t>();
            if(libnccl_handle)
                dlclose(libnccl_handle);
            return ret;
        }
        else
        {
            return 0;
        }
    }

    uint64_t timemory_stop_ncclp(uint64_t id)
    {
        return deactivate_ncclp<nccl_toolset_t, api_t>(id);
    }

    void timemory_register_ncclp() { global_id = timemory_start_ncclp(); }
    void timemory_deregister_ncclp() { global_id = timemory_stop_ncclp(global_id); }

    // Below are for FORTRAN codes
    void     timemory_ncclp_library_ctor_() {}
    uint64_t timemory_start_ncclp_() { return timemory_start_ncclp(); }
    uint64_t timemory_stop_ncclp_(uint64_t id) { return timemory_stop_ncclp(id); }
    void     timemory_register_ncclp_() { timemory_register_ncclp(); }
    void     timemory_deregister_ncclp_() { timemory_deregister_ncclp(); }

}  // extern "C"
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
struct nccl_comm_data : base<nccl_comm_data, void>
{
    using value_type = void;
    using this_type  = nccl_comm_data;
    using base_type  = base<this_type, value_type>;
    using tracker_t  = tim::auto_bundle<TIMEMORY_API, nccl_data_tracker_t*>;
    using data_type  = float;

    TIMEMORY_DEFAULT_OBJECT(nccl_comm_data)

    static void preinit() { timemory_ncclp_library_ctor(); }

    static void global_init()
    {
        auto _data = tim::get_env("TIMEMORY_NCCLP_COMM_DATA", true);
        if(_data)
            tracker_t::get_initializer() = [](tracker_t& cb) {
                cb.initialize<nccl_data_tracker_t>();
            };
    }

    void start() {}
    void stop() {}

    auto NCCL_Type_size(ncclDataType_t datatype)
    {
        switch(datatype)
        {
            case ncclInt8:
            case ncclUint8: return 1;
            case ncclFloat16: return 2;
            case ncclInt32:
            case ncclUint32:
            case ncclFloat32: return 4;
            case ncclInt64:
            case ncclUint64:
            case ncclFloat64: return 8;
            default: return 0;
        };
    }

    // ncclReduce
    void audit(const std::string& _name, const void*, void*, size_t count,
               ncclDataType_t datatype, ncclRedOp_t, int root, ncclComm_t, cudaStream_t)
    {
        int size = NCCL_Type_size(datatype);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", root));
    }

    // ncclSend
    void audit(const std::string& _name, const void*, size_t count,
               ncclDataType_t datatype, int peer, ncclComm_t, cudaStream_t)
    {
        int size = NCCL_Type_size(datatype);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", peer));
    }

    // ncclBcast
    // ncclRecv
    void audit(const std::string& _name, void*, size_t count, ncclDataType_t datatype,
               int root, ncclComm_t, cudaStream_t)
    {
        int size = NCCL_Type_size(datatype);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", root));
    }

    // ncclBroadcast
    void audit(const std::string& _name, const void*, void*, size_t count,
               ncclDataType_t datatype, int root, ncclComm_t, cudaStream_t)
    {
        int size = NCCL_Type_size(datatype);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", root));
    }

    // ncclAllReduce
    // ncclReduceScatter
    void audit(const std::string& _name, const void*, void*, size_t count,
               ncclDataType_t datatype, ncclRedOp_t, ncclComm_t, cudaStream_t)
    {
        int size = NCCL_Type_size(datatype);
        add(_name, count * size);
    }

    // ncclAllGather
    void audit(const std::string& _name, const void*, void*, size_t count,
               ncclDataType_t datatype, ncclComm_t, cudaStream_t)
    {
        int size = NCCL_Type_size(datatype);
        add(_name, count * size);
    }

private:
    template <typename... Args>
    void add(tracker_t& _t, data_type value, Args&&... args)
    {
        _t.store(std::plus<data_type>{}, value);
        TIMEMORY_FOLD_EXPRESSION(add_secondary(_t, std::forward<Args>(args), value));
    }

    template <typename... Args>
    void add(const std::string& _name, data_type value, Args&&... args)
    {
        tracker_t _t(_name);
        add(_t, value, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void add_secondary(tracker_t&, const std::string& _name, data_type value,
                       Args&&... args)
    {
        if(tim::settings::add_secondary())
        {
            tracker_t _s(_name);
            add(_s, value, std::forward<Args>(args)...);
        }
    }
};
}  // namespace component
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_INITIALIZE_STORAGE(nccl_comm_data, nccl_data_tracker_t)
//
//--------------------------------------------------------------------------------------//
