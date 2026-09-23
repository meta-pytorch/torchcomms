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

#include "timemory/components/gotcha/mpip.hpp"
#include "timemory/library.h"
#include "timemory/timemory.hpp"

#include <dlfcn.h>
#include <memory>
#include <set>
#include <unordered_map>

using namespace tim::component;

TIMEMORY_DECLARE_COMPONENT(mpi_comm_data)
//
//--------------------------------------------------------------------------------------//
//
struct mpi_data_tag
{};
//
using mpi_data_tracker_t = data_tracker<float, mpi_data_tag>;
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_STATISTICS_TYPE(mpi_data_tracker_t, float)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_memory_units, mpi_data_tracker_t, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_memory_category, mpi_data_tracker_t, true_type)
//
//--------------------------------------------------------------------------------------//
//
using api_t                   = TIMEMORY_API;
using mpi_toolset_t           = tim::component_tuple<user_mpip_bundle, mpi_comm_data>;
using mpip_handle_t           = mpip_handle<mpi_toolset_t, api_t>;
static uint64_t global_id     = 0;
static void*    libmpi_handle = nullptr;
//
//--------------------------------------------------------------------------------------//
//
extern "C"
{
    void timemory_mpip_library_ctor()
    {
        mpi_data_tracker_t::label()       = "mpi_comm_data";
        mpi_data_tracker_t::description() = "Tracks MPI communication data";
    }

    uint64_t timemory_start_mpip()
    {
        // provide environment variable for enabling/disabling
        if(tim::get_env<bool>("TIMEMORY_ENABLE_MPIP", true))
        {
            // make sure the symbols are loaded to be wrapped
            auto libpath = tim::get_env<std::string>("TIMEMORY_MPI_LIBRARY", "libmpi.so");
            libmpi_handle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_GLOBAL);
            if(!libmpi_handle)
                fprintf(stderr, "%s\n", dlerror());
            dlerror();  // Clear any existing error

            configure_mpip<mpi_toolset_t, api_t>();
            tim::operation::init<user_mpip_bundle>(
                tim::operation::mode_constant<tim::operation::init_mode::global>{});
            auto ret = activate_mpip<mpi_toolset_t, api_t>();
            if(libmpi_handle)
                dlclose(libmpi_handle);
            return ret;
        }
        else
        {
            return 0;
        }
    }

    uint64_t timemory_stop_mpip(uint64_t id)
    {
        return deactivate_mpip<mpi_toolset_t, api_t>(id);
    }

    void timemory_register_mpip() { global_id = timemory_start_mpip(); }
    void timemory_deregister_mpip() { global_id = timemory_stop_mpip(global_id); }

    // Below are for FORTRAN codes
    void     timemory_mpip_library_ctor_() {}
    uint64_t timemory_start_mpip_() { return timemory_start_mpip(); }
    uint64_t timemory_stop_mpip_(uint64_t id) { return timemory_stop_mpip(id); }
    void     timemory_register_mpip_() { timemory_register_mpip(); }
    void     timemory_deregister_mpip_() { timemory_deregister_mpip(); }

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
struct mpi_comm_data : base<mpi_comm_data, void>
{
    using value_type = void;
    using this_type  = mpi_comm_data;
    using base_type  = base<this_type, value_type>;
    using tracker_t  = tim::auto_bundle<TIMEMORY_API, mpi_data_tracker_t*>;
    using data_type  = float;

    TIMEMORY_DEFAULT_OBJECT(mpi_comm_data)

    static void preinit() { timemory_mpip_library_ctor(); }

    static void global_init()
    {
        auto _data = tim::get_env("TIMEMORY_MPIP_COMM_DATA", true);
        if(_data)
            tracker_t::get_initializer() = [](tracker_t& cb) {
                cb.initialize<mpi_data_tracker_t>();
            };
    }

    void start() {}
    void stop() {}

    // MPI_Send
    void audit(const std::string& _name, const void*, int count, MPI_Datatype datatype,
               int dst, int tag, MPI_Comm)
    {
        int size = 0;
        MPI_Type_size(datatype, &size);
        tracker_t _t(_name);
        add(_t, count * size);
        add_secondary(_t, TIMEMORY_JOIN('_', _name, "dst", dst), count * size,
                      TIMEMORY_JOIN('_', _name, "dst", dst, "tag", tag));
    }

    // MPI_Recv
    void audit(const std::string& _name, void*, int count, MPI_Datatype datatype, int dst,
               int tag, MPI_Comm, MPI_Status*)
    {
        int size = 0;
        MPI_Type_size(datatype, &size);
        tracker_t _t(_name);
        add(_t, count * size);
        add_secondary(_t, TIMEMORY_JOIN('_', _name, "dst", dst), count * size,
                      TIMEMORY_JOIN('_', _name, "dst", dst, "tag", tag));
    }

    // MPI_Isend
    void audit(const std::string& _name, const void*, int count, MPI_Datatype datatype,
               int dst, int tag, MPI_Comm, MPI_Request*)
    {
        int size = 0;
        MPI_Type_size(datatype, &size);
        tracker_t _t(_name);
        add(_t, count * size);
        add_secondary(_t, TIMEMORY_JOIN('_', _name, "dst", dst), count * size,
                      TIMEMORY_JOIN('_', _name, "dst", dst, "tag", tag));
    }

    // MPI_Irecv
    void audit(const std::string& _name, void*, int count, MPI_Datatype datatype, int dst,
               int tag, MPI_Comm, MPI_Request*)
    {
        int size = 0;
        MPI_Type_size(datatype, &size);
        tracker_t _t(_name);
        add(_t, count * size);
        add_secondary(_t, TIMEMORY_JOIN('_', _name, "dst", dst), count * size,
                      TIMEMORY_JOIN('_', _name, "dst", dst, "tag", tag));
    }

    // MPI_Bcast
    void audit(const std::string& _name, void*, int count, MPI_Datatype datatype,
               int root, MPI_Comm)
    {
        int size = 0;
        MPI_Type_size(datatype, &size);
        add(_name, count * size, TIMEMORY_JOIN('_', _name, "root", root));
    }

    // MPI_Allreduce
    void audit(const std::string& _name, const void*, void*, int count,
               MPI_Datatype datatype, MPI_Op, MPI_Comm)
    {
        int size = 0;
        MPI_Type_size(datatype, &size);
        add(_name, count * size);
    }

    // MPI_Sendrecv
    void audit(const std::string& _name, const void*, int sendcount,
               MPI_Datatype sendtype, int, int sendtag, void*, int recvcount,
               MPI_Datatype recvtype, int, int recvtag, MPI_Comm, MPI_Status*)
    {
        int send_size = 0;
        int recv_size = 0;
        MPI_Type_size(sendtype, &send_size);
        MPI_Type_size(recvtype, &recv_size);
        tracker_t _t(_name);
        add(_t, sendcount * send_size + recvcount * recv_size);
        add_secondary(_t, TIMEMORY_JOIN('_', _name, "send"), sendcount * send_size,
                      TIMEMORY_JOIN('_', _name, "send", "tag", sendtag));
        add_secondary(_t, TIMEMORY_JOIN('_', _name, "recv"), recvcount * recv_size,
                      TIMEMORY_JOIN('_', _name, "recv", "tag", recvtag));
    }

    // MPI_Gather
    void audit(const std::string& _name, const void*, int sendcount,
               MPI_Datatype sendtype, void*, int recvcount, MPI_Datatype recvtype,
               int root, MPI_Comm)
    {
        int send_size = 0;
        int recv_size = 0;
        MPI_Type_size(sendtype, &send_size);
        MPI_Type_size(recvtype, &recv_size);
        tracker_t _t(_name);
        add(_t, sendcount * send_size + recvcount * recv_size);
        tracker_t _r(TIMEMORY_JOIN('_', _name, "root", root));
        add(_r, sendcount * send_size + recvcount * recv_size);
        add_secondary(_r, TIMEMORY_JOIN('_', _name, "root", root, "send"),
                      sendcount * send_size);
        add_secondary(_r, TIMEMORY_JOIN('_', _name, "root", root, "recv"),
                      recvcount * recv_size);
    }

    // MPI_Scatter
    void audit(const std::string& _name, void*, int sendcount, MPI_Datatype sendtype,
               void*, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm)
    {
        int send_size = 0;
        int recv_size = 0;
        MPI_Type_size(sendtype, &send_size);
        MPI_Type_size(recvtype, &recv_size);
        tracker_t _t(_name);
        add(_t, sendcount * send_size + recvcount * recv_size);
        tracker_t _r(TIMEMORY_JOIN('_', _name, "root", root));
        add(_r, sendcount * send_size + recvcount * recv_size);
        add_secondary(_r, TIMEMORY_JOIN('_', _name, "root", root, "send"),
                      sendcount * send_size);
        add_secondary(_r, TIMEMORY_JOIN('_', _name, "root", root, "recv"),
                      recvcount * recv_size);
    }

    // MPI_Alltoall
    void audit(const std::string& _name, void*, int sendcount, MPI_Datatype sendtype,
               void*, int recvcount, MPI_Datatype recvtype, MPI_Comm)
    {
        int send_size = 0;
        int recv_size = 0;
        MPI_Type_size(sendtype, &send_size);
        MPI_Type_size(recvtype, &recv_size);
        tracker_t _t(_name);
        add(_t, sendcount * send_size + recvcount * recv_size);
        add_secondary(_t, TIMEMORY_JOIN('_', _name, "send"), sendcount * send_size);
        add_secondary(_t, TIMEMORY_JOIN('_', _name, "recv"), recvcount * recv_size);
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
TIMEMORY_INITIALIZE_STORAGE(mpi_comm_data, mpi_data_tracker_t)
//
//--------------------------------------------------------------------------------------//
