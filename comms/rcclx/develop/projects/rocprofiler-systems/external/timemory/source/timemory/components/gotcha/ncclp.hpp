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

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/gotcha/backends.hpp"
#include "timemory/components/gotcha/types.hpp"
#include "timemory/manager/declaration.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/units.hpp"
#include "timemory/variadic/types.hpp"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#if defined(TIMEMORY_USE_NCCL)
#    include <nccl.h>
#endif

#if !defined(NUM_TIMEMORY_NCCLP_WRAPPERS)
#    define NUM_TIMEMORY_NCCLP_WRAPPERS 15
#endif

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Toolset, typename Tag>
void
configure_ncclp(std::set<std::string> permit = {}, std::set<std::string> reject = {});
//
//--------------------------------------------------------------------------------------//
//
template <typename Toolset, typename Tag>
static uint64_t
activate_ncclp();
//
//--------------------------------------------------------------------------------------//
//
template <typename Toolset, typename Tag>
static uint64_t deactivate_ncclp(uint64_t);
//
//--------------------------------------------------------------------------------------//
//
template <typename Toolset, typename Tag>
struct ncclp_handle : base<ncclp_handle<Toolset, Tag>, void>
{
    static constexpr size_t ncclp_wrapper_count = NUM_TIMEMORY_NCCLP_WRAPPERS;

    using value_type = void;
    using this_type  = ncclp_handle<Toolset, Tag>;
    using base_type  = base<this_type, value_type>;

    using string_t       = std::string;
    using nccl_toolset_t = Toolset;
    using ncclp_gotcha_t =
        tim::component::gotcha<ncclp_wrapper_count, nccl_toolset_t, Tag>;
    using ncclp_tuple_t = tim::component_tuple<ncclp_gotcha_t>;
    using toolset_ptr_t = std::shared_ptr<ncclp_tuple_t>;

    static string_t label() { return "ncclp_handle"; }
    static string_t description() { return "Handle for activating NCCL wrappers"; }

    void get() {}

    void start()
    {
        if(get_tool_count()++ == 0)
        {
            get_tool_instance() = std::make_shared<ncclp_tuple_t>("timemory_ncclp");
            get_tool_instance()->start();
        }
    }

    void stop()
    {
        auto idx = --get_tool_count();
        if(get_tool_instance().get())
        {
            get_tool_instance()->stop();
            if(idx == 0)
                get_tool_instance().reset();
        }
    }

    int get_count() { return get_tool_count().load(); }

private:
    struct persistent_data
    {
        std::atomic<short>   m_configured;
        std::atomic<int64_t> m_count;
        toolset_ptr_t        m_tool;
    };

    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }

    static std::atomic<short>& get_configured()
    {
        return get_persistent_data().m_configured;
    }

    static toolset_ptr_t& get_tool_instance() { return get_persistent_data().m_tool; }

    static std::atomic<int64_t>& get_tool_count()
    {
        return get_persistent_data().m_count;
    }
};
//
//======================================================================================//
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
//
#include "timemory/timemory.hpp"
//
//======================================================================================//
//
/// \fn uint64_t tim::component::activate_ncclp()
/// \brief The thread that first activates ncclp will be the thread that turns it off.
/// Function returns the number of new ncclp handles
///
template <typename Toolset, typename Tag>
static uint64_t
tim::component::activate_ncclp()
{
    using handle_t = tim::component::ncclp_handle<Toolset, Tag>;

    static std::shared_ptr<handle_t> _handle;

    if(!_handle.get())
    {
        _handle = std::make_shared<handle_t>();
        _handle->start();

        auto cleanup_functor = [=]() {
            if(_handle)
            {
                _handle->stop();
                _handle.reset();
            }
        };

        std::stringstream ss;
        ss << "timemory-ncclp-" << typeid(Toolset).name() << "-" << typeid(Tag).name();
        tim::manager::instance()->add_cleanup(ss.str(), cleanup_functor);
        return 1;
    }
    return 0;
}
//
//======================================================================================//
//
/// \fn uint64_t tim::component::deactivate_ncclp(uint64_t id)
/// \brief The thread that created the initial ncclp handle will turn off. Returns
/// the number of handles active
///
template <typename Toolset, typename Tag>
static uint64_t
tim::component::deactivate_ncclp(uint64_t id)
{
    if(id > 0)
    {
        std::stringstream ss;
        ss << "timemory-ncclp-" << typeid(Toolset).name() << "-" << typeid(Tag).name();
        tim::manager::instance()->cleanup(ss.str());
        return 0;
    }
    return 1;
}
//
//======================================================================================//
//
#if !defined(TIMEMORY_USE_GOTCHA) || !defined(TIMEMORY_USE_NCCL)
//
template <typename Toolset, typename Tag>
void configure_ncclp(std::set<std::string>, std::set<std::string>)
{}
//
#else
//
template <typename Toolset, typename Tag>
void
tim::component::configure_ncclp(std::set<std::string> permit,
                                std::set<std::string> reject)
{
    static constexpr size_t ncclp_wrapper_count = NUM_TIMEMORY_NCCLP_WRAPPERS;

    using string_t       = std::string;
    using ncclp_gotcha_t = tim::component::gotcha<ncclp_wrapper_count, Toolset, Tag>;

    static bool is_initialized = false;
    if(!is_initialized)
    {
        // generate the gotcha wrappers
        ncclp_gotcha_t::get_initializer() = []() {
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 0, ncclReduce);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 1, ncclBcast);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 2, ncclBroadcast);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 3, ncclAllReduce);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 4, ncclReduceScatter);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 5, ncclAllGather);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 6, ncclCommCuDevice);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 7, ncclCommUserRank);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 8, ncclGroupStart);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 9, ncclGroupEnd);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 10, ncclSend);
            TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 11, ncclRecv);
            // TIMEMORY_C_GOTCHA(ncclp_gotcha_t, 12, ncclCommCount);
        };

        // provide environment variable for suppressing wrappers
        ncclp_gotcha_t::get_reject_list() = [reject]() {
            auto _reject = reject;
            // check environment
            auto reject_list = tim::get_env<string_t>("TIMEMORY_NCCLP_REJECT_LIST", "");
            // add environment setting
            for(const auto& itr : tim::delimit(reject_list))
                _reject.insert(itr);
            return _reject;
        };

        // provide environment variable for selecting wrappers
        ncclp_gotcha_t::get_permit_list() = [permit]() {
            auto _permit = permit;
            // check environment
            auto permit_list = tim::get_env<string_t>("TIMEMORY_NCCLP_PERMIT_LIST", "");
            // add environment setting
            for(const auto& itr : tim::delimit(permit_list))
                _permit.insert(itr);
            return _permit;
        };

        is_initialized = true;
    }
}
//
#endif
//
//======================================================================================//
//
