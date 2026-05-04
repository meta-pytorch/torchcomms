//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file cupti.hpp
 * \headerfile cupti_activity.hpp "timemory/cupti_activity.hpp"
 * Provides implementation of CUPTI routines.
 *
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/cupti/backends.hpp"
#include "timemory/components/cupti/types.hpp"
#include "timemory/settings/declaration.hpp"

#include <functional>
#include <numeric>
#include <string>
#include <vector>

//======================================================================================//

namespace tim
{
namespace component
{
//--------------------------------------------------------------------------------------//
//
//          CUPTI activity tracing component (high-precisin kernel timers)
//
//--------------------------------------------------------------------------------------//
/// \struct tim::component::cupti_activity
/// \brief CUPTI activity tracing component for high-precision kernel timing. For
/// low-precision kernel timing, use \ref tim::component::cuda_event component.
///
struct cupti_activity : public base<cupti_activity, intmax_t>
{
    // required aliases
    using value_type = intmax_t;
    using this_type  = cupti_activity;
    using base_type  = base<cupti_activity, value_type>;

    // component-specific aliases
    using ratio_t           = std::nano;
    using size_type         = std::size_t;
    using string_t          = std::string;
    using receiver_type     = cupti::activity::receiver;
    using kind_vector_type  = std::vector<cupti::activity_kind_t>;
    using get_initializer_t = std::function<kind_vector_type()>;
    using kernel_elapsed_t  = typename cupti::activity::receiver::named_elapsed_t;
    using kernel_names_t    = std::unordered_set<std::string>;

    static std::string label() { return "cupti_activity"; }
    static std::string description()
    {
        return "Wall-clock execution timing for the CUDA API";
    }

    //----------------------------------------------------------------------------------//

    static get_initializer_t& get_initializer()
    {
        static get_initializer_t _instance = []() -> kind_vector_type {
            std::vector<cupti::activity_kind_t> _kinds;
            auto                                lvl = settings::cupti_activity_level();

            /// look up integer codes in <timemory/backends/types/cupti.hpp>
            auto vec = delimit(settings::cupti_activity_kinds());
            for(const auto& itr : vec)
            {
                int iactivity = atoi(itr.c_str());
                if(iactivity > static_cast<int>(CUPTI_ACTIVITY_KIND_INVALID) &&
                   iactivity < static_cast<int>(CUPTI_ACTIVITY_KIND_COUNT))
                {
                    _kinds.push_back(static_cast<cupti::activity_kind_t>(iactivity));
                }
            }

            // if found settings in environment, use those
            if(!_kinds.empty())
            {
                return _kinds;
            }
            if(lvl == 0)
            {
                // general settings for kernels, runtime, overhead
                _kinds = { CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL };
            }
            else if(lvl == 1)
            {
                // general settings for kernels, runtime, memory, overhead
                _kinds = { { CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                             CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET } };
            }
            else if(lvl == 2)
            {
                // general settings for kernels, runtime, memory, overhead, and device
                _kinds = { { CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                             CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
                             CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DEVICE,
                             CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_OVERHEAD } };
            }
            else if(lvl > 2)
            {
                // general settings for kernels, runtime, memory, overhead, device,
                // stream, CDP kernels
                _kinds = { { CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
                             CUPTI_ACTIVITY_KIND_MEMCPY, CUPTI_ACTIVITY_KIND_MEMSET,
                             CUPTI_ACTIVITY_KIND_RUNTIME, CUPTI_ACTIVITY_KIND_DEVICE,
                             CUPTI_ACTIVITY_KIND_DRIVER, CUPTI_ACTIVITY_KIND_OVERHEAD,
                             CUPTI_ACTIVITY_KIND_MARKER, CUPTI_ACTIVITY_KIND_STREAM,
                             CUPTI_ACTIVITY_KIND_CDP_KERNEL } };
            }
            return _kinds;
        };
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static kind_vector_type get_kind_types()
    {
        static kind_vector_type _instance = get_initializer()();
        return _instance;
    }

    //----------------------------------------------------------------------------------//

    static void global_init()
    {
        static std::atomic<short> _once(0);
        if(_once++ > 0)
            return;
        cupti::activity::initialize_trace(get_kind_types());
        cupti::init_driver();
    }

    //----------------------------------------------------------------------------------//

    static void global_finalize() { cupti::activity::finalize_trace(get_kind_types()); }

    //----------------------------------------------------------------------------------//

    static value_type record() { return cupti::activity::get_receiver().get(); }

    //----------------------------------------------------------------------------------//

public:
    TIMEMORY_DEFAULT_OBJECT(cupti_activity)

    // make sure it is removed
    ~cupti_activity() { cupti::activity::get_receiver().remove(this); }

    //----------------------------------------------------------------------------------//
    // start
    //
    void start()
    {
        cupti::activity::start_trace(this, m_depth_change);
        value           = cupti::activity::get_receiver().get();
        m_kernels_index = cupti::activity::get_receiver().get_named_index();
    }

    //----------------------------------------------------------------------------------//

    void stop()
    {
        using namespace tim::component::operators;
        cupti::activity::stop_trace(this);
        auto tmp     = cupti::activity::get_receiver().get();
        auto kernels = cupti::activity::get_receiver().get_named(m_kernels_index, true);

        accum += (tmp - value);
        value = tmp;
        for(const auto& itr : kernels)
            m_kernels_accum[itr.first] += itr.second;
        m_kernels_value = std::move(kernels);
    }

    //----------------------------------------------------------------------------------//

    double get() const { return load() / static_cast<double>(base_type::get_unit()); }

    double get_display() const { return get(); }

    //----------------------------------------------------------------------------------//

    kernel_elapsed_t get_secondary() const { return m_kernels_accum; }

    void set_depth_change(bool v) { m_depth_change = v; }

private:
    bool             m_depth_change  = false;
    uint64_t         m_kernels_index = 0;
    kernel_elapsed_t m_kernels_value;
    kernel_elapsed_t m_kernels_accum;
};

}  // namespace component
}  // namespace tim
