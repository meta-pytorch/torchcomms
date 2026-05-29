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
//

#include "timemory/components/papi/components.hpp"
#include "timemory/components/user_bundle/types.hpp"
#include "timemory/library.h"
#include "timemory/timemory.hpp"

using namespace tim::component;
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DECLARE_COMPONENT(inst_per_cycle)
TIMEMORY_STATISTICS_TYPE(component::inst_per_cycle, double)
#if !defined(TIMEMORY_USE_PAPI)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::inst_per_cycle, false_type)
#endif
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace component
{
struct inst_per_cycle : public base<inst_per_cycle, std::array<long long, 2>>
{
    using value_type = std::array<long long, 2>;
    using hw_t       = tim::component::papi_tuple<PAPI_TOT_INS, PAPI_TOT_CYC>;

    static std::string label() { return "inst_per_cycle"; }
    static std::string description() { return "number of instructions per cycle"; }
    static void        thread_init(storage_type*) { hw_t::thread_init(); }

    void start()
    {
        m_hw.start();
        value = m_hw.get_value();
    }
    void stop()
    {
        m_hw.stop();
        value = m_hw.get_value();
        accum = m_hw.get_accum();
    }
    double get() const { return (accum[1] > 0.0) ? (accum[0] / (1.0 * accum[1])) : 0.0; }

private:
    hw_t m_hw;
};
}  // namespace component
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
extern "C" void
timemory_register_custom_dynamic_instr()
{
    TIMEMORY_PRINT_HERE("%s", "Registering the custom dynamic instrumentation");

    using tim::operation::init;
    using tim::operation::init_mode;
    using tim::operation::mode_constant;

    tim::settings::papi_multiplexing() = true;

    // insert monotonic clock component into structure
    // used by timemory-run in --mode=trace
    user_trace_bundle::configure<monotonic_clock>();

    // insert monotonic clock component into structure
    // used by timemory-run in --mode=region
    timemory_add_components("monotonic_clock");

    // if PAPI enabled at compile-time and run-time
    using hw_t = typename inst_per_cycle::hw_t;
    if(tim::trait::runtime_enabled<hw_t>::get())
    {
        hw_t::configure();
        // used by timemory when --mode=trace
        user_trace_bundle::configure<inst_per_cycle>();
        // used by timemory when --mode=region
        user_global_bundle::configure<inst_per_cycle>();
    }
}
//
//--------------------------------------------------------------------------------------//
//
extern "C" void
timemory_deregister_custom_dynamic_instr()
{
    TIMEMORY_PRINT_HERE("%s", "Deregistering the custom dynamic instrumentation");
}
//
//--------------------------------------------------------------------------------------//
//
