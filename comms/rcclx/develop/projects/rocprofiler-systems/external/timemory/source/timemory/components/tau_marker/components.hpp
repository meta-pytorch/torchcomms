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

/**
 * \file timemory/components/tau_marker/components.hpp
 * \brief Implementation of the tau_marker component(s)
 */

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/tau_marker/macros.hpp"
#include "timemory/components/tau_marker/types.hpp"
#include "timemory/components/types.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/mpl/concepts.hpp"

#include <type_traits>

//======================================================================================//
//
namespace tim
{
namespace component
{
/// \struct tim::component::tau_marker
/// \brief Forwards timemory labels to the TAU (Tuning and Analysis Utilities)
//
struct tau_marker : public base<tau_marker, void>
{
    // timemory component api
    using value_type = void;
    using this_type  = tau_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "tau"; }
    static std::string description()
    {
        return "Forwards markers to TAU instrumentation (via Tau_start and Tau_stop)";
    }

    TIMEMORY_DEFAULT_OBJECT(tau_marker)

    static void global_init();
    static void thread_init();

    static void start(const char* _prefix);
    static void stop(const char* _prefix);

    tau_marker(const char* _prefix)
    : m_prefix(_prefix)
    {}

    void start() { tau_marker::start(m_prefix); }
    void stop() { tau_marker::stop(m_prefix); }
    void set_prefix(const char* _prefix) { m_prefix = _prefix; }

    template <typename Tp>
    static auto start(Tp&& _v, enable_if_t<concepts::is_string_type<Tp>::value, int> = 0)
        -> decltype(tau_marker::start(std::forward<Tp>(_v).data()))
    {
        return tau_marker::start(const_cast<const char*>(std::forward<Tp>(_v).data()));
    }

    template <typename Tp, enable_if_t<concepts::is_string_type<Tp>::value, int> = 0>
    static auto stop(Tp&& _v, enable_if_t<concepts::is_string_type<Tp>::value, int> = 0)
        -> decltype(tau_marker::stop(std::forward<Tp>(_v).data()))
    {
        return tau_marker::stop(const_cast<const char*>(std::forward<Tp>(_v).data()));
    }

private:
    const char* m_prefix = nullptr;
};
//
}  // namespace component
}  // namespace tim

#if defined(TIMEMORY_TAU_MARKER_COMPONENT_HEADER_MODE) &&                                \
    TIMEMORY_TAU_MARKER_COMPONENT_HEADER_MODE > 0
#    include "timemory/components/tau_marker/components.cpp"
#endif
