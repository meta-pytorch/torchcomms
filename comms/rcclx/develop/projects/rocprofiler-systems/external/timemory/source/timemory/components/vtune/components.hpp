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

#pragma once

#include "timemory/components/base.hpp"
#include "timemory/components/vtune/backends.hpp"
#include "timemory/components/vtune/types.hpp"
#include "timemory/units.hpp"

namespace tim
{
namespace component
{
//
/// \struct tim::component::vtune_event
/// \brief Implements `__itt_event`
///
struct vtune_event : public base<vtune_event, void>
{
    using value_type = void;
    using this_type  = vtune_event;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "vtune_event"; }
    static std::string description()
    {
        return "Creates events for Intel profiler running on the application";
    }
    static value_type record() {}

    static void global_init() { ittnotify::pause(); }
    static void global_finalize() { ittnotify::pause(); }

    void start()
    {
        get_index()++;

        if(m_index == 0)
            ittnotify::resume();
        if(!m_created)
        {
            m_event   = ittnotify::create_event(m_prefix);
            m_created = true;
        }
        ittnotify::start_event(m_event);
    }

    void stop()
    {
        auto _index = --get_index();
        ittnotify::end_event(m_event);
        if(_index == 0)
            ittnotify::pause();
    }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }

protected:
    bool               m_created = false;
    std::string        m_prefix;
    int64_t            m_index = -1;
    ittnotify::event_t m_event;

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance(0);
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::component::vtune_frame
/// \brief Implements `__itt_domain`
///
struct vtune_frame : public base<vtune_frame, void>
{
    using value_type = void;
    using this_type  = vtune_frame;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "vtune_frame"; }
    static std::string description()
    {
        return "Creates frames for Intel profiler running on the application";
    }
    static value_type record() {}

    static void global_init() { ittnotify::pause(); }
    static void global_finalize() { ittnotify::pause(); }

    void start()
    {
        get_index()++;

        if(m_index == 0)
            ittnotify::resume();
        if(m_domain == nullptr)
            m_domain = ittnotify::create_domain(m_prefix);
        ittnotify::start_frame(m_domain);
    }

    void stop()
    {
        auto _index = --get_index();
        if(m_domain)
            ittnotify::end_frame(m_domain);
        if(_index == 0)
            ittnotify::pause();
    }

    void set_prefix(const std::string& _prefix) { m_prefix = _prefix; }

protected:
    std::string          m_prefix;
    int64_t              m_index  = -1;
    ittnotify::domain_t* m_domain = nullptr;

private:
    static std::atomic<int64_t>& get_index()
    {
        static std::atomic<int64_t> _instance(0);
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::component::vtune_profiler
/// \brief Implements `__itt_pause()` and `__itt_resume()` to control where the vtune
/// profiler is active
///
struct vtune_profiler
: public base<vtune_profiler, void>
, private policy::instance_tracker<vtune_profiler, false>
{
    using value_type   = void;
    using this_type    = vtune_profiler;
    using base_type    = base<this_type, value_type>;
    using tracker_type = policy::instance_tracker<vtune_profiler, false>;

    static std::string label() { return "vtune_profiler"; }
    static std::string description()
    {
        return "Control switch for Intel profiler running on the application";
    }
    static value_type record() {}

    static void global_init() { ittnotify::pause(); }
    static void global_finalize() { ittnotify::pause(); }

    using tracker_type::m_tot;

    void start()
    {
        tracker_type::start();

        if(m_tot == 0)
            ittnotify::resume();
    }

    void stop()
    {
        tracker_type::stop();

        if(m_tot == 0)
            ittnotify::pause();
    }
};
//
}  // namespace component
}  // namespace tim
