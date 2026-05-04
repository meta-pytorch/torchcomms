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

#include "caliper/ConfigManager.h"
#include "caliper/cali.h"
#include "timemory/api.hpp"
#include "timemory/components/base.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/settings/declaration.hpp"

#if defined(TIMEMORY_PYBIND11_SOURCE)
#    include "pybind11/cast.h"
#    include "pybind11/pybind11.h"
#    include "pybind11/stl.h"
#endif

#include <map>
#include <vector>

//======================================================================================//
//
namespace tim
{
namespace trait
{
//
//--------------------------------------------------------------------------------------//
//
template <>
struct python_args<TIMEMORY_RECORD, component::caliper_loop_marker>
{
    using type = type_list<size_t>;
};
//
template <>
struct python_args<TIMEMORY_MARK_BEGIN, component::caliper_loop_marker>
{
    using type = type_list<size_t>;
};
//
template <>
struct python_args<TIMEMORY_MARK_END, component::caliper_loop_marker>
{
    using type = type_list<size_t>;
};
}  // namespace trait
//
//--------------------------------------------------------------------------------------//
//
namespace component
{
struct caliper_common
{
    using attributes_t = int;

    static void _init()
    {
        if(tim::settings::debug() || tim::settings::verbose() > 0)
            puts("Initializing caliper...");
        cali_init();
    }

    static void init()
    {
        static bool _ini = (_init(), true);
        consume_parameters(_ini);
    }

    static std::string& get_channel() { return get_persistent_data().channel; }

    static attributes_t& get_attributes() { return get_persistent_data().attributes; }

    static attributes_t get_default_attributes()
    {
        return (get_nested() | CALI_ATTR_SCOPE_THREAD);
    }

    static attributes_t get_nested()
    {
        return (tim::settings::flat_profile()) ? 0 : CALI_ATTR_NESTED;
    }

    static auto get_process_scope()
    {
        return attributes_t(get_nested() | CALI_ATTR_SCOPE_PROCESS);
    }

    static auto get_thread_scope()
    {
        return attributes_t(get_nested() | CALI_ATTR_SCOPE_THREAD);
    }

    static auto get_task_scope()
    {
        return attributes_t(get_nested() | CALI_ATTR_SCOPE_TASK);
    }

    static void enable_process_scope() { get_attributes() = get_process_scope(); }
    static void enable_thread_scope() { get_attributes() = get_thread_scope(); }
    static void enable_task_scope() { get_attributes() = get_task_scope(); }

    static std::string label() { return "caliper"; }
    static std::string description()
    {
        return "Forwards markers to Caliper instrumentation";
    }

    // when inherited types are constructed
    caliper_common() { init(); }

private:
    struct persistent_data
    {
        std::string  channel    = "timemory";
        attributes_t attributes = (get_nested() | CALI_ATTR_SCOPE_THREAD);
    };

    static persistent_data& get_persistent_data()
    {
        static auto _instance = persistent_data{};
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::component::caliper_config
/// \brief Component which provides Caliper `cali::ConfigManager`.
///
struct caliper_config
: public base<caliper_config, void>
, private policy::instance_tracker<caliper_config, false>
{
    using value_type         = void;
    using instance_tracker_t = policy::instance_tracker<caliper_config, false>;
    using arg_map_t          = std::map<std::string, std::string>;
    using arg_vec_t          = std::vector<std::string>;

    static std::string label() { return "caliper_config"; }
    static std::string description() { return "Caliper configuration manager"; }

    static auto& get_manager()
    {
        static cali::ConfigManager _instance;
        return _instance;
    }

    static void configure(const arg_vec_t& _args, const arg_map_t& _kwargs = {})
    {
        std::string cmd{};
        {
            std::stringstream ss;
            for(auto& itr : _args)
                ss << "," << itr;
            if(!ss.str().empty())
                cmd = ss.str().substr(1);
        }
        {
            std::stringstream ss;
            for(auto& itr : _kwargs)
            {
                auto _arg = itr.second;
                if(_arg.empty())
                {
                    ss << "," << itr.first;
                }
                else
                {
                    ss << "," << itr.first << "=(" << _arg << ")";
                }
            }
            if(!ss.str().empty())
            {
                if(!cmd.empty())
                    cmd += ",";
                cmd += ss.str().substr(1);
            }
        }
        if(!cmd.empty())
        {
            if(tim::settings::debug() || tim::settings::verbose() > -1)
                std::cerr << "Configuring caliper with :: " << cmd << std::endl;
            get_manager().add(cmd.c_str());
            if(get_manager().error())
            {
                std::cerr << "Caliper config error: " << get_manager().error_msg()
                          << std::endl;
            }
        }
        else
        {
            if(tim::settings::debug() || tim::settings::verbose() > -1)
                std::cerr << "Caliper was not configured" << std::endl;
        }
    }

#if defined(TIMEMORY_PYBIND11_SOURCE)
    //
    /// this is called by python api
    ///
    ///     args --> pybind11::args --> pybind11::tuple
    ///     kwargs --> pybind11::kwargs --> pybind11::dict
    ///
    static void configure(project::python, pybind11::args _args, pybind11::kwargs _kwargs)
    {
        std::string cmd = "";
        {
            std::stringstream ss;
            for(auto& itr : _args)
                ss << "," << itr.cast<std::string>();
            if(!ss.str().empty())
                cmd = ss.str().substr(1);
        }
        {
            std::stringstream ss;
            for(auto& itr : _kwargs)
            {
                auto _arg = itr.second.cast<std::string>();
                if(_arg.empty())
                    ss << "," << itr.first.cast<std::string>();
                else
                {
                    ss << "," << itr.first.cast<std::string>() << "=(" << _arg << ")";
                }
            }
            if(!ss.str().empty())
            {
                if(!cmd.empty())
                    cmd += ",";
                cmd += ss.str().substr(1);
            }
        }
        if(!cmd.empty())
        {
            if(tim::settings::debug() || tim::settings::verbose() > -1)
                std::cerr << "Configuring caliper with :: " << cmd << std::endl;
            get_manager().add(cmd.c_str());
            if(get_manager().error())
                std::cerr << "Caliper config error: " << get_manager().error_msg()
                          << std::endl;
        }
        else
        {
            if(tim::settings::debug() || tim::settings::verbose() > -1)
                std::cerr << "Caliper was not configured" << std::endl;
        }
    }
#endif

    void start()
    {
        auto cnt = instance_tracker_t::start();
        if(cnt == 0)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(tim::settings::debug(), "%s",
                                            "Starting Caliper ConfigManager");
            get_manager().start();
        }
    }

    void stop()
    {
        auto cnt = instance_tracker_t::stop();
        if(cnt == 0)
        {
            TIMEMORY_CONDITIONAL_PRINT_HERE(tim::settings::debug(), "%s",
                                            "Flushing Caliper ConfigManager");
            get_manager().flush();
        }
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::component::caliper_marker
/// \brief Standard marker for the Caliper Performance Analysis Toolbox
///
struct caliper_marker
: public base<caliper_marker, void>
, public caliper_common
{
    // timemory component api
    using value_type = void;
    using this_type  = caliper_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "caliper_marker"; }
    static std::string description()
    {
        return "Generic forwarding of markers to Caliper instrumentation";
    }

    TIMEMORY_DEFAULT_OBJECT(caliper_marker)

    void start()
    {
        if(m_prefix == nullptr)
            return;
        TIMEMORY_DEBUG_PRINT_HERE("'%s'", m_prefix);
        cali_begin_string(m_id, m_prefix);
    }
    void stop()
    {
        if(m_prefix == nullptr)
            return;
        TIMEMORY_DEBUG_PRINT_HERE("'%s'", m_prefix);
        cali_safe_end_string(m_id, m_prefix);
    }

    void set_prefix(const char* _prefix) { m_prefix = _prefix; }

private:
    const char* m_prefix = nullptr;
    cali_id_t   m_id     = cali_create_attribute("timemory", CALI_TYPE_STRING,
                                           caliper_common::get_attributes());

public:
    // emulate CALI_MARK_BEGIN and CALI_MARK_END via static calls which require a string
    static void start(const std::string& _name) { CALI_MARK_BEGIN(_name.c_str()); }
    static void stop(const std::string& _name) { CALI_MARK_END(_name.c_str()); }

#if defined(TIMEMORY_PYBIND11_SOURCE)
    //
    /// this is called by python api
    ///
    ///     Use this to add customizations to the python module. The instance
    ///     of the component is within in a variadic wrapper which is used
    ///     elsewhere to ensure that calling mark_begin(...) on a component
    ///     without that member function is not invalid
    ///
    template <template <typename...> class BundleT>
    static void configure(project::python,
                          pybind11::class_<BundleT<caliper_marker>>& _pyclass)
    {
        // define two lambdas to pass to pybind11 instead of &caliper_marker
        // and &caliper_marker::stop because it would be ambiguous
        auto _bregion = [](const std::string& _name) { caliper_marker::start(_name); };
        auto _eregion = [](const std::string& _name) { caliper_marker::stop(_name); };
        // define these as static member functions so that they can be called
        // without object
        _pyclass.def_static("begin_region", _bregion, "Begin a user-defined region");
        _pyclass.def_static("end_region", _eregion, "End a user-defined region");
        //
        // add CALI_ATTRIBUTES
        pybind11::enum_<cali_attr_properties> _pyattr(
            _pyclass, "Attribute", pybind11::arithmetic(), "Attributes");
        _pyattr.value("Nested", CALI_ATTR_NESTED)
            .value("ThreadScope", CALI_ATTR_SCOPE_THREAD)
            .value("ProcessScope", CALI_ATTR_SCOPE_PROCESS)
            .value("TaskScope", CALI_ATTR_SCOPE_TASK);
        // no need to add bindings to default start/stop, those are already added
    }
#endif
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::component::caliper_loop_marker
/// \brief Loop marker for the Caliper Performance Analysis Toolbox
///
struct caliper_loop_marker
: public base<caliper_loop_marker, void>
, public caliper_common
{
    // timemory component api
    using value_type = void;
    using this_type  = caliper_loop_marker;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "caliper_loop_marker"; }
    static std::string description()
    {
        return "Variant of caliper_marker with support for loop marking";
    }

    TIMEMORY_DEFAULT_OBJECT(caliper_loop_marker)

    void start()
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s", m_prefix);
        cali_begin_string(m_id, m_prefix);
        m_id  = cali_make_loop_iteration_attribute(m_prefix);
        m_itr = 0;
    }
    void stop() { cali_end(m_id); }

    void mark_begin()
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s", m_prefix);
        cali_begin_int(m_id, m_itr++);
    }
    void mark_end()
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s", m_prefix);
        cali_end(m_id);
    }

    template <typename T, enable_if_t<std::is_integral<T>::value, int> = 0>
    void mark_begin(T itr)
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s @ %i", m_prefix, (int) itr);
        m_itr = itr;
        cali_begin_int(m_id, m_itr++);
    }

    template <typename T, enable_if_t<std::is_integral<T>::value, int> = 0>
    void mark_end(T)
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s @ %i", m_prefix, (int) m_itr);
        cali_end(m_id);
    }

    template <typename T, enable_if_t<std::is_integral<T>::value, int> = 0>
    tim::scope::destructor record(T itr)
    {
        TIMEMORY_DEBUG_PRINT_HERE("%s @ %i", m_prefix, (int) itr);
        m_itr = itr;
        cali_begin_int(m_id, m_itr++);
        return tim::scope::destructor([=]() { cali_end(m_id); });
    }

    void set_prefix(const char* _prefix) { m_prefix = _prefix; }

#if defined(TIMEMORY_PYBIND11_SOURCE)
    template <template <typename...> class BundleT>
    static void configure(project::python, pybind11::class_<BundleT<caliper_marker>>&)
    {}
#endif

private:
    size_t      m_itr    = 0;
    cali_id_t   m_id     = cali_create_attribute("timemory", CALI_TYPE_STRING,
                                           caliper_common::get_attributes());
    const char* m_prefix = nullptr;
};
//
}  // namespace component
}  // namespace tim
//
//======================================================================================//
