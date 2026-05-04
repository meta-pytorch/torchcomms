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

#if !defined(TIMEMORY_PYCOMPONENT_BUNDLE_SOURCE)
#    define TIMEMORY_PYCOMPONENT_BUNDLE_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/components/extern.hpp"
#include "timemory/containers/definition.hpp"
#include "timemory/containers/extern.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/properties.hpp"
#include "timemory/variadic/definition.hpp"

using namespace tim::component;

//======================================================================================//
//
namespace pycomponent_bundle
{
//
template <typename UserBundleT>
class pycomponent_bundle
{
public:
    using type               = UserBundleT;
    using component_bundle_t = tim::component_bundle<TIMEMORY_API, UserBundleT>;

public:
    pycomponent_bundle(component_bundle_t* _ptr = nullptr)
    : m_ptr(_ptr)
    {}

    ~pycomponent_bundle()
    {
        if(m_ptr)
            m_ptr->stop();
        delete m_ptr;
    }

    pycomponent_bundle& operator=(component_bundle_t* _ptr)
    {
        if(m_ptr)
            delete m_ptr;
        m_ptr = _ptr;
        return *this;
    }

    void start()
    {
        if(m_ptr)
            m_ptr->start();
    }

    void stop()
    {
        if(m_ptr)
            m_ptr->stop();
    }

    static size_t size() { return type::bundle_size(); }

    static void reset()
    {
        TIMEMORY_DEBUG_PRINT_HERE("size = %lu", (unsigned long) size());
        type::reset();
    }

    template <typename... ArgsT>
    static void configure(ArgsT&&... _args)
    {
        tim::configure<type>(std::forward<ArgsT>(_args)...);
    }

private:
    component_bundle_t* m_ptr = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
namespace init
{
template <typename UserBundleT>
pycomponent_bundle<UserBundleT>*
component_bundle(const std::string& func, const std::string& file, const int line,
                 py::list args)
{
    using component_bundle_t = tim::component_bundle<TIMEMORY_API, UserBundleT>;
    using settings           = tim::settings;
    using source_loc         = tim::source_location;
    using mode               = source_loc::mode;
    namespace scope          = tim::scope;

    std::string sargs = "";
    for(auto itr : args)
    {
        try
        {
            auto v = itr.cast<std::string>();
            sargs  = (sargs.empty()) ? v : TIMEMORY_JOIN('/', sargs, v);
        } catch(...)
        {}
    }

    auto&& _scope = scope::get_default();
    auto&& _loc   = source_loc(mode::complete, func.c_str(), line, file.c_str(), sargs);
    auto&& _obj   = (settings::enabled())
                      ? new component_bundle_t(_loc.get_captured(sargs), true, _scope)
                      : nullptr;
    return new pycomponent_bundle<UserBundleT>(_obj);
}
}  // namespace init
//
//--------------------------------------------------------------------------------------//
//
template <typename UserBundleT, typename ScopeFunc = std::function<void(bool, bool)>>
auto
generate(
    py::module& _pymod, const char* _name, const char* _doc,
    ScopeFunc _scope_set = [](bool, bool) {})
{
    using bundle_t = pycomponent_bundle<UserBundleT>;

    py::class_<bundle_t> comp_bundle(_pymod, _name, _doc);

    auto configure_pybundle = [_scope_set](py::list _args, bool flat_profile,
                                           bool timeline_profile) {
        try
        {
            pytim::pyenum_set_t components{};
            try
            {
                components = pytim::get_enum_set(_args);
            } catch(py::cast_error& e)
            {
                auto _argstr = py::str(_args);
                std::cerr << "Error getting enumeration set from "
                          << _argstr.cast<std::string>() << " : " << e.what()
                          << ". Expected python list" << std::endl;
                timemory_print_demangled_backtrace<32>();
            }

            if(_args.size() == 0)
                components.insert(WALL_CLOCK);

            size_t isize = bundle_t::size();
            if(tim::settings::debug() || tim::settings::verbose() > 3)
            {
                static auto _enum2str = pytim::get_enum_string_map();
                std::string _slist{};
                for(auto& itr : components)
                    _slist += ", " + _enum2str[itr];
                TIMEMORY_PRINT_HERE("configuring pybundle with [%s]",
                                    _slist.substr(2).c_str());
            }

            _scope_set(flat_profile, timeline_profile);

            auto _scope = tim::scope::get_default();
            if(flat_profile)
                _scope += tim::scope::flat{};
            if(timeline_profile)
                _scope += tim::scope::timeline{};

            tim::configure<UserBundleT>(components, _scope);

            if(tim::settings::debug() || tim::settings::verbose() > 3)
            {
                auto fsize = components.size();
                TIMEMORY_PRINT_HERE(
                    "final size: %lu, input size: %lu, components size: %lu\n",
                    (unsigned long) fsize, (unsigned long) isize,
                    (unsigned long) components.size());
            }
        } catch(py::cast_error& e)
        {
            std::cerr << "Error configuring bundle! " << e.what() << std::endl;
            timemory_print_demangled_backtrace<32>();
        }
    };

    tim::operation::init<UserBundleT>(
        tim::operation::mode_constant<tim::operation::init_mode::global>{});

    //==================================================================================//
    //
    //                      Component bundle
    //
    //==================================================================================//

    comp_bundle.def(py::init(&init::component_bundle<UserBundleT>), "Initialization",
                    py::arg("func"), py::arg("file"), py::arg("line"),
                    py::arg("extra") = py::list{});

    comp_bundle.def("start", &bundle_t::start, "Start the bundle");

    comp_bundle.def("stop", &bundle_t::stop, "Stop the bundle");

    comp_bundle.def_static(
        "configure", configure_pybundle, py::arg("components") = py::list{},
        py::arg("flat_profile") = false, py::arg("timeline_profile") = false,
        "Configure the profiler types (default: 'wall_clock')");

    comp_bundle.def_static("reset", &bundle_t::reset,
                           "Reset the components in the bundle");
    return comp_bundle;
}
//
}  // namespace pycomponent_bundle
//
//======================================================================================//
