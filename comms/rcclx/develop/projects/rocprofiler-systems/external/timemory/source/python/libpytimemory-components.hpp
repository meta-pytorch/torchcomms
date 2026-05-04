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

#if !defined(TIMEMORY_PYBIND11_SOURCE)
#    define TIMEMORY_PYBIND11_SOURCE
#endif

//======================================================================================//
// disables a bunch of warnings
//
#include "pybind11/cast.h"
#include "pybind11/embed.h"
#include "pybind11/eval.h"
#include "pybind11/functional.h"
#include "pybind11/iostream.h"
#include "pybind11/numpy.h"
#include "pybind11/operators.h"
#include "pybind11/options.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"
#include "timemory/components/types.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/runtime/enumerate.hpp"
#include "timemory/types.hpp"
#include "timemory/utility/macros.hpp"

namespace py = pybind11;
using namespace py::literals;

//--------------------------------------------------------------------------------------//
//
//                                       GENERAL
//
//--------------------------------------------------------------------------------------//
//
namespace pytim
{
//
template <typename... Tail, typename FuncT, typename ValT>
bool
try_cast_seq(FuncT&&, ValT&, std::ostream* = nullptr,
             tim::enable_if_t<sizeof...(Tail) == 0> = 0)
{
    return false;
}
//
template <typename Tp, typename... Tail, typename FuncT, typename ValT>
bool
try_cast_seq(FuncT&& f, ValT& v, std::ostream* _msg = nullptr)
{
    try
    {
        std::forward<FuncT>(f)(v.template cast<Tp>());
    } catch(py::cast_error& e)
    {
        if(_msg)
            (*_msg) << e.what() << '\n';
        return try_cast_seq<Tail...>(std::forward<FuncT>(f), v, _msg);
    }
    return true;
}
//
using pyenum_set_t = std::set<TIMEMORY_COMPONENT>;
//
/// \fn TIMEMORY_COMPONENT get_enum(py::object args)
/// \param[in] args String or component enumeration for component
/// \param[out] component_enum Return TIMEMORY_COMPONENT enum
///
/// \brief Converts a python specification of component into a C++ enum type
inline TIMEMORY_COMPONENT
get_enum(py::object _obj)
{
    try
    {
        return _obj.cast<TIMEMORY_COMPONENT>();
    } catch(py::cast_error&)
    {}

    try
    {
        auto _sitr = _obj.cast<std::string>();
        // return native components end so that message isn't delivered
        if(_sitr.length() == 0)
            return TIMEMORY_NATIVE_COMPONENTS_END;
        return tim::runtime::enumerate(_sitr);
    } catch(py::cast_error& e)
    {
        std::cerr << e.what() << std::endl;
    }

    return TIMEMORY_COMPONENTS_END;
}
//
/// \fn auto get_enum_set(py::list args)
/// \param[in] args Python list of strings or component enumerations
/// \param[out] components Return a set of TIMEMORY_COMPONENT enums
///
/// \brief Converts a python specification of components into a C++ type
inline auto
get_enum_set(py::list _args)
{
    auto components = pyenum_set_t{};

    for(auto itr : _args)
    {
        TIMEMORY_COMPONENT _citr = TIMEMORY_COMPONENTS_END;
        try
        {
            _citr = get_enum(itr.cast<py::object>());
        } catch(py::cast_error& e)
        {
            std::cerr << e.what() << std::endl;
        }

        if(_citr != TIMEMORY_COMPONENTS_END)
            components.insert(_citr);
        else if(_citr != TIMEMORY_NATIVE_COMPONENTS_END)
        {
            std::string obj_repr = "";
            try
            {
                auto locals = py::dict("obj"_a = itr.cast<py::object>());
                py::exec(R"(
                     obj_repr = "'{}' [type: {}]".format(obj, type(obj).__name__)
                     )",
                         py::globals(), locals);
                obj_repr = locals["obj_repr"].cast<std::string>();
            } catch(py::cast_error&)
            {}

            TIMEMORY_PRINT_HERE("ignoring argument that failed casting to either "
                                "'timemory.component' and string: %s",
                                obj_repr.c_str());
        }
    }
    return components;
}
//
namespace impl
{
//
template <typename Tp>
struct get_type_enums;
//
struct enum_value_set
{};
struct enum_string_map
{};
struct string_enum_map
{};
//
template <typename... Types>
struct get_type_enums<tim::type_list<Types...>>
{
    auto operator()(enum_value_set) const
    {
        static auto _instance = []() {
            return pytim::pyenum_set_t{ tim::component::properties<Types>{}()... };
        }();
        return _instance;
    }
    //
    auto operator()(enum_string_map) const
    {
        using type            = std::map<TIMEMORY_COMPONENT, std::string>;
        static auto _instance = []() {
            return type{ { tim::component::properties<Types>{}(),
                           tim::component::properties<Types>::enum_string() }... };
        }();
        return _instance;
    }
    //
    auto operator()(string_enum_map) const
    {
        using type            = std::map<std::string, TIMEMORY_COMPONENT>;
        static auto _instance = [](const auto& rev) {
            auto ret = type{};
            for(auto& itr : rev)
                ret.insert({ itr.second, itr.first });
        }((*this)(enum_string_map{}));
        return _instance;
    }
    //
};
//
}  // namespace impl
//
template <typename Tp = tim::complete_types_t>
auto
get_type_enums()
{
    return impl::get_type_enums<Tp>{}(impl::enum_value_set{});
}
//
template <typename Tp = tim::complete_types_t>
auto
get_enum_string_map()
{
    return impl::get_type_enums<Tp>{}(impl::enum_string_map{});
}
//
template <typename Tp = tim::complete_types_t>
auto
get_string_enum_map()
{
    return impl::get_type_enums<Tp>{}(impl::string_enum_map{});
}
//
template <typename TupleT>
struct construct_dict
{
    using type = TupleT;
    construct_dict(const TupleT& _tup, py::dict& _dict)
    {
        auto _label = std::get<0>(_tup);
        if(_label.size() > 0)
            _dict[_label.c_str()] = std::get<1>(_tup);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <>
struct construct_dict<std::tuple<std::string, void>>
{
    using type = std::tuple<std::string, void>;
    template <typename... ArgsT>
    construct_dict(ArgsT&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
struct dict
{
    template <typename... Types>
    static py::dict construct(const std::tuple<Types...>& _tup)
    {
        using apply_types = std::tuple<construct_dict<Types>...>;
        py::dict _dict;
        ::tim::mpl::apply<void>::access<apply_types>(_tup, std::ref(_dict));
        return _dict;
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace pytim
//
//--------------------------------------------------------------------------------------//
//
//                                       COMPONENTS
//
//--------------------------------------------------------------------------------------//
//
namespace pycomponents
{
py::module
generate(py::module& _pymod);
}
//
//--------------------------------------------------------------------------------------//
//
//                                      ENUMERATION
//
//--------------------------------------------------------------------------------------//
//
namespace pyenumeration
{
py::enum_<TIMEMORY_NATIVE_COMPONENT>
generate(py::module& _pymod);
}
//
//--------------------------------------------------------------------------------------//
//
//                                       SETTINGS
//
//--------------------------------------------------------------------------------------//
//
namespace pysettings
{
//
py::class_<tim::settings>
generate(py::module& _pymod);
}  // namespace pysettings
//
//--------------------------------------------------------------------------------------//
//
//                                       SIGNALS
//
//--------------------------------------------------------------------------------------//
//
namespace pysignals
{
py::module
generate(py::module& _pymod);
}  // namespace pysignals
//
//--------------------------------------------------------------------------------------//
//
//                                        UNITS
//
//--------------------------------------------------------------------------------------//
//
namespace pyunits
{
py::module
generate(py::module& _pymod);
}  // namespace pyunits
//
//--------------------------------------------------------------------------------------//
//
//                                      HARDWARE_COUNTERS
//
//--------------------------------------------------------------------------------------//
//
namespace pyhardware_counters
{
py::module
generate(py::module& _pymod);
}  // namespace pyhardware_counters
//
//--------------------------------------------------------------------------------------//
//
//                                          APIs
//
//--------------------------------------------------------------------------------------//
//
namespace pyapi
{
py::module
generate(py::module& _pymod);
}  // namespace pyapi
//
//--------------------------------------------------------------------------------------//
//
//                                      COMPONENT_LIST
//
//--------------------------------------------------------------------------------------//
//
namespace pycomponent_list
{
void
generate(py::module& _pymod);
}  // namespace pycomponent_list
//
//--------------------------------------------------------------------------------------//
//
//                                      COMPONENT_BUNDLE
//
//--------------------------------------------------------------------------------------//
//
namespace pycomponent_bundle
{
void
generate(py::module& _pymod);
}  // namespace pycomponent_bundle
//
//--------------------------------------------------------------------------------------//
//
//                                      AUTO_TIMER
//
//--------------------------------------------------------------------------------------//
//
namespace pyauto_timer
{
void
generate(py::module& _pymod);
}  // namespace pyauto_timer
//
//--------------------------------------------------------------------------------------//
//
//                                      RSS USAGE
//
//--------------------------------------------------------------------------------------//
//
namespace pyrss_usage
{
using namespace tim::component;
using rss_usage_t =
    tim::component_bundle_t<TIMEMORY_API, page_rss, peak_rss, num_minor_page_faults,
                            num_major_page_faults, voluntary_context_switch,
                            priority_context_switch>;

py::class_<rss_usage_t>
generate(py::module& _pymod, py::module& _pyunits);
}  // namespace pyrss_usage
//
//--------------------------------------------------------------------------------------//
//
//                                      PROFILER
//
//--------------------------------------------------------------------------------------//
//
namespace pyprofile
{
py::module
generate(py::module& _pymod);
}  // namespace pyprofile
//
//--------------------------------------------------------------------------------------//
//
//                                      TRACER
//
//--------------------------------------------------------------------------------------//
//
namespace pytrace
{
py::module
generate(py::module& _pymod);
//
}  // namespace pytrace
//
//--------------------------------------------------------------------------------------//
//
//                                    STATISTICS
//
//--------------------------------------------------------------------------------------//
//
namespace pystatistics
{
void
generate(py::module& _pymod);
}
//
//--------------------------------------------------------------------------------------//
//
//                                      STORAGE
//
//--------------------------------------------------------------------------------------//
//
namespace pystorage
{
py::module
generate(py::module& _pymod);
}
//
