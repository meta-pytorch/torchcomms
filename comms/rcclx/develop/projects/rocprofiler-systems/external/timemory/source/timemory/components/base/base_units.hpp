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

#include "timemory/components/base/types.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/settings/settings.hpp"
#include "timemory/units.hpp"

#include <type_traits>

namespace tim
{
namespace component
{
//
/// \struct tim::component::base_units<Tp>
/// \tparam Tp the component type
///
/// \brief A helper base class for handling the units of the data in a component.
///
template <typename Tp>
struct base_units
{
    static constexpr bool is_assignable_v = trait::assignable_units<Tp>::value;
    using units_value_type                = typename trait::units<Tp>::type;
    using units_display_type              = typename trait::units<Tp>::display_type;

    static decltype(auto) unit();
    static decltype(auto) display_unit();

    static decltype(auto) get_unit();
    static decltype(auto) get_display_unit();

    static void set_unit(units_value_type);
    static void set_display_unit(units_display_type);

private:
    static void                base_units_update_impl();
    static units_value_type&   base_units_value_impl();
    static units_display_type& base_units_display_impl();
};

//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
inline decltype(auto)
base_units<Tp>::unit()
{
    if constexpr(is_assignable_v)
    {
        return base_units_value_impl();
    }
    else
    {
        if constexpr(std::is_trivially_copyable<units_value_type>::value)
        {
            return units_value_type{ base_units_value_impl() };
        }
        else
        {
            return const_cast<const units_value_type&>(base_units_value_impl());
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
inline decltype(auto)
base_units<Tp>::display_unit()
{
    if constexpr(is_assignable_v)
    {
        return base_units_display_impl();
    }
    else
    {
        if constexpr(std::is_trivially_copyable<units_display_type>::value)
        {
            return units_display_type{ base_units_display_impl() };
        }
        else
        {
            return const_cast<const units_display_type&>(base_units_display_impl());
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
inline void
base_units<Tp>::base_units_update_impl()
{
    constexpr bool _supports_settings_config =
        trait::uses_timing_units<Tp>::value || trait::uses_memory_units<Tp>::value;

    if constexpr(_supports_settings_config)
    {
        static bool _once     = false;
        const auto& _settings = settings::shared_instance();
        if(!_settings || (_once && _settings->get_initialized()))
            return;
        _once = true;

        if constexpr(trait::uses_timing_units<Tp>::value)
        {
            auto&& _v = units::get_timing_unit(_settings->get_timing_units());
            Tp::set_unit(std::get<1>(_v));
            Tp::set_display_unit(std::get<0>(_v));
        }
        else if constexpr(trait::uses_memory_units<Tp>::value)
        {
            auto&& _v = units::get_memory_unit(_settings->get_memory_units());
            Tp::set_unit(std::get<1>(_v));
            Tp::set_display_unit(std::get<0>(_v));
        }
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
inline decltype(auto)
base_units<Tp>::get_unit()
{
    base_units_update_impl();
    return Tp::unit();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
inline decltype(auto)
base_units<Tp>::get_display_unit()
{
    base_units_update_impl();
    return Tp::display_unit();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
inline void
base_units<Tp>::set_unit(units_value_type _v)
{
    if constexpr(is_assignable_v)
    {
        Tp::unit() = std::move(_v);
    }
    else
    {
        base_units_value_impl() = std::move(_v);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
inline void
base_units<Tp>::set_display_unit(units_display_type _v)
{
    if constexpr(is_assignable_v)
    {
        Tp::display_unit() = std::move(_v);
    }
    else
    {
        base_units_display_impl() = std::move(_v);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
typename base_units<Tp>::units_value_type&
base_units<Tp>::base_units_value_impl()
{
    static units_value_type _v = []() -> units_value_type {
        if constexpr(trait::uses_timing_units<Tp>::value)
        {
            return units::sec;
        }
        else if constexpr(trait::uses_memory_units<Tp>::value)
        {
            return units::megabyte;
        }
        else if constexpr(trait::uses_temperature_units<Tp>::value)
        {
            return 1;
        }
        else if constexpr(trait::uses_power_units<Tp>::value)
        {
            return units::watt;
        }
        else if constexpr(trait::uses_percent_units<Tp>::value)
        {
            return 1;
        }
        return 1;
    }();
    return _v;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
typename base_units<Tp>::units_display_type&
base_units<Tp>::base_units_display_impl()
{
    if constexpr(concepts::is_string_type<units_display_type>::value)
    {
        static std::string _init = []() {
            std::string _value = {};
            if(_value.empty())
            {
                if constexpr(trait::uses_timing_units<Tp>::value)
                {
                    _value = units::time_repr(unit());
                }
                else if constexpr(trait::uses_memory_units<Tp>::value)
                {
                    _value = units::mem_repr(unit());
                }
                else if constexpr(trait::uses_temperature_units<Tp>::value)
                {
                    _value = "degC";
                }
                else if constexpr(trait::uses_power_units<Tp>::value)
                {
                    _value = units::power_repr(unit());
                }
                else if constexpr(trait::uses_percent_units<Tp>::value)
                {
                    _value = "%";
                }
            }
            return _value;
        }();

        if constexpr(std::is_same<units_display_type, std::string>::value)
        {
            return _init;
        }
        else if constexpr(std::is_same<units_display_type, std::string_view>::value)
        {
            static auto _v = units_display_type{ _init };
            return _v;
        }
        else
        {
            return _init.c_str();
        }
    }
    else
    {
        static units_display_type _v = {};
        return _v;
    }
}
}  // namespace component
}  // namespace tim
