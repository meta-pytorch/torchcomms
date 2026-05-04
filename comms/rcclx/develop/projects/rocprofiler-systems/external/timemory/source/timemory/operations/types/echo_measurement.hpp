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

/**
 * \file timemory/operations/types/echo_measurement.hpp
 * \brief Definition for various functions for echo_measurement in operations
 */

#pragma once

#include "timemory/mpl/stl.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/settings/declaration.hpp"

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::echo_measurement
/// \brief This operation class echoes DartMeasurements for a CDash dashboard
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct echo_measurement<Tp, false> : public common_utils
{
    template <typename... Args>
    echo_measurement(Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct echo_measurement<Tp, true> : public common_utils
{
    using type         = Tp;
    using attributes_t = std::map<std::string, std::string>;

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename... Args>
    static TIMEMORY_COLD string_t generate_name(const string_t& _prefix, string_t _unit,
                                                Args&&... _args)
    {
        if(settings::dart_label())
        {
            return (_unit.length() > 0 && _unit != "%")
                       ? join("//", type::get_label(), _unit)
                       : type::get_label();
        }

        auto _extra = join('/', std::forward<Args>(_args)...);
        auto _label = uppercase(type::get_label());
        _unit       = replace(_unit, "", { " " });
        string_t _name =
            (_extra.length() > 0) ? join("//", _extra, _prefix) : join("//", _prefix);

        auto _ret = join("//", _label, _name);

        if(_ret.length() > 0 && _ret.at(_ret.length() - 1) == '/')
            _ret.erase(_ret.length() - 1);

        if(_unit.length() > 0 && _unit != "%")
            _ret += "//" + _unit;

        return _ret;
    }

    //------------------------------------------------------------------------------//
    //
    struct impl
    {
        template <typename Tuple, typename... Args, size_t... Nt,
                  enable_if_t<sizeof...(Nt) == 0, char> = 0>
        static TIMEMORY_COLD std::string name_generator(const string_t&, Tuple, Args&&...,
                                                        index_sequence<Nt...>)
        {
            return "";
        }

        template <typename Tuple, typename... Args, size_t Idx, size_t... Nt,
                  enable_if_t<sizeof...(Nt) == 0, char> = 0>
        static TIMEMORY_COLD std::string name_generator(const string_t& _prefix,
                                                        Tuple _units, Args&&... _args,
                                                        index_sequence<Idx, Nt...>)
        {
            return generate_name(_prefix, std::get<Idx>(_units),
                                 std::forward<Args>(_args)...);
        }

        template <typename Tuple, typename... Args, size_t Idx, size_t... Nt,
                  enable_if_t<(sizeof...(Nt) > 0), char> = 0>
        static TIMEMORY_COLD std::string name_generator(const string_t& _prefix,
                                                        Tuple _units, Args&&... _args,
                                                        index_sequence<Idx, Nt...>)
        {
            return join(
                ",",
                name_generator<Tuple>(_prefix, _units, std::forward<Args>(_args)...,
                                      index_sequence<Idx>{}),
                name_generator<Tuple>(_prefix, _units, std::forward<Args>(_args)...,
                                      index_sequence<Nt...>{}));
        }
    };

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename Tuple, typename... Args>
    static TIMEMORY_COLD string_t generate_name(const string_t& _prefix, Tuple _unit,
                                                Args&&... _args)
    {
        constexpr size_t N = std::tuple_size<Tuple>::value;
        return impl::template name_generator<Tuple>(
            _prefix, _unit, std::forward<Args>(_args)..., make_index_sequence<N>{});
    }

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename T, typename... Alloc, typename... Args>
    static TIMEMORY_COLD string_t generate_name(const string_t&          _prefix,
                                                std::vector<T, Alloc...> _unit,
                                                Args&&... _args)
    {
        using namespace tim::stl::ostream;
        string_t ret;
        for(auto& itr : _unit)
        {
            return join(",", ret,
                        generate_name(_prefix, itr, std::forward<Args>(_args)...));
        }
        return ret;
    }

    //----------------------------------------------------------------------------------//
    /// generate a name attribute
    ///
    template <typename T, size_t N, typename... Args>
    static TIMEMORY_COLD string_t generate_name(const string_t&  _prefix,
                                                std::array<T, N> _unit, Args&&... _args)
    {
        using namespace tim::stl::ostream;
        string_t ret;
        for(auto& itr : _unit)
        {
            return join(",", ret,
                        generate_name(_prefix, itr, std::forward<Args>(_args)...));
        }
        return ret;
    }

    //----------------------------------------------------------------------------------//
    /// generate a measurement tag
    ///
    template <typename Vt>
    static TIMEMORY_COLD void generate_measurement(std::ostream&       os,
                                                   const attributes_t& attributes,
                                                   const Vt&           value)
    {
        using namespace tim::stl::ostream;
        os << "<DartMeasurement";
        os << " " << attribute_string("type", "numeric/double");
        for(const auto& itr : attributes)
            os << " " << attribute_string(itr.first, itr.second);
        os << ">" << std::setprecision(type::get_precision()) << value
           << "</DartMeasurement>\n";
    }

    //----------------------------------------------------------------------------------//
    /// generate a measurement tag
    ///
    template <typename Vt, typename... ExtraT>
    static TIMEMORY_COLD void generate_measurement(
        std::ostream& os, attributes_t attributes,
        const std::vector<Vt, ExtraT...>& value)
    {
        auto _default_name = attributes["name"];
        int  i             = 0;
        for(const auto& itr : value)
        {
            std::stringstream ss;
            ss << "INDEX_" << i++ << " ";
            attributes["name"] = ss.str() + _default_name;
            generate_measurement(os, attributes, itr);
        }
    }

    //----------------------------------------------------------------------------------//
    /// generate a measurement tag
    ///
    template <typename Lhs, typename Rhs, typename... ExtraT>
    static TIMEMORY_COLD void generate_measurement(std::ostream&              os,
                                                   attributes_t               attributes,
                                                   const std::pair<Lhs, Rhs>& value)
    {
        auto default_name = attributes["name"];
        auto set_name     = [&](int i) {
            std::stringstream ss;
            ss << "INDEX_" << i << " ";
            attributes["name"] = ss.str() + default_name;
        };

        set_name(0);
        generate_measurement(os, attributes, value.first);
        set_name(1);
        generate_measurement(os, attributes, value.second);
    }

    //----------------------------------------------------------------------------------//
    /// generate the prefix
    ///
    static TIMEMORY_COLD string_t generate_prefix(const strvec_t& hierarchy)
    {
        if(settings::dart_label())
            return string_t("");

        string_t              ret_prefix{};
        string_t              add_prefix{};
        static const strset_t repl_chars = { "\t", "\n", "<", ">" };
        for(const auto& itr : hierarchy)
        {
            auto prefix = itr;
            prefix      = replace(prefix, "", { ">>>" });
            prefix      = replace(prefix, "", { "|_" });
            prefix      = replace(prefix, "_", repl_chars);
            prefix      = replace(prefix, "_", { "__" });
            if(prefix.length() > 0 && prefix.at(prefix.length() - 1) == '_')
                prefix.erase(prefix.length() - 1);
            ret_prefix += add_prefix + prefix;
        }
        return ret_prefix;
    }

    //----------------------------------------------------------------------------------//
    /// assumes type is not a iterable
    ///
    template <typename Up = Tp, typename Vt = typename Up::value_type,
              enable_if_t<is_enabled<Up>::value, char> = 0,
              enable_if_t<!(trait::array_serialization<Up>::value ||
                            trait::iterable_measurement<Up>::value),
                          int>                         = 0>
    TIMEMORY_COLD echo_measurement(Up& obj, const strvec_t& hierarchy)
    {
        auto prefix = generate_prefix(hierarchy);
        auto _unit  = type::get_display_unit();
        auto name   = generate_name(prefix, _unit);
        auto _data  = obj.get();

        attributes_t   attributes = { { "name", name } };
        stringstream_t ss;
        generate_measurement(ss, attributes, _data);
        std::cout << ss.str() << std::flush;
    }

    //----------------------------------------------------------------------------------//
    /// assumes type is iterable
    ///
    template <typename Up = Tp, typename Vt = typename Up::value_type,
              enable_if_t<is_enabled<Up>::value, char> = 0,
              enable_if_t<trait::array_serialization<Up>::value ||
                              trait::iterable_measurement<Up>::value,
                          int>                         = 0>
    TIMEMORY_COLD echo_measurement(Up& obj, const strvec_t& hierarchy)
    {
        auto prefix = generate_prefix(hierarchy);
        auto _data  = obj.get();

        attributes_t   attributes = {};
        stringstream_t ss;

        uint64_t idx     = 0;
        auto     _labels = obj.label_array();
        auto     _dunits = obj.display_unit_array();
        for(auto& itr : _data)
        {
            string_t _extra = (idx < _labels.size()) ? _labels.at(idx) : "";
            string_t _dunit = (idx < _labels.size()) ? _dunits.at(idx) : "";
            ++idx;
            attributes["name"] = generate_name(prefix, _dunit, _extra);
            generate_measurement(ss, attributes, itr);
        }
        std::cout << ss.str() << std::flush;
    }

    template <typename... Args, typename Up = Tp, typename Vt = typename Up::value_type,
              enable_if_t<!is_enabled<Up>::value, char> = 0>
    echo_measurement(Up&, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
