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

/** \file component/properties.hpp
 * \headerfile component/properties.hpp "timemory/component/properties.hpp"
 * Provides properties for components
 *
 */

#pragma once

#include "timemory/backends/process.hpp"
#include "timemory/backends/threading.hpp"
#include "timemory/enum.h"
#include "timemory/environment/types.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/utility/macros.hpp"

#include <cstdio>
#include <cstring>
#include <regex>
#include <set>
#include <string>

namespace tim
{
namespace component
{
//
template <typename... Types>
struct placeholder;
//
using idset_t = std::set<std::string>;
//
struct nothing;
//
struct opaque;
//
namespace factory
{
//
template <typename Toolset, typename Arg, typename... Args>
opaque
get_opaque(Arg&& arg, Args&&... args);
//
template <typename Toolset>
opaque
get_opaque();
//
template <typename Toolset>
std::set<size_t>
get_typeids();
//
}  // namespace factory
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::component::static_properties
/// \tparam Tp Component type
/// \tparam Placeholder Whether or not the component type is a placeholder type that
/// should be ignored during runtime initialization.
///
/// \brief Provides three variants of a `matches` function for determining if a
/// component is identified by a given string or enumeration value
///
template <typename Tp, bool PlaceHolder = concepts::is_placeholder<Tp>::value>
struct static_properties;
//
template <typename Tp>
struct properties;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct state
{
    static bool& has_storage()
    {
        static thread_local bool _instance = false;
        return _instance;
    }
};
//
//--------------------------------------------------------------------------------------//
//
//  single specialization
//
template <>
struct static_properties<void, false>
{
    static bool matches(const char* _ckey, const char* _enum_str, const idset_t& _ids)
    {
        static bool _debug = tim::get_env<bool>(TIMEMORY_SETTINGS_PREFIX "DEBUG", false);
        static const auto regex_consts = std::regex_constants::ECMAScript |
                                         std::regex_constants::icase |
                                         std::regex_constants::optimize;
        std::string _opts{ _enum_str };
        _opts.reserve(_opts.size() + 512);
        for(const auto& itr : _ids)
        {
            if(!itr.empty())
                _opts += "|" + itr;
        }
        auto _option = std::string{ "\\b(" } + _opts + std::string{ ")\\b" };
        try
        {
            if(std::regex_search(_ckey, std::regex{ _option, regex_consts }))
            {
                if(_debug)
                {
                    auto _doption = std::string{ "\\b(" } + _opts + std::string{ ")\\b" };
                    TIMEMORY_PRINTF(
                        stderr,
                        "[component::static_properties::matches] '%s' matches (%s) "
                        "[regex: '%s']\n",
                        _ckey, _opts.c_str(), _doption.c_str());
                    fflush(stderr);
                }
                return true;
            }
        } catch(std::regex_error& err)
        {
            auto _doption = std::string{ "\\b(" } + _opts + std::string{ ")\\b" };
            TIMEMORY_PRINT_HERE(
                "regex error in regex_match(\"%s\", regex{ \"%s\", egrep | icase "
                "| optimize }): %s [real: %s]",
                _ckey, _doption.c_str(), err.what(), _option.c_str());
            TIMEMORY_TESTING_EXCEPTION("regex error in: \"" << _doption << "\" for "
                                                            << _ckey)
        }

        return false;
    }
};
//
//--------------------------------------------------------------------------------------//
//
//  non-placeholder types
//
template <typename Tp>
struct static_properties<Tp, false>
{
    using ptype = properties<Tp>;
    using vtype = static_properties<void, false>;

    static bool matches(int _idx) { return (_idx == ptype{}()); }
    static bool matches(const std::string& _key) { return matches(_key.c_str()); }
    static bool matches(const char* _key)
    {
        // don't allow checks for placeholder types
        static_assert(!concepts::is_placeholder<Tp>::value,
                      "static_properties is instantiating a placeholder type");

        return vtype::matches(_key, ptype::enum_string(), ptype::ids());
    }
};
//
//  placeholder types
//
template <typename Tp>
struct static_properties<Tp, true>
{
    static bool matches(int) { return false; }
    static bool matches(const char*) { return false; }
    static bool matches(const std::string&) { return false; }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::component::properties
/// \tparam Tp Component type
///
/// \brief This is a critical specialization for mapping string and integers to
/// component types at runtime. The `enum_string()` function is the enum id as
/// a string. The `id()` function is (typically) the name of the C++ component
/// as a string. The `ids()` function returns a set of strings which are alternative
/// string identifiers to the enum string or the string ID. Additionally, it
/// provides serializaiton of these values.
///
/// A macro is provides to simplify this specialization:
///
/// \code{.cpp}
/// TIMEMORY_PROPERTY_SPECIALIZATION(wall_clock, TIMEMORY_WALL_CLOCK, "wall_clock",
///                                  "real_clock", "virtual_clock")
/// \endcode
///
/// In the above, the first parameter is the C++ type, the second is the enumeration
/// id, the enum string is automatically generated via preprocessor `#` on the second
/// parameter, the third parameter is the string ID, and the remaining values are placed
/// in the `ids()`. Additionally, this macro specializes the
/// \ref tim::component::enumerator.
///
template <typename Tp>
struct properties : static_properties<Tp>
{
    using type                                = Tp;
    using value_type                          = TIMEMORY_COMPONENT;
    static constexpr TIMEMORY_COMPONENT value = TIMEMORY_COMPONENTS_END;

    static constexpr bool        specialized() { return false; }
    static constexpr const char* enum_string() { return "TIMEMORY_COMPONENTS_END"; }
    static constexpr const char* id() { return ""; }
    static idset_t               ids() { return idset_t{}; }
    template <typename Archive>
    void serialize(Archive&, const unsigned int)
    {}
    TIMEMORY_COMPONENT operator()() { return TIMEMORY_COMPONENTS_END; }
    //
    constexpr operator TIMEMORY_COMPONENT() const { return TIMEMORY_COMPONENTS_END; }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::component::enumerator
/// \tparam Idx Enumeration value
///
/// \brief This is a critical specialization for mapping string and integers to
/// component types at runtime (should always be specialized alongside \ref
/// tim::component::properties) and it is also critical for performing template
/// metaprogramming "loops" over all the components. E.g.:
///
/// \code{.cpp}
/// template <size_t Idx>
/// using Enumerator_t = typename tim::component::enumerator<Idx>::type;
///
/// template <size_t... Idx>
/// auto init(std::index_sequence<Idx...>)
/// {
///     // expand for [0, TIMEMORY_COMPONENTS_END)
///     TIMEMORY_FOLD_EXPRESSION(tim::storage_initializer::get<
///         Enumerator_t<Idx>>());
/// }
///
/// void init()
/// {
///     init(std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
/// }
/// \endcode
///
template <int Idx>
struct enumerator : properties<placeholder<nothing>>
{
    using type                  = placeholder<nothing>;
    static constexpr bool value = false;

    bool operator==(int) const { return false; }
    bool operator==(const char*) const { return false; }
    bool operator==(const std::string&) const { return false; }
};
//
//--------------------------------------------------------------------------------------//
//
template <int Idx>
using enumerator_t = typename enumerator<Idx>::type;
//
template <int Idx>
using properties_t = typename enumerator<Idx>::type;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
