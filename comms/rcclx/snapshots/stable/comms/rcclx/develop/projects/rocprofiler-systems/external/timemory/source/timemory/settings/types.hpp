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

#include "timemory/compat/macros.h"
#include "timemory/process/process.hpp"
#include "timemory/settings/macros.hpp"
#include "timemory/utility/serializer.hpp"
#include "timemory/utility/type_list.hpp"

#include <functional>
#include <map>
#include <set>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
//                              settings
//
//--------------------------------------------------------------------------------------//
//
std::string
get_local_datetime(const char* dt_format, std::time_t* dt_curr = nullptr);
//
//--------------------------------------------------------------------------------------//
//
/// this enumeration is used to specify how setting values were updated
///
enum class setting_update_type : short
{
    default_value = 0,
    env,
    config,
    user,
    unspecified,
};
//
using setting_supported_data_types =
    type_list<bool, std::string, int16_t, int32_t, int64_t, uint16_t, uint32_t, uint64_t,
              size_t, process::id_t, float, double>;
//
struct TIMEMORY_VISIBILITY("default") vsettings;
//
template <typename Tp, typename Vp = Tp>
struct TIMEMORY_VISIBILITY("default") tsettings;
//
struct TIMEMORY_VISIBILITY("default") settings;
//
namespace operation
{
template <typename Tp, typename TagT = void>
struct setting_serialization;
}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim

TIMEMORY_SET_CLASS_VERSION(2, ::tim::settings)
TIMEMORY_SET_CLASS_VERSION(1, ::tim::vsettings)

namespace tim
{
namespace cereal
{
namespace detail
{
template <typename Tp, typename Vp>
struct StaticVersion<tsettings<Tp, Vp>>
{
    static constexpr std::uint32_t version = 1;
};
}  // namespace detail
}  // namespace cereal
}  // namespace tim
