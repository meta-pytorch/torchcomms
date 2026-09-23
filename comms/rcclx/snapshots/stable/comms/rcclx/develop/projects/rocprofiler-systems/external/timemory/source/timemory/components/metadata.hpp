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

#include "timemory/enum.h"
#include "timemory/utility/demangle.hpp"

#include <string>

namespace tim
{
namespace component
{
/// \struct tim::component::metadata
/// \brief Provides forward declaration support for assigning static metadata properties.
/// This is most useful for specialization of template components. If this class
/// is specialized for component, then the component does not need to provide
/// the static member functions `label()` and `description()`.
///
template <typename Tp>
struct metadata
{
    using type                                = Tp;
    using value_type                          = TIMEMORY_COMPONENT;
    static constexpr TIMEMORY_COMPONENT value = TIMEMORY_COMPONENTS_END;
    static std::string                  name();
    static std::string                  label();
    static std::string                  description();
    static std::string                  extra_description() { return ""; }
    static constexpr bool               specialized() { return false; }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
std::string
metadata<Tp>::name()
{
    return try_demangle<Tp>();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
std::string
metadata<Tp>::label()
{
    return try_demangle<Tp>();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
std::string
metadata<Tp>::description()
{
    return try_demangle<Tp>();
}
//
}  // namespace component
}  // namespace tim
