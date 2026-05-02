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

#include "timemory/components/base.hpp"
#include "timemory/mpl/types.hpp"

#include <cstdint>
#include <string>

namespace tim
{
namespace component
{
/// \struct tim::component::placeholder
/// \brief provides nothing, used for dummy types in enum
///
template <typename... Types>
struct placeholder
: public base<placeholder<Types...>, void>
, public Types...
{
    using value_type = void;
    using this_type  = placeholder<Types...>;
    using base_type  = base<this_type, value_type>;

    static std::string label() { return "placeholder"; }
    static std::string description() { return "placeholder"; }
    static void        record() {}
    void               start() {}
    void               stop() {}

    /// generic configuration
    template <typename... Args>
    static void configure(Args&&...)
    {}

    /// specific to gotcha
    template <size_t N, typename... Ret, typename... Args>
    static void configure(Args&&...)
    {}
};

}  // namespace component

namespace trait
{
// always filter placeholder
template <typename... Types>
struct is_available<component::placeholder<Types...>> : std::false_type
{};
}  // namespace trait
}  // namespace tim
