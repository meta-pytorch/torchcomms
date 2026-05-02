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

#include "timemory/components/papi/backends.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/utility/macros.hpp"

#include <initializer_list>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace tim
{
namespace component
{
//
//--------------------------------------------------------------------------------------//
//
//                          Common PAPI configuration
//
//--------------------------------------------------------------------------------------//
//
/// \class papi_event_vector
/// \brief this provides the portability of using strings while retaining the
/// compatibility with using a vector of ints. This might not be the greatest idea but
/// otherwise a lot of things would be broken
class papi_event_vector : public std::vector<std::string>
{
public:
    using base_type = std::vector<std::string>;

    TIMEMORY_DEFAULT_OBJECT(papi_event_vector)

    using base_type::assign;
    using base_type::at;
    using base_type::back;
    using base_type::begin;
    using base_type::capacity;
    using base_type::cbegin;
    using base_type::cend;
    using base_type::crbegin;
    using base_type::crend;
    using base_type::data;
    using base_type::emplace;
    using base_type::emplace_back;
    using base_type::empty;
    using base_type::erase;
    using base_type::front;
    using base_type::get_allocator;
    using base_type::insert;
    using base_type::max_size;
    using base_type::pop_back;
    using base_type::push_back;
    using base_type::rbegin;
    using base_type::rend;
    using base_type::reserve;
    using base_type::resize;
    using base_type::shrink_to_fit;
    using base_type::size;
    using base_type::swap;
    using base_type::operator=;
    using base_type::operator[];

    /// generic constructor for whenever there is not 1 argument which is a vector
    template <typename Arg, typename... Args,
              std::enable_if_t<(sizeof...(Args) > 0 && !concepts::is_vector<Arg>::value),
                               int> = 0>
    papi_event_vector(Arg&&, Args&&...);

    /// converting constructor
    papi_event_vector(const base_type&);

    /// converting constructor
    papi_event_vector(const std::vector<int>&);

    /// converting constructor
    papi_event_vector(const std::initializer_list<int>&);

    /// assignment from a standard string vector
    template <typename... Args>
    papi_event_vector& operator=(const std::vector<std::string, Args...>&);

    /// assignment from a standard int vector
    template <typename... Args>
    papi_event_vector& operator=(const std::vector<int, Args...>&);

    /// conversion to an int vector
    operator std::vector<int>();
};
//
template <
    typename Arg, typename... Args,
    std::enable_if_t<(sizeof...(Args) > 0 && !concepts::is_vector<Arg>::value), int>>
inline papi_event_vector::papi_event_vector(Arg&& _arg, Args&&... _args)
: base_type(std::forward<Arg>(_arg), std::forward<Args>(_args)...)
{}

inline papi_event_vector::papi_event_vector(const base_type& rhs)
: base_type{ rhs }
{}

inline papi_event_vector::papi_event_vector(const std::vector<int>& rhs)
: base_type{}
{
    reserve(size() + rhs.size());
    for(auto&& itr : rhs)
        emplace_back(papi::get_event_info(itr).symbol);
}

inline papi_event_vector::papi_event_vector(const std::initializer_list<int>& rhs)
: base_type{}
{
    reserve(size() + rhs.size());
    for(auto&& itr : rhs)
        emplace_back(papi::get_event_info(itr).symbol);
}

template <typename... Args>
inline papi_event_vector&
papi_event_vector::operator=(const std::vector<std::string, Args...>& rhs)
{
    if(this != &rhs)
        base_type::operator=(rhs);
    return *this;
}

template <typename... Args>
inline papi_event_vector&
papi_event_vector::operator=(const std::vector<int, Args...>& rhs)
{
    reserve(size() + rhs.size());
    for(auto&& itr : rhs)
        emplace_back(papi::get_event_info(itr).symbol);
    return *this;
}

inline papi_event_vector::operator std::vector<int>()
{
    std::vector<int> _v{};
    _v.reserve(size());
    for(auto&& itr : *this)
        _v.emplace_back(papi::get_event_info(itr).event_code);
    return _v;
}
}  // namespace component
}  // namespace tim
