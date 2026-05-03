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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#include "timemory/variadic/component_bundle.hpp"

namespace tim
{
//
template <typename CompTuple, typename CompList>
class [[deprecated("Use component_bundle<T..., L*...>")]] component_hybrid;
//
template <template <typename...> class TupleT, template <typename...> class ListT,
          typename... TupleTypes, typename... ListTypes>
class component_hybrid<TupleT<TupleTypes...>, ListT<ListTypes...>>
: public component_bundle<project::timemory, TupleTypes...,
                          std::add_pointer_t<ListTypes>...>
{
public:
    using base_type  = component_bundle<project::timemory, TupleTypes...,
                                       std::add_pointer_t<ListTypes>...>;
    using value_type = typename base_type::value_type;

    // ...
    template <typename... Args>
    component_hybrid(Args&&... args)
    : base_type(std::forward<Args>(args)...)
    {}

    using base_type::add_secondary;
    using base_type::assemble;
    using base_type::audit;
    using base_type::construct;
    using base_type::count;
    using base_type::data;
    using base_type::derive;
    using base_type::fixed_count;
    using base_type::get;
    using base_type::get_component;
    using base_type::get_labeled;
    using base_type::get_prefix;
    using base_type::get_reference;
    using base_type::get_scope;
    using base_type::get_store;
    using base_type::hash;
    using base_type::init;
    using base_type::initialize;
    using base_type::invoke;
    using base_type::key;
    using base_type::laps;
    using base_type::mark_begin;
    using base_type::mark_end;
    using base_type::measure;
    using base_type::optional_count;
    using base_type::pop;
    using base_type::prefix;
    using base_type::print;
    using base_type::push;
    using base_type::record;
    using base_type::rekey;
    using base_type::reset;
    using base_type::sample;
    using base_type::serialize;
    using base_type::size;
    using base_type::start;
    using base_type::stop;
    using base_type::store;
    using base_type::type_apply;
    using base_type::operator+=;
    using base_type::operator-=;
    using base_type::operator*=;
    using base_type::operator/=;
};

template <typename... T>
using component_hybrid_t = typename component_hybrid<T...>::type;
//
}  // namespace tim
