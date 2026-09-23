//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

#pragma once

#include "timemory/components/base/declaration.hpp"
#include "timemory/components/base/types.hpp"
#include "timemory/mpl/math.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/declaration.hpp"
#include "timemory/units.hpp"

#include <cassert>

namespace tim
{
namespace component
{
//
//======================================================================================//
//
//                              NON-VOID BASE
//
//======================================================================================//
//
template <typename Tp, typename Value>
template <typename Vp, typename Up, enable_if_t<trait::sampler<Up>::value, int>>
void
base<Tp, Value>::add_sample(Vp&& _obj)
{
    auto _storage = static_cast<storage_type*>(get_storage());
    assert(_storage != nullptr);
    if(_storage)
        _storage->add_sample(std::forward<Vp>(_obj));
}
//
//--------------------------------------------------------------------------------------//
//
//          Units, display units, width, precision, labels, display labels
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Up>
void
base<Tp, Value>::print(
    std::ostream&, enable_if_t<!trait::uses_value_storage<Up, Value>::value, long>) const
{}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace component
}  // namespace tim
//

#include "timemory/operations/types/base_printer.hpp"
#include "timemory/operations/types/serialization.hpp"
#include "timemory/tpls/cereal/cereal.hpp"

namespace tim
{
namespace component
{
//
template <typename Tp, typename Value>
template <typename Up>
void
base<Tp, Value>::print(
    std::ostream& os, enable_if_t<trait::uses_value_storage<Up, Value>::value, int>) const
{
    operation::base_printer<Up>(os, static_cast<const Up&>(*this));
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Value>
template <typename Archive, typename Up,
          enable_if_t<!trait::custom_serialization<Up>::value, int>>
void
base<Tp, Value>::save(Archive& ar, const unsigned int version) const
{
    operation::serialization<Type>(static_cast<const Type&>(*this), ar, version);
}
//
template <typename Tp, typename Value>
template <typename Archive, typename Up,
          enable_if_t<!trait::custom_serialization<Up>::value, int>>
void
base<Tp, Value>::load(Archive& ar, const unsigned int version)
{
    laps_type::serialize(ar, version);
    data_type::serialize(ar, version);
    set_is_transient(true);  // assume always transient
}
//
}  // namespace component
}  // namespace tim
