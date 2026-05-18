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
#include "timemory/macros/attributes.hpp"
#include "timemory/utility/bit_flags.hpp"

namespace tim
{
namespace component
{
/// \struct base_state
/// \brief Provide state configuration options for a component instance.
/// The current states are:
///
/// 1. `get_is_running()`   : returns true if a component has started collection.
///
/// 2. `get_is_on_stack()`  : returns true if a component has bookmarked a storage
///                           location.
///
/// 3. `get_is_transient()` : returns true if the value returned from `get()` (or similar)
///                           represents a phase measurement.
///
/// 4. `get_is_flat()`      : returns true if the component bookmarked storage at a
///                           call-stack depth of zero explicitly.
///
/// 5. `get_depth_change()` : used internally by components to determine if a push
///                           operation incremented/decremented the call-stack depth.
///
/// 6. `get_is_invalid()`   : used to indicate that the metric does not have a valid
///                           state. If set to false before a push operation, this will
///                           suppress insertion into storage.
///
struct base_state : private utility::bit_flags<6>
{
protected:
    using base_type = utility::bit_flags<6>;
    using base_type::set;
    using base_type::test;

public:
    using base_type::reset;

    TIMEMORY_INLINE bool get_is_running() const { return test<RunningIdx>(); }
    TIMEMORY_INLINE bool get_is_on_stack() const { return test<OnStackIdx>(); }
    TIMEMORY_INLINE bool get_is_transient() const { return test<TransientIdx>(); }
    TIMEMORY_INLINE bool get_is_flat() const { return test<FlatIdx>(); }
    TIMEMORY_INLINE bool get_depth_change() const { return test<DepthIdx>(); }
    TIMEMORY_INLINE bool get_is_invalid() const { return test<InvalidIdx>(); }

    TIMEMORY_INLINE void set_is_running(bool v) { set<RunningIdx>(v); }
    TIMEMORY_INLINE void set_is_on_stack(bool v) { set<OnStackIdx>(v); }
    TIMEMORY_INLINE void set_is_transient(bool v) { set<TransientIdx>(v); }
    TIMEMORY_INLINE void set_is_flat(bool v) { set<FlatIdx>(v); }
    TIMEMORY_INLINE void set_depth_change(bool v) { set<DepthIdx>(v); }
    TIMEMORY_INLINE void set_is_invalid(bool v) { set<InvalidIdx>(v); }

protected:
    enum State
    {
        RunningIdx   = 0,
        OnStackIdx   = 1,
        TransientIdx = 2,
        FlatIdx      = 3,
        DepthIdx     = 4,
        InvalidIdx   = 5,
    };
};
}  // namespace component
}  // namespace tim
