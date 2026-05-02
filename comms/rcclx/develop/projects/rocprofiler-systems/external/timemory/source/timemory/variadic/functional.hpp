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

#include "timemory/api.hpp"
#include "timemory/components/base/types.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/available.hpp"
#include "timemory/operations/types/cache.hpp"
#include "timemory/operations/types/construct.hpp"
#include "timemory/operations/types/generic.hpp"
#include "timemory/utility/types.hpp"

#include <type_traits>

namespace tim
{
//
namespace invoke
{
//
template <typename... Args>
TIMEMORY_INLINE void
print(std::ostream& os, Args&&... args);
//
template <typename... Args>
TIMEMORY_INLINE void
print(std::ostream& os, const std::string& delim, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  invoke
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class OpT, typename ApiT,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class OpT, template <typename...> class TupleT,
          typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class OpT, typename ApiT,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class OpT, template <typename...> class TupleT,
          typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class OpT, typename ApiT = TIMEMORY_API, typename... Up,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(mpl::piecewise_select<Up...>, TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class OpT, typename ApiT = TIMEMORY_API, typename... Up,
          template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
invoke(mpl::piecewise_select<Up...>, TupleT<Tp&...>& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  construct
//--------------------------------------------------------------------------------------//
//
template <typename TupleT, typename ApiT, typename... Args>
TIMEMORY_INLINE auto
construct(Args&&... args);
//
template <typename TupleT, typename... Args>
TIMEMORY_INLINE auto
construct(Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  destroy
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp>
TIMEMORY_INLINE auto
destroy(TupleT<Tp...>& obj);
//
template <template <typename...> class TupleT, typename... Tp>
TIMEMORY_INLINE auto
destroy(TupleT<Tp...>& obj);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp>
TIMEMORY_INLINE auto
destroy(TupleT<Tp&...>&& obj);
//
template <template <typename...> class TupleT, typename... Tp>
TIMEMORY_INLINE auto
destroy(TupleT<Tp&...>&& obj);
//
//--------------------------------------------------------------------------------------//
//                                  start
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
start(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
start(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
start(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
start(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  stop
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
stop(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
stop(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
stop(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
stop(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  mark
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
mark(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
mark(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
mark(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
mark(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  mark_begin
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
mark_begin(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
mark_begin(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
mark_begin(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
mark_begin(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  mark_end
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
mark_end(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
mark_end(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
mark_end(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
mark_end(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  store
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
store(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
store(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
store(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
store(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  reset
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
reset(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
reset(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
reset(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
reset(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  record
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
record(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
record(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
record(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
record(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  measure
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
measure(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
measure(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
measure(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
measure(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  push
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
push(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
push(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
push(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
push(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  pop
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
pop(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
pop(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
pop(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
pop(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  set_prefix
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
set_prefix(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
set_prefix(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
set_prefix(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
set_prefix(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  set_scope
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
set_scope(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
set_scope(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
set_scope(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
set_scope(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  set_state
//--------------------------------------------------------------------------------------//
//
template <typename ApiT = TIMEMORY_API, template <typename...> class TupleT,
          typename... Tp, typename... Args>
TIMEMORY_INLINE void
set_state(TupleT<Tp...>& obj, Args&&...);
//
template <typename ApiT = TIMEMORY_API, template <typename...> class TupleT,
          typename... Tp, typename... Args>
TIMEMORY_INLINE void
set_state(TupleT<Tp&...>&& obj, Args&&...);
//
//--------------------------------------------------------------------------------------//
//                                  assemble
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
assemble(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
assemble(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
assemble(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
assemble(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  derive
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
derive(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
derive(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
derive(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
derive(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  audit
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
audit(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
audit(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
audit(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
audit(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  add_secondary
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
add_secondary(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
add_secondary(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE void
add_secondary(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE void
add_secondary(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  get
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE auto
get(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE auto
get(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE auto
get(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE auto
get(TupleT<Tp&...>&& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp>
TIMEMORY_INLINE auto
get(TupleT<Tp...>& obj, void*& _ptr, size_t _hash);
//
template <template <typename...> class TupleT, typename... Tp>
TIMEMORY_INLINE auto
get(TupleT<Tp...>& obj, void*& _ptr, size_t _hash);
//
//--------------------------------------------------------------------------------------//
//                                  get_labeled
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE auto
get_labeled(TupleT<Tp...>& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE auto
get_labeled(TupleT<Tp...>& obj, Args&&... args);
//
template <typename ApiT, template <typename...> class TupleT, typename... Tp,
          typename... Args>
TIMEMORY_INLINE auto
get_labeled(TupleT<Tp&...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
TIMEMORY_INLINE auto
get_labeled(TupleT<Tp&...>&& obj, Args&&... args);
//
//--------------------------------------------------------------------------------------//
//                                  get_labeled
//--------------------------------------------------------------------------------------//
//
template <typename ApiT, typename ArchiveT, template <typename...> class TupleT,
          typename... Tp>
TIMEMORY_INLINE auto
serialize(ArchiveT& ar, TupleT<Tp...>& obj);
//
template <typename ArchiveT, template <typename...> class TupleT, typename... Tp>
TIMEMORY_INLINE auto
serialize(ArchiveT& ar, TupleT<Tp...>& obj);
//
template <typename ApiT, typename ArchiveT, template <typename...> class TupleT,
          typename... Tp>
TIMEMORY_INLINE auto
serialize(ArchiveT& ar, TupleT<Tp&...>&& obj);
//
template <typename ArchiveT, template <typename...> class TupleT, typename... Tp>
TIMEMORY_INLINE auto
serialize(ArchiveT& ar, TupleT<Tp&...>&& obj);
//
//--------------------------------------------------------------------------------------//
//                                  get_cache
//--------------------------------------------------------------------------------------//
//
template <typename... BundleT>
TIMEMORY_INLINE auto
get_cache();
//
//--------------------------------------------------------------------------------------//
//
namespace disjoint
{
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
start(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
stop(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark_begin(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
mark_end(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
store(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
reset(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
record(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
measure(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
push(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
pop(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
set_prefix(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
set_scope(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
assemble(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
derive(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
audit(TupleT<Tp...>&& obj, Args&&... args);
//
template <template <typename...> class TupleT, typename... Tp, typename... Args>
void
add_secondary(TupleT<Tp...>&& obj, Args&&... args);
//
}  // namespace disjoint
}  // namespace invoke
}  // namespace tim

#include "timemory/variadic/functional.cpp"
