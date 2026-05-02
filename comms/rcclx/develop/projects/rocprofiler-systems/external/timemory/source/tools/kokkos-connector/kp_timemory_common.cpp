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

#include "kp_timemory.hpp"
#include "timemory/operations/types/storage_initializer.hpp"
#include "timemory/timemory.hpp"

#include <cstdint>

//--------------------------------------------------------------------------------------//

namespace tim
{
template <>
inline auto
invoke_preinit<KokkosMemoryTracker>(long)
{
    KokkosMemoryTracker::label()       = "kokkos_memory";
    KokkosMemoryTracker::description() = "Kokkos Memory tracker";
}
}  // namespace tim

//--------------------------------------------------------------------------------------//

inline memory_map_t&
get_memory_map()
{
    return get_tl_static<memory_map_t>();
}

//--------------------------------------------------------------------------------------//

inline auto&
get_memory_map(SpaceHandle _space)
{
    return get_memory_map()[tim::string_view_t{ _space.name }];
}

//--------------------------------------------------------------------------------------//

inline memory_stack_t&
get_memory_stack()
{
    return get_tl_static<memory_stack_t>();
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_allocate_data(const SpaceHandle space, const char* label, const void* const,
                      const uint64_t size)
{
    if(!tim::settings::enabled())
        return;
    alloc_entry_t{ TIMEMORY_JOIN('/', "kokkos/allocate", space.name, label) }.store(
        std::plus<int64_t>{}, size);
}

extern "C" void
kokkosp_deallocate_data(const SpaceHandle space, const char* label, const void* const,
                        const uint64_t size)
{
    if(!tim::settings::enabled())
        return;
    alloc_entry_t{ TIMEMORY_JOIN('/', "kokkos/deallocate", space.name, label) }.store(
        std::plus<int64_t>{}, size);
}

//--------------------------------------------------------------------------------------//

extern "C" void
kokkosp_begin_deep_copy(SpaceHandle dst_handle, const char* dst_name, const void* dst_ptr,
                        SpaceHandle src_handle, const char* src_name, const void* src_ptr,
                        uint64_t size)
{
    if(!tim::settings::enabled())
        return;
    auto name = TIMEMORY_JOIN('/', "kokkos/deep_copy",
                              TIMEMORY_JOIN('=', dst_handle.name, dst_name),
                              TIMEMORY_JOIN('=', src_handle.name, src_name));
    get_memory_stack().emplace_back(memory_entry_t(name, true));
    get_memory_stack().back().audit(dst_handle, dst_name, dst_ptr, src_handle, src_name,
                                    src_ptr, size);
    get_memory_stack().back().start();
    get_memory_stack().back().store(std::plus<int64_t>{}, size);
}

extern "C" void
kokkosp_end_deep_copy()
{
    if(get_memory_stack().empty())
        return;
    get_memory_stack().back().store(std::minus<int64_t>{}, 0);
    get_memory_stack().back().stop();
    get_memory_stack().pop_back();
}

//--------------------------------------------------------------------------------------//

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(user_kokkosp_bundle, false, void)
TIMEMORY_INITIALIZE_STORAGE(KokkosMemoryTracker)

//--------------------------------------------------------------------------------------//
