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

/// \file timemory/components/user_bundle/types.hpp
/// \brief Forward declaration of user_bundle components. User-bundles are similar to the
/// classical profiling interface where the interface is fixed.

#pragma once

#include "timemory/api.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/ompt/types.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

#if defined(TIMEMORY_USE_EXTERN) && !defined(TIMEMORY_USE_USER_BUNDLE_EXTERN)
#    define TIMEMORY_USE_USER_BUNDLE_EXTERN
#endif

#if !defined(TIMEMORY_USE_USER_BUNDLE_EXTERN) &&                                         \
    !defined(TIMEMORY_USER_BUNDLE_SOURCE) &&                                             \
    !defined(TIMEMORY_USER_BUNDLE_COMPONENT_HEADER_MODE)
#    define TIMEMORY_USER_BUNDLE_COMPONENT_HEADER_MODE
#endif

//======================================================================================//
//
TIMEMORY_DECLARE_TEMPLATE_COMPONENT(user_bundle, size_t Idx, typename Tag)
//
TIMEMORY_BUNDLE_INDEX(global_bundle_idx, 10000)
// TIMEMORY_BUNDLE_INDEX(ompt_bundle_idx, 11110)
TIMEMORY_BUNDLE_INDEX(mpip_bundle_idx, 11111)
TIMEMORY_BUNDLE_INDEX(ncclp_bundle_idx, 11112)
TIMEMORY_BUNDLE_INDEX(trace_bundle_idx, 20000)
TIMEMORY_BUNDLE_INDEX(profiler_bundle_idx, 22000)
TIMEMORY_BUNDLE_INDEX(kokkosp_bundle_idx, 0)

namespace tim
{
namespace component
{
/// \typedef tim::component::user_bundle<global_bundle_idx, project::timemory>
/// tim::component::user_global_bundle
///
/// \brief A specification of components which is used by multiple variadic bundlers and
/// user_bundles as the fall-back set of components if their specific variable is
/// not set. E.g. user_mpip_bundle will use this if TIMEMORY_MPIP_COMPONENTS is not
/// specified
using user_global_bundle = user_bundle<global_bundle_idx, project::timemory>;

// these were deprecated
using user_tuple_bundle = user_global_bundle;
using user_list_bundle  = user_global_bundle;

/// \typedef tim::component::user_bundle<ompt_bundle_idx, project::timemory>
/// tim::component::user_ompt_bundle
///
/// \brief Generic bundle for inserting components at runtime into OMPT call-back system.
/// Configure via TIMEMORY_OMPT_COMPONENTS [environment], settings::ompt_components()
/// [string], or direct insertion
using user_ompt_bundle = user_bundle<ompt_bundle_idx, project::timemory>;

/// \typedef tim::component::user_bundle<mpip_bundle_idx, project::timemory>
/// tim::component::user_mpip_bundle
///
/// \brief Generic bundle for inserting components at runtime around MPI calls. Configure
/// via TIMEMORY_MPIP_COMPONENTS [environment], settings::mpip_components() [string], or
/// direct insertion
using user_mpip_bundle = user_bundle<mpip_bundle_idx, project::timemory>;

/// \typedef tim::component::user_bundle<ncclp_bundle_idx, project::timemory>
/// tim::component::user_ncclp_bundle
///
/// \brief Generic bundle for inserting components at runtime around NCCL calls. Configure
/// via TIMEMORY_NCCLP_COMPONENTS [environment], settings::ncclp_components() [string], or
/// direct insertion
using user_ncclp_bundle = user_bundle<ncclp_bundle_idx, project::timemory>;

/// \typedef tim::component::user_bundle<trace_bundle_idx, project::timemory>
/// tim::component::user_trace_bundle
///
/// \brief Used by `timemory-run` instrumentation tool for dynamic instrumentation
/// at runtime and re-writing binaries with instrumentation. See `timemory-run`
/// documentation for instructions about using this type to insert custom components.
/// This component is also used by the Python line-tracing profiler (i.e. `python -m
/// timemory.trace <OPTIONS> -- <CMD>`
/// Environment variable: `TIMEMORY_TRACE_COMPONENTS`
using user_trace_bundle = user_bundle<trace_bundle_idx, project::timemory>;

/// \typedef tim::component::user_bundle<profiler_bundle_idx, project::timemory>
/// tim::component::user_profiler_bundle
///
/// \brief Used by the Python function profiler, e.g. `python -m timemory.profiler
/// <OPTIONS> -- <CMD>`
/// Environment variable: `TIMEMORY_PROFILER_COMPONENTS`
using user_profiler_bundle = user_bundle<profiler_bundle_idx, project::timemory>;

/// \typedef tim::component::user_bundle<kokkosp_bundle_idx, project::timemory>
/// tim::component::user_kokkosp_bundle
///
/// \brief Bundle used for Kokkos runtime callbacks that are built into the core library.
/// Environment variable: `TIMEMORY_KOKKOS_COMPONENTS`
using user_kokkosp_bundle = user_bundle<kokkosp_bundle_idx, project::kokkosp>;

//
}  // namespace component
}  // namespace tim
//
#if defined(TIMEMORY_COMPILER_INSTRUMENTATION)
//
namespace tim
{
namespace trait
{
//
template <size_t Idx, typename Tag>
struct is_available<component::user_bundle<Idx, Tag>> : std::false_type
{};
//
}  // namespace trait
}  // namespace tim
//
#else
//
#    if !defined(TIMEMORY_USE_OMPT)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_ompt_bundle, false_type)
#    endif
//
#    if !defined(TIMEMORY_USE_MPI) || !defined(TIMEMORY_USE_GOTCHA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_mpip_bundle, false_type)
#    endif
//
#    if !defined(TIMEMORY_USE_NCCL) || !defined(TIMEMORY_USE_GOTCHA)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::user_ncclp_bundle, false_type)
#    endif
//
#endif
//
//--------------------------------------------------------------------------------------//
//
//                              IS USER BUNDLE
//                              REQUIRES PREFIX
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SET_COMPONENT_API(component::user_global_bundle, project::timemory, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::user_mpip_bundle, project::timemory,
                           os::supports_linux)
TIMEMORY_SET_COMPONENT_API(component::user_ncclp_bundle, project::timemory,
                           os::supports_linux)
TIMEMORY_SET_COMPONENT_API(component::user_trace_bundle, project::timemory, os::agnostic)
TIMEMORY_SET_COMPONENT_API(component::user_profiler_bundle, project::timemory,
                           os::agnostic)
//
namespace tim
{
namespace trait
{
//
template <size_t Idx, typename Type>
struct is_user_bundle<component::user_bundle<Idx, Type>> : true_type
{};
//
}  // namespace trait
//
//--------------------------------------------------------------------------------------//
//
namespace concepts
{
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class Tuple, typename... T>
struct has_user_bundle<Tuple<T...>>
{
    using type = typename mpl::get_true_types<trait::is_user_bundle, Tuple<T...>>::type;
    static constexpr bool value = (mpl::get_tuple_size<type>::value != 0);
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace concepts
//
//--------------------------------------------------------------------------------------//
//
namespace operation
{
template <typename T>
struct reset;
//
template <size_t Idx, typename Type>
struct reset<component::user_bundle<Idx, Type>>
{
    using type = component::user_bundle<Idx, Type>;

    TIMEMORY_DEFAULT_OBJECT(reset)

    template <typename... Args>
    explicit reset(type&, Args&&...)
    {}

    template <typename... Args>
    constexpr auto operator()(type&, Args&&...) const
    {}
};
}  // namespace operation
//
//--------------------------------------------------------------------------------------//
//
}  // namespace tim
//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_global_bundle, TIMEMORY_USER_GLOBAL_BUNDLE,
                                 "user_global_bundle", "global_bundle",
                                 "user_tuple_bundle", "tuple_bundle", "user_list_bundle",
                                 "list_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_ompt_bundle, TIMEMORY_USER_OMPT_BUNDLE,
                                 "user_ompt_bundle", "ompt_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_mpip_bundle, TIMEMORY_USER_MPIP_BUNDLE,
                                 "user_mpip_bundle", "mpip", "mpi_tools", "mpi")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_ncclp_bundle, TIMEMORY_USER_NCCLP_BUNDLE,
                                 "user_ncclp_bundle", "ncclp", "nccl_tools", "nccl")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_trace_bundle, TIMEMORY_USER_TRACE_BUNDLE,
                                 "user_trace_bundle", "trace_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_profiler_bundle, TIMEMORY_USER_PROFILER_BUNDLE,
                                 "user_profiler_bundle", "profiler_bundle")
//
TIMEMORY_PROPERTY_SPECIALIZATION(user_kokkosp_bundle, TIMEMORY_USER_KOKKOSP_BUNDLE,
                                 "user_kokkos_bundle", "kokkos_bundle",
                                 "user_kokkosp_bundle", "kokkosp_bundle")
//
TIMEMORY_METADATA_SPECIALIZATION(
    user_global_bundle, "user_global_bundle",
    "Generic bundle for inserting components at runtime",
    "Configure via TIMEMORY_GLOBAL_COMPONENTS [environment], "
    "settings::global_components() [string], or direct insertion")
//
TIMEMORY_METADATA_SPECIALIZATION(
    user_ompt_bundle, "user_ompt_bundle",
    "Generic bundle for inserting components at runtime into OMPT call-back system",
    "Configure via TIMEMORY_OMPT_COMPONENTS [environment], "
    "settings::ompt_components() [string], or direct insertion")
//
TIMEMORY_METADATA_SPECIALIZATION(
    user_mpip_bundle, "user_mpip_bundle",
    "Generic bundle for inserting components at runtime around MPI calls",
    "Configure via TIMEMORY_MPIP_COMPONENTS [environment], "
    "settings::mpip_components() [string], or direct insertion")
//
TIMEMORY_METADATA_SPECIALIZATION(
    user_ncclp_bundle, "user_ncclp_bundle",
    "Generic bundle for inserting components at runtime around NCCL calls",
    "Configure via TIMEMORY_NCCLP_COMPONENTS [environment], "
    "settings::ncclp_components() [string], or direct insertion")
//
TIMEMORY_METADATA_SPECIALIZATION(
    user_profiler_bundle, "user_profiler_bundle",
    "Generic bundle for inserting components at runtime around calls when profiling (via "
    "Python)",
    "Configure via TIMEMORY_PROFILER_COMPONENTS [environment], "
    "settings::profiler_components() [string], or direct insertion")
//
TIMEMORY_METADATA_SPECIALIZATION(
    user_trace_bundle, "user_trace_bundle",
    "Generic bundle for inserting components at runtime around calls when tracing (via "
    "Python or Dyninst)",
    "Configure via TIMEMORY_TRACE_COMPONENTS [environment], "
    "settings::trace_components() [string], or direct insertion")
//
TIMEMORY_METADATA_SPECIALIZATION(
    user_kokkosp_bundle, "user_kokkosp_bundle",
    "Generic bundle for inserting components into Kokkos profiling API",
    "Configure via TIMEMORY_KOKKOS_COMPONENTS [environment], "
    "settings::kokkos_components() [string], or direct insertion")
