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
#include "timemory/components/data_tracker/types.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/components/ompt/backends.hpp"
#include "timemory/components/ompt/macros.hpp"
#include "timemory/enum.h"
#include "timemory/macros/attributes.hpp"
#include "timemory/mpl/type_traits.hpp"

#include <type_traits>

TIMEMORY_DECLARE_TEMPLATE_COMPONENT(user_bundle, size_t Idx, typename Tag = TIMEMORY_API)
//
TIMEMORY_BUNDLE_INDEX(ompt_bundle_idx, 11110)
//
TIMEMORY_COMPONENT_ALIAS(user_ompt_bundle,
                         user_bundle<ompt_bundle_idx, project::timemory>)
//
//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_SET_COMPONENT_API(component::user_ompt_bundle, tpls::openmp, category::external,
                           os::agnostic)
//
namespace tim
{
namespace component
{
template <typename Api = TIMEMORY_API>
struct ompt_handle;

template <typename Api>
struct ompt_data_tracker;

struct ompt_target_data_tag
{};

using ompt_native_handle       = ompt_handle<TIMEMORY_API>;
using ompt_native_data_tracker = ompt_data_tracker<TIMEMORY_API>;
//
using ompt_data_tracker_t = data_tracker<int64_t, ompt_target_data_tag>;
//
}  // namespace component
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_USE_OMPT)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, tpls::openmp, false_type)
#endif
//
namespace tim
{
namespace trait
{
//
//--------------------------------------------------------------------------------------//
/// trait for configuring OMPT components
///
template <typename T>
struct ompt_handle
{
    using type =
        component_tuple<component::user_ompt_bundle, component::ompt_native_data_tracker>;
};
//
#if !defined(TIMEMORY_USE_OMPT)
//
template <typename Api>
struct is_available<component::ompt_handle<Api>> : false_type
{};
//
template <>
struct is_available<component::ompt_native_handle> : false_type
{};
//
template <typename Api>
struct is_available<component::ompt_data_tracker<Api>> : false_type
{};
//
template <>
struct is_available<component::ompt_native_data_tracker> : false_type
{};
//
#endif
//
}  // namespace trait
//
namespace openmp
{
//
//--------------------------------------------------------------------------------------//
//
enum class mode
{
    unspecified_callback,
    begin_callback,
    end_callback,
    store_callback,
    endpoint_callback
};
//
template <mode V>
using mode_constant = std::integral_constant<mode, V>;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::openmp::context_handler
/// \brief this struct provides the methods through which a unique identifier and
/// a label are generated for each OMPT callback.
///
template <typename Api = TIMEMORY_API>
struct context_handler;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::openmp::callback_connector
/// \brief this struct provides the routines through which timemory components
/// are applied to the callbacks
///
template <typename Components, typename Api = TIMEMORY_API>
struct callback_connector;
//
//--------------------------------------------------------------------------------------//
//
template <typename Components, typename ApiT>
struct callback_connector
{
    using api_type    = ApiT;
    using type        = Components;
    using handle_type = component::ompt_handle<api_type>;

    static bool is_enabled();
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::openmp::ompt_wrapper
/// \brief this struct provides the static callback function for OMPT which creates
/// a temporary instance of the connector, e.g. \ref openmp::callback_connector,
/// whose constructor creates a temporary instance of the context handler
/// in order to create a unique identifier and a label and then instruments the callback
/// based on the \tparam Mode template parameter.
///
template <typename Connector, mode Mode, typename... Args>
struct ompt_wrapper
{
    using api_type       = typename Connector::api_type;
    using component_type = typename Connector::type;

    static void callback(Args... args);
};
}  // namespace openmp

namespace ompt
{
using finalize_tool_func_t = void (*)(void);

template <typename ApiT>
finalize_tool_func_t
configure(ompt_function_lookup_t lookup, int, ompt_data_t*);
}  // namespace ompt
}  // namespace tim

TIMEMORY_PROPERTY_SPECIALIZATION(ompt_handle<TIMEMORY_API>, TIMEMORY_OMPT_HANDLE,
                                 "ompt_handle", "ompt", "ompt_handle", "openmp",
                                 "openmp_tools")

#if defined(TIMEMORY_USE_OMPT) && (!defined(TIMEMORY_OMPT_COMPONENT_HEADER_MODE) ||      \
                                   (defined(TIMEMORY_OMPT_COMPONENT_HEADER_MODE) &&      \
                                    TIMEMORY_OMPT_COMPONENT_HEADER_MODE == 0))

extern "C"
{
    extern ompt_start_tool_result_t* ompt_start_tool(
        unsigned int omp_version, const char* runtime_version) TIMEMORY_VISIBLE;
}

#endif
