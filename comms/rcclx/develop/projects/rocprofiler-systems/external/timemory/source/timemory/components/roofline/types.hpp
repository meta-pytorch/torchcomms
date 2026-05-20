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

#include "timemory/components/cuda/backends.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_TEMPLATE_COMPONENT(cpu_roofline, typename... Types)

TIMEMORY_DECLARE_TEMPLATE_COMPONENT(gpu_roofline, typename... Types)

#if !defined(TIMEMORY_USE_ROOFLINE)
#    if defined(TIMEMORY_USE_PAPI) ||                                                    \
        (defined(TIMEMORY_USE_CUPTI) && defined(TIMEMORY_USE_CUDA))
#        define TIMEMORY_USE_ROOFLINE 1
#    else
#        define TIMEMORY_USE_ROOFLINE 0
#    endif
#endif

namespace tim
{
namespace component
{
/// \typedef tim::component::cpu_roofline<float> tim::component::cpu_roofline_sp_flops
///
/// \brief A specialization of \ref tim::component::cpu_roofline for 32-bit floating point
/// operations
using cpu_roofline_sp_flops = cpu_roofline<float>;

/// \typedef tim::component::cpu_roofline<double> tim::component::cpu_roofline_dp_flops
///
/// \brief A specialization of \ref tim::component::cpu_roofline for 64-bit floating point
/// operations
using cpu_roofline_dp_flops = cpu_roofline<double>;

/// \typedef tim::component::cpu_roofline<float, double>
/// tim::component::cpu_roofline_flops
///
/// \brief A specialization of \ref tim::component::cpu_roofline for 32-bit and 64-bit
/// floating point operations
using cpu_roofline_flops = cpu_roofline<float, double>;

/// \typedef tim::component::gpu_roofline<float> tim::component::gpu_roofline_hp_flops
///
/// \brief A specialization of \ref tim::component::gpu_roofline for 16-bit floating point
/// operations (depending on availability).
using gpu_roofline_hp_flops = gpu_roofline<cuda::fp16_t>;

/// \typedef tim::component::gpu_roofline<float> tim::component::gpu_roofline_sp_flops
///
/// \brief A specialization of \ref tim::component::gpu_roofline for 32-bit floating point
/// operations
using gpu_roofline_sp_flops = gpu_roofline<float>;

/// \typedef tim::component::gpu_roofline<double> tim::component::gpu_roofline_dp_flops
///
/// \brief A specialization of \ref tim::component::gpu_roofline for 64-bit floating point
/// operations
using gpu_roofline_dp_flops = gpu_roofline<double>;

/// \typedef tim::component::gpu_roofline<float, double>
/// tim::component::gpu_roofline_flops
///
/// \brief A specialization of \ref tim::component::gpu_roofline for 16-bit (possibly),
/// 32-bit, and 64-bit floating point operations
#if defined(TIMEMORY_USE_CUDA_HALF)
using gpu_roofline_flops = gpu_roofline<cuda::fp16_t, float, double>;
#else
using gpu_roofline_flops = gpu_roofline<float, double>;
#endif
}  // namespace component
}  // namespace tim

//--------------------------------------------------------------------------------------//
//
//                                  APIs
//
//--------------------------------------------------------------------------------------//

TIMEMORY_SET_TEMPLATE_COMPONENT_API(TIMEMORY_ESC(typename... Types),
                                    TIMEMORY_ESC(component::cpu_roofline<Types...>),
                                    tpls::papi, category::external,
                                    category::hardware_counter, category::timing,
                                    os::supports_linux)

TIMEMORY_SET_TEMPLATE_COMPONENT_API(TIMEMORY_ESC(typename... Types),
                                    TIMEMORY_ESC(component::gpu_roofline<Types...>),
                                    tpls::nvidia, category::external,
                                    category::hardware_counter, category::timing,
                                    os::agnostic)

//--------------------------------------------------------------------------------------//
//
//                              STATISTICS
//
//--------------------------------------------------------------------------------------//

TIMEMORY_VARIADIC_STATISTICS_TYPE(component::cpu_roofline, std::vector<double>, typename)
TIMEMORY_VARIADIC_STATISTICS_TYPE(component::gpu_roofline, std::vector<double>, typename)

//--------------------------------------------------------------------------------------//
//
//                              IS AVAILABLE
//
//--------------------------------------------------------------------------------------//
//
//      PAPI
//
#if !defined(TIMEMORY_USE_PAPI) ||                                                       \
    (defined(TIMEMORY_USE_ROOFLINE) && TIMEMORY_USE_ROOFLINE == 0)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::cpu_roofline, false_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_sp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::cpu_roofline_dp_flops, false_type)
#endif

//
//      CUDA and CUPTI
//
#if !defined(TIMEMORY_USE_CUPTI) || !defined(TIMEMORY_USE_CUDA) ||                       \
    (defined(TIMEMORY_USE_ROOFLINE) && TIMEMORY_USE_ROOFLINE == 0)
TIMEMORY_DEFINE_VARIADIC_TRAIT(is_available, component::gpu_roofline, false_type,
                               typename)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_hp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_sp_flops, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_dp_flops, false_type)
#elif !defined(TIMEMORY_USE_CUDA_HALF)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::gpu_roofline_hp_flops, false_type)
#endif

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM UNIT PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_unit_printing, component::cpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_unit_printing, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM LABEL PRINTING
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_label_printing, component::cpu_roofline, true_type,
                               typename)
TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_label_printing, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              ARRAY SERIALIZATION
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(array_serialization, component::cpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              REQUIRES JSON
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(requires_json, component::cpu_roofline, true_type,
                               typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(requires_json, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              SUPPORTS CUSTOM RECORD
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(supports_custom_record, component::cpu_roofline, true_type,
                               typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(supports_custom_record, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              ITERABLE MEASUREMENT
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(iterable_measurement, component::cpu_roofline, true_type,
                               typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(iterable_measurement, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              CUSTOM SERIALIZATION
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_serialization, component::cpu_roofline, true_type,
                               typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(custom_serialization, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              SECONDARY DATA
//
//--------------------------------------------------------------------------------------//

TIMEMORY_DEFINE_VARIADIC_TRAIT(secondary_data, component::gpu_roofline, true_type,
                               typename)

//--------------------------------------------------------------------------------------//
//
//                              FINALIZATION PRIORITY
//
//--------------------------------------------------------------------------------------//

// finalize early
TIMEMORY_DEFINE_VARIADIC_TRAIT(fini_priority, component::cpu_roofline,
                               priority_constant<-4>, typename)

TIMEMORY_DEFINE_VARIADIC_TRAIT(fini_priority, component::gpu_roofline,
                               priority_constant<-4>, typename)

//--------------------------------------------------------------------------------------//
//
//                              UNITS SPECIALIZATIONS
//
//--------------------------------------------------------------------------------------//

namespace tim
{
namespace trait
{
//
template <typename... Types>
struct units<component::cpu_roofline<Types...>>
{
    using type         = double;
    using display_type = std::vector<std::string>;
};
//
}  // namespace trait
}  // namespace tim

//
//======================================================================================//
//
TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_dp_flops, TIMEMORY_CPU_ROOFLINE_DP_FLOPS,
                                 "cpu_roofline_dp_flops", "cpu_roofline_dp",
                                 "cpu_roofline_double")

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_flops, TIMEMORY_CPU_ROOFLINE_FLOPS,
                                 "cpu_roofline_flops", "cpu_roofline")

TIMEMORY_PROPERTY_SPECIALIZATION(cpu_roofline_sp_flops, TIMEMORY_CPU_ROOFLINE_SP_FLOPS,
                                 "cpu_roofline_sp_flops", "cpu_roofline_sp",
                                 "cpu_roofline_single")

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_dp_flops, TIMEMORY_GPU_ROOFLINE_DP_FLOPS,
                                 "gpu_roofline_dp_flops", "gpu_roofline_dp",
                                 "gpu_roofline_double")

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_flops, TIMEMORY_GPU_ROOFLINE_FLOPS,
                                 "gpu_roofline_flops", "gpu_roofline")

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_hp_flops, TIMEMORY_GPU_ROOFLINE_HP_FLOPS,
                                 "gpu_roofline_hp_flops", "gpu_roofline_hp",
                                 "gpu_roofline_half")

TIMEMORY_PROPERTY_SPECIALIZATION(gpu_roofline_sp_flops, TIMEMORY_GPU_ROOFLINE_SP_FLOPS,
                                 "gpu_roofline_sp_flops", "gpu_roofline_sp",
                                 "gpu_roofline_single")
//
//======================================================================================//
//
