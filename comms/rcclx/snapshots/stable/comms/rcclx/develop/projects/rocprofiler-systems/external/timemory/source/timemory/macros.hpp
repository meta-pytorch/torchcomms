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

#include "timemory/compat/macros.h"
#include "timemory/macros/arch.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/macros.hpp"
#include "timemory/utility/macros.hpp"

//======================================================================================//
//
//      PROTECTS COMMAS FROM BEING DELIMITED
//
//          TIMEMORY_STATISTICS_TYPE(foo, std::array<size_t, 10>)
//
//      should be
//
//          TIMEMORY_STATISTICS_TYPE(foo, TIMEMORY_ESC(std::array<size_t, 10>))
//
//======================================================================================//

#if !defined(TIMEMORY_ESC)
#    define TIMEMORY_ESC(...) __VA_ARGS__
#endif

// stringify some macro -- uses TIMEMORY_STRINGIZE2 which does the actual
//   "stringify-ing" after the macro has been substituted by it's result
#if !defined(TIMEMORY_STRINGIZE)
#    define TIMEMORY_STRINGIZE(X) TIMEMORY_STRINGIZE2(X)
#endif

// actual stringifying
#if !defined(TIMEMORY_STRINGIZE2)
#    define TIMEMORY_STRINGIZE2(X) #    X
#endif

//======================================================================================//
//
/// Simple way to define a static variable which avoids the static initialization fiasco
#if !defined(TIMEMORY_STATIC_ACCESSOR)
#    define TIMEMORY_STATIC_ACCESSOR(RETURN_TYPE, FUNC_NAME, ...)                        \
        static RETURN_TYPE& FUNC_NAME()                                                  \
        {                                                                                \
            static RETURN_TYPE _v = __VA_ARGS__;                                         \
            return _v;                                                                   \
        }
#endif

/// Simple way to define a static thread-local variable which avoids the static
/// initialization fiasco
#if !defined(TIMEMORY_STATIC_THREAD_LOCAL_ACCESSOR)
#    define TIMEMORY_STATIC_THREAD_LOCAL_ACCESSOR(RETURN_TYPE, FUNC_NAME, ...)           \
        static RETURN_TYPE& FUNC_NAME()                                                  \
        {                                                                                \
            static thread_local RETURN_TYPE _v = __VA_ARGS__;                            \
            return _v;                                                                   \
        }
#endif

//======================================================================================//
//
//                              COMPONENTS
//
//======================================================================================//

#if !defined(TIMEMORY_FORWARD_DECLARE_COMPONENT)
/// use this macro for forward declarations. Using \ref TIMEMORY_DECLARE_COMPONENT
/// on a pre-existing type will fail because of is_component specialization
#    define TIMEMORY_FORWARD_DECLARE_COMPONENT(NAME)                                     \
        namespace tim                                                                    \
        {                                                                                \
        namespace component                                                              \
        {                                                                                \
        struct NAME;                                                                     \
        }                                                                                \
        }
#endif

//======================================================================================//
//
//      STATISTICS TYPE-TRAIT SPECIALIZATION
//
//======================================================================================//

#if !defined(TIMEMORY_STATISTICS_TYPE)
#    define TIMEMORY_STATISTICS_TYPE(COMPONENT, TYPE)                                    \
        TIMEMORY_TRAIT_TYPE(statistics, TIMEMORY_ESC(COMPONENT), TIMEMORY_ESC(TYPE))
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_TEMPLATE_STATISTICS_TYPE)
#    define TIMEMORY_TEMPLATE_STATISTICS_TYPE(COMPONENT, TYPE, TEMPLATE_TYPE)            \
        TIMEMORY_TEMPLATE_TRAIT_TYPE(statistics, TIMEMORY_ESC(COMPONENT),                \
                                     TIMEMORY_ESC(TEMPLATE_TYPE T), TIMEMORY_ESC(T),     \
                                     TIMEMORY_ESC(TYPE))
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_VARIADIC_STATISTICS_TYPE)
#    define TIMEMORY_VARIADIC_STATISTICS_TYPE(COMPONENT, TYPE, TEMPLATE_TYPE)            \
        TIMEMORY_VARIADIC_TRAIT_TYPE(statistics, TIMEMORY_ESC(COMPONENT),                \
                                     TIMEMORY_ESC(TEMPLATE_TYPE... T),                   \
                                     TIMEMORY_ESC(T...), TIMEMORY_ESC(TYPE))
#endif

//======================================================================================//
//
//                              GOTCHA
//
//======================================================================================//
//
#if defined(TIMEMORY_USE_GOTCHA)
//
//--------------------------------------------------------------------------------------//
//
/// \macro TIMEMORY_C_GOTCHA
/// \brief attempt to generate a GOTCHA wrapper for a C function (unmangled)
///
#    if !defined(TIMEMORY_C_GOTCHA)
#        define TIMEMORY_C_GOTCHA(type, idx, func)                                       \
            type::template instrument<                                                   \
                idx, typename ::tim::mpl::function_traits<decltype(func)>::result_type,  \
                typename ::tim::mpl::function_traits<decltype(func)>::call_type>::       \
                generate(TIMEMORY_STRINGIZE(func))
#    endif
//
//--------------------------------------------------------------------------------------//
//
/// \macro
/// \brief TIMEMORY_C_GOTCHA + ability to pass priority and tool name
///
#    if !defined(TIMEMORY_C_GOTCHA_TOOL)
#        define TIMEMORY_C_GOTCHA_TOOL(type, idx, func, ...)                             \
            type::template instrument<                                                   \
                idx, typename ::tim::mpl::function_traits<decltype(func)>::result_type,  \
                typename ::tim::mpl::function_traits<decltype(func)>::call_type>::       \
                generate(TIMEMORY_STRINGIZE(func), __VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
/// \macro TIMEMORY_CXX_GOTCHA
/// \brief attempt to generate a GOTCHA wrapper for a C++ function by mangling the
/// function name in general, mangling template function is not supported
///
#    if !defined(TIMEMORY_CXX_GOTCHA)
#        define TIMEMORY_CXX_GOTCHA(type, idx, func)                                     \
            type::template instrument<                                                   \
                idx, typename ::tim::mpl::function_traits<decltype(func)>::result_type,  \
                typename ::tim::mpl::function_traits<decltype(func)>::call_type>::       \
                generate(::tim::mangle<decltype(func)>(TIMEMORY_STRINGIZE(func)))
#    endif
//
//--------------------------------------------------------------------------------------//
//
/// \macro TIMEMORY_CXX_GOTCHA_TOOL
/// \brief TIMEMORY_CXX_GOTCHA + ability to pass priority and tool name
///
#    if !defined(TIMEMORY_CXX_GOTCHA_TOOL)
#        define TIMEMORY_CXX_GOTCHA_TOOL(type, idx, func, ...)                           \
            type::template instrument<                                                   \
                idx, typename ::tim::mpl::function_traits<decltype(func)>::result_type,  \
                typename ::tim::mpl::function_traits<decltype(func)>::call_type>::       \
                generate(::tim::mangle<decltype(func)>(TIMEMORY_STRINGIZE(func)),        \
                         __VA_ARGS__)
#    endif
//
//--------------------------------------------------------------------------------------//
//
/// \macro TIMEMORY_CXX_GOTCHA_MEMFUN
/// \brief attempt to generate a GOTCHA wrapper for a C++ function by mangling the
/// function name in general, mangling template function is not supported
///
#    if !defined(TIMEMORY_CXX_GOTCHA_MEMFUN)
#        define TIMEMORY_CXX_GOTCHA_MEMFUN(type, idx, func)                              \
            type::template instrument<                                                   \
                idx, typename ::tim::mpl::function_traits<decltype(&func)>::result_type, \
                typename ::tim::mpl::function_traits<decltype(&func)>::call_type>::      \
                generate(::tim::mangle<decltype(&func)>(TIMEMORY_STRINGIZE(func)))
#    endif
//
//--------------------------------------------------------------------------------------//
//
/// \macro TIMEMORY_DERIVED_GOTCHA
/// \brief generate a GOTCHA wrapper for function with identical args but different name
/// -- useful for C++ template function where the mangled name is determined
///    via `nm --dynamic <EXE>`
///
#    if !defined(TIMEMORY_DERIVED_GOTCHA)
#        define TIMEMORY_DERIVED_GOTCHA(type, idx, func, ...)                            \
            type::template instrument<                                                   \
                idx, typename ::tim::mpl::function_traits<decltype(func)>::result_type,  \
                typename ::tim::mpl::function_traits<decltype(func)>::call_type>::       \
                generate(__VA_ARGS__)
#    endif
//
#else
//
#    if !defined(TIMEMORY_C_GOTCHA)
#        define TIMEMORY_C_GOTCHA(...)
#    endif
#    if !defined(TIMEMORY_C_GOTCHA_TOOL)
#        define TIMEMORY_C_GOTCHA_TOOL(...)
#    endif
#    if !defined(TIMEMORY_CXX_GOTCHA)
#        define TIMEMORY_CXX_GOTCHA(...)
#    endif
#    if !defined(TIMEMORY_CXX_GOTCHA_TOOL)
#        define TIMEMORY_CXX_GOTCHA_TOOL(...)
#    endif
#    if !defined(TIMEMORY_CXX_GOTCHA_MEMFUN)
#        define TIMEMORY_CXX_GOTCHA_MEMFUN(...)
#    endif
#    if !defined(TIMEMORY_DERIVED_GOTCHA)
#        define TIMEMORY_DERIVED_GOTCHA(...)
#    endif
//
#endif
//
//  backwards compatibility
//
#if !defined(TIMEMORY_CXX_MEMFUN_GOTCHA)
#    define TIMEMORY_CXX_MEMFUN_GOTCHA(...) TIMEMORY_CXX_GOTCHA_MEMFUN(__VA_ARGS__)
#endif

//======================================================================================//
//
//                              CUPTI
//
//======================================================================================//
//
#if defined(TIMEMORY_USE_CUPTI)

//--------------------------------------------------------------------------------------//

#    if !defined(TIMEMORY_CUDA_DRIVER_API_CALL)
#        define TIMEMORY_CUDA_DRIVER_API_CALL(apiFuncCall)                               \
            {                                                                            \
                CUresult _status = apiFuncCall;                                          \
                if(_status != CUDA_SUCCESS)                                              \
                {                                                                        \
                    TIMEMORY_PRINTF_WARNING(                                             \
                        stderr, "%s:%d: error: function '%s' failed with error: %d.\n",  \
                        __FILE__, __LINE__, #apiFuncCall, _status);                      \
                }                                                                        \
            }
#    endif

//--------------------------------------------------------------------------------------//

#    if !defined(TIMEMORY_CUPTI_CALL)
#        define TIMEMORY_CUPTI_CALL(call)                                                \
            {                                                                            \
                CUptiResult _status = call;                                              \
                if(_status != CUPTI_SUCCESS)                                             \
                {                                                                        \
                    const char* errstr;                                                  \
                    cuptiGetResultString(_status, &errstr);                              \
                    TIMEMORY_PRINTF_WARNING(                                             \
                        stderr, "%s:%d: error: function '%s' failed with error: %s.\n",  \
                        __FILE__, __LINE__, #call, errstr);                              \
                }                                                                        \
            }
#    endif

//--------------------------------------------------------------------------------------//

#else  // !defined(TIMEMORY_USE_CUPTI)
//
#    if !defined(TIMEMORY_CUDA_DRIVER_API_CALL)
#        define TIMEMORY_CUDA_DRIVER_API_CALL(...)
#    endif
//
#    if !defined(TIMEMORY_CUPTI_CALL)
#        define TIMEMORY_CUPTI_CALL(...)
#    endif
//
#endif  // !defined(TIMEMORY_USE_CUPTI)

//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_CUPTI_BUFFER_SIZE)
#    define TIMEMORY_CUPTI_BUFFER_SIZE (32 * 1024)
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_CUPTI_ALIGN_SIZE)
#    define TIMEMORY_CUPTI_ALIGN_SIZE (8)
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_CUPTI_ALIGN_BUFFER)
#    define TIMEMORY_CUPTI_ALIGN_BUFFER(buffer, align)                                   \
        (((uintptr_t)(buffer) & ((align) -1))                                            \
             ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align) -1)))               \
             : (buffer))
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_CUPTI_PROFILER_NAME_SHORT)
#    define TIMEMORY_CUPTI_PROFILER_NAME_SHORT 128
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_CUPTI_PROFILER_NAME_LONG)
#    define TIMEMORY_CUPTI_PROFILER_NAME_LONG 512
#endif

//======================================================================================//
//
//                              CUDA
//
//======================================================================================//
//
#if defined(TIMEMORY_USE_CUDA)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_CUDA_RUNTIME_API_CALL)
#        define TIMEMORY_CUDA_RUNTIME_API_CALL(apiFuncCall)                              \
            {                                                                            \
                ::tim::cuda::error_t err = apiFuncCall;                                  \
                if(err != ::tim::cuda::success_v && (int) err != 0)                      \
                {                                                                        \
                    TIMEMORY_PRINTF_WARNING(                                             \
                        stderr, "%s:%d: error: function '%s' failed with error: %s.\n",  \
                        __FILE__, __LINE__, #apiFuncCall,                                \
                        ::tim::cuda::get_error_string(err));                             \
                }                                                                        \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_CUDA_RUNTIME_API_CALL_THROW)
#        define TIMEMORY_CUDA_RUNTIME_API_CALL_THROW(apiFuncCall)                        \
            {                                                                            \
                ::tim::cuda::error_t err = apiFuncCall;                                  \
                if(err != ::tim::cuda::success_v && (int) err != 0)                      \
                {                                                                        \
                    char errmsg[std::numeric_limits<uint16_t>::max()];                   \
                    snprintf(errmsg, std::numeric_limits<uint16_t>::max() - 1,           \
                             "%s:%d: error: function '%s' failed with error: %s.\n",     \
                             __FILE__, __LINE__, #apiFuncCall,                           \
                             ::tim::cuda::get_error_string(err));                        \
                    throw std::runtime_error(errmsg);                                    \
                }                                                                        \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_CUDA_RUNTIME_CHECK_ERROR)
#        define TIMEMORY_CUDA_RUNTIME_CHECK_ERROR(err)                                   \
            {                                                                            \
                if(err != ::tim::cuda::success_v && (int) err != 0)                      \
                {                                                                        \
                    TIMEMORY_PRINTF_WARNING(                                             \
                        stderr, "%s:%d: error check failed with: code %i -- %s.\n",      \
                        __FILE__, __LINE__, (int) err,                                   \
                        ::tim::cuda::get_error_string(err));                             \
                }                                                                        \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#else
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_CUDA_RUNTIME_API_CALL)
#        define TIMEMORY_CUDA_RUNTIME_API_CALL(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_CUDA_RUNTIME_API_CALL_THROW)
#        define TIMEMORY_CUDA_RUNTIME_API_CALL_THROW(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_CUDA_RUNTIME_CHECK_ERROR)
#        define TIMEMORY_CUDA_RUNTIME_CHECK_ERROR(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#endif

#if defined(TIMEMORY_USE_NVML)
#    if !defined(TIMEMORY_NVML_RUNTIME_CHECK_ERROR)
#        define TIMEMORY_NVML_RUNTIME_CHECK_ERROR(apiFuncCall, ...)                      \
            {                                                                            \
                auto err = apiFuncCall;                                                  \
                if(err != NVML_SUCCESS && (int) err != 0)                                \
                {                                                                        \
                    TIMEMORY_PRINTF_WARNING(                                             \
                        stderr,                                                          \
                        "%s:%d: error check failed with: code %i -- %s.\nFunction "      \
                        "call: %s\n",                                                    \
                        __FILE__, __LINE__, (int) err, nvmlErrorString(err),             \
                        #apiFuncCall);                                                   \
                    __VA_ARGS__;                                                         \
                }                                                                        \
            }
#    endif
#else
#    if !defined(TIMEMORY_NVML_RUNTIME_CHECK_ERROR)
#        define TIMEMORY_NVML_RUNTIME_CHECK_ERROR(...)
#    endif
#endif

//======================================================================================//
//
//                              HIP
//
//======================================================================================//
//
#if defined(TIMEMORY_USE_HIP)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_HIP_RUNTIME_API_CALL)
#        define TIMEMORY_HIP_RUNTIME_API_CALL(apiFuncCall)                               \
            {                                                                            \
                ::tim::hip::error_t err = apiFuncCall;                                   \
                if(err != ::tim::hip::success_v && (int) err != 0)                       \
                {                                                                        \
                    TIMEMORY_PRINTF_WARNING(                                             \
                        stderr, "%s:%d: error: function '%s' failed with error: %s.\n",  \
                        __FILE__, __LINE__, #apiFuncCall,                                \
                        ::tim::hip::get_error_string(err));                              \
                }                                                                        \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_HIP_RUNTIME_API_CALL_THROW)
#        define TIMEMORY_HIP_RUNTIME_API_CALL_THROW(apiFuncCall)                         \
            {                                                                            \
                ::tim::hip::error_t err = apiFuncCall;                                   \
                if(err != ::tim::hip::success_v && (int) err != 0)                       \
                {                                                                        \
                    char errmsg[std::numeric_limits<uint16_t>::max()];                   \
                    snprintf(errmsg, std::numeric_limits<uint16_t>::max() - 1,           \
                             "%s:%d: error: function '%s' failed with error: %s.\n",     \
                             __FILE__, __LINE__, #apiFuncCall,                           \
                             ::tim::hip::get_error_string(err));                         \
                    throw std::runtime_error(errmsg);                                    \
                }                                                                        \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_HIP_RUNTIME_CHECK_ERROR)
#        define TIMEMORY_HIP_RUNTIME_CHECK_ERROR(err)                                    \
            {                                                                            \
                if(err != ::tim::hip::success_v && (int) err != 0)                       \
                {                                                                        \
                    TIMEMORY_PRINTF_WARNING(                                             \
                        stderr, "%s:%d: error check failed with: code %i -- %s.\n",      \
                        __FILE__, __LINE__, (int) err,                                   \
                        ::tim::hip::get_error_string(err));                              \
                }                                                                        \
            }
#    endif
//
//--------------------------------------------------------------------------------------//
//
#else
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_HIP_RUNTIME_API_CALL)
#        define TIMEMORY_HIP_RUNTIME_API_CALL(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_HIP_RUNTIME_API_CALL_THROW)
#        define TIMEMORY_HIP_RUNTIME_API_CALL_THROW(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(TIMEMORY_HIP_RUNTIME_CHECK_ERROR)
#        define TIMEMORY_HIP_RUNTIME_CHECK_ERROR(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#endif

//======================================================================================//
//
//                              GPU
//
//======================================================================================//
//
#if !defined(TIMEMORY_GPU_RUNTIME_API_CALL)
#    if defined(TIMEMORY_USE_CUDA)
#        define TIMEMORY_GPU_RUNTIME_API_CALL(...)                                       \
            TIMEMORY_CUDA_RUNTIME_API_CALL(__VA_ARGS__)
#    elif defined(TIMEMORY_USE_HIP)
#        define TIMEMORY_GPU_RUNTIME_API_CALL(...)                                       \
            TIMEMORY_HIP_RUNTIME_API_CALL(__VA_ARGS__)
#    else
#        define TIMEMORY_GPU_RUNTIME_API_CALL(...)
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_GPU_RUNTIME_API_CALL_THROW)
#    if defined(TIMEMORY_USE_CUDA)
#        define TIMEMORY_GPU_RUNTIME_API_CALL_THROW(...)                                 \
            TIMEMORY_CUDA_RUNTIME_API_CALL_THROW(__VA_ARGS__)
#    elif defined(TIMEMORY_USE_HIP)
#        define TIMEMORY_GPU_RUNTIME_API_CALL_THROW(...)                                 \
            TIMEMORY_HIP_RUNTIME_API_CALL_THROW(__VA_ARGS__)
#    else
#        define TIMEMORY_GPU_RUNTIME_API_CALL_THROW(...)
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//
#if !defined(TIMEMORY_GPU_RUNTIME_CHECK_ERROR)
#    if defined(TIMEMORY_USE_CUDA)
#        define TIMEMORY_GPU_RUNTIME_CHECK_ERROR(...)                                    \
            TIMEMORY_CUDA_RUNTIME_CHECK_ERROR(__VA_ARGS__)
#    elif defined(TIMEMORY_USE_HIP)
#        define TIMEMORY_GPU_RUNTIME_CHECK_ERROR(...)                                    \
            TIMEMORY_HIP_RUNTIME_CHECK_ERROR(__VA_ARGS__)
#    else
#        define TIMEMORY_GPU_RUNTIME_CHECK_ERROR(...)
#    endif
#endif
//
//--------------------------------------------------------------------------------------//
//

//======================================================================================//
//
//                              LIKWID
//
//======================================================================================//
//
#if !defined(TIMEMORY_USE_LIKWID)
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_MARKER_INIT)
#        define LIKWID_MARKER_INIT
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_MARKER_THREADINIT)
#        define LIKWID_MARKER_THREADINIT
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_MARKER_SWITCH)
#        define LIKWID_MARKER_SWITCH
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_MARKER_REGISTER)
#        define LIKWID_MARKER_REGISTER(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_MARKER_CLOSE)
#        define LIKWID_MARKER_CLOSE
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_MARKER_GET)
#        define LIKWID_MARKER_GET(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_MARKER_RESET)
#        define LIKWID_MARKER_RESET(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_WITH_NVMON)
#        define LIKWID_NVMARKER_INIT
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_WITH_NVMON)
#        define LIKWID_NVMARKER_THREADINIT
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_WITH_NVMON)
#        define LIKWID_NVMARKER_SWITCH
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_WITH_NVMON)
#        define LIKWID_NVMARKER_REGISTER(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_WITH_NVMON)
#        define LIKWID_NVMARKER_CLOSE
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_WITH_NVMON)
#        define LIKWID_NVMARKER_GET(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#    if !defined(LIKWID_WITH_NVMON)
#        define LIKWID_NVMARKER_RESET(...)
#    endif
//
//--------------------------------------------------------------------------------------//
//
#endif

//======================================================================================//
//
//                                      OTHERS
//
//======================================================================================//
//
#include "timemory/components/macros.hpp"
#include "timemory/variadic/macros.hpp"
