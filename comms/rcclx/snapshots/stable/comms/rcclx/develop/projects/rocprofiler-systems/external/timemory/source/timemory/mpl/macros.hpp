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

#include <type_traits>

#if !defined(TIMEMORY_ESC)
#    define TIMEMORY_ESC(...) __VA_ARGS__
#endif

//======================================================================================//
//
//      DECLARE TYPE-TRAIT TYPE
//
//      e.g. TIMEMORY_DECLARE_TYPE_TRAIT(pretty_archive, typename T)
//
//======================================================================================//

#if !defined(TIMEMORY_DECLARE_TYPE_TRAIT)
#    define TIMEMORY_DECLARE_TYPE_TRAIT(NAME, ...)                                       \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <__VA_ARGS__>                                                           \
        struct NAME;                                                                     \
        }                                                                                \
        }
#endif

//======================================================================================//
//
//      GENERIC TYPE-TRAIT SPECIALIZATION (for true_type/false_type traits)
//
//======================================================================================//

#if !defined(TIMEMORY_DEFINE_CONCRETE_TRAIT)
#    define TIMEMORY_DEFINE_CONCRETE_TRAIT(TRAIT, TYPE, VALUE)                           \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct TRAIT<TYPE> : VALUE                                                       \
        {};                                                                              \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_DEFINE_TEMPLATE_TRAIT)
#    define TIMEMORY_DEFINE_TEMPLATE_TRAIT(TRAIT, TYPE, VALUE, TEMPLATE_TYPE)            \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TEMPLATE_TYPE T>                                                       \
        struct TRAIT<TYPE<T>> : VALUE                                                    \
        {};                                                                              \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_DEFINE_VARIADIC_TRAIT)
#    define TIMEMORY_DEFINE_VARIADIC_TRAIT(TRAIT, TYPE, VALUE, TEMPLATE_TYPE)            \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TEMPLATE_TYPE... T>                                                    \
        struct TRAIT<TYPE<T...>> : VALUE                                                 \
        {};                                                                              \
        }                                                                                \
        }
#endif

//======================================================================================//
//
//      GENERIC TYPE-TRAIT SPECIALIZATION (for defining ::type traits)
//
//======================================================================================//

#if !defined(TIMEMORY_TRAIT_TYPE)
#    define TIMEMORY_TRAIT_TYPE(TRAIT, TYPE, ...)                                        \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <>                                                                      \
        struct TRAIT<TYPE>                                                               \
        {                                                                                \
            using type = __VA_ARGS__;                                                    \
        };                                                                               \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_TEMPLATE_TRAIT_TYPE)
#    define TIMEMORY_TEMPLATE_TRAIT_TYPE(TRAIT, TYPE, TEMPLATE_PARAM, TEMPLATE_ARG, ...) \
        namespace tim                                                                    \
        {                                                                                \
        namespace trait                                                                  \
        {                                                                                \
        template <TEMPLATE_PARAM>                                                        \
        struct TRAIT<TYPE<TEMPLATE_ARG>>                                                 \
        {                                                                                \
            using type = __VA_ARGS__;                                                    \
        };                                                                               \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

#if !defined(TIMEMORY_VARIADIC_TRAIT_TYPE)
#    define TIMEMORY_VARIADIC_TRAIT_TYPE(TRAIT, TYPE, TEMPLATE_PARAM, TEMPLATE_ARG, ...) \
        TIMEMORY_TEMPLATE_TRAIT_TYPE(TRAIT, TYPE, TIMEMORY_ESC(TEMPLATE_PARAM),          \
                                     TIMEMORY_ESC(TEMPLATE_ARG), __VA_ARGS__)
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
using true_type  = std::true_type;
using false_type = std::false_type;
}  // namespace tim
