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

#include "timemory/macros/attributes.hpp"
#include "timemory/math/abs.hpp"
#include "timemory/math/assign.hpp"
#include "timemory/math/divide.hpp"
#include "timemory/math/fwd.hpp"
#include "timemory/math/max.hpp"
#include "timemory/math/min.hpp"
#include "timemory/math/minus.hpp"
#include "timemory/math/multiply.hpp"
#include "timemory/math/percent_diff.hpp"
#include "timemory/math/plus.hpp"
#include "timemory/math/pow.hpp"
#include "timemory/math/sqr.hpp"
#include "timemory/math/sqrt.hpp"

namespace tim
{
namespace math
{
/// \struct tim::math::compute
/// \brief Struct for performing math operations on complex data structures without using
/// globally overload operators (e.g. `lhs += rhs`) and generic functions (`lhs =
/// abs(rhs)`)
///
template <typename Tp, typename Up = Tp>
struct compute
{
    using this_type  = compute<Tp, Up>;
    using type       = Tp;
    using value_type = Up;

    static TIMEMORY_INLINE decltype(auto) abs(const type& _v)
    {
        return this_type::abs(_v, 0);
    }

    static TIMEMORY_INLINE decltype(auto) sqr(const type& _v)
    {
        return this_type::sqr(_v, 0);
    }

    static TIMEMORY_INLINE decltype(auto) sqrt(const type& _v)
    {
        return this_type::sqrt(_v, 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) min(const type& _l, V&& _r)
    {
        return this_type::min(_l, std::forward<V>(_r), 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) max(const type& _l, V&& _r)
    {
        return this_type::max(_l, std::forward<V>(_r), 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) percent_diff(const type& _l, V&& _r)
    {
        return this_type::percent_diff(_l, std::forward<V>(_r), 0);
    }

    // reference
    template <typename V>
    static TIMEMORY_INLINE decltype(auto) plus(type& _l, V&& _r)
    {
        return this_type::plus(_l, std::forward<V>(_r), 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) minus(type& _l, V&& _r)
    {
        return this_type::minus(_l, std::forward<V>(_r), 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) multiply(type& _l, V&& _r)
    {
        return this_type::multiply(_l, std::forward<V>(_r), 0);
    }

    template <typename V>
    static TIMEMORY_INLINE decltype(auto) divide(type& _l, V&& _r)
    {
        return this_type::divide(_l, std::forward<V>(_r), 0);
    }

    // const ref
    template <typename V>
    static TIMEMORY_INLINE auto plus(const type& _l, V&& _r)
    {
        type _t{ _l };
        return this_type::plus(_t, std::forward<V>(_r), 0);
    }

    template <typename V>
    static TIMEMORY_INLINE auto minus(const type& _l, V&& _r)
    {
        type _t{ _l };
        return this_type::minus(_t, std::forward<V>(_r), 0);
    }

    template <typename V>
    static TIMEMORY_INLINE auto multiply(const type& _l, V&& _r)
    {
        type _t{ _l };
        return this_type::multiply(_t, std::forward<V>(_r), 0);
    }

    template <typename V>
    static TIMEMORY_INLINE auto divide(const type& _l, V&& _r)
    {
        type _t{ _l };
        return this_type::divide(_t, std::forward<V>(_r), 0);
    }

private:
    //----------------------------------------------------------------------------------//
    // tim::math overload available
    //
    template <typename V>
    static TIMEMORY_INLINE auto abs(V&& _v, int)
        -> decltype(::tim::math::abs(std::forward<V>(_v)))
    {
        return ::tim::math::abs(std::forward<V>(_v));
    }

    template <typename V>
    static TIMEMORY_INLINE auto sqr(V&& _v, int)
        -> decltype(::tim::math::sqr(std::forward<V>(_v)))
    {
        return ::tim::math::sqr(std::forward<V>(_v));
    }

    template <typename V>
    static TIMEMORY_INLINE auto sqrt(V&& _v, int)
        -> decltype(::tim::math::sqrt(std::forward<V>(_v)))
    {
        return ::tim::math::sqrt(std::forward<V>(_v));
    }

    template <typename V>
    static TIMEMORY_INLINE auto min(const type& _l, V&& _r, int)
        -> decltype(::tim::math::min(_l, std::forward<V>(_r),
                                     get_index_sequence<type>::value),
                    ::tim::math::min(_l, std::forward<V>(_r)))
    {
        return ::tim::math::min(_l, std::forward<V>(_r));
    }

    template <typename V>
    static TIMEMORY_INLINE auto max(const type& _l, V&& _r, int)
        -> decltype(::tim::math::max(_l, std::forward<V>(_r),
                                     get_index_sequence<type>::value),
                    ::tim::math::max(_l, std::forward<V>(_r)))
    {
        return ::tim::math::max(_l, std::forward<V>(_r));
    }

    template <typename V>
    static TIMEMORY_INLINE auto percent_diff(const type& _l, V&& _r, int)
        -> decltype(::tim::math::percent_diff(_l, std::forward<V>(_r),
                                              get_index_sequence<type>::value, 0),
                    ::tim::math::percent_diff(_l, std::forward<V>(_r)))
    {
        return ::tim::math::percent_diff(_l, std::forward<V>(_r));
    }

    template <typename V>
    static TIMEMORY_INLINE auto plus(type& _l, V&& _r, int)
        -> decltype(::tim::math::plus(_l, std::forward<V>(_r),
                                      get_index_sequence<type>::value, 0),
                    std::declval<type&>())
    {
        return ::tim::math::plus(_l, std::forward<V>(_r));
    }

    template <typename V>
    static TIMEMORY_INLINE auto minus(type& _l, V&& _r, int)
        -> decltype(::tim::math::minus(_l, std::forward<V>(_r),
                                       get_index_sequence<type>::value, 0),
                    std::declval<type&>())
    {
        return ::tim::math::minus(_l, std::forward<V>(_r));
    }

    template <typename V>
    static TIMEMORY_INLINE auto multiply(type& _l, V&& _r, int)
        -> decltype(::tim::math::multiply(_l, std::forward<V>(_r),
                                          get_index_sequence<type>::value, 0),
                    std::declval<type&>())
    {
        return ::tim::math::multiply(_l, std::forward<V>(_r));
    }

    template <typename V, typename U = void>
    static TIMEMORY_INLINE auto divide(type& _l, V&& _r, int)
        -> decltype(::tim::math::divide(_l, std::forward<V>(_r),
                                        get_index_sequence<type>::value, 0),
                    std::declval<type&>())
    {
        return ::tim::math::divide(_l, std::forward<V>(_r));
    }

    //----------------------------------------------------------------------------------//
    // no tim::math overload available
    //
    template <typename V>
    static TIMEMORY_INLINE auto abs(V&& _v, long)
    {
        return _v;
    }

    template <typename V>
    static TIMEMORY_INLINE auto sqr(V&& _v, long)
    {
        return _v;
    }

    template <typename V>
    static TIMEMORY_INLINE auto sqrt(V&& _v, long)
    {
        return _v;
    }

    template <typename V>
    static TIMEMORY_INLINE auto min(const type& _l, V&&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto max(const type& _l, V&&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto percent_diff(const type& _l, V&&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto& plus(type& _l, V&&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto& minus(type& _l, V&&, long)
    {
        return _l;
    }

    template <typename V>
    static TIMEMORY_INLINE auto& multiply(type& _l, V&&, long)
    {
        return _l;
    }

    template <typename V, typename U = void>
    static TIMEMORY_INLINE auto& divide(type& _l, V&&, long)
    {
        return _l;
    }
};

//--------------------------------------------------------------------------------------//

}  // namespace math
}  // namespace tim

#define TIMEMORY_MATH_NULL_TYPE_COMPUTE(TYPE)                                            \
    namespace tim                                                                        \
    {                                                                                    \
    namespace math                                                                       \
    {                                                                                    \
    template <>                                                                          \
    struct compute<TYPE, TYPE>                                                           \
    {                                                                                    \
        using type = TYPE;                                                               \
        static type abs(const type&) { return type{}; }                                  \
        static type sqr(const type&) { return type{}; }                                  \
        static type sqrt(const type&) { return type{}; }                                 \
        static type max(const type&, const type&) { return type{}; }                     \
        static type min(const type&, const type&) { return type{}; }                     \
        static type percent_diff(const type&, const type&) { return type{}; }            \
                                                                                         \
        template <typename Vp>                                                           \
        static decltype(auto) plus(type& lhs, const Vp&)                                 \
        {                                                                                \
            return lhs;                                                                  \
        }                                                                                \
        template <typename Vp>                                                           \
        static decltype(auto) minus(type& lhs, const Vp&)                                \
        {                                                                                \
            return lhs;                                                                  \
        }                                                                                \
        template <typename Vp>                                                           \
        static decltype(auto) multiply(type& lhs, const Vp&)                             \
        {                                                                                \
            return lhs;                                                                  \
        }                                                                                \
        template <typename Vp>                                                           \
        static decltype(auto) divide(type& lhs, const Vp&)                               \
        {                                                                                \
            return lhs;                                                                  \
        }                                                                                \
    };                                                                                   \
    }                                                                                    \
    }

TIMEMORY_MATH_NULL_TYPE_COMPUTE(std::tuple<>)
TIMEMORY_MATH_NULL_TYPE_COMPUTE(null_type)
TIMEMORY_MATH_NULL_TYPE_COMPUTE(type_list<>)
