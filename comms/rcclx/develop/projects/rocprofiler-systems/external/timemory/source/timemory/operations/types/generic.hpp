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

/**
 * \file timemory/operations/types/generic.hpp
 * \brief Definition for various functions for generic in operations
 */

#pragma once

#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/demangle.hpp"

#include <type_traits>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::generic_operator
/// \brief This operation class is similar to pointer_operator but can handle non-pointer
/// types
///
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp, typename Op, typename Tag>
struct generic_operator
{
    static_assert(!is_optional<Tp>::value, "Error! type is optional type");
    static_assert(std::is_same<concepts::unqualified_type_t<Tp>, Tp>::value,
                  "Error! type has qualifiers");

    using type = remove_optional_t<Tp>;

    TIMEMORY_DELETED_OBJECT(generic_operator)

private:
    template <typename Up>
    static bool check()
    {
        using U = remove_optional_t<concepts::unqualified_type_t<decay_t<Up>>>;
        static_assert(std::is_same<U, type>::value, "Error! Up != type");

        if constexpr(trait::runtime_enabled<Tp>::value &&
                     trait::runtime_enabled<Op>::value &&
                     trait::runtime_enabled<Tag>::value)
        {
            return (trait::runtime_enabled<Tp>::get() &&
                    trait::runtime_enabled<Op>::get() &&
                    trait::runtime_enabled<Tag>::get());
        }
        else if constexpr(trait::runtime_enabled<Tp>::value &&
                          trait::runtime_enabled<Op>::value)
        {
            return (trait::runtime_enabled<Tp>::get() &&
                    trait::runtime_enabled<Op>::get());
        }
        else if constexpr(trait::runtime_enabled<Op>::value &&
                          trait::runtime_enabled<Tag>::value)
        {
            return (trait::runtime_enabled<Op>::get() &&
                    trait::runtime_enabled<Tag>::get());
        }
        else if constexpr(trait::runtime_enabled<Tp>::value &&
                          trait::runtime_enabled<Tag>::value)
        {
            return (trait::runtime_enabled<Tp>::get() &&
                    trait::runtime_enabled<Tag>::get());
        }
        return true;
    }

    template <typename Up>
    static bool is_invalid(Up& obj)
    {
        // use a type-list for checking multiple types
        using passthrough_t =
            type_list<operation::set_is_invalid<Tp>, operation::get_is_invalid<Tp, false>,
                      operation::get_is_invalid<Tp, true>>;
        if(is_one_of<Op, passthrough_t>::value)
            return false;
        return operation::get_is_invalid<Tp, false>{}(obj);
    }

    //----------------------------------------------------------------------------------//
    //
    //      Pointers
    //
    //----------------------------------------------------------------------------------//
public:
    template <typename Up, typename... Args, typename Rp = type,
              enable_if_t<trait::is_available<Rp>::value, int> = 0,
              enable_if_t<std::is_pointer<Up>::value, int>     = 0>
    TIMEMORY_INLINE explicit generic_operator(Up obj, Args&&... args)
    {
        // rely on compiler to optimize this away if supports_runtime_checks if false
        if(!check<Up>())
            return;

        // check the component is valid before applying
        if(obj && !is_invalid(*obj))
            sfinae(obj, 0, 0, 0, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args, typename Rp = type,
              enable_if_t<trait::is_available<Rp>::value, int> = 0,
              enable_if_t<std::is_pointer<Up>::value, int>     = 0>
    TIMEMORY_INLINE explicit generic_operator(Up obj, Up rhs, Args&&... args)
    {
        // rely on compiler to optimize this away if supports_runtime_checks if false
        if(!check<Up>())
            return;

        // check the components are valid before applying
        if(obj && rhs && !is_invalid(*obj) && !is_invalid(*rhs))
            sfinae(obj, rhs, 0, 0, 0, std::forward<Args>(args)...);
    }

    //----------------------------------------------------------------------------------//
private:
    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto pointer_sfinae(Up obj, int, int, int, Args&&... args)
        -> decltype(Op{ *obj, std::forward<Args>(args)... })
    {
        return Op{ *obj, std::forward<Args>(args)... };
    }

    template <typename Up, typename... Args, typename OpT = Op,
              enable_if_t<std::is_default_constructible<OpT>::value> = 0>
    TIMEMORY_INLINE auto pointer_sfinae(Up obj, int, int, long, Args&&... args)
        -> decltype(std::declval<Op>()(*obj, std::forward<Args>(args)...))
    {
        return Op{}(*obj, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto pointer_sfinae(Up obj, int, long, long, Args&&...)
        -> decltype(Op{ *obj })
    {
        return Op{ *obj };
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE void pointer_sfinae(Up, long, long, long, Args&&...)
    {}

    //----------------------------------------------------------------------------------//

    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto pointer_sfinae(Up obj, Up rhs, int, int, int, Args&&... args)
        -> decltype(Op{ *obj, *rhs, std::forward<Args>(args)... })
    {
        return Op{ *obj, *rhs, std::forward<Args>(args)... };
    }

    template <typename Up, typename... Args, typename OpT = Op,
              enable_if_t<std::is_default_constructible<OpT>::value> = 0>
    TIMEMORY_INLINE auto pointer_sfinae(Up obj, Up rhs, int, int, long, Args&&... args)
        -> decltype(std::declval<Op>()(*obj, *rhs, std::forward<Args>(args)...))
    {
        return Op{}(*obj, *rhs, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto pointer_sfinae(Up obj, Up rhs, int, long, long, Args&&...)
        -> decltype(Op{ *obj, *rhs })
    {
        return Op{ *obj, *rhs };
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE void pointer_sfinae(Up, Up, long, long, long, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    //
    //      References
    //
    //----------------------------------------------------------------------------------//
public:
    template <typename Up, typename... Args, typename Rp = Tp,
              enable_if_t<trait::is_available<Rp>::value, int> = 0,
              enable_if_t<!std::is_pointer<Up>::value, int>    = 0>
    TIMEMORY_INLINE explicit generic_operator(Up& obj, Args&&... args)
    {
        // rely on compiler to optimize this away if supports_runtime_checks if false
        if(!check<Up>())
            return;

        // check the component is valid before applying
        if constexpr(is_optional<concepts::unqualified_type_t<Up>>::value)
        {
            if(obj && !is_invalid(*obj))
                sfinae(*obj, 0, 0, 0, std::forward<Args>(args)...);
        }
        else
        {
            if(!is_invalid(obj))
                sfinae(obj, 0, 0, 0, std::forward<Args>(args)...);
        }
    }

    template <typename Up, typename... Args, typename Rp = Tp,
              enable_if_t<trait::is_available<Rp>::value, int> = 0,
              enable_if_t<!std::is_pointer<Up>::value, int>    = 0>
    TIMEMORY_INLINE explicit generic_operator(Up& obj, Up& rhs, Args&&... args)
    {
        // rely on compiler to optimize this away if supports_runtime_checks if false
        if(!check<Up>())
            return;

        if constexpr(is_optional<concepts::unqualified_type_t<Up>>::value)
        {
            if(obj && rhs && !is_invalid(*obj) && !is_invalid(rhs))
                sfinae(*obj, *rhs, 0, 0, 0, std::forward<Args>(args)...);
        }
        else
        {
            // check the components are valid before applying
            if(!is_invalid(obj) && !is_invalid(rhs))
                sfinae(obj, rhs, 0, 0, 0, std::forward<Args>(args)...);
        }
    }

    //----------------------------------------------------------------------------------//
private:
    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto sfinae(Up& obj, int, int, int, Args&&... args)
        -> decltype(Op{ obj, std::forward<Args>(args)... })
    {
        return Op{ obj, std::forward<Args>(args)... };
    }

    template <typename Up, typename... Args, typename OpT = Op,
              enable_if_t<std::is_default_constructible<OpT>::value> = 0>
    TIMEMORY_INLINE auto sfinae(Up& obj, int, int, long, Args&&... args)
        -> decltype(std::declval<Op>()(obj, std::forward<Args>(args)...))
    {
        return Op{}(obj, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto sfinae(Up& obj, int, long, long, Args&&...)
        -> decltype(Op{ obj })
    {
        return Op{ obj };
    }

    template <typename Up, typename... Args,
              enable_if_t<std::is_pointer<Up>::value, int> = 0>
    TIMEMORY_INLINE decltype(auto) sfinae(Up& obj, long, long, long, Args&&... args)
    {
        // some operations want a raw pointer, e.g. generic_deleter
        return pointer_sfinae(obj, 0, 0, 0, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args,
              enable_if_t<!std::is_pointer<Up>::value, int> = 0>
    TIMEMORY_INLINE void sfinae(Up&, long, long, long, Args&&...)
    {}

    //----------------------------------------------------------------------------------//

    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto sfinae(Up& obj, Up& rhs, int, int, int, Args&&... args)
        -> decltype(Op{ obj, rhs, std::forward<Args>(args)... })
    {
        return Op{ obj, rhs, std::forward<Args>(args)... };
    }

    template <typename Up, typename... Args, typename OpT = Op,
              enable_if_t<std::is_default_constructible<OpT>::value> = 0>
    TIMEMORY_INLINE auto sfinae(Up& obj, Up& rhs, int, int, long, Args&&... args)
        -> decltype(std::declval<Op>()(obj, rhs, std::forward<Args>(args)...))
    {
        return Op{}(obj, rhs, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE auto sfinae(Up& obj, Up& rhs, int, long, long, Args&&...)
        -> decltype(Op{ obj, rhs })
    {
        return Op{ obj, rhs };
    }

    template <typename Up, typename... Args,
              enable_if_t<std::is_pointer<Up>::value, int> = 0>
    TIMEMORY_INLINE decltype(auto) sfinae(Up& obj, Up& rhs, long, long, long,
                                          Args&&... args)
    {
        // some operations want a raw pointer, e.g. generic_deleter
        return pointer_sfinae(obj, rhs, 0, 0, 0, std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args,
              enable_if_t<!std::is_pointer<Up>::value, int> = 0>
    TIMEMORY_INLINE void sfinae(Up&, Up&, long, long, long, Args&&...)
    {}

    //----------------------------------------------------------------------------------//
    //
    //      Not available
    //
    //----------------------------------------------------------------------------------//
public:
    // if the type is not available, never do anything
    template <typename Up, typename... Args, typename Rp = Tp,
              enable_if_t<!trait::is_available<Rp>::value, int> = 0>
    TIMEMORY_INLINE generic_operator(Up&, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct generic_deleter
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(generic_deleter)

    template <typename Up = Tp>
    TIMEMORY_INLINE explicit generic_deleter(Up&& _obj)
    {
        using unqual_type = std::remove_pointer_t<concepts::unqualified_type_t<Up>>;
        static_assert(concepts::is_unqualified_same<unqual_type, type>::value,
                      "Error! should be same type");
        (*this)(std::forward<Up>(_obj));
    }

    template <typename Up>
    auto operator()(Up&& _obj)
    {
        sfinae(std::forward<Up>(_obj), 0);
        return _obj;
    }

private:
    template <typename Up>
    static auto sfinae(Up& _obj, int,
                       std::enable_if_t<!std::is_pointer<Up>::value &&
                                            concepts::is_dynamic_alloc<Up>::value,
                                        int> = 0) -> decltype(_obj.reset(), void())
    {
        _obj.reset();
    }

    template <typename Up>
    static auto sfinae(Up& _obj, int,
                       std::enable_if_t<std::is_pointer<Up>::value, int> = 0)
    {
        delete _obj;
        _obj = nullptr;
    }

    template <typename Up>
    static void sfinae(Up&&, long)
    {
        static_assert(!concepts::is_dynamic_alloc<Up>::value,
                      "Error! Up is a dynamic allocation type");
    }
};
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct generic_counter
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(generic_counter)

    template <typename Up>
    generic_counter(const Up& obj, uint64_t& count)
    {
        count += (*this)(obj);
    }

    template <typename Up>
    TIMEMORY_INLINE uint64_t operator()(const Up& obj) const
    {
        if constexpr(is_optional<Up>::value || std::is_pointer<Up>::value)
        {
            return (trait::runtime_enabled<type>::get() && obj) ? 1 : 0;
        }
        else
        {
            return (trait::runtime_enabled<type>::get()) ? 1 : 0;
        }
    }

    template <typename Up>
    uint64_t& operator()(const Up& obj, uint64_t& count) const
    {
        return (count += (*this)(obj));
    }
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
