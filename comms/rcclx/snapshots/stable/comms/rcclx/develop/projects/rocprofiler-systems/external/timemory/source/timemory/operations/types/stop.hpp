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
#include "timemory/mpl/quirks.hpp"
#include "timemory/operations/declaration.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/macros.hpp"

#include <type_traits>

namespace tim
{
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
//
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct stop
{
    using type = Tp;

    template <typename U>
    using base_t = typename U::base_type;

    TIMEMORY_DEFAULT_OBJECT(stop)

    TIMEMORY_HOT explicit stop(type& obj) { impl(obj); }
    TIMEMORY_HOT explicit stop(type& obj, quirk::unsafe&&) { impl(obj, quirk::unsafe{}); }

    template <typename Arg, typename... Args>
    TIMEMORY_HOT stop(type& obj, Arg&& arg, Args&&... args)
    {
        impl(obj, std::forward<Arg>(arg), std::forward<Args>(args)...);
    }

    template <typename... Args>
    TIMEMORY_HOT auto operator()(type& obj, Args&&... args) const
    {
        using RetT = decltype(sfinae(obj, 0, 0, std::forward<Args>(args)...));
        if constexpr(std::is_default_constructible<RetT>::value)
        {
            if(is_running<Tp, true>{}(obj))
            {
                return sfinae(obj, 0, 0, std::forward<Args>(args)...);
            }
            return get_return<RetT>();
        }
        else
        {
            if(is_running<Tp, true>{}(obj))
            {
                sfinae(obj, 0, 0, std::forward<Args>(args)...);
            }
        }
    }

    template <typename... Args>
    TIMEMORY_HOT auto operator()(type& obj, quirk::unsafe&&, Args&&... args) const
    {
        return sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    template <typename T>
    auto get_return(enable_if_t<std::is_void<T>::value, int> = 0) const
    {}

    template <typename T>
    auto get_return(enable_if_t<!std::is_void<T>::value, long> = 0) const
    {
        static_assert(std::is_default_constructible<T>::value,
                      "Error! start() returns a type that is not default constructible! "
                      "You must specialize operation::start<T> struct");
        return T{};
    }

    template <typename... Args>
    TIMEMORY_HOT void impl(type& obj, Args&&... args) const;

    // resolution #1 (best)
    template <typename Up, typename... Args>
    TIMEMORY_HOT auto sfinae(Up& obj, int, int, Args&&... args) const
        -> decltype(obj.stop(std::forward<Args>(args)...))
    {
        set_stopped<Tp>{}(obj);
        return obj.stop(std::forward<Args>(args)...);
    }

    // resolution #2
    template <typename Up, typename... Args>
    TIMEMORY_HOT auto sfinae(Up& obj, int, long, Args&&...) const -> decltype(obj.stop())
    {
        set_stopped<Tp>{}(obj);
        return obj.stop();
    }

    // resolution #3 (worst) - no member function
    template <typename Up, typename... Args>
    void sfinae(Up&, long, long, Args&&...) const
    {
        SFINAE_WARNING(type);
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
struct priority_stop
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(priority_stop)

    template <typename... Args>
    TIMEMORY_HOT explicit priority_stop(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    static TIMEMORY_HOT auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        if constexpr(std::is_default_constructible<stop<Tp>>::value)
        {
            return stop<Tp>{}(obj, std::forward<Args>(args)...);
        }
        else
        {
            return stop<Tp>{ obj, std::forward<Args>(args)... };
        }
    }

    //  does not satisfy mpl condition
    template <typename Up, typename... Args>
    static TIMEMORY_INLINE void sfinae(Up&, false_type&&, Args&&...)
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
struct standard_stop
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(standard_stop)

    template <typename... Args>
    TIMEMORY_HOT explicit standard_stop(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    static TIMEMORY_HOT auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        if constexpr(std::is_default_constructible<stop<Tp>>::value)
        {
            return stop<Tp>{}(obj, std::forward<Args>(args)...);
        }
        else
        {
            return stop<Tp>{ obj, std::forward<Args>(args)... };
        }
    }

    //  does not satisfy mpl condition
    template <typename Up, typename... Args>
    static TIMEMORY_INLINE void sfinae(Up&, false_type&&, Args&&...)
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
struct delayed_stop
{
    using type = Tp;

    TIMEMORY_DEFAULT_OBJECT(delayed_stop)

    template <typename... Args>
    TIMEMORY_HOT explicit delayed_stop(type& obj, Args&&... args);

private:
    //  satisfies mpl condition
    template <typename Up, typename... Args>
    static TIMEMORY_HOT auto sfinae(Up& obj, true_type&&, Args&&... args)
    {
        if constexpr(std::is_default_constructible<stop<Tp>>::value)
        {
            return stop<Tp>{}(obj, std::forward<Args>(args)...);
        }
        else
        {
            return stop<Tp>{ obj, std::forward<Args>(args)... };
        }
    }

    //  does not satisfy mpl condition
    template <typename Up, typename... Args>
    static TIMEMORY_INLINE void sfinae(Up&, false_type&&, Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
void
stop<Tp>::impl(type& obj, Args&&... args) const
{
    if(is_running<Tp, true>{}(obj))
    {
        set_stopped<Tp>{}(obj);
        sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
priority_stop<Tp>::priority_stop(type& obj, Args&&... args)
{
    using sfinae_type =
        conditional_t<(trait::stop_priority<Tp>::value < 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
standard_stop<Tp>::standard_stop(type& obj, Args&&... args)
{
    using sfinae_type =
        conditional_t<(trait::stop_priority<Tp>::value == 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
template <typename... Args>
delayed_stop<Tp>::delayed_stop(type& obj, Args&&... args)
{
    using sfinae_type =
        conditional_t<(trait::stop_priority<Tp>::value > 0), true_type, false_type>;
    sfinae(obj, sfinae_type{}, std::forward<Args>(args)...);
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
