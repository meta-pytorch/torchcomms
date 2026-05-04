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
 * \file timemory/operations/types.hpp
 * \brief Declare the operations types
 */

#pragma once

#include "timemory/backends/dmp.hpp"
#include "timemory/macros/attributes.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/operations/macros.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/storage/types.hpp"
#include "timemory/variadic/types.hpp"

#include <functional>
#include <iosfwd>
#include <type_traits>
#include <utility>

namespace std
{
//
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
TSTAG(struct)
tuple_size<::tim::type_list<Types...>>
{
public:
    static constexpr size_t value = sizeof...(Types);
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace std

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct remove_pointers;
//
template <template <typename...> class Tuple, typename... Tp>
struct remove_pointers<Tuple<Tp...>>
{
    using type = Tuple<std::remove_pointer_t<Tp>...>;
};
//
template <typename Tp>
using remove_pointers_t = typename remove_pointers<Tp>::type;
//
template <typename Tp>
struct basic_tree;
//
namespace node
{
template <typename Tp>
struct tree;
}
//
//--------------------------------------------------------------------------------------//
//
namespace data
{
struct stream;
}
//
namespace utility
{
using stream = data::stream;
}
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct statistics;
//
//--------------------------------------------------------------------------------------//
//
//                              operations
//
//--------------------------------------------------------------------------------------//
//
//  components that provide the invocation (i.e. WHAT the components need to do)
//
namespace operation
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Up, typename Ap = trait::is_available_t<Up>>
struct has_data;
//
template <typename Up>
struct has_data<Up, true_type>
{
    // shorthand for non-void
    using Vp                    = typename Up::value_type;
    static constexpr bool value = !std::is_void<Vp>::value;
};
//
template <typename Up>
struct has_data<Up, false_type>
{
    // shorthand for non-void
    using Vp                    = void;
    static constexpr bool value = !std::is_void<Vp>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Up>
struct is_enabled
{
    // shorthand for available + non-void
    using Vp                    = typename Up::value_type;
    static constexpr bool value = trait::is_available<Up>::value && has_data<Up>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Up>
struct enabled_value_storage
{
    // shorthand for available + uses_value_storage
    static constexpr bool value =
        trait::is_available<Up>::value && trait::uses_value_storage<Up>::value;
};
//
//--------------------------------------------------------------------------------------//
//
template <typename U>
using is_enabled_t = typename is_enabled<U>::type;
//
//--------------------------------------------------------------------------------------//
//
namespace internal
{
template <typename U>
auto
resolve_record_type(int) -> decltype(
    U::record(),
    typename mpl::function_traits<decltype(std::declval<U>().record())>::result_type())
{
    return U::record();
}
//
//--------------------------------------------------------------------------------------//
//
template <typename U>
void
resolve_record_type(long)
{}
}  // namespace internal
//
//--------------------------------------------------------------------------------------//
//
template <typename U>
decltype(internal::resolve_record_type<U>(0))
resolve_record_type()
{
    return internal::resolve_record_type<U>(0);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type>
struct check_record_type
{
    using type =
        typename mpl::function_traits<decltype(&resolve_record_type<T>)>::result_type;
    static constexpr bool value =
        (!std::is_void<V>::value && !std::is_void<type>::value && is_enabled<T>::value &&
         std::is_same<V, type>::value);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Up, typename Vp>
struct stats_enabled
{
    static constexpr bool value =
        (trait::record_statistics<Up>::value && !concepts::is_null_type<Vp>::value);
};
//
template <typename Up, typename Vp>
struct stats_enabled<Up, statistics<Vp>>
{
    static constexpr bool value =
        (trait::record_statistics<Up>::value && !concepts::is_null_type<Vp>::value);
};
//
//--------------------------------------------------------------------------------------//
//
struct init_mode
{
    enum value
    {
        thread,
        global
    };
};
//
using fini_mode = init_mode;
//
template <int ModeV>
using mode_constant = std::integral_constant<int, ModeV>;
//
//--------------------------------------------------------------------------------------//
//
template <typename U>
using has_data_t = typename has_data<U>::type;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct async;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct init;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct init_storage;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct fini_storage;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct construct;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct set_prefix;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct TIMEMORY_VISIBLE set_scope;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct set_data;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct set_state;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct push_node;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct pop_node;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct record;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct reset;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct measure;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct sample;
//
//--------------------------------------------------------------------------------------//
//
template <typename Ret, typename Lhs, typename Rhs>
struct compose;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct insert
{
    TIMEMORY_DEFAULT_OBJECT(insert)

    template <typename Up, typename... Args>
    TIMEMORY_HOT auto operator()(Up& obj, Args&&... args) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0, 0, std::forward<Args>(args)...);
    }

private:
    template <typename Up, typename... Args>
    static TIMEMORY_HOT auto sfinae(Up& obj, int, int, Args&&... args)
        -> decltype(obj.insert(std::forward<Args>(args)...))
    {
        return obj.insert(std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    static TIMEMORY_HOT auto sfinae(Up& obj, int, long, Args&&... args)
        -> decltype(obj->insert(std::forward<Args>(args)...))
    {
        return obj->insert(std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    static TIMEMORY_INLINE auto sfinae(Up&, long, long, Args&&...) -> std::nullptr_t
    {
        return nullptr;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct stack_push
{
    TIMEMORY_DEFAULT_OBJECT(stack_push)

    template <typename Up, typename Vp>
    TIMEMORY_HOT void operator()(Up& obj, Vp&& arg) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        sfinae(obj, 0, 0, std::forward<Vp>(arg));
    }

private:
    template <typename Up, typename Vp>
    TIMEMORY_HOT auto sfinae(Up& obj, int, int, Vp&& arg) const
        -> decltype(obj.stack_push(std::forward<Vp>(arg)), void())
    {
        obj.stack_push(std::forward<Vp>(arg));
    }

    template <typename Up, typename Vp>
    TIMEMORY_HOT auto sfinae(Up& obj, int, long, Vp&& arg) const
        -> decltype(obj->stack_push(std::forward<Vp>(arg)), void())
    {
        if(obj)
            obj->stack_push(std::forward<Vp>(arg));
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE void sfinae(Up&, long, long, Args&&...) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct pop
{
    TIMEMORY_DEFAULT_OBJECT(pop)

    template <typename Up, typename... Args>
    TIMEMORY_HOT decltype(auto) operator()(Up& obj, Args&&... args) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0, std::forward<Args>(args)...);
    }

private:
    template <typename Up, typename... Args>
    TIMEMORY_HOT auto sfinae(Up& obj, int, Args&&... args) const
        -> decltype(obj.pop(std::forward<Args>(args)...))
    {
        return obj.pop(std::forward<Args>(args)...);
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE void sfinae(Up&, long, Args&&...) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct stack_pop
{
    TIMEMORY_DEFAULT_OBJECT(stack_pop)

    template <typename Up, typename Vp>
    TIMEMORY_HOT void operator()(Up& obj, Vp&& arg) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        sfinae(obj, 0, 0, std::forward<Vp>(arg));
    }

private:
    template <typename Up, typename Vp>
    TIMEMORY_HOT auto sfinae(Up& obj, int, int, Vp&& arg) const
        -> decltype(obj.stack_pop(std::forward<Vp>(arg)), void())
    {
        obj.stack_pop(std::forward<Vp>(arg));
    }

    template <typename Up, typename Vp>
    TIMEMORY_HOT auto sfinae(Up& obj, int, long, Vp&& arg) const
        -> decltype(obj->stack_pop(std::forward<Vp>(arg)), void())
    {
        if(obj)
            obj->stack_pop(std::forward<Vp>(arg));
    }

    template <typename Up, typename... Args>
    TIMEMORY_INLINE void sfinae(Up&, long, long, Args&&...) const
    {}
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_started
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which the component provides
/// to internally store whether or not it is currently within a phase measurement (to
/// prevent restarts)
template <typename T>
struct set_started
{
    TIMEMORY_DEFAULT_OBJECT(set_started)

    template <typename Up>
    TIMEMORY_HOT auto operator()(Up& obj) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0);
    }

private:
    template <typename Up>
    static TIMEMORY_HOT auto sfinae(Up& obj, int) -> decltype(obj.set_started())
    {
        return obj.set_started();
    }

    template <typename Up>
    static TIMEMORY_INLINE auto sfinae(Up&, long) -> void
    {}
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_stopped
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which the component provides
/// to internally store whether or not it is currently within a phase measurement (to
/// prevent stopping when it hasn't been started)
template <typename T>
struct set_stopped
{
    TIMEMORY_DEFAULT_OBJECT(set_stopped)

    template <typename Up>
    TIMEMORY_HOT auto operator()(Up& obj) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0);
    }

private:
    template <typename Up>
    static TIMEMORY_HOT auto sfinae(Up& obj, int) -> decltype(obj.set_stopped())
    {
        return obj.set_stopped();
    }

    template <typename Up>
    static TIMEMORY_INLINE auto sfinae(Up&, long) -> void
    {}
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_depth_change
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which the component provides
/// to internally store whether or not the component triggered a depth change when it
/// was push to the call-stack or when it was popped from the call-stack
template <typename T>
struct set_depth_change;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_is_flat
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which the component provides
/// to internally store whether or not the component triggered a depth change when it
/// was push to the call-stack or when it was popped from the call-stack
template <typename T>
struct set_is_flat;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_is_on_stack
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which the component provides
/// to internally store whether or not the component is referenced in persistent storage
template <typename T>
struct set_is_on_stack;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_is_invalid
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which sets whether or not
/// the component is in a valid state for data access and updates.
template <typename T>
struct set_is_invalid;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_is_running
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which sets whether or not
/// the component is currently collecting.
template <typename T>
struct set_is_running;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::set_iterator
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which sets a pointer to
/// a reference value in persistant storage.
template <typename T>
struct set_iterator;
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::is_running
/// \tparam T Component type
/// \tparam DefaultValue The value to return if the member function is not provided
///
/// \brief This operation attempts to call a member function which provides whether or not
/// the component currently within a phase measurement
template <typename T, bool DefaultValue>
struct is_running
{
    TIMEMORY_DEFAULT_OBJECT(is_running)

    template <typename Up>
    TIMEMORY_HOT auto operator()(const Up& obj) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0);
    }

private:
    template <typename Up>
    static auto sfinae(const Up& obj, int) -> decltype(obj.get_is_running())
    {
        return obj.get_is_running();
    }

#if !defined(TIMEMORY_NVCC_COMPILER)
    // NVCC compiler has errors when object has private is_running function.
    // Particularly, the papi_vector class which has private inheritance
    // from papi_common<Tp>, which has a is_running() function.
    template <typename Up>
    static auto sfinae(const Up& obj, long) -> decltype(obj.is_running())
    {
        return obj.is_running();
    }
#endif

    template <typename Up>
    static auto sfinae(const Up&, ...) -> bool
    {
        return DefaultValue;
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::get_is_flat
/// \tparam T Component type
/// \tparam DefaultValue The value to return if the member function is not provided
///
/// \brief This operation attempts to call a member function which provides whether or not
/// the component pushed to storage at a depth of zero.
template <typename T, bool DefaultValue>
struct get_is_flat
{
    TIMEMORY_DEFAULT_OBJECT(get_is_flat)

    template <typename Up>
    TIMEMORY_INLINE auto operator()(const Up& obj) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0);
    }

private:
    template <typename Up>
    static auto sfinae(const Up& obj, int) -> decltype(obj.get_is_flat())
    {
        return obj.get_is_flat();
    }

    template <typename Up>
    static auto sfinae(const Up&, ...) -> bool
    {
        return DefaultValue;
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::get_is_invalid
/// \tparam T Component type
/// \tparam DefaultValue The value to return if the member function is not provided
///
/// \brief This operation attempts to call a member function which provides whether or not
/// the component is in a valid state for data access and updates.
template <typename T = void, bool DefaultValue = false>
struct get_is_invalid
{
    TIMEMORY_DEFAULT_OBJECT(get_is_invalid)

    template <typename Up>
    TIMEMORY_INLINE auto operator()(const Up& obj) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0);
    }

private:
    template <typename Up>
    static auto sfinae(const Up& obj, int) -> decltype(obj.get_is_invalid())
    {
        return obj.get_is_invalid();
    }

    template <typename Up>
    static auto sfinae(const Up&, ...) -> bool
    {
        return DefaultValue;
    }
};
//
template <>
struct get_is_invalid<void, false>
{
    TIMEMORY_DEFAULT_OBJECT(get_is_invalid)

    template <typename Up>
    TIMEMORY_INLINE auto operator()(Up&& obj) const
    {
        using Tp = decay_t<Up>;
        static_assert(!std::is_void<Tp>::value, "Infinite recursion");
        return get_is_invalid<Tp, false>{}(std::forward<Up>(obj));
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::get_is_on_stack
/// \tparam T Component type
/// \tparam DefaultValue The value to return if the member function is not provided
///
/// \brief This operation attempts to call a member function which provides whether or not
/// the component is in a valid state for data access and updates.
template <typename T, bool DefaultValue>
struct get_is_on_stack
{
    TIMEMORY_DEFAULT_OBJECT(get_is_on_stack)

    template <typename Up>
    TIMEMORY_INLINE auto operator()(const Up& obj) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0);
    }

private:
    template <typename Up>
    static auto sfinae(const Up& obj, int) -> decltype(obj.get_is_on_stack())
    {
        return obj.get_is_on_stack();
    }

    template <typename Up>
    static auto sfinae(const Up&, ...) -> bool
    {
        return DefaultValue;
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::get_depth
/// \tparam T Component type
/// \tparam DefaultDepth value to return if get_depth() function not available
///
/// \brief This operation attempts to call a member function which provides a depth value.
template <typename T, int64_t DefaultDepth = 0>
struct get_depth
{
    TIMEMORY_DEFAULT_OBJECT(get_depth)

    template <typename Up, enable_if_t<!std::is_pointer<decay_t<Up>>::value, int> = 0>
    TIMEMORY_INLINE auto operator()(Up& obj) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0, 0);
    }

    template <typename Up, enable_if_t<std::is_pointer<decay_t<Up>>::value, int> = 0>
    TIMEMORY_INLINE auto operator()(Up obj) const
    {
        if(!obj)
            return DefaultDepth;

        return sfinae(*obj, 0, 0);
    }

private:
    template <typename Up>
    auto sfinae(Up& obj, int, int) const -> decltype(obj.get_depth())
    {
        return obj.get_depth();
    }

    template <typename Up>
    auto sfinae(Up& obj, int, long) const -> decltype(obj.depth())
    {
        return obj.depth();
    }

    template <typename Up>
    auto sfinae(Up&&, long, long) const
    {
        return DefaultDepth;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get_storage;
//
/// \struct tim::operation::set_storage
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which provides a pointer to
/// the data storage structure for a component which should be updated for
/// aggregation/logging.
template <typename T>
struct set_storage
{
    friend struct get_storage<T>;
    static constexpr size_t max_threads = TIMEMORY_MAX_THREADS;
    using type                          = T;
    using storage_array_t               = std::array<storage<type>*, max_threads>;

    TIMEMORY_DEFAULT_OBJECT(set_storage)

    TIMEMORY_INLINE auto operator()(storage<type>* _storage, size_t _idx) const
    {
        get().at(_idx) = static_cast<storage<type>*>(_storage);
    }

    TIMEMORY_INLINE auto operator()(type& _obj, size_t _idx) const
    {
        get().at(_idx) = static_cast<storage<type>*>(_obj.get_storage());
    }

private:
    static storage_array_t& get()
    {
        static storage_array_t _v = { nullptr };
        return _v;
    }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::get_storage
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which provides a pointer to
/// the data storage structure for a component which should be updated for
/// aggregation/logging.
template <typename T>
struct get_storage
{
    using type = T;

    TIMEMORY_DEFAULT_OBJECT(get_storage)

    TIMEMORY_INLINE auto operator()() const
    {
        type _obj{};
        return (*this)(_obj);
    }

    TIMEMORY_INLINE auto operator()(const type& obj) const
    {
        return static_cast<storage<type>*>(obj.get_storage());
    }

    TIMEMORY_INLINE auto operator()(size_t _idx) const
    {
        return operation::set_storage<T>::get().at(_idx);
    }

    TIMEMORY_INLINE auto operator()(type&, size_t _idx) const { return (*this)(_idx); }
};
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::operation::get_iterator
/// \tparam T Component type
///
/// \brief This operation attempts to call a member function which provides a pointer to
/// a component which exists in storage.
template <typename T>
struct get_iterator
{
    TIMEMORY_DEFAULT_OBJECT(get_iterator)

    template <typename Up>
    TIMEMORY_INLINE auto operator()(Up& obj) const
    {
        static_assert(!std::is_pointer<Up>::value,
                      "SFINAE tests will always fail with pointer types");
        return sfinae(obj, 0);
    }

private:
    template <typename Up>
    static auto sfinae(Up& obj, int) -> decltype(obj.get_iterator())
    {
        return obj.get_iterator();
    }

    template <typename Up>
    static auto sfinae(Up&&, ...) -> std::nullptr_t
    {
        return nullptr;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct priority_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct standard_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct delayed_start;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct priority_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct standard_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct delayed_stop;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct mark;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct mark_begin;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct mark_end;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct store;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct audit;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct plus;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct minus;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct multiply;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct divide;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get_data;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct get_labeled_data;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct base_printer;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_header;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_statistics;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct print_storage;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct add_secondary;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct add_statistics;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct serialization;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct extra_serialization;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, bool Enabled = trait::echo_enabled<T>::value>
struct echo_measurement;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct copy;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct assemble;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct derive;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct generic_deleter;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct generic_counter;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename Op, typename Tag = TIMEMORY_API>
struct generic_operator;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct cache;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct fini;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct cleanup;
//
//--------------------------------------------------------------------------------------//
//
template <typename Tag = TIMEMORY_API>
struct decode;
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct dummy
{
    static_assert(std::is_default_constructible<T>::value,
                  "Type is not default constructible and therefore a dummy object (for "
                  "placeholders and meta-programming) cannot be automatically generated. "
                  "Please specialize tim::operation::dummy<T> to provide operator()() "
                  "which returns a dummy object.");

    TIMEMORY_DEFAULT_OBJECT(dummy)

    TIMEMORY_INLINE T operator()() const
    {
        T _v{};
        operation::set_is_invalid<T>{}(_v, true);
        return _v;
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
struct python_class_name;
//
//--------------------------------------------------------------------------------------//
//
template <typename DataT, typename T>
struct annotate;
//
//--------------------------------------------------------------------------------------//
//
namespace finalize
{
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct mpi_get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct upc_get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct dmp_get;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct print;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type, bool has_data>
struct merge;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct merge<Type, true>
{
    static constexpr bool has_data = true;
    using storage_type             = impl::storage<Type, has_data>;
    using singleton_t              = typename storage_type::singleton_type;
    using graph_t                  = typename storage_type::graph_type;
    using result_type              = typename storage_type::result_array_t;

    template <typename Tp>
    using vector_t = std::vector<Tp>;

    TIMEMORY_DEFAULT_OBJECT(merge)

    TIMEMORY_COLD merge(storage_type& lhs, storage_type& rhs);
    TIMEMORY_COLD merge(result_type& lhs, result_type& rhs);

    // unary
    template <typename Tp>
    TIMEMORY_COLD basic_tree<Tp> operator()(const basic_tree<Tp>& _bt);

    template <typename Tp>
    TIMEMORY_COLD vector_t<basic_tree<Tp>> operator()(
        const vector_t<basic_tree<Tp>>& _bt);

    template <typename Tp>
    TIMEMORY_COLD vector_t<basic_tree<Tp>> operator()(
        const vector_t<vector_t<basic_tree<Tp>>>& _bt, size_t _root = 0);

    // binary
    template <typename Tp>
    TIMEMORY_COLD basic_tree<Tp> operator()(const basic_tree<Tp>&, const basic_tree<Tp>&);

    template <typename Tp>
    TIMEMORY_COLD vector_t<basic_tree<Tp>> operator()(const vector_t<basic_tree<Tp>>&,
                                                      const vector_t<basic_tree<Tp>>&);
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct merge<Type, false>
{
    static constexpr bool has_data = false;
    using storage_type             = impl::storage<Type, has_data>;
    using singleton_t              = typename storage_type::singleton_type;
    using graph_t                  = typename storage_type::graph_type;
    using result_type              = typename storage_type::result_array_t;

    merge(storage_type& lhs, storage_type& rhs);
    merge(result_type&, const result_type&) {}
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct flamegraph;
//
//--------------------------------------------------------------------------------------//
//
template <typename Type>
struct ctest_notes;
//
//--------------------------------------------------------------------------------------//
//
namespace base
{
//
//--------------------------------------------------------------------------------------//
//
struct print
{
    using this_type   = print;
    using stream_type = std::shared_ptr<utility::stream>;
    using settings_t  = std::shared_ptr<settings>;

    explicit TIMEMORY_COLD print(bool       _forced_json = false,
                                 settings_t _settings    = settings::shared_instance())
    : m_settings(std::move(_settings))
    , json_forced(_forced_json)
    {
        if(m_settings)
        {
            debug          = m_settings->get_debug();
            verbose        = m_settings->get_verbose();
            max_call_stack = m_settings->get_max_depth();
        }
    }

    TIMEMORY_COLD print(std::string _label, bool _forced_json,
                        settings_t _settings = settings::shared_instance())
    : m_settings(std::move(_settings))
    , json_forced(_forced_json)
    , label(std::move(_label))
    {
        if(m_settings)
        {
            debug          = m_settings->get_debug();
            verbose        = m_settings->get_verbose();
            max_call_stack = m_settings->get_max_depth();
        }
    }

    TIMEMORY_COLD virtual ~print() = default;

    TIMEMORY_COLD virtual void setup()        = 0;
    TIMEMORY_COLD virtual void execute()      = 0;
    TIMEMORY_COLD virtual void read_json()    = 0;
    TIMEMORY_COLD virtual void print_dart()   = 0;
    TIMEMORY_COLD virtual void update_data()  = 0;
    TIMEMORY_COLD virtual void print_custom() = 0;

    TIMEMORY_COLD virtual void write(std::ostream& os, stream_type stream);
    TIMEMORY_COLD virtual void print_cout(stream_type stream);
    TIMEMORY_COLD virtual void print_text(const std::string& fname, stream_type stream);
    TIMEMORY_COLD virtual void print_plot(const std::string& fname, std::string suffix);

    TIMEMORY_COLD auto get_label() const { return label; }
    TIMEMORY_COLD auto get_text_output_name() const { return text_outfname; }
    TIMEMORY_COLD auto get_json_output_name() const { return json_outfname; }
    TIMEMORY_COLD auto get_json_input_name() const { return json_inpfname; }
    TIMEMORY_COLD auto get_text_diff_name() const { return text_diffname; }
    TIMEMORY_COLD auto get_json_diff_name() const { return json_diffname; }

    TIMEMORY_COLD void set_debug(bool v) { debug = v; }
    TIMEMORY_COLD void set_update(bool v) { update = v; }
    TIMEMORY_COLD void set_verbose(int32_t v) { verbose = v; }
    TIMEMORY_COLD void set_max_call_stack(int64_t v) { max_call_stack = v; }

    TIMEMORY_COLD int64_t get_max_depth() const
    {
        return (max_depth > 0)
                   ? max_depth
                   : std::min<int64_t>(max_call_stack, m_settings->get_max_depth());
    }

    TIMEMORY_COLD bool dart_output()
    {
        if(!m_settings)
        {
            TIMEMORY_PRINT_HERE("%s", "Null pointer to settings! Disabling");
            return false;
        }
        return m_settings->get_dart_output();
    }
    TIMEMORY_COLD bool file_output()
    {
        if(!m_settings)
        {
            TIMEMORY_PRINT_HERE("%s", "Null pointer to settings! Disabling");
            return false;
        }
        return m_settings->get_file_output();
    }
    TIMEMORY_COLD bool cout_output()
    {
        if(!m_settings)
        {
            TIMEMORY_PRINT_HERE("%s", "Null pointer to settings! Disabling");
            return false;
        }
        return m_settings->get_cout_output();
    }
    TIMEMORY_COLD bool tree_output()
    {
        if(!m_settings)
        {
            TIMEMORY_PRINT_HERE("%s", "Null pointer to settings! Disabling");
            return false;
        }
        return (m_settings->get_tree_output() || json_forced) &&
               m_settings->get_file_output();
    }
    TIMEMORY_COLD bool json_output()
    {
        if(!m_settings)
        {
            TIMEMORY_PRINT_HERE("%s", "Null pointer to settings! Disabling");
            return false;
        }
        return (m_settings->get_json_output() || json_forced) &&
               m_settings->get_file_output();
    }
    TIMEMORY_COLD bool text_output()
    {
        if(!m_settings)
        {
            TIMEMORY_PRINT_HERE("%s", "Null pointer to settings! Disabling");
            return false;
        }
        return m_settings->get_text_output() && m_settings->get_file_output();
    }
    TIMEMORY_COLD bool plot_output()
    {
        if(!m_settings)
        {
            TIMEMORY_PRINT_HERE("%s", "Null pointer to settings! Disabling");
            return false;
        }
        return m_settings->get_plot_output() && m_settings->get_json_output() &&
               m_settings->get_file_output();
    }
    TIMEMORY_COLD bool flame_output()
    {
        if(!m_settings)
        {
            TIMEMORY_PRINT_HERE("%s", "Null pointer to settings! Disabling");
            return false;
        }
        return m_settings->get_flamegraph_output() && m_settings->get_file_output();
    }

protected:
    // do not lint misc-non-private-member-variables-in-classes
    settings_t  m_settings     = settings::shared_instance();          // NOLINT
    bool        debug          = false;                                // NOLINT
    bool        update         = true;                                 // NOLINT
    bool        json_forced    = false;                                // NOLINT
    bool        node_init      = dmp::is_initialized();                // NOLINT
    int32_t     node_rank      = dmp::rank();                          // NOLINT
    int32_t     node_size      = dmp::size();                          // NOLINT
    int32_t     verbose        = 0;                                    // NOLINT
    int64_t     max_depth      = 0;                                    // NOLINT
    int64_t     max_call_stack = std::numeric_limits<int64_t>::max();  // NOLINT
    std::string label          = {};                                   // NOLINT
    std::string description    = {};                                   // NOLINT
    std::string text_outfname  = {};                                   // NOLINT
    std::string json_outfname  = {};                                   // NOLINT
    std::string json_inpfname  = {};                                   // NOLINT
    std::string text_diffname  = {};                                   // NOLINT
    std::string json_diffname  = {};                                   // NOLINT
    stream_type data_stream    = stream_type{};                        // NOLINT
    stream_type diff_stream    = stream_type{};                        // NOLINT
    std::function<void(const std::string&)> m_print;
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace base
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print<Tp, true> : public base::print
{
    static constexpr bool has_data = true;
    using base_type                = base::print;
    using this_type                = print<Tp, has_data>;
    using type                     = Tp;
    using storage_type             = impl::storage<Tp, has_data>;
    using result_type              = typename storage_type::dmp_result_t;
    using result_node              = typename storage_type::result_node;
    using graph_type               = typename storage_type::graph_t;
    using graph_node               = typename storage_type::graph_node;
    using hierarchy_type           = std::vector<uint64_t>;
    using callback_type            = std::function<void(this_type*)>;
    using stream_type              = std::shared_ptr<utility::stream>;
    using basic_tree_type          = basic_tree<node::tree<Tp>>;
    using basic_tree_vector_type   = std::vector<basic_tree_type>;
    using result_tree = std::map<std::string, std::vector<basic_tree_vector_type>>;

    static callback_type& get_default_callback()
    {
        static callback_type _instance = [](this_type*) {};
        return _instance;
    }

    explicit TIMEMORY_COLD print(
        storage_type* _data, const settings_t& _settings = settings::shared_instance());

    print(const std::string& _label, storage_type* _data,
          const settings_t& _settings = settings::shared_instance());

    ~print() override = default;

    TIMEMORY_COLD void execute() override
    {
        if(!data)
            return;

        if(update)
        {
            update_data();
        }
        else
        {
            setup();
        }

        if(node_init && node_rank > 0)
            return;

        if(file_output())
        {
            if(json_output() || tree_output())
                print_json(json_outfname, node_results, node_tree);
            if(text_output())
                print_text(text_outfname, data_stream);
            if(plot_output())
                print_plot(json_outfname, "");
        }

        if(cout_output())
        {
            print_cout(data_stream);
        }

        if(dart_output())
            print_dart();

        if(!node_input.empty() && !node_delta.empty() && settings::diff_output())
        {
            if(file_output())
            {
                if(json_output())
                {
                    auto _empty_tree = result_tree{};
                    print_json(json_diffname, node_delta, _empty_tree);
                }
                if(text_output())
                    print_text(text_diffname, diff_stream);
                if(plot_output())
                {
                    std::stringstream ss;
                    ss << "Difference vs. " << json_inpfname;
                    print_plot(json_diffname, ss.str());
                }
            }

            if(cout_output())
            {
                print_cout(diff_stream);
            }
        }

        print_custom();
    }

    TIMEMORY_COLD void update_data() override;
    TIMEMORY_COLD void setup() override;
    TIMEMORY_COLD void read_json() override;

    TIMEMORY_COLD void print_dart() override;
    TIMEMORY_COLD void print_custom() override
    {
        try
        {
            callback(this);
        } catch(std::exception& e)
        {
            fprintf(stderr, "Exception: %s\n", e.what());
        }
    }
    TIMEMORY_COLD void        write_stream(stream_type& stream, result_type& results);
    TIMEMORY_COLD void        print_json(const std::string& fname, result_type& results,
                                         result_tree& rt);
    TIMEMORY_COLD const auto& get_data() const { return data; }
    TIMEMORY_COLD const auto& get_node_results() const { return node_results; }
    TIMEMORY_COLD const auto& get_node_input() const { return node_input; }
    TIMEMORY_COLD const auto& get_node_delta() const { return node_delta; }

    TIMEMORY_COLD std::vector<result_node*> get_flattened(result_type& results)
    {
        std::vector<result_node*> flat;
        for(auto& ritr : results)
        {
            for(auto& itr : ritr)
            {
                flat.push_back(&itr);
            }
        }
        return flat;
    }

protected:
    // do not lint misc-non-private-member-variables-in-classes
    storage_type* data         = nullptr;                 // NOLINT
    callback_type callback     = get_default_callback();  // NOLINT
    result_type   node_results = {};                      // NOLINT
    result_type   node_input   = {};                      // NOLINT
    result_type   node_delta   = {};                      // NOLINT
    result_tree   node_tree    = {};                      // NOLINT
};
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
struct print<Tp, false> : base::print
{
    template <typename... Args>
    print(Args&&...)
    {}
};
//
//--------------------------------------------------------------------------------------//
//
}  // namespace finalize
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
using insert_node = push_node<T>;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
