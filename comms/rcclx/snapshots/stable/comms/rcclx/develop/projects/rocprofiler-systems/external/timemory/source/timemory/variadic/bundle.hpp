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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

#pragma once

#ifndef TIMEMORY_VARIADIC_BUNDLE_HPP_
#    define TIMEMORY_VARIADIC_BUNDLE_HPP_
#endif

#include "timemory/general/source_location.hpp"
#include "timemory/hash/types.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/operations/types.hpp"
#include "timemory/utility/demangle.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/variadic/base_bundle.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <type_traits>

namespace tim
{
/// \class tim::bundle
/// \brief Static polymorphic base class for component bundlers
template <typename... Tp>
class bundle;
//
/// \class tim::bundle< Tag, BundleT, TupleT >
/// \tparam Tag API tag type, e.g. `TIMEMORY_API`
/// \tparam BundleT The empty or empty + tag derived type.
/// \tparam TupleT The set of components wrapped in a type which provides the appropriate
/// aliases.
///
/// \brief Example:
/// `bundle<Tag, component_bundle<Foo>, mixed_wrapper_types<concat<Bar, Baz>>>` will
/// use `Tag` + `trait::is_available<Tag>` or `trait::runtime_available<Tag>` to disable
/// this bundle at compile-time or run-time, respectively. It will covert
/// `component_bundle<Foo>` to `component_bundle<Foo, Bar, Baz>` for purposes of function
/// signatures and it will instantiate `std::tuple<...>` depending on the compile-time
/// availability of `Bar` and `Baz`. `mixed_wrapper_types` is a dummy type with the
/// appropriate aliases to perform these conversions. Here is a theoretical implementation
/// of `mixed_wrapper_types` (which supports allocating components on the stack and the
/// heap):
///
/// \code{.cpp}
/// template <typename... T>
/// struct heap_wrapper_types
/// {
///     TIMEMORY_DELETED_OBJECT(heap_wrapper_types)
///
///     /// the set of types, unaltered, in a type_list
///     using type_list_type = type_list<T...>;
///
///     /// the set of types without any pointers
///     using reference_type = type_list<std::remove_pointer_t<T>...>;
///
///     /// type list of the available types
///     using available_type = type_list_t<reference_type>;
///
///     /// the original bundle type
///     template <typename BundleT>
///     using this_type = convert_t<type_list<T...>, BundleT>;
///
///     /// the type after available_t<concat<...>>
///     template <typename BundleT>
///     using type = convert_t<available_type, BundleT>;
///
///     /// conversion to equivalent wrapper requiring explicit start/stop
///     template <typename BundleT>
///     using component_type = convert_t<type_list<T...>, BundleT>;
///
///     /// conversion to equivalent wrapper which automatically starts/stops
///     template <typename BundleT>
///     using auto_type = concepts::auto_type_t<convert_t<type_list_type, BundleT>>;
///
///     /// the valid types to instantiate in a tuple
///     template <typename ApiT = TIMEMORY_API>
///     using data_type = conditional_t<
///         trait::is_available<ApiT>::value,
///         convert_t<add_pointer_if_not_t<non_placeholder_t<non_quirk_t<type_list_t<T...>>>>,
///                   std::tuple<>>,
///         std::tuple<>>;
/// };
/// \endcode
template <typename Tag, typename BundleT, typename TupleT>
class bundle<Tag, BundleT, TupleT>
: public api_bundle<Tag, typename TupleT::available_type>
{
    template <typename U>
    using remove_pointer_decay_t = std::remove_pointer_t<decay_t<U>>;

    template <typename... Tp>
    friend class impl::base_bundle;

    template <typename... Tp>
    friend class auto_base_bundle;

    template <typename... Tp>
    friend class auto_bundle;

    template <typename... Tp>
    friend class auto_tuple;

    template <typename... Tp>
    friend class auto_list;

#if defined(TIMEMORY_USE_DEPRECATED)
    template <typename TupleC, typename ListC>
    friend class component_hybrid;

    template <typename TupleC, typename ListC>
    friend class auto_hybrid;
#endif

    using internal_tag = tim::variadic::impl::internal_tag;

protected:
    using apply_v        = mpl::apply<void>;
    using bundle_type    = api_bundle<Tag, typename TupleT::available_type>;
    using string_t       = typename bundle_type::string_t;
    using reference_type = typename TupleT::reference_type;

public:
    using captured_location_t = source_location::captured;
    using this_type           = typename TupleT::template this_type<BundleT>;
    using type                = typename TupleT::template type<BundleT>;
    using component_type      = typename TupleT::template component_type<BundleT>;
    using data_type           = typename TupleT::template data_type<Tag>;
    using type_list_type      = typename TupleT::type_list_type;
    using value_type          = data_type;
    using size_type           = typename bundle_type::size_type;
    using initializer_type    = std::function<void(this_type&)>;
    using transient_func_t    = utility::transient_function<void(this_type&)>;

public:
    static initializer_type& get_initializer();

    template <typename T, typename... U>
    using quirk_config = tim::variadic::impl::quirk_config<T, reference_type, U...>;

    /// Query whether type matches this_type
    template <typename U>
    static constexpr bool is_this_type()
    {
        return std::is_same<decay_t<U>, this_type>::value;
    }

public:
    bundle();

    template <typename... T>
    explicit bundle(const string_t& _key, quirk::config<T...>,
                    transient_func_t = get_initializer());

    template <typename... T>
    explicit bundle(hash_value_t _hash, quirk::config<T...>,
                    transient_func_t = get_initializer());

    template <typename... T>
    explicit bundle(const captured_location_t& _loc, quirk::config<T...>,
                    transient_func_t = get_initializer());

    template <typename... T>
    explicit bundle(const string_t& _key, bool _store, quirk::config<T...>,
                    transient_func_t = get_initializer());

    template <typename... T>
    explicit bundle(const captured_location_t& _loc, bool _store, quirk::config<T...>,
                    transient_func_t = get_initializer());

    explicit bundle(hash_value_t _hash, bool _store = true,
                    scope::config _scope = scope::get_default(),
                    transient_func_t     = get_initializer());

    explicit bundle(const string_t& _key, bool _store = true,
                    scope::config _scope = scope::get_default(),
                    transient_func_t     = get_initializer());

    explicit bundle(const captured_location_t& _loc, bool _store = true,
                    scope::config _scope = scope::get_default(),
                    transient_func_t     = get_initializer());

    explicit bundle(hash_value_t _hash, scope::config _scope,
                    transient_func_t = get_initializer());

    explicit bundle(const string_t& _key, scope::config _scope,
                    transient_func_t = get_initializer());

    explicit bundle(const captured_location_t& _loc, scope::config _scope,
                    transient_func_t = get_initializer());

    ~bundle();
    bundle(const bundle& rhs);
    bundle(bundle&&) = default;

    // this_type operators
    bundle& operator=(const bundle& rhs);
    bundle& operator=(bundle&&) = default;

    this_type& operator-=(const this_type& rhs);
    this_type& operator+=(const this_type& rhs);

    // generic operators
    template <typename Op, enable_if_t<!is_this_type<Op>()> = 0>
    this_type& operator-=(Op&& rhs)
    {
        invoke::invoke<operation::minus, Tag>(m_data, std::forward<Op>(rhs));
        return get_this_type();
    }

    template <typename Op, enable_if_t<!is_this_type<Op>()> = 0>
    this_type& operator+=(Op&& rhs)
    {
        invoke::invoke<operation::plus, Tag>(m_data, std::forward<Op>(rhs));
        return get_this_type();
    }

    template <typename Op, enable_if_t<!is_this_type<Op>()> = 0>
    this_type& operator*=(Op&& rhs)
    {
        invoke::invoke<operation::multiply, Tag>(m_data, std::forward<Op>(rhs));
        return get_this_type();
    }

    template <typename Op, enable_if_t<!is_this_type<Op>()> = 0>
    this_type& operator/=(Op&& rhs)
    {
        invoke::invoke<operation::divide, Tag>(m_data, std::forward<Op>(rhs));
        return get_this_type();
    }

    // friend operators
    friend this_type operator+(this_type lhs, const this_type& rhs) { return lhs += rhs; }
    friend this_type operator-(this_type lhs, const this_type& rhs) { return lhs -= rhs; }

    template <typename Op>
    friend enable_if_t<!std::is_arithmetic<Op>::value, this_type> operator*(this_type lhs,
                                                                            const Op& rhs)
    {
        return lhs *= rhs;
    }

    template <typename Op>
    friend enable_if_t<!std::is_arithmetic<Op>::value, this_type> operator/(this_type lhs,
                                                                            const Op& rhs)
    {
        return lhs /= rhs;
    }

    friend std::ostream& operator<<(std::ostream& os, const bundle& obj)
    {
        obj.print<true, true>(os);
        return os;
    }

    bundle clone(bool store, scope::config _scope = scope::get_default());

public:
    /// Query at compile-time whether a user_bundle exists in the set of components.
    /// user_bundle are more restricted versions of component bundlers but allow
    /// runtime insertion of components.
    static constexpr bool has_user_bundle() { return bundle_type::has_user_bundle_v; }

    /// Query at compile-time whether initialization can occur on the stack
    template <typename U>
    static constexpr bool can_stack_init()
    {
        using T = remove_optional_t<remove_pointer_decay_t<U>>;
        return trait::is_available<T>::value &&
               (is_one_of<T, data_type>::value ||
                is_one_of<std::optional<T>, data_type>::value);
    }

    /// Query at compile-time whether initialization can occur on the heap
    template <typename U>
    static constexpr bool can_heap_init()
    {
        using T = remove_pointer_decay_t<U>;
        return trait::is_available<T>::value && is_one_of<T*, data_type>::value;
    }

    /// Query at compile-time whether initialization can occur via a placement new.
    /// Placement new init allows for the combination of optional initialization without
    /// a heap allocation. Not currently available.
    template <typename U>
    static constexpr bool can_placement_init()
    {
        return false;
    }

    /// Query at compile-time whether the specified type can be initialized
    template <typename U>
    static constexpr bool can_init()
    {
        using T = remove_pointer_decay_t<U>;
        return can_stack_init<T>() || can_heap_init<T>() || can_placement_init<T>() ||
               (trait::is_available<T>::value && has_user_bundle());
    }

    /// Query at compile-time whether initialization will occur on the heap.
    /// `can_heap_init<T>() && !will_heap_init<T>()` will indicate that a
    /// stack-allocated instance of the same type exists.
    template <typename U>
    static constexpr bool will_heap_init()
    {
        using T = remove_pointer_decay_t<U>;
        return can_init<T>() && !can_stack_init<T>();
    }

    /// Query at compile-time whether initialization will happen with an opaque wrapper
    /// (i.e. via user bundle). In this situation, initialization arguments are ignored
    /// and the component will only be initialized with the \ref tim::scope::config of the
    /// bundle.
    template <typename U>
    static constexpr bool will_opaque_init()
    {
        using T = remove_pointer_decay_t<U>;
        return !can_stack_init<T>() && !can_heap_init<T>() &&
               (trait::is_available<T>::value && has_user_bundle());
    }

    /// Query at compile-time whether type can be constructed with the given argument
    /// types. if type is not available, then it probably won't be defined, so this
    /// places the `trait::is_available<T>` check in the template parameters and the
    /// `std::is_constructible<T, ...>` in the body to avoid undefined behavior
    template <typename U, typename... Args, typename T = remove_pointer_decay_t<U>,
              enable_if_t<trait::is_available<T>::value> = 0>
    static constexpr bool is_constructible()
    {
        return std::is_constructible<T, Args...>::value;
    }

    /// Return false if the type is not available
    template <typename U, typename... Args, typename T = remove_pointer_decay_t<U>,
              enable_if_t<!trait::is_available<T>::value> = 0>
    static constexpr bool is_constructible()
    {
        return false;
    }

    /// Query at compile-time whether type supports default construction. If type is not
    /// available, then it probably won't be defined, so this places the
    /// `trait::is_available<T>` check in the template parameters and the
    /// `std::is_constructible<T, ...>` in the body to avoid undefined behavior
    template <typename U, typename T = remove_pointer_decay_t<U>,
              enable_if_t<trait::is_available<T>::value> = 0>
    static constexpr bool is_default_constructible()
    {
        return std::is_default_constructible<T>::value;
    }

    /// Return false if the type is not available
    template <typename U, typename T = remove_pointer_decay_t<U>,
              enable_if_t<!trait::is_available<T>::value> = 0>
    static constexpr bool is_default_constructible()
    {
        return false;
    }

public:
    /// requests the component initialize their storage
    static void init_storage();

    /// tells each component to push itself into the call-stack hierarchy
    this_type& push(int64_t _tid = threading::get_id());

    /// tells each component to pop itself off of the call-stack hierarchy
    this_type& pop(int64_t _tid = threading::get_id());

    /// selective push
    template <typename... Tp>
    this_type& push(mpl::piecewise_select<Tp...>, int64_t _tid = threading::get_id());

    /// selective push
    template <typename... Tp>
    this_type& push(mpl::piecewise_ignore<Tp...>, int64_t _tid = threading::get_id());

    /// selective push with scope configuration
    template <typename... Tp>
    this_type& push(mpl::piecewise_select<Tp...>, scope::config,
                    int64_t _tid = threading::get_id());

    /// selective push with scope configuration
    template <typename... Tp>
    this_type& push(mpl::piecewise_ignore<Tp...>, scope::config,
                    int64_t _tid = threading::get_id());

    /// selective pop
    template <typename... Tp>
    this_type& pop(mpl::piecewise_select<Tp...>, int64_t _tid = threading::get_id());

    /// selective pop
    template <typename... Tp>
    this_type& pop(mpl::piecewise_ignore<Tp...>, int64_t _tid = threading::get_id());

    /// requests each component record a measurment
    template <typename... Args>
    this_type& measure(Args&&...);

    /// requests each component take a sample (if supported)
    template <typename... Args>
    this_type& sample(Args&&...);

    /// invokes start on all the components
    template <typename... Args>
    this_type& start(Args&&...);

    /// invokes stop on all the components
    template <typename... Args>
    this_type& stop(Args&&...);

    /// requests each component perform a measurement
    template <typename... Args>
    this_type& record(Args&&...);

    /// invokes reset member function on all the components
    template <typename... Args>
    this_type& reset(Args&&...);

    /// variant of start() which excludes push()
    template <typename... Args>
    this_type& start(mpl::lightweight, Args&&...);

    /// variant of stop() which excludes pop()
    template <typename... Args>
    this_type& stop(mpl::lightweight, Args&&...);

    /// variant of start() which only gets applied to Tp types
    template <typename... Tp, typename... Args>
    this_type& start(mpl::piecewise_select<Tp...>, Args&&...);

    /// variant of start() which gets applied to non-Tp types
    template <typename... Tp, typename... Args>
    this_type& start(mpl::piecewise_ignore<Tp...>, Args&&...);

    /// variant of stop() which only gets applied to Tp types
    template <typename... Tp, typename... Args>
    this_type& stop(mpl::piecewise_select<Tp...>, Args&&...);

    /// variant of stop() which gets applied to non-Tp types
    template <typename... Tp, typename... Args>
    this_type& stop(mpl::piecewise_ignore<Tp...>, Args&&...);

    using bundle_type::get_prefix;
    using bundle_type::get_scope;
    using bundle_type::get_store;
    using bundle_type::hash;
    using bundle_type::key;
    using bundle_type::laps;
    using bundle_type::prefix;
    using bundle_type::size;
    using bundle_type::store;
    using bundle_type::tid;

    /// query the number of (compile-time) fixed components
    static constexpr uint64_t fixed_count();

    /// query the number of (run-time) optional components
    static constexpr uint64_t optional_count();

    /// number of objects that will be performing measurements
    uint64_t count();

    /// when chaining together operations, this function enables executing a function
    /// inside the chain
    template <typename FuncT, typename... Args>
    decltype(auto) execute(FuncT&& func, Args&&... args);

    /// construct the objects that have constructors with matching arguments
    template <typename... Args>
    this_type& construct(Args&&... _args);

    /// provide preliminary info to the objects with matching arguments. This is typically
    /// used to notify a component that it has been bundled alongside another component
    /// that it can extract data from.
    template <typename... Args>
    this_type& assemble(Args&&... _args);

    /// provide conclusive info to the objects with matching arguments. This is typically
    /// used by components to extract data from another component it has been bundled
    /// alongside, e.g. the cpu_util component can extract data from \ref
    /// tim::component::wall_clock and \ref tim::component::cpu_clock
    template <typename... Args>
    this_type& derive(Args&&... _args);

    /// mark an atomic event
    template <typename... Args>
    this_type& mark(Args&&... _args);

    /// mark a beginning position in the execution (typically used by asynchronous
    /// structures)
    template <typename... Args>
    this_type& mark_begin(Args&&... _args);

    /// mark a beginning position in the execution (typically used by asynchronous
    /// structures)
    template <typename... Args>
    this_type& mark_end(Args&&... _args);

    /// store a value
    template <typename... Args>
    this_type& store(Args&&... _args);

    /// allow the components to inspect the incoming arguments before start
    /// or out-going return value before returning (typically using in GOTCHA components)
    template <typename... Args>
    this_type& audit(Args&&... _args);

    /// perform an add_secondary operation. This operation allows components to add
    /// additional entries to storage which are their direct descendant
    template <typename... Args>
    this_type& add_secondary(Args&&... _args);

    /// perform an add_secondary operation. This operation allows components to add
    /// additional entries to storage which are their direct descendant
    template <typename... Args>
    this_type& update_statistics(Args&&... _args);

    /// pass data onto the components
    template <typename... Args>
    this_type& set_data(Args&&... _args);

    /// generic member function for invoking user-provided operations
    /// \tparam OpT Operation struct
    template <template <typename> class OpT, typename... Args>
    this_type& invoke(Args&&... _args);

    /// generic member function for invoking user-provided operations on a specific
    /// set of component types
    /// \tparam OpT Operation struct
    template <template <typename> class OpT, typename... Tp, typename... Args>
    this_type& invoke(mpl::piecewise_select<Tp...>, Args&&... _args);

    /// generic member function for invoking user-provided operations on all types
    /// that are not listed
    /// \tparam OpT Operation struct
    template <template <typename> class OpT, typename... Tp, typename... Args>
    this_type& invoke(mpl::piecewise_ignore<Tp...>, Args&&... _args);

    template <template <typename> class OpT, typename... Tp, typename... Args>
    this_type& invoke_with(type_list<Tp...>&&, Args&&... _args);

    template <template <typename> class OpT, size_t... Idx, typename... Args>
    this_type& invoke_with(std::index_sequence<Idx...>&&, Args&&... _args);

    template <bool PrintPrefix = true, bool PrintLaps = true>
    this_type& print(std::ostream& os, bool _endl = false) const;

    /// returns a tuple of invoking get() on all the components
    template <typename... Args>
    auto get(Args&&...) const;

    /// returns a tuple of the component label + invoking get() on all the components
    template <typename... Args>
    auto get_labeled(Args&&...) const;

    /// returns a reference to the underlying tuple of components
    data_type& data();

    /// returns a const reference to the underlying tuple of components
    const data_type& data() const;

    /// get a component from the bundle
    template <typename U>
    decltype(auto) get();

    /// get a component from the bundle
    template <typename U>
    decltype(auto) get() const;

    /// get a component from the bundle and apply function if the pointer is valid
    template <typename U, typename FuncT>
    decltype(auto) get(FuncT&&);

    /// get a component from the bundle
    template <typename U, typename FuncT>
    decltype(auto) get(FuncT&&) const;

    /// performs an opaque search. Opaque searches are generally provided by user_bundles
    /// with a functor such as this:
    ///
    /// \code{.cpp}
    /// auto _get = [](void* v_this, void*& ptr, size_t _hash) {
    /// {
    ///     if(!ptr && v_this && _hash == typeid_hash<Tp>())
    ///     {
    ///         Tp* _this = static_cast<Tp*>(v_this);
    ///         _this->get(ptr, _hash);
    ///     }
    ///     return ptr;
    /// };
    /// \endcode
    ///
    /// And the component provides this function:
    ///
    /// \code{.cpp}
    /// template <typename Tp, typename Value>
    /// void
    /// base<Tp, Value>::get(void*& ptr, size_t _hash) const
    /// {
    ///     if(!ptr && _hash == typeid_hash<Tp>())
    ///         ptr = reinterpret_cast<void*>(const_cast<base_type*>(this));
    /// }
    /// \endcode
    ///
    this_type& get(void*& ptr, hash_value_t _hash) const;

    /// this is a simple alternative to get<T>() when used from SFINAE in operation
    /// namespace which has a struct get also templated. Usage there can cause error
    /// with older compilers
    template <typename U>
    auto get_component(
        enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                        is_one_of<remove_pointer_decay_t<U>, reference_type>::value,
                    int> = 0);

    /// returns a reference from a stack component instead of a pointer
    template <typename U>
    auto& get_reference(
        enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                        is_one_of<remove_pointer_decay_t<U>, reference_type>::value,
                    int> = 0);

    /// create an optional type that is in variadic list AND is available AND
    /// accepts arguments
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    enable_if_t<will_heap_init<T>() && !will_opaque_init<T>(), bool> init(
        Args&&... _args, enable_if_t<is_constructible<T, Args...>(), int> = 0)
    {
        T*& _obj = std::get<index_of<T*, data_type>::value>(m_data);
        if(!_obj)
        {
            if(settings::debug() && settings::verbose() > 1)
            {
                printf("[bundle::init]> initializing type '%s'...\n",
                       demangle<T>().c_str());
            }
            if(!bundle_type::init_buffer())
            {
                _obj = new T{ std::forward<Args>(_args)... };
            }
            else
            {
                auto _val = T{ std::forward<Args>(_args)... };
                _obj      = m_buffer->write(&_val).second;
            }
            set_prefix(_obj, internal_tag{});
            set_scope(_obj, internal_tag{});
            return true;
        }

        static std::atomic<int> _count(0);
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            std::string _id = demangle<T>();
            fprintf(stderr,
                    "[bundle::init]> skipping re-initialization of type"
                    " \"%s\"...\n",
                    _id.c_str());
        }

        return false;
    }

    /// create an optional type that is in variadic list AND is available but is not
    /// constructible with provided arguments
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    enable_if_t<will_heap_init<T>() && !will_opaque_init<T>(), bool> init(
        Args&&...,
        enable_if_t<!is_constructible<T, Args...>() && is_default_constructible<T>(),
                    long> = 0)
    {
        T*& _obj = std::get<index_of<T*, data_type>::value>(m_data);
        if(!_obj)
        {
            if(settings::debug() && settings::verbose() > 1)
            {
                fprintf(stderr, "[bundle::init]> initializing type '%s'...\n",
                        demangle<T>().c_str());
            }
            if(!bundle_type::init_buffer())
            {
                _obj = new T{};
            }
            else
            {
                T in{};
                _obj = m_buffer->write(&in).second;
            }
            set_prefix(_obj, internal_tag{});
            set_scope(_obj, internal_tag{});
            return true;
        }

        static std::atomic<int> _count(0);
        if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
        {
            fprintf(stderr,
                    "[bundle::init]> skipping re-initialization of type "
                    "\"%s\"...\n",
                    demangle<T>().c_str());
        }

        return false;
    }

    /// try to re-create a stack object with provided arguments
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    bool init(Args&&... _args, enable_if_t<can_stack_init<T>(), int> = 0)
    {
        IF_CONSTEXPR(can_heap_init<T>())
        {
            static std::atomic<int> _count(0);
            if((settings::verbose() > 1 || settings::debug()) && _count++ == 0)
            {
                fprintf(stderr,
                        "[bundle::init]> type exists as a heap-allocated instance "
                        "and stack-allocated instance: \"%s\"...\n",
                        demangle<T>().c_str());
            }
        }

        if constexpr(is_one_of<std::optional<T>, data_type>::value)
        {
            std::optional<T>& _obj =
                std::get<index_of<std::optional<T>, data_type>::value>(m_data);
            if constexpr(std::is_constructible<T, Args...>::value)
                _obj = T{ std::forward<Args>(_args)... };
            else if constexpr(std::is_default_constructible<T>::value)
                _obj = T{};
            else
                static_assert(std::is_empty<T>::value,
                              "Error! unsupported initialization");
            set_prefix(&*_obj, internal_tag{});
            set_scope(&*_obj, internal_tag{});
        }
        else
        {
            T& _obj = std::get<index_of<T, data_type>::value>(m_data);
            operation::construct<T>{ _obj, std::forward<Args>(_args)... };
            set_prefix(&_obj, internal_tag{});
            set_scope(&_obj, internal_tag{});
        }
        return true;
    }

    /// if a type is not in variadic list but a \ref tim::component::user_bundle is
    /// available, add it in there
    template <typename U, typename T = remove_pointer_decay_t<U>, typename... Args>
    bool init(Args&&..., enable_if_t<will_opaque_init<T>(), long> = 0)
    {
        using bundle_t = decay_t<decltype(
            std::get<0>(std::declval<typename bundle_type::user_bundle_types>()))>;
        static_assert(trait::is_user_bundle<bundle_t>::value, "Error! Not a user_bundle");
        this->init<remove_optional_t<bundle_t>>();
        auto* _bundle = this->get<bundle_t>();
        if(_bundle)
        {
            _bundle->insert(component::factory::get_opaque<T>(m_scope),
                            component::factory::get_typeids<T>());
            return true;
        }
        return false;
    }

    /// do nothing if type not available, not one of the variadic types, and there
    /// is no user bundle available
    template <typename U, typename T = remove_optional_t<decay_t<U>>, typename... Args>
    enable_if_t<!trait::is_available<T>::value ||
                    !(is_one_of<T*, data_type>::value || is_one_of<T, data_type>::value ||
                      is_one_of<std::optional<T>, data_type>::value ||
                      bundle_type::has_user_bundle_v),
                bool>
    init(Args&&...)
    {
        return false;
    }

    /// \brief variadic initialization
    /// \tparam T components to initialize
    /// \tparam Args arguments to pass to the construction of the component
    template <typename... T, typename... Args>
    std::array<bool, sizeof...(T)> initialize(Args&&... args);

    /// delete any optional types currently allocated
    template <typename... Tail>
    this_type& disable();

    /// apply a member function to a stack type that is in variadic list AND is available
    template <typename T, typename Func, typename... Args,
              enable_if_t<is_one_of<T, data_type>::value, int> = 0>
    this_type& type_apply(Func&& _func, Args&&... _args)
    {
        auto* _obj = get<T>();
        ((*_obj).*(_func))(std::forward<Args>(_args)...);
        return get_this_type();
    }

    /// apply a member function to either a heap type or a type that is in a user_bundle
    template <typename T, typename Func, typename... Args,
              enable_if_t<trait::is_available<T>::value, int> = 0>
    this_type& type_apply(Func&& _func, Args&&... _args);

    /// ignore applying a member function because the type is not present
    template <typename T, typename Func, typename... Args,
              enable_if_t<!trait::is_available<T>::value, int> = 0>
    this_type& type_apply(Func&&, Args&&...);

    void       set_prefix(const string_t&) const;
    this_type& set_prefix(hash_value_t) const;
    this_type& set_prefix(captured_location_t) const;
    this_type& set_scope(scope::config);

    TIMEMORY_INLINE void rekey(const string_t& _key);
    TIMEMORY_INLINE void rekey(captured_location_t _loc);
    TIMEMORY_INLINE void rekey(uint64_t _hash);

    /// returns a stack-object for calling stop
    scope::transient_destructor get_scope_destructor();

    /// returns a stack-object for calling some member functions when the scope is exited.
    scope::transient_destructor get_scope_destructor(
        utility::transient_function<void(this_type&)>);

    const data_type& get_data() const { return m_data; }

protected:
    template <typename T>
    void set_scope(T* obj, internal_tag) const;

    template <typename T>
    void set_prefix(T* obj, internal_tag) const;

protected:
    // objects
    using bundle_type::m_buffer;
    using bundle_type::m_config;
    using bundle_type::m_enabled;
    using bundle_type::m_hash;
    using bundle_type::m_is_active;
    using bundle_type::m_is_pushed;
    using bundle_type::m_laps;
    using bundle_type::m_scope;
    using bundle_type::m_store;
    mutable data_type m_data{};

private:
    this_type& get_this_type() { return static_cast<this_type&>(*this); }
    this_type& get_this_type() const
    {
        return const_cast<this_type&>(static_cast<const this_type&>(*this));
    }

    static this_type*& get_last_instance()
    {
        static thread_local this_type* _instance = nullptr;
        return _instance;
    }

    static void update_last_instance(this_type*  _new_instance,
                                     this_type*& _old_instance = get_last_instance(),
                                     bool        _stop_last    = false)
    {
        if(_stop_last && _old_instance && _old_instance != _new_instance)
            _old_instance->stop();
        _old_instance = _new_instance;
    }

    template <template <typename> class OpT, typename... Tp, typename... Args>
    void invoke_piecewise(type_list<Tp...>, Args&&... _args);

public:
    // archive serialization
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int);
};
//
//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
constexpr uint64_t
bundle<Tag, BundleT, TupleT>::fixed_count()
{
    return (size() -
            mpl::get_tuple_size<
                typename mpl::get_true_types<tim::is_optional, data_type>::type>::value);
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
constexpr uint64_t
bundle<Tag, BundleT, TupleT>::optional_count()
{
    return mpl::get_tuple_size<
        typename mpl::get_true_types<tim::is_optional, data_type>::type>::value;
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename FuncT, typename... Args>
decltype(auto)
bundle<Tag, BundleT, TupleT>::execute(FuncT&& func, Args&&... args)
{
    return mpl::execute(get_this_type(),
                        std::forward<FuncT>(func)(std::forward<Args>(args)...));
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
decltype(auto)
bundle<Tag, BundleT, TupleT>::get()
{
    U* _val = nullptr;
    if constexpr(is_one_of<U, reference_type>::value)
        _val = tim::variadic::impl::get<U, Tag>(m_data);

    if(!_val)
    {
        void* _ptr = nullptr;
        tim::variadic::impl::get<Tag>(m_data, _ptr, typeid_hash<U>());
        if(_ptr)
            _val = static_cast<U*>(_ptr);
    }

    return _val;
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
decltype(auto)
bundle<Tag, BundleT, TupleT>::get() const
{
    return tim::variadic::impl::get<U, Tag>(m_data);
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U, typename FuncT>
decltype(auto)
bundle<Tag, BundleT, TupleT>::get(FuncT&& _func)
{
    auto* _obj = tim::variadic::impl::get<U, Tag>(m_data);
    if(_obj)
        return std::forward<FuncT>(_func)(_obj);
    using return_type = decltype(_func(std::declval<decltype(_obj)>()));
#if defined(CXX17)
    if constexpr(std::is_void<decay_t<return_type>>::value)
        return;
    else if constexpr(std::is_pointer<decay_t<return_type>>::value)
        return static_cast<decay_t<return_type>>(nullptr);
    else
        return decltype(_func(std::declval<decltype(_obj)>())){};
#else
    return return_type();
#endif
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U, typename FuncT>
decltype(auto)
bundle<Tag, BundleT, TupleT>::get(FuncT&& _func) const
{
    auto* _obj = tim::variadic::impl::get<U, Tag>(m_data);
    if(_obj)
        return std::forward<FuncT>(_func)(_obj);
    using return_type = decltype(_func(std::declval<decltype(_obj)>()));
#if defined(CXX17)
    if constexpr(std::is_void<decay_t<return_type>>::value)
        return;
    else if constexpr(std::is_pointer<decay_t<return_type>>::value)
        return static_cast<decay_t<return_type>>(nullptr);
    else
        return decltype(_func(std::declval<decltype(_obj)>())){};
#else
    return return_type();
#endif
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
auto bundle<Tag, BundleT, TupleT>::get_component(
    enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                    is_one_of<remove_pointer_decay_t<U>, reference_type>::value,
                int>)
{
    return get<remove_pointer_decay_t<U>>();
}

//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
template <typename U>
auto& bundle<Tag, BundleT, TupleT>::get_reference(
    enable_if_t<trait::is_available<remove_pointer_decay_t<U>>::value &&
                    is_one_of<remove_pointer_decay_t<U>, reference_type>::value,
                int>)
{
    using Up = remove_pointer_decay_t<U>;
    if constexpr(is_one_of<std::optional<Up>, data_type>::value)
        return std::get<index_of<std::optional<Up>, data_type>::value>(m_data);
    else if constexpr(is_one_of<Up*, data_type>::value)
        return std::get<index_of<Up*, data_type>::value>(m_data);
    else
        return std::get<index_of<Up, data_type>::value>(m_data);
}

//
//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
void
bundle<Tag, BundleT, TupleT>::rekey(const string_t& _key)
{
    m_hash = add_hash_id(_key);
    set_prefix(_key);
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
void
bundle<Tag, BundleT, TupleT>::rekey(captured_location_t _loc)
{
    m_hash = _loc.get_hash();
    set_prefix(_loc);
}
//
//----------------------------------------------------------------------------------//
//
template <typename Tag, typename BundleT, typename TupleT>
void
bundle<Tag, BundleT, TupleT>::rekey(uint64_t _hash)
{
    m_hash = _hash;
    set_prefix(_hash);
}
}  // namespace tim

#ifndef TIMEMORY_VARIADIC_BUNDLE_CPP_
#    include "timemory/variadic/bundle.cpp"
#endif
