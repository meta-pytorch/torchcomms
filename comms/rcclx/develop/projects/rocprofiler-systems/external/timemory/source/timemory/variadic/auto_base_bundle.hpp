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
//

#pragma once

#include "timemory/general/source_location.hpp"
#include "timemory/mpl/filters.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/variadic/component_bundle.hpp"
#include "timemory/variadic/macros.hpp"
#include "timemory/variadic/types.hpp"

#include <cstdint>
#include <string>

namespace tim
{
/// \class tim::auto_base_bundle
/// \brief Static polymorphic base class for automatic start/stop bundlers
//
/// \class tim::auto_base_bundle< Tag, CompT, BundleT >
/// \tparam Tag Unique identifying type for the bundle which when \ref
/// tim::trait::is_available<Tag> is false at compile-time or \ref
/// tim::trait::runtime_enabled<Tag>() is false at runtime, then none of the components
/// will be collected
/// \tparam CompT The empty or empty + tag non-auto type which will be wrapped
/// \tparam BundleT Derived data type
///
/// \brief Example:
/// `auto_base_bundle<Tag, component_bundle<Tag>, auto_bundle<Tag, Types...>>` will
/// use `Tag` + `trait::is_available<Tag>` or `trait::runtime_available<Tag>` to disable
/// this bundle at compile-time or run-time, respectively. It will wrap auto-start/stop
/// around `component_bundle<Tag, Types...>` and use `auto_bundle<Tag, Types...>` for
/// function signatures.
template <typename Tag, typename CompT, typename BundleT>
class auto_base_bundle<Tag, CompT, BundleT>
: public concepts::wrapper
, public concepts::variadic
, public concepts::auto_wrapper
, public concepts::tagged
{
    static_assert(concepts::is_api<Tag>::value,
                  "Error! The first template parameter of an 'BundleT' must "
                  "statisfy the 'is_api' concept");

public:
    using this_type           = BundleT;
    using auto_type           = this_type;
    using base_type           = convert_t<BundleT, CompT>;
    using component_type      = typename base_type::component_type;
    using tuple_type          = typename component_type::tuple_type;
    using data_type           = typename component_type::data_type;
    using sample_type         = typename component_type::sample_type;
    using bundle_type         = typename component_type::bundle_type;
    using initializer_type    = std::function<void(this_type&)>;
    using transient_func_t    = utility::transient_function<void(this_type&)>;
    using captured_location_t = typename component_type::captured_location_t;
    using value_type          = component_type;

    static constexpr bool has_gotcha_v      = component_type::has_gotcha_v;
    static constexpr bool has_user_bundle_v = component_type::has_user_bundle_v;

public:
    template <typename T, typename... U>
    using quirk_config =
        tim::variadic::impl::quirk_config<T, convert_t<BundleT, type_list<>>, U...>;

public:
    //
    static void init_storage() { component_type::init_storage(); }
    //
    static initializer_type& get_initializer()
    {
        static initializer_type _instance = [](this_type&) {};
        return _instance;
    }
    //
    static initializer_type& get_finalizer()
    {
        static initializer_type _instance = [](this_type&) {};
        return _instance;
    }

public:
    auto_base_bundle();

    template <typename... T>
    auto_base_bundle(string_view_cref_t, quirk::config<T...>,
                     transient_func_t = get_initializer());

    template <typename... T>
    auto_base_bundle(const captured_location_t&, quirk::config<T...>,
                     transient_func_t = get_initializer());

    explicit auto_base_bundle(string_view_cref_t, scope::config = scope::get_default(),
                              bool report_at_exit = settings::destructor_report(),
                              transient_func_t    = get_initializer());

    explicit auto_base_bundle(const captured_location_t&,
                              scope::config       = scope::get_default(),
                              bool report_at_exit = settings::destructor_report(),
                              transient_func_t    = get_initializer());

    explicit auto_base_bundle(size_t, scope::config = scope::get_default(),
                              bool report_at_exit = settings::destructor_report(),
                              transient_func_t    = get_initializer());

    explicit auto_base_bundle(component_type& tmp, scope::config = scope::get_default(),
                              bool report_at_exit = settings::destructor_report());

    template <typename Arg, typename... Args>
    auto_base_bundle(string_view_cref_t, bool store, scope::config _scope,
                     transient_func_t, Arg&&, Args&&...);

    template <typename Arg, typename... Args>
    auto_base_bundle(const captured_location_t&, bool store, scope::config _scope,
                     transient_func_t, Arg&&, Args&&...);

    template <typename Arg, typename... Args>
    auto_base_bundle(size_t, bool store, scope::config _scope, transient_func_t, Arg&&,
                     Args&&...);

    ~auto_base_bundle();

    // copy and move
    auto_base_bundle(const auto_base_bundle&) = default;
    auto_base_bundle(auto_base_bundle&&)      = default;
    auto_base_bundle& operator=(const auto_base_bundle&) = default;
    auto_base_bundle& operator=(auto_base_bundle&&) = default;

    static constexpr std::size_t size() { return component_type::size(); }

public:
    // public member functions
    component_type&       get_component() { return m_temporary; }
    const component_type& get_component() const { return m_temporary; }

    /// implicit conversion to underlying component_type
    operator component_type&() { return m_temporary; }
    /// implicit conversion to const ref of underlying component_type
    operator const component_type&() const { return m_temporary; }

    /// query the number of (compile-time) fixed components
    static constexpr auto fixed_count() { return component_type::fixed_count(); }
    /// query the number of (run-time) optional components
    static constexpr auto optional_count() { return component_type::optional_count(); }
    /// count number of active components in an instance
    auto count() { return (m_enabled) ? m_temporary.count() : 0; }
    /// when chaining together operations, this enables executing a function inside the
    /// chain
    template <typename FuncT, typename... Args>
    decltype(auto) execute(FuncT&& func, Args&&... args)
    {
        return mpl::execute(static_cast<this_type&>(*this),
                            std::forward<FuncT>(func)(std::forward<Args>(args)...));
    }

    /// push components into call-stack storage
    template <typename... Args>
    this_type& push(Args&&... args);

    /// pop components off call-stack storage
    template <typename... Args>
    this_type& pop(Args&&... args);

    /// execute a measurement
    template <typename... Args>
    this_type& measure(Args&&... args);

    /// record some data
    template <typename... Args>
    this_type& record(Args&&... args);

    /// execute a sample
    template <typename... Args>
    this_type& sample(Args&&... args);

    /// invoke start member function on all components
    template <typename... Args>
    this_type& start(Args&&... args);

    /// invoke stop member function on all components
    template <typename... Args>
    this_type& stop(Args&&... args);

    /// invoke assemble member function on all components to determine if measurements can
    /// be derived from other components in a bundle
    template <typename... Args>
    this_type& assemble(Args&&... args);

    /// invoke derive member function on all components to extract measurements from other
    /// components in the bundle
    template <typename... Args>
    this_type& derive(Args&&... args);

    /// invoke mark member function on all components
    template <typename... Args>
    this_type& mark(Args&&... args);

    /// invoke mark_begin member function on all components
    template <typename... Args>
    this_type& mark_begin(Args&&... args);

    /// invoke mark_begin member function on all components
    template <typename... Args>
    this_type& mark_end(Args&&... args);

    /// invoke store member function on all components
    template <typename... Args>
    this_type& store(Args&&... args);

    /// invoke audit member function on all components
    template <typename... Args>
    this_type& audit(Args&&... args);

    /// add secondary data
    template <typename... Args>
    this_type& add_secondary(Args&&... args);

    /// reset the data
    template <typename... Args>
    this_type& reset(Args&&... args);

    /// modify the scope of the push operation
    template <typename... Args>
    this_type& set_scope(Args&&... args);

    /// set the key
    template <typename... Args>
    this_type& set_prefix(Args&&... args);

    /// invoke the provided operation on all components
    template <template <typename> class OpT, typename... Args>
    this_type& invoke(Args&&... _args);

    /// invoke get member function on all components to get their data
    template <typename... Args>
    auto get(Args&&... args) const
    {
        return m_temporary.get(std::forward<Args>(args)...);
    }

    /// invoke get member function on all components to get data labeled with component
    /// name
    template <typename... Args>
    auto get_labeled(Args&&... args) const
    {
        return m_temporary.get_labeled(std::forward<Args>(args)...);
    }

    /// check whether enabled
    bool enabled() const;

    /// check whether reporting at exit
    bool report_at_exit() const;

    /// check whether storing data
    bool store() const;

    /// get the number of laps
    int64_t laps() const;

    /// get the key hash
    uint64_t hash() const;

    /// get the key
    std::string key() const;

    /// get tuple data
    data_type& data();

    /// get tuple data
    const data_type& data() const;

    /// write to stdout during destruction
    this_type& report_at_exit(bool val);

    /// update key
    this_type& rekey(string_view_cref_t _key);

    /// update key
    this_type& rekey(captured_location_t _loc);

    /// update key
    this_type& rekey(uint64_t _hash);

    /// returns a stack-object for calling stop
    scope::transient_destructor get_scope_destructor();

    /// returns a stack-object for calling some member functions when the scope is exited.
    scope::transient_destructor get_scope_destructor(
        utility::transient_function<void(this_type&)>);

public:
    template <typename Tp, typename... Args>
    auto init(Args&&... args)
    {
        m_temporary.template init<Tp>(std::forward<Args>(args)...);
    }

    template <typename... Tp, typename... Args>
    auto initialize(Args&&... args)
    {
        m_temporary.template initialize<Tp...>(std::forward<Args>(args)...);
    }

    template <typename... Tail>
    void disable();

    template <typename Tp, typename... Args>
    decltype(auto) get(Args&&... args)
    {
        return m_temporary.template get<Tp>(std::forward<Args>(args)...);
    }

    template <typename Tp, typename... Args>
    decltype(auto) get(Args&&... args) const
    {
        return m_temporary.template get<Tp>(std::forward<Args>(args)...);
    }

    void get(void*& ptr, size_t hash) const { m_temporary.get(ptr, hash); }

    template <typename T>
    auto get_component()
        -> decltype(std::declval<component_type>().template get_component<T>())
    {
        return m_temporary.template get_component<T>();
    }

    decltype(auto) get_data() const { return m_temporary.get_data(); }

    this_type& operator+=(const this_type& rhs)
    {
        m_temporary += rhs.m_temporary;
        return static_cast<this_type&>(*this);
    }

    this_type& operator-=(const this_type& rhs)
    {
        m_temporary -= rhs.m_temporary;
        return static_cast<this_type&>(*this);
    }

    friend this_type operator+(const this_type& lhs, const this_type& rhs)
    {
        return this_type{ lhs } += rhs;
    }

    friend this_type operator-(const this_type& lhs, const this_type& rhs)
    {
        return this_type{ lhs } -= rhs;
    }

protected:
    void internal_init(transient_func_t _init);

    template <typename Arg, typename... Args>
    void internal_init(transient_func_t _init, Arg&& _arg, Args&&... _args);

protected:
    bool            m_enabled          = true;
    bool            m_report_at_exit   = false;
    component_type* m_reference_object = nullptr;
    component_type  m_temporary;

private:
    this_type& get_this_type() { return static_cast<this_type&>(*this); }
    this_type& get_this_type() const
    {
        return const_cast<this_type&>(static_cast<const this_type&>(*this));
    }
};
//
}  // namespace tim
