//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/**
 * \file timemory/components/gotcha/components.hpp
 * \brief Implementation of the gotcha component(s)
 */

#pragma once

#ifndef TIMEMORY_COMPONENTS_GOTCHA_COMPONENTS_HPP_
#    define TIMEMORY_COMPONENTS_GOTCHA_COMPONENTS_HPP_
#endif

#include "timemory/components/base.hpp"
#include "timemory/components/gotcha/backends.hpp"
#include "timemory/components/gotcha/types.hpp"
#include "timemory/macros.hpp"
#include "timemory/mpl/apply.hpp"
#include "timemory/mpl/function_traits.hpp"
#include "timemory/mpl/quirks.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/settings/declaration.hpp"
#include "timemory/units.hpp"
#include "timemory/utility/locking.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/types.hpp"
#include "timemory/variadic/types.hpp"

#include <type_traits>

//======================================================================================//
//
namespace tim
{
namespace component
{
//
//======================================================================================//
//
// template params:
//      Nt             ==  max number of GOTCHA wrappers
//      BundleT     ==  {auto,component}_{tuple,list,bundle}
//      DiffT ==  extra param to differentiate when Nt and BundleT are same
//
//  TODO: filter any gotcha components out of BundleT
//
/// \struct tim::component::gotcha
/// \tparam Nt Max number of functions which will wrapped by this component
/// \tparam BundleT Component bundle to wrap around the function(s)
/// \tparam DiffT Differentiator type to distinguish different sets of wrappers with
/// identical values of `Nt` and `BundleT` (or provide function call operator if replacing
/// functions instead of wrapping functions)
///
/// \brief The gotcha component rewrites the global offset table such that calling the
/// wrapped function actually invokes either a function which is wrapped by timemory
/// instrumentation or is replaced by a timemory component with an function call operator
/// (`operator()`) whose return value and arguments exactly match the original function.
/// This component is only available on Linux and can only by applied to external,
/// dynamically-linked functions (i.e. functions defined in a shared library).
/// If the `BundleT` template parameter is a non-empty component bundle, this component
/// will surround the original function call with:
///
/// \code{.cpp}
/// bundle_type _obj{ "<NAME-OF-ORIGINAL-FUNCTION>" };
/// _obj.construct(_args...);
/// _obj.start();
/// _obj.audit("<NAME-OF-ORIGINAL-FUNCTION>", _args...);
///
/// Ret _ret = <CALL-ORIGINAL-FUNCTION>
///
/// _obj.audit("<NAME-OF-ORIGINAL-FUNCTION>", _ret);
/// _obj.stop();
/// \endcode
///
/// If the `BundleT` template parameter is an empty variadic class, e.g. `std::tuple<>`,
/// `tim::component_tuple<>`, etc., and the `DiffT` template parameter is a timemory
/// component, the assumption is that the `DiffT` component has a function call operator
/// which should replace the original function call, e.g. `void* malloc(size_t)` can be
/// replaced with a component with `void* operator()(size_t)`, e.g.:
///
/// \code{.cpp}
/// // replace 'double exp(double)'
/// struct exp_replace : base<exp_replace, void>
/// {
///     double operator()(double value)
///     {
///         float result = expf(static_cast<float>(value));
///         return static_cast<double>(result);
///     }
/// };
/// \endcode
///
/// Example usage:
///
/// \code{.cpp}
/// #include <timemory/timemory.hpp>
///
/// #include <cassert>
/// #include <cmath>
/// #include <tuple>
///
/// using empty_tuple_t = std::tuple<>;
/// using base_bundle_t = tim::component_tuple<wall_clock, cpu_clock>;
/// using gotcha_wrap_t = tim::component::gotcha<2, base_bundle_t, void>;
/// using gotcha_repl_t = tim::component::gotcha<2, empty_tuple_t, exp_replace>;
/// using impl_bundle_t = tim::mpl::append_type_t<base_bundle_t,
///                                               tim::type_list<gotcha_wrap_t,
///                                                              gotcha_repl_t>>;
///
/// void init_wrappers()
/// {
///     // wraps the sin and cos math functions
///     gotcha_wrap_t::get_initializer() = []()
///     {
///         TIMEMORY_C_GOTCHA(gotcha_wrap_t, 0, sin);   // index 0 replaces sin
///         TIMEMORY_C_GOTCHA(gotcha_wrap_t, 1, cos);   // index 1 replace cos
///     };
///
///     // replaces the 'exp' function which may be 'exp' in symbols table
///     // or '__exp_finite' in symbols table (use `nm <bindary>` to determine)
///     gotcha_repl_t::get_initializer() = []()
///     {
///         TIMEMORY_C_GOTCHA(gotcha_repl_t, 0, exp);
///         TIMEMORY_DERIVED_GOTCHA(gotcha_repl_t, 1, exp, "__exp_finite");
///     };
/// }
///
/// // the following is useful to avoid having to call 'init_wrappers()' explicitly:
/// // use comma operator to call 'init_wrappers' and return true
/// static auto called_init_at_load = (init_wrappers(), true);
///
/// int main()
/// {
///     assert(called_init_at_load == true);
///
///     double angle = 45.0 * (M_PI / 180.0);
///
///     impl_bundle_t _obj{ "main" };
///
///     // gotcha wrappers not activated yet
///     printf("cos(%f) = %f\n", angle, cos(angle));
///     printf("sin(%f) = %f\n", angle, sin(angle));
///     printf("exp(%f) = %f\n", angle, exp(angle));
///
///     // gotcha wrappers are reference counted according to start/stop
///     _obj.start();
///
///     printf("cos(%f) = %f\n", angle, cos(angle));
///     printf("sin(%f) = %f\n", angle, sin(angle));
///     printf("exp(%f) = %f\n", angle, exp(angle));
///
///     _obj.stop();
///
///     // gotcha wrappers will be deactivated
///     printf("cos(%f) = %f\n", angle, cos(angle));
///     printf("sin(%f) = %f\n", angle, sin(angle));
///     printf("exp(%f) = %f\n", angle, exp(angle));
///
///     return 0;
/// }
/// \endcode
template <size_t Nt, typename BundleT, typename DiffT>
struct gotcha
: public base<gotcha<Nt, BundleT, DiffT>, void>
, public concepts::external_function_wrapper
{
    static_assert(concepts::has_gotcha<BundleT>::value == false,
                  "Error! {auto,component}_{list,tuple,bundle} in a GOTCHA specification "
                  "cannot include another gotcha_component");

    using value_type   = void;
    using this_type    = gotcha<Nt, BundleT, DiffT>;
    using base_type    = base<this_type, value_type>;
    using storage_type = typename base_type::storage_type;
    using tuple_type   = concepts::tuple_type_t<BundleT>;
    using bundle_type  = concepts::component_type_t<BundleT>;

    friend struct operation::record<this_type>;
    friend struct operation::start<this_type>;
    friend struct operation::stop<this_type>;
    friend struct operation::set_started<this_type>;
    friend struct operation::set_stopped<this_type>;

    template <typename Tp>
    using array_t = std::array<Tp, Nt>;

    using binding_t     = backend::gotcha::binding_t;
    using wrappee_t     = backend::gotcha::wrappee_t;
    using wrappid_t     = backend::gotcha::string_t;
    using error_t       = backend::gotcha::error_t;
    using constructor_t = std::function<void()>;
    using destructor_t  = std::function<void()>;
    using atomic_bool_t = std::atomic<bool>;

    using select_list_t = std::set<std::string>;

    using config_t          = void;
    using get_initializer_t = std::function<config_t()>;
    using get_select_list_t = std::function<select_list_t()>;

    static constexpr size_t components_size = mpl::get_tuple_size<tuple_type>::value;
    static constexpr bool replaces = backend::gotcha::replaces<DiffT, tuple_type>::value;
    using operator_type = typename std::conditional<replaces, DiffT, void>::type;

    static constexpr size_t capacity() { return Nt; }

    static std::string label();
    static std::string description();
    static value_type  record() {}

    static get_initializer_t& get_initializer();

    /// when a permit list is provided, only these functions are wrapped by GOTCHA
    static get_select_list_t& get_permit_list();

    /// reject listed functions are never wrapped by GOTCHA
    static get_select_list_t& get_reject_list();

    static bool& get_default_ready();

    /// add function names at runtime to suppress wrappers
    static void add_global_suppression(const std::string& func);

    /// get an array of whether the wrappers are filled and ready
    static auto get_ready();

    /// set filled wrappers to array of ready values
    static auto set_ready(bool val);

    /// set filled wrappers to array of ready values
    static auto set_ready(const std::array<bool, Nt>& values);

    /// generates the gotcha bindings
    template <size_t N, typename Ret, typename... Args>
    static bool construct(const std::string& _func, int _priority,
                          const std::string& _tool);

    /// invokes construct
    template <size_t N, typename Ret, typename... Args>
    static bool configure(const gotcha_config<N, Ret, Args...>&);

    template <size_t N, typename Ret, typename... Args>
    static bool configure(const std::string& _func, int _priority = 0,
                          const std::string& _tool = {});

    template <size_t N, typename Ret, typename... Args>
    static bool configure(const std::vector<std::string>& _funcs, int _priority = 0,
                          const std::string& _tool = {});

    template <size_t N>
    static bool revert();

    static bool& is_configured() { return get_persistent_data().m_is_configured; }
    static locking::spin_mutex& get_mutex() { return get_persistent_data().m_mutex; }
    static auto                 get_info();

    static void configure();
    static void enable() { configure(); }
    static void disable();

    static void global_finalize();
    static void thread_init();

    static gotcha_data* at(size_t);

public:
    TIMEMORY_DEFAULT_OBJECT(gotcha)

    void start();
    void stop();

public:
    template <size_t N, typename Ret, typename... Args>
    struct instrument
    {
        static void generate(const std::string& _func, const std::string& _tool = "",
                             int _priority = 0);
    };

    //----------------------------------------------------------------------------------//

#if !defined(TIMEMORY_NVCC_COMPILER)
    template <size_t N, typename Ret, typename... Args>
    struct instrument<N, Ret, type_list<Args...>> : instrument<N, Ret, Args...>
    {};

    template <size_t N, typename Ret, typename... Args>
    struct instrument<N, Ret, std::tuple<Args...>> : instrument<N, Ret, Args...>
    {};
#endif

    template <size_t N, typename Ret, typename... Args>
    static void gotcha_factory(const std::string& _func, const std::string& _tool = "",
                               int _priority = 0);

private:
    //----------------------------------------------------------------------------------//
    //
    struct persistent_data
    {
        persistent_data()
        {
            size_t _idx = 0;
            for(auto& itr : m_data)
            {
                itr.ready = get_default_ready();
                itr.index = _idx++;
            }
        }

        ~persistent_data() = default;
        TIMEMORY_DELETE_COPY_MOVE_OBJECT(persistent_data)

        bool                  m_is_configured = false;
        int                   m_verbose       = -1;
        std::atomic<int64_t>  m_started{ 0 };
        array_t<gotcha_data>  m_data;
        locking::spin_mutex   m_mutex;
        std::set<std::string> m_suppress    = { "malloc", "calloc", "free" };
        get_initializer_t     m_initializer = []() {
            for(const auto& itr : get_data())
                itr.constructor();
        };
        get_select_list_t m_reject_list = []() { return select_list_t{}; };
        get_select_list_t m_permit_list = []() { return select_list_t{}; };
    };

    //----------------------------------------------------------------------------------//
    //
    static persistent_data& get_persistent_data()
    {
        static persistent_data _instance;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// \fn array_t<gotcha_data>& get_data()
    /// \brief Gotcha wrapper data
    static array_t<gotcha_data>& get_data() { return get_persistent_data().m_data; }

    //----------------------------------------------------------------------------------//
    /// \fn std::atomic<int64_t>& get_started()
    /// \brief Global counter of active gotchas started
    static std::atomic<int64_t>& get_started() { return get_persistent_data().m_started; }

    //----------------------------------------------------------------------------------//
    /// \fn int64_t& get_thread_started()
    /// \brief Thread-local counter of activate gotchas
    static int64_t& get_thread_started()
    {
        static thread_local int64_t _instance = 0;
        return _instance;
    }

    //----------------------------------------------------------------------------------//
    /// \fn std::set<std::string>& get_suppresses()
    /// \brief global suppression when being used
    static std::set<std::string>& get_suppresses()
    {
        return get_persistent_data().m_suppress;
    }

    //----------------------------------------------------------------------------------//
    /// \brief bool is_permitted()
    /// Check the permit list and reject list for whether the component is permitted
    /// to be wrapped.
    template <size_t N, typename Ret, typename... Args>
    static bool is_permitted(const std::string& _func);

    //----------------------------------------------------------------------------------//

    template <size_t N>
    static void check_error(error_t _ret, const std::string& _prefix);

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static binding_t construct_binder(const std::string& _func)
    {
        auto& _data = get_data()[N];

        constexpr bool _is_fast =
            trait::gotcha_trait<trait::fast_gotcha, this_type, N>::value;
        if constexpr(_is_fast)
        {
            _data.wrapper = (void*) this_type::fast_func<N, Ret, Args...>;
            return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
        }
        else
        {
            if constexpr(replaces)
            {
                _data.wrapper = (void*) this_type::replace_func<N, Ret, Args...>;
                return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
            }
            else
            {
                _data.wrapper = (void*) this_type::wrap<N, Ret, Args...>;
                return binding_t{ _func.c_str(), _data.wrapper, &_data.wrappee };
            }
        }
    }

    //----------------------------------------------------------------------------------//

    template <typename Comp, typename Ret, typename... Args>
    static Ret invoke(gotcha_data&& _data, Comp& _comp, Ret (*_func)(Args...),
                      Args&&... _args)
    {
        if constexpr(backend::gotcha::replaces<DiffT, tuple_type>::value)
        {
            constexpr bool set_data_v =
                !quirk::has_quirk<quirk::static_data, BundleT>::value;
            auto* _obj = _comp.template get<DiffT>();
            if(_data.is_active && _obj)
            {
                return gotcha_invoker<DiffT, Ret, set_data_v>{}(
                    *_obj, std::forward<gotcha_data>(_data), _func,
                    std::forward<Args>(_args)...);
            }
            else
            {
                return _func(std::forward<Args>(_args)...);
            }
        }
        else if constexpr(!backend::gotcha::replaces<DiffT, tuple_type>::value)
        {
            return _func(std::forward<Args>(_args)...);
        }
        else
        {
            static_assert(std::is_empty<this_type>::value,
                          "Error! invoke did not satisfy any expected conditions");
        }
        consume_parameters(_data, _comp);
    }

    //----------------------------------------------------------------------------------//

    static inline void toggle_suppress_on(bool* _bsuppress, bool& _did)
    {
        if(_bsuppress && *_bsuppress == false)
        {
            *(_bsuppress) = true;
            _did          = true;
        }
    }

    static inline void toggle_suppress_off(bool* _bsuppress, bool& _did)
    {
        if(_bsuppress && _did == true && *_bsuppress == true)
        {
            *(_bsuppress) = false;
            _did          = false;
        }
    }

    //----------------------------------------------------------------------------------//

    template <size_t N, typename Ret, typename... Args>
    static TIMEMORY_NOINLINE Ret wrap(Args... _args);

    template <size_t N, typename Ret, typename... Args>
    static TIMEMORY_NOINLINE Ret replace_func(Args... _args);

    template <size_t N, typename Ret, typename... Args>
    static TIMEMORY_INLINE Ret fast_func(Args... _args);

private:
    template <typename Tp>
    static auto init_storage(int) -> decltype(Tp::init_storage())
    {
        return Tp::init_storage();
    }

    template <typename Tp>
    static auto init_storage(long)
    {}

public:
    static const array_t<gotcha_data>& get_gotcha_data()
    {
        return get_persistent_data().m_data;
    }
};
//
}  // namespace component
}  // namespace tim

#if !defined(TIMEMORY_COMPONENTS_GOTCHA_COMPONENTS_CPP_)
#    include "timemory/components/gotcha/components.cpp"
#endif
