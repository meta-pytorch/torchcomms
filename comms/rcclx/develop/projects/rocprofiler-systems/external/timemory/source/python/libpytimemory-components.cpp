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

#if !defined(TIMEMORY_PYCOMPONENTS_SOURCE)
#    define TIMEMORY_PYCOMPONENTS_SOURCE
#endif

#if !defined(TIMEMORY_USE_EXTERN)
#    define TIMEMORY_USE_EXTERN
#endif

#if !defined(TIMEMORY_USE_COMPONENT_EXTERN)
#    define TIMEMORY_USE_COMPONENT_EXTERN
#endif

#include "libpytimemory-components.hpp"

#include "timemory/components/extern.hpp"
#include "timemory/data/ring_buffer_allocator.hpp"
#include "timemory/enum.h"
#include "timemory/operations/types/python_class_name.hpp"
#include "timemory/timemory.hpp"

#include <set>
#include <string>

namespace pycomponents
{
/*template <typename Tp>
auto&
get_allocator()
{
    static tim::data::ring_buffer_allocator<Tp> _v{};
    return _v;
}
//
template <typename Tp, typename... Args>
Tp*
create(Args&&... args)
{
    auto& alloc = get_allocator<Tp>();
    Tp*   ptr   = alloc.allocate(1);
    alloc.construct(ptr, std::forward<Args>(args)...);
    return ptr;
}
//
template <typename Tp>
void
destroy(Tp* ptr)
{
    auto& alloc = get_allocator<Tp>();
    alloc.destroy(ptr);
    alloc.deallocate(ptr, 1);
}*/
//
template <typename Tp, typename... Args>
Tp*
create(Args&&... args)
{
    return new Tp{ std::forward<Args>(args)... };
}
}  // namespace pycomponents

namespace pyinternal
{
//
struct cinfo
{
    bool                  available  = false;
    bool                  arithmetic = false;
    std::string           class_name = {};
    std::string           id         = {};
    std::set<std::string> ids        = {};

    template <typename Tp>
    auto find(Tp&& _v) const
    {
        return ids.find(std::forward<Tp>(_v));
    }
};
//
//--------------------------------------------------------------------------------------//
/// variadic wrapper around each component allowing to to accept arguments that it
/// doesn't actually accept and implement functions it does not actually implement
template <typename T>
using pytuple_t = tim::lightweight_tuple<T>;
/// a python object generator function via a string ID
using keygen_t = std::function<py::object()>;
/// component info paired to a generator function
using keyset_t = std::pair<cinfo, keygen_t>;
/// a python object generator function via an enumeration ID
using indexgen_t = std::function<py::object(int)>;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type,
          std::enable_if_t<!std::is_same<V, void>::value, int> = 0>
static inline auto
get(py::class_<pytuple_t<T>>& _pyclass)
    -> decltype(std::get<0>(std::declval<pytuple_t<T>>().get()), void())
{
    using bundle_t = pytuple_t<T>;
    auto _get      = [](bundle_t* obj) { return obj->template get<T>()->get(); };
    auto _get_raw  = [](bundle_t* obj) {
        if(tim::trait::base_has_accum<T>::value)
            return obj->template get<T>()->get_accum();
        else
            return obj->template get<T>()->get_value();
    };

    _pyclass.def_property_readonly_static(
        "has_value", [](py::object) { return true; },
        "Whether the component has an accessible value");
    _pyclass.def("get", _get,
                 R"(
Get the value for the component in the units designated by the component and
the current settings, e.g. if settings.timing_units = "msec" and the
component has the type-trait 'uses_timing_units' set to true, this will return
the time in milliseconds. Use get_raw() to avoid unit-conversion.
)");
    _pyclass.def("get_raw", _get_raw,
                 R"(
Get the value without any unit conversions.
This may be the identical to get_value() or get_accum() depending on the
type-traits of the component.
)");

    auto _get_value = [](bundle_t* obj) { return obj->template get<T>()->get_value(); };
    _pyclass.def("get_value", _get_value,
                 R"(
Get the current value.
Use with care: depending on the design of the component, this may just be an
incomplete/intermediate representation of the raw value, such as the starting
time-stamp or the starting values of the hardware counters, when start() was
called and stop() was not called.
)");

    IF_CONSTEXPR(tim::trait::base_has_accum<T>::value)
    {
        auto _get_accum = [](bundle_t* obj) {
            return obj->template get<T>()->get_accum();
        };
        _pyclass.def("get_accum", _get_accum,
                     R"(
Get the accumulated value of the component.
When this function is available, this is generally safer than calling get_value()
since the accumulated value is typically only updated during the stop() member
function whereas the value returned from get_value() is typically updated
during both start() and stop(), e.g.:

    start() { value = record(); }
    stop()
    {
        value = (record() - value);
        accum += value;
    }
)");
    }

    IF_CONSTEXPR(tim::trait::base_has_last<T>::value)
    {
        auto _get_last = [](bundle_t* obj) { return obj->template get<T>()->get_last(); };
        _pyclass.def("get_last", _get_last,
                     R"(
Get the latest recorded value.
This may or may not differ from get_value() depending on the component and when it is
called. Generally this function is made available when the value returned by get_value()
is used to record an intermediate value between start() and stop() but it is worthwhile to
make the latest updated value available.
)");
    }
}
//
//--------------------------------------------------------------------------------------//
//
static bool print_repr_in_interactive = false;
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type>
static inline void
repr(py::class_<pytuple_t<T>>& _pyclass,
     std::enable_if_t<!std::is_same<V, void>::value, int> = 0)
{
    auto _repr = [](pytuple_t<T>* obj) {
        static auto _main     = py::module::import("__main__");
        static bool _has_main = (_main) ? py::hasattr(_main, "__file__") : false;
        if(!obj || (!_has_main && !print_repr_in_interactive))
            return std::string{};
        std::stringstream ss;
        {
            tim::cereal::MinimalJSONOutputArchive ar(ss);
            ar(tim::cereal::make_nvp(tim::demangle<pytuple_t<T>>(), *obj));
        }
        return ss.str();
    };
    auto _str = [](pytuple_t<T>* obj) {
        if(!obj)
            return std::string{};
        std::stringstream ss;
        if(obj->hash() == 0)
            obj->template print<false, true>(ss, false);
        else
            obj->template print<true, true>(ss, false);
        return ss.str();
    };
    _pyclass.def("__repr__", _repr, "String representation");
    _pyclass.def("__str__", _str, "String representation");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type>
static inline void
repr(py::class_<pytuple_t<T>>&, std::enable_if_t<std::is_same<V, void>::value, int> = 0)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename V = typename T::value_type,
          std::enable_if_t<std::is_same<V, void>::value, int> = 0>
static inline void
get(py::class_<pytuple_t<T>>& _pyclass)
{
    using bundle_t = pytuple_t<T>;
    auto _none     = [](bundle_t*) { return py::none{}; };
    _pyclass.def("get", _none, "Component does not return value");
    _pyclass.def_property_readonly_static(
        "has_value", [](py::object) { return false; },
        "Whether the component has an accessible value");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
record(py::class_<pytuple_t<T>>& _pyclass, int, int)
    -> decltype(T::record(std::declval<Args>()...), void())
{
    auto _record = [](Args... _args) { return T::record(_args...); };
    _pyclass.def_static("record", _record, "Get the record of a measurement");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
record(py::class_<pytuple_t<T>>& _pyclass, int, long) -> decltype(
    std::declval<pytuple_t<T>>().template get<T>()->record(std::declval<Args>()...),
    void())
{
    using bundle_t = pytuple_t<T>;
    auto _record   = [](bundle_t* obj, Args... args) {
        return obj->template get<T>()->record(args...);
    };
    _pyclass.def("record", _record, "Get the record of a measurement");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline void
record(py::class_<pytuple_t<T>>&, long, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
sample(py::class_<pytuple_t<T>>& _pyclass, int, int)
    -> decltype(T::sample(std::declval<Args>()...), void())
{
    auto _sample = [](Args... _args) { return T::sample(_args...); };
    _pyclass.def_static("sample", _sample, "Get the sample of a measurement");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
sample(py::class_<pytuple_t<T>>& _pyclass, int, long) -> decltype(
    std::declval<pytuple_t<T>>().template get<T>()->sample(std::declval<Args>()...),
    void())
{
    using bundle_t = pytuple_t<T>;
    auto _sample   = [](bundle_t* obj, Args... args) {
        return obj->template get<T>()->sample(args...);
    };
    _pyclass.def("sample", _sample, "Get the sample of a measurement");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline void
sample(py::class_<pytuple_t<T>>&, long, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
static inline auto
get_unit(py::class_<pytuple_t<T>>& _pyclass, int, int) -> decltype(T::get_unit(), void())
{
    auto _get_unit = []() { return T::get_unit(); };
    _pyclass.def_static("unit", _get_unit, "Get the units for the type");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
static inline auto
get_unit(py::class_<pytuple_t<T>>& _pyclass, int, long)
    -> decltype(std::declval<pytuple_t<T>>().template get<T>()->get_unit(), void())
{
    using bundle_t = pytuple_t<T>;
    auto _get_unit = [](bundle_t* obj) { return obj->template get<T>()->get_unit(); };
    _pyclass.def("unit", _get_unit, "Get the units of the object");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline void
get_unit(py::class_<pytuple_t<T>>& _pyclass, long, long)
{
    _pyclass.def_static(
        "unit", []() { return 1; }, "Get the units of the object");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
get_display_unit(py::class_<pytuple_t<T>>& _pyclass, int, int)
    -> decltype(T::get_display_unit(), void())
{
    auto _get_display_unit = []() { return T::get_display_unit(); };
    _pyclass.def_static("display_unit", _get_display_unit,
                        "Get the display units of the type");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
get_display_unit(py::class_<pytuple_t<T>>& _pyclass, int, long)
    -> decltype(std::declval<pytuple_t<T>>().template get<T>()->get_display_unit(),
                void())
{
    using bundle_t         = pytuple_t<T>;
    auto _get_display_unit = [](bundle_t* obj) {
        return obj->template get<T>()->get_display_unit();
    };
    _pyclass.def("display_unit", _get_display_unit,
                 "Get the display units of the object");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline void
get_display_unit(py::class_<pytuple_t<T>>& _pyclass, long, long)
{
    _pyclass.def_static(
        "display_unit", []() { return ""; }, "Get the display units of the object");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
configure(py::class_<pytuple_t<T>>& _pyclass, int, int, Args&&... args)
    -> decltype(T::configure(tim::project::python{}, std::forward<Args>(args)...), void())
{
    auto _configure = [](Args&&... _args) {
        T::configure(tim::project::python{}, std::forward<Args>(_args)...);
    };

    std::stringstream ss;
    ss << "Configure " << tim::demangle<T>() << " globally. Args: ("
       << TIMEMORY_JOIN(", ", tim::demangle<Args>()...) << ")";
    _pyclass.def_static("configure", _configure, ss.str().c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
configure(py::class_<pytuple_t<T>>& _pyclass, int, long, Args&&... args)
    -> decltype(std::declval<pytuple_t<T>>().template get<T>()->configure(
                    tim::project::python{}, std::forward<Args>(args)...),
                void())
{
    using bundle_t  = pytuple_t<T>;
    auto _configure = [](bundle_t* obj, Args&&... _args) {
        obj->template get<T>()->configure(tim::project::python{},
                                          std::forward<Args>(_args)...);
    };

    std::stringstream ss;
    ss << "Configure " << tim::demangle<T>() << " instance. Args: ("
       << TIMEMORY_JOIN(", ", tim::demangle<Args>()...) << ")";
    _pyclass.def("configure", _configure, ss.str().c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T, typename... Args>
static inline auto
configure(py::class_<pytuple_t<T>>&, long, long, Args&&...)
{}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
static inline auto
configure(py::class_<pytuple_t<T>>& _pyclass, int)
    -> decltype(T::configure(tim::project::python{}, _pyclass))
{
    T::configure(tim::project::python{}, _pyclass);
}
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
static inline auto
configure(py::class_<pytuple_t<T>>&, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <int OpT, typename... Args>
struct process_args;
//
template <int OpT>
struct process_args<OpT>
{
    template <typename U>
    static void generate(py::class_<U>&)
    {}
};
//
template <int OpT, typename Arg, typename... Args>
struct process_args<OpT, Arg, Args...>
{
    template <typename U, int Op = OpT,
              std::enable_if_t<Op == TIMEMORY_CONSTRUCT, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _init = [](Arg arg, Args... args) {
            U* obj = pycomponents::create<U>(arg, args...);
            obj->construct(arg, args...);
            return obj;
        };
        _pycomp.def(py::init(_init), "Construct");
    }

    template <typename U, int Op = OpT, std::enable_if_t<Op == TIMEMORY_GET, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) { return obj->get(arg, args...); };
        _pycomp.def("get", _func, "Get some value from the component");
    }

    template <typename U, int Op = OpT, std::enable_if_t<Op == TIMEMORY_AUDIT, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return &obj->audit(arg, args...);
        };
        _pycomp.def("audit", _func, "Audit incoming or outgoing values");
    }

    template <typename U, int Op = OpT, std::enable_if_t<Op == TIMEMORY_START, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return &obj->start(arg, args...);
        };
        _pycomp.def("start", _func, "Start measurement");
    }

    template <typename U, int Op = OpT, std::enable_if_t<Op == TIMEMORY_STOP, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return &obj->stop(arg, args...);
        };
        _pycomp.def("stop", _func, "Stop measurement");
    }

    template <typename U, int Op = OpT, std::enable_if_t<Op == TIMEMORY_STORE, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return &obj->store(arg, args...);
        };
        _pycomp.def("store", _func, "Store measurement");
    }

    template <typename U, int Op = OpT, std::enable_if_t<Op == TIMEMORY_RECORD, int> = 0>
    static void generate(py::class_<pytuple_t<U>>& _pycomp)
    {
        pyinternal::record<U, Arg, Args...>(_pycomp, 0, 0);
    }

    template <typename U, int Op = OpT, std::enable_if_t<Op == TIMEMORY_SAMPLE, int> = 0>
    static void generate(py::class_<pytuple_t<U>>& _pycomp)
    {
        pyinternal::sample<U, Arg, Args...>(_pycomp, 0, 0);
    }

    template <typename U, int Op = OpT, std::enable_if_t<Op == TIMEMORY_MEASURE, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return &obj->measure(arg, args...);
        };
        _pycomp.def("measure", _func, "Take a measurement");
    }

    template <typename U, int Op = OpT,
              std::enable_if_t<Op == TIMEMORY_MARK_BEGIN, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return &obj->mark_begin(arg, args...);
        };
        _pycomp.def("mark_begin", _func, "Mark a begin point");
    }

    template <typename U, int Op = OpT,
              std::enable_if_t<Op == TIMEMORY_MARK_END, int> = 0>
    static void generate(py::class_<U>& _pycomp)
    {
        auto _func = [](U* obj, Arg arg, Args... args) {
            return &obj->mark_end(arg, args...);
        };
        _pycomp.def("mark_end", _func, "Mark an end point");
    }
};
//
template <int OpT, typename... Args, typename... Tail>
struct process_args<OpT, tim::type_list<Args...>, Tail...>
{
    template <typename U>
    static void generate(py::class_<U>& _pycomp)
    {
        process_args<OpT, Args...>::generate(_pycomp);
        process_args<OpT, Tail...>::generate(_pycomp);
    }
};
//
//--------------------------------------------------------------------------------------//
//
template <template <typename...> class TupleT, typename T, size_t... Idx>
static void
operations(py::class_<TupleT<T>>& _pyclass, std::index_sequence<Idx...>)
{
    using tim::trait::python_args_t;
    TIMEMORY_FOLD_EXPRESSION(
        pyinternal::process_args<Idx, python_args_t<Idx, T>>::generate(_pyclass));
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, typename T>
static void
generate_properties(py::class_<pytuple_t<T>>& /*_pycomp*/)
{
    /*
    //----------------------------------------------------------------------------------//
    //
    //      Properties
    //
    //----------------------------------------------------------------------------------//
    using property_t = tim::component::properties<T>;

    py::class_<property_t> _pyprop(_pycomp, "Properties", "Static properties class");


    static auto        _id_idx = static_cast<TIMEMORY_NATIVE_COMPONENT>(Idx);
    static std::string _id_str = property_t::id();
    static std::string _enum_str = property_t::enum_string();
    static auto        _ids_set  = []() {
        py::list _ret{};
        for(const auto& itr : property_t::ids())
            _ret.append(itr);
        return _ret;
    }();
    auto _match_int = [](TIMEMORY_NATIVE_COMPONENT eid) {
        return property_t::matches(static_cast<int>(eid));
    };
    auto _match_str = [](const std::string& str) { return property_t::matches(str); };

    _pyprop.def_property_readonly_static(
        "enum_string", [](py::object) { return _enum_str; },
        "Get the string version of the enumeration ID");

    _pyprop.def_property_readonly_static(
        "enum_value", [](py::object) { return _id_idx; },
        "Get the enumeration ID for the component");

    _pyprop.def_property_readonly_static(
        "id", [](py::object) { return _id_str; },
        "Get the primary string ID for the component");

    _pyprop.def_property_readonly_static(
        "ids", [](py::object) { return _ids_set; },
        "Get the secondary string IDs for the component");

    _pyprop.def_static(
        "matches", _match_int,
        "Returns whether the provided enum is a matching identifier for the type");
    _pyprop.def_static(
        "matches", _match_str,
        "Returns whether the provided string is a matching identifier for the type");

    auto _as_json = []() {
        using archive_t   = tim::cereal::MinimalJSONOutputArchive;
        using api_t       = tim::project::python;
        using policy_type = tim::policy::output_archive<archive_t, api_t>;
        std::stringstream ss;
        property_t        prop{};
        {
            auto oa = policy_type::get(ss);
            (*oa)(tim::cereal::make_nvp("properties", prop));
        }
        auto json_module = py::module::import("json");
        return json_module.attr("loads")(ss.str());
    };

    _pyprop.def_static("as_json", _as_json, "Get the properties as a JSON dictionary");
    */
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, size_t N>
static void
generate(py::module& _pymod, std::array<bool, N>& _boolgen,
         std::array<keyset_t, N>& _keygen,
         std::enable_if_t<
             tim::component::enumerator<Idx>::value &&
                 !tim::concepts::is_placeholder<tim::component::enumerator_t<Idx>>::value,
             int> = 0)
{
    using T = typename tim::component::enumerator<Idx>::type;
    if(tim::concepts::is_placeholder<T>::value)
        return;
    using property_t = tim::component::properties<T>;
    using metadata_t = tim::component::metadata<T>;
    using bundle_t   = pytuple_t<T>;
    using value_type = tim::trait::component_value_type_t<T>;

    static_assert(property_t::specialized(), "Error! Missing specialization");

    std::string id           = tim::operation::python_class_name<T>{}();
    std::string cid          = property_t::id();
    bool        is_arith     = std::is_arithmetic<value_type>::value;
    bool        is_func_wrap = tim::concepts::is_external_function_wrapper<T>::value;

    auto _init  = []() { return pycomponents::create<bundle_t>(); };
    auto _sinit = [](std::string key) { return pycomponents::create<bundle_t>(key); };
    auto _hash  = [](bundle_t* obj) { return obj->hash(); };
    auto _key   = [](bundle_t* obj) { return obj->key(); };
    auto _laps  = [](bundle_t* obj) { return obj->laps(); };
    auto _rekey = [](bundle_t* obj, std::string _key) { obj->rekey(_key); };

    auto _isub = [](bundle_t* lhs, bundle_t* rhs) {
        if(lhs && rhs)
            *lhs -= *rhs;
        return lhs;
    };

    py::class_<bundle_t> _pycomp(_pymod, id.c_str(), T::description().c_str());

    // these have direct mappings to the component
    _pycomp.def(py::init(_init), "Creates component");
    _pycomp.def(py::init(_sinit), "Creates component with a label");
    _pycomp.def(
        "push", [](bundle_t* obj) { return &obj->push(); }, "Push into the call-graph");
    _pycomp.def(
        "pop", [](bundle_t* obj) { return &obj->pop(); }, "Pop off the call-graph");
    _pycomp.def(
        "start", [](bundle_t* obj) { return &obj->start(); }, "Start measurement");
    _pycomp.def(
        "stop", [](bundle_t* obj) { return &obj->stop(); }, "Stop measurement");
    _pycomp.def(
        "measure", [](bundle_t* obj) { return &obj->measure(); }, "Take a measurement");
    _pycomp.def(
        "reset", [](bundle_t* obj) { return &obj->reset(); }, "Reset the values");
    _pycomp.def(
        "mark_begin", [](bundle_t* obj) { return &obj->mark_begin(); },
        "Mark an begin point");
    _pycomp.def(
        "mark_end", [](bundle_t* obj) { return &obj->mark_end(); }, "Mark an end point");

    // these require further evaluation
    pyinternal::get(_pycomp);
    pyinternal::repr(_pycomp);
    pyinternal::record(_pycomp, 0, 0);
    pyinternal::sample(_pycomp, 0, 0);
    pyinternal::get_unit(_pycomp, 0, 0);
    pyinternal::get_display_unit(_pycomp, 0, 0);
    pyinternal::configure(_pycomp, 0);
    pyinternal::configure(_pycomp, 0, 0, py::args{}, py::kwargs{});
    pyinternal::operations(_pycomp, std::make_index_sequence<TIMEMORY_OPERATION_END>{});

    // these are operations on the bundler
    _pycomp.def("hash", _hash, "Get the current hash");
    _pycomp.def("key", _key, "Get the identifier");
    _pycomp.def("rekey", _rekey, "Change the identifier");
    _pycomp.def("laps", _laps, "Get the number of laps");

    // operators
    _pycomp.def(py::self + py::self);
    _pycomp.def(py::self - py::self);
    _pycomp.def(py::self += py::self);
    _pycomp.def("__isub__", _isub, "Subtract rhs from lhs", py::is_operator());

    auto _label = []() {
        if(metadata_t::specialized())
            return metadata_t::label();
        return T::label();
    };

    auto _desc = []() {
        if(metadata_t::specialized())
            return TIMEMORY_JOIN("", metadata_t::description(), ". ",
                                 metadata_t::extra_description());
        return T::description();
    };

    namespace operation = tim::operation;

    auto _global_init = []() {
        operation::init<T>{}(operation::mode_constant<operation::init_mode::global>{});
    };

    auto _thread_init = []() {
        operation::init<T>{}(operation::mode_constant<operation::init_mode::thread>{});
    };

    auto _global_fini = []() {
        operation::fini<T>{}(operation::mode_constant<operation::fini_mode::global>{});
    };

    auto _thread_fini = []() {
        operation::fini<T>{}(operation::mode_constant<operation::fini_mode::thread>{});
    };

    auto _true = [](py::object) { return true; };

    _pycomp.def_static("label", _label, "Get the label for the type");
    _pycomp.def_static("description", _desc, "Get the description for the type");
    _pycomp.def_property_readonly_static("available", _true,
                                         "Whether the component is available");
    _pycomp.def_static("global_init", _global_init,
                       "Execute global initialization routine");
    _pycomp.def_static("thread_init", _thread_init,
                       "Execute thread initialization routine");
    _pycomp.def_static("global_finalize", _global_fini,
                       "Execute global finalization routine");
    _pycomp.def_static("thread_finalize", _thread_fini,
                       "Execute thread finalization routine");

    std::set<std::string> _keys = property_t::ids();
    _keys.insert(id);
    _boolgen[Idx] = true;
    _keygen[Idx]  = { cinfo{ true, is_arith && !is_func_wrap, id, cid, _keys },
                     []() { return py::cast(pycomponents::create<bundle_t>()); } };

    auto idx = static_cast<TIMEMORY_NATIVE_COMPONENT>(Idx);
    _pycomp.def_static(
        "index", [idx]() { return idx; }, "Enumeration ID for the component");

    _pycomp.def_static(
        "id", [cid]() { return cid; }, "(Primary) String ID for the component");

    // generate_properties<Idx, T>(_pycomp);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Idx, size_t N>
static void
generate(py::module& _pymod, std::array<bool, N>& _boolgen,
         std::array<keyset_t, N>& _keygen,
         std::enable_if_t<
             !tim::component::enumerator<Idx>::value &&
                 !tim::concepts::is_placeholder<tim::component::enumerator_t<Idx>>::value,
             long> = 0)
{
    using T = typename tim::component::enumerator<Idx>::type;
    if(tim::concepts::is_placeholder<T>::value)
        return;
    using property_t  = tim::component::properties<T>;
    using bundle_t    = pytuple_t<T>;
    std::string id    = tim::operation::python_class_name<T>{}();
    std::string cid   = property_t::id();
    std::string _desc = "not available";

    auto _init  = []() { return new bundle_t{}; };
    auto _sinit = [](std::string) { return new bundle_t{}; };
    auto _noop  = [](bundle_t*) {};
    auto _none  = [](bundle_t*) { return py::none{}; };
    auto _rekey = [](bundle_t*, std::string) {};
    auto _repr  = [](bundle_t*) { return std::string(""); };

    py::class_<bundle_t> _pycomp(_pymod, id.c_str(), _desc.c_str());

    _pycomp.def(py::init(_init), "Creates component");
    _pycomp.def(py::init(_sinit), "Creates component with a label");
    _pycomp.def("push", _noop, "Push into the call-graph");
    _pycomp.def("pop", _noop, "Pop off the call-graph");
    _pycomp.def("start", _noop, "Start measurement");
    _pycomp.def("stop", _noop, "Stop measurement");
    _pycomp.def("measure", _noop, "Take a measurement");
    _pycomp.def("reset", _noop, "Reset the values");
    _pycomp.def("mark_begin", _noop, "Mark an begin point");
    _pycomp.def("mark_end", _noop, "Mark an end point");
    _pycomp.def("get", _none, "No value available");
    _pycomp.def("value", _none, "No value available");

    // these are operations on the bundler
    _pycomp.def("hash", _none, "Get the current hash");
    _pycomp.def("key", _none, "Get the identifier");
    _pycomp.def("rekey", _rekey, "Change the identifier");
    _pycomp.def("laps", _none, "Get the number of laps");

    // operators
    _pycomp.def("__add__", _none, "Get addition of two components", py::is_operator());
    _pycomp.def("__sub__", _none, "Get difference between two components",
                py::is_operator());
    _pycomp.def("__iadd__", _none, "Add rhs to lhs", py::is_operator());
    _pycomp.def("__isub__", _none, "Subtract rhs from lhs", py::is_operator());
    _pycomp.def("__repr__", _repr, "String representation");

    auto _false = [](py::object) { return false; };

    _pycomp.def_static(
        "unit", []() { return 1; }, "Get the units for the type");
    _pycomp.def_static(
        "display_unit", []() { return ""; }, "Get the unit repr for the type");
    _pycomp.def_property_readonly_static("available", _false,
                                         "Whether the component is available");
    _pycomp.def_property_readonly_static("has_value", _false,
                                         "Whether the component has an accessible value");

    std::set<std::string> _keys = property_t::ids();
    _keys.insert(id);
    _boolgen[Idx] = false;
    _keygen[Idx] = { cinfo{ false, false, id, cid, _keys }, []() { return py::none{}; } };

    auto idx = static_cast<TIMEMORY_NATIVE_COMPONENT>(Idx);
    _pycomp.def_static(
        "index", [idx]() { return idx; }, "Enumeration ID for the component");

    _pycomp.def_static(
        "id", [cid]() { return cid; }, "(Primary) String ID for the component");

    // generate_properties<Idx, T>(_pycomp);
}
//
//--------------------------------------------------------------------------------------//
//
template <
    size_t Idx, size_t N,
    std::enable_if_t<
        tim::concepts::is_placeholder<tim::component::enumerator_t<Idx>>::value, int> = 0>
static void
generate(py::module&, std::array<bool, N>&, std::array<keyset_t, N>&)
{}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx, size_t N = sizeof...(Idx)>
static void
components(py::module& _pymod, std::array<bool, N>& _boolgen,
           std::array<keyset_t, N>& _keygen, std::index_sequence<Idx...>)
{
    TIMEMORY_FOLD_EXPRESSION(pyinternal::generate<Idx>(_pymod, _boolgen, _keygen));
}
//
}  // namespace pyinternal
//
//======================================================================================//
//
namespace pycomponents
{
//
py::module
generate(py::module& _pymod)
{
    py::module _pycomp = _pymod.def_submodule(
        "component",
        "Stand-alone classes for the components. Unless push() and pop() are called on "
        "these objects, they will not store any data in the timemory call-graph (if "
        "applicable)");

    _pycomp.attr("_print_repr_in_interactive") = pyinternal::print_repr_in_interactive;

    constexpr size_t                    N = TIMEMORY_COMPONENTS_END;
    std::array<bool, N>                 _boolgen;
    std::array<pyinternal::keyset_t, N> _keygen;
    _boolgen.fill(false);
    _keygen.fill(pyinternal::keyset_t{ {}, []() -> py::function { return py::none{}; } });

    pyinternal::components(_pycomp, _boolgen, _keygen,
                           std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});

    std::vector<std::pair<bool, std::string>> _available_and_names{};
    std::vector<std::string>                  _names{};
    std::vector<std::string>                  _available_names{};
    std::vector<std::string>                  _available_arith_names{};

    _available_and_names.reserve(N);
    _names.reserve(N);
    _available_names.reserve(N);
    _available_arith_names.reserve(N);

    for(const auto& itr : _keygen)
    {
        _available_and_names.emplace_back(itr.first.available, itr.first.id);
        if(itr.first.id.empty())
            continue;
        if(itr.first.arithmetic)
            _available_arith_names.emplace_back(itr.first.id);
        _names.emplace_back(itr.first.id);
        if(itr.first.available)
            _available_names.emplace_back(itr.first.id);
    }

    _available_and_names.shrink_to_fit();
    _available_names.shrink_to_fit();
    _names.shrink_to_fit();

    auto _is_available = [_available_and_names](py::object _obj) {
        auto _enum_val = pytim::get_enum(_obj);
        if(_enum_val >= TIMEMORY_COMPONENTS_END)
            return false;
        return _available_and_names.at(static_cast<size_t>(_enum_val)).first;
    };

    auto _keygenerator = [_keygen, _boolgen](std::string _key) {
        TIMEMORY_DEBUG_PRINT_HERE("pycomponents::get_generator :: looking for %s",
                                  _key.c_str());
        size_t i = 0;
        for(const auto& itr : _keygen)
        {
            if(!_boolgen[i++])
                continue;
            if(itr.first.ids.find(_key) != itr.first.ids.end())
                return itr.second;
        }
        pyinternal::keygen_t _nogen = []() -> py::object { return py::none{}; };
        return _nogen;
    };

    auto _indexgenerator = [_keygen, _boolgen](TIMEMORY_NATIVE_COMPONENT _id) {
        TIMEMORY_DEBUG_PRINT_HERE("pycomponents::get_generator :: looking for %i",
                                  (int) _id);
        size_t i = static_cast<size_t>(_id);
        if(!_boolgen[i])
        {
            pyinternal::keygen_t _nogen = []() -> py::object { return py::none{}; };
            return _nogen;
        }
        return _keygen[i].second;
    };

    _pycomp.def("is_available", _is_available,
                "Query whether a component type is available. Accepts string IDs and "
                "enumerations, e.g. \"wall_clock\" or timemory.component.wall_clock");
    _pycomp.def("get_generator", _keygenerator,
                "Get a functor for generating the component whose class name or string "
                "IDs (see `timemory-avail -s`) match the given key");
    _pycomp.def("get_generator", _indexgenerator,
                "Get a functor for generating the component whose enumeration ID (see "
                "`help(timemory.component.id)`) match the given enumeration ID");
    _pycomp.def(
        "get_types", [_names]() { return _names; },
        "Get the names of all the component types regardless of whether "
        "available or not");
    _pycomp.def(
        "get_available_types",
        [_available_names, _available_arith_names](bool arith_only) {
            return (arith_only) ? _available_arith_names : _available_names;
        },
        "Get the names of all the component types which are available. If "
        "arithmetic=True, will only return components whose value type is a basic "
        "arithmetic type",
        py::arg("arithmetic") = false);

    return _pycomp;
}
}  // namespace pycomponents
//
//======================================================================================//
