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

#if !defined(TIMEMORY_PYSTATISTICS_SOURCE)
#    define TIMEMORY_PYSTATISTICS_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/components/types.hpp"
#include "timemory/data/statistics.hpp"
#include "timemory/utility/demangle.hpp"

namespace pystatistics
{
//
template <typename Tp>
struct valid_statistics_type
{
    using invalid_types = tim::type_list<tim::true_type, tim::false_type, tim::null_type,
                                         void, bool, std::tuple<>, tim::type_list<>>;
    static constexpr bool value =
        tim::trait::is_available<Tp>::value && !tim::is_one_of<Tp, invalid_types>::value;
};
//
template <size_t Idx>
struct statistics_type
{
    using type = typename tim::trait::statistics<tim::component::enumerator_t<Idx>>::type;
};
//
template <size_t Idx>
using statistics_type_t = typename statistics_type<Idx>::type;
//
static inline std::string
get_class_name(std::string id)
{
    auto pos = std::string::npos;
    for(auto&& itr : { " ", ":", ",", "<", ">" })
        while((pos = id.find(itr)) != std::string::npos)
            id = id.replace(pos, 1, "_");

    while((pos = id.find("__")) != std::string::npos)
        id = id.replace(pos, 2, "_");

    while((pos = id.find("std_1_")) != std::string::npos)
        id = id.replace(pos, 6, "std_");

    while(id.back() == '_')
        id = id.substr(0, id.length() - 1);

    return id;
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
auto
construct(py::module& _pymod, int, tim::enable_if_t<valid_statistics_type<Tp>::value> = 0)
    -> decltype(tim::demangle<Tp>(), void())
{
    using value_type        = Tp;
    using statistics_type   = tim::statistics<Tp>;
    using pystatistics_type = py::class_<statistics_type>;

    static std::set<std::string> _created{};
    auto                         _type = tim::demangle<Tp>();
    auto              _name = get_class_name(std::string("Statistics_") + _type);
    std::stringstream _desc;
    _desc << "Statistics class for " << _type << " data type";

    if(_created.count(_name) > 0)
        return;
    _created.insert(_name);

    pystatistics_type _pystat(_pymod, _name.c_str(), _desc.str().c_str());

    auto _init  = []() { return new statistics_type{}; };
    auto _vinit = [](const Tp& _val) { return new statistics_type(_val); };

    _pystat.def(py::init(_init), "Creates statistics type");
    _pystat.def(py::init(_vinit), "Creates statistics type with initial value");
    _pystat.def("reset", &statistics_type::reset, "Reset all values");
    _pystat.def("count", &statistics_type::get_count, "Get the total number of values");
    _pystat.def("min", &statistics_type::get_min, "Get the minimum value");
    _pystat.def("max", &statistics_type::get_max, "Get the maximum value");
    _pystat.def("mean", &statistics_type::get_mean, "Get the mean value");
    _pystat.def("sum", &statistics_type::get_sum, "Get the sum of the values");
    _pystat.def("sqr", &statistics_type::get_sqr,
                "Get the sum of the square of the values");
    _pystat.def("variance", &statistics_type::get_variance,
                "Get the variance of the values");
    _pystat.def("stddev", &statistics_type::get_stddev,
                "Get the standard deviation of the values");

    auto _add = [](statistics_type* lhs, value_type rhs) { return (*lhs += rhs); };
    auto _sub = [](statistics_type* lhs, value_type rhs) { return (*lhs -= rhs); };

    auto _isub = [](statistics_type* lhs, statistics_type* rhs) {
        if(lhs && rhs)
            *lhs -= *rhs;
        return lhs;
    };

    auto _repr = [](statistics_type* obj) {
        std::stringstream ss;
        if(obj)
            ss << *obj;
        return ss.str();
    };

    // operators
    _pystat.def(py::self + py::self);
    _pystat.def(py::self - py::self);
    _pystat.def(py::self += py::self);
    _pystat.def("__isub__", _isub, "Subtract rhs from lhs", py::is_operator());
    _pystat.def("__iadd__", _add, "Add value", py::is_operator());
    _pystat.def("__isub__", _sub, "Subtract value", py::is_operator());
    _pystat.def("__repr__", _repr, "String representation");
}
//
//--------------------------------------------------------------------------------------//
//
template <typename Tp>
void
construct(py::module&, long)
{}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx>
constexpr auto construct(std::index_sequence<Idx...>)
{
    return tim::type_list<statistics_type_t<Idx>...>{};
}
//
//--------------------------------------------------------------------------------------//
//
template <typename... Tp>
auto
construct(py::module& _pymod, tim::type_list<Tp...>)
{
    TIMEMORY_FOLD_EXPRESSION(construct<Tp>(_pymod, 0));
}
//
//--------------------------------------------------------------------------------------//
//
void
generate(py::module& _pymod)
{
    auto _types = construct(std::make_index_sequence<TIMEMORY_NATIVE_COMPONENTS_END>{});
    // std::cerr << "Types: " << tim::demangle<decltype(_types)>() << std::endl;
    construct(_pymod, _types);
    (void) _types;
    (void) _pymod;
}
//
}  // namespace pystatistics
