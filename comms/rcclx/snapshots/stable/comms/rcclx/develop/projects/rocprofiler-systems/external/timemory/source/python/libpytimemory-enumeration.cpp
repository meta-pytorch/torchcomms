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

#if !defined(TIMEMORY_PYENUMERATION_SOURCE)
#    define TIMEMORY_PYENUMERATION_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/components/definition.hpp"
#include "timemory/runtime/properties.hpp"

//======================================================================================//
//
namespace pyenumeration
{
template <size_t Idx>
static void
generate(py::enum_<TIMEMORY_NATIVE_COMPONENT>& _pyenum)
{
    using T                       = typename tim::component::enumerator<Idx>::type;
    using property_t              = tim::component::properties<T>;
    constexpr bool is_placeholder = tim::concepts::is_placeholder<T>::value;

    if(is_placeholder)
        return;

    // ensure specialized if not placeholder
    static_assert(is_placeholder || property_t::specialized(),
                  "Error! Missing specialization for non-placeholder type");

    std::string id = property_t::enum_string();
    for(auto& itr : id)
        itr = tolower(itr);
    _pyenum.value(id.c_str(), static_cast<TIMEMORY_NATIVE_COMPONENT>(property_t{}()),
                  T::description().c_str());
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t... Idx>
static void
components(py::enum_<TIMEMORY_NATIVE_COMPONENT>& _pyenum, std::index_sequence<Idx...>)
{
    TIMEMORY_FOLD_EXPRESSION(pyenumeration::generate<Idx>(_pyenum));
}
//
//--------------------------------------------------------------------------------------//
//
py::enum_<TIMEMORY_NATIVE_COMPONENT>
generate(py::module& _pymod)
{
    py::enum_<TIMEMORY_NATIVE_COMPONENT> _pyenum(
        _pymod, "id", py::arithmetic(), "Component enumerations for timemory module");
    pyenumeration::components(_pyenum,
                              std::make_index_sequence<TIMEMORY_COMPONENTS_END>{});
    _pyenum.export_values();
    return _pyenum;
}
}  // namespace pyenumeration
//
//======================================================================================//
