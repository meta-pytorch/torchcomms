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

#if !defined(TIMEMORY_PYCOMPONENT_LIST_SOURCE)
#    define TIMEMORY_PYCOMPONENT_LIST_SOURCE
#endif

#include "libpytimemory-components.hpp"
#include "timemory/components.hpp"
#include "timemory/components/extern.hpp"
#include "timemory/runtime/initialize.hpp"
#include "timemory/runtime/properties.hpp"
#include "timemory/variadic/definition.hpp"

using namespace tim::component;

using auto_list_t        = tim::available_auto_list_t;
using component_list_t   = typename auto_list_t::component_type;
using component_enum_vec = std::vector<TIMEMORY_COMPONENT>;

//======================================================================================//
//
namespace pycomponent_list
{
//
//--------------------------------------------------------------------------------------//
//
component_enum_vec
components_list_to_vec(py::list pystr_list)
{
    std::vector<std::string> str_list{};
    str_list.reserve(pystr_list.size());
    for(auto itr : pystr_list)
        str_list.emplace_back(itr.cast<std::string>());
    return tim::enumerate_components(str_list);
}
//
//--------------------------------------------------------------------------------------//
//
component_enum_vec
components_enum_to_vec(py::list enum_list)
{
    component_enum_vec vec{};
    vec.reserve(enum_list.size());
    for(auto itr : enum_list)
        vec.emplace_back(itr.cast<TIMEMORY_COMPONENT>());
    return vec;
}
//
//--------------------------------------------------------------------------------------//
//
component_list_t*
create_component_list(const std::string& obj_tag, const component_enum_vec& components)
{
    using quirk_config_t =
        tim::quirk::config<tim::quirk::explicit_push, tim::quirk::explicit_pop>;
    return new component_list_t{ obj_tag, quirk_config_t{},
                                 [&components](component_list_t& _cl) {
                                     tim::initialize(_cl, components);
                                 } };
}
//
//--------------------------------------------------------------------------------------//
//
class component_list_decorator
{
public:
    component_list_decorator(component_list_t* _ptr = nullptr)
    : m_ptr(_ptr)
    {
        if(m_ptr)
        {
            m_ptr->push();
            m_ptr->start();
        }
    }

    ~component_list_decorator()
    {
        if(m_ptr)
        {
            m_ptr->stop();
            m_ptr->pop();
        }
        delete m_ptr;
    }

    component_list_decorator& operator=(component_list_t* _ptr)
    {
        if(m_ptr)
        {
            m_ptr->stop();
            m_ptr->pop();
            delete m_ptr;
        }
        m_ptr = _ptr;
        if(m_ptr)
        {
            m_ptr->push();
            m_ptr->start();
        }
        return *this;
    }

private:
    component_list_t* m_ptr = nullptr;
};
//
//--------------------------------------------------------------------------------------//
//
namespace init
{
//
component_list_t*
component_list(py::object farg, py::object sarg)
{
    py::list    _comp{};
    std::string _key{};
    try
    {
        _comp = farg.cast<py::list>();
        _key  = sarg.cast<std::string>();
    } catch(py::cast_error& e)
    {
        std::cerr << e.what() << std::endl;
        try
        {
            _comp = sarg.cast<py::list>();
            _key  = farg.cast<std::string>();
        } catch(py::cast_error& e)
        {
            std::cerr << e.what() << std::endl;
            return nullptr;
        }
    }

    return create_component_list(_key, components_enum_to_vec(_comp));
}
//
component_list_decorator*
component_decorator(py::list components, const std::string& key)
{
    component_list_decorator* _ptr = new component_list_decorator{};
    if(!tim::settings::enabled())
        return _ptr;

    return &(*_ptr = create_component_list(
                 key, components_enum_to_vec(std::move(components))));
}
//
}  // namespace init
//
//--------------------------------------------------------------------------------------//
//
void
generate(py::module& _pymod)
{
    py::class_<component_list_t> comp_list(_pymod, "component_tuple",
                                           "Generic component_tuple");

    py::class_<component_list_decorator> comp_decorator(
        _pymod, "component_decorator", "Component list used in decorators");

    //==================================================================================//
    //
    //                      TIMEMORY COMPONENT_TUPLE
    //
    //==================================================================================//
    comp_list.def(py::init(&init::component_list), "Initialization",
                  py::arg("components") = py::list{}, py::arg("key") = std::string{},
                  py::return_value_policy::automatic);
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "push", [](component_list_t* self) { self->push(); },
        "Push components into storage");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "pop", [](component_list_t* self) { self->pop(); },
        "Finalize the component in storage");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "start", [](component_list_t* self) { self->start(); }, "Start component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "stop", [](component_list_t* self) { self->stop(); }, "Stop component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "report", [](component_list_t* self) { std::cout << *(self) << std::endl; },
        "Report component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "__str__",
        [](component_list_t* self) {
            std::stringstream ss;
            ss << *(self);
            return ss.str();
        },
        "Stringify component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "reset", [](component_list_t* self) { self->reset(); },
        "Reset the component tuple");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "get_raw", [](component_list_t* self) { return self->get(); },
        "Get the component list data");
    //----------------------------------------------------------------------------------//
    comp_list.def(
        "get",
        [](component_list_t* self) {
            return pytim::dict::construct(self->get_labeled());
        },
        "Get the component list data (labeled)");
    //----------------------------------------------------------------------------------//
    comp_decorator.def(py::init(&init::component_decorator), "Initialization",
                       py::return_value_policy::automatic);
}
}  // namespace pycomponent_list
//
//======================================================================================//
