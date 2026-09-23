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

#include <cctype>
#include <cstddef>
#include <string>

namespace tim
{
//
namespace component
{
template <typename Tp>
struct properties;
}
//
namespace operation
{
//
template <typename Tp>
struct python_class_name;
//
//--------------------------------------------------------------------------------------//
//
///
/// \struct tim::operation::python_class_name
/// \brief This class generates the class name for a component according to the standard
/// Python naming convention
///
//
//--------------------------------------------------------------------------------------//
//
template <>
struct python_class_name<void>
{
    std::string operator()(std::string id) const;
};
//
template <typename Tp>
struct python_class_name : python_class_name<void>
{
    using type      = Tp;
    using base_type = python_class_name<void>;

    using base_type::operator();
    std::string      operator()() const
    {
        using properties_t = component::properties<Tp>;
        static_assert(properties_t::specialized(),
                      "Error! Cannot get python class name if the component properties "
                      "have not been specialized");
        return this->base_type::operator()(properties_t::enum_string());
    }
};
//
//--------------------------------------------------------------------------------------//
//
inline std::string
python_class_name<void>::operator()(std::string id) const
{
    if(id.empty())
        return std::string{};

    for(auto& itr : id)
        itr = ::tolower(itr);

    // capitalize after every delimiter
    for(size_t i = 0; i < id.size(); ++i)
    {
        if(i == 0)
            id.at(i) = ::toupper(id.at(i));
        else
        {
            if((id.at(i) == '_' || id.at(i) == '-') && i + 1 < id.length())
            {
                id.at(i + 1) = ::toupper(id.at(i + 1));
                ++i;
            }
        }
    }
    // remove all delimiters
    for(auto ditr : { '_', '-' })
    {
        size_t _pos = 0;
        while((_pos = id.find(ditr)) != std::string::npos)
            id = id.erase(_pos, 1);
    }

    return id;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace operation
}  // namespace tim
