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

#include "timemory/variadic/auto_base_bundle.hpp"
#include "timemory/variadic/types.hpp"

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace tim
{
//--------------------------------------------------------------------------------------//
/// \class tim::auto_bundle
/// \tparam Tag unique identifying type for the bundle which when \ref
/// tim::trait::is_available<Tag> is false at compile-time or \ref
/// tim::trait::runtime_enabled<Tag>() is false at runtime, then none of the components
/// will be collected
/// \tparam Types Specification of the component types to bundle together
///
/// \brief This is a variadic component wrapper which combines the features of \ref
/// tim::auto_tuple<T...> and \ref tim::auto_list<U..>. The "T" types (compile-time fixed,
/// allocated on stack) should be specified as usual, the "U" types (runtime-time
/// optional, allocated on the heap) should be specified as a pointer. Initialization
/// of the optional types is similar to \ref tim::auto_list<U...> but no environment
/// variable is built-in since, ideally, this environment variable should be customized
/// based on the \ref Tag template parameter.
///
/// \code{.cpp}
///
/// // dummy type identifying the context
/// struct FooApi {};
///
/// using bundle_t = tim::auto_bundle<FooApi, wall_clock, cpu_clock*>;
///
/// void foo_init() // user initialization routine
/// {
///     bundle_t::get_initializer() = [](bundle_t& b)
///     {
///         static auto env_enum = tim::enumerate_components(
///             tim::delimit(tim::get_env<string_t>("FOO_COMPONENTS", "wall_clock")));
///
///         :im::initialize(b, env_enum);
///     };
/// }
/// void bar()
/// {
///     // will record whichever components are specified by "FOO_COMPONENT" in
///     // environment, which "wall_clock" as the default
///
///     auto bar = bundle_t("foo");
///     // ...
/// }
///
/// int main(int argc, char** argv)
/// {
///     tim::timemory_init(argc, argv);
///
///     foo_init();
///
///     bar();
///
///     tim::timemory_finalize();
/// }
/// \endcode
///
/// The above code will record wall-clock, cpu-clock, and peak-rss. The intermediate
/// storage will happen on the stack and when the destructor is called, it will add itself
/// to the call-graph
///
template <typename Tag, typename... Types>
class auto_bundle<Tag, Types...>
: public auto_base_bundle<Tag, component_bundle<Tag>, auto_bundle<Tag, Types...>>
, public concepts::mixed_wrapper
{
    using poly_base =
        auto_base_bundle<Tag, component_bundle<Tag>, auto_bundle<Tag, Types...>>;

public:
    using this_type      = auto_bundle<Tag, Types...>;
    using base_type      = component_bundle<Tag, Types...>;
    using auto_type      = this_type;
    using component_type = typename base_type::component_type;
    using type           = convert_t<typename component_type::type, auto_bundle<Tag>>;

    template <typename... Args>
    explicit auto_bundle(Args&&... args);

    // copy and move
    ~auto_bundle()                  = default;
    auto_bundle(const auto_bundle&) = default;
    auto_bundle(auto_bundle&&)      = default;
    auto_bundle& operator=(const auto_bundle&) = default;
    auto_bundle& operator=(auto_bundle&&) = default;

    static constexpr std::size_t size() { return poly_base::size(); }

public:
    this_type&           print(std::ostream& os, bool _endl = false) const;
    friend std::ostream& operator<<(std::ostream& os, const this_type& obj)
    {
        obj.print(os, false);
        return os;
    }
};
//
template <typename Tag, typename... Types>
template <typename... Args>
auto_bundle<Tag, Types...>::auto_bundle(Args&&... args)
: poly_base{ std::forward<Args>(args)... }
{}
//
template <typename Tag, typename... Types>
auto_bundle<Tag, Types...>&
auto_bundle<Tag, Types...>::print(std::ostream& os, bool _endl) const
{
    os << poly_base::m_temporary;
    if(_endl)
        os << '\n';
    return const_cast<this_type&>(*this);
}
//
//======================================================================================//
//
template <typename Tag, typename... Types>
auto
get(const auto_bundle<Tag, Types...>& _obj)
{
    return get(_obj.get_component());
}

//--------------------------------------------------------------------------------------//

template <typename Tag, typename... Types>
auto
get_labeled(const auto_bundle<Tag, Types...>& _obj)
{
    return get_labeled(_obj.get_component());
}
//
}  // namespace tim

//======================================================================================//
//
// variadic versions
//
#if !defined(TIMEMORY_VARIADIC_BLANK_AUTO_BUNDLE)
#    define TIMEMORY_VARIADIC_BLANK_AUTO_BUNDLE(tag, ...)                                \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_bundle<__VA_ARGS__>;     \
        TIMEMORY_BLANK_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

#if !defined(TIMEMORY_VARIADIC_BASIC_AUTO_BUNDLE)
#    define TIMEMORY_VARIADIC_BASIC_AUTO_BUNDLE(tag, ...)                                \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_bundle<__VA_ARGS__>;     \
        TIMEMORY_BASIC_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

#if !defined(TIMEMORY_VARIADIC_AUTO_BUNDLE)
#    define TIMEMORY_VARIADIC_AUTO_BUNDLE(tag, ...)                                      \
        using TIMEMORY_TYPEDEF_VARIABLE(__LINE__) = ::tim::auto_bundle<__VA_ARGS__>;     \
        TIMEMORY_MARKER(TIMEMORY_TYPEDEF_VARIABLE(__LINE__), tag);
#endif

//======================================================================================//
//
//      std::get operator
//
//======================================================================================//
//
namespace std
{
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename Tag, typename... Types>
typename std::tuple_element<N, std::tuple<Types...>>::type&
get(tim::auto_bundle<Tag, Types...>& obj)
{
    return get<N>(obj.data());
}
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename Tag, typename... Types>
const typename std::tuple_element<N, std::tuple<Types...>>::type&
get(const tim::auto_bundle<Tag, Types...>& obj)
{
    return get<N>(obj.data());
}
//
//--------------------------------------------------------------------------------------//
//
template <std::size_t N, typename Tag, typename... Types>
auto
get(tim::auto_bundle<Tag, Types...>&& obj)
    -> decltype(get<N>(std::forward<tim::auto_bundle<Tag, Types...>>(obj).data()))
{
    using obj_type = tim::auto_bundle<Tag, Types...>;
    return get<N>(std::forward<obj_type>(obj).data());
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace std
