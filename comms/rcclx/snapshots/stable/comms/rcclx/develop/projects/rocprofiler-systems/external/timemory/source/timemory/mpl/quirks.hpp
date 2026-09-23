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

#include "timemory/mpl/concepts.hpp"
#include "timemory/mpl/types.hpp"
#include "timemory/utility/types.hpp"

#include <ostream>

namespace tim
{
namespace quirk
{
//
template <typename... Types>
struct config
{
    using type       = type_list<Types...>;
    using value_type = void;

    friend std::ostream& operator<<(std::ostream& _os, const config&) { return _os; }
};

//
template <typename QuirkT, typename Tp>
struct has_quirk
: std::conditional_t<std::is_same<QuirkT, Tp>::value, std::true_type, std::false_type>
{};

//
template <typename QuirkT, typename... Tp>
struct has_quirk<QuirkT, config<Tp...>>
{
    using type = std::conditional_t<is_one_of<QuirkT, config<Tp...>>::value,
                                    std::true_type, std::false_type>;
    static constexpr bool value = type::value;

    constexpr bool operator()() const { return type::value; }
};

/// \struct tim::quirk::config
/// \brief a variadic type which holds zero or more quirks that are passed to the
/// constructor of a component bundler.
/// \code{.cpp}
/// namespace quirk = tim::quirk;
/// using foo_t = tim::component_tuple<wall_clock>;
///
/// foo_t f("example", quirk::config<quirk::auto_start, quirk::flat_scope>{});
/// ...
/// f.stop();
/// \endcode
template <typename... Types>
struct config;

template <typename T>
struct is_config : false_type
{};

template <typename... Types>
struct is_config<config<Types...>> : true_type
{};

/// \struct tim::quirk::auto_start
/// \brief Will cause non-auto bundlers to invoke start() during construction. If
/// included as a template parameter of the bundler, it will have no effect.
/// Usage:
/// - bundler constructor w/in \ref tim::quirk::config object
/// - bundler template parameter
///
/// \code{.cpp}
/// // usage as template parameter
/// using bundle_t = tim::component_tuple<foo, tim::quirk::auto_start>;
///
/// void bar()
/// {
///     using bundle_t = tim::component_tuple<foo>;
///
///     // usage in constructor
///     bundle_t obj{ "bar", tim::quirk::config<tim::quirk:auto_start>{} };
/// }
/// \endcode
struct auto_start : concepts::quirk_type
{};

/// \struct tim::quirk::auto_stop
/// \brief This quirk is irrelevant. This is the default behavior for all bundlers. See
/// \ref tim::quirk::explicit_stop to suppress this behavior
struct auto_stop : concepts::quirk_type
{};

/// \struct tim::quirk::explicit_start
/// \brief Will cause auto bundlers to suppress calling start during construction. If
/// included as a template parameter of the non-auto bundler, it will have no effect.
/// Usage:
/// - bundler constructor w/in \ref tim::quirk::config object
/// - bundler template parameter
///
/// \code{.cpp}
/// // usage as template parameter
/// using bundle_t = tim::auto_tuple<foo, tim::quirk::explicit_start>;
///
/// void bar()
/// {
///     using bundle_t = tim::auto_tuple<foo>;
///
///     // usage in constructor
///     bundle_t obj{ "bar", tim::quirk::config<tim::quirk:explicit_start>{} };
///     obj.start(); // now required
/// }
/// \endcode
struct explicit_start : concepts::quirk_type
{};

/// \struct tim::quirk::explicit_stop
/// \brief Will cause bundlers to suppress calling stop during destruction.
/// Usage:
/// - bundler template parameter
///
/// \code{.cpp}
/// // usage as template parameter
/// using foo_bundle_t = tim::auto_tuple<foo, tim::quirk::explicit_stop>;
/// using baz_bundle_t = tim::component_tuple<foo, tim::quirk::explicit_stop>;
/// \endcode
struct explicit_stop : concepts::quirk_type
{};

/// \struct tim::quirk::explicit_push
/// \brief Will suppress the implicit `push()` within `start()` for the bundlers with this
/// characteristic
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
///
/// \code{.cpp}
/// // usage as template parameter
/// using bundle_t = tim::component_tuple<foo, tim::quirk::explicit_push>;
/// \endcode
struct explicit_push : concepts::quirk_type
{};

/// \struct tim::quirk::explicit_pop
/// \brief Will suppress the implicit `pop()` within `stop()` for the bundlers with this
/// characteristic. Combining this with \ref tim::quirk::explicit_push will effectively
/// allow the measurements within the bundler to only be recorded locally and statistics
/// to not be updated during intermediate measurements.
/// Usage:
/// - bundler template parameter
///
/// \code{.cpp}
/// // usage as template parameter
/// using bundle_t = tim::component_tuple<tim::component::wall_clock,
///                                       tim::quirk::explicit_push,
///                                       tim::quirk::explicit_pop>;
///
/// static bundle_t fibonacci_total{ "total" };
///
/// long fibonacci(long n)
/// {
///     bundle_t tmp{};
///     tmp.start();
///     auto result = (n < 2) ? n : (fibonacci(n-1) + fibonacci(n-2));
///     fibonacci_total += tmp.stop();
///     return result;
/// }
///
/// long foo(long n)
/// {
///     // create new "fibonacci_total" entry in call-graph. Pushing will reset
///     // any previous measurements
///     bundle_total.push();
///
///     // invoke this function when foo returns
///     tim::scope::destructor _dtor{ []() { fibonacci_total.pop(); } };
///
///     return fibonacci(n);
/// }
/// \endcode
struct explicit_pop : concepts::quirk_type
{};

/// \struct tim::quirk::exit_report
/// \brief Will cause auto-bundlers to write itself to stdout during destruction.
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
struct exit_report : concepts::quirk_type
{};

/// \struct tim::quirk::no_init
/// \brief Will cause bundlers to suppress calling any routines related to initializing
/// routines during construction. This is useful to override the default-initializer for a
/// bundler type
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
struct no_init : concepts::quirk_type
{};

/// \struct tim::quirk::no_store
/// \brief Will cause bundlers to suppress any implicit entries into the component
/// storage. This behavior is the default for tim::lightweight_bundle and is meaningless
/// in that context. It is quite similar to adding both \ref tim::quirk::explicit_push
/// and \ref tim::quirk::explicit_pop, however it effectively propagates \ref
/// tim::quirk::explicit_pop when used within the constructor.
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
struct no_store : concepts::quirk_type
{};

/// \struct tim::quirk::tree_scope
/// \brief Will cause bundlers to ignore the global settings and enforce hierarchical
/// storage in the call-graph.
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
struct tree_scope
: scope::tree
, concepts::quirk_type
{};

/// \struct tim::quirk::flat_scope
/// \brief Will cause bundlers to ignore the global settings and enforce flat storage in
/// the call-graph.
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
struct flat_scope
: scope::flat
, concepts::quirk_type
{};

/// \struct tim::quirk::timeline_scope
/// \brief Will cause bundlers to ignore the global settings and enforce timeline storage.
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
struct timeline_scope
: scope::timeline
, concepts::quirk_type
{};
//

/// \struct tim::quirk::stop_last_bundle
/// \brief Will cause a bundler to stop the "parent" bundler (of the same type). It can be
/// used as template parameter but be aware that
/// `tim::component_tuple<foo, tim::quirk::stop_last_bundle>` is NOT the same type as
/// `tim::component_tuple<foo>`.
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
struct stop_last_bundle : concepts::quirk_type
{};

/// \struct tim::quirk::unsafe
/// \brief When present, this argument instructs to skip any safety checks. Example checks
/// include: checking whether a component is in a stop state before startings, checking
/// whether a component has been started before stopping, checking whether push/pop has
/// been applied before applying the inverse
/// Usage:
/// - constructor of bundler within a \ref tim::quirk::config object
/// - bundler template parameter
/// - first argument to a bundler member function
struct unsafe : concepts::quirk_type
{};

/// \struct tim::quirk::fast
/// \brief When present, this argument instructs to pick the fast possible implementation
/// Usage:
/// - the second template argument to a gotcha component. The third template parameter
///   must have a function operator accepting the: 1. gotcha data object, 2. a function
///   pointer to the callee, and 3. the arguments.
///
struct fast : concepts::quirk_type
{};

/// \struct tim::quirk::static_data
/// \brief When present, this argument instructs to assume set_data calls are cached
/// statically Usage:
/// - the second template argument to a gotcha component. The third template parameter
///   must have a function operator accepting the: 1. gotcha data object, 2. a function
///   pointer to the callee, and 3. the arguments.
///
struct static_data : concepts::quirk_type
{};
//
}  // namespace quirk

namespace concepts
{
template <>
struct tuple_type<quirk::fast>
{
    using type = std::tuple<>;
};
//
template <>
struct component_type<quirk::fast>
{
    using type = lightweight_tuple<>;
};
//
template <typename... Tp>
struct tuple_type<quirk::config<Tp...>>
{
    using type = std::tuple<>;
};
//
template <typename... Tp>
struct component_type<quirk::config<Tp...>>
{
    using type = lightweight_tuple<>;
};
}  // namespace concepts
}  // namespace tim
