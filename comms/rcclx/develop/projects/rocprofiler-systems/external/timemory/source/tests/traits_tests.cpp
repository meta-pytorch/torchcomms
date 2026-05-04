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

#include "test_macros.hpp"

TIMEMORY_TEST_DEFAULT_MAIN

#include "timemory/timemory.hpp"
#include "timemory/tpls/cereal/archives.hpp"
#include "timemory/tpls/cereal/cereal/archives/binary.hpp"
#include "timemory/tpls/cereal/cereal/archives/json.hpp"
#include "timemory/tpls/cereal/cereal/archives/portable_binary.hpp"
#include "timemory/tpls/cereal/cereal/archives/xml.hpp"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

//--------------------------------------------------------------------------------------//

namespace details
{
//--------------------------------------------------------------------------------------//
//  Get the current tests name
//
inline std::string
get_test_name()
{
    return std::string(::testing::UnitTest::GetInstance()->current_test_suite()->name()) +
           "." + ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

// this function consumes approximately "n" milliseconds of real time
inline void
do_sleep(long n)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(n));
}

// this function consumes an unknown number of cpu resources
inline long
fibonacci(long n)
{
    return (n < 2) ? n : (fibonacci(n - 1) + fibonacci(n - 2));
}

// this function consumes approximately "t" milliseconds of cpu time
void
consume(long n)
{
    // a mutex held by one lock
    mutex_t mutex;
    // acquire lock
    lock_t hold_lk(mutex);
    // associate but defer
    lock_t try_lk(mutex, std::defer_lock);
    // get current time
    auto now = std::chrono::steady_clock::now();
    // try until time point
    while(std::chrono::steady_clock::now() < (now + std::chrono::milliseconds(n)))
        try_lk.try_lock();
}

// get a random entry from vector
template <typename Tp>
size_t
random_entry(const std::vector<Tp>& v)
{
    std::mt19937 rng;
    rng.seed(std::random_device()());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, v.size() - 1);
    return v.at(dist(rng));
}

}  // namespace details

//--------------------------------------------------------------------------------------//

class traits_tests : public ::testing::Test
{
protected:
    TIMEMORY_TEST_DEFAULT_SUITE_BODY
};

//--------------------------------------------------------------------------------------//

TIMEMORY_DECLARE_COMPONENT(void_component)
TIMEMORY_DECLARE_COMPONENT(int64_component)
TIMEMORY_DECLARE_COMPONENT(array_component)
TIMEMORY_TEMPLATE_COMPONENT(template_component, typename T, T)
TIMEMORY_TEMPLATE_COMPONENT(variadic_component, TIMEMORY_ESC(typename T, size_t... N), T,
                            N...)

TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_accum, component::array_component, false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(base_has_last, component::int64_component, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::template_component<int64_t>,
                               false_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::template_component<int32_t>,
                               false_type)

TIMEMORY_TRAIT_TYPE(data, component::void_component, void)
TIMEMORY_TRAIT_TYPE(data, component::array_component, std::array<double, 4>)
TIMEMORY_TEMPLATE_TRAIT_TYPE(data, component::template_component, typename T, T,
                             std::pair<int, T>)
TIMEMORY_TEMPLATE_TRAIT_TYPE(data, component::variadic_component,
                             TIMEMORY_ESC(typename T, size_t... N), TIMEMORY_ESC(T, N...),
                             std::tuple<std::array<T, N>...>)

namespace tim  // NOLINT
{
namespace component
{
struct void_component : base<void_component>
{};
struct int64_component : base<int64_component, int64_t>
{};
struct array_component : base<array_component>
{};
template <typename T>
struct template_component : base<template_component<T>>
{};
template <typename T, size_t... N>
struct variadic_component : base<variadic_component<T, N...>>
{};
}  // namespace component
}  // namespace tim

using namespace tim::component;
using variadic_type_t = std::tuple<std::array<double, 1>, std::array<double, 3>>;

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, data)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE), tim::trait::data_t<__VA_ARGS__>>::value;       \
    printf("[%s]> data type for %-38s :: %s\n", details::get_test_name().c_str(),        \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<tim::trait::data_t<__VA_ARGS__>>().c_str())

    auto void_test  = TYPE_TEST(void, void_component);
    auto int_test   = TYPE_TEST(tim::type_list<>, int64_component);
    auto array_test = TYPE_TEST(TIMEMORY_ESC(std::array<double, 4>), array_component);
    auto temp_i64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int64_t>), template_component<int64_t>);
    auto temp_i32_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int32_t>), template_component<int32_t>);
    auto temp_u64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, uint64_t>), template_component<uint64_t>);
    auto var_test = TYPE_TEST(variadic_type_t, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, component_value_type)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE),                                                \
                 tim::trait::component_value_type_t<__VA_ARGS__>>::value;                \
    printf("[%s]> component value type for %-38s :: %s\n",                               \
           details::get_test_name().c_str(),                                             \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<tim::trait::component_value_type_t<__VA_ARGS__>>().c_str())

    auto void_test     = TYPE_TEST(void, void_component);
    auto int_test      = TYPE_TEST(int64_t, int64_component);
    auto array_test    = TYPE_TEST(TIMEMORY_ESC(std::array<double, 4>), array_component);
    auto temp_i64_test = TYPE_TEST(void, template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(void, template_component<int32_t>);
    auto temp_u64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, uint64_t>), template_component<uint64_t>);
    auto var_test = TYPE_TEST(variadic_type_t, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, value_type)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE), typename __VA_ARGS__::value_type>::value;      \
    printf("[%s]> value type for %-38s :: %s\n", details::get_test_name().c_str(),       \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<typename __VA_ARGS__::value_type>().c_str())

    auto void_test  = TYPE_TEST(void, void_component);
    auto int_test   = TYPE_TEST(int64_t, int64_component);
    auto array_test = TYPE_TEST(TIMEMORY_ESC(std::array<double, 4>), array_component);
    auto temp_i64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int64_t>), template_component<int64_t>);
    auto temp_i32_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int32_t>), template_component<int32_t>);
    auto temp_u64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, uint64_t>), template_component<uint64_t>);
    auto var_test = TYPE_TEST(variadic_type_t, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, accum_type)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE), typename __VA_ARGS__::accum_type>::value;      \
    printf("[%s]> accum type for %-38s :: %s\n", details::get_test_name().c_str(),       \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<typename __VA_ARGS__::accum_type>().c_str())

    auto void_test  = TYPE_TEST(void, void_component);
    auto int_test   = TYPE_TEST(int64_t, int64_component);
    auto array_test = TYPE_TEST(tim::null_type, array_component);
    auto temp_i64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int64_t>), template_component<int64_t>);
    auto temp_i32_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, int32_t>), template_component<int32_t>);
    auto temp_u64_test =
        TYPE_TEST(TIMEMORY_ESC(std::pair<int, uint64_t>), template_component<uint64_t>);
    auto var_test = TYPE_TEST(variadic_type_t, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, last_type)
{
#define TYPE_TEST(DATA_TYPE, ...)                                                        \
    std::is_same<TIMEMORY_ESC(DATA_TYPE), typename __VA_ARGS__::last_type>::value;       \
    printf("[%s]> last type for %-38s :: %s\n", details::get_test_name().c_str(),        \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<typename __VA_ARGS__::last_type>().c_str())

    auto void_test     = TYPE_TEST(void, void_component);
    auto int_test      = TYPE_TEST(int64_t, int64_component);
    auto array_test    = TYPE_TEST(tim::null_type, array_component);
    auto temp_i64_test = TYPE_TEST(tim::null_type, template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(tim::null_type, template_component<int32_t>);
    auto temp_u64_test = TYPE_TEST(tim::null_type, template_component<uint64_t>);
    auto var_test      = TYPE_TEST(tim::null_type, variadic_component<double, 1, 3>);

    EXPECT_TRUE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, generates_output)
{
#define TYPE_TEST(...)                                                                   \
    tim::trait::generates_output<__VA_ARGS__>::value;                                    \
    printf("[%s]> output type for %-38s :: %s\n", details::get_test_name().c_str(),      \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           tim::demangle<tim::trait::generates_output<__VA_ARGS__>::type>().c_str())

    auto void_test     = TYPE_TEST(void_component);
    auto int_test      = TYPE_TEST(int64_component);
    auto array_test    = TYPE_TEST(array_component);
    auto temp_i64_test = TYPE_TEST(template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(template_component<int32_t>);
    auto temp_u64_test = TYPE_TEST(template_component<uint64_t>);
    auto var_test      = TYPE_TEST(variadic_component<double, 1, 3>);

    EXPECT_FALSE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_TRUE(temp_i64_test);
    EXPECT_TRUE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, collects_data)
{
#define TYPE_TEST(...)                                                                   \
    tim::trait::collects_data<__VA_ARGS__>::value;                                       \
    printf("[%s]> collects_data type for %-38s :: %s\n",                                 \
           details::get_test_name().c_str(),                                             \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           (tim::trait::collects_data<__VA_ARGS__>::value) ? "true" : "false")

    auto void_test     = TYPE_TEST(void_component);
    auto int_test      = TYPE_TEST(int64_component);
    auto array_test    = TYPE_TEST(array_component);
    auto temp_i64_test = TYPE_TEST(template_component<int64_t>);
    auto temp_i32_test = TYPE_TEST(template_component<int32_t>);
    auto temp_u64_test = TYPE_TEST(template_component<uint64_t>);
    auto var_test      = TYPE_TEST(variadic_component<double, 1, 3>);

    EXPECT_FALSE(void_test);
    EXPECT_TRUE(int_test);
    EXPECT_TRUE(array_test);
    EXPECT_FALSE(temp_i64_test);
    EXPECT_FALSE(temp_i32_test);
    EXPECT_TRUE(temp_u64_test);
    EXPECT_TRUE(var_test);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, sizeof)
{
#define TYPE_TEST(...)                                                                   \
    sizeof(__VA_ARGS__);                                                                 \
    printf("[%s]> sizeof(%-38s) = %lu\n", details::get_test_name().c_str(),              \
           tim::demangle<__VA_ARGS__>().substr(16).c_str(),                              \
           (unsigned long) sizeof(__VA_ARGS__))

    auto void_test     = TYPE_TEST(void_component);
    auto int64_test    = TYPE_TEST(int64_component);
    auto array_test    = TYPE_TEST(array_component);
    auto temp_i32_test = TYPE_TEST(template_component<int32_t>);
    auto temp_i64_test = TYPE_TEST(template_component<int64_t>);
    auto temp_u64_test = TYPE_TEST(template_component<uint64_t>);
    auto var_test      = TYPE_TEST(variadic_component<double, 1, 3>);

    EXPECT_EQ(void_test, 1);
    EXPECT_EQ(int64_test, 48);
    EXPECT_EQ(temp_i32_test, 40);
    EXPECT_EQ(temp_i64_test, 56);
    EXPECT_EQ(temp_u64_test, 56);
    EXPECT_EQ(array_test, 56);
    EXPECT_EQ(var_test, 88);
#undef TYPE_TEST
}

//--------------------------------------------------------------------------------------//

template <template <typename> class Predicate, typename... Types>
struct validate
{
    using tuple_type = std::tuple<Types...>;

    explicit validate(bool val) { (*this)(val, std::index_sequence_for<Types...>{}); }

    template <size_t Idx, size_t... Tail>
    auto operator()(bool expected, std::index_sequence<Idx, Tail...>,
                    std::enable_if_t<sizeof...(Tail) == 0, int> = 0)
    {
        using type  = typename std::tuple_element<Idx, tuple_type>::type;
        bool result = Predicate<type>::value;
        EXPECT_TRUE(result == expected) << "    " << tim::demangle<Predicate<type>>()
                                        << " != " << std::boolalpha << expected;
        if(tim::settings::verbose() > 0 && result == expected)
        {
            std::cout << "[" << details::get_test_name() << "]> "
                      << tim::demangle<Predicate<type>>()
                      << "::value == " << std::boolalpha << expected << '\n';
        }
    }

    template <size_t Idx, size_t... Tail>
    auto operator()(bool val, std::index_sequence<Idx, Tail...>,
                    std::enable_if_t<(sizeof...(Tail) > 0), long> = 0)
    {
        (*this)(val, std::index_sequence<Idx>{});
        (*this)(val, std::index_sequence<Tail...>{});
    }
};

template <template <typename> class Predicate, template <typename...> class TupleT,
          typename... Types>
struct validate<Predicate, TupleT<Types...>> : validate<Predicate, Types...>
{
    explicit validate(bool val)
    : validate<Predicate, Types...>(val)
    {}
};

//--------------------------------------------------------------------------------------//

TEST_F(traits_tests, concepts)
{
    using namespace tim;

    auto _verbose = 1;
    std::swap(settings::verbose(), _verbose);

    using bundles =
        type_list<component_tuple<>, component_list<>,
                  component_bundle<project::timemory>, auto_tuple<>, auto_list<>,
                  auto_bundle<project::timemory>, lightweight_tuple<>>;

    using comp_bundles =
        type_list<component_tuple<>, component_list<>,
                  component_bundle<project::timemory>, lightweight_tuple<>>;

    using auto_bundles =
        type_list<auto_tuple<>, auto_list<>, auto_bundle<project::timemory>>;

    using stack_bundles = type_list<component_tuple<>, auto_tuple<>, lightweight_tuple<>>;

    using heap_bundles = type_list<component_list<>, auto_list<>>;

    using mixed_bundles =
        type_list<component_bundle<project::timemory>, auto_bundle<project::timemory>>;

    using input_archives =
        type_list<tim::cereal::JSONInputArchive, tim::cereal::XMLInputArchive,
                  tim::cereal::PortableBinaryInputArchive,
                  tim::cereal::BinaryInputArchive>;

    using output_archives =
        type_list<tim::cereal::PrettyJSONOutputArchive,
                  tim::cereal::MinimalJSONOutputArchive, tim::cereal::XMLOutputArchive,
                  tim::cereal::PortableBinaryOutputArchive,
                  tim::cereal::BinaryOutputArchive>;

    using archives = tim::convert_t<output_archives, input_archives>;

    static_assert((std::tuple_size<input_archives>::value +
                   std::tuple_size<output_archives>::value) ==
                      std::tuple_size<archives>::value,
                  "size<input_archives> + size<output_archives> != size<archives>");

    validate<trait::is_available, available_types_t>{ true };
    puts("");
    validate<concepts::is_component, available_types_t>{ true };
    validate<concepts::is_component, bundles>{ false };
    puts("");
    validate<concepts::is_variadic, bundles>{ true };
    puts("");
    validate<concepts::is_wrapper, bundles>{ true };
    validate<concepts::is_wrapper, available_types_t>{ false };
    puts("");
    validate<concepts::is_comp_wrapper, comp_bundles>{ true };
    validate<concepts::is_comp_wrapper, auto_bundles>{ false };
    puts("");
    validate<concepts::is_auto_wrapper, comp_bundles>{ false };
    validate<concepts::is_auto_wrapper, auto_bundles>{ true };
    puts("");
    validate<concepts::is_stack_wrapper, stack_bundles>{ true };
    validate<concepts::is_stack_wrapper, heap_bundles>{ false };
    validate<concepts::is_stack_wrapper, mixed_bundles>{ false };
    puts("");
    validate<concepts::is_heap_wrapper, stack_bundles>{ false };
    validate<concepts::is_heap_wrapper, heap_bundles>{ true };
    validate<concepts::is_heap_wrapper, mixed_bundles>{ false };
    puts("");
    validate<concepts::is_mixed_wrapper, stack_bundles>{ false };
    validate<concepts::is_mixed_wrapper, heap_bundles>{ false };
    validate<concepts::is_mixed_wrapper, mixed_bundles>{ true };
    puts("");
    validate<concepts::is_archive, bundles>{ false };
    validate<concepts::is_archive, stack_bundles>{ false };
    validate<concepts::is_archive, heap_bundles>{ false };
    validate<concepts::is_archive, mixed_bundles>{ false };
    validate<concepts::is_archive, available_types_t>{ false };
    validate<concepts::is_archive, archives>{ true };
    puts("");
    validate<concepts::is_input_archive, input_archives>{ true };
    validate<concepts::is_input_archive, output_archives>{ false };
    puts("");
    validate<concepts::is_output_archive, input_archives>{ false };
    validate<concepts::is_output_archive, output_archives>{ true };

    std::swap(settings::verbose(), _verbose);
}

//--------------------------------------------------------------------------------------//
